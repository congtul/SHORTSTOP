"""Unshielded rollout harness for CALVIN (Stage 7b) -- Propose -> execute
directly, no Certify/Repair. Used to measure the *baseline* risk exposure
(violation_rate, success_rate) that a shield would need to improve on.

Interface confirmed by reading the real, installed `mdt_policy` checkout
(`mdt/wrappers/hulc_wrapper.py`, `mdt/evaluation/mdt_evaluate.py`'s
`rollout()`), not guessed:
  - `env.get_obs()` / `env.get_info()` / `env.step(action_tensor)` /
    `env.reset(robot_obs=..., scene_obs=...)` -- HulcWrapper's real
    methods.
  - `env.step()` needs a **torch tensor** action of shape (7,) (relative
    actions -- confirmed via `HulcWrapper.step`'s `assert len(action)==7`
    branch), not a plain numpy row -- our policy clients return numpy
    chunks, so this module converts per-step before calling `env.step()`.
  - the RAW, unprocessed 15D proprioceptive state (matching
    `get_env_state_for_initial_condition`'s layout: ee_pos[0:3],
    ee_orn[3:6], gripper_width[6], joint_positions[7:14],
    gripper_action[14]) is `obs["robot_obs_raw"]` -- a torch tensor,
    *not* `obs["robot_obs"]` (that key is CALVIN's own `process_state()`
    output, a possibly-different-shape tensor prepared for the model's
    own input, not guaranteed to keep this layout).
  - `task_oracle.get_task_info_for_set(start_info, current_info, {subtask})`
    returns a non-empty set once `subtask` is completed -- CALVIN's own
    real success checker, reused as-is (see `rollout()`).

Obstacle handling follows the design decision in
docs/STAGE7B_CALVIN_PIPELINE_DESIGN.md: purely privileged/geometric (see
shortstop.calvin_obstacle), and -- matching shortstop.env.ReachAvoid2D's
`done = violated or reached or timeout` -- a subtask attempt stops the
instant it is violated, so `violated` and `reached` can never both be True
for the same attempt (except the same-step edge case), and no attempt
that entered X_u is ever counted as a success by continuing past it.
"""
import numpy as np

from .robot_geometry import sphere_centers

ROBOT_OBS_RAW_JOINT_SLICE = slice(7, 14)


def _joint_angles_from_obs(obs):
    return obs["robot_obs_raw"].detach().cpu().numpy()[ROBOT_OBS_RAW_JOINT_SLICE]


def _to_action_tensor(action_row):
    import torch
    return torch.as_tensor(np.asarray(action_row, dtype=np.float32))


def _violates(obs, obstacle):
    if obstacle is None:
        return False
    joint_angles = _joint_angles_from_obs(obs)
    centers = sphere_centers(joint_angles)
    return bool(np.any(np.linalg.norm(centers - obstacle.center, axis=1) <= obstacle.radius))


def _lang_goal(lang_embeddings, val_annotations, subtask):
    """Builds the goal dict MDTVAgent.forward() needs: `get_lang_goal()`'s
    own dict, plus the `lang_text` key forward() reads directly (not
    returned by get_lang_goal() itself)."""
    lang_annotation = val_annotations[subtask][0]
    goal = lang_embeddings.get_lang_goal(lang_annotation)
    goal["lang_text"] = lang_annotation
    return goal


def run_calvin_unshielded_subtask(
    env, policy, task_oracle, lang_embeddings, subtask, val_annotations,
    ep_len=360, replan_steps=10, obstacle=None,
):
    """One subtask attempt, unshielded: `policy.propose(...)`'s first
    candidate is executed directly, no filtering at all -- matches
    shortstop.experiment.run_episode's `else: first_action = candidates[0][0]`
    branch (no shield = execute the first candidate).

    Returns {'violated': bool, 'reached': bool}. `obstacle`
    (shortstop.env.Obstacle or None): if given, checked (ground truth,
    from the real post-step robot_obs_raw) after every real env.step();
    None runs the pure CALVIN-official baseline with no check at all.
    """
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()

    violated = False
    reached = False
    steps_taken = 0
    while steps_taken < ep_len:
        candidates = policy.propose({**obs, "goal": goal})
        chunk = candidates[0]

        for action_row in chunk[:replan_steps]:
            obs, _, _, current_info = env.step(_to_action_tensor(action_row))
            steps_taken += 1

            if _violates(obs, obstacle):
                violated = True
                break

            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            if len(current_task_info) > 0:
                reached = True
                break

            if steps_taken >= ep_len:
                break

        if violated or reached:
            break

    return {"violated": violated, "reached": reached}


def run_calvin_unshielded_sequence(
    env, policy, task_oracle, lang_embeddings, initial_condition, eval_sequence, val_annotations,
    get_env_state_for_initial_condition, ep_len=360, replan_steps=10, obstacle_fn=None,
):
    """One full sequence attempt (up to `len(eval_sequence)` subtasks),
    stopping at the first failed/violated subtask -- mirrors CALVIN's own
    `evaluate_sequence()` truncation exactly (see
    shortstop/calvin_metrics.py's docstring for why this matters for fair
    cross-baseline comparison).

    `obstacle_fn(joint_angles, reference_chunk) -> Obstacle`, or None to
    run with no obstacle at all (the pure CALVIN-official baseline).
    Returns a list of 0..len(eval_sequence) per-subtask {'violated',
    'reached'} dicts -- feed this (one list per launched sequence) into
    shortstop.calvin_metrics.build_fixed_cohort_slots.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_condition)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    attempts = []
    for subtask in eval_sequence:
        obstacle = None
        if obstacle_fn is not None:
            obs = env.get_obs()
            joint_angles = _joint_angles_from_obs(obs)
            reference_chunk = policy.propose({**obs, "goal": _lang_goal(lang_embeddings, val_annotations, subtask)})[0]
            obstacle = obstacle_fn(joint_angles, reference_chunk)

        result = run_calvin_unshielded_subtask(
            env, policy, task_oracle, lang_embeddings, subtask, val_annotations,
            ep_len=ep_len, replan_steps=replan_steps, obstacle=obstacle,
        )
        attempts.append(result)
        if not result["reached"]:
            break
    return attempts
