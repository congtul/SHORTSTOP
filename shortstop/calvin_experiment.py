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

Ground-truth collision check (_clearance) uses the *full-arm capsule
chain* (robot_geometry.capsule_segments -- one capsule per link, link0
through link7) plus a gripper-tip capsule (flange to
robot_geometry.gripper_tip_position(), covering panda_hand + the fingers
past the flange). A collision partway along a link (elbow/forearm/wrist
links are 0.14-0.35m long, far longer than a single point's own radius),
on a link with no dedicated check point at all, or at the fingertips
(which point *ahead* of the flange in the direction of travel, past
where a flange-centered point alone would ever detect anything) is
covered. shortstop.arm_reach's reachtube (used by the shield's own
Certify step, once wired in) was upgraded in step to check the same
9-frame chain (robot_geometry.FRAME_RADIUS/panda_frames(), not a coarser
named subset of it) -- see its module docstring for the residual
per-point-box (vs. exact capsule) approximation that remains there.
"""
import numpy as np

from .robot_geometry import (
    GRIPPER_TIP_RADIUS,
    capsule_segments,
    gripper_tip_position,
    panda_frames,
    point_to_segment_distance,
)

ROBOT_OBS_RAW_JOINT_SLICE = slice(7, 14)


def _joint_angles_from_obs(obs):
    return obs["robot_obs_raw"].detach().cpu().numpy()[ROBOT_OBS_RAW_JOINT_SLICE]


def _to_action_tensor(action_row):
    import torch
    return torch.as_tensor(np.asarray(action_row, dtype=np.float32))


def _clearance(obs, obstacle):
    """min over (the flange->fingertip capsule, every one of the 8 link
    capsules) of (distance to obstacle center - obstacle.radius - this
    primitive's own radius) -- signed: <= 0 means violated (the arm's
    *volume*, not just a handful of sample points, touches the
    obstacle), > 0 is how far the closest surface is from the obstacle's
    boundary. `None` if there is no obstacle at all (nothing to measure
    clearance against).

    See module docstring for why this checks the full capsule chain -- a
    sphere/capsule is a physical volume, and real collision happens at
    center/axis-distance <= obstacle.radius + the primitive's own radius;
    omitting either the arm's thickness, the *length* of a link between
    its two named endpoints, or the fingers reaching out past the flange
    would under-count real risk.
    """
    if obstacle is None:
        return None
    joint_angles = _joint_angles_from_obs(obs)

    flange = panda_frames(joint_angles)[-1]
    tip = gripper_tip_position(joint_angles)
    d = point_to_segment_distance(obstacle.center, flange, tip)
    clearances = [d - obstacle.radius - GRIPPER_TIP_RADIUS]

    for point_a, point_b, link_radius in capsule_segments(joint_angles):
        d = point_to_segment_distance(obstacle.center, point_a, point_b)
        clearances.append(d - obstacle.radius - link_radius)

    return float(min(clearances))


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
    ep_len=360, replan_steps=10, obstacle_fn=None, record_trajectory=False,
    record_camera_frames=False,
):
    """One subtask attempt, unshielded: `policy.propose(...)`'s first
    candidate is executed directly, no filtering at all -- matches
    shortstop.experiment.run_episode's `else: first_action = candidates[0][0]`
    branch (no shield = execute the first candidate).

    Returns {'violated': bool, 'reached': bool, 'min_clearance': float or
    None}, plus (only when `record_trajectory=True`) 'trajectory' (list of
    (9, 3) panda_frames() arrays -- the whole chain, base through flange,
    not a coarser named subset of it -- one per step incl. the starting
    pose) and 'obstacle' (the Obstacle actually used, or None), plus
    (only when `record_camera_frames=True`) 'camera_frames' (list of
    HxWx3 uint8 rgb_static arrays, one per step incl. the starting pose
    -- the SAME camera image the policy itself conditions on, fetched via
    an extra `env.env.get_obs()` call since HulcWrapper.step() only
    returns its own *transformed* obs, not the raw pixel array; a real
    CALVIN env only -- FakeEnv-based unit tests never set this True). See
    shortstop.calvin_obstacle_viz for turning either recording into an
    MP4. Both recordings are opt-in and off by default: they're only
    meant for a single illustrative attempt at a time (debug
    visualization), not every attempt of a large sweep -- the
    memory/list-building (and, for camera frames, extra render-call) cost
    is not worth paying when nobody is going to render it.

    `min_clearance` is the smallest `_clearance()` value seen over every
    step of this attempt (see its docstring) -- `None` when `obstacle_fn`
    is None (nothing to measure). `obstacle_fn(joint_angles, chunk) ->
    Obstacle`, or None to run the pure CALVIN-official baseline with no
    check at all. Deliberately placed from the *first* proposed chunk of
    this subtask -- the same chunk that actually gets executed, not a
    separate speculative `propose()` call -- so an unshielded run with vs.
    without an obstacle_fn issues the exact same number/order of
    policy.propose() calls and (given the caller reseeds identically
    before each sequence) follows the identical trajectory up to the
    point the obstacle is hit. A separate reference-chunk call would burn
    an extra draw of the diffusion policy's noise and desync the two
    runs from step 1, defeating the whole point of the comparison.
    """
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()

    obstacle = None
    violated = False
    reached = False
    min_clearance = None
    steps_taken = 0
    first_chunk = True
    trajectory = [panda_frames(_joint_angles_from_obs(obs))] if record_trajectory else None
    camera_frames = [env.env.get_obs()["rgb_obs"]["rgb_static"]] if record_camera_frames else None
    while steps_taken < ep_len:
        candidates = policy.propose({**obs, "goal": goal})
        chunk = candidates[0]

        if first_chunk and obstacle_fn is not None:
            obstacle = obstacle_fn(_joint_angles_from_obs(obs), chunk)
        first_chunk = False

        for action_row in chunk[:replan_steps]:
            obs, _, _, current_info = env.step(_to_action_tensor(action_row))
            steps_taken += 1

            if record_trajectory:
                trajectory.append(panda_frames(_joint_angles_from_obs(obs)))
            if record_camera_frames:
                camera_frames.append(env.env.get_obs()["rgb_obs"]["rgb_static"])

            clearance = _clearance(obs, obstacle)
            if clearance is not None:
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                if clearance <= 0:
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

    result = {"violated": violated, "reached": reached, "min_clearance": min_clearance}
    if record_trajectory:
        result["trajectory"] = trajectory
        result["obstacle"] = obstacle
    if record_camera_frames:
        result["camera_frames"] = camera_frames
    return result


def run_calvin_unshielded_sequence(
    env, policy, task_oracle, lang_embeddings, initial_condition, eval_sequence, val_annotations,
    get_env_state_for_initial_condition, ep_len=360, replan_steps=10, obstacle_fn=None,
    record_trajectory=False, record_camera_frames=False,
):
    """One full sequence attempt (up to `len(eval_sequence)` subtasks),
    stopping at the first failed/violated subtask -- mirrors CALVIN's own
    `evaluate_sequence()` truncation exactly (see
    shortstop/calvin_metrics.py's docstring for why this matters for fair
    cross-baseline comparison).

    `obstacle_fn(joint_angles, chunk) -> Obstacle`, or None to run with no
    obstacle at all (the pure CALVIN-official baseline) -- forwarded as-is
    to run_calvin_unshielded_subtask for each subtask (see its docstring
    for why the obstacle is derived from the actually-executed chunk
    instead of a separate speculative propose() call). `record_trajectory`
    and `record_camera_frames` likewise forwarded as-is to every subtask
    -- see that function's docstring; only use these for a single
    sequence you intend to visualize (shortstop.calvin_obstacle_viz), not
    a full metrics sweep.
    Returns a list of 0..len(eval_sequence) per-subtask {'violated',
    'reached'} dicts -- feed this (one list per launched sequence) into
    shortstop.calvin_metrics.build_fixed_cohort_slots.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_condition)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    attempts = []
    for subtask in eval_sequence:
        result = run_calvin_unshielded_subtask(
            env, policy, task_oracle, lang_embeddings, subtask, val_annotations,
            ep_len=ep_len, replan_steps=replan_steps, obstacle_fn=obstacle_fn,
            record_trajectory=record_trajectory, record_camera_frames=record_camera_frames,
        )
        attempts.append(result)
        if not result["reached"]:
            break
    return attempts
