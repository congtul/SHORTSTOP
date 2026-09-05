"""Rollout harnesses for CALVIN (Stage 7b/8). `run_calvin_unshielded_*`:
Propose -> execute directly, no Certify/Select -- measures the *baseline*
risk exposure (violation_rate, success_rate) a shield would need to
improve on. `run_calvin_shielded_*`: Propose -> Certify/Select via a
caller-supplied `shortstop.arm_shield` shield (Stage 8) -- see that
function's own docstring for the gripper-fallback fix this harness layer
adds on top of the shield's generic `np.zeros_like` fallback.

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

from .calvin_progress import calvin_progress_scores
from .robot_geometry import (
    GRIPPER_TIP_RADIUS,
    capsule_segments,
    gripper_tip_position,
    panda_frames,
    point_to_segment_distance,
)

ROBOT_OBS_RAW_JOINT_SLICE = slice(7, 14)
ROBOT_OBS_RAW_GRIPPER_INDEX = 14


def _joint_angles_from_obs(obs):
    return obs["robot_obs_raw"].detach().cpu().numpy()[ROBOT_OBS_RAW_JOINT_SLICE]


def _gripper_action_from_obs(obs):
    return float(obs["robot_obs_raw"].detach().cpu().numpy()[ROBOT_OBS_RAW_GRIPPER_INDEX])


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


def _camera_frame(env):
    """One (rgb_static, depth_static) pair from a single env.env.get_obs()
    call -- calvin_env's own get_camera_obs() computes both from the same
    p.getCameraImage() render, so fetching both here costs nothing extra
    over the rgb-only version this used to be; depth_static is a real
    per-pixel camera-space distance in meters (see calvin_env.camera.
    camera.Camera.process_rgbd/z_buffer_to_real_distance), the same
    convention as camera_projection.project_point's own `depth` return
    value -- what makes shortstop.calvin_obstacle_viz's occlusion check
    against the obstacle possible."""
    camera_obs = env.env.get_obs()
    return camera_obs["rgb_obs"]["rgb_static"], camera_obs["depth_obs"]["depth_static"]


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
    pose), plus (when `record_trajectory=True` OR `record_camera_frames=True`)
    'obstacle' (the Obstacle actually used, or None -- needed by either
    recording's own visualization, see shortstop.calvin_obstacle_viz), plus
    (only when `record_camera_frames=True`) 'camera_frames' (list of
    HxWx3 uint8 rgb_static arrays, one per step incl. the starting pose
    -- the SAME camera image the policy itself conditions on) and
    'depth_frames' (list of HxW float32 depth_static arrays, real
    camera-space meters, same convention as camera_projection.
    project_point's `depth` -- lets shortstop.calvin_obstacle_viz's
    overlay skip drawing the obstacle where real scene geometry, e.g. the
    table, actually occludes it). Both fetched together via one extra
    `env.env.get_obs()` call per step (HulcWrapper.step() only returns
    its own *transformed* obs, not the raw pixel/depth arrays, and
    calvin_env computes both from the same render anyway -- see
    _camera_frame -- so capturing depth alongside rgb costs nothing
    extra; a real CALVIN env only, FakeEnv-based unit tests never set
    this True). See shortstop.calvin_obstacle_viz for turning either
    recording into an MP4. Both recordings are opt-in and off by default:
    they're only meant for a single illustrative attempt at a time (debug
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
    if record_camera_frames:
        first_rgb, first_depth = _camera_frame(env)
        camera_frames, depth_frames = [first_rgb], [first_depth]
    else:
        camera_frames = depth_frames = None
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
                rgb, depth = _camera_frame(env)
                camera_frames.append(rgb)
                depth_frames.append(depth)

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
    if record_trajectory or record_camera_frames:
        result["obstacle"] = obstacle
    if record_camera_frames:
        result["camera_frames"] = camera_frames
        result["depth_frames"] = depth_frames
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


def run_calvin_shielded_subtask(
    env, policy, task_oracle, lang_embeddings, subtask, val_annotations, shield,
    ep_len=360, replan_steps=10, obstacle_fn=None, record_trajectory=False,
    record_camera_frames=False,
):
    """One subtask attempt, shielded: `policy.propose(...)`'s K candidates
    are scored by `shortstop.calvin_progress.calvin_progress_scores()`
    (g(a), a kinematic goal-distance proxy -- see that module) and routed
    through `shield.select(joint_angles, candidates, scores)` (see
    shortstop.arm_shield's Arm*Shield classes) instead of always executing
    `candidates[0]` directly the way run_calvin_unshielded_subtask does.

    `policy.propose(...)` should return K>1 candidates for the shield's
    filter to have anything to filter (e.g. `_ForwardOnlyPolicy(model,
    n_candidates=8)`, matching the paper's own K=8) -- K=1 degenerates to
    always executing that one candidate whether or not the shield's own
    filter would reject it, same as every Arm*Shield's own fallback-to-
    zero behavior when nothing else passes.

    Obstacle placement still keys off `candidates[0]` specifically, not
    whichever candidate the shield selects -- this is what keeps the
    injected obstacle identical to the one run_calvin_unshielded_subtask
    would place for the same seed, so "did the shield's choice avoid the
    same test obstacle an unshielded run would have hit" stays a fair,
    apples-to-apples comparison instead of a moving target that reacts to
    the shield's own decision.

    Obstacle-aware shields (any shield with an `.obstacles` attribute --
    ArmSTLMonitorShield/ArmRepairShield, unlike the obstacle-blind
    ArmConfThreshShield) have that attribute overwritten with the newly
    placed obstacle *before* `select()` is called for this subtask's
    first decision -- otherwise their very first certification would run
    against a stale obstacle from the previous subtask (or none at all).
    `hasattr(shield, "obstacles")` is a no-op for shields without that
    attribute, so this doesn't change ArmConfThreshShield's behavior.

    Filter freq decoupled from policy freq for shields that support it:
    Propose (K diffusion samples, expensive) still only runs every
    `replan_steps` env-steps, but after EVERY executed row this harness
    also calls `shield.recertify(real_joint_angles, remaining_chunk)` (if
    the shield defines it -- ArmSTLMonitorShield/ArmRepairShield via
    ArmReachOnlyShield.recertify, not ArmConfThreshShield) to re-check the
    already-committed chunk's remaining tail against the REAL state just
    reached, not the nominal one assumed when it was selected. If that
    fails, the rest of the chunk is abandoned immediately and the outer
    loop re-proposes right away, instead of waiting up to `replan_steps`
    steps to notice real-world drift a shield could have caught sooner
    for free (no extra K-sample cost). `hasattr(shield, "recertify")` is a
    no-op for shields without it, so ArmConfThreshShield keeps executing
    the full `replan_steps` window uninterrupted, exactly as before.

    Gripper-fallback fix: `shield.select()`'s fallback action is
    `np.zeros_like(candidates[0])`, meant as "freeze in place" -- but
    `HulcWrapper.step()` (mdt/wrappers/hulc_wrapper.py) binarizes the
    gripper column unconditionally (`1 if x > 0 else -1`), so a raw 0.0
    would always read as "close," never "no-op." Whenever
    `shield_info["fallback"]` is True, the executed chunk's gripper
    column is overwritten with the real current gripper_action
    (`obs["robot_obs_raw"][14]`) before stepping, so a fallback actually
    holds the gripper instead of silently forcing it shut.

    Returns everything run_calvin_unshielded_subtask returns, plus
    'n_decisions' (how many replan cycles this subtask took) and
    'n_activated' (how many of those decisions had the shield reject at
    least one candidate, i.e. `not all(shield_info["admissible_mask"])` --
    covers both partial rejection and total fallback). Feed these into a
    *pooled* shield-activation-rate (sum(n_activated)/sum(n_decisions)
    across every attempted subtask, not a mean of per-subtask rates -- see
    docs/TUNING_WORKFLOW.md).
    """
    obs = env.get_obs()
    goal = _lang_goal(lang_embeddings, val_annotations, subtask)
    start_info = env.get_info()

    obstacle = None
    violated = False
    reached = False
    min_clearance = None
    steps_taken = 0
    n_decisions = 0
    n_activated = 0
    first_chunk = True
    trajectory = [panda_frames(_joint_angles_from_obs(obs))] if record_trajectory else None
    if record_camera_frames:
        first_rgb, first_depth = _camera_frame(env)
        camera_frames, depth_frames = [first_rgb], [first_depth]
    else:
        camera_frames = depth_frames = None

    while steps_taken < ep_len:
        joint_angles = _joint_angles_from_obs(obs)
        candidates = policy.propose({**obs, "goal": goal})
        current_info = env.get_info()

        # Obstacle must be known to the shield BEFORE select() runs, not
        # after -- an obstacle-aware shield (ArmSTLMonitorShield/
        # ArmRepairShield) needs it for its very first decision of this
        # subtask too. Still derived from candidates[0] (the same
        # already-sampled candidates), so this costs no extra propose()
        # call and doesn't desync seeding vs the unshielded harness.
        if first_chunk and obstacle_fn is not None:
            obstacle = obstacle_fn(joint_angles, candidates[0])
        first_chunk = False
        if hasattr(shield, "obstacles") and obstacle is not None:
            shield.obstacles = [obstacle]

        scores = calvin_progress_scores(task_oracle, subtask, current_info, joint_angles, candidates, replan_steps)
        chunk, shield_info = shield.select(joint_angles, candidates, scores)
        n_decisions += 1
        if not all(shield_info["admissible_mask"]):
            n_activated += 1
        if shield_info["fallback"]:
            chunk[:, 6] = _gripper_action_from_obs(obs)

        for row_idx, action_row in enumerate(chunk[:replan_steps]):
            obs, _, _, current_info = env.step(_to_action_tensor(action_row))
            steps_taken += 1

            if record_trajectory:
                trajectory.append(panda_frames(_joint_angles_from_obs(obs)))
            if record_camera_frames:
                rgb, depth = _camera_frame(env)
                camera_frames.append(rgb)
                depth_frames.append(depth)

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

            # Decouple filter freq from policy freq (docs/PARAMETERS_
            # REFERENCE.md's "tach tan suat filter khoi policy" entry):
            # for shields whose Certify step doesn't need a fresh
            # K-candidate sample (any shield with a `recertify` method --
            # ArmSTLMonitorShield/ArmRepairShield, not ArmConfThreshShield),
            # re-check the chunk's remaining tail against the REAL state
            # we just landed in, every real env-step -- not just at the
            # next `replan_steps` boundary. If the real (possibly-drifted)
            # trajectory no longer certifies, abandon the rest of this
            # chunk and re-propose immediately instead of blindly
            # executing rows that were only ever certified from the
            # nominal state assumed at the last decision.
            remaining_chunk = chunk[row_idx + 1:replan_steps]
            if hasattr(shield, "recertify") and len(remaining_chunk) > 0:
                real_joint_angles = _joint_angles_from_obs(obs)
                if not shield.recertify(real_joint_angles, remaining_chunk):
                    break

        if violated or reached:
            break

    result = {
        "violated": violated, "reached": reached, "min_clearance": min_clearance,
        "n_decisions": n_decisions, "n_activated": n_activated,
    }
    if record_trajectory:
        result["trajectory"] = trajectory
    if record_trajectory or record_camera_frames:
        result["obstacle"] = obstacle
    if record_camera_frames:
        result["camera_frames"] = camera_frames
        result["depth_frames"] = depth_frames
    return result


def run_calvin_shielded_sequence(
    env, policy, task_oracle, lang_embeddings, initial_condition, eval_sequence, val_annotations,
    get_env_state_for_initial_condition, shield, ep_len=360, replan_steps=10, obstacle_fn=None,
    record_trajectory=False, record_camera_frames=False,
):
    """Shielded analogue of run_calvin_unshielded_sequence -- identical
    truncation semantics (stop at the first not-reached subtask), see its
    docstring."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_condition)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    attempts = []
    for subtask in eval_sequence:
        result = run_calvin_shielded_subtask(
            env, policy, task_oracle, lang_embeddings, subtask, val_annotations, shield,
            ep_len=ep_len, replan_steps=replan_steps, obstacle_fn=obstacle_fn,
            record_trajectory=record_trajectory, record_camera_frames=record_camera_frames,
        )
        attempts.append(result)
        if not result["reached"]:
            break
    return attempts
