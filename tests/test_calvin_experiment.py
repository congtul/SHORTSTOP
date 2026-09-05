"""Exercises shortstop.calvin_experiment's control-flow logic (replan
cycling, ground-truth violation check, stop-on-violation, task_oracle
success check) against fakes standing in for HulcWrapper/task_oracle/
lang_embeddings -- no real CALVIN/mdt_policy session involved. See
docs/CALVIN_SETUP.md for the real interface this mirrors.
"""
import functools

import numpy as np
import torch

from shortstop.arm_shield import ArmConfThreshShield, ArmSTLMonitorShield
from shortstop.arm_reach import propagate_arm_tube
from shortstop.calvin_experiment import (
    _base_transform_from_env,
    _clearance,
    run_calvin_shielded_subtask,
    run_calvin_unshielded_sequence,
    run_calvin_unshielded_subtask,
)
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates
from shortstop.env import Obstacle
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS, panda_frames

SUBTASK = "fake_subtask"

# A real, well-within-JOINT_LIMITS Franka "ready" pose -- NOT np.zeros: see
# tests/test_arm_shield.py's own Q_HOME for why q=0 is itself physically
# invalid for joint 4 alone, which matters for any test exercising a real
# shield's select()/recertify() (now enforcing JOINT_LIMITS).
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


class FakeEnv:
    """joint_angles[0] += action[0] each step -- just enough dynamics to
    move the sphere-chain in a controlled, predictable way. `scene_positions`
    (name -> [x,y,z]) and `gripper_action` are read by get_info()/get_obs()
    for the shielded-harness tests (calvin_progress_scores's target-object
    lookup, and the gripper-fallback fix) -- unused/benign for the plain
    unshielded tests below."""

    def __init__(self):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        self.gripper_action = 1.0
        self.gripper_width = 0.0  # closed by default -- matches the old fixed-radius model's floor
        self.scene_positions = {}
        self.last_gripper_action = None

    def reset(self, robot_obs=None, scene_obs=None):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        robot_obs_raw = np.zeros(15, dtype=np.float32)
        robot_obs_raw[6] = self.gripper_width
        robot_obs_raw[7:14] = self.joint_angles
        robot_obs_raw[14] = self.gripper_action
        return {"robot_obs_raw": torch.tensor(robot_obs_raw)}

    def get_info(self):
        return {
            "step_count": self.step_count,
            "scene_info": {
                "movable_objects": {
                    name: {"current_pos": pos} for name, pos in self.scene_positions.items()
                },
            },
        }

    def step(self, action_tensor):
        action = action_tensor.detach().cpu().numpy()
        self.last_gripper_action = float(action[6])
        self.joint_angles = self.joint_angles.copy()
        self.joint_angles[0] += action[0]
        self.step_count += 1
        return self.get_obs(), 0.0, False, self.get_info()


class FakeTaskOracle:
    def __init__(self, success_after_steps):
        self.success_after_steps = success_after_steps

    def get_task_info_for_set(self, start_info, current_info, subtask_set):
        if current_info["step_count"] >= self.success_after_steps:
            return set(subtask_set)
        return set()


class FakeTaskOracleWithTasks(FakeTaskOracle):
    """Adds `.tasks` ({subtask_name: functools.partial(fn, obj_name, ...)},
    matching calvin_env.envs.tasks.Tasks's own shape) -- needed by
    shortstop.calvin_progress.calvin_progress_scores for the shielded
    harness, not by the plain unshielded one."""

    def __init__(self, success_after_steps, tasks):
        super().__init__(success_after_steps)
        self.tasks = tasks


class FakePolicy:
    """Always proposes 1 candidate chunk moving joint 0 by `delta`/step."""

    def __init__(self, delta=0.1, horizon=10):
        self.delta = delta
        self.horizon = horizon

    def propose(self, observation):
        del observation
        chunk = np.zeros((self.horizon, 7))
        chunk[:, 0] = self.delta
        return [chunk]


class FakeMultiCandidatePolicy:
    """Always proposes the same fixed list of candidate chunks, regardless
    of observation -- deterministic stand-in for a K-candidate policy."""

    def __init__(self, chunks):
        self.chunks = chunks

    def propose(self, observation):
        del observation
        return list(self.chunks)


def _joint0_chunk(delta, horizon=10):
    chunk = np.zeros((horizon, 7))
    chunk[:, 0] = delta
    return chunk


def _predicted_endpoint(joint_angles, chunk):
    tube = propagate_arm_tube(joint_angles, chunk, w_bar=0.0, model_error=0.0)
    return tube[-1][FLANGE_FRAME_INDEX].center()


class FakeLangEmbeddings:
    def get_lang_goal(self, task):
        return {"lang": task}


VAL_ANNOTATIONS = {SUBTASK: [SUBTASK]}


def test_unshielded_subtask_succeeds_when_never_flagged():
    env = FakeEnv()
    env.reset()
    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS,
        ep_len=360, replan_steps=10, obstacle_fn=None,
    )
    assert result == {"violated": False, "reached": True, "min_clearance": None, "steps_taken": 12}


def test_unshielded_subtask_stops_at_violation_and_never_reaches():
    env = FakeEnv()
    env.reset()
    # obstacle placed exactly where the arm is after 6 steps -- well
    # before the task_oracle would succeed at step 12
    obstacle_joint_angles = np.zeros(N_JOINTS)
    obstacle_joint_angles[0] = 0.1 * 6
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS,
        ep_len=360, replan_steps=10, obstacle_fn=lambda joint_angles, chunk: obstacle,
    )
    assert result["violated"] is True
    assert result["reached"] is False  # violated => never gets a chance to also succeed
    assert result["min_clearance"] <= 0.0  # violated => clearance hit zero/negative


def test_base_transform_from_env_defaults_to_identity_without_env_dot_robot():
    """FakeEnv (like every other test double in this file) has no
    `.env.robot` chain at all -- _base_transform_from_env must fall back
    to identity rather than raising, so g(a)'s new base-frame correction
    (calvin_progress_scores's base_position/base_orientation) is a no-op
    for every FakeEnv-based test in this file, exactly as before the fix."""
    env = FakeEnv()
    base_position, base_orientation = _base_transform_from_env(env)
    assert base_position == (0.0, 0.0, 0.0)
    assert base_orientation == (0.0, 0.0, 0.0, 1.0)


def test_base_transform_from_env_reads_the_real_robot_base_pose_when_present():
    class _FakeRobot:
        base_position = [-0.34, -0.46, 0.24]
        base_orientation = [0.0, 0.0, 0.0, 1.0]

    class _FakeInnerEnv:
        robot = _FakeRobot()

    class _FakeWrapper:
        env = _FakeInnerEnv()

    base_position, base_orientation = _base_transform_from_env(_FakeWrapper())
    assert base_position == [-0.34, -0.46, 0.24]
    assert base_orientation == [0.0, 0.0, 0.0, 1.0]


def test_clearance_tracks_the_real_gripper_width_not_a_fixed_worst_case_bound():
    """Regression test for the finger_tip_capsules fix (Category A.4):
    _clearance() used to assume the gripper was ALWAYS at its worst-case
    (fully open) spread, regardless of the real, current
    gripper_opening_width -- an obstacle placed exactly where an OPEN
    finger's own tip would be must read as clear when the gripper is
    actually closed, and as violated once it's actually open (same
    joint_angles both times -- only gripper_width differs)."""
    from shortstop.robot_geometry import finger_tip_capsules

    env = FakeEnv()
    env.reset()
    q = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0])  # unfolded -- avoids q=0's own degenerate self-overlap
    env.joint_angles = q.copy()
    (_, left_far), _ = finger_tip_capsules(q, gripper_width=0.08)
    obstacle = Obstacle(center=left_far, radius=0.01)

    env.gripper_width = 0.0
    assert _clearance(env.get_obs(), obstacle) > 0.0  # closed -- finger nowhere near the obstacle

    env.gripper_width = 0.08
    assert _clearance(env.get_obs(), obstacle) <= 0.0  # open -- finger tip is exactly at the obstacle


def test_violation_rate_and_success_rate_differ_with_vs_without_obstacle():
    """The exact comparison requested: same policy/task, run once with no
    obstacle at all and once with a privileged obstacle placed on the
    arm's own path -- confirm success_rate drops and violation_rate rises.
    """
    def make_sequence_result(obstacle):
        env = FakeEnv()
        return run_calvin_unshielded_sequence(
            env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(),
            initial_condition={}, eval_sequence=[SUBTASK], val_annotations=VAL_ANNOTATIONS,
            get_env_state_for_initial_condition=lambda ic: (None, None),
            ep_len=360, replan_steps=10,
            obstacle_fn=(lambda joint_angles, reference_chunk: obstacle) if obstacle is not None else None,
        )

    obstacle_joint_angles = np.zeros(N_JOINTS)
    obstacle_joint_angles[0] = 0.1 * 6
    obstacle = Obstacle(center=panda_frames(obstacle_joint_angles)[-1], radius=0.02)

    n_sequences = 20
    results_without = [make_sequence_result(None) for _ in range(n_sequences)]
    results_with = [make_sequence_result(obstacle) for _ in range(n_sequences)]

    slots_without = build_fixed_cohort_slots(results_without, subtasks_per_sequence=1)
    slots_with = build_fixed_cohort_slots(results_with, subtasks_per_sequence=1)

    violation_without, success_without = fixed_cohort_rates(slots_without)
    violation_with, success_with = fixed_cohort_rates(slots_with)

    assert violation_without == 0.0
    assert success_without == 1.0
    assert violation_with == 1.0  # every run takes the identical deterministic path through the obstacle
    assert success_with == 0.0


def test_shielded_subtask_executes_the_shields_higher_scoring_admissible_candidate():
    env = FakeEnv()
    env.reset()

    positive_delta = _joint0_chunk(0.1)
    negative_delta = _joint0_chunk(-0.1)
    policy = FakeMultiCandidatePolicy([positive_delta, negative_delta])

    # target object sits exactly at the +delta candidate's own predicted
    # endpoint -> its g(a) (-distance) is unambiguously the higher of the two
    target_pos = _predicted_endpoint(np.zeros(N_JOINTS), positive_delta)
    env.scene_positions = {"block_red": target_pos.tolist()}
    task_oracle = FakeTaskOracleWithTasks(
        success_after_steps=10**9,  # never triggers within this single-decision window
        tasks={SUBTASK: functools.partial(lambda *a, **k: None, "block_red")},
    )

    shield = ArmConfThreshShield(disagreement_threshold=10.0, replan_steps=10)  # generous -> both candidates admissible

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=None,
    )

    assert result["n_decisions"] == 1
    assert result["n_activated"] == 0  # nothing rejected -- both admissible
    assert env.joint_angles[0] > 0  # executed the +delta candidate (higher g(a), closer to target)


def test_shielded_subtask_wires_the_obstacle_into_an_obstacle_aware_shield_before_its_first_decision():
    """Regression test for the obstacle-then-select reordering fix: an
    obstacle-aware shield (has a `.obstacles` attribute, e.g.
    ArmSTLMonitorShield) must have it set to the ACTUALLY placed obstacle
    before its very first select() call of a subtask -- not left over
    from whatever it was constructed with. Checked by constructing the
    shield with a far-away placeholder obstacle, then placing the real
    obstacle exactly on one candidate's own predicted endpoint (mirroring
    sample_obstacle_from_reference_chunk's real behavior) -- if the
    reordering fix works, that candidate is rejected on the very first
    decision; if the shield were still certifying against the stale
    far-away placeholder, it would wrongly pass as admissible.

    Starts from Q_HOME (not np.zeros -- see this module's own Q_HOME
    docstring) and moves along y (not x) for `other`: select() now also
    enforces JOINT_LIMITS (see ArmReachOnlyShield._trajectory_within_
    joint_limits), and a purely-x displacement large enough to clear the
    flange's own inflation radius drives some joint well past its real
    limit over 10 repeated rows -- verified directly (not guessed) that
    this dx/dy pair stays within JOINT_LIMITS for both candidates while
    still separating geometrically."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    def _xyz_chunk(dx, dy, dz, horizon=10):
        chunk = np.zeros((horizon, 7))
        chunk[:, 0] = dx
        chunk[:, 1] = dy
        chunk[:, 2] = dz
        return chunk

    candidate = _xyz_chunk(0.02, 0.0, 0.0)
    other = _xyz_chunk(0.0, 0.2, 0.0)  # different axis -- clears the flange's own inflation radius
    policy = FakeMultiCandidatePolicy([candidate, other])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})

    far_away = Obstacle(center=np.array([100.0, 100.0, 100.0]), radius=0.02)
    shield = ArmSTLMonitorShield(obstacles=[far_away], epsilon=0.0)

    def obstacle_fn(joint_angles, chunk):
        return Obstacle(center=_predicted_endpoint(joint_angles, chunk), radius=0.05)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=obstacle_fn,
    )

    assert result["n_activated"] == 1  # candidate (candidates[0]) got rejected
    # `other`'s dx=0 -- FakeEnv's own dynamics (joint_angles[0] += action[0])
    # only ever moves index 0, so it staying exactly unchanged from Q_HOME
    # confirms `other` (not `candidate`, whose dx=0.02 would have moved it)
    # is what actually got executed.
    assert env.joint_angles[0] == Q_HOME[0]


def test_shielded_subtask_reports_latency_and_ground_truth_rejection_precision():
    """Regression test for the new generic metrics instrumentation
    (shortstop.calvin_experiment._candidate_clearance +
    run_calvin_shielded_subtask's latencies_ms/rejected_total/
    rejected_truly_unsafe): reuses the exact same obstacle-placed-at-
    candidate's-own-endpoint setup as the obstacle-wiring regression test
    above, but checks the NEW fields instead. `candidate` genuinely hits
    the obstacle (ground-truth clearance -0.11, confirmed directly) and
    `other` genuinely clears it (+0.15) -- so this is a real, not a
    false-positive, rejection: rejected_total and rejected_truly_unsafe
    must both be 1, not just rejected_total."""
    env = FakeEnv()
    env.reset()
    env.joint_angles = Q_HOME.copy()

    def _xyz_chunk(dx, dy, dz, horizon=10):
        chunk = np.zeros((horizon, 7))
        chunk[:, 0] = dx
        chunk[:, 1] = dy
        chunk[:, 2] = dz
        return chunk

    candidate = _xyz_chunk(0.02, 0.0, 0.0)
    other = _xyz_chunk(0.0, 0.2, 0.0)
    policy = FakeMultiCandidatePolicy([candidate, other])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = ArmSTLMonitorShield(obstacles=[], epsilon=0.0)

    def obstacle_fn(joint_angles, chunk):
        return Obstacle(center=_predicted_endpoint(joint_angles, chunk), radius=0.05)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=obstacle_fn,
    )

    assert len(result["latencies_ms"]) == result["n_decisions"] == 1
    assert all(t >= 0.0 for t in result["latencies_ms"])
    assert result["rejected_total"] == 1
    assert result["rejected_truly_unsafe"] == 1


def test_shielded_subtask_fallback_holds_the_gripper_instead_of_forcing_it_closed():
    """Regression test: shield.select()'s fallback action is
    np.zeros_like(candidates[0]) -- the harness must overwrite its gripper
    column with the real current gripper_action before stepping, or a
    fallback would (via HulcWrapper.step()'s real binarization, not
    exercised by this fake env) silently force the gripper closed
    regardless of its actual prior state. Checked here at the harness
    layer directly: whatever env.step() actually received.

    Also exercises the fallback-window shrink (2026-09-05): a TOTAL
    fallback only commits 1 env-step before re-proposing, not the full
    `replan_steps` window -- `FakeMultiCandidatePolicy` is stateless (same
    2 candidates every call, see its own docstring) and this threshold
    always disagrees, so EVERY decision here is a fallback -> with
    ep_len=10 and a 1-step window, that's 10 decisions, not 1."""
    env = FakeEnv()
    env.reset()
    env.gripper_action = 0.42  # distinctive marker, neither 0 nor +-1

    policy = FakeMultiCandidatePolicy([_joint0_chunk(0.1), _joint0_chunk(-0.1)])
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = ArmConfThreshShield(disagreement_threshold=1e-9, replan_steps=10)  # tiny -> both candidates disagree -> fallback

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=None,
    )

    assert result["n_decisions"] == 10  # 1-step fallback window -> 10 decisions over ep_len=10
    assert result["n_activated"] == 10
    assert result["n_fallback"] == 10
    assert np.isclose(env.last_gripper_action, 0.42)  # held (float32 roundtrip), not forced to 0/-1
    assert env.joint_angles[0] == 0.0  # froze in place -- fallback's position columns are 0


class FakeRecertifyShield:
    """Test double isolating the harness's per-step recertify WIRING
    (does it call recertify() after every executed row and abandon the
    rest of the chunk on failure) from shield MATH correctness (already
    covered by tests/test_arm_shield.py's ArmSTLMonitorShield/recertify
    tests). select() always executes candidates[0] with everything
    admissible. recertify() returns False exactly once, on its `fail_at`-th
    call (0-indexed) -- True on every other call."""

    def __init__(self, fail_at):
        self.fail_at = fail_at
        self.n_recertify_calls = 0

    def select(self, joint_angles, candidates, scores):
        del joint_angles, scores
        mask = [True] * len(candidates)
        return candidates[0], {"fallback": False, "n_admissible": len(candidates), "admissible_mask": mask}

    def recertify(self, joint_angles, remaining_chunk):
        del joint_angles, remaining_chunk
        call_idx = self.n_recertify_calls
        self.n_recertify_calls += 1
        return call_idx != self.fail_at


def test_shielded_subtask_recertifies_every_step_and_reproposes_early_on_failure():
    """Regression test for decoupling filter freq from policy freq
    (docs/PARAMETERS_REFERENCE.md's "tach tan suat filter khoi policy"
    entry): a shield with a `recertify` method must have it called after
    EVERY executed row of the currently-committed chunk, not only at the
    next `replan_steps` boundary -- and a single failure must abandon the
    rest of that chunk immediately (re-propose right away) instead of
    blindly finishing all `replan_steps` rows.

    FakePolicy always proposes the identical 10-row, delta=0.1 chunk
    regardless of observation, so re-proposing produces indistinguishable
    per-row motion -- what a working recertify wiring changes is *when*
    propose()/select() get called again. `fail_at=3` rejects the tail
    right after the 4th executed row (row_idx 0,1,2 pass; row_idx 3
    fails), forcing a second decision; with `ep_len=replan_steps=10`, that
    second decision's remaining 6 rows all pass recertify (call indices
    4..8, none equal to 3), so the harness makes exactly 2 decisions
    total, not 1 -- without the wiring, ep_len==replan_steps means a
    single decision would always finish the whole episode.
    """
    env = FakeEnv()
    env.reset()
    policy = FakePolicy(delta=0.1, horizon=10)
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeRecertifyShield(fail_at=3)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=None,
    )

    assert result["n_decisions"] == 2
    assert np.isclose(env.joint_angles[0], 0.1 * 10)  # all 10 rows still executed, just across 2 decisions


class FakeResolveShield:
    """Test double isolating the harness's per-step `resolve` WIRING (does
    it call resolve() after every executed row, actually EXECUTE the
    chunk resolve() returns instead of the stale original, prefer it over
    recertify when both exist, and abandon the rest of the chunk on a
    None) from shield MATH correctness (covered by tests/test_arm_shield.py's
    ArmMPCFilterShield.resolve() tests). select() always executes
    candidates[0] with everything admissible. resolve() returns the
    remaining chunk UNCHANGED for its first `switch_at` calls (0-indexed),
    then a version with a DISTINCTIVE delta (0.5, unmistakably different
    from FakePolicy's 0.1) from call `switch_at` onward -- and `None`
    exactly at `fail_at` (if given), simulating an infeasible re-solve."""

    def __init__(self, switch_at, fail_at=None):
        self.switch_at = switch_at
        self.fail_at = fail_at
        self.n_resolve_calls = 0
        self.n_recertify_calls = 0  # must stay 0 if resolve is defined -- see the harness's own priority

    def select(self, joint_angles, candidates, scores):
        del joint_angles, scores
        mask = [True] * len(candidates)
        return candidates[0], {"fallback": False, "n_admissible": len(candidates), "admissible_mask": mask}

    def resolve(self, joint_angles, remaining_chunk):
        del joint_angles
        call_idx = self.n_resolve_calls
        self.n_resolve_calls += 1
        if self.fail_at is not None and call_idx == self.fail_at:
            return None
        if call_idx >= self.switch_at:
            switched = remaining_chunk.copy()
            switched[:, 0] = 0.5
            return switched
        return remaining_chunk

    def recertify(self, joint_angles, remaining_chunk):
        del joint_angles, remaining_chunk
        self.n_recertify_calls += 1
        return True


def test_shielded_subtask_executes_resolves_result_not_the_stale_chunk():
    """Regression test for the receding-horizon resolve() wiring
    (2026-09-05, docs/PARAMETERS_REFERENCE.md's "tach tan suat filter khoi
    policy" entry): resolve()'s returned chunk must actually be what gets
    executed on later rows of this same decision, not merely checked and
    discarded the way a boolean recertify() is. Row 0..3 execute at
    FakePolicy's original delta=0.1 (the switch (at resolve call_idx=3,
    fired right after executing row_idx=3) only affects chunk[4:10]
    going forward); rows 4..9 execute at the switched delta=0.5. Also
    confirms resolve() takes priority over recertify() when a shield
    defines both -- recertify must never be called."""
    env = FakeEnv()
    env.reset()
    policy = FakePolicy(delta=0.1, horizon=10)
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeResolveShield(switch_at=3)

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=None,
    )

    assert result["n_decisions"] == 1  # never failed -- one decision covers all 10 rows
    assert shield.n_resolve_calls == 9  # called after each of the first 9 rows (row 9 has no remaining tail)
    assert shield.n_recertify_calls == 0  # resolve takes priority -- recertify never touched
    assert np.isclose(env.joint_angles[0], 0.1 * 4 + 0.5 * 6)  # rows 0-3 at 0.1, rows 4-9 at 0.5


def test_shielded_subtask_abandons_the_chunk_when_resolve_returns_none():
    """resolve() returning None (infeasible re-solve) must be treated
    exactly like recertify() returning False: abandon the rest of this
    chunk immediately, re-propose right away -- not silently execute more
    rows of a chunk resolve() itself just said it couldn't re-certify."""
    env = FakeEnv()
    env.reset()
    policy = FakePolicy(delta=0.1, horizon=10)
    task_oracle = FakeTaskOracleWithTasks(success_after_steps=10**9, tasks={})
    shield = FakeResolveShield(switch_at=10**9, fail_at=3)  # never switches, fails right after row_idx=3

    result = run_calvin_shielded_subtask(
        env, policy, task_oracle, FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS, shield,
        ep_len=10, replan_steps=10, obstacle_fn=None,
    )

    assert result["n_decisions"] == 2  # abandoned after row_idx=3, forcing a second decision
    assert np.isclose(env.joint_angles[0], 0.1 * 10)  # all 10 rows still executed, just across 2 decisions
