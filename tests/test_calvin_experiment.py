"""Exercises shortstop.calvin_experiment's control-flow logic (replan
cycling, ground-truth violation check, stop-on-violation, task_oracle
success check) against fakes standing in for HulcWrapper/task_oracle/
lang_embeddings -- no real CALVIN/mdt_policy session involved. See
docs/CALVIN_SETUP.md for the real interface this mirrors.
"""
import functools

import numpy as np
import torch

from shortstop.arm_shield import ArmConfThreshShield
from shortstop.arm_reach import propagate_arm_tube
from shortstop.calvin_experiment import (
    run_calvin_shielded_subtask,
    run_calvin_unshielded_sequence,
    run_calvin_unshielded_subtask,
)
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates
from shortstop.env import Obstacle
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS, panda_frames

SUBTASK = "fake_subtask"


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
        self.scene_positions = {}
        self.last_gripper_action = None

    def reset(self, robot_obs=None, scene_obs=None):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        robot_obs_raw = np.zeros(15, dtype=np.float32)
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
    assert result == {"violated": False, "reached": True, "min_clearance": None}


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


def test_shielded_subtask_fallback_holds_the_gripper_instead_of_forcing_it_closed():
    """Regression test: shield.select()'s fallback action is
    np.zeros_like(candidates[0]) -- the harness must overwrite its gripper
    column with the real current gripper_action before stepping, or a
    fallback would (via HulcWrapper.step()'s real binarization, not
    exercised by this fake env) silently force the gripper closed
    regardless of its actual prior state. Checked here at the harness
    layer directly: whatever env.step() actually received."""
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

    assert result["n_activated"] == 1
    assert np.isclose(env.last_gripper_action, 0.42)  # held (float32 roundtrip), not forced to 0/-1
    assert env.joint_angles[0] == 0.0  # froze in place -- fallback's position columns are 0
