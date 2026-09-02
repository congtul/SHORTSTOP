"""Exercises shortstop.calvin_experiment's control-flow logic (replan
cycling, ground-truth violation check, stop-on-violation, task_oracle
success check) against fakes standing in for HulcWrapper/task_oracle/
lang_embeddings -- no real CALVIN/mdt_policy session involved. See
docs/CALVIN_SETUP.md for the real interface this mirrors.
"""
import numpy as np
import torch

from shortstop.calvin_experiment import run_calvin_unshielded_sequence, run_calvin_unshielded_subtask
from shortstop.calvin_metrics import build_fixed_cohort_slots, fixed_cohort_rates
from shortstop.env import Obstacle
from shortstop.robot_geometry import N_JOINTS, sphere_centers

SUBTASK = "fake_subtask"


class FakeEnv:
    """joint_angles[0] += action[0] each step -- just enough dynamics to
    move the sphere-chain in a controlled, predictable way."""

    def __init__(self):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0

    def reset(self, robot_obs=None, scene_obs=None):
        self.joint_angles = np.zeros(N_JOINTS)
        self.step_count = 0
        return self.get_obs()

    def get_obs(self):
        robot_obs_raw = np.zeros(15, dtype=np.float32)
        robot_obs_raw[7:14] = self.joint_angles
        return {"robot_obs_raw": torch.tensor(robot_obs_raw)}

    def get_info(self):
        return {"step_count": self.step_count}

    def step(self, action_tensor):
        action = action_tensor.detach().cpu().numpy()
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


class FakeLangEmbeddings:
    def get_lang_goal(self, task):
        return {"lang": task}


VAL_ANNOTATIONS = {SUBTASK: [SUBTASK]}


def test_unshielded_subtask_succeeds_when_never_flagged():
    env = FakeEnv()
    env.reset()
    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS,
        ep_len=360, replan_steps=10, obstacle=None,
    )
    assert result == {"violated": False, "reached": True}


def test_unshielded_subtask_stops_at_violation_and_never_reaches():
    env = FakeEnv()
    env.reset()
    # obstacle placed exactly where the arm is after 6 steps -- well
    # before the task_oracle would succeed at step 12
    obstacle_joint_angles = np.zeros(N_JOINTS)
    obstacle_joint_angles[0] = 0.1 * 6
    obstacle = Obstacle(center=sphere_centers(obstacle_joint_angles)[-1], radius=0.02)

    result = run_calvin_unshielded_subtask(
        env, FakePolicy(), FakeTaskOracle(success_after_steps=12), FakeLangEmbeddings(), SUBTASK, VAL_ANNOTATIONS,
        ep_len=360, replan_steps=10, obstacle=obstacle,
    )
    assert result["violated"] is True
    assert result["reached"] is False  # violated => never gets a chance to also succeed


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
    obstacle = Obstacle(center=sphere_centers(obstacle_joint_angles)[-1], radius=0.02)

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
