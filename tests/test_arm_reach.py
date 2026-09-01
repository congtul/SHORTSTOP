import numpy as np

from shortstop.arm_reach import arm_find_counterexample, arm_robustness_to_go, propagate_arm_tube
from shortstop.env import Obstacle
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES


def _zero_chunk(horizon=4, action_dim=7):
    return np.zeros((horizon, action_dim))


def test_propagate_arm_tube_zero_action_keeps_spheres_at_current_position():
    q = np.random.default_rng(0).uniform(-0.3, 0.3, size=N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(), w_bar=0.0, model_error=0.0)

    assert len(tube) == 5  # horizon 4 + the initial (realized) entry
    for step in tube[1:]:
        assert set(step.keys()) == set(SPHERE_NAMES)
        for box in step.values():
            assert np.allclose(box.low, box.high, atol=1e-9)  # zero disturbance -> zero-width


def test_propagate_arm_tube_inflates_with_disturbance_and_model_error():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=1), w_bar=0.05, model_error=0.02)
    box = tube[1][SPHERE_NAMES[0]]
    assert np.allclose(box.high - box.low, 2 * 0.07, atol=1e-9)  # inflate(r) widens by r each side


def test_arm_robustness_to_go_matches_manual_min_over_spheres_and_obstacles():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=2), w_bar=0.0, model_error=0.0)

    # obstacle placed exactly at the gripper's step-1 position -> should be
    # the binding (most negative) term
    gripper_pos = tube[1][SPHERE_NAMES[-1]].center()
    obstacles = [Obstacle(center=gripper_pos, radius=0.05)]

    value = arm_robustness_to_go(tube, obstacles)
    assert np.isclose(value, -0.05, atol=1e-6)  # inside by exactly the radius


def test_arm_find_counterexample_identifies_the_violating_sphere_and_step():
    q = np.zeros(N_JOINTS)
    # nonzero action so the gripper actually moves step to step -- with a
    # zero action every step is a no-op and the "obstacle at step 2's
    # position" would coincide with step 1's position too, since the
    # configuration never changes.
    chunk = np.tile([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1))
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    gripper_pos = tube[2][SPHERE_NAMES[-1]].center()
    obstacles = [Obstacle(center=gripper_pos, radius=0.05)]

    ce = arm_find_counterexample(tube, obstacles)
    assert ce["step"] == 2
    assert ce["sphere"] == SPHERE_NAMES[-1]
    assert np.isclose(ce["robustness"], -0.05, atol=1e-6)
