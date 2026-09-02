import numpy as np

from shortstop.env import Obstacle
from shortstop.keypose_reach import (
    inverse_kinematics_position,
    path_find_counterexample,
    path_robustness_to_go,
    propagate_path_tube,
)
from shortstop.planner import mock_get_path
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES, panda_frames


def test_inverse_kinematics_position_converges_to_a_reachable_target():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    target = current + np.array([0.05, 0.02, -0.03])

    q_solved = inverse_kinematics_position(target, q0)
    achieved = panda_frames(q_solved)[-1]
    assert np.linalg.norm(achieved - target) < 1e-3


def test_propagate_path_tube_starts_at_current_and_matches_path_length():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    target = current + np.array([0.05, 0.0, 0.0])
    path_points = mock_get_path(q0, target, n_waypoints=5)

    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    assert len(tube) == 5
    assert np.allclose(tube[0][SPHERE_NAMES[-1]].center(), current)


def test_path_robustness_to_go_detects_an_obstacle_on_the_path():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    target = current + np.array([0.1, 0.0, 0.0])
    path_points = mock_get_path(q0, target, n_waypoints=6)
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)

    midpoint = tube[3][SPHERE_NAMES[-1]].center()
    obstacle = Obstacle(center=midpoint, radius=0.02)

    value = path_robustness_to_go(tube, [obstacle])
    assert value < 0.0

    ce = path_find_counterexample(tube, [obstacle])
    assert ce["step"] == 3
    assert ce["sphere"] == SPHERE_NAMES[-1]
