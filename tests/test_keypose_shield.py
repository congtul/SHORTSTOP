import numpy as np

from shortstop.env import Obstacle
from shortstop.keypose_reach import propagate_path_tube
from shortstop.keypose_shield import KeyposeReachOnlyShield, KeyposeRepairShield, KeyposeSTLShield
from shortstop.planner import mock_get_path
from shortstop.robot_geometry import N_JOINTS, SPHERE_NAMES, panda_frames


def _keypose(position):
    return np.concatenate([position, [0.0, 0.0, 0.0, 1.0], [1.0]])  # pos, identity quat, gripper open


def test_keypose_reach_only_shield_rejects_a_target_that_hits_an_obstacle():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    unsafe = _keypose(current + np.array([0.1, 0.0, 0.0]))
    safe = _keypose(current + np.array([-0.1, 0.0, 0.0]))

    path_points = mock_get_path(q0, unsafe[:3])
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    shield = KeyposeReachOnlyShield(obstacles=[obstacle], w_bar=0.0, planner_fn=mock_get_path, model_error=0.0)
    action, info = shield.select(q0, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_keypose_stl_shield_rejects_within_margin():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    keypose = _keypose(current + np.array([0.1, 0.0, 0.0]))
    path_points = mock_get_path(q0, keypose[:3])
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    gripper_end = tube[-1][SPHERE_NAMES[-1]].center()

    obstacle = Obstacle(center=gripper_end, radius=0.02)
    shield = KeyposeSTLShield(
        obstacles=[obstacle], w_bar=0.0, planner_fn=mock_get_path, model_error=0.0, epsilon=0.05,
    )
    _, info = shield.select(q0, [keypose], scores=[1.0])
    assert info["admissible_mask"] == [False]


def test_keypose_repair_shield_fixes_a_rejected_target_by_replanning():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    keypose = _keypose(current + np.array([0.1, 0.0, 0.0]))
    path_points = mock_get_path(q0, keypose[:3])
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    shield = KeyposeRepairShield(
        obstacles=[obstacle], w_bar=0.0, planner_fn=mock_get_path, model_error=0.0, epsilon=0.02,
        trust_region=0.2, step_size=0.1, max_repair_iters=3,
    )
    action, info = shield.select(q0, [keypose], scores=[1.0])

    assert info["repair_attempted"]
    assert info["repair_succeeded"]
    assert not np.allclose(action, keypose)
    assert np.allclose(action[3:], keypose[3:])  # only position (0:3) was touched


def test_keypose_repair_shield_calls_planner_fn_again_for_each_repair_attempt():
    """The whole point of the v2 design: repair must re-plan, not edit a
    path in place -- verify planner_fn is actually invoked more than once
    (once for the initial certify, at least once more per repair
    iteration) rather than reusing a cached path."""
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    keypose = _keypose(current + np.array([0.1, 0.0, 0.0]))
    path_points = mock_get_path(q0, keypose[:3])
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    call_count = {"n": 0}

    def counting_planner(joint_angles, target_position):
        call_count["n"] += 1
        return mock_get_path(joint_angles, target_position)

    shield = KeyposeRepairShield(
        obstacles=[obstacle], w_bar=0.0, planner_fn=counting_planner, model_error=0.0, epsilon=0.02,
        trust_region=0.2, step_size=0.1, max_repair_iters=3,
    )
    shield.select(q0, [keypose], scores=[1.0])
    assert call_count["n"] >= 2  # initial certify + at least one repair replan


def test_keypose_repair_shield_falls_back_when_it_cannot_fix_in_time():
    q0 = np.zeros(N_JOINTS)
    current = panda_frames(q0)[-1]
    keypose = _keypose(current + np.array([0.1, 0.0, 0.0]))
    path_points = mock_get_path(q0, keypose[:3])
    tube = propagate_path_tube(path_points, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    shield = KeyposeRepairShield(
        obstacles=[obstacle], w_bar=0.0, planner_fn=mock_get_path, model_error=0.0, epsilon=0.02,
        trust_region=1e-6, step_size=1e-6, max_repair_iters=1,
    )
    action, info = shield.select(q0, [keypose], scores=[1.0])
    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(keypose))
