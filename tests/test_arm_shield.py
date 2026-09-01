import numpy as np

from shortstop.arm_shield import ArmReachOnlyShield, ArmRepairShield, ArmSTLShield
from shortstop.env import Obstacle
from shortstop.robot_geometry import N_JOINTS


def _straight_chunk(dx, horizon=4):
    step = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.tile(step, (horizon, 1))


def test_arm_reach_only_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = np.zeros(N_JOINTS)
    unsafe = _straight_chunk(0.05)
    safe = _straight_chunk(-0.05)

    # obstacle placed where the "unsafe" chunk's gripper ends up
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import SPHERE_NAMES
    tube = propagate_arm_tube(q, unsafe, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    shield = ArmReachOnlyShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_arm_stl_shield_rejects_within_margin_even_if_reach_only_would_accept():
    q = np.zeros(N_JOINTS)
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import SPHERE_NAMES
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    gripper_end = tube[-1][SPHERE_NAMES[-1]].center()

    # obstacle just outside the true collision radius but inside STL's margin
    obstacle = Obstacle(center=gripper_end, radius=0.02)
    shield = ArmSTLShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.05)
    _, info = shield.select(q, [chunk], scores=[1.0])
    assert info["admissible_mask"] == [False]


def test_arm_repair_shield_fixes_a_rejected_candidate_and_still_certifies_it():
    q = np.zeros(N_JOINTS)
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import SPHERE_NAMES
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=0.2, step_size=0.1, max_repair_iters=3,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["repair_attempted"]
    assert info["repair_succeeded"]
    assert info["admissible_mask"] == [True]
    assert not np.allclose(action, chunk)  # actually got modified


def test_arm_repair_shield_falls_back_when_repair_cannot_fix_it_in_time():
    q = np.zeros(N_JOINTS)
    chunk = _straight_chunk(0.05)
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import SPHERE_NAMES
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][SPHERE_NAMES[-1]].center(), radius=0.05)

    # tiny trust region + tiny step -> repair can't move far enough to clear
    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=1e-6, step_size=1e-6, max_repair_iters=1,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])
    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(chunk))
