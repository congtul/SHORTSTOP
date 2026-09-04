import numpy as np

from shortstop.arm_reach import propagate_arm_tube
from shortstop.arm_shield import ArmConfThreshShield, ArmReachOnlyShield, ArmRepairShield, ArmSTLShield
from shortstop.env import Obstacle
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS


def _straight_chunk(dx, horizon=4):
    step = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.tile(step, (horizon, 1))


def test_arm_reach_only_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = np.zeros(N_JOINTS)
    unsafe = _straight_chunk(0.05)
    safe = _straight_chunk(-0.05)

    # obstacle placed where the "unsafe" chunk's gripper ends up
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import FLANGE_FRAME_INDEX
    tube = propagate_arm_tube(q, unsafe, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmReachOnlyShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_arm_stl_shield_rejects_within_margin_even_if_reach_only_would_accept():
    q = np.zeros(N_JOINTS)
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import FLANGE_FRAME_INDEX
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    gripper_end = tube[-1][FLANGE_FRAME_INDEX].center()

    # obstacle just outside the true collision radius but inside STL's margin
    obstacle = Obstacle(center=gripper_end, radius=0.02)
    shield = ArmSTLShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.05)
    _, info = shield.select(q, [chunk], scores=[1.0])
    assert info["admissible_mask"] == [False]


def test_arm_repair_shield_fixes_a_rejected_candidate_and_still_certifies_it():
    q = np.zeros(N_JOINTS)
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import FLANGE_FRAME_INDEX
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

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
    from shortstop.robot_geometry import FLANGE_FRAME_INDEX
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    # tiny trust region + tiny step -> repair can't move far enough to clear
    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=1e-6, step_size=1e-6, max_repair_iters=1,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])
    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(chunk))


def _endpoint(q, chunk):
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    return tube[-1][FLANGE_FRAME_INDEX].center()


def test_arm_conf_thresh_shield_rejects_the_outlier_but_keeps_the_agreeing_candidates():
    q = np.zeros(N_JOINTS)
    agree_a = _straight_chunk(0.05)
    agree_b = _straight_chunk(0.05001)  # same direction, tiny float variation -> near-identical endpoint
    outlier = _straight_chunk(-0.05)  # opposite direction -> far from the other two

    distances_from_centroid = [
        np.linalg.norm(e - np.mean([_endpoint(q, c) for c in (agree_a, agree_b, outlier)], axis=0))
        for e in [_endpoint(q, c) for c in (agree_a, agree_b, outlier)]
    ]
    threshold = (max(distances_from_centroid[:2]) + distances_from_centroid[2]) / 2

    shield = ArmConfThreshShield(disagreement_threshold=threshold, replan_steps=4)
    action, info = shield.select(q, [agree_a, agree_b, outlier], scores=[1.0, 2.0, 3.0])

    assert info["admissible_mask"] == [True, True, False]
    assert np.allclose(action, agree_b)  # admissible with the higher score (2.0 > 1.0)


def test_arm_conf_thresh_shield_picks_highest_score_among_admissible():
    q = np.zeros(N_JOINTS)
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.05001)  # close enough to agree at a generous threshold

    shield = ArmConfThreshShield(disagreement_threshold=10.0, replan_steps=4)
    action, info = shield.select(q, [a, b], scores=[5.0, 1.0])

    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, a)


def test_arm_conf_thresh_shield_falls_back_when_every_candidate_disagrees():
    q = np.zeros(N_JOINTS)
    a = _straight_chunk(0.05)
    b = _straight_chunk(-0.05)

    shield = ArmConfThreshShield(disagreement_threshold=1e-9, replan_steps=4)
    action, info = shield.select(q, [a, b], scores=[1.0, 2.0])

    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(a))


def test_arm_conf_thresh_shield_measures_disagreement_only_over_the_first_replan_steps_rows():
    """Regression test for the replan_steps fix: `a` and `b` are IDENTICAL
    for their first 2 rows, then diverge sharply on rows 3-4 -- rows that
    would never be committed if replan_steps=2. With replan_steps=2 both
    candidates' endpoints coincide exactly (disagreement=0, trivially
    admissible at any threshold); with replan_steps=4 (the full chunk)
    they end up ~1.1m apart (disagreement>0, rejected at a tight
    threshold). Same two candidates, same threshold -- only replan_steps
    changes, so a different verdict proves the endpoint really is being
    computed over the truncated prefix, not silently still the full chunk.
    """
    q = np.zeros(N_JOINTS)
    a = np.array([[0.05, 0, 0, 0, 0, 0, 0]] * 4)
    b = np.array([[0.05, 0, 0, 0, 0, 0, 0], [0.05, 0, 0, 0, 0, 0, 0],
                  [-0.5, 0, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(a[:2], b[:2])  # identical prefix by construction
    assert not np.allclose(_endpoint(q, a), _endpoint(q, b))  # but diverge by the full chunk's end

    threshold = 0.3  # strictly between 0 (prefix disagreement) and ~0.55 (full-chunk disagreement)

    shield_prefix = ArmConfThreshShield(disagreement_threshold=threshold, replan_steps=2)
    shield_full = ArmConfThreshShield(disagreement_threshold=threshold, replan_steps=4)

    _, info_prefix = shield_prefix.select(q, [a, b], scores=[1.0, 1.0])
    _, info_full = shield_full.select(q, [a, b], scores=[1.0, 1.0])

    assert info_prefix["admissible_mask"] == [True, True]
    assert info_full["admissible_mask"] == [False, False]
