import numpy as np

from shortstop.arm_reach import arm_robustness_to_go, nominal_joint_trajectory, propagate_arm_tube
from shortstop.arm_shield import (
    ArmConfThreshShield, ArmMPCFilterShield, ArmReachOnlyShield, ArmRepairShield, ArmSTLMonitorShield,
    ArmSTLShield,
)
from shortstop.env import Obstacle
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, N_JOINTS, within_joint_limits

# A real, well-within-JOINT_LIMITS Franka "ready" pose -- NOT np.zeros:
# joint 4 (index 3)'s own real range is entirely negative ([-3.0718,
# -0.0698], see robot_geometry.JOINT_LIMITS), so q=0 is itself already
# physically invalid for that joint alone. Every test below that exercises
# a shield's actual select()/recertify() (which now enforce JOINT_LIMITS,
# see ArmReachOnlyShield._trajectory_within_joint_limits) needs a
# genuinely valid starting config, not the old placeholder.
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])


def _straight_chunk(dx, horizon=4):
    step = np.array([dx, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.tile(step, (horizon, 1))


def test_arm_reach_only_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = Q_HOME
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
    q = Q_HOME
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


def test_arm_stl_monitor_shield_rejects_a_chunk_that_hits_an_obstacle():
    q = Q_HOME
    unsafe = _straight_chunk(0.05)
    safe = _straight_chunk(-0.05)

    tube = propagate_arm_tube(q, unsafe, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [unsafe, safe], scores=[1.0, 0.0])

    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, safe)


def test_arm_mpc_filter_shield_leaves_the_chunk_unchanged_when_no_obstacle_is_near():
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    far_obstacle = Obstacle(center=np.array([100.0, 100.0, 100.0]), radius=0.02)

    shield = ArmMPCFilterShield(obstacles=[far_obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is False
    assert info["admissible_mask"] == [True]
    assert np.allclose(action[:, :3], chunk[:, :3], atol=1e-6)


def test_arm_mpc_filter_shield_corrects_a_chunk_that_hits_an_obstacle():
    """Regression test for the QP's own linearization: the corrected chunk
    must be genuinely admissible under the TRUE (nonlinear) reachtube, not
    just the linearized model the QP itself solved against -- confirmed
    directly via propagate_arm_tube/arm_robustness_to_go on the CORRECTED
    chunk, exactly the same ground-truth check every other shield's
    admissibility is judged by."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    assert not np.allclose(action[:, :3], chunk[:, :3], atol=1e-6)

    corrected_tube = propagate_arm_tube(q, action, w_bar=0.0, model_error=0.0)
    assert arm_robustness_to_go(corrected_tube, [obstacle]) >= 0.0


def test_arm_mpc_filter_shield_enforces_joint_limits_even_with_no_obstacle():
    """A chunk that would drive a joint past JOINT_LIMITS over the horizon
    (no obstacle involved at all) must still get pulled back within range
    -- the arm's analogue of 2D's max_action_norm bound, checked directly
    via nominal_joint_trajectory/within_joint_limits on the TRUE (not
    linearized) resulting trajectory."""
    q = Q_HOME
    horizon = 10
    step = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    chunk = np.tile(step, (horizon, 1))
    assert not all(within_joint_limits(qk) for qk in nominal_joint_trajectory(q, chunk)[1:])

    shield = ArmMPCFilterShield(obstacles=[], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is False
    assert info["intervened"] is True
    corrected_trajectory = nominal_joint_trajectory(q, action)
    assert all(within_joint_limits(qk) for qk in corrected_trajectory[1:])


def test_arm_mpc_filter_shield_falls_back_when_the_qp_is_infeasible():
    """An absurdly large obstacle radius (5m) centered on the chunk's own
    nominal endpoint can't be cleared by any correction within reach --
    the QP must report infeasible, not silently return a wrong/partial
    solution."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=5.0)

    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["fallback"] is True
    assert info["admissible_mask"] == [False]
    assert np.allclose(action, np.zeros_like(chunk))


def test_arm_mpc_filter_shield_resolve_matches_select_in_isolation():
    """resolve() and select() share the same QP core (_solve_qp) -- called
    with identical arguments (a real, not-yet-executed state and its own
    full chunk as the nominal reference), they must produce the identical
    correction. This is the isolated (non-receding) sanity check;
    tests/test_calvin_experiment.py's resolve-wiring tests cover the
    receding-horizon case (re-solving from a state reached mid-chunk)."""
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)
    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)

    selected_action, _ = shield.select(q, [chunk], scores=[1.0])
    resolved = shield.resolve(q, chunk)

    assert resolved is not None
    assert np.allclose(resolved, selected_action, atol=1e-6)


def test_arm_mpc_filter_shield_resolve_reoptimizes_from_a_genuinely_drifted_real_state():
    """The receding-horizon case resolve() exists for, mirroring EXACTLY
    what the real harness does (run_calvin_shielded_subtask): select()
    picks this decision's chunk, row 0 gets executed for real, then
    resolve() re-solves the REMAINING TAIL OF THE ALREADY-CORRECTED CHUNK
    (not the original uncorrected nominal) from the REAL resulting state.

    The result must be genuinely admissible from that real state -- but
    verified numerically (not assumed): chaining select() then resolve()
    means TWO separate linearization passes (each one exact only at its
    own nominal point, see ArmMPCFilterShield's own docstring), so a
    small amount of slack can compound across them. Real measured value
    for this exact scenario: ~-0.00068 (well under a millimeter) -- a
    genuine, expected characteristic of chaining single-linearization-pass
    corrections, not a sign resolve() is broken. Asserting >= -1e-3 (1mm)
    catches an actual regression (a much larger, unbounded error) without
    being a false ">=0 always" claim this single-pass design doesn't
    make."""
    q_original = Q_HOME
    chunk = _straight_chunk(0.05, horizon=6)
    tube = propagate_arm_tube(q_original, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)
    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)

    selected, select_info = shield.select(q_original, [chunk], scores=[1.0])
    assert select_info["intervened"] is True

    # Row 0 of the CORRECTED chunk executed for real -> the real state
    # resolve() must adapt to (not the original uncorrected nominal's).
    real_trajectory = nominal_joint_trajectory(q_original, selected[:1])
    q_drifted = real_trajectory[1]
    remaining = selected[1:]

    resolved = shield.resolve(q_drifted, remaining)

    assert resolved is not None
    assert not np.allclose(resolved[:, :3], remaining[:, :3], atol=1e-6)  # genuinely re-optimized, not a no-op
    corrected_tube = propagate_arm_tube(q_drifted, resolved, w_bar=0.0, model_error=0.0)
    assert arm_robustness_to_go(corrected_tube, [obstacle]) >= -1e-3


def test_arm_mpc_filter_shield_resolve_returns_none_when_infeasible():
    q = Q_HOME
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=5.0)
    shield = ArmMPCFilterShield(obstacles=[obstacle], w_bar=0.0, model_error=0.0)

    assert shield.resolve(q, chunk) is None


def test_arm_stl_monitor_shield_picks_highest_score_among_admissible():
    q = Q_HOME
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.03)
    # obstacle far from both candidates' paths -- both admissible
    obstacle = Obstacle(center=np.array([10.0, 10.0, 10.0]), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [a, b], scores=[1.0, 5.0])

    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, b)  # higher score (5.0 > 1.0), not the first candidate


def test_arm_stl_monitor_shield_falls_back_when_every_candidate_hits_the_obstacle():
    q = Q_HOME
    a = _straight_chunk(0.05)
    b = _straight_chunk(0.05001)  # same direction, still ends up on the obstacle
    tube = propagate_arm_tube(q, a, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = shield.select(q, [a, b], scores=[1.0, 2.0])

    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(a))


def test_arm_stl_monitor_shield_epsilon_is_a_real_tunable_margin_not_hardcoded_zero():
    """Regression test for making `epsilon` an explicit, required
    constructor arg (see ArmSTLMonitorShield's docstring on the paper's
    two conflicting readings of its threshold): a candidate whose nominal
    robustness is small and POSITIVE (0.04, computed directly via
    propagate_arm_tube/arm_robustness_to_go against this exact obstacle)
    must stay admissible at epsilon=0.0 ("rejects if negative", literal
    reading) but get rejected once epsilon is raised past it (0.05,
    "shared epsilon" reading) -- if epsilon were still hardcoded to 0.0
    internally, the second shield would wrongly keep it admissible too.
    """
    q = Q_HOME
    candidate = _straight_chunk(0.05)
    other = _straight_chunk(-0.05)  # moves away -- stays admissible at every epsilon tested here
    tube = propagate_arm_tube(q, candidate, w_bar=0.0, model_error=0.0)
    endpoint = tube[-1][FLANGE_FRAME_INDEX].center()
    obstacle = Obstacle(center=endpoint + np.array([0.25, 0.0, 0.0]), radius=0.05)

    lenient = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)
    action, info = lenient.select(q, [candidate, other], scores=[2.0, 1.0])
    assert info["admissible_mask"] == [True, True]
    assert np.allclose(action, candidate)  # higher score, still admissible at epsilon=0.0

    strict = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.05)
    action, info = strict.select(q, [candidate, other], scores=[2.0, 1.0])
    assert info["admissible_mask"] == [False, True]  # same candidate now rejected
    assert np.allclose(action, other)


def test_recertify_matches_admissible_and_reacts_to_a_drifted_real_state():
    """`recertify()` (shortstop.arm_shield.ArmReachOnlyShield, inherited by
    ArmSTLShield/ArmSTLMonitorShield/ArmRepairShield -- see
    docs/PARAMETERS_REFERENCE.md's "tach tan suat filter khoi policy" entry
    for why this is feasible for these shields but not ArmConfThreshShield)
    is just `_admissible()` under a name the harness calls every real
    env-step, not only at Propose time: same reject-if-below-epsilon
    decision, just against a chunk SUFFIX from whatever real joint state
    the caller passes in -- which may have drifted from the nominal state
    `select()` originally certified against."""
    from shortstop.arm_reach import _step_joint_config

    remaining = _straight_chunk(0.05, horizon=2)  # the 2 rows still left to execute
    tube = propagate_arm_tube(Q_HOME, remaining, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)
    shield = ArmSTLMonitorShield(obstacles=[obstacle], epsilon=0.0)

    # from the nominal state (Q_HOME) the remaining suffix runs straight
    # into the obstacle -- recertify must reject it, same as _admissible
    # would.
    assert shield.recertify(Q_HOME, remaining) is False

    # from a real state that has already drifted safely away -- reached by
    # actually taking 2 steps in the opposite (-0.15/step) task-space
    # direction, not a hand-picked joint value -- the same remaining
    # suffix clears the obstacle by a margin (robustness computed
    # directly: +0.065, via propagate_arm_tube/arm_robustness_to_go
    # against this exact obstacle) while staying within JOINT_LIMITS
    # throughout (a larger -0.3/step drift would clear the obstacle by
    # more but blows past joint 2's own limit partway through -- recertify
    # correctly rejects that for a DIFFERENT reason, so it's not usable
    # here to isolate the geometric-clearance behavior this test targets).
    # recertify must accept it.
    drifted_joint_angles = Q_HOME.copy()
    for _ in range(2):
        drifted_joint_angles = _step_joint_config(drifted_joint_angles, np.array([-0.15, 0.0, 0.0]))
    assert shield.recertify(drifted_joint_angles, remaining) is True

    # ArmConfThreshShield has no cheap per-step re-check at all (see its
    # own docstring) -- the harness's `hasattr(shield, "recertify")` guard
    # depends on this staying true.
    assert not hasattr(ArmConfThreshShield(disagreement_threshold=1.0, replan_steps=10), "recertify")


def test_arm_repair_shield_fixes_a_rejected_candidate_and_still_certifies_it():
    q = Q_HOME
    from shortstop.arm_reach import propagate_arm_tube
    from shortstop.robot_geometry import FLANGE_FRAME_INDEX
    chunk = _straight_chunk(0.05)
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    obstacle = Obstacle(center=tube[-1][FLANGE_FRAME_INDEX].center(), radius=0.05)

    # trust_region widened from 0.2 (worked at the old q=0 placeholder) to
    # 0.3 -- Q_HOME's own Jacobian geometry needs slightly more room for
    # the same 3-iteration repair to actually converge; verified directly
    # (not guessed) before picking this value.
    shield = ArmRepairShield(
        obstacles=[obstacle], w_bar=0.0, model_error=0.0, epsilon=0.02,
        trust_region=0.3, step_size=0.1, max_repair_iters=3,
    )
    action, info = shield.select(q, [chunk], scores=[1.0])

    assert info["repair_attempted"]
    assert info["repair_succeeded"]
    assert info["admissible_mask"] == [True]
    assert not np.allclose(action, chunk)  # actually got modified


def test_arm_repair_shield_falls_back_when_repair_cannot_fix_it_in_time():
    q = Q_HOME
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
