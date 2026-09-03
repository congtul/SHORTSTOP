import numpy as np

from shortstop.arm_reach import arm_find_counterexample, arm_robustness_to_go, propagate_arm_tube
from shortstop.env import Obstacle
from shortstop.robot_geometry import FLANGE_FRAME_INDEX, FRAME_RADIUS, N_JOINTS


def _zero_chunk(horizon=4, action_dim=7):
    return np.zeros((horizon, action_dim))


def test_propagate_arm_tube_zero_action_keeps_frames_at_current_position():
    q = np.random.default_rng(0).uniform(-0.3, 0.3, size=N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(), w_bar=0.0, model_error=0.0)

    assert len(tube) == 5  # horizon 4 + the initial (realized) entry
    for step in tube[1:]:
        assert set(step.keys()) == set(range(9))  # the whole chain, not a coarser subset
        for i, box in step.items():
            # zero disturbance/model_error -> the box's *center* doesn't
            # move, but it's still inflated by the frame's own physical
            # radius (see propagate_arm_tube's docstring), so it's not
            # zero-width.
            assert np.allclose(box.high - box.low, 2 * FRAME_RADIUS[i], atol=1e-9)


def test_propagate_arm_tube_inflates_with_disturbance_and_model_error():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=1), w_bar=0.05, model_error=0.02)
    box = tube[1][0]
    expected_r = 0.05 + 0.02 + FRAME_RADIUS[0]  # disturbance + model_error + own radius
    assert np.allclose(box.high - box.low, 2 * expected_r, atol=1e-9)  # inflate(r) widens by r each side


def test_arm_robustness_to_go_matches_manual_min_over_frames_and_obstacles():
    q = np.zeros(N_JOINTS)
    tube = propagate_arm_tube(q, _zero_chunk(horizon=2), w_bar=0.0, model_error=0.0)

    # obstacle placed exactly at the flange's step-1 position -> should be
    # the binding (most negative) term
    flange_pos = tube[1][FLANGE_FRAME_INDEX].center()
    obstacles = [Obstacle(center=flange_pos, radius=0.05)]

    value = arm_robustness_to_go(tube, obstacles)
    assert np.isclose(value, -0.05, atol=1e-6)  # inside by exactly the radius


def test_arm_find_counterexample_identifies_the_violating_step_and_robustness():
    q = np.zeros(N_JOINTS)
    # nonzero action so the flange actually moves step to step -- with a
    # zero action every step is a no-op and the "obstacle at step 2's
    # position" would coincide with step 1's position too, since the
    # configuration never changes. Per-step displacement (0.3) is
    # deliberately well beyond any frame's own radius (<=0.16, see
    # robot_geometry.FRAME_RADIUS) so step 1 and step 2 stay clearly
    # separated even after propagate_arm_tube's box is inflated by each
    # frame's physical radius -- a small displacement comparable to the
    # inflation amount can make the (axis-aligned-box) closest-point
    # computation degenerate and unable to tell adjacent steps apart.
    chunk = np.tile([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 1))
    tube = propagate_arm_tube(q, chunk, w_bar=0.0, model_error=0.0)
    flange_pos = tube[2][FLANGE_FRAME_INDEX].center()
    obstacles = [Obstacle(center=flange_pos, radius=0.05)]

    ce = arm_find_counterexample(tube, obstacles)
    assert ce["step"] == 2
    assert np.isclose(ce["robustness"], -0.05, atol=1e-6)
    # NOT asserting ce["frame"] == FLANGE_FRAME_INDEX: frame 7 (wrist) sits
    # only ~0.107m from frame 8 (flange, see FLANGE_OFFSET) -- closer than
    # the sum of their own FRAME_RADIUS inflation (0.11 + 0.16) -- so their
    # inflated boxes overlap here and either can legitimately be reported
    # as the (tied) worst violator. Same box-degeneracy caveat this
    # module's docstring already documents for *steps*, now also possible
    # *between adjacent frames* now that the chain checks all 9 of them
    # instead of 4 sparse, widely-separated points.
    assert ce["frame"] in (7, FLANGE_FRAME_INDEX)
