import numpy as np

from shortstop.baselines import CBFShield, ConfThreshShield, MPCFilterShield, STLMonitorShield
from shortstop.env import Obstacle
from shortstop.reach import nominal_rollout


def test_conf_thresh_rejects_the_outlier_candidate():
    """5 candidates cluster tightly around [1, 0]; one wild outlier heads
    the opposite way. Conf-Thresh has no obstacle/geometry knowledge at all
    -- it must reject purely because the outlier disagrees with the rest.
    """
    tight = [np.tile([1.0, 0.0], (8, 1))] * 5
    outlier = np.tile([-1.0, 1.0], (8, 1))
    candidates = tight + [outlier]

    shield = ConfThreshShield(goal=[5.0, 0.0], obstacles=[], dt=0.1, w_bar=0.0, disagreement_threshold=0.35)
    _, info = shield.select([0.0, 0.0], candidates)

    assert all(info["admissible_mask"][:5])
    assert not info["admissible_mask"][5]


def test_stl_monitor_rejects_chunk_through_obstacle_and_accepts_a_clear_miss():
    obstacle = Obstacle(center=[0.5, 0.0], radius=0.3)
    shield = STLMonitorShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)

    through = np.tile([1.0, 0.0], (8, 1))
    clear = np.tile([0.0, 1.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [through, clear])
    assert info["admissible_mask"] == [False, True]
    assert np.allclose(action, clear)


def test_stl_monitor_has_no_margin_unlike_stl_shield():
    """Table VII's shield margin (STLShield's epsilon) is a ShortStop-only
    calibration -- STL-Monitor rejects only at exactly rho < 0, so a
    near-miss chunk that grazes just outside the obstacle (rho slightly
    positive) must be accepted here even though STLShield with a real
    margin would reject it (see test_stl_shield_rejects_within_margin...
    in test_shield.py, same obstacle placement).
    """
    obstacle = Obstacle(center=[0.6, 0.35], radius=0.3)
    shield = STLMonitorShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([1.0, 0.0], (8, 1))

    _, info = shield.select([0.0, 0.0], [chunk])
    assert info["admissible_mask"] == [True]


def test_mpc_filter_corrects_the_first_action_to_clear_the_obstacle():
    obstacle = Obstacle(center=[0.12, 0.0], radius=0.05)
    shield = MPCFilterShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([1.0, 0.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert info["intervened"]
    assert not np.allclose(action[0], chunk[0])
    corrected_next = np.array([0.0, 0.0]) + action[0] * 0.1
    assert np.linalg.norm(corrected_next - obstacle.center) >= obstacle.radius - 1e-6
    # the QP optimizes the whole horizon, but only step 0's constraint binds
    # here (later nominal points are already clear of the obstacle), so
    # those entries should come back ~unchanged (up to solver tolerance)
    assert np.allclose(action[1:], chunk[1:], atol=1e-3)


def test_mpc_filter_catches_a_collision_several_steps_out():
    """The obstacle sits exactly at the predicted position of step 5 (index
    4), not step 1 -- a 1-step-lookahead filter (the earlier, simplified
    version of this class) would see step 1's prediction is clear and do
    nothing, sailing straight through the obstacle 4 steps later. The
    full-horizon QP must catch this instead.
    """
    # Offset slightly off the dead-ahead line: dead-center would need pure
    # forward speed to escape, but a_nominal's x-component is already at
    # max_action_norm (no room left to go faster) -- offsetting gives the QP
    # a lateral direction to push into instead.
    obstacle = Obstacle(center=[0.5, 0.03], radius=0.05)
    shield = MPCFilterShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([1.0, 0.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert info["intervened"]
    path = nominal_rollout([0.0, 0.0], action, 0.1)
    assert all(np.linalg.norm(p - obstacle.center) >= obstacle.radius - 1e-6 for p in path[1:])


def test_mpc_filter_leaves_a_clear_action_untouched():
    obstacle = Obstacle(center=[0.5, 0.0], radius=0.05)
    shield = MPCFilterShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([0.0, 1.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert not info["intervened"]
    assert np.allclose(action, chunk)


def test_cbf_shield_pushes_away_when_already_inside_the_unsafe_set():
    obstacle = Obstacle(center=[0.05, 0.0], radius=0.1)
    alpha = 1.0
    shield = CBFShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0, alpha=alpha)
    chunk = np.tile([1.0, 0.0], (8, 1))
    state = np.array([0.0, 0.0])

    action, info = shield.select(state, [chunk])

    assert info["intervened"]
    diff = state - obstacle.center
    h = float(diff @ diff - obstacle.radius ** 2)
    grad_h = 2.0 * diff
    assert grad_h @ action[0] >= -alpha * h - 1e-6


def test_cbf_shield_leaves_a_safe_action_untouched():
    obstacle = Obstacle(center=[5.0, 5.0], radius=0.1)  # far away, irrelevant
    shield = CBFShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([1.0, 0.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert not info["intervened"]
    assert np.allclose(action, chunk)
