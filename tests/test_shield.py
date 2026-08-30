import numpy as np

from shortstop.env import Obstacle
from shortstop.shield import CEShield, ReachOnlyShield, RepairShield, STLShield


def test_shield_rejects_chunk_through_obstacle():
    obstacle = Obstacle(center=[0.5, 0.0], radius=0.3)
    shield = ReachOnlyShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)

    unsafe_chunk = np.tile([5.0, 0.0], (8, 1))
    safe_chunk = np.tile([0.0, 5.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [unsafe_chunk, safe_chunk])
    assert not info["fallback"]
    assert np.allclose(action, safe_chunk)


def test_shield_falls_back_when_all_unsafe():
    obstacle = Obstacle(center=[0.0, 0.0], radius=5.0)
    shield = ReachOnlyShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)

    chunk = np.tile([1.0, 0.0], (8, 1))
    action, info = shield.select([0.0, 0.0], [chunk])
    assert info["fallback"]
    assert np.allclose(action, np.zeros_like(chunk))


def test_stl_shield_rejects_within_margin_even_without_direct_intersection():
    """Stage 2 vs. Stage 1: a chunk that literally never touches the obstacle
    (gap 0.05) should still be rejected once a 0.1 safety margin is required.
    """
    obstacle = Obstacle(center=[0.6, 0.35], radius=0.3)
    chunk = np.tile([1.0, 0.0], (8, 1))

    reach_only = ReachOnlyShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    _, info = reach_only.select([0.0, 0.0], [chunk])
    assert not info["fallback"]

    stl_shield = STLShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0, epsilon=0.1)
    _, info = stl_shield.select([0.0, 0.0], [chunk])
    assert info["fallback"]


def test_ce_shield_reports_counterexample_for_rejected_chunk_only():
    obstacle = Obstacle(center=[0.5, 0.0], radius=0.3)
    shield = CEShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0, epsilon=0.0)

    unsafe_chunk = np.tile([1.0, 0.0], (8, 1))
    safe_chunk = np.tile([0.0, 1.0], (8, 1))
    _, info = shield.select([0.0, 0.0], [unsafe_chunk, safe_chunk])

    assert info["counterexamples"][0] is not None
    assert info["counterexamples"][0]["obstacle"] is obstacle
    assert info["counterexamples"][1] is None


def test_repair_shield_fixes_a_near_miss_chunk():
    """Algorithm 1 only takes *one* gradient step per rejected candidate, so
    this obstacle placement must be fixable in a single step for the test to
    be meaningful -- not something requiring multiple rounds.
    """
    obstacle = Obstacle(center=[0.5, 0.15], radius=0.3)
    shield = RepairShield(
        goal=[5.0, 0.0],
        obstacles=[obstacle],
        dt=0.1,
        w_bar=0.0,
        epsilon=0.05,
        trust_region=0.6,
    )
    chunk = np.tile([1.0, 0.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert not info["fallback"]
    assert info["repair_attempted"]
    assert info["repair_succeeded"]
    assert not np.allclose(action, chunk)  # actually got modified


def test_repair_shield_default_matches_algorithm_1_single_shot():
    """Algorithm 1 gives up after one failed repair attempt -- no retry.
    This obstacle placement needs >=2 rounds to fix, so the paper-faithful
    default (max_repair_iters=1) must fail it, while explicitly asking for
    more rounds (a CEGIS-style extension beyond the paper) succeeds.
    """
    obstacle = Obstacle(center=[0.4, 0.15], radius=0.3)
    chunk = np.tile([1.0, 0.0], (8, 1))

    default_shield = RepairShield(
        goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0,
        epsilon=0.05, trust_region=0.6,
    )
    _, info = default_shield.select([0.0, 0.0], [chunk])
    assert info["fallback"]
    assert not info["repair_succeeded"]

    multi_round_shield = RepairShield(
        goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0,
        epsilon=0.05, trust_region=0.6, max_repair_iters=3,
    )
    _, info = multi_round_shield.select([0.0, 0.0], [chunk])
    assert not info["fallback"]
    assert info["repair_succeeded"]


def test_repair_shield_falls_back_when_obstacle_unavoidable():
    obstacle = Obstacle(center=[0.0, 0.0], radius=5.0)  # engulfs every direction
    shield = RepairShield(goal=[5.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0)
    chunk = np.tile([1.0, 0.0], (8, 1))

    action, info = shield.select([0.0, 0.0], [chunk])

    assert info["fallback"]
    assert info["repair_attempted"]
    assert not info["repair_succeeded"]
    assert np.allclose(action, np.zeros_like(chunk))
