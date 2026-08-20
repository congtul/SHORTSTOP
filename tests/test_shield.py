import numpy as np

from shortstop.env import Obstacle
from shortstop.shield import ReachOnlyShield


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
