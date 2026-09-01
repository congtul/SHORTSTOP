"""Confirms MPCFilterShield/CBFShield's action_dim parametrization (added
for Stage 7a's 3D arm task-space) actually works in 3D, not just "doesn't
break the existing 2D tests" -- ConfThreshShield/STLMonitorShield need no
such test since they were already dimension-agnostic (no action_dim at
all, see shortstop/baselines.py's module docstring reasoning)."""
import numpy as np

from shortstop.baselines import CBFShield, MPCFilterShield
from shortstop.env import Obstacle


def test_mpc_filter_shield_corrects_a_3d_action_around_a_3d_obstacle():
    obstacle = Obstacle(center=[0.12, 0.0, 0.0], radius=0.05)
    shield = MPCFilterShield(
        goal=[5.0, 0.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0, action_dim=3,
    )
    chunk = np.tile([1.0, 0.0, 0.0], (4, 1))

    action, info = shield.select([0.0, 0.0, 0.0], [chunk])
    assert info["intervened"]
    assert action.shape == (4, 3)
    corrected_next = np.array([0.0, 0.0, 0.0]) + action[0] * 0.1
    assert np.linalg.norm(corrected_next - obstacle.center) >= obstacle.radius - 1e-6


def test_cbf_shield_pushes_away_in_3d():
    obstacle = Obstacle(center=[0.05, 0.0, 0.0], radius=0.1)
    alpha = 1.0
    shield = CBFShield(
        goal=[5.0, 0.0, 0.0], obstacles=[obstacle], dt=0.1, w_bar=0.0, alpha=alpha, action_dim=3,
    )
    chunk = np.tile([1.0, 0.0, 0.0], (4, 1))
    state = np.array([0.0, 0.0, 0.0])

    action, info = shield.select(state, [chunk])
    assert info["intervened"]
    diff = state - obstacle.center
    h = float(diff @ diff - obstacle.radius ** 2)
    grad_h = 2.0 * diff
    assert grad_h @ action[0] >= -alpha * h - 1e-6
