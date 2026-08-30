import numpy as np

from shortstop.calibration import calibrate_w_bar, collect_disturbance_residuals
from shortstop.env import ReachAvoid2D


def _make_env(w_bar):
    return ReachAvoid2D(
        start=[0.0, 0.0], goal=[5.0, 0.0], obstacles=[], dt=0.1, w_bar=w_bar, max_steps=5
    )


def test_collect_disturbance_residuals_bounded_by_true_w_bar():
    """Every residual is exactly ||w_t||, w_t ~ Uniform(-w_bar, w_bar)^2, so
    none can exceed the true w_bar's diagonal (sqrt(2) * w_bar).
    """
    rng = np.random.default_rng(0)
    residuals = collect_disturbance_residuals(lambda: _make_env(0.1), n_episodes=50, rng=rng)

    assert len(residuals) > 0
    assert np.all(residuals <= 0.1 * np.sqrt(2) + 1e-9)
    assert np.all(residuals >= 0.0)


def test_calibrate_w_bar_tracks_the_true_disturbance_scale():
    """The calibrated estimate should land in the right ballpark of the true
    w_bar (a high quantile times a safety factor > 1), not be wildly off in
    either direction.
    """
    rng = np.random.default_rng(0)
    true_w_bar = 0.1
    calibrated = calibrate_w_bar(
        lambda: _make_env(true_w_bar), n_episodes=200, quantile=0.99, safety_factor=1.25, rng=rng
    )

    assert 0.5 * true_w_bar < calibrated < 2.0 * true_w_bar
