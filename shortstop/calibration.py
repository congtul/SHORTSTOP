"""Table VII's calibration recipe: "Model-error quantile: 99th (x1.25)".

The paper does not hand the shield the true disturbance/model-error bound --
in the field you cannot know that in advance for a frozen, already-deployed
policy. Instead it calibrates the certified error radius from a high
quantile of residuals observed on held-out transitions, scaled up by a
safety factor for extra margin.

This 2D prototype's f_hat already equals the true dynamics exactly (see
reach.propagate's docstring), so there is no model-mismatch term to
calibrate -- the one quantity the shield does *not* actually know a priori
is the disturbance bound w_bar itself (shortstop.experiment.run_episode
currently hands the shield the ground-truth w_bar as a convenience). This
module estimates it the way the paper estimates epsilon: run held-out
episodes under the real (noisy) env, measure the realized per-step
disturbance ||x_{t+1} - (x_t + a_t*dt)||, and take a high quantile of that,
scaled by a safety factor.
"""
import numpy as np


def collect_disturbance_residuals(make_env, n_episodes, rng):
    """Roll out `make_env()` under random actions and record the realized
    per-step disturbance magnitude ||x_{t+1} - (x_t + a_t*dt)|| -- the part
    of the transition the shield's nominal model cannot predict.
    """
    residuals = []
    for _ in range(n_episodes):
        env = make_env()
        state = env.reset()
        done = False
        while not done:
            action = rng.uniform(-env.max_action_norm, env.max_action_norm, size=2)
            predicted = np.clip(state + action * env.dt, env.low, env.high)
            state, done, _ = env.step(action)
            residuals.append(float(np.linalg.norm(state - predicted)))
    return np.asarray(residuals)


def calibrate_w_bar(make_env, n_episodes=200, quantile=0.99, safety_factor=1.25, rng=None):
    """Table VII: eps = quantile(residuals) * safety_factor, applied here to
    the disturbance bound the shield certifies against instead of the true
    (privileged) w_bar. Returns a float usable directly as a shield's
    `w_bar` argument.
    """
    rng = rng if rng is not None else np.random.default_rng()
    residuals = collect_disturbance_residuals(make_env, n_episodes, rng)
    return float(np.quantile(residuals, quantile) * safety_factor)
