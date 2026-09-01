import numpy as np


class Box:
    """Axis-aligned box (interval) over-approximation of a reachable set."""

    __slots__ = ("low", "high")

    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)

    @classmethod
    def point(cls, x):
        x = np.asarray(x, dtype=float)
        return cls(x.copy(), x.copy())

    def inflate(self, r):
        return Box(self.low - r, self.high + r)

    def shift(self, delta):
        return Box(self.low + delta, self.high + delta)

    def center(self):
        return (self.low + self.high) / 2.0

    def closest_point(self, p):
        return np.clip(np.asarray(p, dtype=float), self.low, self.high)

    def intersects_circle(self, center, radius):
        closest = self.closest_point(center)
        return np.linalg.norm(closest - np.asarray(center, dtype=float)) <= radius

    def intersects_any(self, obstacles):
        return any(self.intersects_circle(o.center, o.radius) for o in obstacles)


def propagate(box, action, dt, w_bar, model_error=0.0):
    """Formula (1): R_{k+1} = f_hat(R_k, a_{t+k}) (+) B(eps_k + w_bar).

    f_hat here is the exact point-mass model x_{t+1} = x_t + a*dt, so all box
    growth comes from the bounded disturbance w_bar plus an optional
    model-error term (model_error=0 in Phase 1, since f_hat == f exactly).
    """
    delta = np.asarray(action, dtype=float) * dt
    r = model_error + w_bar
    return box.shift(delta).inflate(r)


def propagate_tube(x0, actions, dt, w_bar, model_error=0.0):
    """Propagate a reachtube R_0..R_H from a point start x0 and an action chunk."""
    tube = [Box.point(x0)]
    for a in actions:
        tube.append(propagate(tube[-1], a, dt, w_bar, model_error))
    return tube


def nominal_rollout(x0, actions, dt):
    """Deterministic (noise-free) rollout, i.e. the reachtube's center path."""
    states = [np.asarray(x0, dtype=float).copy()]
    for a in actions:
        states.append(states[-1] + np.asarray(a, dtype=float) * dt)
    return states
