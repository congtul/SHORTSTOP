import numpy as np


class Obstacle:
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def contains(self, point):
        return np.linalg.norm(np.asarray(point) - self.center) <= self.radius


class ReachAvoid2D:
    """Point-mass 2D reach-avoid environment (Sec. V, Reach-Avoid-2D prototype).

    Dynamics: x_{t+1} = x_t + a_t * dt + w_t, with ||w_t||_inf <= w_bar.
    """

    def __init__(
        self,
        start,
        goal,
        goal_radius=0.3,
        obstacles=None,
        bounds=((-5.0, -5.0), (5.0, 5.0)),
        dt=0.1,
        w_bar=0.02,
        max_action_norm=1.0,
        max_steps=200,
        rng=None,
    ):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.goal_radius = goal_radius
        self.obstacles = obstacles or []
        self.low = np.asarray(bounds[0], dtype=float)
        self.high = np.asarray(bounds[1], dtype=float)
        self.dt = dt
        self.w_bar = w_bar
        self.max_action_norm = max_action_norm
        self.max_steps = max_steps
        self.rng = rng if rng is not None else np.random.default_rng()
        self.state = None
        self.t = 0

    def reset(self):
        self.state = self.start.copy()
        self.t = 0
        return self.state.copy()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -self.max_action_norm, self.max_action_norm)
        noise = self.rng.uniform(-self.w_bar, self.w_bar, size=2)
        self.state = self.state + action * self.dt + noise
        self.state = np.clip(self.state, self.low, self.high)
        self.t += 1

        violated = any(o.contains(self.state) for o in self.obstacles)
        reached = np.linalg.norm(self.state - self.goal) <= self.goal_radius
        done = bool(violated or reached or self.t >= self.max_steps)
        info = {"violated": bool(violated), "reached": bool(reached)}
        return self.state.copy(), done, info
