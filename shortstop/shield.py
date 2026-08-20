import numpy as np

from .reach import nominal_rollout, propagate_tube


class ReachOnlyShield:
    """Phase 1 shield: Propose -> Reach -> reject on tube/obstacle intersection.

    No STL robustness-to-go (Phase 2), no counterexample search (Phase 3),
    no repair (Phase 4) yet.
    Selection among admissible chunks uses a progress-to-goal score as a
    stand-in for g(a) in Eq. (5); falls back to braking (zero action) if no
    candidate chunk is admissible.
    """

    def __init__(self, goal, obstacles, dt, w_bar, model_error=0.0):
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles
        self.dt = dt
        self.w_bar = w_bar
        self.model_error = model_error

    def _admissible(self, state, chunk):
        tube = propagate_tube(state, chunk, self.dt, self.w_bar, self.model_error)
        return not any(box.intersects_any(self.obstacles) for box in tube[1:])

    def _score(self, state, chunk):
        final = nominal_rollout(state, chunk, self.dt)[-1]
        return -np.linalg.norm(final - self.goal)

    def select(self, state, candidates):
        mask = [self._admissible(state, c) for c in candidates]
        admissible = [c for c, ok in zip(candidates, mask) if ok]
        if not admissible:
            fallback = np.zeros_like(candidates[0])
            return fallback, {"fallback": True, "n_admissible": 0, "admissible_mask": mask}
        best = max(admissible, key=lambda c: self._score(state, c))
        return best, {"fallback": False, "n_admissible": len(admissible), "admissible_mask": mask}
