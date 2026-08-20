import numpy as np


class GaussianChunkPolicy:
    """Stand-in for a trained generative policy (Sec. III.C).

    Samples K candidate action chunks of length H as Gaussian perturbations
    around a reference "go straight to the goal" velocity command. This is
    enough to exercise the shield pipeline end-to-end before a real
    diffusion/flow policy is trained (Phase 6).
    """

    def __init__(self, goal, horizon=8, n_candidates=8, noise_std=0.3, max_speed=1.0, rng=None):
        self.goal = np.asarray(goal, dtype=float)
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.noise_std = noise_std
        self.max_speed = max_speed
        self.rng = rng if rng is not None else np.random.default_rng()

    def _reference_action(self, state):
        direction = self.goal - np.asarray(state, dtype=float)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2)
        return direction / norm * self.max_speed

    def propose(self, state):
        ref = self._reference_action(state)
        chunks = []
        for _ in range(self.n_candidates):
            noise = self.rng.normal(0.0, self.noise_std, size=(self.horizon, 2))
            chunk = np.clip(ref + noise, -self.max_speed, self.max_speed)
            chunks.append(chunk)
        return chunks
