import numpy as np
import torch

from .env import encode_obstacles


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


class DiffusionChunkPolicy:
    """Drop-in replacement for GaussianChunkPolicy: samples K candidate
    action chunks from a trained diffusion model (shortstop/
    diffusion_policy.py, trained by scripts/train_diffusion_policy.py on
    the scripted-expert dataset), conditioned on (state, this episode's
    obstacles).

    Takes an already-loaded model/schedule/normalization stats rather than
    a checkpoint path, so it does no disk I/O -- cheap to construct fresh
    per episode the way shortstop/experiment.py:run_episode does for every
    policy. Load the checkpoint once (shortstop.diffusion_policy.
    load_checkpoint) and close over the pieces in a `policy_factory`
    (see scripts/run_ablation_diffusion.py).

    Only ever trained for the fixed goal (4.0, 0.0) that
    shortstop.experiment.make_scenario always uses -- goal isn't part of
    the model's input at all (Stage 6a's dataset never varied it), so it
    isn't accepted as a constructor argument here.
    """

    def __init__(self, model, schedule, cond_mean, cond_std, obstacles,
                 n_candidates=8, num_inference_steps=10, rng=None):
        self.model = model
        self.schedule = schedule
        self.cond_mean = cond_mean
        self.cond_std = cond_std
        self.obstacle_vec = encode_obstacles(obstacles)
        self.n_candidates = n_candidates
        self.num_inference_steps = num_inference_steps
        self.rng = rng if rng is not None else np.random.default_rng()

    def propose(self, state):
        cond_raw = np.concatenate([np.asarray(state, dtype=float), self.obstacle_vec])
        cond_norm = (cond_raw - self.cond_mean) / self.cond_std
        cond = torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0).repeat(self.n_candidates, 1)

        torch.manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        samples = self.schedule.ddim_sample(
            self.model, cond,
            chunk_shape=(self.model.horizon, self.model.action_dim),
            num_inference_steps=self.num_inference_steps,
        )
        return [samples[i].numpy() for i in range(self.n_candidates)]
