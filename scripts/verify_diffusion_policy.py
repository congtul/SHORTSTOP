"""Structural verification report for the Stage 6a diffusion policy model
(shortstop/diffusion_policy.py) -- no training, just confirms the network
and DDPM/DDIM plumbing are wired correctly against the real dataset shapes
built by scripts/build_dataset.py.

Usage:
    .venv/Scripts/python.exe scripts/verify_diffusion_policy.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from shortstop.diffusion_policy import ConditionalUnet1D, DDPMSchedule


def main(dataset_path="results/expert_dataset.npz"):
    data = np.load(dataset_path)
    states, obstacle_vecs, chunks = data["states"], data["obstacle_vecs"], data["action_chunks"]
    horizon = int(data["horizon"])
    action_dim = chunks.shape[-1]
    cond_dim = states.shape[-1] + obstacle_vecs.shape[-1]
    print(f"dataset: {dataset_path}  n_windows={len(states)}  horizon={horizon}  cond_dim={cond_dim}")

    model = ConditionalUnet1D(action_dim=action_dim, horizon=horizon, cond_dim=cond_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    schedule = DDPMSchedule(num_diffusion_steps=100)

    batch = 64
    idx = np.random.default_rng(0).choice(len(states), size=batch, replace=False)
    x0 = torch.tensor(chunks[idx], dtype=torch.float32)
    cond = torch.tensor(
        np.concatenate([states[idx], obstacle_vecs[idx]], axis=-1), dtype=torch.float32
    )

    t0 = time.perf_counter()
    loss = schedule.training_loss(model, x0, cond)
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(
        f"one training_loss + backward on a real batch of {batch}: "
        f"loss={loss.item():.4f}  all_grads_finite={grad_ok}  "
        f"({(time.perf_counter() - t0) * 1000:.1f} ms)"
    )

    t0 = time.perf_counter()
    n_candidates = 8
    sample_cond = cond[:1].repeat(n_candidates, 1)  # K candidates for 1 real (state, obstacle) input
    samples = schedule.ddim_sample(
        model, sample_cond, chunk_shape=(horizon, action_dim), num_inference_steps=10
    )
    print(
        f"ddim_sample: {n_candidates} candidate chunks for 1 (state, obstacle_vec) input, "
        f"shape={tuple(samples.shape)}  ({(time.perf_counter() - t0) * 1000:.1f} ms, untrained weights)"
    )


if __name__ == "__main__":
    main()
