"""Train the Stage 6a diffusion policy on the scripted-expert dataset.

Standardizes the conditioning input [state, obstacle_vec] using training-set
mean/std (action chunks are left as-is: they're already unit-speed vectors,
roughly in [-1, 1], a fine scale for DDPM as is). Holds out a validation
split purely to monitor for overfitting -- with a model this small
(~100K params) relative to the dataset (~37K windows) that's not expected,
but it's a one-line sanity check to have.

Usage:
    .venv/Scripts/python.exe scripts/train_diffusion_policy.py \
        [n_steps] [dataset_path] [checkpoint_path]
    (defaults: n_steps=3000, dataset_path=results/expert_dataset.npz,
    checkpoint_path=results/diffusion_policy.pt)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from shortstop.diffusion_policy import ConditionalUnet1D, DDPMSchedule, save_checkpoint


def load_dataset(dataset_path):
    data = np.load(dataset_path)
    states = data["states"].astype(np.float32)
    obstacle_vecs = data["obstacle_vecs"].astype(np.float32)
    chunks = data["action_chunks"].astype(np.float32)
    horizon = int(data["horizon"])
    cond = np.concatenate([states, obstacle_vecs], axis=-1)
    return cond, chunks, horizon


def main(
    n_steps=3000,
    dataset_path="results/expert_dataset.npz",
    checkpoint_path="results/diffusion_policy.pt",
    batch_size=128,
    lr=1e-3,
    val_fraction=0.1,
    log_every=200,
    seed=0,
):
    cond, chunks, horizon = load_dataset(dataset_path)
    action_dim = chunks.shape[-1]
    cond_dim = cond.shape[-1]

    rng = np.random.default_rng(seed)
    n = len(cond)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    cond_mean = cond[train_idx].mean(axis=0)
    cond_std = cond[train_idx].std(axis=0) + 1e-6
    cond_norm = (cond - cond_mean) / cond_std

    torch.manual_seed(seed)
    cond_t = torch.tensor(cond_norm, dtype=torch.float32)
    chunks_t = torch.tensor(chunks, dtype=torch.float32)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)

    model = ConditionalUnet1D(action_dim=action_dim, horizon=horizon, cond_dim=cond_dim)
    schedule = DDPMSchedule(num_diffusion_steps=100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(
        f"train windows: {len(train_idx)}  val windows: {len(val_idx)}  "
        f"model params: {sum(p.numel() for p in model.parameters()):,}"
    )

    t0 = time.perf_counter()
    running_train_loss = []
    last_val_loss = None
    for step in range(1, n_steps + 1):
        batch_idx = train_idx_t[torch.randint(0, len(train_idx_t), (batch_size,))]
        loss = schedule.training_loss(model, chunks_t[batch_idx], cond_t[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss.append(loss.item())

        if step % log_every == 0 or step == n_steps:
            model.eval()
            with torch.no_grad():
                val_batch = val_idx_t[torch.randint(0, len(val_idx_t), (min(batch_size, len(val_idx_t)),))]
                last_val_loss = schedule.training_loss(model, chunks_t[val_batch], cond_t[val_batch]).item()
            model.train()
            elapsed = time.perf_counter() - t0
            print(
                f"step {step:5d}/{n_steps}  train_loss={np.mean(running_train_loss[-log_every:]):.4f}  "
                f"val_loss={last_val_loss:.4f}  ({elapsed:.1f}s elapsed)"
            )

    train_info = {
        "dataset_path": str(dataset_path),
        "n_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "n_train_windows": len(train_idx),
        "n_val_windows": len(val_idx),
        "final_train_loss": float(np.mean(running_train_loss[-log_every:])),
        "final_val_loss": last_val_loss,
    }
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(checkpoint_path, model, schedule, cond_mean, cond_std, train_info)
    print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    dataset_path = sys.argv[2] if len(sys.argv) > 2 else "results/expert_dataset.npz"
    checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else "results/diffusion_policy.pt"
    main(n_steps=n_steps, dataset_path=dataset_path, checkpoint_path=checkpoint_path)
