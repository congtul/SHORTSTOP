"""Small conditional diffusion policy for Stage 6a (Reach-Avoid-2D).

Denoiser: a 1D-conv U-Net over the chunk's time axis, FiLM-conditioned on
(state, obstacle_vec, diffusion timestep) -- matches the paper's own
description of pi_theta's architecture (Table VII: "1D-conv U-Net denoiser
... 100 training diffusion steps, 10 inference DDIM steps"), just sized for
this toy 2D setting (2D action, horizon 8, no image encoder needed).

`DDPMSchedule` carries the noise schedule plus the two things training and
inference actually need: `training_loss` (forward diffusion + denoising
MSE) and `ddim_sample` (fast few-step generation). Nothing in this file
runs a training loop -- see tests/test_diffusion_policy.py for structural
verification (shapes, gradient flow) before any of that is attempted.
"""
import math

import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the diffusion
    timestep (an integer in [0, num_diffusion_steps)), so the denoiser can
    tell how noisy its input currently is.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMConvBlock(nn.Module):
    """Conv1d -> GroupNorm -> FiLM (condition-predicted scale/shift) -> Mish."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.act = nn.Mish()

    def forward(self, x, cond):
        x = self.norm(self.conv(x))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return self.act(x)


class ConditionalUnet1D(nn.Module):
    """Denoiser eps_theta(chunk_noisy, t, state, obstacle_vec) -> predicted noise.

    Two downsample/upsample stages (horizon 8 -> 4 -> 2 -> 4 -> 8), skip
    connections between matching resolutions -- a small U-Net, not the full
    multi-stage one used for image-conditioned policies, since there is no
    image here at all.
    """

    def __init__(self, action_dim=2, horizon=8, cond_dim=11, time_dim=32, base_channels=32):
        super().__init__()
        if horizon % 4 != 0:
            raise ValueError("horizon must be divisible by 4 for two downsample stages")
        self.horizon = horizon
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.base_channels = base_channels

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )
        full_cond_dim = cond_dim + time_dim

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.down1 = FiLMConvBlock(action_dim, c1, full_cond_dim)
        self.down2 = FiLMConvBlock(c1, c2, full_cond_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid = FiLMConvBlock(c2, c3, full_cond_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up1 = FiLMConvBlock(c3 + c2, c2, full_cond_dim)
        self.up2 = FiLMConvBlock(c2 + c1, c1, full_cond_dim)
        self.out = nn.Conv1d(c1, action_dim, 1)

    def forward(self, chunk, cond, timestep):
        """chunk: (B, horizon, action_dim); cond: (B, cond_dim);
        timestep: (B,) int64 diffusion step in [0, num_diffusion_steps).
        Returns predicted noise, same shape as `chunk`.
        """
        x = chunk.transpose(1, 2)  # (B, action_dim, horizon)
        full_cond = torch.cat([cond, self.time_mlp(timestep)], dim=-1)

        h1 = self.down1(x, full_cond)
        h2 = self.down2(self.pool(h1), full_cond)
        m = self.mid(self.pool(h2), full_cond)

        u1 = self.up1(torch.cat([self.upsample(m), h2], dim=1), full_cond)
        u2 = self.up2(torch.cat([self.upsample(u1), h1], dim=1), full_cond)

        return self.out(u2).transpose(1, 2)  # (B, horizon, action_dim)


class DDPMSchedule:
    """Linear-beta DDPM noise schedule plus the training loss and DDIM
    sampler built on top of it (Table VII: 100 training steps / 10
    inference DDIM steps).
    """

    def __init__(self, num_diffusion_steps=100, beta_start=1e-4, beta_end=0.02):
        self.num_diffusion_steps = num_diffusion_steps
        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # (num_diffusion_steps,)
        self.betas = betas
        self.alphas = alphas

    def to(self, device):
        self.alpha_bars = self.alpha_bars.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self

    def q_sample(self, x0, t, noise):
        """Forward diffusion: x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise."""
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
        return alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

    def training_loss(self, model, x0, cond):
        """One DDPM training step: sample a random t and noise, predict
        the noise back out, MSE against the true noise.
        """
        batch = x0.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, (batch,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = model(x_t, cond, t)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, model, cond, chunk_shape, num_inference_steps=10, eta=0.0):
        """Deterministic-by-default DDIM sampler (eta=0), stepping through a
        subsequence of `num_inference_steps` timesteps instead of all
        `num_diffusion_steps` -- this is the "10 inference steps" side of
        the paper's 100-train/10-infer split.
        """
        device = cond.device
        batch = cond.shape[0]
        step_indices = torch.linspace(
            self.num_diffusion_steps - 1, 0, num_inference_steps, device=device
        ).long()

        x = torch.randn((batch, *chunk_shape), device=device)
        for i, t in enumerate(step_indices):
            t_batch = t.expand(batch)
            alpha_bar_t = self.alpha_bars[t]
            pred_noise = model(x, cond, t_batch)
            x0_pred = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()

            if i == len(step_indices) - 1:
                x = x0_pred
                continue
            t_next = step_indices[i + 1]
            alpha_bar_next = self.alpha_bars[t_next]
            dir_xt = (1 - alpha_bar_next).sqrt() * pred_noise
            x = alpha_bar_next.sqrt() * x0_pred + dir_xt
        return x


def save_checkpoint(path, model, schedule, cond_mean, cond_std, train_info):
    """Canonical on-disk format shared by scripts/train_diffusion_policy.py
    and whatever later loads this for inference (Stage 6a's Step 5: swap
    in for GaussianChunkPolicy). `cond_mean`/`cond_std` are the training-set
    standardization stats for `[state, obstacle_vec]` -- required at
    inference time too, so they travel with the weights.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "action_dim": model.action_dim,
                "horizon": model.horizon,
                "cond_dim": model.cond_dim,
                "base_channels": model.base_channels,
            },
            "schedule_config": {"num_diffusion_steps": schedule.num_diffusion_steps},
            "cond_mean": np.asarray(cond_mean),
            "cond_std": np.asarray(cond_std),
            "train_info": train_info,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    """Inverse of save_checkpoint: returns (model, schedule, cond_mean, cond_std, train_info)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = ConditionalUnet1D(**ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    schedule = DDPMSchedule(**ckpt["schedule_config"])
    return model, schedule, ckpt["cond_mean"], ckpt["cond_std"], ckpt["train_info"]
