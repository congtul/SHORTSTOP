import torch

from shortstop.diffusion_policy import ConditionalUnet1D, DDPMSchedule


def _model():
    return ConditionalUnet1D(action_dim=2, horizon=8, cond_dim=11, base_channels=16)


def test_unet_forward_preserves_chunk_shape():
    model = _model()
    batch = 5
    chunk = torch.randn(batch, 8, 2)
    cond = torch.randn(batch, 11)
    t = torch.randint(0, 100, (batch,))

    out = model(chunk, cond, t)
    assert out.shape == chunk.shape


def test_unet_rejects_horizon_not_divisible_by_four():
    try:
        ConditionalUnet1D(horizon=6)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for horizon=6")


def test_training_loss_is_finite_and_backprop_reaches_every_parameter():
    model = _model()
    schedule = DDPMSchedule(num_diffusion_steps=100)
    x0 = torch.randn(4, 8, 2)
    cond = torch.randn(4, 11)

    loss = schedule.training_loss(model, x0, cond)
    assert torch.isfinite(loss)

    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient at {name}"


def test_ddim_sample_shape_and_reproducible_with_fixed_seed():
    model = _model()
    schedule = DDPMSchedule(num_diffusion_steps=100)
    cond = torch.randn(3, 11)

    torch.manual_seed(0)
    sample1 = schedule.ddim_sample(model, cond, chunk_shape=(8, 2), num_inference_steps=10)
    torch.manual_seed(0)
    sample2 = schedule.ddim_sample(model, cond, chunk_shape=(8, 2), num_inference_steps=10)

    assert sample1.shape == (3, 8, 2)
    assert torch.allclose(sample1, sample2)  # eta=0 DDIM: deterministic given the same seed


def test_model_is_small():
    model = _model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000
