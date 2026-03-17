"""E2E tests for flow matching DiffLoss: overfit, checkpoint roundtrip, pixel range, gradient health."""
import copy
import torch
from models.mar import MAR


def _make_tiny_flow_model(**kwargs):
    defaults = dict(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
        diffloss_type='flow', num_sampling_steps='10',
    )
    defaults.update(kwargs)
    model = MAR(**defaults)
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu
    return model


def test_flow_overfit_loss_decreases():
    """Train 30 steps on a fixed batch; loss must decrease."""
    torch.manual_seed(42)
    model = _make_tiny_flow_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 1000, (4,))

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    assert last5 < first5, (
        f"Loss did not decrease: first5_avg={first5:.4f} -> last5_avg={last5:.4f}"
    )


def test_flow_checkpoint_roundtrip():
    """Save and reload state_dict; sampling output must match exactly."""
    torch.manual_seed(123)
    model = _make_tiny_flow_model()
    model.eval()

    state_dict = copy.deepcopy(model.state_dict())

    model2 = _make_tiny_flow_model()
    model2.load_state_dict(state_dict)
    model2.eval()

    labels = torch.zeros(2).long()
    with torch.no_grad():
        torch.manual_seed(999)
        out1 = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)
        torch.manual_seed(999)
        out2 = model2.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)

    assert torch.allclose(out1, out2, atol=1e-5), "Checkpoint roundtrip outputs differ"


def test_flow_generated_pixel_range():
    """After a few training steps, sampled output must be finite (not NaN/Inf)."""
    torch.manual_seed(42)
    model = _make_tiny_flow_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    labels = torch.randint(0, 1000, (4,))

    for _ in range(5):
        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert torch.isfinite(out).all().item(), "Sample contains non-finite values after training"


def test_flow_gradient_health():
    """All trainable params in FlowDiffLoss net should receive gradients after a few steps.

    At init, adaLN modulation layers are zero-initialized, so time_embed and cond_embed
    have zero gradients (their output y gets multiplied by zero in all gates). After a few
    training steps the gates open and gradients flow through.
    """
    torch.manual_seed(0)
    model = _make_tiny_flow_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 1000, (4,))

    # Train a few steps so adaLN gates open up
    for _ in range(5):
        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()
        optimizer.step()

    # After training, all params should have non-zero finite gradients
    diffloss_params = {name: p for name, p in model.diffloss.named_parameters() if p.requires_grad}
    no_grad = [name for name, p in diffloss_params.items() if p.grad is None]
    assert len(no_grad) == 0, f"FlowDiffLoss params with no gradient: {no_grad}"

    for name, p in diffloss_params.items():
        assert torch.isfinite(p.grad).all().item(), f"Non-finite gradient in {name}"
        assert p.grad.abs().sum().item() > 0, f"Zero gradient in {name}"


def test_flow_cfg_sampling_differs_from_uncond():
    """After training, CFG sampling should produce different outputs than unconditional sampling.

    At init, adaLN gates are zero so conditioning has no effect and CFG changes nothing.
    After a few training steps the model becomes condition-aware and CFG should matter.
    """
    torch.manual_seed(77)
    model = _make_tiny_flow_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 1000, (4,))
    for _ in range(10):
        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    labels = torch.zeros(2).long()

    with torch.no_grad():
        torch.manual_seed(42)
        out_nocfg = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)
        torch.manual_seed(42)
        out_cfg = model.sample_tokens(bsz=2, num_iter=2, cfg=3.0, labels=labels)

    assert not torch.allclose(out_nocfg, out_cfg, atol=1e-4), "CFG and non-CFG outputs are identical after training"


def test_flow_heun_overfit_loss_decreases():
    """Same overfit test but with Heun solver, verifying it doesn't break training/sampling."""
    torch.manual_seed(42)
    model = _make_tiny_flow_model(flow_sampling_method='heun')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 1000, (4,))

    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        loss = model(imgs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Heun: loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Also verify sampling works with Heun after training
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert torch.isfinite(out).all().item()
