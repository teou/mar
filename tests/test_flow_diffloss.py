import torch
from models.diffloss import FlowDiffLoss
from models.mar import MAR


def _make_tiny_flow_diffloss(**kwargs):
    defaults = dict(
        target_channels=16, z_channels=32, depth=1, width=64,
        num_sampling_steps=5, P_mean=-0.8, P_std=0.8,
        noise_scale=1.0, t_eps=0.05, sampling_method='euler',
    )
    defaults.update(kwargs)
    return FlowDiffLoss(**defaults)


def _make_tiny_mar_flow(**kwargs):
    defaults = dict(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
        diffloss_type='flow', num_sampling_steps='5',
    )
    defaults.update(kwargs)
    model = MAR(**defaults)
    # deterministic orders for CPU tests
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu
    return model


def test_flow_diffloss_forward_finite_loss():
    dl = _make_tiny_flow_diffloss()
    target = torch.randn(8, 16)
    z = torch.randn(8, 32)
    loss = dl(target, z)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_flow_diffloss_sample_shape():
    dl = _make_tiny_flow_diffloss()
    z = torch.randn(4, 32)
    dl.eval()
    with torch.no_grad():
        out = dl.sample(z, temperature=1.0, cfg=1.0)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all().item()


def test_flow_diffloss_sample_with_cfg():
    dl = _make_tiny_flow_diffloss()
    # CFG requires z of size 2*N (cond + uncond)
    z = torch.randn(8, 32)
    dl.eval()
    with torch.no_grad():
        out = dl.sample(z, temperature=1.0, cfg=2.0)
    assert out.shape == (8, 16)
    assert torch.isfinite(out).all().item()


def test_mar_with_flow_diffloss():
    model = _make_tiny_mar_flow()
    imgs = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 1000, (2,))
    loss = model(imgs, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_mar_flow_sample_tokens():
    model = _make_tiny_mar_flow()
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all().item()


def test_flow_diffloss_heun_sampling():
    dl = _make_tiny_flow_diffloss(sampling_method='heun')
    z = torch.randn(4, 32)
    dl.eval()
    with torch.no_grad():
        out = dl.sample(z, temperature=1.0, cfg=1.0)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all().item()


def test_flow_diffloss_no_vae_mode():
    """Full no-VAE + flow matching pipeline: forward + sample."""
    model = _make_tiny_mar_flow()
    # forward
    imgs = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 1000, (2,))
    loss = model(imgs, labels)
    assert torch.isfinite(loss).item()

    # sample
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=2.0, labels=torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all().item()
