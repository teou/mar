import torch
from models.mar import MAR


def _make_tiny_model():
    """Tiny no-VAE model: img_size=32, patch_size=4, vae_stride=1, vae_embed_dim=3."""
    model = MAR(
        img_size=32,
        vae_stride=1,
        patch_size=4,
        encoder_embed_dim=64,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        vae_embed_dim=3,
        diffloss_d=1,
        diffloss_w=64,
    )
    # deterministic orders for CPU tests
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu
    return model


def test_no_vae_model_init():
    model = _make_tiny_model()
    assert model.seq_len == 64  # (32/1/4)^2
    assert model.token_embed_dim == 48  # 3 * 4^2


def test_patchify_unpatchify_roundtrip_rgb():
    model = _make_tiny_model()
    imgs = torch.randn(2, 3, 32, 32)
    tokens = model.patchify(imgs)
    assert tokens.shape == (2, 64, 48)
    reconstructed = model.unpatchify(tokens)
    assert reconstructed.shape == (2, 3, 32, 32)
    assert torch.allclose(imgs, reconstructed)


def test_no_vae_forward_finite_loss():
    model = _make_tiny_model()
    imgs = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 1000, (2,))
    loss = model(imgs, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_no_vae_sample_tokens_shape():
    model = _make_tiny_model()
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all().item()
