import torch

from models.mar import MAR


def test_forward_video_returns_finite_scalar_on_cpu():
    model = MAR(
        img_size=32,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=64,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        vae_embed_dim=2,
        diffloss_d=1,
        diffloss_w=64,
        context_len=4,
    )

    # avoid CUDA-only path in sample_orders for CPU tests
    def _sample_orders_cpu(bsz):
        seq_len = model.seq_len
        return torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1).long()

    model.sample_orders = _sample_orders_cpu

    bsz, t = 2, 4
    seq_len, token_dim = model.seq_len, model.token_embed_dim
    context_tokens = torch.randn(bsz, t, seq_len, token_dim)
    target_tokens = torch.randn(bsz, seq_len, token_dim)

    loss = model.forward_video(context_tokens, target_tokens, labels=None)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_encode_context_changes_with_input():
    model = MAR(
        img_size=32,
        vae_stride=16,
        patch_size=1,
        encoder_embed_dim=64,
        encoder_depth=1,
        encoder_num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_num_heads=4,
        vae_embed_dim=2,
        diffloss_d=1,
        diffloss_w=64,
        context_len=4,
    )

    bsz, t = 1, 4
    seq_len, token_dim = model.seq_len, model.token_embed_dim
    c1 = torch.zeros(bsz, t, seq_len, token_dim)
    c2 = torch.ones(bsz, t, seq_len, token_dim)

    e1 = model.encode_context(c1)
    e2 = model.encode_context(c2)
    assert not torch.allclose(e1, e2)
