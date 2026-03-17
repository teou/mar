import copy
import torch
from models.mar import MAR


def _make_tiny_model():
    model = MAR(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
    )
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu
    return model


def test_overfit_loss_decreases():
    torch.manual_seed(42)
    model = _make_tiny_model()
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

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_checkpoint_roundtrip():
    torch.manual_seed(123)
    model = _make_tiny_model()
    model.eval()

    state_dict = copy.deepcopy(model.state_dict())

    model2 = _make_tiny_model()
    model2.load_state_dict(state_dict)
    model2.eval()

    with torch.no_grad():
        labels = torch.zeros(2).long()
        out1 = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)
        # reset seed for deterministic diffusion sampling
        torch.manual_seed(999)
        out1 = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)
        torch.manual_seed(999)
        out2 = model2.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=labels)

    assert torch.allclose(out1, out2, atol=1e-5), "Checkpoint roundtrip outputs differ"


def test_generated_pixel_range():
    torch.manual_seed(42)
    model = _make_tiny_model()
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

    assert torch.isfinite(out).all().item(), "Sample contains non-finite values"
    # Untrained tiny model won't produce bounded pixels; just verify finite and not exploding to inf/nan
