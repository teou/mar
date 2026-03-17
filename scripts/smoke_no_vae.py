#!/usr/bin/env python
"""Standalone smoke test for no-VAE pixel-space MAR. No external dependencies needed."""
import sys
import torch
from models.mar import MAR


def main():
    print("Creating tiny no-VAE MAR model (img_size=32, patch_size=4) ...")
    model = MAR(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
    )

    # deterministic orders for CPU
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu

    print(f"  seq_len={model.seq_len}, token_embed_dim={model.token_embed_dim}")

    # forward pass
    imgs = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 1000, (2,))
    loss = model(imgs, labels)
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    print(f"  Forward loss: {loss.item():.4f} (finite)")

    # sample
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all().item(), "Sample contains non-finite values"
    print(f"  Sample shape: {out.shape} (all finite)")

    print("Smoke test PASSED")


if __name__ == "__main__":
    main()
