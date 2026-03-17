#!/usr/bin/env python
"""Standalone smoke test for flow matching DiffLoss (no-VAE mode). No external dependencies needed."""
import sys
import torch
from models.mar import MAR


def main():
    torch.manual_seed(0)

    print("=== Flow Matching Smoke Test (no-VAE, Euler) ===")
    model = MAR(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
        diffloss_type='flow', num_sampling_steps='10',
    )

    # deterministic orders for CPU
    def _sample_orders_cpu(bsz):
        return torch.arange(model.seq_len).unsqueeze(0).repeat(bsz, 1).long()
    model.sample_orders = _sample_orders_cpu

    print(f"  seq_len={model.seq_len}, token_embed_dim={model.token_embed_dim}")
    print(f"  diffloss type: {type(model.diffloss).__name__}")

    # 1. Forward pass
    imgs = torch.randn(2, 3, 32, 32)
    labels = torch.randint(0, 1000, (2,))
    loss = model(imgs, labels)
    assert torch.isfinite(loss).item(), f"Loss is not finite: {loss.item()}"
    print(f"  Forward loss: {loss.item():.4f} (finite)")

    # 2. Backward pass
    loss.backward()
    grad_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
    assert len(grad_norms) > 0, "No gradients computed"
    assert all(v < 1e6 for v in grad_norms.values()), f"Gradient explosion detected"
    print(f"  Backward OK, {len(grad_norms)} params have gradients")

    # 3. Sample (no CFG)
    model.eval()
    with torch.no_grad():
        out = model.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert out.shape == (2, 3, 32, 32), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all().item(), "Sample contains non-finite values"
    print(f"  Sample (no CFG) shape: {out.shape} (all finite)")

    # 4. Sample (with CFG)
    with torch.no_grad():
        out_cfg = model.sample_tokens(bsz=2, num_iter=2, cfg=2.0, labels=torch.zeros(2).long())
    assert out_cfg.shape == (2, 3, 32, 32), f"Unexpected shape: {out_cfg.shape}"
    assert torch.isfinite(out_cfg).all().item(), "CFG sample contains non-finite values"
    print(f"  Sample (CFG=2.0) shape: {out_cfg.shape} (all finite)")

    # 5. Heun sampler
    print("\n=== Flow Matching Smoke Test (no-VAE, Heun) ===")
    model_heun = MAR(
        img_size=32, vae_stride=1, patch_size=4,
        encoder_embed_dim=64, encoder_depth=1, encoder_num_heads=4,
        decoder_embed_dim=64, decoder_depth=1, decoder_num_heads=4,
        vae_embed_dim=3, diffloss_d=1, diffloss_w=64,
        diffloss_type='flow', num_sampling_steps='10',
        flow_sampling_method='heun',
    )
    model_heun.sample_orders = _sample_orders_cpu
    model_heun.eval()
    with torch.no_grad():
        out_heun = model_heun.sample_tokens(bsz=2, num_iter=2, cfg=1.0, labels=torch.zeros(2).long())
    assert out_heun.shape == (2, 3, 32, 32)
    assert torch.isfinite(out_heun).all().item()
    print(f"  Sample (Heun, no CFG) shape: {out_heun.shape} (all finite)")

    print("\nSmoke test PASSED")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
