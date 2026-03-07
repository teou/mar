import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from util.video_loader import VideoFrameDataset
from models.vae import AutoencoderKL
from models.mar import mar_base


def make_tiny_dummy_videos(root: Path, num_videos=2, num_frames=8, size=64):
    root.mkdir(parents=True, exist_ok=True)
    for vid in range(num_videos):
        vdir = root / f"video_{vid:03d}"
        vdir.mkdir(parents=True, exist_ok=True)
        for t in range(num_frames):
            arr = np.zeros((size, size, 3), dtype=np.uint8)
            arr[..., 0] = (t * 20 + vid * 30) % 255
            arr[..., 1] = (np.arange(size)[None, :] + t * 5) % 255
            arr[..., 2] = (np.arange(size)[:, None] + vid * 10) % 255
            Image.fromarray(arr).save(vdir / f"{t:06d}.png")


def psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_data_path", type=str, default="./tmp_videos")
    parser.add_argument("--context_len", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--vae_path", type=str, default="pretrained_models/vae/kl16.ckpt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--create_dummy", action="store_true")
    args = parser.parse_args()

    if args.create_dummy:
        make_tiny_dummy_videos(Path(args.video_data_path), size=args.img_size)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ds = VideoFrameDataset(args.video_data_path, context_len=args.context_len, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path=args.vae_path).to(args.device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    model = mar_base(
        img_size=args.img_size,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        diffloss_d=1,
        diffloss_w=256,
        context_len=args.context_len,
    ).to(args.device)

    context, target, _ = next(iter(loader))
    context = context.to(args.device)
    target = target.to(args.device)

    with torch.no_grad():
        b, t, c, h, w = context.shape
        ctx_flat = context.view(b * t, c, h, w)
        ctx_lat = vae.encode(ctx_flat).sample().mul_(0.2325)
        ctx_tok = model.patchify(ctx_lat).view(b, t, -1, model.token_embed_dim)

        tgt_lat = vae.encode(target).sample().mul_(0.2325)
        tgt_tok = model.patchify(tgt_lat)

        loss = model.forward_video(ctx_tok, tgt_tok)

        # baseline: copy last frame
        pred_baseline = context[:, -1]
        target_01 = (target + 1) / 2
        pred_01 = (pred_baseline + 1) / 2
        metric_psnr = psnr(pred_01.clamp(0, 1), target_01.clamp(0, 1))

    print("Smoke OK")
    print(f"loss={loss.item():.6f}")
    print(f"baseline_psnr={metric_psnr.item():.4f}")


if __name__ == "__main__":
    main()
