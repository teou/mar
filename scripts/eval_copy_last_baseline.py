import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms

from util.video_loader import VideoFrameDataset


def psnr(x, y, eps=1e-8):
    mse = torch.mean((x - y) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_data_path", type=str, required=True)
    parser.add_argument("--context_len", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=50)
    args = parser.parse_args()

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ds = VideoFrameDataset(args.video_data_path, context_len=args.context_len, transform=tfm)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    psnr_vals = []
    for i, (context, target, _meta) in enumerate(loader):
        pred = context[:, -1]
        pred_01 = ((pred + 1) / 2).clamp(0, 1)
        tgt_01 = ((target + 1) / 2).clamp(0, 1)
        psnr_vals.append(psnr(pred_01, tgt_01).item())
        if i + 1 >= args.max_batches:
            break

    avg_psnr = sum(psnr_vals) / max(1, len(psnr_vals))
    print(f"copy-last baseline avg PSNR over {len(psnr_vals)} batches: {avg_psnr:.4f}")


if __name__ == "__main__":
    main()
