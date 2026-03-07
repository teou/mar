import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

from util.video_loader import VideoFrameDataset
from models.vae import AutoencoderKL
from models.mar import mar_base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_data_path', type=str, required=True)
    parser.add_argument('--report_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--context_len', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vae_path', type=str, default='pretrained_models/vae/kl16.ckpt')
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    ds = VideoFrameDataset(args.video_data_path, context_len=args.context_len, transform=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=True)
    context, target, _ = next(iter(loader))
    context = context.to(args.device)
    target = target.to(args.device)

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
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    for _ in range(args.steps):
        with torch.no_grad():
            b, t, c, h, w = context.shape
            ctx_lat = vae.encode(context.view(b * t, c, h, w)).sample().mul_(0.2325)
            ctx_tok = model.patchify(ctx_lat).view(b, t, -1, model.token_embed_dim)
            tgt_lat = vae.encode(target).sample().mul_(0.2325)
            tgt_tok = model.patchify(tgt_lat)

        loss = model.forward_video(ctx_tok, tgt_tok)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    ckpt_path = report_dir / 'tiny_overfit_ckpt.pth'
    torch.save({'model': model.state_dict(), 'steps': len(losses)}, ckpt_path)

    loaded = mar_base(
        img_size=args.img_size,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        diffloss_d=1,
        diffloss_w=256,
        context_len=args.context_len,
    ).to(args.device)
    loaded.load_state_dict(torch.load(ckpt_path, map_location=args.device)['model'])
    loaded.eval()

    with torch.no_grad():
        b, t, c, h, w = context.shape
        ctx_lat = vae.encode(context.view(b * t, c, h, w)).sample().mul_(0.2325)
        ctx_tok = loaded.patchify(ctx_lat).view(b, t, -1, loaded.token_embed_dim)
        pred_lat = loaded.sample_next_frame(ctx_tok, num_iter=1, cfg=1.0, temperature=1.0, progress=False)
        pred_img = vae.decode(pred_lat / 0.2325)

    def denorm(x):
        return ((x + 1) / 2).clamp(0, 1)

    ctx_p = to_pil_image(denorm(context[0, -1].cpu()))
    tgt_p = to_pil_image(denorm(target[0].cpu()))
    pred_p = to_pil_image(denorm(pred_img[0].cpu()))

    w, h = ctx_p.size
    trip = Image.new('RGB', (w * 3 + 40, h + 40), 'white')
    d = ImageDraw.Draw(trip)
    trip.paste(ctx_p, (10, 30))
    trip.paste(tgt_p, (20 + w, 30))
    trip.paste(pred_p, (30 + 2 * w, 30))
    d.text((10, 8), 'Context(last)', fill='black')
    d.text((20 + w, 8), 'Ground Truth', fill='black')
    d.text((30 + 2 * w, 8), 'Predicted', fill='black')
    trip_path = report_dir / 'triptych_context_gt_pred.png'
    trip.save(trip_path)

    pw, ph, m = 800, 320, 45
    plot = Image.new('RGB', (pw, ph), 'white')
    d = ImageDraw.Draw(plot)
    d.line((m, ph - m, pw - m, ph - m), fill='black', width=2)
    d.line((m, m, m, ph - m), fill='black', width=2)
    min_l, max_l = min(losses), max(losses)
    span = max(max_l - min_l, 1e-6)
    pts = []
    for i, l in enumerate(losses):
        x = m + (pw - 2 * m) * (i / max(1, len(losses) - 1))
        y = ph - m - (ph - 2 * m) * ((l - min_l) / span)
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill='blue', width=2)

    start5 = sum(losses[:5]) / 5
    end5 = sum(losses[-5:]) / 5
    imp = start5 - end5
    d.text((m, 10), f'Tiny-overfit loss curve (steps={len(losses)})', fill='black')
    d.text((pw - 260, 10), f'start5={start5:.4f}', fill='black')
    d.text((pw - 260, 28), f'end5={end5:.4f}', fill='black')
    d.text((pw - 260, 46), f'improvement={imp:.4f}', fill='black')
    loss_path = report_dir / 'tiny_overfit_loss_curve.png'
    plot.save(loss_path)

    metrics = {
        'dataset_path': args.video_data_path,
        'steps': len(losses),
        'loss_start5_avg': start5,
        'loss_end5_avg': end5,
        'loss_improvement': imp,
        'ckpt_path': str(ckpt_path),
        'triptych_image': str(trip_path),
        'loss_curve_image': str(loss_path),
        'pred_latent_shape': list(pred_lat.shape),
    }
    (report_dir / 'tiny_overfit_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
