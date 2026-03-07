# MAR 视频下一帧改造 - 真实数据集评估版报告

生成时间：2026-03-07

---

## 1. 评估目标

把此前的 MVP（视频下一帧预测）从 dummy 数据验证，升级到**真实视频帧数据**验证，输出可复现、可视化、可审计的结果。

---

## 2. 真实数据集构建方式

### 数据来源
- 原始视频：`data/real_videos_raw/bbb.mp4`
- 下载来源：公开样例视频（samplelib）

下载命令：
```bash
cd /Users/niubobo/.openclaw/workspace/mar
mkdir -p data/real_videos_raw
curl -L --fail -o data/real_videos_raw/bbb.mp4 https://samplelib.com/lib/preview/mp4/sample-5s.mp4
```

### 构建流程
- 从视频按时间间隔抽帧（每 3 帧取 1 帧）
- 重采样到 128x128
- 生成 3 个短视频序列目录：
  - `data/real_eval_frames/video_000`（10 帧）
  - `data/real_eval_frames/video_001`（10 帧）
  - `data/real_eval_frames/video_002`（10 帧）

构建命令：
```bash
cd /Users/niubobo/.openclaw/workspace/mar
PYTHONPATH=. python3 - <<'PY'
from pathlib import Path
import imageio.v2 as iio
from PIL import Image
import numpy as np

video_path = Path('data/real_videos_raw/bbb.mp4')
out_root = Path('data/real_eval_frames')
out_root.mkdir(parents=True, exist_ok=True)

for p in out_root.glob('video_*'):
    for f in p.glob('*.png'):
        f.unlink()
    p.rmdir()

reader = iio.get_reader(str(video_path))
frames = []
for idx, frame in enumerate(reader):
    if idx % 3 == 0:
        frames.append(frame)
reader.close()

starts = [0, 8, 16]
clip_len = 10
for vid, st in enumerate(starts):
    vdir = out_root / f'video_{vid:03d}'
    vdir.mkdir(parents=True, exist_ok=True)
    clip = frames[st:st+clip_len]
    for i, fr in enumerate(clip):
        img = Image.fromarray(np.asarray(fr).astype('uint8')).convert('RGB').resize((128, 128))
        img.save(vdir / f'{i:06d}.png')
PY
```

### 评估使用参数
- `context_len=4`
- 图像输入 `img_size=64`
- 设备：CPU

---

## 3. 实现说明（核心改造摘要）

为了让 E2E 在本机 CPU 环境可跑通，我做了三层改造：

1) **视频任务链路**
- `util/video_loader.py`：连续帧样本 `(context, target, meta)`
- `main_mar.py`：新增 `video_next_frame` 任务参数
- `engine_mar.py`：视频 batch 的 VAE 编码 + token 训练路径
- `models/mar.py`：`encode_context` / `forward_video` / `sample_next_frame`

2) **采样设备兼容性修复（关键）**
- `models/mar.py`
- `models/diffloss.py`
- `diffusion/gaussian_diffusion.py`
- 把残留 `.cuda()` 改成 device-aware（CPU/GPU 通用）

3) **验证脚本**
- `scripts/smoke_video_next_frame.py`
- `scripts/eval_copy_last_baseline.py`

---

## 4. 测试设计

### A) 单元测试（UT）
文件：
- `tests/test_video_loader.py`
- `tests/test_mar_video_forward.py`

覆盖点：
- 数据 shape 与时序正确性
- 帧不足异常处理
- `forward_video` 标量/有限值
- 条件输入变化时，context 编码变化

### B) 端到端测试（E2E）
1. Smoke：
- 验证“数据 -> VAE -> MAR forward”链路可运行

2. Baseline：
- 复制最后一帧（copy-last）PSNR 作为参考线

3. Tiny-overfit：
- 固定真实视频样本跑 30 步
- 检查 loss 是否下降
- 保存/加载 ckpt
- 推理 + 可视化（Context/GT/Pred 三联图）

---

## 5. 运行结果

## 5.1 UT 结果
命令：
`python3 -m pytest -q tests/test_video_loader.py tests/test_mar_video_forward.py`

结果：
- **4 passed, 1 warning**

原始日志：`unit_tests.txt`

## 5.2 Smoke 结果
命令：
```bash
cd /Users/niubobo/.openclaw/workspace/mar
PYTHONPATH=. python3 scripts/smoke_video_next_frame.py \
  --video_data_path ./data/real_eval_frames \
  --img_size 64 \
  --context_len 4 \
  --batch_size 1 \
  --device cpu \
  --vae_path pretrained_models/vae/kl16.ckpt
```

结果：
- `Smoke OK`
- `loss=1.001144`
- `baseline_psnr=31.9340`

原始日志：`smoke_test.txt`

## 5.3 Baseline 结果
命令：
```bash
cd /Users/niubobo/.openclaw/workspace/mar
PYTHONPATH=. python3 scripts/eval_copy_last_baseline.py \
  --video_data_path ./data/real_eval_frames \
  --context_len 4 \
  --img_size 64 \
  --batch_size 2 \
  --max_batches 8
```

结果：
- `copy-last baseline avg PSNR over 8 batches: 30.5839`

原始日志：`baseline_eval.txt`

## 5.4 Tiny-overfit 结果（真实样本）
命令：
```bash
cd /Users/niubobo/.openclaw/workspace/mar
PYTHONPATH=. python3 scripts/tiny_overfit_real_eval.py \
  --video_data_path ./data/real_eval_frames \
  --report_dir ./reports/e2e_report_real_2026-03-07 \
  --img_size 64 \
  --context_len 4 \
  --steps 30 \
  --lr 3e-5 \
  --device cpu \
  --vae_path pretrained_models/vae/kl16.ckpt
```

结果：
- `steps=30`
- `loss_start5_avg=1.0173`
- `loss_end5_avg=0.9207`
- `loss_improvement=+0.0966`（loss 下降，符合预期）
- `pred_latent_shape=[1,16,4,4]`

原始日志：`tiny_overfit_run.txt`
原始指标：`tiny_overfit_metrics.json`

---

## 6. 可视化输出

1. **loss 曲线图**
- `tiny_overfit_loss_curve.png`

2. **Context / Ground Truth / Predicted 三联图**
- `triptych_context_gt_pred.png`

---

## 7. 结论

在真实视频帧数据上，本次 MVP 具备：
- 单测通过
- E2E 链路可跑
- 基线可计算
- tiny-overfit 下 loss 有下降
- 可视化样例可导出

结论：**真实数据集评估版报告已完成，可用于当前阶段验收。**

---

## 附：报告目录

```
reports/e2e_report_real_2026-03-07/
  REPORT.md
  unit_tests.txt
  smoke_test.txt
  baseline_eval.txt
  tiny_overfit_metrics.json
  tiny_overfit_loss_curve.png
  triptych_context_gt_pred.png
```
