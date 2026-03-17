# MAR - Autoregressive Image Generation without Vector Quantization

NeurIPS 2024. 用 Diffusion Loss 替代 VQ + cross-entropy，在连续值空间自回归建模。

## 项目结构

```
models/mar.py       # MAR 模型（encoder-decoder ViT + masked AR）
models/diffloss.py  # Diffusion Loss（DDPM DiffLoss + Flow Matching FlowDiffLoss）
models/vae.py       # KL-VAE tokenizer（stride=16, dim=16）
main_mar.py         # 训练入口（DDP）
engine_mar.py       # 训练/评估引擎
main_cache.py       # VAE latent 缓存
diffusion/          # 高斯扩散（cosine schedule, SpacedDiffusion）
util/               # 数据加载、分布式工具、学习率调度
tests/              # 单元测试 + E2E 测试
scripts/            # Smoke 测试脚本
```

## 模型架构

- **Encoder**：ViT + 64 buffer tokens，full attention
- **Decoder**：ViT + mask token → 条件向量 z → DiffLoss
- **Masked AR**：训练随机 mask（截断高斯, min=0.7），推理 cosine unmask
- **视频模式**：next-frame prediction，context 通过 full-sequence prefix injection
- **DiffLoss**：小型 MLP（默认 3 层 1024 宽），AdaLN 调制，支持 CFG。两种模式：
  - `--diffloss_type ddpm`（默认）：DDPM ε-prediction，cosine schedule，1000 步离散时间
  - `--diffloss_type flow`：JiT-style flow matching，x-prediction + v-loss，logit-normal 时间采样，Euler/Heun ODE solver

| 模型 | dim | layers | heads | 参数量 |
|------|-----|--------|-------|--------|
| MAR-B | 768 | 24 | 12 | 208M |
| MAR-L | 1024 | 32 | 16 | 479M |
| MAR-H | 1280 | 40 | 16 | 943M |

## 训练模式

### 默认（VAE）
标准流程：图像 → VAE encode → latent × 0.2325 → MAR → DiffLoss → VAE decode

### No-VAE 像素空间（`--no_vae`）
跳过 VAE，直接在 RGB patch 上训练。自动设 `vae_stride=1, vae_embed_dim=3`。
与 `--use_cached` 互斥。patchify/unpatchify 天然支持任意 (C,H,W)，模型零改动。

| 模式 | vae_stride | vae_embed_dim | patch_size | token_embed_dim |
|------|-----------|--------------|-----------|----------------|
| VAE 默认 | 16 | 16 | 1 | 16 |
| No-VAE 256×256 | 1 | 3 | 16 | 768 |
| No-VAE 64×64 | 1 | 3 | 4 | 48 |

## DiffLoss 类型

| 类型 | `--diffloss_type` | 预测目标 | Loss | 时间采样 | 采样器 | CFG 空间 |
|------|------------------|---------|------|---------|-------|---------|
| DDPM | `ddpm` | ε (noise) | MSE + VLB | 均匀离散 [0,999] | DDPM reverse | ε-space |
| Flow | `flow` | x (clean) | v-loss (加权 x-MSE) | logit-normal 连续 | Euler/Heun ODE | v-space |

Flow Matching 关键参数（`--flow_*`）：
- `P_mean=-0.8, P_std=0.8`：logit-normal 分布参数，控制训练时间步分布
- `noise_scale=1.0`：噪声缩放
- `t_eps=0.05`：(1-t) 下界 clamp，防止 t→1 时除零
- `sampling_method={euler,heun}`：ODE 求解器

## 关键技术细节

- VAE latent 缩放因子：`0.2325`
- Mask 采样：截断高斯（均值=seq_len, std=0.25×seq_len, min_ratio=0.7）
- 生成顺序：完全随机排列
- CFG：训练 10% label dropout，推理 ε = ε_uncond + ω(ε_cond - ε_uncond)

## 环境

- **miniconda** conda env `mar`
- 运行：`conda run -n mar python ...` / `conda run -n mar python -m pytest ...`
- Python 3.10, PyTorch 2.10.0, timm 0.9.12

## 测试

```bash
conda run -n mar python -m pytest tests/ -v
conda run -n mar python scripts/smoke_no_vae.py
conda run -n mar python scripts/smoke_flow_matching.py
```
