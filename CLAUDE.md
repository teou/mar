# MAR - Autoregressive Image Generation without Vector Quantization

## 论文

NeurIPS 2024 论文实现。作者：Tianhong Li (MIT), Yonglong Tian (Google DeepMind), He Li (清华), Mingyang Deng (MIT), Kaiming He (MIT)。

核心思想：用 **Diffusion Loss** 替代离散 VQ tokenizer + cross-entropy loss，使自回归模型能在连续值空间上直接建模 per-token 概率分布。

代码仓库：https://github.com/LTH14/mar

## 项目结构

```
models/
  mar.py          # MAR 模型主体（encoder-decoder Transformer + masked AR）
  diffloss.py     # Diffusion Loss 模块（SimpleMLPAdaLN 去噪网络）
  vae.py          # KL-VAE tokenizer（预训练，stride=16，embed_dim=16）
diffusion/
  gaussian_diffusion.py  # 高斯扩散过程（1000步 cosine schedule）
  respace.py             # 推理加速（SpacedDiffusion，默认100步）
  diffusion_utils.py     # 工具函数
main_mar.py       # 训练入口（DDP 分布式训练）
engine_mar.py     # 训练/评估引擎（train_one_epoch, evaluate, cache_latents）
main_cache.py     # VAE latent 预计算缓存
util/
  misc.py         # 分布式训练工具
  loader.py       # 图像数据加载（ImageFolder + CachedFolder）
  video_loader.py # 视频帧数据加载
  lr_sched.py     # 学习率调度
demo/
  gradio_app.py   # Gradio 交互式 demo
tests/            # 单元测试
pretrained_models/
  vae/kl16.ckpt   # 预训练 KL-16 VAE
reports/          # 实验报告
```

## 模型架构

### MAR 模型 (`models/mar.py`)
- **Encoder**：ViT encoder + 可学习位置编码，前置 64 个 buffer `[cls]` token
- **Decoder**：ViT decoder + mask token，输出条件向量 z 给 DiffLoss
- **Masked AR**：训练时随机 mask（ratio 从截断高斯采样，min=0.7），推理时逐步 unmask（cosine schedule，默认 64 步）
- **双向注意力**：encoder 和 decoder 均使用 full attention（非 causal）
- **视频模式**：支持 next-frame prediction，context frames 通过 full-sequence prefix injection 注入（非 mean-pooling）

### 三个模型规模

| 模型 | dim | layers | heads | 参数量 | FID (w/ CFG) |
|------|-----|--------|-------|--------|-------------|
| MAR-B | 768 | 24 | 12 | 208M | 2.31 |
| MAR-L | 1024 | 32 | 16 | 479M | 1.78 |
| MAR-H | 1280 | 40 | 16 | 943M | 1.55 |

### Diffusion Loss (`models/diffloss.py`)
- 小型 MLP（默认 3 层，1024 宽度，~21M 参数）
- AdaLN 调制（timestep embedding + condition z）
- 训练：去噪损失 + 可选 VLB
- 推理：反向扩散采样，支持 CFG 和温度控制

## 训练配置

- **优化器**：AdamW (lr=8e-4, weight_decay=0.02, betas=(0.9, 0.95))
- **Batch size**：2048（全局）
- **Epochs**：400（默认），800（Table 4 最佳结果）
- **Warmup**：100 epochs linear warmup → constant lr
- **Diffusion batch multiplier**：4（每个 z 采样 4 次 timestep t）
- **EMA**：momentum=0.9999
- **数据集**：ImageNet 256×256
- **支持 VAE latent 缓存**：2x 训练加速
- **Gradient checkpointing**：支持

## 推理配置

- **MAR 步数**：64（默认消融）/ 256（最佳结果）
- **扩散采样步数**：100
- **CFG scale**：3.0（默认）
- **温度**：1.0（默认）

## 近期开发重点

### 最新改动 (2026-03)
- **context prefix injection**（commit 2405c12）：视频 next-frame prediction 的核心改进。将 context frames 从 mean-pooling 压缩改为 full-sequence prefix 注入，保留完整时空信息。新增 factorized position embedding（temporal + spatial）。

### 实验报告 (2026-03)
- OpenClaw + Codex 实验（多个 commit）：围绕视频预测功能进行的 E2E 实验和验证报告
- NV Cosmos tokenizer 实验（commit c63380b）：尝试用 NVIDIA Cosmos tokenizer 替代 KL-VAE

## 关键技术细节

- **VAE latent 归一化**：乘以 `0.2325` 缩放因子
- **Mask 采样**：截断高斯（均值=序列长度，std=0.25*序列长度，min_ratio=0.7）
- **生成顺序**：完全随机排列（区别于 MAGE 的 confidence-based 排序）
- **温度采样**：扩散采样时缩放 noise σ_t * δ by τ
- **CFG 实现**：训练时 10% label dropout，推理时 ε = ε_uncond + ω(ε_cond - ε_uncond)

## 环境

- **运行环境**：miniconda，conda env 名称 `mar`（路径 `/opt/homebrew/Caskroom/miniconda/base/envs/mar`）
- 运行命令统一用 `conda run -n mar python ...` 或 `conda run -n mar python -m pytest ...`
- Python 3.10, PyTorch 2.10.0
- timm 0.9.12（ViT blocks）
- 训练硬件：8×V100/H100 GPU × 多节点
