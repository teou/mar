# PLAN: No-VAE 像素空间训练模式

## Context

论文 Appendix D.1 已验证 MAR 可直接在像素空间训练（ImageNet 64×64, FID=2.93）。当前代码强依赖 KL-VAE 做 encode/decode。本次改动增加 `--no_vae` 开关，跳过 VAE，让 MAR 直接在 RGB 像素 patch 上训练和生成，保持 seq_len=256。

**关键洞察**：`models/mar.py` 的 `patchify/unpatchify` 是纯 reshape 操作，天然支持任意 `(C, H, W)` 输入，无需添加 Conv。模型架构零改动。

## 参数对照

| 模式 | vae_stride | vae_embed_dim | patch_size | seq_len | token_embed_dim |
|------|-----------|--------------|-----------|---------|----------------|
| VAE (默认) | 16 | 16 | 1 | 256 | 16 |
| No-VAE 256×256 | 1 | 3 | 16 | 256 | 768 |
| No-VAE 64×64 | 1 | 3 | 4 | 256 | 48 |

---

## 改动清单

### 1. `main_mar.py` — 添加 `--no_vae` 开关

- 新增 `--no_vae` 参数（`action='store_true'`）
- 当 `--no_vae` 时自动覆盖：`vae_stride=1, vae_embed_dim=3`
- 校验 `--no_vae` 与 `--use_cached` 互斥
- 条件跳过 VAE 加载（`vae = None`）
- 打印 no-vae 模式信息（seq_len, token_embed_dim）

### 2. `engine_mar.py` — 训练和评估跳过 VAE

**`train_one_epoch()`（image_gen 分支，约 line 73-90）：**
```python
if getattr(args, 'no_vae', False):
    x = samples  # 图像已经是 [-1,1] 归一化，直接传入
else:
    # 原有 VAE encode 逻辑不变
```

**`train_one_epoch()`（video_next_frame 分支，约 line 54-72）：**
```python
if getattr(args, 'no_vae', False):
    # 跳过 vae.encode + 0.2325，直接 patchify 原始帧
    context_tokens = model.patchify(context_flat).view(bsz, t, -1, token_dim)
    target_tokens = model.patchify(target_frame)
else:
    # 原有逻辑不变
```

**`evaluate()`（约 line 180）：**
```python
sampled_tokens = model_without_ddp.sample_tokens(...)
if getattr(args, 'no_vae', False):
    sampled_images = sampled_tokens  # 已经是 [B,3,H,W]
else:
    sampled_images = vae.decode(sampled_tokens / 0.2325)
```

### 3. `models/mar.py` — 无改动

- `patchify()`：`[B,3,256,256]` → `[B,256,768]`（patch_size=16）— 已验证正确
- `unpatchify()`：反向操作 — 已验证正确
- `forward()`：接收 `[B,C,H,W]` 输入，内部调 patchify — 无需改动
- `sample_tokens()`：输出经 unpatchify 后为 `[B,3,256,256]` — 正确

### 4. `main_cache.py` — 无改动

No-VAE 模式不需要缓存，两者互斥。

---

## 测试计划

### UT: `tests/test_no_vae.py`

遵循 `tests/test_mar_video_forward.py` 的模式（小模型、CPU、mock sample_orders）。

**tiny 配置**：`img_size=32, vae_stride=1, patch_size=4, vae_embed_dim=3, encoder_embed_dim=64, depth=1, heads=4, diffloss_d=1, diffloss_w=64`
- seq_len = (32/1/4)² = 64
- token_embed_dim = 3×4² = 48

| 测试 | 验证点 |
|------|--------|
| `test_no_vae_model_init` | seq_len=64, token_embed_dim=48 |
| `test_patchify_unpatchify_roundtrip_rgb` | `[B,3,32,32]` → patchify → unpatchify → 与原图一致 |
| `test_no_vae_forward_finite_loss` | forward 返回有限标量 loss |
| `test_no_vae_sample_tokens_shape` | sample_tokens 输出 shape = `[2,3,32,32]` |

### Smoke: `scripts/smoke_no_vae.py`

独立脚本，无外部依赖（不需要 VAE checkpoint、不需要数据集）。

```
步骤1: 创建 tiny MAR 模型 (img_size=32, no-vae params)
步骤2: 随机图像 forward → 检查 loss 有限
步骤3: sample_tokens (num_iter=2) → 检查输出 shape 和 finite
步骤4: 打印 "Smoke test PASSED"
```

运行方式：`python scripts/smoke_no_vae.py`

### E2E: `tests/test_no_vae_e2e.py`

| 测试 | 验证点 |
|------|--------|
| `test_overfit_loss_decreases` | 固定 batch 训练 20 步，loss[-1] < loss[0] |
| `test_checkpoint_roundtrip` | save/load state_dict 后，同 seed 推理输出一致 |
| `test_generated_pixel_range` | 训练几步后生成的图像值有界（|x| < 10） |

E2E 测试直接用 model + AdamW 手动循环，不经过 `engine_mar.py`（避免 DDP/CUDA 依赖）。

---

## 验证步骤

实现完成后，按以下顺序执行验证：

```bash
# 1. 单元测试
pytest tests/test_no_vae.py -v

# 2. Smoke 测试
python scripts/smoke_no_vae.py

# 3. E2E 测试
pytest tests/test_no_vae_e2e.py -v

# 4. 回归：确保原有测试不被破坏
pytest tests/ -v

# 5. （可选）实际训练 dry-run
# 256×256, patch_size=16, seq_len=256
python main_mar.py --no_vae --img_size 256 --patch_size 16 --epochs 1 --batch_size 4
# 64×64, patch_size=4, seq_len=256
python main_mar.py --no_vae --img_size 64 --patch_size 4 --epochs 1 --batch_size 4
```

---

## 文件变更总结

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `main_mar.py` | 修改 | ~15 行（参数+条件加载） |
| `engine_mar.py` | 修改 | ~20 行（3 处 if/else 分支） |
| `models/mar.py` | 不改 | — |
| `tests/test_no_vae.py` | 新建 | ~80 行（4 个 UT） |
| `scripts/smoke_no_vae.py` | 新建 | ~40 行 |
| `tests/test_no_vae_e2e.py` | 新建 | ~90 行（3 个 E2E） |
