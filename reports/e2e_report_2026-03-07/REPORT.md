# MAR 视频下一帧改造 - E2E 可视化报告

生成时间：2026-03-07

---

## 1) 我是怎么实现的（实现说明）

### 目标
把原始 MAR（类条件图像生成）改造成 MVP 版视频下一帧预测：
- 输入：前 `n` 帧（默认 `n=4`）
- 输出：第 `n+1` 帧（当前以 latent 空间预测为主）

### 主要改造点

1. **数据层**
- 新增 `util/video_loader.py`
- 支持按视频文件夹读取连续帧，输出 `(context_frames, target_frame, meta)`

2. **训练入口与参数**
- 修改 `main_mar.py`
- 新增参数：
  - `--task {image_gen,video_next_frame}`
  - `--video_data_path`
  - `--context_len`

3. **训练循环**
- 修改 `engine_mar.py`
- 在 `video_next_frame` 任务中：
  - 对 context/target 逐帧 VAE 编码
  - context tokens 作为条件，target tokens 作为监督

4. **模型结构（MAR）**
- 修改 `models/mar.py`
- 新增：
  - `encode_context(...)`
  - `forward_video(...)`
  - `sample_next_frame(...)`
- `forward(...)` 支持 tuple 视频输入

5. **设备兼容修复（关键）**
- 修改 `models/mar.py`、`models/diffloss.py`、`diffusion/gaussian_diffusion.py`
- 将残留 `.cuda()` 改为 device-aware，支持 CPU 跑通 E2E

---

## 2) 测试是怎么设计的

### A. 单元测试（UT）

文件：
- `tests/test_video_loader.py`
- `tests/test_mar_video_forward.py`

覆盖点：
1. **数据集 shape/时序正确性**
   - context 形状是否为 `[T, C, H, W]`
   - target 形状是否为 `[C, H, W]`
   - 索引是否严格满足 `context < target`
2. **异常场景**
   - 帧数不足时是否正确抛错
3. **模型前向稳定性**
   - `forward_video` 返回标量且 finite
4. **条件有效性**
   - 不同 context 输入，`encode_context` 输出应不同

### B. E2E 验证（脚本）

1. `scripts/smoke_video_next_frame.py`
- 验证链路：数据 -> VAE 编码 -> MAR 视频前向
- 输出：`loss` 与基线 `baseline_psnr`

2. `scripts/eval_copy_last_baseline.py`
- 计算 “复制最后一帧” baseline 的平均 PSNR

3. Tiny-overfit（报告内复现实验）
- 固定单样本做 30 步训练，观察 loss 下降
- 保存/加载 ckpt
- 跑一次推理并生成可视化结果

---

## 3) 测试运行结果

### UT 结果
- 命令：
  - `python3 -m pytest -q tests/test_video_loader.py tests/test_mar_video_forward.py`
- 结果：
  - **4 passed, 1 warning**
- 原始日志：`unit_tests.txt`

### Smoke 结果
- 结果：
  - `Smoke OK`
  - `loss=0.980068`
  - `baseline_psnr=26.6181`
- 原始日志：`smoke_test.txt`

### Baseline 结果
- 结果：
  - `copy-last baseline avg PSNR over 4 batches: 26.6181`
- 原始日志：`baseline_eval.txt`

### Tiny-overfit 结果
- `steps=30`
- `loss_start5_avg=1.0189`
- `loss_end5_avg=0.9718`
- `loss_improvement=+0.0471`（下降，符合“可学习”预期）
- `pred_latent_shape=[1,16,4,4]`
- 原始指标：`tiny_overfit_metrics.json`

---

## 4) 可视化材料

1. **Loss 曲线**
- 文件：`tiny_overfit_loss_curve.png`

2. **三联图（Context / GT / Pred）**
- 文件：`triptych_context_gt_pred.png`

> 注：当前 Pred 是通过 `sample_next_frame -> latent` 再经 VAE decode 得到；这是 MVP 阶段验证链路可运行的可视化，不代表最终效果上限。

---

## 5) 结论与下一步

### 当前结论
- 功能改造已形成闭环：**UT通过 + Smoke通过 + Baseline有数 + Tiny-overfit下降 + 推理可出图**。
- 当前阶段可以判定为：**MVP 验证通过**。

### 建议下一步
1. 把评估从 dummy 数据切到真实视频数据集
2. 增加正式指标（PSNR/SSIM/LPIPS）并和 baseline 同一协议对齐
3. 把预测图与 GT 做批量导出，形成更完整实验报告

---

## 附：报告目录

```
reports/e2e_report_2026-03-07/
  REPORT.md
  unit_tests.txt
  smoke_test.txt
  baseline_eval.txt
  tiny_overfit_metrics.json
  tiny_overfit_loss_curve.png
  triptych_context_gt_pred.png
```
