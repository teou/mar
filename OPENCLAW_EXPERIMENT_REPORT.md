# OpenClaw 完成 MAR 改造任务实验报告

> 基于 `chat_history.md` 对话记录整理
> 实验主题：通过 OpenClaw 从需求提出到代码改造、测试验证、报告交付的完整闭环

## 0. 实验环境简述

- **Agent框架**：OpenClaw（最新版）
- **模型**：OpenAI Codex 5.3（`openai-codex/gpt-5.3-codex`）
- **执行形态**：OpenClaw + Codex 协同完成需求分解、代码改造、测试验证与报告产出
- **工作目录**：`~/.openclaw/workspace/mar`

---

## 1. 实验目标

验证在真实协作场景中，OpenClaw 是否能完成以下闭环：

1. 获取并理解外部项目（`LTH14/mar`）
2. 将高层需求转为可执行改造计划
3. 进行代码实现并持续推进
4. 通过 UT + E2E 方式验证改动有效性
5. 产出可视化报告材料

核心需求是：

- 将原本图像生成导向的 MAR 改造为「输入前 n 帧视频，自回归预测下一帧」的 MVP 路线。

### 1.1 原始用户需求引用（摘自 chat_history）

以下为本实验中最关键的原始需求片段：

1. 「**ok 你研究一下这个repo，学会了以后，研究一下这个repo如何把输入改成视频， 输入前n帧视频自回归地预测下一帧视频**」
2. 「**你先做个计划，准备怎么改，改完了以后通过什么样的ut或端到端测试验证你的改动，你对代码的改动要能验证**」
3. 「**这个计划先记下来，然后按这个计划开始改**」
4. 「**继续补完、你的测试要跑起来 看一下测试的结果 是否符合预期**」
5. 「**整理下各种材料，介绍下你怎么实现的，你的测试怎么设计的，测试的运行结果是什么，形成一个可视化的报告发给我**」

这些原始指令共同定义了目标：**不仅要实现功能，还要可验证、可复盘、可交付**。

### 1.2 目标难度解析

从工程角度，这不是"单点改代码"任务，而是一个中高复杂度的系统改造任务，难度主要来自四层：

1. **技术难度（中高）**
   - 原 MAR 偏图像 token 生成，目标却是视频时序预测；
   - 需要把"随机 mask 恢复"思路迁移到"前 n 帧条件 + 下一帧生成"；
   - 涉及数据、模型、训练接口联动改造，不是单文件修补。

2. **验证难度（高）**
   - 仅"能跑通"不等于"改对了"；
   - 必须设计 UT（shape/时序/依赖性）+ E2E（smoke/baseline/overfit/ckpt）形成证据链；
   - 需要定义基线（copy-last-frame）避免主观判断。

3. **工程环境难度（中）**
   - 过程中出现设备兼容性问题（`.cuda()` 硬编码导致 CPU 失败）；
   - 需要把关键路径改为 device-aware，才能确保在当前机器可稳定复现。

4. **协作与过程控制难度（中）**
   - 目标不是一次性交付，而是"持续推进 + 节点汇报 + 最终可视化报告"；
   - 中途还出现过上下文断裂插曲，增加了过程管理复杂度。

综合评估：**本目标属于"中高难度的工程闭环任务"，其难点在验证与持续可追踪性，而不只是功能编码本身。**

### 1.3 人类 P6 vs OpenClaw+Codex 工期对比（100%投入估算）

> 说明：本节为工程估算口径，基于本实验聊天时间线与已交付产物（代码改造 + UT/E2E + 报告）。

#### 结论

- **人类 P6 研发工程师（单人、100%投入）**：约 **1.5~3 个工作日（12~24 小时）**
- **OpenClaw + Codex（本次实际）**：约 **6.5~7.5 小时**（按聊天时间线折算）

#### OpenClaw+Codex 本次用时依据（聊天时间线）

- 03:01：明确改造目标（前 n 帧预测下一帧）
- 03:05~03:25：完成计划与第一批实现推进
- 08:43 后：恢复上下文并继续测试/验收闭环
- 09:57：交付可视化报告

据此，从"目标明确"到"可验证交付（测试 + 报告）"为约 **7 小时量级**（含中断插曲与沟通开销）。

#### 人类 P6 典型工时拆分（同口径估算）

1. 代码阅读与方案设计：**1~3h**
2. 数据/模型/训练链路改造：**4~8h**
3. UT + E2E + 兼容性修复：**4~8h**
4. 可视化整理与技术报告：**2~4h**

合计常见落在 **12~24h**。

#### 对比结论（速度）

在本实验范围内，OpenClaw+Codex 从需求到"可验证交付"约 7 小时；按同口径估算，单人 P6 全投入通常需要 12~24 小时。前者在速度上约快 **1.7x~3.4x**，但稳定性更依赖上下文连续性与过程治理（如状态持久化、里程碑记录）。

---

## 2. 对话与执行过程总结（时间线）

### 阶段 A：需求澄清与环境打通

- 用户提出：研究 `litianhong/mar`（后提供仓库链接）。
- 初期出现能力边界（无法直接访问外网仓库）→ 通过本地 `git clone` 方式打通。
- 形成初步技术判断：
  - 原项目是图像 token 生成，不是视频时间因果预测。
  - 需要从 2D 图像补全逻辑改为时序条件生成逻辑。

### 阶段 B：方案设计与验收标准定义

- 给出路线 A（MVP）与路线 B（高成本）两种方案；最终走路线 A。
- 用户要求先做计划 + 可验证测试。
- 产出计划包含：
  - 改造范围（data / engine / model / scripts）
  - 分阶段目标（Phase 1/2/3）
  - UT 与 E2E 验收口径（shape、时序、loss、baseline 对比、ckpt 推理）

### 阶段 C：实现推进与里程碑提交

根据聊天记录，关键提交与产物包括：

- `452aa6a`：视频下一帧 MVP 主体改造 + 测试骨架
- `dae53f5`：smoke 与 baseline 评估脚本
- `07fb0d0`：修复设备兼容问题（CPU/device portability）
- `4e34573`：清理残留 `.cuda()`，保证 CPU E2E 可跑
- `4a61f06`：可视化 E2E 报告（含图表、日志、指标）

### 阶段 D：中断插曲（关键观察）

- 聊天中出现一次明显上下文断裂：
  - 用户问「之前 mar 改代码任务进展」，助手先回答"没有历史上下文/未在执行"。
  - 后续通过翻历史会话才恢复并确认：任务确实已执行并有 commit。
- 该插曲说明：
  - 跨会话记忆/上下文连续性是实验中的真实风险点。
  - 需要显式"任务状态持久化"机制，避免用户二次追问时信息丢失。

---

## 3. 技术实现摘要

本次实现选择「路线 A（MVP）」：

1. **数据层**：从图片样本切到视频连续帧样本
2. **模型层**：引入 `forward_video(...)` / `sample_next_frame(...)` 等视频路径
3. **训练层**：支持 context frame + target frame 的训练输入
4. **验证层**：补齐 smoke、baseline、tiny-overfit、ckpt 加载推理

关键设计思想：

- 不一次性重写为 3D VAE 高复杂方案；
- 先构建"能跑、能测、能收敛"的最小可用闭环。

---

## 4. 测试设计与结果（来自聊天记录）

### 4.1 单元测试（UT）

关注点：

- Video loader shape 与时序是否正确
- video forward 是否产出有效标量 loss
- context 改变时输出是否变化（依赖性）

结果：

- `pytest -q tests/test_video_loader.py tests/test_mar_video_forward.py`
- **4 passed**（符合预期）

### 4.2 端到端（E2E）

关注点：

- smoke 全链路是否打通
- baseline（copy-last-frame）是否可计算
- tiny-overfit 是否出现 loss 下降
- ckpt 保存/加载后是否可推理

记录到的结果（聊天中汇报）：

- Smoke：`loss ≈ 0.98`，`baseline_psnr ≈ 26.62`
- Baseline：`copy-last avg PSNR ≈ 26.62`
- Tiny-overfit：loss 出现下降（约 +0.047 改善）
- ckpt 加载后推理可运行（形状正常）

### 4.3 真实视频数据端到端验证（详细过程）

为避免仅在 dummy 数据上“自洽”，本实验进一步执行了真实视频数据验证，流程如下。

#### Step 1：下载真实 mp4 样例视频

- 原始视频：`data/real_videos_raw/bbb.mp4`
- 来源：公开 sample 视频

示例命令：

```bash
cd /Users/niubobo/.openclaw/workspace/mar
mkdir -p data/real_videos_raw
curl -L --fail -o data/real_videos_raw/bbb.mp4 https://samplelib.com/lib/preview/mp4/sample-5s.mp4
```

#### Step 2：从 mp4 抽帧并构建评估数据集

- 抽帧策略：每 3 帧取 1 帧
- 预处理：统一 resize 到 `128x128`
- 组织方式：切分为 3 段短序列，每段 10 帧
  - `data/real_eval_frames/video_000`
  - `data/real_eval_frames/video_001`
  - `data/real_eval_frames/video_002`

该格式与 `util/video_loader.py` 约定一致，可直接用于 `context_len=4` 的视频下一帧任务。

#### Step 3：跑真实数据 smoke（链路可运行性）

命令（节选）：

```bash
PYTHONPATH=. python3 scripts/smoke_video_next_frame.py \
  --video_data_path ./data/real_eval_frames \
  --img_size 64 --context_len 4 --batch_size 1 --device cpu \
  --vae_path pretrained_models/vae/kl16.ckpt
```

结果（真实数据报告记录）：

- `Smoke OK`
- `loss=1.001144`
- `baseline_psnr=31.9340`

#### Step 4：跑 baseline（copy-last-frame）

命令（节选）：

```bash
PYTHONPATH=. python3 scripts/eval_copy_last_baseline.py \
  --video_data_path ./data/real_eval_frames \
  --context_len 4 --img_size 64 --batch_size 2 --max_batches 8
```

结果：

- `copy-last baseline avg PSNR over 8 batches: 30.5839`

#### Step 5：跑 tiny-overfit（可学习性验证）

命令（节选）：

```bash
PYTHONPATH=. python3 scripts/openclaw_tiny_overfit_real_eval.py \
  --video_data_path ./data/real_eval_frames \
  --report_dir ./reports/e2e_report_real_2026-03-07 \
  --img_size 64 --context_len 4 --steps 30 --lr 3e-5 --device cpu \
  --vae_path pretrained_models/vae/kl16.ckpt
```

结果：

- `loss_start5_avg=1.0173`
- `loss_end5_avg=0.9207`
- `loss_improvement=+0.0966`（loss 明显下降）
- `pred_latent_shape=[1,16,4,4]`

并产出可视化：

- `tiny_overfit_loss_curve.png`
- `triptych_context_gt_pred.png`

#### Step 6：结果文件与可审计材料

完整材料已沉淀在：

- `reports/e2e_report_real_2026-03-07/REPORT.md`
- `reports/e2e_report_real_2026-03-07/unit_tests.txt`
- `reports/e2e_report_real_2026-03-07/smoke_test.txt`
- `reports/e2e_report_real_2026-03-07/baseline_eval.txt`
- `reports/e2e_report_real_2026-03-07/tiny_overfit_metrics.json`

### 4.4 这些 E2E 是否最终证明“满足需求”？

结论分层如下：

1. **已被证明（MVP层）**
   - 代码改造链路成立：前 n 帧条件输入 + 下一帧目标训练路径可运行；
   - 在真实视频帧数据上可完成训练/推理/可视化；
   - 具备可测试、可复现、可审计证据链（UT + E2E + 日志 + 图）。

2. **尚未被完全证明（产品/研究完备层）**
   - 目前主要证明“可运行与可学习”，尚未在更大规模真实数据上完成系统性指标对比（如 PSNR/SSIM/LPIPS 全量统计）；
   - 尚未给出多视频、多场景、长时预测稳定性的充分统计结论。

因此，本次结论应表述为：

- **本次代码改动已满足“视频下一帧预测 MVP 验证目标”**；
- 若按“生产级/研究完备”标准，还需补充大规模评估与更严格对照实验。

---

## 5. 实验经验与教训

### 5.1 做得好的地方

1. 先定义验收口径，再改代码，减少"改完无法证明有效"的风险。
2. 采用分阶段推进（计划→实现→验证→可视化报告），过程透明。
3. 遇到 CPU/CUDA 兼容问题时，快速定位到 `.cuda()` 硬编码并系统性修复。

### 5.2 暴露的问题

1. **上下文连续性问题**：中途出现"已做工作暂时不可见"的断点。
2. 主动汇报节奏不稳定：有重复确认、重复问"是否继续"的交互噪音。
3. 在用户高频催促（hello/重复请求）场景下，缺少统一"进度看板链接"应对方式。

### 5.3 可复用改进建议

1. 在项目根目录维护 `STATUS.md`（当前阶段、最近 commit、下一步）。
2. 关键里程碑后自动写入 `chat_history.md` 或 `progress_log.md`，降低跨会话丢失。
3. 固化"每阶段必须输出"模板：
   - 改了什么
   - 怎么测
   - 测试结果
   - 剩余风险
4. 避免重复确认型消息，改为"默认持续推进 + 里程碑汇报"。

---

## 6. 本次实验结论

本实验证明：

- OpenClaw 在真实工程任务中可以完成从需求到交付的工程闭环；
- 在有明确计划和测试验收标准时，执行质量显著更高；
- 最大风险不在"能不能改代码"，而在"跨会话状态保持与进度可追踪性"。

总体判断：

- **实验成功（MVP 层面）**，并形成了可复用的方法论（计划先行、测试驱动、阶段汇报、结果可视化）。

---

## 7. 实验中的重大问题复盘（时间线 + 根因 + 改进）

以下问题按影响度排序，重点覆盖"发生了什么、为什么发生、改哪里、怎么改"。

### P1. 上下文断裂：已做工作被误判为未做（最高优先级）

**时间线**

- 03-07 03:00~09:57：任务持续推进，已有多次 commit、测试、E2E。
- 03-07 08:43~08:47：用户询问 mar 改造进展时，先出现"没有历史上下文/未在执行"的错误判断。
- 随后通过"翻历史会话"恢复真实状态，确认已有提交和推进记录。

**根因分析**

1. 缺少项目内"单一真相源"（Single Source of Truth）记录任务状态。
2. 过度依赖会话临时上下文，跨段后容易丢脉络。
3. 回答进展前缺少"先查 repo 事实"的硬约束流程。

**改进建议（改哪里）**

- 新增 `mar/STATUS.md`
- 新增 `mar/progress_log.md`
- 固化工作流：每个里程碑必须写状态

**改进建议（怎么改）**

- 每次关键动作后落盘：当前阶段（Plan/Impl/Test/Report）、最新 commit、已完成/阻塞/下一步。
- 回答"当前进展"前必须先执行：
  1. `git log --oneline -n 5`
  2. `git status --short`
  3. 读取 `STATUS.md`
- 未查事实时，不给进展结论。

### P2. 设备兼容性问题：`.cuda()` 硬编码导致 E2E 阻塞

**时间线**

- UT 跑通后进入 E2E。
- 03-07 09:09：明确定位 CPU 环境失败（`torch.cuda.is_available() = False`），失败原因是残留 `.cuda()` 硬编码。
- 后续修复 `models/mar.py`、`models/diffloss.py`、`diffusion/gaussian_diffusion.py` 后，E2E 恢复。

**根因分析**

1. 默认假设 GPU 可用，device 管理不统一。
2. UT 对 CPU-only 路径覆盖不足，问题未提前暴露。

**改进建议（改哪里）**

- 模型与扩散核心路径（上述三个文件）
- `tests/` 增强 device 维度测试

**改进建议（怎么改）**

- 统一 `to(device)`，禁止裸 `.cuda()`。
- 新增本地门禁：grep 扫描 `.cuda(`（白名单除外）。
- 把 CPU smoke 设为必跑项；GPU 可用时补充 GPU smoke。

### P3. 沟通节奏噪音：重复确认"要不要继续"

**时间线**

- 03:10~03:26 出现多次重复确认型消息。
- 用户明确提出"以后不要问，持续推进直到完成目标"。

**根因分析**

1. 缺少默认执行策略约定。
2. 缺少固定里程碑汇报模板，导致打断式沟通。

**改进建议（改哪里）**

- 新增 `mar/RUNBOOK.md`
- 在协作约定中明确"默认持续推进"

**改进建议（怎么改）**

- 仅在 blocker（权限、破坏性动作、方向冲突）时中断请示。
- 固定里程碑更新频率（如 45~60 分钟/次）。
- 汇报模板固定四项：改了什么 / 怎么验证 / 结果 / 下一步。

### P4. 高频催促场景缺少进度看板锚点

**时间线**

- 09:37 后用户多次重复要求"整理材料 + 可视化报告"。
- 最终交付报告，但中间缺少统一可追踪入口。

**根因分析**

- 进度信息散落在聊天消息中，而非 repo 固定索引文件。

**改进建议（改哪里）**

- 新增 `mar/reports/README.md`
- 与 `mar/STATUS.md` 联动

**改进建议（怎么改）**

- 固定"当前产物索引页"并持续更新：
  - 最新报告路径
  - 最新图表路径
  - 最新测试日志路径
  - 最新 commit
- 聊天回复优先给"看板位置 + 关键状态"。

### 小结

- 核心成功点：计划先行 + 测试驱动，最终完成 MVP 验证闭环。
- 核心失败点：状态管理未产品化，造成阶段性上下文断裂。
- 最大改进杠杆：把"进度与事实"沉淀到 repo 文件（`STATUS.md` + `progress_log.md` + `reports/README.md`）。

---

## 8. 建议的后续动作

1. 在 `mar` 项目内新增 `STATUS.md` 与 `RUNBOOK.md`。
2. 把当前 E2E 报告与本报告串联成一个总索引页（`reports/README.md`）。
3. 如需"研究版"结题，可继续补：
   - 更多数据集上的对比
   - 多组超参实验
   - 可视化质量评估（GT vs Pred）

---

## 附录：本报告使用的材料

- `mar/chat_history.md`
- 聊天中提及的提交：`452aa6a`、`dae53f5`、`07fb0d0`、`4e34573`、`4a61f06`
- 聊天中汇报的 UT/E2E 结果与可视化产物路径（`reports/e2e_report_2026-03-07/`）
