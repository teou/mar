# OpenClaw 严格一致性检查（复跑）

- Date: 2026-03-07 11:42:58 CST
- Git HEAD: 9d9f82a47e40d2a8a9ae1cd80c070690b9c9fb69

## 1) 关键文件存在性
- ✅ main_mar.py
- ✅ engine_mar.py
- ✅ models/mar.py
- ✅ models/diffloss.py
- ✅ diffusion/gaussian_diffusion.py
- ✅ util/video_loader.py
- ✅ tests/test_video_loader.py
- ✅ tests/test_mar_video_forward.py
- ✅ scripts/smoke_video_next_frame.py
- ✅ scripts/eval_copy_last_baseline.py
- ✅ scripts/openclaw_tiny_overfit_real_eval.py
- ✅ reports/e2e_report_real_2026-03-07/REPORT.md

## 2) 语义断言
- main_mar video args:
100:    parser.add_argument('--video_data_path', default='', type=str,
102:    parser.add_argument('--task', default='image_gen', type=str, choices=['image_gen', 'video_next_frame'],
104:    parser.add_argument('--context_len', default=4, type=int,
105:                        help='number of context frames for video_next_frame task')
175:    if args.task == 'video_next_frame':
176:        if not args.video_data_path:
177:            raise ValueError("--video_data_path is required when --task video_next_frame")
178:        dataset_train = VideoFrameDataset(args.video_data_path, context_len=args.context_len, transform=transform_train)
219:        context_len=args.context_len,
- engine_mar video branch:
54:        if args.task == 'video_next_frame':
64:                context_tokens = model.module.patchify(x_ctx) if hasattr(model, 'module') else model.patchify(x_ctx)
69:                target_tokens = model.module.patchify(x_tgt) if hasattr(model, 'module') else model.patchify(x_tgt)
- mar video APIs:
43:                 context_len=4,
66:        self.context_len = context_len
280:    def forward_video(self, context_tokens, target_tokens, labels=None):
297:    def sample_next_frame(self, context_tokens, num_iter=64, cfg=1.0, cfg_schedule="linear", temperature=1.0, progress=False):
- no hardcoded cuda:
  models/mar.py
  (none)
  models/diffloss.py
  (none)
  diffusion/gaussian_diffusion.py
  (none)

## 3) 单测快检
```
....                                                                     [100%]
=============================== warnings summary ===============================
../../../../../Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/utils/generic.py:441
  /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    _torch_pytree._register_pytree_node(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
4 passed, 1 warning in 10.85s
```
