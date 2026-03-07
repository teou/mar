# OpenClaw 严格一致性检查

- Date: 2026-03-07 11:39:53 CST
- Repo: /Users/niubobo/.openclaw/workspace/mar
- Git HEAD: cd43e19f30fd15bb903031ccc13a7fa4de41e210
- Git Branch: main

## 1) Git 工作区状态
```
?? reports/strict_consistency_openclaw_2026-03-07/
```

## 2) 关键文件存在性检查
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
- ✅ scripts/tiny_overfit_real_eval.py
- ✅ reports/e2e_report_real_2026-03-07/REPORT.md
- ✅ reports/e2e_report_real_2026-03-07/unit_tests.txt
- ✅ reports/e2e_report_real_2026-03-07/smoke_test.txt
- ✅ reports/e2e_report_real_2026-03-07/baseline_eval.txt
- ✅ reports/e2e_report_real_2026-03-07/tiny_overfit_metrics.json
- ✅ reports/e2e_report_real_2026-03-07/tiny_overfit_loss_curve.png
- ✅ reports/e2e_report_real_2026-03-07/triptych_context_gt_pred.png

## 3) 语义一致性断言（grep）
- video task args in main_mar.py
100:    parser.add_argument('--video_data_path', default='', type=str,
102:    parser.add_argument('--task', default='image_gen', type=str, choices=['image_gen', 'video_next_frame'],
104:    parser.add_argument('--context_len', default=4, type=int,
105:                        help='number of context frames for video_next_frame task')
175:    if args.task == 'video_next_frame':
176:        if not args.video_data_path:
177:            raise ValueError("--video_data_path is required when --task video_next_frame")
178:        dataset_train = VideoFrameDataset(args.video_data_path, context_len=args.context_len, transform=transform_train)
219:        context_len=args.context_len,
- video path in engine_mar.py
54:        if args.task == 'video_next_frame':
64:                context_tokens = model.module.patchify(x_ctx) if hasattr(model, 'module') else model.patchify(x_ctx)
69:                target_tokens = model.module.patchify(x_tgt) if hasattr(model, 'module') else model.patchify(x_tgt)
- video APIs in models/mar.py
43:                 context_len=4,
66:        self.context_len = context_len
280:    def forward_video(self, context_tokens, target_tokens, labels=None):
297:    def sample_next_frame(self, context_tokens, num_iter=64, cfg=1.0, cfg_schedule="linear", temperature=1.0, progress=False):
- device hardcode checks (.cuda)
  models/mar.py:
  (none)
  models/diffloss.py:
  (none)
  diffusion/gaussian_diffusion.py:
  (none)

## 4) 关键文件 SHA256
9a9896811d4aca41bbfbf8007cd64c5f5b35a95e6c3ffb23d1a6d356147df082  main_mar.py
9f9df33e926c4c06c12eb075f032025c0238e14d4bcb7ea0a15e351646afee6f  engine_mar.py
d42db0924f40fd4dd6ee1bc2202598255d819c344ca42284516b9472f09dff6b  models/mar.py
0c4fbb5ca28d382e2c5d90aca8b98a01d02745ad0aa4edf0121e26803598932a  models/diffloss.py
41d40ef0da129471b660026d4ef36d339283a5c088c7f95fc5c1b4be04a37bcf  diffusion/gaussian_diffusion.py
7389ab4c3353014af0b81e1cf3ebe2f4be9badd8c272615d13991a5e625fa4b0  util/video_loader.py
b5110e19c9900a6fbf62193229dc4bbeb5e5ee583c87e4cc80ff0c420fff5681  tests/test_video_loader.py
d31451d44fabcadcc698e4b4ef00e79af23141c079d43e4b9a55c63dee4366cf  tests/test_mar_video_forward.py
1ae34828bd216c56012b86fbdedc489d37cbc0ee49a10b4168fec51f17ddd5ec  scripts/smoke_video_next_frame.py
cbafa838c1d58d2f98e3c96dbc848588fd33f8265753c40ed2c08a6808bbd84c  scripts/eval_copy_last_baseline.py
4f37917d8f80f7c90b999520421ed362d96b283ac2e6aaefdc5e5d96a6f36d23  scripts/tiny_overfit_real_eval.py
188b4b7c7aace2bc417cc21d0ca91e4305b8fd9823911930ebdedacd340c16c5  reports/e2e_report_real_2026-03-07/REPORT.md
1b505fd768aa7a5706f17c654adace5d0bbfa2d853c23ea93abaff6dbd6ed2c0  reports/e2e_report_real_2026-03-07/unit_tests.txt
22594c78bec2ac6a18356bb0d2f1ba67ac3932f4511cb8c27e3a5d4f1740b83a  reports/e2e_report_real_2026-03-07/smoke_test.txt
20dcb8d4fe298dcdb3c94fea033e1cf739ac7f8cf180c522f5b2da1865c3130d  reports/e2e_report_real_2026-03-07/baseline_eval.txt
0d7a3fabc2ceecad2fb36a74448d366d51aa519316cf7c7bd67bd2b29f08aa06  reports/e2e_report_real_2026-03-07/tiny_overfit_metrics.json

## 5) 快速执行回归（UT）
```
....                                                                     [100%]
=============================== warnings summary ===============================
../../../../../Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/utils/generic.py:441
  /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
    _torch_pytree._register_pytree_node(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
4 passed, 1 warning in 9.91s
```
