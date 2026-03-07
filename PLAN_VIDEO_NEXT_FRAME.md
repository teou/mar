# Video Next-Frame Adaptation Plan (MAR)

## Goal
Adapt MAR from class-conditional image generation to conditional next-frame prediction:
- Input: previous n frames
- Output: next frame
- First milestone: MVP with 2D VAE per-frame latent encoding

## Scope
1. Data pipeline for video clips
2. Model API extension for video conditioning
3. Training loop updates
4. Unit tests + end-to-end smoke validation

## Milestones
### Phase 1: Data + API plumbing
- Add `util/video_loader.py` with `VideoFrameDataset`
- Add CLI flags in `main_mar.py`:
  - `--task {image_gen,video_next_frame}`
  - `--video_data_path`
  - `--context_len`
- Add minimal collate/loader support

Validation:
- UT: dataset shapes and temporal order
- Smoke: one forward pass

### Phase 2: Model adaptation (MVP)
- Extend `MAR` with video-condition forward path:
  - `forward_video(context_tokens, target_tokens, labels=None)`
  - `sample_next_frame(context_tokens, ...)`
- Keep DiffLoss unchanged in MVP
- Condition on context via prepended context buffer + time embedding

Validation:
- UT: loss scalar and finite
- UT: output shape/range in inference
- UT: output changes when context changes

### Phase 3: Training + E2E checks
- Wire `engine_mar.py` for video batches
- Add single-GPU smoke script for short run

Validation:
- E2E: tiny overfit set reduces loss
- E2E: checkpoint load + inference output image
- E2E: baseline compare against copy-last-frame (PSNR/SSIM)

## Deliverables
- Code diffs
- Test files and pass logs
- Sample outputs (context/gt/pred)
- Follow-up TODOs for full spatiotemporal model
