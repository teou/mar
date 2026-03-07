import os

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from util.video_loader import VideoFrameDataset


def _write_dummy_frame(path, size=(32, 32), value=0):
    img = Image.new("RGB", size, color=(value, value, value))
    img.save(path)


def test_video_loader_shapes_and_order(tmp_path):
    root = tmp_path / "videos"
    v0 = root / "video_000"
    v0.mkdir(parents=True)

    for i in range(6):
        _write_dummy_frame(v0 / f"{i:06d}.png", value=i)

    to_tensor = lambda x: pil_to_tensor(x).float() / 255.0
    ds = VideoFrameDataset(str(root), context_len=4, transform=to_tensor)

    assert len(ds) == 2
    context, target, meta = ds[0]
    assert context.shape == (4, 3, 32, 32)
    assert target.shape == (3, 32, 32)
    assert meta["context_indices"] == [0, 1, 2, 3]
    assert meta["target_index"] == 4


def test_video_loader_requires_enough_frames(tmp_path):
    root = tmp_path / "videos"
    v0 = root / "video_short"
    v0.mkdir(parents=True)
    for i in range(2):
        _write_dummy_frame(v0 / f"{i:06d}.png")

    try:
        VideoFrameDataset(str(root), context_len=4, transform=None)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
