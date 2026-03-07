import os
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_frames(video_dir: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(video_dir)):
        if name.lower().endswith(IMG_EXTS):
            files.append(os.path.join(video_dir, name))
    return files


class VideoFrameDataset(Dataset):
    """
    Directory layout:
      root/
        video_000/
          000001.png
          000002.png
          ...
        video_001/
          ...

    Returns:
      context: Tensor [T, C, H, W]
      target: Tensor [C, H, W]
      meta: dict with indices for testing/debugging
    """

    def __init__(self, root: str, context_len: int = 4, transform=None):
        self.root = root
        self.context_len = context_len
        self.transform = transform

        self.samples: List[Tuple[List[str], str, dict]] = []
        self._build_index()

    def _build_index(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Video dataset path does not exist: {self.root}")

        for video_name in sorted(os.listdir(self.root)):
            video_dir = os.path.join(self.root, video_name)
            if not os.path.isdir(video_dir):
                continue

            frames = _list_frames(video_dir)
            if len(frames) <= self.context_len:
                continue

            for target_idx in range(self.context_len, len(frames)):
                context_paths = frames[target_idx - self.context_len: target_idx]
                target_path = frames[target_idx]
                meta = {
                    "video": video_name,
                    "context_indices": list(range(target_idx - self.context_len, target_idx)),
                    "target_index": target_idx,
                }
                self.samples.append((context_paths, target_path, meta))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid video samples found under {self.root}. "
                f"Need at least context_len+1 frames per video directory."
            )

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path: str):
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(img)
        # fallback to tensor-like behavior if no transform (avoid numpy dependency)
        return pil_to_tensor(img).float() / 255.0

    def __getitem__(self, index: int):
        context_paths, target_path, meta = self.samples[index]
        context = [self._load_frame(p) for p in context_paths]
        context = torch.stack(context, dim=0)
        target = self._load_frame(target_path)
        return context, target, meta
