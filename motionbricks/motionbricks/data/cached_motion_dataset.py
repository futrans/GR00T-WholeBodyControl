from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CachedMotionDataset(Dataset):
    def __init__(self, cache_root: str | Path, *, dtype: torch.dtype = torch.float32):
        self.cache_root = Path(cache_root)
        self.dtype = dtype
        self.manifest = json.loads((self.cache_root / "manifest.json").read_text(encoding="utf-8"))
        self.entries = [
            json.loads(line)
            for line in (self.cache_root / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        motion_name = self.manifest.get("files", {}).get("motion", "motion.npy")
        self.motion = np.load(self.cache_root / motion_name, mmap_mode="r")
        if self.motion.ndim != 2:
            raise ValueError(f"cached motion must be [N, D], got {self.motion.shape}")

        feature_dim = int(self.manifest.get("feature_dim", self.motion.shape[1]))
        if self.motion.shape[1] != feature_dim:
            raise ValueError(
                f"cached motion dim {self.motion.shape[1]} != manifest feature_dim {feature_dim}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, object]:
        entry = self.entries[idx]
        offset = int(entry["offset"])
        num_frames = int(entry["num_frames"])
        motion = np.array(self.motion[offset : offset + num_frames], dtype=np.float32, copy=True)
        return {
            "keyid": str(entry["sequence_id"]),
            "motion": torch.as_tensor(motion, dtype=self.dtype),
        }
