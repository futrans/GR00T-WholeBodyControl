from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_CACHE_SCHEMA_VERSION = "motionbricks_bones_seed_g1_packed_cache_v1"
_FEATURE_DIM = 414
_QPOS_DIM = 36


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

        self._validate_cache()

    def _validate_cache(self) -> None:
        if self.manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
            raise ValueError("manifest schema_version mismatch")

        if int(self.manifest.get("feature_dim", -1)) != _FEATURE_DIM:
            raise ValueError("manifest feature_dim mismatch")

        if int(self.manifest.get("qpos_dim", -1)) != _QPOS_DIM:
            raise ValueError("manifest qpos_dim mismatch")

        if self.motion.shape[1] != _FEATURE_DIM:
            raise ValueError(
                f"cached motion dim {self.motion.shape[1]} != manifest feature_dim {_FEATURE_DIM}"
            )

        if int(self.manifest.get("num_sequences", -1)) != len(self.entries):
            raise ValueError("manifest num_sequences mismatch")

        if int(self.manifest.get("total_frames", -1)) != self.motion.shape[0]:
            raise ValueError("manifest total_frames mismatch")

        expected_offset = 0
        total_frames = self.motion.shape[0]
        for entry in self.entries:
            sequence_id = str(entry["sequence_id"])
            offset = int(entry["offset"])
            num_frames = int(entry["num_frames"])
            feature_dim = int(entry["feature_dim"])
            qpos_dim = int(entry["qpos_dim"])

            if offset != expected_offset:
                raise ValueError(f"index offset for {sequence_id} mismatch")

            if feature_dim != _FEATURE_DIM:
                raise ValueError(f"index feature_dim for {sequence_id} mismatch")

            if qpos_dim != _QPOS_DIM:
                raise ValueError(f"index qpos_dim for {sequence_id} mismatch")

            if offset < 0 or num_frames < 0 or offset + num_frames > total_frames:
                raise ValueError(f"index window for {sequence_id} exceeds motion array")

            expected_offset += num_frames

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, object]:
        entry = self.entries[idx]
        sequence_id = str(entry["sequence_id"])
        offset = int(entry["offset"])
        num_frames = int(entry["num_frames"])
        if offset < 0 or num_frames < 0 or offset + num_frames > self.motion.shape[0]:
            raise ValueError(f"index window for {sequence_id} exceeds motion array")
        motion = np.array(self.motion[offset : offset + num_frames], dtype=np.float32, copy=True)
        return {
            "keyid": sequence_id,
            "motion": torch.as_tensor(motion, dtype=self.dtype),
        }
