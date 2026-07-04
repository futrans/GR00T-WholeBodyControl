from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from motionbricks.data.cached_motion_dataset import CachedMotionDataset
from motionbricks.data.synthetic_dataset import SyntheticMotionDataset, collate_batch


def build_motion_training_dataset(
    *,
    feat_dim: int,
    dataset_cache: str | None,
    num_samples: int,
    min_frames: int,
    max_frames: int,
) -> Dataset:
    if dataset_cache:
        return CachedMotionDataset(dataset_cache)
    return SyntheticMotionDataset(
        feat_dim=feat_dim,
        num_samples=num_samples,
        min_frames=min_frames,
        max_frames=max_frames,
    )


def build_motion_training_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        persistent_workers=num_workers > 0,
    )
