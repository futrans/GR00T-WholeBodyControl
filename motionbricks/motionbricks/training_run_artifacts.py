from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def build_trainer_artifacts(
    *,
    run_dir: str | None,
    save_every_n_steps: int,
    csv_logger_cls: type,
    checkpoint_cls: type,
) -> tuple[bool, Any, list[Any]]:
    if not run_dir:
        return False, False, []

    run_path = Path(run_dir)
    checkpoint_dir = run_path / "checkpoints"
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = csv_logger_cls(save_dir=str(run_path), name="logs", version="")
    checkpoint_callback = checkpoint_cls(
        dirpath=str(checkpoint_dir),
        filename="step-{step}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=save_every_n_steps,
    )
    return True, logger, [checkpoint_callback]
