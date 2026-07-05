from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def parse_wandb_tags(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    tags = [part.strip() for part in raw_value.split(",") if part.strip()]
    return tags or None


def build_trainer_artifacts(
    *,
    run_dir: str | None,
    save_every_n_steps: int,
    csv_logger_cls: type,
    checkpoint_cls: type,
    logger_mode: str = "csv",
    wandb_logger_cls: type | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: list[str] | None = None,
) -> tuple[bool, Any, list[Any]]:
    if not run_dir:
        return False, False, []
    if logger_mode not in {"none", "csv", "wandb", "both"}:
        raise ValueError(f"unsupported logger mode: {logger_mode}")
    if logger_mode in {"wandb", "both"} and wandb_logger_cls is None:
        raise ValueError("wandb logger mode requires wandb_logger_cls")

    run_path = Path(run_dir)
    checkpoint_dir = run_path / "checkpoints"
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    loggers: list[Any] = []
    if logger_mode in {"csv", "both"}:
        loggers.append(csv_logger_cls(save_dir=str(run_path), name="logs", version=""))
    if logger_mode in {"wandb", "both"}:
        loggers.append(
            wandb_logger_cls(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                name=wandb_name,
                save_dir=str(run_path),
                tags=wandb_tags,
            )
        )
    if logger_mode == "none":
        logger: Any = False
    elif len(loggers) == 1:
        logger = loggers[0]
    else:
        logger = loggers
    checkpoint_callback = checkpoint_cls(
        dirpath=str(checkpoint_dir),
        filename="step-{step}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=save_every_n_steps,
    )
    return True, logger, [checkpoint_callback]
