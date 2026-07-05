"""Root model training script using synthetic data.

Demonstrates how the root backbone training pipeline works without
requiring the actual motion dataset. Loads the saved model config from
the checkpoint directory and trains on randomly generated motion tensors.

The root model does not require a pretrained VQVAE — it directly
predicts continuous root motion values.

Usage:
    python scripts/train_root.py --max_steps 100
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_RESULT_DIR = str(Path(__file__).resolve().parents[1] / "out")

from motionbricks.data.training_dataset_factory import (
    build_motion_training_dataloader,
    build_motion_training_dataset,
)
from motionbricks.helper.pl_util import load_motion_rep
from motionbricks.training_run_artifacts import build_trainer_artifacts, parse_wandb_tags, positive_int


def load_config(result_dir: str, max_steps: int):
    """Load and patch hparams.yaml for single-GPU training."""
    version_dir = os.path.join(result_dir, "motionbricks_root", "version_1")
    hparams_path = os.path.join(version_dir, "hparams.yaml")
    conf = OmegaConf.load(hparams_path)

    with open_dict(conf):
        # resolve data paths to the version directory (where skeleton/stats live)
        conf.data = {"folder": version_dir, "text_embeddings": None}
        conf.skeleton.folder = os.path.join(version_dir, "skeleton")
        conf.motion_rep.stats.folder = os.path.join(version_dir, "stats", "motion")

        # single-GPU training overrides
        conf.trainer.devices = 1
        conf.trainer.num_nodes = 1
        conf.trainer.max_steps = max_steps
        conf.trainer.accelerator = "auto"
        conf.trainer.strategy = "auto"
        conf.trainer.enable_progress_bar = True
        conf.trainer.log_every_n_steps = 10
        conf.trainer.val_check_interval = max_steps
        conf.trainer.num_sanity_val_steps = 0

        # resolve ${trainer.max_steps} in scheduler
        conf.model.scheduler.num_training_steps = max_steps

        # remove keys with unresolvable ${hydra:...} interpolations
        conf.id = "synthetic"
        conf.run_dir = "."
        conf.out_dir = result_dir

    # resolve all ${} interpolations, then re-wrap as DictConfig
    resolved = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(resolved)

    return conf, version_dir


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main():
    _configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Root model training")
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR,
                        help="Directory containing pretrained checkpoints")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of synthetic samples in dataset")
    parser.add_argument("--dataset_cache", type=str, default=None,
                        help="MotionBricks packed cache root")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader worker count")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Optional directory for CSV logs and checkpoints")
    parser.add_argument("--save_every_n_steps", type=positive_int, default=500,
                        help="Checkpoint interval when --run_dir is set")
    parser.add_argument("--logger", choices=["none", "csv", "wandb", "both"], default="csv",
                        help="Logger backend when --run_dir is set")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="Comma-separated wandb tags")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    conf, version_dir = load_config(args.result_dir, args.max_steps)
    if args.dataset_cache:
        with open_dict(conf):
            conf.motion_rep.stats.folder = os.path.join(args.dataset_cache, "stats", "motion")

    # instantiate skeleton and motion representation
    motion_rep = load_motion_rep(conf)
    feat_dim = len(motion_rep.indices['all'])

    dataset = build_motion_training_dataset(
        feat_dim=feat_dim,
        dataset_cache=args.dataset_cache,
        num_samples=args.num_samples,
        min_frames=200,
        max_frames=400,
    )
    dataloader = build_motion_training_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    # instantiate networks and model
    model_conf = copy.deepcopy(conf.model)
    with open_dict(model_conf):
        # instantiate backbone network (needs full motion_rep for dual_rep access)
        backbone_net = instantiate(
            model_conf.backbone_network,
            motion_rep=motion_rep,
            _recursive_=False,
        )

        # build optimizer and scheduler as partials
        optimizer_fn = instantiate(model_conf.optimizer)
        scheduler_fn = instantiate(model_conf.scheduler) if model_conf.scheduler else None

        model = instantiate(
            model_conf,
            pose_vqvae_network=None,
            root_vqvae_network=None,
            backbone_network=backbone_net,
            motion_rep=motion_rep,
            optimizer=optimizer_fn,
            scheduler=scheduler_fn,
            _recursive_=False,
        )

    enable_checkpointing, logger, callbacks = build_trainer_artifacts(
        run_dir=args.run_dir,
        save_every_n_steps=args.save_every_n_steps,
        csv_logger_cls=CSVLogger,
        checkpoint_cls=ModelCheckpoint,
        logger_mode=args.logger,
        wandb_logger_cls=WandbLogger,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_name=args.wandb_name,
        wandb_tags=parse_wandb_tags(args.wandb_tags),
    )

    trainer = pl.Trainer(
        max_steps=conf.trainer.max_steps,
        devices=conf.trainer.devices,
        num_nodes=conf.trainer.num_nodes,
        accelerator=conf.trainer.accelerator,
        strategy=conf.trainer.strategy,
        precision=conf.trainer.precision,
        gradient_clip_val=conf.trainer.gradient_clip_val,
        enable_progress_bar=conf.trainer.enable_progress_bar,
        log_every_n_steps=conf.trainer.log_every_n_steps,
        num_sanity_val_steps=0,
        enable_checkpointing=enable_checkpointing,
        logger=logger,
        callbacks=callbacks,
    )

    print(f"Starting root model training for {args.max_steps} steps...")
    print(f"  Feature dim: {feat_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dataset type: {'cached' if args.dataset_cache else 'synthetic'}")
    if args.dataset_cache:
        print(f"  Dataset cache: {args.dataset_cache}")
    if args.run_dir:
        print(f"  Run dir: {args.run_dir}")
        print(f"  Checkpoint interval: {args.save_every_n_steps} steps")
        print(f"  Logger: {args.logger}")
    print(f"  Dataset size: {len(dataset)}")
    trainer.fit(model, train_dataloaders=dataloader)
    print("Training complete.")


if __name__ == "__main__":
    main()
