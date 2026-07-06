from scripts.train_vqvae import load_config


def test_load_config_accepts_explicit_parallel_trainer_overrides(tmp_path):
    result_dir = tmp_path / "result_dir"
    version_dir = result_dir / "motionbricks_vqvae" / "version_1"
    version_dir.mkdir(parents=True)
    (version_dir / "hparams.yaml").write_text(
        """
data: {}
skeleton:
  folder: old
motion_rep:
  stats:
    folder: old
trainer:
  devices: 1
  num_nodes: 1
  max_steps: 1
  accelerator: auto
  strategy: auto
  enable_progress_bar: false
  log_every_n_steps: 1
  val_check_interval: 1
  num_sanity_val_steps: 0
model:
  scheduler:
    num_training_steps: 1
""".lstrip(),
        encoding="utf-8",
    )

    conf, _ = load_config(
        str(result_dir),
        max_steps=1000,
        devices=8,
        num_nodes=1,
        accelerator="gpu",
        strategy="ddp",
    )

    assert conf.trainer.devices == 8
    assert conf.trainer.num_nodes == 1
    assert conf.trainer.accelerator == "gpu"
    assert conf.trainer.strategy == "ddp"
    assert conf.trainer.max_steps == 1000
    assert conf.model.scheduler.num_training_steps == 1000
