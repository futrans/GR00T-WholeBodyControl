from pathlib import Path

from scripts.train_pose import load_config


def test_load_config_accepts_explicit_vqvae_checkpoint_name(tmp_path):
    result_dir = tmp_path / "result_dir"
    version_dir = result_dir / "motionbricks_pose" / "version_1"
    version_dir.mkdir(parents=True)
    vqvae_path = result_dir / "motionbricks_vqvae" / "version_1" / "checkpoints" / "model-step=2000000.ckpt"
    (version_dir / "hparams.yaml").write_text(
        f"""
data: {{}}
skeleton:
  folder: old
motion_rep:
  stats:
    folder: old
trainer:
  devices: 8
  num_nodes: 2
  max_steps: 1
  accelerator: gpu
  strategy: ddp
  enable_progress_bar: false
  log_every_n_steps: 1
  val_check_interval: 1
  num_sanity_val_steps: 0
model:
  scheduler:
    num_training_steps: 1
  args:
    vqvae_model_ckpt_path: {vqvae_path.as_posix()}
""".lstrip(),
        encoding="utf-8",
    )

    conf, _ = load_config(str(result_dir), max_steps=123, vqvae_ckpt="last.ckpt")

    assert Path(conf.model.args.vqvae_model_ckpt_path).name == "last.ckpt"
    assert Path(conf.model.args.vqvae_model_ckpt_path).parent == vqvae_path.parent
