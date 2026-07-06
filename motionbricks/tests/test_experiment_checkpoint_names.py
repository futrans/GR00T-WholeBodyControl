from argparse import Namespace

from motionbricks.exp_setup.experiment import get_path_dir


def test_get_path_dir_keeps_official_default_checkpoint_names():
    ckpt_info = get_path_dir("default")

    assert ckpt_info["vqvae_ckpt"] == "model-step=2000000.ckpt"
    assert ckpt_info["pose_model_ckpt"] == "model-step=2000000.ckpt"
    assert ckpt_info["root_model_ckpt"] == "model-step=2000000.ckpt"


def test_get_path_dir_accepts_explicit_checkpoint_name_overrides():
    args = Namespace(
        vqvae_ckpt="vqvae-last.ckpt",
        pose_ckpt="pose-last.ckpt",
        root_ckpt="root-last.ckpt",
    )

    ckpt_info = get_path_dir("default", args=args)

    assert ckpt_info["vqvae_ckpt"] == "vqvae-last.ckpt"
    assert ckpt_info["pose_model_ckpt"] == "pose-last.ckpt"
    assert ckpt_info["root_model_ckpt"] == "root-last.ckpt"
