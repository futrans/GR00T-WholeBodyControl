from motionbricks.training_run_artifacts import build_trainer_artifacts


class DummyCSVLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyTensorBoardLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyWandbLogger:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyCheckpoint:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_tensorboard_csv_logger_mode_writes_summary_to_stable_run_subdir(tmp_path):
    run_dir = tmp_path / "vqvae"

    enable_checkpointing, logger, callbacks = build_trainer_artifacts(
        run_dir=str(run_dir),
        save_every_n_steps=500,
        csv_logger_cls=DummyCSVLogger,
        tensorboard_logger_cls=DummyTensorBoardLogger,
        checkpoint_cls=DummyCheckpoint,
        logger_mode="tensorboard_csv",
    )

    assert enable_checkpointing is True
    assert len(logger) == 2
    assert isinstance(logger[0], DummyCSVLogger)
    assert isinstance(logger[1], DummyTensorBoardLogger)
    assert logger[0].kwargs["save_dir"] == str(run_dir)
    assert logger[0].kwargs["name"] == "logs"
    assert logger[0].kwargs["version"] == ""
    assert logger[1].kwargs["save_dir"] == str(run_dir / "tensorboard")
    assert logger[1].kwargs["name"] == ""
    assert logger[1].kwargs["version"] == ""
    assert len(callbacks) == 2


def test_wandb_mode_still_requires_wandb_logger_cls(tmp_path):
    try:
        build_trainer_artifacts(
            run_dir=str(tmp_path / "run"),
            save_every_n_steps=500,
            csv_logger_cls=DummyCSVLogger,
            tensorboard_logger_cls=DummyTensorBoardLogger,
            checkpoint_cls=DummyCheckpoint,
            logger_mode="wandb",
        )
    except ValueError as exc:
        assert "wandb logger mode requires wandb_logger_cls" in str(exc)
    else:
        raise AssertionError("wandb mode without wandb_logger_cls should fail")
