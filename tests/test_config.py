"""Tests for the training configuration."""

import dataclasses
from pathlib import Path

import pytest

from cifar10c_calibration.config import TrainConfig


def test_checkpoint_path_includes_seed() -> None:
    config = TrainConfig(seed=3)
    assert config.checkpoint_path == Path("outputs/checkpoints/resnet18_seed3.pt")


def test_config_is_immutable() -> None:
    config = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]
