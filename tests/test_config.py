"""Tests for the training configuration."""

import dataclasses
import json
from pathlib import Path

import pytest

from cifar10c_calibration.config import FAST_DEV_RUN_IMAGES, TrainConfig

PROJECT_CONFIG = Path(__file__).parents[1] / "configs" / "resnet18_cifar10.toml"


def _write_toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_checkpoint_path_includes_seed() -> None:
    config = TrainConfig(seed=3)
    assert config.checkpoint_path == Path("outputs/checkpoints/resnet18_seed3.pt")


def test_config_is_immutable() -> None:
    config = TrainConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 1  # type: ignore[misc]


def test_from_toml_reads_values_and_keeps_defaults(tmp_path: Path) -> None:
    path = _write_toml(tmp_path, 'epochs = 5\nlr = 0.05\noutput_dir = "runs"\n')

    config = TrainConfig.from_toml(path)

    assert config.epochs == 5
    assert config.lr == 0.05
    assert config.output_dir == Path("runs")  # converted from str
    assert config.batch_size == TrainConfig().batch_size  # not in file: default


def test_from_toml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_toml(tmp_path, "learning_rate = 0.1\n")
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig.from_toml(path)


def test_project_config_file_loads() -> None:
    TrainConfig.from_toml(PROJECT_CONFIG)


def test_with_overrides_skips_none() -> None:
    config = TrainConfig().with_overrides(seed=3, epochs=None)
    assert config.seed == 3
    assert config.epochs == TrainConfig().epochs


def test_to_dict_is_json_serialisable() -> None:
    as_dict = TrainConfig().to_dict()
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert as_dict["data_dir"] == "data"


def test_fast_dev_run_is_tiny_and_saved_separately() -> None:
    config = TrainConfig(epochs=40).fast_dev_run()
    assert config.epochs == 1
    assert config.max_images == FAST_DEV_RUN_IMAGES
    assert config.output_dir == Path("outputs/fast_dev_run")


def test_summary_path_sits_next_to_checkpoint() -> None:
    config = TrainConfig(seed=3)
    assert config.summary_path == Path("outputs/checkpoints/resnet18_seed3.json")


def test_logits_path_includes_seed() -> None:
    config = TrainConfig(seed=3)
    assert config.logits_path == Path("outputs/logits/resnet18_seed3.pt")
