"""Tests for the CIFAR-10-C loader, on a tiny fake copy of the dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data import (
    CORRUPTIONS,
    SEVERITIES,
    CIFAR10CData,
    eval_transform,
)

IMAGES_PER_SEVERITY = 4


def _fake_config(tmp_path: Path, max_images: int | None = None) -> TrainConfig:
    """A config pointing at a fake CIFAR-10-C with one corruption, "fog".

    Every image at severity s has all its pixels set to 40 * s.
    """
    root = tmp_path / "CIFAR-10-C"
    root.mkdir()
    n_total = IMAGES_PER_SEVERITY * len(SEVERITIES)

    labels = np.arange(n_total) % 10
    pixel_values = np.repeat(
        np.array(SEVERITIES, dtype=np.uint8) * 40, IMAGES_PER_SEVERITY
    )
    images = np.broadcast_to(pixel_values[:, None, None, None], (n_total, 32, 32, 3))
    np.save(root / "labels.npy", labels)
    np.save(root / "fog.npy", images)

    config = TrainConfig(data_dir=tmp_path, num_workers=0, max_images=max_images)
    return config


def _load_all(loader: DataLoader[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Every image and label in ``loader``, concatenated."""
    batches = list(loader)
    images = torch.cat([batch[0] for batch in batches])
    labels = torch.cat([batch[1] for batch in batches])
    return images, labels


def test_there_are_15_corruptions() -> None:
    assert len(set(CORRUPTIONS)) == 15


def test_loader_returns_only_the_requested_severity(tmp_path: Path) -> None:
    data = CIFAR10CData(_fake_config(tmp_path))

    images, labels = _load_all(data.loader("fog", severity=3))

    expected_labels = torch.arange(8, 12) % 10
    expected_image = eval_transform()(np.full((32, 32, 3), 120, dtype=np.uint8))
    assert torch.equal(labels, expected_labels)
    assert all(torch.equal(image, expected_image) for image in images)


def test_max_images_limits_each_loader(tmp_path: Path) -> None:
    data = CIFAR10CData(_fake_config(tmp_path, max_images=2))

    images, _ = _load_all(data.loader("fog", severity=1))

    assert len(images) == 2


def test_unknown_corruption_or_severity_is_rejected(tmp_path: Path) -> None:
    data = CIFAR10CData(_fake_config(tmp_path))
    with pytest.raises(ValueError, match="corruption"):
        data.loader("rain", severity=1)
    with pytest.raises(ValueError, match="Severity"):
        data.loader("fog", severity=6)


def test_missing_dataset_gives_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_cifar10c"):
        CIFAR10CData(TrainConfig(data_dir=tmp_path))
