"""Tests for the CIFAR-10 split and transforms (no download needed)."""

import numpy as np
import torch
from PIL import Image

from cifar10c_calibration.data import eval_transform, split_indices, train_transform


def test_split_sizes() -> None:
    train_idx, val_idx = split_indices(50_000, 5_000)
    assert len(train_idx) == 45_000
    assert len(val_idx) == 5_000


def test_split_has_no_overlap_and_covers_everything() -> None:
    train_idx, val_idx = split_indices(50_000, 5_000)
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx) | set(val_idx) == set(range(50_000))


def test_split_is_reproducible() -> None:
    assert split_indices(1_000, 100) == split_indices(1_000, 100)


def test_different_seeds_give_different_splits() -> None:
    assert split_indices(1_000, 100, seed=0) != split_indices(1_000, 100, seed=1)


def _random_image() -> Image.Image:
    pixels = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def test_transforms_output_normalised_tensor_of_right_shape() -> None:
    image = _random_image()
    for transform in (train_transform(), eval_transform()):
        tensor = transform(image)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32


def test_eval_transform_is_deterministic() -> None:
    image = _random_image()
    assert torch.equal(eval_transform()(image), eval_transform()(image))