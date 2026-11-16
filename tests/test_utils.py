"""Tests for utils module"""

import torch

from cifar10c_calibration.utils import get_device, set_seed


def test_same_seed_gives_same_numbers() -> None:
    set_seed(0)
    first = torch.rand(5)
    set_seed(0)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_get_device_returns_a_known_device() -> None:
    device = get_device()
    assert device.type in {"cuda", "mps", "cpu"}
