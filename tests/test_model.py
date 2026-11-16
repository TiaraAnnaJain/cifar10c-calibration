"""Tests for the CIFAR ResNet-18."""

import torch
from torch import nn

from cifar10c_calibration.models import resnet18_cifar


def test_output_is_one_logit_per_class() -> None:
    model = resnet18_cifar()
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (4, 10)


def test_stem_is_cifar_style() -> None:
    model = resnet18_cifar()
    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)


def test_parameter_count_matches_resnet18() -> None:
    model = resnet18_cifar()
    n_params = sum(p.numel() for p in model.parameters())
    assert 11_000_000 < n_params < 11_300_000
