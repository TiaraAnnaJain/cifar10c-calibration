"""Tests for the CIFAR ResNet-18."""

from pathlib import Path

import torch
from torch import nn

from cifar10c_calibration.models import load_resnet18_cifar, resnet18_cifar


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


def test_load_resnet18_cifar_restores_weights(tmp_path: Path) -> None:
    model = resnet18_cifar()
    path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": {}}, path)

    loaded = load_resnet18_cifar(path, torch.device("cpu"))

    weights = "conv1.weight"
    assert torch.equal(loaded.state_dict()[weights], model.state_dict()[weights])
    assert not loaded.training  # ready for inference
