"""Run a trained model on every evaluation set and save its logits."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data import CORRUPTIONS, SEVERITIES, CIFAR10CData, CIFAR10Data
from cifar10c_calibration.engine import predict_logits

Predictions = dict[str, tuple[torch.Tensor, torch.Tensor]]


def corrupted_set_name(corruption: str, severity: int) -> str:
    """Name of one CIFAR-10-C set."""
    name = f"{corruption}_s{severity}"
    return name


def evaluation_sets(config: TrainConfig) -> Iterator[tuple[str, DataLoader[Any]]]:
    """Validation, clean test, then all 75 corrupted sets, loaded one at a time."""
    clean = CIFAR10Data(config)
    yield "val", clean.val_loader()
    yield "test", clean.test_loader()

    corrupted = CIFAR10CData(config)
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            name = corrupted_set_name(corruption=corruption, severity=severity)
            loader = corrupted.loader(corruption=corruption, severity=severity)
            yield name, loader


def predict_all(
    model: nn.Module, sets: Iterable[tuple[str, DataLoader[Any]]], device: torch.device
) -> Predictions:
    """Logits and labels for every set in "sets"."""
    predictions: Predictions = {}
    for name, loader in sets:
        predictions[name] = predict_logits(model, loader, device)
        print(f" {name}")
    return predictions


def save_predictions(predictions: Predictions, path: Path) -> None:
    """Save predictions to path, creating its folder if necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, path)


def load_predictions(path: Path) -> Predictions:
    """Predictions saved by save_predictions."""
    predictions: Predictions = torch.load(path)
    return predictions
