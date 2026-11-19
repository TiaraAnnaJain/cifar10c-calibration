"""Tests for running a model over evaluation sets and saving the logits."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10c_calibration.prediction import (
    corrupted_set_name,
    load_predictions,
    predict_all,
    save_predictions,
)

DEVICE = torch.device("cpu")


def _toy_loader(n_images: int) -> DataLoader[tuple[torch.Tensor, ...]]:
    images = torch.randn(n_images, 2)
    labels = torch.randint(0, 2, (n_images,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    return loader


def test_corrupted_set_name() -> None:
    assert corrupted_set_name("fog", 3) == "fog_s3"


def test_predict_all_gives_logits_and_labels_for_every_set() -> None:
    sets = [("val", _toy_loader(6)), ("fog_s1", _toy_loader(10))]

    predictions = predict_all(nn.Linear(2, 2), sets, DEVICE)

    assert list(predictions) == ["val", "fog_s1"]
    logits, labels = predictions["fog_s1"]
    assert logits.shape == (10, 2)
    assert labels.shape == (10,)


def test_predictions_round_trip(tmp_path: Path) -> None:
    predictions = predict_all(nn.Linear(2, 2), [("val", _toy_loader(6))], DEVICE)
    path = tmp_path / "logits" / "seed0.pt"

    save_predictions(predictions, path)
    loaded = load_predictions(path)

    assert torch.equal(loaded["val"][0], predictions["val"][0])
    assert torch.equal(loaded["val"][1], predictions["val"][1])
