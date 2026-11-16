"""Tests for the training and inference loops, on a tiny synthetic problem."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10c_calibration.engine import predict_logits, train_one_epoch

DEVICE = torch.device("cpu")


def _toy_loader(shuffle: bool = True) -> DataLoader[tuple[torch.Tensor, ...]]:
    """64 points in 2D; the class is 1 when the first coordinate is positive."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 2, generator=generator)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=shuffle)


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    loader = _toy_loader()

    first = train_one_epoch(model, loader, optimizer, DEVICE)
    for _ in range(20):
        last = train_one_epoch(model, loader, optimizer, DEVICE)

    assert last.loss < first.loss
    assert last.accuracy > 0.9


def test_predict_logits_shapes_and_order() -> None:
    model = nn.Linear(2, 2)
    loader = _toy_loader(shuffle=False)

    logits, labels = predict_logits(model, loader, DEVICE)

    assert logits.shape == (64, 2)
    assert labels.shape == (64,)
    assert torch.equal(labels, loader.dataset.tensors[1])  # type: ignore[attr-defined]


def test_predict_logits_does_not_change_weights() -> None:
    model = nn.Linear(2, 2)
    before = [p.clone() for p in model.parameters()]

    predict_logits(model, _toy_loader(), DEVICE)

    assert all(
        torch.equal(b, a) for b, a in zip(before, model.parameters(), strict=True)
    )
