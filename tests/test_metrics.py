"""Tests for the scoring functions."""

import torch

from cifar10c_calibration.metrics import accuracy


def test_accuracy_counts_correct_predictions() -> None:
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 0.0], [1.0, 4.0]])
    labels = torch.tensor([0, 1, 1, 1])
    assert accuracy(logits, labels) == 0.75
