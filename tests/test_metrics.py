"""Tests for the scoring functions."""

import math

import pytest
import torch

from cifar10c_calibration.calibration.metrics import (
    accuracy,
    brier,
    ece,
    nll,
    reliability_bins,
)


def _confident_logits(confidence: float, labels: torch.Tensor) -> torch.Tensor:
    """2-class logits that always predict class 0 with the given confidence."""
    probs = torch.tensor([confidence, 1 - confidence]).repeat(len(labels), 1)
    logits = probs.log()
    return logits


def test_accuracy_counts_correct_predictions() -> None:
    logits = torch.tensor([[2.0, 1.0], [0.0, 3.0], [5.0, 0.0], [1.0, 4.0]])
    labels = torch.tensor([0, 1, 1, 1])
    assert accuracy(logits, labels) == 0.75


def test_nll_of_uniform_prediction_is_log_num_classes() -> None:
    logits = torch.zeros(8, 10)
    labels = torch.arange(8)
    assert nll(logits, labels) == pytest.approx(math.log(10))


def test_brier_is_zero_when_certain_and_right() -> None:
    labels = torch.tensor([0, 1, 1])
    logits = torch.nn.functional.one_hot(labels, num_classes=2).float() * 100
    assert brier(logits, labels) == pytest.approx(0.0)


def test_brier_of_coin_flip_is_half() -> None:
    logits = torch.zeros(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    assert brier(logits, labels) == pytest.approx(0.5)


def test_ece_is_gap_between_confidence_and_accuracy() -> None:
    labels = torch.tensor([0, 1] * 50)
    logits = _confident_logits(0.8, labels)
    assert ece(logits, labels) == pytest.approx(0.3)


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    labels = torch.tensor([0, 0, 0, 1])
    logits = _confident_logits(0.75, labels)
    assert ece(logits, labels) == pytest.approx(0.0, abs=1e-6)


def test_reliability_bins_count_every_image_once() -> None:
    logits = torch.randn(500, 10, generator=torch.Generator().manual_seed(0))
    labels = torch.randint(0, 10, (500,), generator=torch.Generator().manual_seed(1))
    bins = reliability_bins(logits, labels, num_bins=15)
    assert bins.num_images.sum() == 500
    assert len(bins.confidence) == len(bins.accuracy) == 15
