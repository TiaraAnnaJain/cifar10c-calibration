"""Tests for combining ensemble members."""

import torch

from cifar10c_calibration.calibration import ensemble_logits


def test_single_member_gives_same_probabilities() -> None:
    logits = torch.randn(6, 10)
    combined = ensemble_logits([logits])
    assert torch.allclose(combined.softmax(dim=1), logits.softmax(dim=1))


def test_probabilities_are_averaged() -> None:
    confident = torch.tensor([[0.9, 0.1]]).log()
    unsure = torch.tensor([[0.5, 0.5]]).log()

    combined = ensemble_logits([confident, unsure])

    assert torch.allclose(combined.softmax(dim=1), torch.tensor([[0.7, 0.3]]))


def test_output_has_one_row_per_image() -> None:
    members = [torch.randn(6, 10) for _ in range(5)]
    combined = ensemble_logits(members)
    assert combined.shape == (6, 10)
