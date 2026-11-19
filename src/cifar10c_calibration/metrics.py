"""Scores computed from a model's logits and the true labels"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

NUM_BINS = 15


class ReliabilityBins(NamedTuple):
    """Average confidence, accuracy and number of images in confidence bin."""

    confidence: torch.Tensor
    accuracy: torch.Tensor
    num_images: torch.Tensor


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of images whose highest logit is the true class."""
    predictions = logits.argmax(dim=1)
    fraction_correct = (predictions == labels).float().mean().item()
    return fraction_correct


def nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Average negative log likelihood given to the true class."""
    loss = F.cross_entropy(logits, labels).item()
    return loss


def brier(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Average squared distance between the probabilities and the one-hot label."""
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    score = ((probs - one_hot) ** 2).sum(dim=1).mean().item()
    return score


def reliability_bins(
    logits: torch.Tensor, labels: torch.Tensor, num_bins: int = NUM_BINS
) -> ReliabilityBins:
    """Group images into equal-width confidence bins and summarise each bin."""
    confidence, predictions = logits.softmax(dim=1).max(dim=1)
    correct = (predictions == labels).float()

    bin_ids = (confidence * num_bins).ceil().long().clamp(1, num_bins) - 1
    count = torch.bincount(bin_ids, minlength=num_bins)
    confidence_sum = torch.bincount(bin_ids, weights=confidence, minlength=num_bins)
    correct_sum = torch.bincount(bin_ids, weights=correct, minlength=num_bins)

    non_zero_count = count.clamp(min=1)
    bins = ReliabilityBins(
        confidence=confidence_sum / non_zero_count,
        accuracy=correct_sum / non_zero_count,
        num_images=count,
    )
    return bins


def ece(logits: torch.Tensor, labels: torch.Tensor, num_bins: int = NUM_BINS) -> float:
    """Expected calibration error: the confidence-accuracy gap, averaged over bins."""
    bins = reliability_bins(logits, labels, num_bins)
    weights = bins.num_images / bins.num_images.sum()
    gaps = (bins.accuracy - bins.confidence).abs()
    error = (weights * gaps).sum().item()
    return error
