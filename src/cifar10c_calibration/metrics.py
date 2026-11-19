"""Scores computed from a model's logits and the true labels"""

import torch


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of images whose highest logit is the true class."""
    predictions = logits.argmax(dim=1)
    fraction_correct = (predictions == labels).float().mean().item()
    return fraction_correct
