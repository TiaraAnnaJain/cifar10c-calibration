"""Training and inference loops."""

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class EpochStats(NamedTuple):
    """Average loss and accuracy over one pass through a dataset."""

    loss: float
    accuracy: float


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochStats:
    """Run one full pass, updating the model's weights.

    Returns:
        The training loss and accuracy averaged over every image in the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_images += batch_size

    return EpochStats(total_loss / total_images, total_correct / total_images)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model without updating the model's weights."""
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)
