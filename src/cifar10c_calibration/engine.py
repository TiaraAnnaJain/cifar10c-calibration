"""Training and inference loops."""

import dataclasses
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from cifar10c_calibration.config import TrainConfig


class EpochStats(NamedTuple):
    """Average loss and accuracy over one pass through a dataset."""

    loss: float
    accuracy: float


class Trainer:
    """Owns the model, optimiser and learning-rate schedule for one training run."""

    def __init__(
        self, model: nn.Module, config: TrainConfig, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs
        )

    def fit(self, train_loader: DataLoader[Any]) -> list[EpochStats]:
        """Train for ``config.epochs`` epochs.

        Returns:
            The training loss and accuracy of every epoch, in order.
        """
        history = []
        for epoch in range(1, self.config.epochs + 1):
            stats = self._train_one_epoch(train_loader)
            self.scheduler.step()
            history.append(stats)
            print(
                f"epoch {epoch:>3}/{self.config.epochs}  "
                f"loss {stats.loss:.4f}  acc {stats.accuracy:.4f}"
            )
        return history

    def save_checkpoint(self) -> None:
        """Save the weights and the config that produced them."""
        path = self.config.checkpoint_path
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataclasses.asdict(self.config).items()
        }
        torch.save({"model_state": self.model.state_dict(), "config": config}, path)

    def _train_one_epoch(self, loader: DataLoader[Any]) -> EpochStats:
        """Run one full pass over ``loader``, updating the model's weights."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            self.optimizer.step()

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
    """Run the model without updating the model's weights.

    Returns:
        ``(logits, labels)`` on the CPU.
    """
    model.eval()
    all_logits = []
    all_labels = []

    for images, labels in loader:
        logits = model(images.to(device))
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)
