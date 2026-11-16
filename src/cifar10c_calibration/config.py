"""Training configuration"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Settings for training one CIFAR-10 ResNet-18."""

    seed: int = 0
    epochs: int = 40
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    val_size: int = 5_000
    num_workers: int = 2

    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def checkpoint_path(self) -> Path:
        """Where this seed's trained weights are saved"""
        return self.output_dir / "checkpoints" / f"resnet18_seed{self.seed}.pt"
