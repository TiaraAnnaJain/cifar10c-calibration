"""Training configuration"""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

FAST_DEV_RUN_IMAGES = 256


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
    max_images: int | None = None

    data_dir: Path = field(default_factory=lambda: Path("data"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_toml(cls, path: Path) -> Self:
        """Load a config from a TOML file. Missing keys keep their default values."""
        with path.open("rb") as file:
            values = tomllib.load(file)

        unknown = values.keys() - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")

        for key in ("data_dir", "output_dir"):
            if key in values:
                values[key] = Path(values[key])
        return cls(**values)

    def with_overrides(self, **overrides: Any) -> Self:
        """Returns a copy with the given fields replaced; "None" values are skipped"""
        given = {key: value for key, value in overrides.items() if value is not None}
        return dataclasses.replace(self, **given)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict of every field. Paths become strings"""
        dictionary = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in dataclasses.asdict(self).items()
        }
        return dictionary

    def fast_dev_run(self) -> Self:
        """1 epoch copy for testing the pipeline, saved in its own folder"""
        config = dataclasses.replace(
            self,
            epochs=1,
            num_workers=0,
            max_images=FAST_DEV_RUN_IMAGES,
            output_dir=self.output_dir / "fast_dev_run",
        )
        return config

    @property
    def checkpoint_path(self) -> Path:
        """Where this seed's trained weights are saved"""
        path = self.output_dir / "checkpoints" / f"resnet18_seed{self.seed}.pt"
        return path

    @property
    def summary_path(self) -> Path:
        """Where this seed's readable training summary is saved."""
        path = self.checkpoint_path.with_suffix(".json")
        return path

    @property
    def logits_path(self) -> Path:
        """Where this seed's logits on every evaluation set are saved."""
        path = self.output_dir / "logits" / f"resnet18_seed{self.seed}.pt"
        return path
