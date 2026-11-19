"""CIFAR-10 split into train (45k), validation (5k) and test (10k) loaders."""

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data.splits import split_indices
from cifar10c_calibration.data.transforms import eval_transform, train_transform


class CIFAR10Data:
    """CIFAR-10 split into train, validation and test sets."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config

        train_full = self._load(train=True, augment=True)
        val_full = self._load(train=True, augment=False)
        test_full = self._load(train=False, augment=False)

        train_idx, val_idx = split_indices(len(train_full), config.val_size)
        test_idx = list(range(len(test_full)))

        if config.max_images is not None:
            n = config.max_images
            train_idx, val_idx, test_idx = train_idx[:n], val_idx[:n], test_idx[:n]

        self.train_set = Subset(train_full, train_idx)
        self.val_set = Subset(val_full, val_idx)
        self.test_set = Subset(test_full, test_idx)

    def train_loader(self) -> DataLoader[Any]:
        """Augmented training images, reshuffled every epoch."""
        shuffle_generator = torch.Generator().manual_seed(self.config.seed)
        loader = self._batch(self.train_set, shuffle=True, generator=shuffle_generator)
        return loader

    def val_loader(self) -> DataLoader[Any]:
        """Validation images, used only to fit the temperature."""
        loader = self._batch(self.val_set, shuffle=False)
        return loader

    def test_loader(self) -> DataLoader[Any]:
        """Clean test images, for the final score."""
        loader = self._batch(self.test_set, shuffle=False)
        return loader

    def _load(self, *, train: bool, augment: bool) -> datasets.CIFAR10:
        """Read CIFAR-10 from disk, downloading it the first time."""
        transform = train_transform() if augment else eval_transform()
        dataset = datasets.CIFAR10(
            root=self.config.data_dir, train=train, transform=transform, download=True
        )
        return dataset

    def _batch(
        self,
        dataset: Dataset[Any],
        *,
        shuffle: bool,
        generator: torch.Generator | None = None,
    ) -> DataLoader[Any]:
        """Wrap a dataset in a loader that yields batches."""
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
        return loader
