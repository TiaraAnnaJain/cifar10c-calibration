"""CIFAR-10 data normalisation, augmentation, train/validation split and loaders"""

from typing import Any, NamedTuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from cifar10c_calibration.config import TrainConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

SPLIT_SEED = 0


def train_transform() -> transforms.Compose:
    """Augment and normalise a training image."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """Normalise an evaluation image without augmentation."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def split_indices(
    n_total: int, val_size: int, seed: int = SPLIT_SEED
) -> tuple[list[int], list[int]]:
    """Randomly split n_total into disjoint train and validation indices.

    Args:
        n_total: Total number of images to split.
        val_size: Number of images to put in the validation set.
        seed: Seed for the shuffle. Defaults to ``SPLIT_SEED`` so every model
            uses the same split; kept separate from the model seed so no
            ensemble member ever trains on the validation images.

    Returns:
        (train_indices, val_indices), with n_total - val_size and
        val_size elements respectively.
    """
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator).tolist()
    return permutation[val_size:], permutation[:val_size]


class CIFAR10Loaders(NamedTuple):
    """CIFAR-1 data loaders."""

    train: DataLoader[Any]
    val: DataLoader[Any]
    test: DataLoader[Any]


def get_cifar10_loaders(config: TrainConfig) -> CIFAR10Loaders:
    """Download CIFAR-10 and build train/val/test loaders."""
    root = config.data_dir
    train_full = datasets.CIFAR10(
        root=root, train=True, transform=train_transform(), download=True
    )
    val_full = datasets.CIFAR10(
        root=root, train=True, transform=eval_transform(), download=True
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, transform=eval_transform(), download=True
    )

    train_idx, val_idx = split_indices(len(train_full), config.val_size)

    shuffle_generator = torch.Generator().manual_seed(config.seed)

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(val_full, val_idx)

    train_loader = _make_loader(
        train_subset, config, shuffle=True, generator=shuffle_generator
    )
    val_loader = _make_loader(val_subset, config, shuffle=False)
    test_loader = _make_loader(test_set, config, shuffle=False)

    return CIFAR10Loaders(train=train_loader, val=val_loader, test=test_loader)


def _make_loader(
    dataset: Dataset[Any],
    config: TrainConfig,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    """Wraps a dataset in a DataLoader."""
    return DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
