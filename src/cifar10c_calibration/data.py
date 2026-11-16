"""CIFAR-10 data normalisation, augmentation, train/validation split and loaders"""

from typing import List, Tuple

import torch
from torchvision import transforms


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


def split_indices(n_total: int, val_size: int, seed: int = SPLIT_SEED) -> Tuple[List[int], List[int]]:
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