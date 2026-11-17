"""Splitting the CIFAR-10 training images into train and validation sets."""

import torch

SPLIT_SEED = 0


def split_indices(
    n_total: int, val_size: int, seed: int = SPLIT_SEED
) -> tuple[list[int], list[int]]:
    """Shuffle ``range(n_total)`` and cut it into train and validation indices.

    Returns:
        (train_indices, val_indices), with n_total - val_size and val_size
        elements respectively.
    """
    generator = torch.Generator().manual_seed(seed)
    permutation: list[int] = torch.randperm(n_total, generator=generator).tolist()

    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices
