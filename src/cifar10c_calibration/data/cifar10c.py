"""CIFAR-10-C: the CIFAR-10 test images under 15 corruptions at 5 severities."""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data.transforms import eval_transform

URL = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
MD5 = "56bf5dcef84df0e2308c6dcbcbbd8499"
FOLDER = "CIFAR-10-C"

CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
SEVERITIES = (1, 2, 3, 4, 5)


def download_cifar10c(data_dir: Path) -> None:
    """Download CIFAR-10_c and extract it into data_dir/CIFAR-10-C"""
    download_and_extract_archive(
        url=URL, download_root=str(data_dir), md5=MD5, remove_finished=True
    )


class ArrayDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images and labels held in NumPy arrays, transformed like the clean test set."""

    def __init__(
        self, images: npt.NDArray[np.uint8], labels: npt.NDArray[np.integer[Any]]
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = eval_transform()

    def __len__(self) -> int:
        size = len(self.labels)
        return size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.transform(self.images[index])
        label = int(self.labels[index])
        return image, label


class CIFAR10CData:
    """One test loader per (corruption, severity) pair."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.root = config.data_dir / FOLDER
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"{self.root} not found: run scripts/download_cifar10c.py first"
            )

        self.labels = np.load(self.root / "labels.npy")
        self.images_per_severity = len(self.labels) // len(SEVERITIES)

    def loader(self, corruption: str, severity: int) -> DataLoader[Any]:
        """Test images with one corruption at one severity."""
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption {corruption!r}")
        if severity not in SEVERITIES:
            raise ValueError(f"Severity must be 1-5, got {severity!r}")

        start = (severity - 1) * self.images_per_severity
        stop = start + self.images_per_severity
        if self.config.max_images is not None:
            stop = min(stop, start + self.config.max_images)

        all_severities = np.load(self.root / f"{corruption}.npy", mmap_mode="r")
        images = np.array(all_severities[start:stop])
        labels = self.labels[start:stop]

        loader = DataLoader(
            ArrayDataset(images, labels),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader
