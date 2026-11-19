"""Download CIFAR-10-C into the data folder."""

import argparse
from pathlib import Path

from cifar10c_calibration.data import download_cifar10c


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Download CIFAR-10-C.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    download_cifar10c(args.data_dir)


if __name__ == "__main__":
    main()
