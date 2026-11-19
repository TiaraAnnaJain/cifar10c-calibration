"""Save each trained model's logits on validation, clean test and CIFAR-10-C."""

import argparse
from pathlib import Path

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.models import load_resnet18_cifar
from cifar10c_calibration.prediction import (
    evaluation_sets,
    predict_all,
    save_predictions,
)
from cifar10c_calibration.utils import get_device

DEFAULT_CONFIG = Path("configs/resnet18_cifar10.toml")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description="Save logits for each trained model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="a few images per set, using the fast-dev-run checkpoints",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    base_config = TrainConfig.from_toml(args.config).with_overrides(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    if args.fast_dev_run:
        base_config = base_config.fast_dev_run()
    device = get_device()

    for seed in args.seeds:
        config = base_config.with_overrides(seed=seed)
        print(f"seed {seed} | device {device}")

        model = load_resnet18_cifar(config.checkpoint_path, device)
        predictions = predict_all(model, evaluation_sets(config), device)
        save_predictions(predictions, config.logits_path)
        print(f"saved {config.logits_path}")


if __name__ == "__main__":
    main()
