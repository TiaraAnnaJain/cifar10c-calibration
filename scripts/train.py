"""Train one CIFAR-10 ResNet-18 for a given seed and save its checkpoint."""

import argparse
from pathlib import Path

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data import CIFAR10Data
from cifar10c_calibration.engine import Trainer
from cifar10c_calibration.models import resnet18_cifar
from cifar10c_calibration.utils import get_device, set_seed

DEFAULT_CONFIG = Path("configs/resnet18_cifar10.toml")


def parse_args(default_path: Path) -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Train one CIFAR-10 ResNet-18.")
    parser.add_argument("--config", type=Path, default=default_path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="1 epoch on a few images, to test the pipeline",
    )

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args(default_path=DEFAULT_CONFIG)
    config = TrainConfig.from_toml(args.config).with_overrides(
        seed=args.seed,
        epochs=args.epochs,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    if args.fast_dev_run:
        config = config.fast_dev_run()

    set_seed(config.seed)
    device = get_device()
    print(f"seed {config.seed} | device {device} | epochs {config.epochs}")

    data = CIFAR10Data(config)
    trainer = Trainer(model=resnet18_cifar(), config=config, device=device)
    trainer.fit(data.train_loader())

    val_acc = trainer.evaluate(data.val_loader())
    test_acc = trainer.evaluate(data.test_loader())
    print(
        f"done in {trainer.train_minutes:.1f} min | "
        f"val acc {val_acc:.4f} | test acc {test_acc:.4f}"
    )

    trainer.save_checkpoint()
    trainer.save_summary(val_accuracy=val_acc, test_accuracy=test_acc)
    print(f"saved {config.checkpoint_path} and {config.summary_path.name}")


if __name__ == "__main__":
    main()
