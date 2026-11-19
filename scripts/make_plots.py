"""Draw the metric-vs-severity plots and the reliability diagrams."""

import argparse
from pathlib import Path

import pandas as pd

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.data import CORRUPTIONS
from cifar10c_calibration.evaluation import Evaluator
from cifar10c_calibration.plots import (
    METRIC_LABELS,
    plot_metric_vs_severity,
    plot_reliability_diagrams,
)
from cifar10c_calibration.prediction import corrupted_set_name, load_predictions

DEFAULT_CONFIG = Path("configs/resnet18_cifar10.toml")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path("results")

RELIABILITY_DIAGRAMS = {
    "reliability_clean": ("Clean test set", ["test"]),
    "reliability_severity5": (
        "CIFAR-10-C, severity 5 (all 15 corruptions)",
        [corrupted_set_name(corruption, 5) for corruption in CORRUPTIONS],
    ),
}


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Plot the results table.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="plot the fast-dev-run results and keep the figures out of results/",
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config = TrainConfig.from_toml(args.config).with_overrides(
        output_dir=args.output_dir
    )
    if args.fast_dev_run:
        config = config.fast_dev_run()
    results_dir = config.output_dir if args.fast_dev_run else RESULTS_DIR
    figures_dir = results_dir / "figures"

    results = pd.read_csv(results_dir / "results.csv")
    for metric in METRIC_LABELS:
        path = figures_dir / f"{metric}_vs_severity.png"
        plot_metric_vs_severity(results, metric, path)
        print(f"saved {path}")

    seed_predictions = {
        seed: load_predictions(config.with_overrides(seed=seed).logits_path)
        for seed in args.seeds
    }
    evaluator = Evaluator(seed_predictions)
    for file_name, (title, names) in RELIABILITY_DIAGRAMS.items():
        path = figures_dir / f"{file_name}.png"
        method_logits = evaluator.pooled_logits(names, seed=args.seeds[0])
        plot_reliability_diagrams(method_logits, title, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
