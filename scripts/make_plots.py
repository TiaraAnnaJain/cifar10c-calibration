"""Draw every metric against corruption severity from the results table.

Examples:
    uv run python scripts/make_plots.py
    uv run python scripts/make_plots.py --fast-dev-run
"""

import argparse
from pathlib import Path

import pandas as pd

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.plots import METRIC_LABELS, plot_metric_vs_severity

DEFAULT_CONFIG = Path("configs/resnet18_cifar10.toml")
RESULTS_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Plot the results table.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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

    results = pd.read_csv(results_dir / "results.csv")
    for metric in METRIC_LABELS:
        path = results_dir / "figures" / f"{metric}_vs_severity.png"
        plot_metric_vs_severity(results, metric, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
