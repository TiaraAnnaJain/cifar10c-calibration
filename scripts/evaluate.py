"""Score every method on clean test and CIFAR-10-C, and save the results table.

Examples:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --seeds 0 --fast-dev-run
"""

import argparse
from pathlib import Path

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.evaluation import CLEAN, Evaluator
from cifar10c_calibration.prediction import load_predictions

DEFAULT_CONFIG = Path("configs/resnet18_cifar10.toml")
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
RESULTS_PATH = Path("results/results.csv")


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Build the results table.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="use the fast-dev-run logits and keep the results out of results/",
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
    results_path = (
        config.output_dir / "results.csv" if args.fast_dev_run else RESULTS_PATH
    )

    seed_predictions = {
        seed: load_predictions(config.with_overrides(seed=seed).logits_path)
        for seed in args.seeds
    }
    evaluator = Evaluator(seed_predictions)
    for seed, temperature in evaluator.temperatures.items():
        print(f"seed {seed} | temperature {temperature:.3f}")

    results = evaluator.results()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    clean = results[results["corruption"] == CLEAN]
    print(clean.groupby("method")[["accuracy", "ece", "nll", "brier"]].mean())
    print(f"saved {results_path} ({len(results)} rows)")


if __name__ == "__main__":
    main()
