"""Figures built from the results table."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHOD_LABELS = {
    "single": "Single model",
    "temperature": "Temperature scaling",
    "ensemble": "Deep ensemble",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "ece": "ECE",
    "nll": "NLL",
    "brier": "Brier score",
}


def severity_summary(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Median and quartiles of "metric" across corruptions, per method and severity.

    Seeds are averaged first, so each corruption counts once.
    """
    per_corruption = results.groupby(
        ["method", "corruption", "severity"], as_index=False
    )[metric].mean()
    grouped = per_corruption.groupby(["method", "severity"])[metric]
    summary = pd.DataFrame(
        {
            "median": grouped.median(),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        }
    ).reset_index()
    return summary


def plot_metric_vs_severity(results: pd.DataFrame, metric: str, path: Path) -> None:
    """Line per method: median over corruptions, shaded between the quartiles."""
    summary = severity_summary(results, metric)
    figure, axes = plt.subplots(figsize=(6, 4))

    for method, label in METHOD_LABELS.items():
        rows = summary[summary["method"] == method]
        (line,) = axes.plot(rows["severity"], rows["median"], marker="o", label=label)
        axes.fill_between(
            rows["severity"],
            rows["q25"],
            rows["q75"],
            color=line.get_color(),
            alpha=0.2,
        )

    axes.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"])
    axes.set_xlabel("Corruption severity")
    axes.set_ylabel(METRIC_LABELS[metric])
    axes.set_title(f"{METRIC_LABELS[metric]} on CIFAR-10-C")
    axes.grid(alpha=0.3)
    axes.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
