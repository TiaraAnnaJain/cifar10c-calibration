"""Figures built from the results table."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cifar10c_calibration.calibration.metrics import NUM_BINS, ece, reliability_bins

METHOD_LABELS = {
    "single": "Single model",
    "temperature": "Temperature scaling",
    "ensemble": "Deep ensemble",
}
METHOD_COLORS = {
    "single": "#e0709b",
    "temperature": "#5b3fb8",
    "ensemble": "#17804d",
}
DIAGONAL_COLOR = "#8a8a86"
GRID_COLOR = "#e4e4e0"
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
        color = METHOD_COLORS[method]
        axes.plot(
            rows["severity"],
            rows["median"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            label=label,
        )
        axes.fill_between(
            rows["severity"], rows["q25"], rows["q75"], color=color, alpha=0.15
        )

    axes.set_xticks(range(6), ["clean", "1", "2", "3", "4", "5"])
    axes.set_xlabel("Corruption severity")
    axes.set_ylabel(METRIC_LABELS[metric])
    axes.set_title(f"{METRIC_LABELS[metric]} on CIFAR-10-C")
    _style(axes)
    axes.legend(frameon=False)

    _save(figure, path)


def plot_reliability_diagrams(
    method_logits: dict[str, tuple[torch.Tensor, torch.Tensor]], title: str, path: Path
) -> None:
    """One panel per method: accuracy in each confidence bin against the diagonal."""
    figure, all_axes = plt.subplots(
        1, len(METHOD_LABELS), figsize=(12, 4), sharey=True, layout="constrained"
    )
    bin_centres = (torch.arange(NUM_BINS) + 0.5) / NUM_BINS

    for axes, (method, label) in zip(all_axes, METHOD_LABELS.items(), strict=True):
        logits, labels = method_logits[method]
        bins = reliability_bins(logits, labels)
        has_images = bins.num_images > 0

        axes.bar(
            bin_centres[has_images].numpy(),
            bins.accuracy[has_images].numpy(),
            width=1 / NUM_BINS,
            color=METHOD_COLORS[method],
            edgecolor="white",  # thin gap between neighbouring bars
            linewidth=1,
            label="Accuracy",
        )
        axes.plot(
            [0, 1], [0, 1], "--", color=DIAGONAL_COLOR, label="Perfect calibration"
        )
        axes.set_title(f"{label} (ECE {ece(logits, labels):.3f})")
        axes.set_xlabel("Confidence")
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        _style(axes)
        axes.legend(loc="upper left", frameon=False)

    all_axes[0].set_ylabel("Accuracy")
    figure.suptitle(title)
    _save(figure, path)


def _style(axes: Axes) -> None:
    """Light grid behind the data, and no top or right border."""
    axes.grid(color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)


def _save(figure: Figure, path: Path) -> None:
    """Save "figure" as a PNG at "path" and free its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
