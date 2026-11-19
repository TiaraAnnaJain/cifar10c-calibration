"""Figures built from the results table."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.figure import Figure

from cifar10c_calibration.calibration.metrics import NUM_BINS, ece, reliability_bins

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
            edgecolor="black",
            label="Accuracy",
        )
        axes.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
        axes.set_title(f"{label} (ECE {ece(logits, labels):.3f})")
        axes.set_xlabel("Confidence")
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)

    all_axes[0].set_ylabel("Accuracy")
    all_axes[0].legend(loc="upper left")
    figure.suptitle(title)
    _save(figure, path)


def _save(figure: Figure, path: Path) -> None:
    """Save "figure" as a PNG at "path" and free its memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
