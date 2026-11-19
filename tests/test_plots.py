"""Tests for the results figures."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from cifar10c_calibration.plots import (
    plot_metric_vs_severity,
    plot_reliability_diagrams,
    severity_summary,
)

METHODS = ("single", "temperature", "ensemble")


def _fake_results() -> pd.DataFrame:
    """Every method, 2 seeds: clean test, plus fog and snow at severity 1.

    Fog accuracy is 0.6 and 0.8 across the seeds; snow is 0.4 for both.
    """
    rows = []
    for method in METHODS:
        for seed, fog_accuracy in ((0, 0.6), (1, 0.8)):
            where = {"method": method, "seed": seed}
            rows.append(
                {**where, "corruption": "clean", "severity": 0, "accuracy": 0.9}
            )
            rows.append(
                {**where, "corruption": "fog", "severity": 1, "accuracy": fog_accuracy}
            )
            rows.append({**where, "corruption": "snow", "severity": 1, "accuracy": 0.4})
    table = pd.DataFrame(rows)
    return table


def test_severity_summary_averages_seeds_then_corruptions() -> None:
    summary = severity_summary(_fake_results(), "accuracy")

    severity_1 = summary[(summary["method"] == "single") & (summary["severity"] == 1)]

    assert severity_1["median"].item() == pytest.approx(0.55)


def test_severity_summary_has_one_row_per_method_and_severity() -> None:
    summary = severity_summary(_fake_results(), "accuracy")
    assert len(summary) == len(METHODS) * 2


def test_plot_is_saved(tmp_path: Path) -> None:
    path = tmp_path / "figures" / "accuracy.png"

    plot_metric_vs_severity(_fake_results(), "accuracy", path)

    assert path.stat().st_size > 0


def test_reliability_diagrams_are_saved(tmp_path: Path) -> None:
    labels = torch.arange(100) % 10
    method_logits = {method: (torch.randn(100, 10), labels) for method in METHODS}
    path = tmp_path / "figures" / "reliability.png"

    plot_reliability_diagrams(method_logits, "Fake data", path)

    assert path.stat().st_size > 0
