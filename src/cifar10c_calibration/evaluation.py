"""Score the single, temperature scaled and ensemble methods on every test set."""

from typing import Any

import pandas as pd
import torch

from cifar10c_calibration.calibration import (
    TemperatureScaler,
    accuracy,
    brier,
    ece,
    ensemble_logits,
    nll,
)
from cifar10c_calibration.data import CORRUPTIONS, SEVERITIES
from cifar10c_calibration.prediction import Predictions, corrupted_set_name

CLEAN = "clean"


def scored_sets() -> list[tuple[str, str, int]]:
    """(set name, corruption, severity) for clean test, then all 75 corrupted sets."""
    sets = [("test", CLEAN, 0)]
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            name = corrupted_set_name(corruption, severity)
            sets.append((name, corruption, severity))
    return sets


def score(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Every metric for one set of logits."""
    scores = {
        "accuracy": accuracy(logits, labels),
        "ece": ece(logits, labels),
        "nll": nll(logits, labels),
        "brier": brier(logits, labels),
    }
    return scores


class Evaluator:
    """Scores each seed alone, each seed temperature-scaled, and the seed ensemble."""

    def __init__(self, seed_predictions: dict[int, Predictions]) -> None:
        self.seed_predictions = seed_predictions
        self.scalers = {
            seed: self._fit_scaler(predictions)
            for seed, predictions in seed_predictions.items()
        }

    @property
    def temperatures(self) -> dict[int, float]:
        """The temperature fitted on each seed's validation logits."""
        temperatures = {
            seed: scaler.temperature for seed, scaler in self.scalers.items()
        }
        return temperatures

    def results(self) -> pd.DataFrame:
        """One row per method, seed, corruption and severity."""
        rows = []
        for name, corruption, severity in scored_sets():
            rows.extend(self._score_set(name, corruption, severity))
        table = pd.DataFrame(rows)
        table["seed"] = table["seed"].astype("Int64")
        return table

    def pooled_logits(
        self, names: list[str], seed: int
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Each method's logits and labels on the sets "names", joined together."""
        single, scaled, combined, all_labels = [], [], [], []
        for name in names:
            logits, labels = self.seed_predictions[seed][name]
            members = [
                predictions[name][0] for predictions in self.seed_predictions.values()
            ]
            single.append(logits)
            scaled.append(self.scalers[seed].scale(logits))
            combined.append(ensemble_logits(members))
            all_labels.append(labels)

        labels = torch.cat(all_labels)
        pooled = {
            "single": (torch.cat(single), labels),
            "temperature": (torch.cat(scaled), labels),
            "ensemble": (torch.cat(combined), labels),
        }
        return pooled

    def _fit_scaler(self, predictions: Predictions) -> TemperatureScaler:
        """A temperature scaler fitted on validation only, never on test."""
        val_logits, val_labels = predictions["val"]
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        return scaler

    def _score_set(
        self, name: str, corruption: str, severity: int
    ) -> list[dict[str, Any]]:
        """Rows for every method on one evaluation set."""
        where = {"corruption": corruption, "severity": severity}
        rows = []
        member_logits = []

        for seed, predictions in self.seed_predictions.items():
            logits, labels = predictions[name]
            scaled_logits = self.scalers[seed].scale(logits)
            rows.append(_row("single", seed, where, logits, labels))
            rows.append(_row("temperature", seed, where, scaled_logits, labels))
            member_logits.append(logits)

        combined = ensemble_logits(member_logits)
        rows.append(_row("ensemble", None, where, combined, labels))
        return rows


def _row(
    method: str,
    seed: int | None,
    where: dict[str, Any],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """One results-table row: what was scored, then its scores."""
    row = {"method": method, "seed": seed, **where, **score(logits, labels)}
    return row
