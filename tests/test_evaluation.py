"""Tests for building the results table, on random fake predictions."""

import pandas as pd
import torch

from cifar10c_calibration.evaluation import CLEAN, Evaluator, scored_sets
from cifar10c_calibration.prediction import Predictions

SEEDS = [0, 1]
N_IMAGES = 50


def _fake_predictions(seed: int) -> Predictions:
    """Random logits for validation and every scored set, with shared labels."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(N_IMAGES) % 10
    names = ["val"] + [name for name, _, _ in scored_sets()]
    predictions = {
        name: (torch.randn(N_IMAGES, 10, generator=generator) * 3, labels)
        for name in names
    }
    return predictions


def _results() -> pd.DataFrame:
    evaluator = Evaluator({seed: _fake_predictions(seed) for seed in SEEDS})
    results = evaluator.results()
    return results


def test_scored_sets_are_clean_test_plus_75_corrupted() -> None:
    sets = scored_sets()
    assert len(sets) == 76
    assert sets[0] == ("test", CLEAN, 0)


def test_one_row_per_method_seed_and_set() -> None:
    results = _results()
    rows_per_set = 2 * len(SEEDS) + 1
    assert len(results) == len(scored_sets()) * rows_per_set
    assert set(results["method"]) == {"single", "temperature", "ensemble"}


def test_temperature_scaling_keeps_accuracy() -> None:
    results = _results()
    single = results[results["method"] == "single"]["accuracy"].to_numpy()
    scaled = results[results["method"] == "temperature"]["accuracy"].to_numpy()
    assert (single == scaled).all()


def test_ensemble_rows_have_no_seed() -> None:
    results = _results()
    assert results[results["method"] == "ensemble"]["seed"].isna().all()


def test_one_temperature_per_seed() -> None:
    evaluator = Evaluator({seed: _fake_predictions(seed) for seed in SEEDS})
    assert list(evaluator.temperatures) == SEEDS
    assert all(t > 0 for t in evaluator.temperatures.values())
