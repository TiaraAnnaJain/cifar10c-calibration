"""Tests for temperature scaling."""

import pytest
import torch

from cifar10c_calibration.calibration import TemperatureScaler, nll

TRUE_TEMPERATURE = 3.0


def _overconfident_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Logits that are exactly TRUE_TEMPERATURE times too confident."""
    generator = torch.Generator().manual_seed(0)
    calibrated_logits = torch.randn(20_000, 10, generator=generator) * 2
    labels = torch.multinomial(
        calibrated_logits.softmax(dim=1), num_samples=1, generator=generator
    ).squeeze(1)
    overconfident_logits = calibrated_logits * TRUE_TEMPERATURE
    return overconfident_logits, labels


def test_unfitted_scaler_leaves_logits_unchanged() -> None:
    logits = torch.randn(5, 10)
    assert torch.equal(TemperatureScaler().scale(logits), logits)


def test_fit_recovers_the_true_temperature() -> None:
    logits, labels = _overconfident_data()
    scaler = TemperatureScaler()

    scaler.fit(logits, labels)

    assert scaler.temperature == pytest.approx(TRUE_TEMPERATURE, rel=0.05)


def test_fit_lowers_nll() -> None:
    logits, labels = _overconfident_data()
    scaler = TemperatureScaler()

    scaler.fit(logits, labels)

    assert nll(scaler.scale(logits), labels) < nll(logits, labels)


def test_scaling_does_not_change_predictions() -> None:
    logits, labels = _overconfident_data()
    scaler = TemperatureScaler()

    scaler.fit(logits, labels)

    assert torch.equal(scaler.scale(logits).argmax(dim=1), logits.argmax(dim=1))
