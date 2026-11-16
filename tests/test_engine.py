"""Tests for the trainer and inference loop, on a tiny synthetic problem."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cifar10c_calibration.config import TrainConfig
from cifar10c_calibration.engine import Trainer, predict_logits

DEVICE = torch.device("cpu")


def _toy_loader(shuffle: bool = True) -> DataLoader[tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(64, 2, generator=generator)
    y = (x[:, 0] > 0).long()
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=shuffle)


def _toy_config(tmp_path: Path) -> TrainConfig:
    return TrainConfig(epochs=20, lr=0.5, weight_decay=0.0, output_dir=tmp_path)


def test_fit_reduces_loss(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trainer = Trainer(nn.Linear(2, 2), _toy_config(tmp_path), DEVICE)

    history = trainer.fit(_toy_loader())

    assert len(history) == 20
    assert history[-1].loss < history[0].loss
    assert history[-1].accuracy > 0.9


def test_learning_rate_decays_to_zero(tmp_path: Path) -> None:
    trainer = Trainer(nn.Linear(2, 2), _toy_config(tmp_path), DEVICE)

    trainer.fit(_toy_loader())

    assert trainer.optimizer.param_groups[0]["lr"] < 1e-6


def test_save_checkpoint_round_trips(tmp_path: Path) -> None:
    config = _toy_config(tmp_path)
    trainer = Trainer(nn.Linear(2, 2), config, DEVICE)

    trainer.save_checkpoint()
    saved = torch.load(config.checkpoint_path)

    restored = nn.Linear(2, 2)
    restored.load_state_dict(saved["model_state"])
    assert torch.equal(restored.weight, trainer.model.state_dict()["weight"])
    assert saved["config"]["seed"] == config.seed


def test_predict_logits_shapes_and_order() -> None:
    model = nn.Linear(2, 2)
    loader = _toy_loader(shuffle=False)

    logits, labels = predict_logits(model, loader, DEVICE)

    assert logits.shape == (64, 2)
    assert labels.shape == (64,)
    assert torch.equal(labels, loader.dataset.tensors[1])  # type: ignore[attr-defined]


def test_predict_logits_does_not_change_weights() -> None:
    model = nn.Linear(2, 2)
    before = [p.clone() for p in model.parameters()]

    predict_logits(model, _toy_loader(), DEVICE)

    assert all(
        torch.equal(b, a) for b, a in zip(before, model.parameters(), strict=True)
    )
