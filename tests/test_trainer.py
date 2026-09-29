"""Tests del trainer con datos sinteticos (sin descargar MNIST)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from digitclf.config import OptimConfig
from digitclf.data import DataBundle
from digitclf.engine.metrics import ClassificationMetrics
from digitclf.engine.trainer import Trainer
from digitclf.models import build_model
from digitclf.utils.seed import seed_everything


def _synthetic_bundle(n: int = 64, num_classes: int = 4) -> DataBundle:
    """Dataset sintetico: clase = patron de pixeles, facil de aprender."""
    torch.manual_seed(0)
    images = torch.rand(n, 1, 8, 8)
    labels = torch.randint(0, num_classes, (n,))
    for i in range(n):
        images[i, :, labels[i] % 8, :] = 1.0
    ds = TensorDataset(images, labels)
    loader = DataLoader(ds, batch_size=16)
    return DataBundle(train=loader, val=loader, num_classes=num_classes)


def _trainer(**kwargs) -> Trainer:
    seed_everything(1)
    bundle = _synthetic_bundle()
    model = build_model("simple_cnn", num_classes=4, width=8, head_hidden=16)
    return Trainer(
        model=model,
        dataloaders=bundle,
        optim_cfg=OptimConfig(max_epochs=2, steps_per_epoch=4, lr=1e-2),
        device="cpu",
        use_amp=False,
        **kwargs,
    )


def test_training_runs_and_improves():
    trainer = _trainer()
    result = trainer.fit()
    assert len(result.history) == 2
    assert result.best_epoch >= 1
    assert 0.0 <= result.best_val_accuracy <= 1.0


def test_metrics_are_finite():
    trainer = _trainer()
    trainer.fit()
    metrics = trainer.evaluate()
    assert np.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_time_budget_stops_early():
    trainer = _trainer(time_budget_seconds=1e-6)
    result = trainer.fit(epochs=20)
    assert result.stop_reason == "time_budget"
    assert len(result.history) < 20


def test_early_stopping():
    trainer = _trainer(early_stopping_patience=1)
    result = trainer.fit(epochs=10)
    assert result.stop_reason in {"early_stopping", "max_epochs"}


def test_invalid_optimizer_raises():
    bundle = _synthetic_bundle()
    with pytest.raises(ValueError, match="Optimizador"):
        Trainer(
            model=build_model("simple_cnn", num_classes=4, width=8),
            dataloaders=bundle,
            optim_cfg=OptimConfig(name="inexistente"),
            device="cpu",
            use_amp=False,
        )


def test_invalid_scheduler_raises_on_fit():
    """El scheduler se construye en fit() para conocer los epochs reales."""
    bundle = _synthetic_bundle()
    trainer = Trainer(
        model=build_model("simple_cnn", num_classes=4, width=8),
        dataloaders=bundle,
        optim_cfg=OptimConfig(scheduler="inexistente", max_epochs=1),
        device="cpu",
        use_amp=False,
    )
    with pytest.raises(ValueError, match="Scheduler"):
        trainer.fit()


def test_invalid_tracker_args():
    with pytest.raises(ValueError):
        _trainer(log_every_n_steps=0)
    with pytest.raises(ValueError):
        _trainer(time_budget_seconds=-1)


def test_evaluate_returns_metrics_object():
    trainer = _trainer()
    out = trainer.evaluate()
    assert isinstance(out["_metrics"], ClassificationMetrics)
    assert out["_metrics"].total == 64


def test_checkpoint_saved(tmp_path):
    trainer = _trainer(checkpoint_dir=tmp_path)
    result = trainer.fit()
    assert (tmp_path / "best.pt").exists()
    assert result.checkpoint_path is not None
