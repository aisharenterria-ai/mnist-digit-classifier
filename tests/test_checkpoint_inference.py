"""Tests de checkpointing y del predictor de inferencia."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from digitclf.config import DataConfig, config_from_dict
from digitclf.engine.checkpoint import export_state_dict, load_checkpoint, save_checkpoint
from digitclf.inference import DigitPredictor
from digitclf.models import build_model


def _checkpoint(tmp_path):
    """Checkpoint coherente: la config guardada describe el modelo guardado."""
    cfg = config_from_dict(
        {"name": "t", "wandb": {"enabled": False}, "model": {"name": "simple_cnn", "params": {"width": 8}}}
    )
    model = build_model(cfg.model.name, **cfg.model.params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = save_checkpoint(
        tmp_path / "best.pt",
        model,
        optimizer,
        epoch=3,
        metrics={"accuracy": 0.99},
        config=cfg.to_dict(),
    )
    return path, cfg


def test_checkpoint_roundtrip(tmp_path):
    path, _ = _checkpoint(tmp_path)
    model = build_model("simple_cnn", width=8)
    payload = load_checkpoint(path, model)
    assert payload["epoch"] == 3
    assert payload["metrics"]["accuracy"] == 0.99


def test_checkpoint_requires_state_dict(tmp_path):
    torch.save({"epoch": 1}, tmp_path / "bad.pt")
    with pytest.raises(KeyError):
        load_checkpoint(tmp_path / "bad.pt", build_model("simple_cnn", width=8))


def test_export_state_dict(tmp_path):
    model = build_model("simple_cnn", width=8)
    path = export_state_dict(tmp_path / "weights.pt", model)
    assert path.exists()


def test_predictor_from_checkpoint(tmp_path):
    path, _ = _checkpoint(tmp_path)
    predictor = DigitPredictor.from_checkpoint(path, device="cpu")
    img = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    label = predictor.predict(img)
    assert 0 <= label <= 9


def test_predictor_from_numpy_and_tensor():
    model = build_model("simple_cnn", width=8)
    predictor = DigitPredictor(model, DataConfig(), device="cpu")
    arr = np.zeros((28, 28), dtype=np.uint8)
    assert 0 <= predictor.predict(arr) <= 9
    assert 0 <= predictor.predict(torch.zeros(28, 28)) <= 9
    label, conf = predictor.predict_with_confidence(arr)
    assert 0 <= label <= 9 and 0.0 <= conf <= 1.0


def test_predictor_resizes_images():
    model = build_model("simple_cnn", width=8)
    predictor = DigitPredictor(model, DataConfig(), device="cpu")
    tensor = predictor.preprocess(Image.fromarray(np.zeros((64, 64), dtype=np.uint8)))
    assert tensor.shape == (1, 1, 28, 28)
