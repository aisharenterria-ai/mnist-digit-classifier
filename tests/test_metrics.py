"""Tests de metricas."""

from __future__ import annotations

import torch

from digitclf.engine.metrics import ClassificationMetrics


def test_perfect_predictions():
    m = ClassificationMetrics(num_classes=10)
    targets = torch.arange(10)
    m.update(targets.clone(), targets)
    assert m.accuracy == 1.0
    assert abs(float(m.f1().mean()) - 1.0) < 1e-9


def test_confusion_matrix_orientation():
    """Fila = real, columna = predicho."""
    m = ClassificationMetrics(num_classes=3)
    targets = torch.tensor([0, 0, 1, 2])
    preds = torch.tensor([0, 1, 1, 0])
    m.update(preds, targets)
    cm = m.confusion_matrix()
    assert cm[0, 0] == 1 and cm[0, 1] == 1
    assert cm[1, 1] == 1
    assert cm[2, 0] == 1
    assert m.total == 4
    assert abs(m.accuracy - 0.5) < 1e-9


def test_perfect_class_scores():
    targets = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    m = ClassificationMetrics(num_classes=3)
    m.update(targets.clone(), targets)
    s = m.summary()
    assert abs(s["macro_precision"] - 1.0) < 1e-9
    assert abs(s["macro_recall"] - 1.0) < 1e-9
    assert abs(s["macro_f1"] - 1.0) < 1e-9


def test_reset_and_accumulate():
    m = ClassificationMetrics(num_classes=2)
    t = torch.tensor([0, 1])
    m.update(t.clone(), t)
    first = m.total
    m.update(t.clone(), t)
    assert m.total == first * 2
    m.reset()
    assert m.total == 0


def test_mismatched_shapes_raise():
    m = ClassificationMetrics(num_classes=2)
    try:
        m.update(torch.tensor([0, 1]), torch.tensor([0]))
    except ValueError:
        return
    raise AssertionError("Debe fallar con formas distintas")


def test_top_confusions():
    m = ClassificationMetrics(num_classes=3)
    targets = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([1, 1, 1, 1])
    m.update(preds, targets)
    top = m.top_confusions(3)
    assert top[0]["true"] == "0" and top[0]["pred"] == "1"


def test_num_classes_validation():
    try:
        ClassificationMetrics(num_classes=1)
    except ValueError:
        return
    raise AssertionError("num_classes=1 debe ser invalido")
