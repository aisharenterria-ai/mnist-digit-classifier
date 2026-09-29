"""Tests de transformaciones y datamodule (sin descarga real de datos)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torchvision import transforms

from digitclf.config import DataConfig
from digitclf.data.transforms import build_eval_transform, build_train_transform, denormalize


def test_eval_transform_is_deterministic():
    cfg = DataConfig()
    tfm = build_eval_transform(cfg)
    img = np.random.rand(28, 28).astype(np.float32)
    assert torch.allclose(tfm(img), tfm(img))


def test_train_transform_with_augment_is_stochastic():
    cfg = DataConfig(augment=True, random_affine_degrees=20)
    tfm = build_train_transform(cfg)
    img = np.random.rand(28, 28).astype(np.float32)
    outputs = [tfm(img) for _ in range(6)]
    assert any(not torch.allclose(outputs[0], o) for o in outputs[1:])


def test_train_transform_without_augment_is_deterministic():
    cfg = DataConfig(augment=False)
    tfm = build_train_transform(cfg)
    img = np.random.rand(28, 28).astype(np.float32)
    assert torch.allclose(tfm(img), tfm(img))


def test_output_shape_and_normalization():
    cfg = DataConfig()
    img = np.random.rand(28, 28).astype(np.float32)
    out = build_eval_transform(cfg)(img)
    assert out.shape == (1, 28, 28)
    # Se comprueba la normalizacion contra el calculo directo
    expected = (torch.from_numpy(img) - cfg.normalize_mean[0]) / cfg.normalize_std[0]
    assert torch.allclose(out, expected, atol=1e-6)


def test_random_affine_scale_is_a_range():
    """El scale debe ser una tupla (min, max) segun el API de torchvision."""
    cfg = DataConfig(augment=True, random_affine_scale=0.1)
    tfm = build_train_transform(cfg)
    affine = [t for t in tfm.transforms if isinstance(t, transforms.RandomAffine)][0]
    assert isinstance(affine.scale, tuple) and len(affine.scale) == 2
    assert affine.scale[0] < 1.0 < affine.scale[1]


def test_random_erasing_optional():
    cfg = DataConfig(augment=True, random_erasing_p=0.5)
    tfm = build_train_transform(cfg)
    assert any(isinstance(t, transforms.RandomErasing) for t in tfm.transforms)


def test_denormalize_inverts_normalization():
    cfg = DataConfig()
    original = torch.rand(1, 28, 28)
    normalized = (original - cfg.normalize_mean[0]) / cfg.normalize_std[0]
    assert torch.allclose(denormalize(normalized, cfg.normalize_mean, cfg.normalize_std), original, atol=1e-5)


def test_grayscale_normalization_rejected():
    with pytest.raises(ValueError):
        DataConfig(normalize_mean=(0.1, 0.2, 0.3))
