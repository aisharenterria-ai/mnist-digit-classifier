"""Tests de modelos: formas de salida y conteo de parametros."""

from __future__ import annotations

import pytest
import torch

from digitclf.models import available_models, build_model
from digitclf.models.mlp import MLP
from digitclf.models.resnet_mnist import ResNetMNIST
from digitclf.models.simple_cnn import SimpleCNN


@pytest.mark.parametrize("name", available_models())
def test_forward_shape(name: str):
    model = build_model(name)
    out = model(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 10)


@pytest.mark.parametrize("name", available_models())
def test_num_classes_custom(name: str):
    model = build_model(name, num_classes=7)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 7)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="no registrado"):
        build_model("transformer_gpt")


def test_simple_cnn_param_count_is_reasonable():
    params = sum(p.numel() for p in SimpleCNN().parameters())
    assert 100_000 < params < 2_000_000


def test_model_is_in_eval_and_grad_enabled():
    """eval() desactiva dropout/BN, pero el gradiente depende del contexto."""
    model = SimpleCNN()
    model.eval()
    with torch.no_grad():
        assert not model(torch.randn(4, 1, 28, 28)).requires_grad
    model.train()
    assert model(torch.randn(4, 1, 28, 28)).requires_grad
    assert any(m.training for m in model.modules())


def test_backward_propagates():
    model = SimpleCNN(width=8)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 1, 28, 28)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_invalid_hyperparameters():
    with pytest.raises(ValueError):
        SimpleCNN(num_classes=1)
    with pytest.raises(ValueError):
        SimpleCNN(width=4)
    with pytest.raises(ValueError):
        MLP(hidden_sizes=())
    with pytest.raises(ValueError):
        ResNetMNIST(num_blocks=0)


def test_mlp_accepts_flat_input():
    assert MLP()(torch.randn(3, 784)).shape == (3, 10)
