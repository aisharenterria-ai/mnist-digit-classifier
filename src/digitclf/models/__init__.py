"""Modelos de clasificacion de digitos y registro de arquitecturas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from digitclf.models.mlp import MLP
from digitclf.models.resnet_mnist import ResNetMNIST
from digitclf.models.simple_cnn import SimpleCNN

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "mlp": MLP,
    "resnet_mnist": ResNetMNIST,
}

__all__ = [
    "MODEL_REGISTRY",
    "MLP",
    "ResNetMNIST",
    "SimpleCNN",
    "available_models",
    "build_model",
    "count_parameters",
]


def available_models() -> list[str]:
    """Nombres de las arquitecturas registradas."""
    return sorted(MODEL_REGISTRY)


def build_model(name: str, **params: Any) -> nn.Module:
    """Instancia un modelo por nombre.

    Args:
        name: clave de ``MODEL_REGISTRY`` (case-insensitive).
        **params: argumentos del constructor del modelo.

    Returns:
        Modelo ``nn.Module`` sin inicializar pesos.

    Raises:
        ValueError: si la arquitectura no existe.
    """
    key = name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{name}' no registrado. Disponibles: {available_models()}"
        )
    return MODEL_REGISTRY[key](**params)


def count_parameters(model: nn.Module) -> int:
    """Numero total de parametros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
