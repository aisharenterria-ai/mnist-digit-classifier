"""Seleccion de dispositivo y utilidades de modelos."""

from __future__ import annotations

from typing import Any

import torch


def get_device(device: str | torch.device = "auto") -> torch.device:
    """Resuelve el dispositivo a usar.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"`` o un ``torch.device``.

    Returns:
        Un ``torch.device`` concreto.

    Raises:
        ValueError: si el dispositivo solicitado no esta disponible.
    """
    if isinstance(device, torch.device):
        resolved = device
    elif device == "auto":
        resolved = _auto_device()
    else:
        resolved = torch.device(device)
    _assert_available(resolved)
    return resolved


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _assert_available(device: torch.device) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Se pidio CUDA pero no hay GPU disponible")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Se pidio MPS pero no hay Apple Silicon con MPS habilitado")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Cuenta los parametros del modelo (util para reports y W&B)."""
    params: Any = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def describe_device(device: torch.device) -> str:
    """Descripcion legible del dispositivo para los logs."""
    if device.type == "cuda":
        return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"
