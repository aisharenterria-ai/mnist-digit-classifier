"""Guardado y carga de checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Guarda un checkpoint completo del entrenamiento.

    Incluye el estado del optimizador y la config, de forma que el entrenamiento
    puede reanudarse y el modelo desplegarse sin el codigo que lo creo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
        "config": config or {},
        **(extra or {}),
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Carga un checkpoint en el modelo (y opcionalmente en el optimizador)."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(f"El checkpoint {path} no contiene 'model_state_dict'")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def export_state_dict(path: str | Path, model: nn.Module) -> Path:
    """Exporta solo los pesos (formato ligero para inferencia/deploy)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
