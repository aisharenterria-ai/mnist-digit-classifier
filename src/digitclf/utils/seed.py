"""Gestion de semillas para experiments reproducibles."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Fija la semilla en Python, NumPy y PyTorch.

    Args:
        seed: semilla base.
        deterministic: si es ``True`` activa los algoritmos deterministas de cuDNN
            (reproducibilidad a costa de algo de rendimiento).

    Returns:
        La semilla aplicada.
    """
    if seed < 0:
        raise ValueError("seed debe ser >= 0")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    return seed


def worker_init_fn(worker_id: int, base_seed: int = 42) -> None:
    """Sembrar los DataLoader workers de forma distinta pero determinista."""
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
