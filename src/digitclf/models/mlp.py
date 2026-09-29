"""Perceptron multicapa: baseline para comparar contra la CNN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Red completamente conectada con normalizacion y dropout.

    Sirve como linea base honesta: si la CNN no supera claramente a este modelo,
    la CNN no esta aportando nada.
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_features: int = 28 * 28,
        hidden_sizes: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes debe ser >= 2")
        hidden_sizes = tuple(hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes no puede estar vacio")

        layers: list[nn.Module] = []
        prev = in_features
        for units in hidden_sizes:
            layers += [
                nn.Linear(prev, units),
                nn.BatchNorm1d(units),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = units
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W) o (N, H*W) -> (N, num_classes)``."""
        x = x.flatten(1)
        return self.net(x)
