"""Figuras para reports, notebook y W&B."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np

# Backend Agg por defecto: no depende de un display y no rompe el entrenamiento
# headless. En un notebook, "%matplotlib inline" tiene precedencia y las figuras se
# renderizan en la celda.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def plot_sample_grid(images: torch.Tensor, labels: torch.Tensor | None = None, n: int = 8):
    """Rejilla de ejemplos reales del dataset."""
    from digitclf.data.transforms import denormalize

    images = denormalize(images[:n].cpu())
    fig, axes = plt.subplots(1, min(n, len(images)), figsize=(2 * min(n, len(images)), 2.4))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if labels is not None:
            ax.set_title(f"label={int(labels[i])}", fontsize=8)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(confusion: torch.Tensor, class_names: Sequence[str] | None = None):
    """Matriz de confusion normalizada como heatmap."""
    cm = confusion.float()
    cm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)
    names = list(class_names or [str(i) for i in range(cm.shape[0])])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion (normalizada por fila)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def plot_training_history(history: Sequence, key: str = "accuracy"):
    """Curvas de loss/accuracy de train y validacion."""
    epochs = [e.epoch for e in history]
    train_key = f"train_{key}"
    val_key = f"val_{key}"
    fig, ax = plt.subplots(figsize=(7, 4))
    if hasattr(history[0], train_key):
        ax.plot(epochs, [getattr(e, train_key) for e in history], marker="o", label=f"train_{key}")
    if hasattr(history[0], val_key):
        ax.plot(epochs, [getattr(e, val_key) for e in history], marker="s", label=f"val_{key}")
    ax.set_xlabel("Epoca")
    ax.set_ylabel(key)
    ax.set_title(f"Historia de entrenamiento: {key}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_error_analysis(images, targets, preds, n: int = 12, class_names=None):
    """Muestra los errores mas frecuentes para analisis de errores."""
    wrong = np.where(np.asarray(targets) != np.asarray(preds))[0][:n]
    from digitclf.data.transforms import denormalize

    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.text(0.5, 0.5, "Sin errores en este lote", ha="center", va="center")
        ax.axis("off")
        return fig

    images = denormalize(torch.as_tensor(np.asarray(images), dtype=torch.float32))
    cols = min(len(wrong), n)
    fig, axes = plt.subplots(1, cols, figsize=(1.8 * cols, 2.6))
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, wrong, strict=False):
        ax.imshow(images[int(idx)].squeeze().numpy(), cmap="gray")
        ax.set_title(f"real {int(targets[idx])}\npred {int(preds[idx])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Analisis de errores", fontsize=10)
    fig.tight_layout()
    return fig
