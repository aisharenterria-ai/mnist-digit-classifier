"""Metricas de clasificacion implementadas en PyTorch (sin dependencias extra)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ClassificationMetrics:
    """Metricas multiclase (accuracy, macro P/R/F1 y confusion matrix).

    Se acumula batch a batch usando una matriz de confusion, que contiene
    toda la informacion necesaria y evita concatenar tensores gigantes en memoria.
    """

    num_classes: int = 10
    class_names: list[str] | None = None
    confusion: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("num_classes debe ser >= 2")
        self.confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float64)
        if self.class_names is None:
            self.class_names = [str(i) for i in range(self.num_classes)]
        elif len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names tiene {len(self.class_names)} entradas "
                f"pero num_classes={self.num_classes}"
            )

    # ------------------------------------------------------------------ acumulacion
    def reset(self) -> ClassificationMetrics:
        """Reinicia los contadores."""
        self.confusion.zero_()
        return self

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> ClassificationMetrics:
        """Acumula un batch de predicciones.

        Args:
            preds: etiquetas predichas ``(N,)``.
            targets: etiquetas reales ``(N,)``.

        Nota:
            El conteo se hace en CPU porque MPS no soporta float64 y la
            matriz de confusion se acumula en float64 (exacta al sumar enteros).
        """
        preds = preds.detach().flatten().to(device="cpu", dtype=torch.long)
        targets = targets.detach().flatten().to(device="cpu", dtype=torch.long)
        if preds.shape != targets.shape:
            raise ValueError(
                f"preds y targets deben tener la misma forma: {preds.shape} vs {targets.shape}"
            )
        if targets.numel() == 0:
            return self
        idx = targets * self.num_classes + preds
        counts = torch.bincount(idx, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes).to(torch.float64)
        return self

    # ------------------------------------------------------------------ metricas
    @property
    def total(self) -> int:
        """Numero de ejemplos acumulados."""
        return int(self.confusion.sum().item())

    @property
    def accuracy(self) -> float:
        """Exactitud global."""
        total = self.total
        if total == 0:
            return 0.0
        correct = float(torch.diagonal(self.confusion).sum().item())
        return correct / total

    def precision(self) -> torch.Tensor:
        """Precision por clase (0 si la clase nunca se predice)."""
        tp = self.confusion.diagonal()
        predicted = self.confusion.sum(dim=0)
        return torch.where(predicted > 0, tp / predicted.clamp(min=1), torch.zeros_like(tp))

    def recall(self) -> torch.Tensor:
        """Recall por clase (0 si la clase nunca aparece en targets)."""
        tp = self.confusion.diagonal()
        actual = self.confusion.sum(dim=1)
        return torch.where(actual > 0, tp / actual.clamp(min=1), torch.zeros_like(tp))

    def f1(self) -> torch.Tensor:
        """F1-score por clase."""
        p, r = self.precision(), self.recall()
        denom = p + r
        return torch.where(denom > 0, 2 * p * r / denom.clamp(min=1e-12), torch.zeros_like(p))

    def confusion_matrix(self) -> torch.Tensor:
        """Matriz de confusion: filas = real, columnas = predicho."""
        return self.confusion.clone()

    def normalized_confusion(self) -> torch.Tensor:
        """Matriz de confusion normalizada por fila (recursos de cada clase)."""
        row_sums = self.confusion.sum(dim=1, keepdim=True).clamp(min=1)
        return self.confusion / row_sums

    # ------------------------------------------------------------------ reportes
    def summary(self) -> dict[str, float]:
        """Diccionario de metricas agregadas."""
        return {
            "accuracy": self.accuracy,
            "macro_precision": float(self.precision().mean().item()),
            "macro_recall": float(self.recall().mean().item()),
            "macro_f1": float(self.f1().mean().item()),
            "total": self.total,
        }

    def per_class_table(self) -> list[dict[str, object]]:
        """Detalle por clase listo para W&B o reports."""
        p, r, f1 = self.precision(), self.recall(), self.f1()
        support = self.confusion.sum(dim=1)
        return [
            {
                "class": self.class_names[c],
                "support": int(support[c].item()),
                "precision": round(float(p[c].item()), 5),
                "recall": round(float(r[c].item()), 5),
                "f1": round(float(f1[c].item()), 5),
            }
            for c in range(self.num_classes)
        ]

    def top_confusions(self, k: int = 5) -> list[dict[str, object]]:
        """Pares (real, predicho) mas frecuentes fuera de la diagonal."""
        cm = self.confusion.clone()
        cm.fill_diagonal_(0)
        flat = cm.flatten()
        top = torch.topk(flat, k=min(k, flat.numel()))
        out = []
        for value, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            if value == 0:
                continue
            real, pred = divmod(idx, self.num_classes)
            out.append(
                {
                    "true": self.class_names[real],
                    "pred": self.class_names[pred],
                    "count": int(value),
                }
            )
        return out
