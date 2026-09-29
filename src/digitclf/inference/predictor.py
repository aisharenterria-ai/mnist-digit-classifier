"""Predictor de digitos: carga un checkpoint y clasifica imagenes en 28x28.

Pensado para el uso en produccion: no necesita torchvision datasets ni el
datamodule, solo PIL/numpy + el state_dict.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from digitclf.config import DataConfig
from digitclf.engine.checkpoint import load_checkpoint
from digitclf.models import build_model
from digitclf.utils.device import get_device
from digitclf.utils.logging import get_logger

LOGGER = get_logger(__name__)
ImageLike = Image.Image | np.ndarray | torch.Tensor | str | Path


class DigitPredictor:
    """Clasificador de digitos cargado desde un checkpoint del entrenamiento."""

    def __init__(
        self,
        model: nn.Module,
        data_cfg: DataConfig,
        device: str | torch.device = "auto",
    ) -> None:
        self.model = model
        self.device = get_device(device)
        self.model.to(self.device).eval()
        self.data_cfg = data_cfg
        self.mean = float(data_cfg.normalize_mean[0])
        self.std = float(data_cfg.normalize_std[0])

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path | None = None,
        device: str | torch.device = "auto",
    ) -> DigitPredictor:
        """Reconstruye el modelo desde un checkpoint (con su config guardada)."""
        import yaml

        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        saved_cfg = payload.get("config") or {}
        if config_path is not None:
            saved_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or saved_cfg

        data_cfg = _build_data_config(saved_cfg.get("data", {}))
        model_cfg = saved_cfg.get("model", {})
        model = build_model(model_cfg.get("name", "simple_cnn"), **model_cfg.get("params", {}))
        load_checkpoint(checkpoint_path, model, map_location="cpu")
        LOGGER.info("Predictor inicializado desde %s", checkpoint_path)
        return cls(model, data_cfg, device=device)

    # ------------------------------------------------------------------ preproc
    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Convierte cualquier imagen soportada en un tensor normalizado ``(1,1,28,28)``."""
        arr = self._to_array(image)
        tensor = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0).unsqueeze(0)
        if tensor.shape[-1] != self.data_cfg.image_size or tensor.shape[-2] != self.data_cfg.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(self.data_cfg.image_size, self.data_cfg.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (tensor - self.mean) / self.std

    def _to_array(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                return np.array(img.convert("L"), dtype=np.float32)
        if isinstance(image, Image.Image):
            return np.array(image.convert("L"), dtype=np.float32)
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy().astype(np.float32)
            arr = arr * 255.0 if arr.max() <= 1.0 else arr
        else:
            arr = np.asarray(image, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze()
        return arr

    # ------------------------------------------------------------------ prediccion
    @torch.no_grad()
    def predict_proba(self, image: ImageLike) -> torch.Tensor:
        """Devuelve el vector de probabilidades ``(num_classes,)``."""
        batch = self.preprocess(image).to(self.device)
        logits = self.model(batch)
        return torch.softmax(logits, dim=1).squeeze(0).cpu()

    def predict(self, image: ImageLike) -> int:
        """Devuelve la etiqueta mas probable."""
        return int(self.predict_proba(image).argmax().item())

    def predict_batch(self, images: Sequence[ImageLike]) -> list[int]:
        """Predice una lista de imagenes."""
        return [self.predict(img) for img in images]

    def predict_with_confidence(self, image: ImageLike) -> tuple[int, float]:
        """Devuelve ``(etiqueta, confianza)``."""
        probs = self.predict_proba(image)
        best = int(probs.argmax().item())
        return best, float(probs[best].item())

    @torch.no_grad()
    def predict_dataset(self, loader: Iterable, return_probs: bool = False):
        """Predice sobre un DataLoader de la competition (batch, label)."""
        preds: list[torch.Tensor] = []
        probs: list[torch.Tensor] = []
        ids: list[torch.Tensor] = []
        self.model.eval()
        for batch in loader:
            images = batch[0].to(self.device, non_blocking=True)
            logits = self.model(images)
            preds.append(logits.argmax(dim=1).cpu())
            if return_probs:
                probs.append(torch.softmax(logits, dim=1).cpu())
            if len(batch) > 1:
                ids.append(batch[1].cpu())
        return (
            torch.cat(preds),
            torch.cat(ids) if ids else None,
            torch.cat(probs) if probs else None,
        )


def _build_data_config(data: dict) -> DataConfig:
    from digitclf.config import config_from_dict

    cfg = config_from_dict({"data": data or {}})
    return cfg.data
