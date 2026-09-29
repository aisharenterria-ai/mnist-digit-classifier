"""Transformaciones de imagen para MNIST."""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms

from digitclf.config import DataConfig


def build_train_transform(cfg: DataConfig) -> transforms.Compose:
    """Construye la cadena de transformaciones de entrenamiento.

    El aumento es conservador a proposito: MNIST ya esta alineado, asi que
    rotaciones o cutting planes agresivos empeoran la generalizacion.
    """
    steps: list[transforms.Transform] = [transforms.ToTensor()]
    if cfg.augment:
        steps.append(
            transforms.RandomAffine(
                degrees=cfg.random_affine_degrees,
                translate=(cfg.random_affine_translate, cfg.random_affine_translate),
                scale=(1.0 - cfg.random_affine_scale, 1.0 + cfg.random_affine_scale),
                fill=0,
            )
        )
    steps.append(transforms.Normalize(cfg.normalize_mean, cfg.normalize_std))
    if cfg.random_erasing_p > 0:
        steps.append(
            transforms.RandomErasing(
                p=cfg.random_erasing_p,
                value="random",
                scale=(0.02, 0.12),
            )
        )
    return transforms.Compose(steps)


def build_eval_transform(cfg: DataConfig) -> transforms.Compose:
    """Transformaciones deterministas para validacion, test e inferencia."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ]
    )


def denormalize(
    image: object,
    mean: Sequence[float] = (0.1307,),
    std: Sequence[float] = (0.3081,),
):
    """Invierte la normalizacion para poder visualizar imagenes."""
    import torch

    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    mean_t = torch.as_tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_t = torch.as_tensor(std, dtype=image.dtype).view(-1, 1, 1)
    return image * std_t + mean_t
