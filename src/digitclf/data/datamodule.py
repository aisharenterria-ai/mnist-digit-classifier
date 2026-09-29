"""Datamodule: encapsula la obtencion de datos y la construccion de dataloaders.

Aislar esto permite que el ``Trainer`` no sepa nada de torchvision ni de donde
salen los datos (facilita testear con datos sinteticos).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST

from digitclf.config import DataConfig
from digitclf.data.transforms import build_eval_transform, build_train_transform
from digitclf.utils.logging import get_logger
from digitclf.utils.seed import worker_init_fn

LOGGER = get_logger(__name__)


@dataclass
class DataBundle:
    """Dataloaders listos para entrenar."""

    train: DataLoader
    val: DataLoader
    test: DataLoader | None = None
    num_classes: int = 10
    class_names: list[str] | None = None

    def __post_init__(self) -> None:
        if self.class_names is None:
            self.class_names = [str(i) for i in range(self.num_classes)]

    def sizes(self) -> dict[str, int]:
        """Numero de ejemplos de cada split."""
        sizes = {"train": len(self.train.dataset), "val": len(self.val.dataset)}
        if self.test is not None:
            sizes["test"] = len(self.test.dataset)
        return sizes


class MNISTDataModule:
    """Descarga/prepara MNIST y produce dataloaders reproducibles.

    Uso tipico::

        dm = MNISTDataModule(DataConfig(), seed=42)
        bundle = dm.build()
        model = SimpleCNN()
        Trainer(model, bundle, OptimConfig()).fit()
    """

    def __init__(self, cfg: DataConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed
        self._train_eval: Dataset | None = None
        self._train_transform = build_train_transform(cfg)
        self._eval_transform = build_eval_transform(cfg)

    # ------------------------------------------------------------------ datasets
    @property
    def raw_dataset(self) -> Dataset:
        """Dataset completo de entrenamiento con transformaciones de train."""
        if self._train_eval is None:
            self._train_eval = MNIST(
                root=self.cfg.root,
                train=True,
                download=self.cfg.download,
                transform=self._train_transform,
            )
        return self._train_eval

    def split_indices(self, val_split: float | None = None) -> tuple[list[int], list[int]]:
        """Divide el train set de forma estratificada y determinista."""
        val_split = self.cfg.val_split if val_split is None else val_split
        n_total = len(self.raw_dataset)
        n_val = int(round(n_total * val_split))
        if n_val == 0:
            raise ValueError("val_split produce un set de validacion vacio")

        targets = list(self.raw_dataset.targets.numpy())
        by_class: dict[int, list[int]] = {}
        for idx, label in enumerate(targets):
            by_class.setdefault(int(label), []).append(idx)

        rng = random.Random(self.seed)
        val_indices: list[int] = []
        for _, indices in sorted(by_class.items()):
            shuffled = indices[:]
            rng.shuffle(shuffled)
            n_class_val = int(round(len(shuffled) * val_split))
            val_indices.extend(shuffled[:n_class_val])

        val_set = set(val_indices)
        train_indices = [i for i in range(n_total) if i not in val_set]
        return train_indices, sorted(val_indices)

    def _eval_view(self) -> Dataset:
        """El mismo train set pero con transformaciones deterministas."""
        return MNIST(
            root=self.cfg.root,
            train=True,
            download=self.cfg.download,
            transform=self._eval_transform,
        )

    # ------------------------------------------------------------------ loaders
    def build(self) -> DataBundle:
        """Construye los dataloaders de train/val/test."""
        train_idx, val_idx = self.split_indices()
        train_ds = Subset(self.raw_dataset, train_idx)
        val_ds = Subset(self._eval_view(), val_idx)

        bundle = DataBundle(
            train=self._loader(train_ds, shuffle=True),
            val=self._loader(val_ds, shuffle=False),
            test=self._loader(self._test_dataset(), shuffle=False) if self._has_test() else None,
            num_classes=self.cfg.num_classes,
        )
        LOGGER.info(
            "Dataloaders listos: train=%s val=%s test=%s",
            len(train_ds),
            len(val_ds),
            len(bundle.test.dataset) if bundle.test else 0,
        )
        return bundle

    def _has_test(self) -> bool:
        raw = Path(self.cfg.root) / "MNIST" / "raw"
        return raw.exists() and any(raw.glob("t10k-images-idx3-ubyte*"))

    def _test_dataset(self) -> Dataset:
        return MNIST(
            root=self.cfg.root,
            train=False,
            download=self.cfg.download,
            transform=self._eval_transform,
        )

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        pin_memory = self.cfg.pin_memory and torch.cuda.is_available()
        # partial (y no lambda) porque con num_workers>0 el callable debe ser picklable
        init_fn = partial(worker_init_fn, base_seed=self.seed) if self.cfg.num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=self.cfg.num_workers > 0,
            worker_init_fn=init_fn,
            generator=generator,
        )
