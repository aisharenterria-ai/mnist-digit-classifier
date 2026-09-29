#!/usr/bin/env python3
"""Genera predicciones y un CSV de submission para la competicion.

Acepta imagenes sueltas (PNG/JPG) y produce el formato de submission estandar
con columnas ``ImageId,Label``.

    # Desde imagenes
    python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --images samples/*.png --output submission.csv

    # Desde el test set de MNIST
    python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --split test --output submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digitclf.inference import DigitPredictor  # noqa: E402
from digitclf.utils.logging import setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint best.pt")
    parser.add_argument("--images", nargs="*", default=[], help="Imagenes a clasificar")
    parser.add_argument("--split", choices=["test", "val"], default=None, help="Usar un split de MNIST")
    parser.add_argument("--limit", type=int, default=None, help="Maximo de imagenes a predecir (por defecto: todas)")
    parser.add_argument("--output", default="submission.csv", help="Ruta del CSV de salida")
    parser.add_argument("--device", default="auto")
    return parser


def write_submission(rows: list[tuple[int, int]], output: str | Path) -> Path:
    """Escribe el CSV en formato de competition."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ImageId", "Label"])
        writer.writerows(rows)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    predictor = DigitPredictor.from_checkpoint(args.checkpoint, device=args.device)
    rows: list[tuple[int, int]] = []

    if args.split:
        from torchvision.datasets import MNIST

        ds = MNIST(root=predictor.data_cfg.root, train=(args.split == "val"), download=True)
        preds, _, _ = predictor.predict_dataset(_loader(ds, predictor, args.limit))
        rows = [(i + 1, int(p)) for i, p in enumerate(preds)]
    elif args.images:
        paths = sorted(Path(p) for p in args.images)
        for i, path in enumerate(paths, start=1):
            label, conf = predictor.predict_with_confidence(path)
            rows.append((i, label))
            print(f"{path.name}: {label} (confianza {conf:.3f})")
    else:
        raise SystemExit("Indica --images o --split")

    path = write_submission(rows, args.output)
    print(f"\n{len(rows)} predicciones escritas en {path}")
    return 0


def _loader(dataset, predictor, limit: int | None = None):
    """DataLoader sobre el dataset con las transformaciones de evaluacion."""
    from torch.utils.data import DataLoader, Subset

    from digitclf.data.transforms import build_eval_transform

    tfm = build_eval_transform(predictor.data_cfg)
    if limit is not None:
        dataset = Subset(dataset, range(min(len(dataset), limit)))
    return DataLoader(_Transformed(dataset, tfm), batch_size=256, shuffle=False)


class _Transformed:
    """Aplica una transformacion a un dataset que devuelve (img, label)."""

    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return self.transform(image), label


if __name__ == "__main__":
    raise SystemExit(main())
