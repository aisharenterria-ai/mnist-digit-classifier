#!/usr/bin/env python3
"""Evalua un checkpoint entrenado sobre el dataset de test.

    python scripts/evaluate.py runs/mnist_simplecnn/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digitclf.config import load_config  # noqa: E402
from digitclf.data import MNISTDataModule  # noqa: E402
from digitclf.engine.checkpoint import load_checkpoint  # noqa: E402
from digitclf.engine.metrics import ClassificationMetrics  # noqa: E402
from digitclf.models import build_model  # noqa: E402
from digitclf.utils.device import get_device  # noqa: E402
from digitclf.utils.logging import setup_logging  # noqa: E402
from digitclf.utils.seed import seed_everything  # noqa: E402
from digitclf.viz import plot_confusion_matrix  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Ruta al checkpoint best.pt")
    parser.add_argument("--config", default=None, help="YAML de configuracion del experimento")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-figure", default=None, help="Ruta para guardar la matriz de confusion")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging()

    config = load_config(args.config)
    config.data.download = True
    device = get_device(args.device)
    seed_everything(config.seed, config.deterministic)

    payload = __import__("torch").load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = (payload.get("config") or {}).get("model", {})
    model = build_model(model_cfg.get("name", "simple_cnn"), **model_cfg.get("params", {}))
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device).eval()

    data_cfg = (payload.get("config") or {}).get("data") or {}
    if data_cfg:
        from digitclf.config import config_from_dict

        config.data = config_from_dict({"data": data_cfg}).data

    bundle = MNISTDataModule(config.data, seed=config.seed).build()
    loader = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[args.split]
    if loader is None:
        raise SystemExit(f"El split '{args.split}' no esta disponible")

    metrics = ClassificationMetrics(num_classes=config.data.num_classes, class_names=bundle.class_names)
    criterion = __import__("torch").nn.CrossEntropyLoss()
    total_loss, total = 0.0, 0
    with __import__("torch").no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += float(criterion(logits, targets).item()) * targets.size(0)
            total += targets.size(0)
            metrics.update(logits.argmax(dim=1), targets)

    summary = metrics.summary()
    summary["loss"] = total_loss / max(1, total)
    print(json.dumps(summary, indent=2))
    print("\nMetricas por clase:")
    print(json.dumps(metrics.per_class_table(), indent=2))
    print("\nConfusiones mas frecuentes:")
    print(json.dumps(metrics.top_confusions(5), indent=2))

    if args.save_figure:
        plot_confusion_matrix(metrics.confusion_matrix(), bundle.class_names).savefig(
            args.save_figure, dpi=150
        )
        print(f"Matriz de confusion guardada en {args.save_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
