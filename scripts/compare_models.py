#!/usr/bin/env python3
"""Compara varias configuraciones en una sola ejecucion (baseline vs CNN vs ResNet).

    python scripts/compare_models.py --epochs 5 --models simple_cnn mlp resnet_mnist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digitclf.config import load_config, parse_overrides  # noqa: E402
from digitclf.models import available_models, build_model, count_parameters  # noqa: E402
from digitclf.pipeline import Experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mlp", "simple_cnn", "resnet_mnist"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="clave=valor")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_config(args.config, parse_overrides(args.overrides))

    rows = []
    for model_name in args.models:
        if model_name not in available_models():
            raise SystemExit(f"Modelos validos: {available_models()}")
        cfg = base.replace(
            name=f"{base.name}_{model_name}",
            **{"model.name": model_name, "model.params": {}, "optim.max_epochs": args.epochs},
        )
        model = build_model(cfg.model.name, **cfg.model.params)
        experiment = Experiment(cfg, use_wandb=not args.no_wandb)
        experiment.model = model
        result, _ = experiment.run()
        rows.append(
            {
                "model": model_name,
                "params": count_parameters(model),
                "best_epoch": result.best_epoch,
                "val_accuracy": round(result.best_val_accuracy, 4),
                "val_macro_f1": round(result.best_val_macro_f1, 4),
                "seconds": round(result.total_seconds, 1),
            }
        )

    print("\n" + "=" * 72)
    print(f"{'model':<16}{'params':>12}{'val_acc':>10}{'macro_f1':>11}{'seg':>8}")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['model']:<16}{row['params']:>12,}{row['val_accuracy']:>10.4f}"
            f"{row['val_macro_f1']:>11.4f}{row['seconds']:>8.1f}"
        )
    print("=" * 72)

    out = Path("runs/comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
