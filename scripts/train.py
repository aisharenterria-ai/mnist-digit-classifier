#!/usr/bin/env python3
"""Entrena un modelo de clasificacion de digitos MNIST.

Ejemplos::

    # Entrenamiento completo con la config por defecto
    python scripts/train.py

    # Entrenamiento rapido en CPU
    python scripts/train.py --config configs/quick.yaml

    # Overrides en linea
    python scripts/train.py --set optim.lr=3e-3 optim.max_epochs=20 model.name=mlp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar el script sin instalar el paquete
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from digitclf.config import load_config, parse_overrides  # noqa: E402
from digitclf.pipeline import Experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Ruta a un YAML de configuracion")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="clave=valor",
        help="Overrides en notacion punteada (p.ej. optim.lr=0.003)",
    )
    parser.add_argument("--name", default=None, help="Nombre del experimento")
    parser.add_argument("--epochs", type=int, default=None, help="Numero maximo de epocas")
    parser.add_argument("--batch-size", type=int, default=None, help="Tamano de lote")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    parser.add_argument("--no-wandb", action="store_true", help="Desactiva el tracking de W&B")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"], help="Modo de W&B")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=None,
        help="Segundos maximos de entrenamiento (requisito de la competicion)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = parse_overrides(args.overrides)
    if args.name:
        overrides["name"] = args.name
    if args.epochs is not None:
        overrides["optim.max_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["data.batch_size"] = args.batch_size
    if args.device:
        overrides["device"] = args.device
    if args.time_budget is not None:
        overrides["time_budget_seconds"] = args.time_budget
    if args.no_wandb:
        overrides["wandb.enabled"] = False
    if args.wandb_mode:
        overrides["wandb.enabled"] = args.wandb_mode != "disabled"
        overrides["wandb.mode"] = args.wandb_mode

    config = load_config(args.config, overrides)
    experiment = Experiment(config, use_wandb=not args.no_wandb)
    result, artifacts = experiment.run()

    print("\n" + "=" * 60)
    print(result.summary())
    print(f"Artefactos: {artifacts.run_dir}")
    print(f"Checkpoint: {artifacts.checkpoint}")
    test_metrics = getattr(experiment, "test_metrics", None)
    if test_metrics:
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
