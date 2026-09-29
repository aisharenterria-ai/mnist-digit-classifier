"""Pipeline de alto nivel: une config -> datos -> modelo -> trainer -> artefactos.

Los scripts y el notebook usan este unico punto de entrada, de forma que
cualquier cambio en la orquestacion no obliga a tocar cada consumidor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from digitclf.config import ExperimentConfig
from digitclf.data import MNISTDataModule
from digitclf.engine.checkpoint import save_checkpoint
from digitclf.engine.trainer import Trainer, TrainingResult
from digitclf.models import build_model, count_parameters
from digitclf.tracking import WandbTracker
from digitclf.utils.device import describe_device, get_device
from digitclf.utils.logging import get_logger, setup_logging
from digitclf.utils.seed import seed_everything
from digitclf.viz import plot_confusion_matrix, plot_sample_grid, plot_training_history

LOGGER = get_logger(__name__)


@dataclass
class ExperimentArtifacts:
    """Rutas generadas por una ejecucion."""

    run_dir: Path
    checkpoint: Path
    history_json: Path
    metrics_json: Path
    confusion_png: Path | None = None
    history_png: Path | None = None


class Experiment:
    """Ejecuta un experimento completo y deja los artefactos en disco.

    Uso minimo::

        exp = Experiment(ExperimentConfig(name="demo"))
        result = exp.run()
        print(result.summary())
    """

    def __init__(self, config: ExperimentConfig, use_wandb: bool = True) -> None:
        self.config = config
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_file=self.run_dir / "train.log", force=True)
        seed_everything(config.seed, config.deterministic)

        self.device = get_device(config.device)
        self.data = MNISTDataModule(config.data, seed=config.seed)
        self.bundle = None
        self.model = build_model(config.model.name, **config.model.params)
        self.tracker = WandbTracker(config.wandb, config.to_dict()) if use_wandb else None

    def run(self) -> tuple[TrainingResult, ExperimentArtifacts]:
        """Entrena, evalua y persiste todos los artefactos."""
        LOGGER.info("\n%s\n%s", self.config.summary(), "-" * 60)

        self.bundle = self.data.build()
        self._log_dataset_summary()

        trainer = Trainer(
            model=self.model,
            dataloaders=self.bundle,
            optim_cfg=self.config.optim,
            device=self.device,
            tracker=self.tracker,
            checkpoint_dir=self.run_dir,
            use_amp=True,
            log_every_n_steps=self.config.log_every_n_steps,
            early_stopping_patience=self.config.early_stopping_patience,
            time_budget_seconds=self.config.time_budget_seconds,
            save_checkpoint=self.config.save_checkpoint,
        )

        if self.tracker and self.config.wandb.log_sample_grid:
            images, labels = next(iter(self.bundle.train))
            self.tracker.log_figure(plot_sample_grid(images, labels, n=16), "Muestras de entrenamiento")

        result = trainer.fit()
        artifacts = self._persist(result, trainer)

        if self.tracker:
            self.tracker.log_metrics({f"final/{k}": v for k, v in result.final_metrics.items()})
            self.tracker.log_figure(plot_training_history(result.history, "accuracy"), "curva_accuracy")
            if artifacts.confusion_png:
                self.tracker.log_figure(artifacts.confusion_png, "matriz_confusion")

        self._log_final_metrics(result)
        return result, artifacts

    # ------------------------------------------------------------------ artefactos
    def _log_dataset_summary(self) -> None:
        sizes = self.bundle.sizes()
        LOGGER.info(
            "Dataset | %s | parametros: %s | dispositivo: %s",
            sizes,
            f"{count_parameters(self.model):,}",
            describe_device(self.device),
        )

    def _persist(self, result: TrainingResult, trainer: Trainer) -> ExperimentArtifacts:
        history_json = self.run_dir / "history.json"
        metrics_json = self.run_dir / "metrics.json"
        history_json.write_text(
            json.dumps([vars(e) for e in result.history], indent=2), encoding="utf-8"
        )
        metrics_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        (self.run_dir / "config.yaml").write_text(
            __import__("yaml").safe_dump(self.config.to_dict(), sort_keys=False), encoding="utf-8"
        )

        val = trainer.evaluate()
        cm = val["_metrics"].confusion_matrix()
        confusion_png = self.run_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, self.bundle.class_names).savefig(confusion_png, dpi=150)
        history_png = self.run_dir / "training_history.png"
        plot_training_history(result.history, "accuracy").savefig(history_png, dpi=150)

        checkpoint = self.run_dir / "best.pt"
        if self.config.save_checkpoint:
            save_checkpoint(
                checkpoint,
                trainer.model,
                trainer.optimizer,
                epoch=result.best_epoch,
                metrics=result.final_metrics,
                config=self.config.to_dict(),
            )

        return ExperimentArtifacts(
            run_dir=self.run_dir,
            checkpoint=checkpoint,
            history_json=history_json,
            metrics_json=metrics_json,
            confusion_png=confusion_png,
            history_png=history_png,
        )

    def _log_final_metrics(self, result: TrainingResult) -> None:
        LOGGER.info("\n%s", result.summary())
        if self.bundle.test is not None:
            test_metrics = self._evaluate_test(result)
            if test_metrics is not None:
                LOGGER.info("TEST FINAL: %s", test_metrics)
        if self.tracker:
            self.tracker.log_metrics({f"final/{k}": v for k, v in result.final_metrics.items()})

    def _evaluate_test(self, result: TrainingResult) -> dict[str, float] | None:
        checkpoint = self.run_dir / "best.pt"
        if not checkpoint.exists() or self.bundle is None:
            return None
        from digitclf.engine.checkpoint import load_checkpoint

        load_checkpoint(checkpoint, self.model, map_location=self.device)
        trainer = Trainer(
            model=self.model,
            dataloaders=self.bundle,
            optim_cfg=self.config.optim,
            device=self.device,
        )
        test_metrics = trainer.evaluate(self.bundle.test)
        self.test_metrics: dict[str, Any] = {k: v for k, v in test_metrics.items() if not k.startswith("_")}
        return self.test_metrics
