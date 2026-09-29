"""Entrenador reutilizable: cualquier ``nn.Module`` + dataloaders -> metricas.

El ``Trainer`` no sabe nada de MNIST ni de W&B; recibe un tracker opcional con
una interfaz minima (``log_metrics`` / ``log_figure`` / ``finish``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from digitclf.config import OptimConfig
from digitclf.engine.checkpoint import save_checkpoint
from digitclf.engine.metrics import ClassificationMetrics
from digitclf.utils.device import count_parameters, describe_device, get_device
from digitclf.utils.logging import get_logger

LOGGER = get_logger(__name__)


class MetricLogger(Protocol):
    """Contrato minimo que el Trainer necesita de un tracker."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_figure(self, figure: Any, caption: str) -> None: ...


@dataclass
class EpochLog:
    """Historico de una epoca."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    lr: float
    epoch_seconds: float


@dataclass
class TrainingResult:
    """Resultado completo de un entrenamiento."""

    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    final_metrics: dict[str, float] = field(default_factory=dict)
    history: list[EpochLog] = field(default_factory=list)
    checkpoint_path: Path | None = None
    stopped_early: bool = False
    stop_reason: str = "max_epochs"
    total_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Version serializable del resultado."""
        return {
            "best_epoch": self.best_epoch,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_macro_f1": self.best_val_macro_f1,
            "final_metrics": self.final_metrics,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "total_seconds": self.total_seconds,
            "history": [vars(e) for e in self.history],
        }

    def summary(self) -> str:
        """Resumen legible para la consola."""
        return (
            f"Mejor epoca: {self.best_epoch} | "
            f"val_acc: {self.best_val_accuracy:.4f} | "
            f"val_macro_f1: {self.best_val_macro_f1:.4f} | "
            f"motivo de parada: {self.stop_reason}"
        )


class Trainer:
    """Bucle de entrenamiento con AMP, scheduler, early stopping y time budget.

    Args:
        model: red neuronal a entrenar.
        dataloaders: ``DataBundle`` con ``train`` y ``val``.
        optim_cfg: optimizador, scheduler y presupuesto de epochs.
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ...
        tracker: logger de metricas (W&B o nulo).
        checkpoint_dir: donde guardar el mejor checkpoint.
        use_amp: activa mixed precision en CUDA.
        log_every_n_steps: frecuencia de log por step.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Any,
        optim_cfg: OptimConfig,
        device: str | torch.device = "auto",
        tracker: MetricLogger | None = None,
        checkpoint_dir: str | Path | None = None,
        use_amp: bool = True,
        log_every_n_steps: int = 50,
        early_stopping_patience: int | None = None,
        time_budget_seconds: float | None = None,
        save_checkpoint: bool = True,
    ) -> None:
        if log_every_n_steps < 1:
            raise ValueError("log_every_n_steps debe ser >= 1")
        if time_budget_seconds is not None and time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds debe ser > 0")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience debe ser >= 1")

        self.cfg = optim_cfg
        self.device = get_device(device)
        self.model = model.to(self.device)
        self.dataloaders = dataloaders
        self.num_classes = getattr(dataloaders, "num_classes", 10)
        self.class_names = getattr(dataloaders, "class_names", None)
        self.tracker = tracker
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.save_checkpoint = save_checkpoint and self.checkpoint_dir is not None
        self.log_every_n_steps = log_every_n_steps
        self.early_stopping_patience = early_stopping_patience
        self.time_budget_seconds = time_budget_seconds

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self._scheduler_name = (self.cfg.scheduler or "none").lower()
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history: list[EpochLog] = []
        self.best_epoch = -1
        self.best_val_accuracy = 0.0
        self.best_val_metrics: dict[str, float] = {}
        self._best_state: dict[str, torch.Tensor] | None = None
        self._global_step = 0
        self._epochs_without_improvement = 0

        LOGGER.info(
            "Trainer listo en %s | parametros entrenables: %s",
            describe_device(self.device),
            f"{count_parameters(self.model):,}",
        )

    # ------------------------------------------------------------------ componentes
    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        name = self.cfg.name.lower()
        common = {"lr": self.cfg.lr, "weight_decay": self.cfg.weight_decay}
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), momentum=self.cfg.momentum, **common)
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), betas=tuple(self.cfg.betas), **common)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), betas=tuple(self.cfg.betas), **common)
        if name == "rmsprop":
            return torch.optim.RMSprop(self.model.parameters(), momentum=self.cfg.momentum, **common)
        raise ValueError(f"Optimizador no soportado: {self.cfg.name}")

    def _build_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Crea el scheduler segun los epochs reales de esta ejecucion."""
        name = (self.cfg.scheduler or "none").lower()
        if name in {"none", "null", ""}:
            return None
        if name == "onecycle":
            steps = self.cfg.steps_per_epoch or len(self.dataloaders.train)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.lr,
                epochs=epochs,
                steps_per_epoch=steps,
                pct_start=self.cfg.pct_start,
                anneal_strategy="cos",
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, epochs // 3), gamma=0.1
            )
        raise ValueError(f"Scheduler no soportado: {self.cfg.scheduler}")

    # ------------------------------------------------------------------ loops
    def train_one_epoch(self, epoch: int) -> EpochLog:
        """Ejecuta una epoca de entrenamiento."""
        self.model.train()
        metrics = ClassificationMetrics(self.num_classes, self.class_names)
        running_loss = 0.0
        start = time.perf_counter()

        for images, targets in self.dataloaders.train:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if targets.ndim > 1:
                targets = targets.argmax(dim=1)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._step_scheduler_per_batch()

            running_loss += float(loss.item()) * targets.size(0)
            metrics.update(logits.detach().argmax(dim=1), targets)
            self._global_step += 1

            if self.tracker and self._global_step % self.log_every_n_steps == 0:
                self.tracker.log_metrics(
                    {
                        "train/batch_loss": float(loss.item()),
                        "train/batch_accuracy": float(
                            (logits.detach().argmax(1) == targets).float().mean().item()
                        ),
                        "train/lr": float(self.optimizer.param_groups[0]["lr"]),
                    },
                    step=self._global_step,
                )

        self._step_scheduler_per_epoch()

        seconds = time.perf_counter() - start
        return EpochLog(
            epoch=epoch,
            train_loss=running_loss / max(1, metrics.total),
            train_accuracy=metrics.accuracy,
            val_loss=0.0,
            val_accuracy=0.0,
            val_macro_f1=0.0,
            lr=float(self.optimizer.param_groups[0]["lr"]),
            epoch_seconds=seconds,
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable | None = None) -> dict[str, float]:
        """Evalua el modelo y devuelve metricas + perdida media."""
        loader = loader or self.dataloaders.val
        self.model.eval()
        metrics = ClassificationMetrics(self.num_classes, self.class_names)
        total_loss = 0.0
        total = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if targets.ndim > 1:
                targets = targets.argmax(dim=1)

            with torch.autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            total_loss += float(loss.item()) * targets.size(0)
            total += targets.size(0)
            metrics.update(logits.argmax(dim=1), targets)

        out = metrics.summary()
        out["loss"] = total_loss / max(1, total)
        out["_metrics"] = metrics  # type: ignore[assignment]
        return out

    def _step_scheduler_per_batch(self) -> None:
        """OneCycleLR avanza dentro de la epoca; el resto, entre epocas."""
        if self.scheduler is not None and self._scheduler_name == "onecycle":
            self.scheduler.step()

    def _step_scheduler_per_epoch(self) -> None:
        if self.scheduler is not None and self._scheduler_name != "onecycle":
            self.scheduler.step()

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        epochs: int | None = None,
        on_epoch_end: Callable[[EpochLog, dict[str, float]], None] | None = None,
    ) -> TrainingResult:
        """Entrena el modelo y devuelve el mejor resultado.

        Para por epochs, early stopping o ``time_budget_seconds`` (util cuando
        la competicion impone un limite de tiempo de ejecucion).
        """
        total_epochs = epochs or self.cfg.max_epochs
        self.scheduler = self._build_scheduler(total_epochs)
        start_time = time.perf_counter()
        stop_reason = "max_epochs"
        stopped_early = False

        for epoch in range(1, total_epochs + 1):
            epoch_log = self.train_one_epoch(epoch)
            val_metrics = self.evaluate()
            val_metrics_clean = {k: v for k, v in val_metrics.items() if not k.startswith("_")}
            epoch_log.val_loss = val_metrics_clean["loss"]
            epoch_log.val_accuracy = val_metrics_clean["accuracy"]
            epoch_log.val_macro_f1 = val_metrics_clean["macro_f1"]
            self.history.append(epoch_log)

            improved = epoch_log.val_accuracy > self.best_val_accuracy
            if improved:
                self.best_val_accuracy = epoch_log.val_accuracy
                self.best_val_metrics = val_metrics_clean
                self.best_epoch = epoch
                self._epochs_without_improvement = 0
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
                if self.save_checkpoint and self.checkpoint_dir is not None:
                    save_checkpoint(
                        self.checkpoint_dir / "best.pt",
                        self.model,
                        self.optimizer,
                        epoch=epoch,
                        metrics=val_metrics_clean,
                    )
            else:
                self._epochs_without_improvement += 1

            LOGGER.info(
                "epoch %02d/%02d | train_loss %.4f | train_acc %.4f | val_loss %.4f | "
                "val_acc %.4f | macro_f1 %.4f | %.1fs",
                epoch,
                total_epochs,
                epoch_log.train_loss,
                epoch_log.train_accuracy,
                epoch_log.val_loss,
                epoch_log.val_accuracy,
                epoch_log.val_macro_f1,
                epoch_log.epoch_seconds,
            )

            if self.tracker:
                self.tracker.log_metrics(
                    {
                        "train/loss": epoch_log.train_loss,
                        "train/accuracy": epoch_log.train_accuracy,
                        "val/loss": epoch_log.val_loss,
                        "val/accuracy": epoch_log.val_accuracy,
                        "val/macro_f1": epoch_log.val_macro_f1,
                        "train/lr": epoch_log.lr,
                        "train/epoch_seconds": epoch_log.epoch_seconds,
                    },
                    step=epoch,
                )

            if on_epoch_end is not None:
                on_epoch_end(epoch_log, val_metrics_clean)

            if self._time_budget_exceeded(start_time):
                stop_reason = "time_budget"
                stopped_early = total_epochs > epoch
                LOGGER.warning("Presupuesto de tiempo agotado en la epoca %d", epoch)
                break

            if (
                self.early_stopping_patience is not None
                and self._epochs_without_improvement >= self.early_stopping_patience
            ):
                stop_reason = "early_stopping"
                stopped_early = total_epochs > epoch
                LOGGER.warning(
                    "Early stopping en la epoca %d (sin mejora en %d epocas)",
                    epoch,
                    self._epochs_without_improvement,
                )
                break

        self._restore_best()
        checkpoint_path = None
        if self.save_checkpoint and self.checkpoint_dir is not None:
            checkpoint_path = self.checkpoint_dir / "best.pt"

        return TrainingResult(
            best_epoch=self.best_epoch,
            best_val_accuracy=self.best_val_accuracy,
            best_val_macro_f1=self.best_val_metrics.get("macro_f1", 0.0),
            final_metrics=self.best_val_metrics,
            history=self.history,
            checkpoint_path=checkpoint_path,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
            total_seconds=time.perf_counter() - start_time,
        )

    def _time_budget_exceeded(self, start_time: float) -> bool:
        if self.time_budget_seconds is None:
            return False
        return (time.perf_counter() - start_time) >= self.time_budget_seconds

    def _restore_best(self) -> None:
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            LOGGER.info("Pesos del mejor modelo restaurados (epoca %d)", self.best_epoch)

    @property
    def metrics(self) -> ClassificationMetrics:
        """Metricas de la ultima evaluacion."""
        val = self.evaluate()
        return val["_metrics"]
