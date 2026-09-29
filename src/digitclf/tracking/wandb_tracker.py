"""Tracker de Weights & Biases con degradacion elegante.

Si no hay API key o no hay red, cae automaticamente a ``offline`` (guarda los
runs en disco, sincronizables despues con ``wandb sync``) y nunca rompe el
entrenamiento. Esto permite trabajar sin cuenta y subir los resultados luego.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from digitclf.config import WandbConfig
from digitclf.utils.logging import get_logger

LOGGER = get_logger(__name__)


class WandbTracker:
    """Envoltura del SDK de W&B con las operaciones que el trainer necesita.

    Implements el protocolo ``MetricLogger``: ``log_metrics`` y ``log_figure``.
    Si ``wandb`` no esta instalado o no hay API key, todos los metodos se
    convierten en no-ops y el entrenamiento continua igual.
    """

    def __init__(self, cfg: WandbConfig, config_dict: dict[str, Any] | None = None) -> None:
        self.cfg = cfg
        self.run: Any = None
        self._enabled = bool(cfg.enabled)
        self._summary: dict[str, float] = {}

        if not self._enabled:
            LOGGER.info("W&B deshabilitado por configuracion")
            return

        try:
            import wandb  # noqa: F401
        except ImportError:
            LOGGER.warning(
                "wandb no esta instalado. Instala con 'pip install wandb' para activar el tracking."
            )
            self._enabled = False
            return

        mode = self._resolve_mode()
        self.run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            name=cfg.run_name,
            tags=cfg.tags,
            group=cfg.group,
            mode=mode,
            config=config_dict or {},
            reinit=True,
        )
        LOGGER.info("W&B inicializado | project=%s | mode=%s | url=%s", cfg.project, mode, self.run.url)

    def _resolve_mode(self) -> str | None:
        if self.cfg.mode:
            return self.cfg.mode
        has_key = bool(os.environ.get("WANDB_API_KEY"))
        if not has_key:
            LOGGER.warning(
                "No se encontro WANDB_API_KEY: usando modo offline. "
                "Los runs quedan en ./wandb/ y se sincronizan con 'wandb sync'."
            )
            return "offline"
        return "online"

    # ------------------------------------------------------------------ logging
    def _is_active(self) -> bool:
        """False tras finish(), para ignorar logs posteriores al cierre del run."""
        return bool(self._enabled and self.run is not None)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Registra metricas escalares en el run."""
        if not self._is_active():
            return
        self._summary.update(metrics)
        self.run.log(metrics, step=step)

    def log_figure(self, figure: Any, caption: str) -> None:
        """Registra una figura matplotlib."""
        if not self._is_active():
            return
        self.run.log({caption: figure})

    def log_table(self, table: list[dict[str, Any]], caption: str) -> None:
        """Registra una tabla (por ejemplo, metricas por clase)."""
        if not self._is_active() or not table:
            return
        import wandb

        columns = list(table[0].keys())
        self.run.log(
            {
                caption: wandb.Table(
                    columns=columns,
                    data=[[row[col] for col in columns] for row in table],
                )
            }
        )

    def log_model(self, artifact_name: str, metadata: dict[str, Any] | None = None) -> None:
        """Sube el checkpoint ``best.pt`` del run como artefacto de W&B."""
        if not self._is_active() or not self.cfg.log_model_artifact:
            return
        import wandb

        checkpoint = Path(str(self.run.dir)) / "files" / "best.pt"
        if not checkpoint.exists():
            LOGGER.warning("No se encontro best.pt para subir como artefacto")
            return
        artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata or {})
        artifact.add_file(str(checkpoint), name="best.pt")
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        """Cierra el run y sincroniza si es posible."""
        if not self._is_active():
            return
        self.run.summary["final"] = self._summary
        self.run.finish()
        self._enabled = False
        LOGGER.info("Run de W&B finalizado")

    def __enter__(self) -> WandbTracker:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish()


class NullTracker:
    """Tracker que no hace nada (util para tests y CPU sin W&B)."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """No-op."""

    def log_figure(self, figure: Any, caption: str) -> None:
        """No-op."""

    def finish(self) -> None:
        """No-op."""
