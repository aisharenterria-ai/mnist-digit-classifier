"""Logging limpio a consola y a fichero."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%H:%M:%S"
_CONFIGURED = False


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configura el logging raiz del proyecto.

    Args:
        level: nivel de log (``logging.DEBUG`` o ``"INFO"``).
        log_file: si se indica, tambien escribe en ese fichero.
        force: reconfigura aunque ya exista una configuracion previa.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATEFMT,
        handlers=handlers,
        force=True,
    )
    logging.getLogger("PIL").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger con el nombre del modulo."""
    return logging.getLogger(name)
