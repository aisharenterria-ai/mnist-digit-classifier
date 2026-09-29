"""Configuracion declarativa del experimento.

Toda la configuracion vive en dataclasses: asi el experimento es reproducible,
validable y serializable a YAML/JSON para W&B.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml

T = TypeVar("T")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class DataConfig:
    """Parametros de datos y aumento."""

    root: str = "data"
    num_classes: int = 10
    image_size: int = 28
    val_split: float = 0.1
    batch_size: int = 128
    num_workers: int = 0
    pin_memory: bool = True
    download: bool = True
    normalize_mean: Sequence[float] = field(default_factory=lambda: (0.1307,))
    normalize_std: Sequence[float] = field(default_factory=lambda: (0.3081,))
    augment: bool = True
    random_affine_degrees: float = 10.0
    random_affine_translate: float = 0.1
    random_affine_scale: float = 0.1
    random_erasing_p: float = 0.0
    csv_train_path: str | None = None
    csv_test_path: str | None = None

    def __post_init__(self) -> None:
        # Normalizar a tupla para que el roundtrip por YAML sea estable
        self.normalize_mean = tuple(float(v) for v in self.normalize_mean)
        self.normalize_std = tuple(float(v) for v in self.normalize_std)
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split debe estar en [0, 1): {self.val_split}")
        if self.batch_size < 1:
            raise ValueError("batch_size debe ser >= 1")
        if len(self.normalize_mean) != 1 or len(self.normalize_std) != 1:
            raise ValueError("Por ahora se espera un valor por canal (imagenes grayscale)")
        if self.num_workers < 0:
            raise ValueError("num_workers debe ser >= 0")


@dataclass
class ModelConfig:
    """Arquitectura del modelo."""

    name: str = "simple_cnn"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("El nombre del modelo no puede estar vacio")


@dataclass
class OptimConfig:
    """Optimizador, learning rate y regularizacion."""

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: Sequence[float] = field(default_factory=lambda: (0.9, 0.999))
    scheduler: str = "onecycle"
    max_epochs: int = 10
    steps_per_epoch: int | None = None
    pct_start: float = 0.3
    grad_clip: float | None = 5.0
    label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        self.betas = tuple(float(b) for b in self.betas)
        if self.lr <= 0:
            raise ValueError(f"lr debe ser > 0: {self.lr}")
        if len(self.betas) != 2:
            raise ValueError("betas debe tener exactamente 2 valores")
        if self.max_epochs < 1:
            raise ValueError("max_epochs debe ser >= 1")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing debe estar en [0, 1)")
        if self.scheduler == "onecycle" and not 0.0 < self.pct_start <= 1.0:
            raise ValueError("pct_start debe estar en (0, 1]")


@dataclass
class WandbConfig:
    """Integracion con Weights & Biases."""

    enabled: bool = True
    project: str = "mnist-digit-classifier"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=lambda: ["mnist", "pytorch"])
    group: str | None = None
    mode: str | None = None
    log_model_artifact: bool = True
    log_sample_grid: bool = True

    def __post_init__(self) -> None:
        if self.mode is not None and self.mode not in {
            "online",
            "offline",
            "disabled",
            "dryrun",
            "export",
        }:
            raise ValueError(f"mode de W&B no valido: {self.mode}")


@dataclass
class ExperimentConfig:
    """Configuracion raiz del experimento."""

    name: str = "mnist_baseline"
    seed: int = 42
    deterministic: bool = True
    device: str = "auto"
    output_dir: str = "runs"
    save_checkpoint: bool = True
    time_budget_seconds: float | None = None
    early_stopping_patience: int | None = None
    log_every_n_steps: int = 50
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed debe ser >= 0")
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds debe ser > 0")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience debe ser >= 1")
        for section in ("data", "model", "optim", "wandb"):
            if not is_dataclass(getattr(self, section)):
                raise TypeError(f"La seccion '{section}' debe ser un dataclass")

    @property
    def run_dir(self) -> Path:
        """Directorio de salida del experimento."""
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        """Convierte la configuracion a diccionario plano."""
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        """Guarda la configuracion en YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path

    def replace(self, **updates: Any) -> ExperimentConfig:
        """Devuelve una copia con ``updates`` aplicados (soporta notacion punteada)."""
        return apply_overrides(self, updates)

    def summary(self) -> str:
        """Resumen legible de la configuracion para la consola."""
        d = self.to_dict()
        lines = [
            f"Experimento : {d['name']}",
            f"Modelo      : {d['model']['name']} {d['model']['params'] or ''}".rstrip(),
            f"Optimizador : {d['optim']['name']} lr={d['optim']['lr']} "
            f"epochs={d['optim']['max_epochs']} sched={d['optim']['scheduler']}",
            f"Datos       : batch={d['data']['batch_size']} val_split={d['data']['val_split']} "
            f"augment={d['data']['augment']}",
            f"W&B         : enabled={d['wandb']['enabled']} project={d['wandb']['project']}",
            f"Dispositivo : {self.device}",
        ]
        return "\n".join(lines)


def _merge_into(instance: T, updates: Mapping[str, Any]) -> T:
    """Aplica overrides en notacion punteada sobre una instancia dataclass."""
    for dotted_key, value in updates.items():
        if value is None:
            continue
        parts = dotted_key.split(".")
        target: Any = instance
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"Seccion inexistente en la config: '{part}'")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise KeyError(f"Parametro inexistente en la config: '{dotted_key}'")
        current = getattr(target, leaf)
        setattr(target, leaf, _coerce_like(current, value))
    return instance


def _coerce_like(current: Any, value: Any) -> Any:
    """Convierte ``value`` al tipo de ``current`` para no romper dataclasses tipados."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)
    if isinstance(current, list) and not isinstance(value, (list, tuple)):
        return [v.strip() for v in str(value).split(",") if v.strip()]
    if isinstance(current, tuple) and not isinstance(value, (list, tuple)):
        value = [v.strip() for v in str(value).split(",") if v.strip()]
    origin = get_origin(type(current))
    if origin in (list, tuple, set):
        args = [a for a in get_args(type(current)) if a is not type(None)]
        if args and isinstance(args[0], type) and isinstance(value, (list, tuple)):
            return type(current)(args[0](v) for v in value) if origin is tuple else value
        return value
    if isinstance(current, float):
        return float(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    return value


def apply_overrides(instance: T, updates: Mapping[str, Any]) -> T:
    """Copia profunda de ``instance`` con los overrides aplicados."""
    clone = copy.deepcopy(instance)
    _merge_into(clone, updates)
    _revalidate(clone)
    return clone


def _revalidate(instance: Any) -> None:
    """Re-ejecuta ``__post_init__`` en cada dataclass tras aplicar overrides."""
    if not is_dataclass(instance):
        return
    for f in fields(instance):
        value = getattr(instance, f.name)
        if is_dataclass(value):
            _revalidate(value)
    post_init = getattr(instance, "__post_init__", None)
    if callable(post_init):
        post_init()


def _build(cls: type[T], data: Mapping[str, Any]) -> T:
    """Construye un dataclass validando nombres desconocidos."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Parametros desconocidos para {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    """Construye una :class:`ExperimentConfig` desde un diccionario (parcial o completo)."""
    data = dict(data or {})
    data.pop("name_version", None)
    root_fields = {f.name for f in fields(ExperimentConfig)}
    kwargs: dict[str, Any] = {k: v for k, v in data.items() if k in root_fields}
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "optim": OptimConfig,
        "wandb": WandbConfig,
    }
    for key, section_cls in sections.items():
        section_data = data.get(key) or {}
        base = kwargs.get(key) or {}
        merged = {**base, **section_data}
        kwargs[key] = _build(section_cls, merged)
    return ExperimentConfig(**kwargs)


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Carga la configuracion desde YAML y aplica overrides.

    Args:
        path: ruta a un YAML. Si es ``None`` se usa ``configs/default.yaml``.
        overrides: overrides en notacion punteada, p.ej. ``{"optim.lr": 3e-3}``.

    Returns:
        Configuracion validada.
    """
    if path is None:
        path = CONFIGS_DIR / "default.yaml"
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
    config = config_from_dict(raw)
    if overrides:
        config = apply_overrides(config, overrides)
    return config


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Convierte ``["optim.lr=0.01", "data.augment=false"]`` en diccionario."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override invalido (se espera clave=valor): '{pair}'")
        key, raw_value = pair.split("=", 1)
        overrides[key.strip()] = raw_value.strip()
    return overrides
