"""digitclf: paquete de clasificacion de digitos manuscritos (MNIST) en PyTorch.

El paquete separa cada responsabilidad en modulos:

- ``config``    : configuracion declarativa (dataclasses + YAML).
- ``data``      : datasets, transformaciones y dataloaders.
- ``models``    : arquitecturas y registro de modelos.
- ``engine``    : trainer, metricas y gestion de checkpoints.
- ``tracking``  : integracion con Weights & Biases (con fallback local).
- ``inference`` : prediccion sobre imagenes crudas y generacion de submission.
- ``utils``     : semilla, device, logging y profileo.
- ``viz``       : figuras para reports y para W&B.
"""

from digitclf.config import ExperimentConfig, load_config
from digitclf.engine.metrics import ClassificationMetrics
from digitclf.engine.trainer import Trainer, TrainingResult
from digitclf.models import build_model
from digitclf.utils.device import get_device
from digitclf.utils.seed import seed_everything

__version__ = "1.0.0"

__all__ = [
    "ClassificationMetrics",
    "ExperimentConfig",
    "TrainingResult",
    "Trainer",
    "build_model",
    "get_device",
    "load_config",
    "seed_everything",
    "__version__",
]
