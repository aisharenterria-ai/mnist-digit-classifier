# mnist-digit-classifier

Clasificador de digitos manuscritos (MNIST) en PyTorch con tracking en Weights & Biases.

**99.29% de accuracy en el test set** · 54 tests · arquitectura por clases

<!-- Badges -->
[![CI](https://github.com/aisharenterria-ai/mnist-digit-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/aisharenterria-ai/mnist-digit-classifier/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Ejecucion en un clic

### En el navegador (sin instalar nada)

<div align="center">
  <a href="https://colab.research.google.com/github/aisharenterria-ai/mnist-digit-classifier/blob/main/notebooks/01_mnist_entrenamiento.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" width="150" alt="Abrir en Colab">
  </a>
</div>

El enlace de arriba abre el notebook directamente en Google Colab con el repositorio ya montado. La primera celda se encarga de instalar las dependencias.

### En tu maquina

```bash
git clone https://github.com/aisharenterria-ai/mnist-digit-classifier.git
cd mnist-digit-classifier

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py --config configs/quick.yaml   # ~30 s: verifica que todo funciona
```

### Verificacion rapida (sin entrenar)

```bash
python -m pytest tests/ -q      # 54 tests
python -m ruff check src        # lint
```

### Entrenamiento completo

```bash
python scripts/train.py --config configs/full.yaml --epochs 15
```

---

## Que hace este proyecto

Toma una imagen en blanco y negro de **28×28 pixeles** y devuelve la probabilidad de que sea cada uno de los 10 digitos (0-9). La prediccion es la de mayor probabilidad.

**Casos de uso reales:**

- **Lectura automatica de cheques y formularios** (OCR especializado).
- **Clasificacion de codigos postales manuscritos** en sistemas de correo.
- **Control de acceso** con padlocks de botones o pulseras de PIN.
- **Diagnostico medico** al interpretar recetas manuscritas.
- **Juegos y educacion** que reconozcan digitos dibujados a mano.

---

## Resultados

Entrenamiento real verificado en Apple Silicon (MPS), `configs/full.yaml`, 8 epocas, 62 segundos:

| Modelo | Parametros | Test accuracy | Macro F1 | Tiempo |
|---|---:|---:|---:|---:|
| `SimpleCNN` (principal) | 270,722 | **99.29%** | 0.9928 | 62 s |
| `ResNetMNIST` | 174,970 | 99.36% | 0.9936 | 35 s (3 ep) |
| `MLP` (baseline) | 537,354 | 97.38% | 0.9736 | 12 s (3 ep) |

Metricas del modelo principal sobre las 10,000 imagenes del test set:

```json
{
  "accuracy": 0.9929,
  "macro_precision": 0.99298,
  "macro_recall": 0.99279,
  "macro_f1": 0.99288,
  "loss": 0.0197
}
```

**La CNN supera al MLP usando 5× menos parametros**: la estructura espacial de la imagen es informacion aprovechable, y una red densa tiene que reconstruirla desde cero.

---

## Como funciona

### 1. Flujo de datos

```
MNIST (60,000 train + 10,000 test)
   │
   ├─ split estratificado 90/10  →  train / validacion
   │     (cada digito mantiene su proporcion)
   │
   ├─ train:      ToTensor → RandomAffine (±10°) → Normalize(0.1307, 0.3081)
   └─ validacion: ToTensor → Normalize   (determinista, sin augmentation)
```

La normalizacion usa la media y desviacion estandar reales de MNIST en lugar de dividir entre 255: centra los pixeles en cero y aporta ~0.3 puntos mas de accuracy. El augmentation es **conservador** a proposito (MNIST ya viene alineado; rotaciones agresivas degradan el rendimiento).

### 2. Arquitectura

```
Entrada (1, 28, 28)
   │
   ├─ Block 1: Conv(1→32, 3×3)  + BN + ReLU + MaxPool          → (32, 14, 14)
   ├─ Block 2: Conv(32→32, 3×3) + BN + ReLU + MaxPool + Drop   → (32, 7, 7)
   ├─ Block 3: Conv(32→64, 3×3) + BN + ReLU                    → (64, 7, 7)
   ├─ Block 4: Conv(64→64, 3×3) + BN + ReLU + MaxPool + Drop   → (64, 3, 3)
   ├─ AdaptiveAvgPool2d((1,1))                                 → (64, 1, 1)
   ├─ Flatten → Linear(64→128) + ReLU + Dropout
   └─ Linear(128→10)                                           → 10 logits
```

| Componente | Razon |
|---|---|
| **Conv layers** | Detectan bordes, curvas y angulos; un MLP tendria que aprender esas formas desde cero. |
| **BatchNorm** | Acelera la convergencia y estabiliza el entrenamiento. Se desactiva solo en inferencia. |
| **MaxPool** | Reduce la resolucion 2× manteniendo las activaciones mas fuertes. ~75% menos computo. |
| **Dropout (0.25)** | Apaga neuronas al azar e impide que la red memorice imagenes concretas. |
| **AdaptiveAvgPool** | Hace el modelo invariante a desplazamientos pequenos del trazo. |
| **Sin softmax en el forward** | `CrossEntropyLoss` ya aplica log-softmax internamente; ponerlo dos veces es un error clasico. |

### 3. Entrenamiento

| Decision | Valor | Motivo |
|---|---|---|
| Optimizador | **AdamW** (lr=1e-3, wd=1e-4) | Desacopla el decaido del gradiente; generaliza mejor que Adam. |
| Scheduler | **OneCycleLR** (coseno) | Calienta el lr y lo baja en coseno: mejor minimo que un lr fijo. |
| Epocas | 8–15 | MNIST converge rapido; mas epocas empiezan a sobreajustar. |
| Gradient clipping | 5.0 | Evita actualizaciones descontroladas ante batches atipicos. |
| AMP | Activa en CUDA | ~2× mas rapido en GPU sin perder calidad. |

Se guarda el checkpoint con **mejor accuracy de validacion**, no el ultimo: es la unica forma honesta de estimar el rendimiento en test.

### 4. Metricas

Ademas del accuracy, el proyecto calcula **precision, recall y F1 por digito** y la **matriz de confusion** normalizada. Las confusiones tipicas (`4→9`, `3→5`, `7→2`, `8→9`) son semanticamente razonables: un 9 con el borde superior cerrado parece un 8. Corregirlas exigiria datos que el dataset no contiene.

---

## Estructura

```
mnist-digit-classifier/
├── configs/                          # Configuraciones YAML
│   ├── default.yaml                  # Entrenamiento estandar
│   ├── quick.yaml                    # Smoke test en CPU (~30 s)
│   └── full.yaml                     # Entrenamiento completo con W&B
├── notebooks/
│   └── 01_mnist_entrenamiento.ipynb  # Entrega: proceso completo documentado
├── scripts/
│   ├── train.py                      # Entrenar desde terminal
│   ├── evaluate.py                   # Evaluar un checkpoint
│   ├── predict.py                    # Generar submission.csv
│   └── compare_models.py             # Benchmark de arquitecturas
├── src/digitclf/
│   ├── config.py                     # Dataclasses validadas + YAML + overrides
│   ├── pipeline.py                   # Orquestacion config → datos → modelo → trainer
│   ├── viz.py                        # Figuras para reports y W&B
│   ├── data/
│   │   ├── datamodule.py             # MNISTDataModule: descarga, split, loaders
│   │   └── transforms.py             # Cadena de transformaciones
│   ├── models/
│   │   ├── simple_cnn.py             # CNN principal (~270k params)
│   │   ├── mlp.py                    # Baseline red densa
│   │   ├── resnet_mnist.py           # ResNet ligera (~175k params)
│   │   └── __init__.py               # Registro: build_model("simple_cnn")
│   ├── engine/
│   │   ├── trainer.py                # Trainer: fit, AMP, schedulers, early stopping
│   │   ├── metrics.py                # ClassificationMetrics (sin sklearn)
│   │   └── checkpoint.py             # Guardado/carga de pesos
│   ├── tracking/
│   │   └── wandb_tracker.py          # W&B con fallback offline automatico
│   ├── inference/
│   │   └── predictor.py              # DigitPredictor para imagenes sueltas
│   └── utils/
│       ├── seed.py                   # Reproducibilidad
│       ├── device.py                 # CPU / CUDA / MPS automatico
│       └── logging.py                # Logging a consola y archivo
└── tests/                            # 54 tests
    ├── test_config.py                # Validacion, YAML, overrides
    ├── test_models.py                # Formas de tensor, param count
    ├── test_metrics.py               # Accuracy, macro F1, confusion matrix
    ├── test_trainer.py               # Entrenamiento con datos sinteticos
    ├── test_transforms.py            # Cadenas de transformacion
    └── test_checkpoint_inference.py  # Guardado/carga y prediccion
```

---

## Comandos

```bash
# Entrenar
python scripts/train.py                                     # config por defecto
python scripts/train.py --config configs/quick.yaml         # rapido (~30 s, CPU)
python scripts/train.py --config configs/full.yaml          # completo con W&B
python scripts/train.py --set optim.lr=3e-3 model.name=resnet_mnist
python scripts/train.py --time-budget 1800                  # limite de tiempo

# Evaluar
python scripts/evaluate.py runs/mnist_simplecnn/best.pt --save-figure confusion.png

# Predecir
python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --split test --output submission.csv
python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --images foto1.png foto2.png

# Comparar arquitecturas
python scripts/compare_models.py --epochs 5

# Tests
python -m pytest tests/ -v
```

---

## API de Python

```python
from digitclf.config import load_config
from digitclf.pipeline import Experiment
from digitclf.inference import DigitPredictor

# Entrenar
config = load_config("configs/full.yaml")
result, artifacts = Experiment(config).run()
print(result.summary())

# Cambiar cualquier parametro sin editar codigo
config = config.replace(**{"optim.lr": 3e-3, "model.name": "resnet_mnist"})

# Inferir
predictor = DigitPredictor.from_checkpoint(artifacts.checkpoint)
label, confidence = predictor.predict_with_confidence("mi_digito.png")
```

---

## Weights & Biases

1. Crear cuenta gratis en [wandb.ai](https://wandb.ai)
2. Copiar la API key de [wandb.ai/authorize](https://wandb.ai/authorize)
3. Pegarla en la celda 2 del notebook: `wandb.login("tu_api_key")`

**Sin cuenta el proyecto funciona igual.** `WandbTracker` detecta que no hay `WANDB_API_KEY` y cae automaticamente en **modo offline**: los runs se guardan en `./wandb/` y se sincronizan despues con:

```bash
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-<id>
```

Se registra: configuracion completa, loss y accuracy por epoca, learning rate, metricas por digito, curvas de aprendizaje y matriz de confusion.

---

## Reproducibilidad

`seed_everything()` fija la semilla en Python, NumPy y PyTorch y activa los algoritmos deterministas de cuDNN. La division train/validacion es **estratificada y determinista**.

Esto importa mas de lo que parece: sin estratificacion el modelo puede heredar un validacion con mas 3s que 8s, aprender a sobrerrepresentar el 3 y reportar un accuracy enganoso. El F1 macro existe justamente para detectar ese sesgo.

---

## Buenas practicas aplicadas

- **Separacion de responsabilidades**: `config`, `data`, `models`, `engine`, `tracking` e `inference` no se importan de forma circular. El `Trainer` no sabe que existe MNIST.
- **Configuracion declarativa**: dataclasses validadas; cambiar de modelo u optimizador es una linea, no una edicion de codigo.
- **Registro de modelos**: `build_model("simple_cnn", width=64)` permite anadir arquitecturas sin tocar el trainer.
- **Inyeccion de dependencias**: el tracker se pasa al trainer; sin W&B no hay nada que cambiar.
- **Degradacion elegante**: sin API key, sin GPU o sin `wandb` instalado, el proyecto sigue funcionando.
- **54 tests** sobre metricas, checkpointing, formas de tensor y validacion de config; corren en segundos con datos sinteticos.
- **CI en GitHub Actions**: lint + tests + entrenamiento smoke en cada push.
- **Artefactos por corrida**: `config.yaml`, `history.json`, `metrics.json`, `best.pt` y figuras.

---

## Licencia

MIT. Dataset MNIST distribuido por Yann LeCun et al. bajo [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
