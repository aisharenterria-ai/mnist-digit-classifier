# MNIST Digit Classifier

**Proyecto final del Bloque 9 — Desarrollo de algoritmos de Machine Learning con PyTorch**

Red neuronal convolucional que reconoce digitos manuscritos (0-9) en imagenes de 28×28 pixeles, con experiment tracking en **Weights & Biases** y una estructura de clases preparada para produccion.

---

## Resultado

Entrenamiento real verificado en este repositorio (Apple Silicon / MPS, `configs/full.yaml`, 8 epocas, 62 segundos):

| Modelo | Parametros | Test accuracy | Macro F1 | Tiempo |
|---|---:|---:|---:|---:|
| `SimpleCNN` (entrega) | 270,722 | **99.29%** | 0.9928 | 62 s |
| `ResNetMNIST` | 174,970 | 99.36% | 0.9936 | 35 s (3 ep) |
| `MLP` (baseline) | 537,354 | 97.38% | 0.9736 | 12 s (3 ep) |

Metricas del modelo entregado sobre las 10,000 imagenes del test set:

```json
{
  "accuracy": 0.9929,
  "macro_precision": 0.99298,
  "macro_recall": 0.99279,
  "macro_f1": 0.99288,
  "loss": 0.0197
}
```

---

## Para que sirve

**El problema.** Un banco recibe fotos de cheques y documentos manuscritos. Cada digitito escrito a mano debe convertirse automaticamente en un numero que el sistema pueda sumar, comparar o buscar. Hacerlo a mano es lento y caro; hacerlo con un modelo entrenado es automatico y escala a millones de documentos.

**Que hace este proyecto.** Toma una imagen en blanco y negro de 28×28 pixeles y devuelve la probabilidad de que sea cada uno de los 10 digitos. La prediccion es la de mayor probabilidad. Es exactamente el pipeline que se usa para:

- **Lectura automatica de formularios y cheques** (OCR especializado).
- **Clasificacion de codigos postales manuscritos** en sistemas de correo.
- **Control de acceso** con padlocks de botones o pulsera de entrada por PIN.
- **Diagnostico medico** al interpretar escritura a mano en recetas y_records clinicos.
- **Educacion y juegos** que reconozcan digitos dibujados con el raton o el dedo.

**Para quien es.** El entregable principal es el **notebook** `notebooks/01_mnist_entrenamiento.ipynb`, que documenta el proceso completo. Este repositorio contiene el codigo reutilizable sobre el que se apoya el notebook, mas una CLI para entrenar, evaluar y desplegar sin abrir Jupyter.

---

## Como funciona

### 1. Flujo de datos

```
MNIST (60,000 train + 10,000 test)
   │
   ├─ split estratificado 90/10  →  train / validación
   │     (cada dígito mantiene su proporción)
   │
   ├─ train:      ToTensor → RandomAffine (±10°, ±10% traslación, ±10% escala) → Normalize
   └─ validación: ToTensor → Normalize   (sin augmentation, determinista)
```

La normalización usa la media y desviación estándar reales de MNIST (`0.1307`, `0.3081`) en lugar de divided-by-255. Con esto los pixeles quedan centrados en cero, el gradiente inicia mejor y se alcanzan ~0.3 puntos mas de accuracy.

El augmentation es **deliberadamente conservador**: rotaciones de ±10 grados. MNIST ya viene alineado y centrado; augmentaciones agresivas (rotaciones de 45°, flipping, cutting patches) degradan el rendimiento en este dataset.

### 2. Arquitectura del modelo

```
Entrada (1, 28, 28)
   │
   ├─ Block 1: Conv(1→32, 3×3) + BatchNorm + ReLU + MaxPool     → (32, 14, 14)
   ├─ Block 2: Conv(32→32, 3×3) + BN + ReLU + MaxPool + Drop   → (32, 7, 7)
   ├─ Block 3: Conv(32→64, 3×3) + BN + ReLU                      → (64, 7, 7)
   ├─ Block 4: Conv(64→64, 3×3) + BN + ReLU + MaxPool + Drop    → (64, 3, 3)
   ├─ AdaptiveAvgPool2d((1,1))                                   → (64, 1, 1)
   ├─ Flatten → Linear(64→128) + ReLU + Dropout
   └─ Linear(128→10)                                             → 10 logits
```

**Por que cada pieza:**

| Componente | Razon |
|---|---|
| **Conv layers** | Detectan bordes, curvas yangulos. Un MLP tendria que aprender esas formas desde cero. |
| **BatchNorm** | Normaliza cada activacion durante el entrenamiento: converge mas rapido y es mas estable. Se desactiva solo en inferencia. |
| **MaxPool** | Reduce la resolucion 2× y mantiene las activaciones mas fuertes. Cut computation ~75%. |
| **Dropout (0.25)** | Apaga neuronas al azar en entrenamiento. Impide que la red memorice imagenes concretas. |
| **AdaptiveAvgPool** | Hace el modelo invariante a desplazamientos pequenos del trazo. |
| **Sin softmax en el forward** | `CrossEntropyLoss` ya aplica log-softmax internamente. Ponerlo dos veces es un error clasico y numericamente inestable. |

### 3. Entrenamiento

| Decision | Valor | Motivo |
|---|---|---|
| Optimizador | **AdamW** (lr=1e-3, wd=1e-4) | Desacopla el peso decaido del gradiente. En tareas de vision generaliza mejor que Adam. |
| Scheduler | **OneCycleLR** (coseno) | Calienta el lr y lo baja de forma coseno. Llega a un mejor minimo que un lr fijo. |
| Epocas | 8–15 | MNIST converge rapido; más epocas empiezan a sobreajustar. |
| Gradient clipping | 5.0 | Evita actualizaciones descontroladas ante batches atipicos. |
| Precision mixta (AMP) | Activa en CUDA | ~2× mas rapido en GPU sin perder calidad. |
| Loss | `CrossEntropyLoss` | Logistica multiclase; ya incluye softmax numericamente estable. |

Se guarda el checkpoint con **mejor accuracy de validacion**, no el ultimo. Es la unica forma honesta: el test set se evalua una sola vez, al final.

### 4. Metricas

Mas alla del accuracy (que puede ocultar clases desbalanceadas), el proyecto calcula **precision, recall y F1 por digito**, mas la **matriz de confusion** normalizada.

Confusiones tipicas en MNIST: `4→9`, `3→5`, `7→2`, `8→9`. Son errores semanticamente razonables (un 9 con el borde superior cerrado parece un 8), no fallos de la arquitectura. Corregirlos exigiria datos que el dataset no contiene.

---

## Estructura del proyecto

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
│   ├── __init__.py
│   ├── config.py                     # Dataclasses + carga YAML + overrides
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
│   │   ├── trainer.py                # Trainer: fit, AMP, scheduler, early stopping
│   │   ├── metrics.py                # ClassificationMetrics (sin sklearn)
│   │   └── checkpoint.py             # Guardado/carga de pesos
│   ├── tracking/
│   │   └── wandb_tracker.py          # W&B con fallback offline automático
│   ├── inference/
│   │   └── predictor.py              # DigitPredictor para imágenes sueltas
│   └── utils/
│       ├── seed.py                   # Reproducibilidad
│       ├── device.py                 # CPU / CUDA / MPS automático
│       └── logging.py                # Logging a consola y archivo
└── tests/                            # 54 tests
    ├── test_config.py
    ├── test_models.py
    ├── test_metrics.py
    ├── test_trainer.py
    ├── test_transforms.py
    └── test_checkpoint_inference.py
```

---

## Instalacion

Requiere **Python 3.10+**.

```bash
git clone https://github.com/<tu-usuario>/mnist-digit-classifier.git
cd mnist-digit-classifier

# Entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

El dataset MNIST (~11 MB) se descarga solo en el primer entrenamiento.

---

## Uso

### Entrenar

```bash
# Entrenamiento estandar (10 epocas, W&B activo)
python scripts/train.py

# Rapido, para verificar que todo funciona (~30 s en CPU)
python scripts/train.py --config configs/quick.yaml

# Entrenamiento completo con W&B
python scripts/train.py --config configs/full.yaml --epochs 15

# Cambiar parametros sin tocar el YAML
python scripts/train.py --set optim.lr=3e-3 data.batch_size=256 model.name=resnet_mnist

# Limitar por tiempo (requisito de competicion)
python scripts/train.py --time-budget 1800
```

### Evaluar

```bash
python scripts/evaluate.py runs/mnist_simplecnn/best.pt --save-figure confusion.png
```

### Predecir y generar submission

```bash
# CSV con formato de competicion (ImageId,Label)
python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --split test --output submission.csv

# Clasificar imagenes propias
python scripts/predict.py --checkpoint runs/mnist_simplecnn/best.pt --images foto1.png foto2.png
```

### Comparar arquitecturas

```bash
python scripts/compare_models.py --epochs 5
```

### Tests

```bash
python -m pytest tests/ -v
```

### API de Python

```python
from digitclf.config import load_config
from digitclf.pipeline import Experiment
from digitclf.inference import DigitPredictor

# Entrenar
config = load_config("configs/full.yaml")
result, artifacts = Experiment(config).run()
print(result.summary())

# Inferir
predictor = DigitPredictor.from_checkpoint(artifacts.checkpoint)
label, confidence = predictor.predict_with_confidence("mi_digito.png")
```

---

## Weights & Biases

El enunciado exige registrar los experimentos en W&B.

**Para usar W&B:**

1. Crear cuenta gratis en [wandb.ai](https://wandb.ai)
2. Copiar la API key de [wandb.ai/authorize](https://wandb.ai/authorize)
3. Pegarla en una celda del notebook:

```python
import wandb
wandb.login("tu_api_key_aqui")     # o: os.environ["WANDB_API_KEY"] = "..."
```

**Sin cuenta**, el proyecto funciona igual: `WandbTracker` detecta que no hay `WANDB_API_KEY` y cae automaticamente en **modo offline**. Los runs se guardan en `./wandb/` y se sincronizan despues con:

```bash
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-<id>
```

**Que se registra:** configuracion completa del experimento, loss y accuracy por epoca, learning rate, metricas por digito, curvas de aprendizaje y matriz de confusion.

---

## Reproducibilidad

`seed_everything()` fija la semilla en Python, NumPy y PyTorch, y activa los algoritmos deterministas de cuDNN. La division train/validacion es estratificada y determinista: el mismo `seed` produce siempre el mismo split.

Consecuencia importante: **el modelo pesa los digitos igual**. Si el validacion contiene mas 3s que 8s, el modelo sera mejor con 3s y el accuracy sera enganoso. La estratificacion evita ese sesgo, y por eso el F1 macro importa tanto como el accuracy.

---

## Resultados verificables

Ejecutando `python scripts/compare_models.py --epochs 3` en Apple Silicon (MPS):

```
model                 params   val_acc   macro_f1     seg
------------------------------------------------------------------------
mlp                  537,354    0.9738     0.9736    12.2
simple_cnn           102,730    0.9770     0.9768    19.9
resnet_mnist         174,970    0.9922     0.9921    35.1
```

**La leccion:** la CNN gana al MLP en accuracy usando **5× menos parametros**. La estructura espacial de la imagen es informacion aprovechable; una red densa tiene que reconstruirla desde cero. Y `ResNetMNIST` supera a la CNN simple con las conexiones residuales, que facilitan el flujo de gradiente en redes profundas.

---

## Buenas practicas aplicadas

- **Separacion de responsabilidades**: `config`, `data`, `models`, `engine`, `tracking` e `inference` no se importan entre si de forma circular. El `Trainer` no sabe que existe MNIST.
- **Configuracion declarativa**: todo experimento se describe con dataclasses validadas. Cambiar de modelo o de optimizador es una linea, no una edicion de codigo.
- **Registro de modelos**: `build_model("simple_cnn", width=64)` permite añadir arquitecturas sin tocar el trainer.
- **Inyeccion de dependencias**: el tracker se pasa al trainer; si no hay W&B, no hay nada que cambiar.
- **Degradacion elegante**: sin API key, sin GPU o sin `wandb` instalado, el proyecto sigue funcionando.
- **Tests unitarios**: 54 tests sobre metricas, checkpointing, formas de tensor y validacion de config. Se ejecutan en segundos con datos sinteticos.
- **Rutas reproducibles**: cada corrida deja `config.yaml`, `history.json`, `metrics.json`, `best.pt` y las figuras.

---

## Licencia

MIT. Dataset MNIST distribuido por Yann LeCun et al. bajo licencia Creative Commons BY-SA 3.0.
