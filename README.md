# YOLO Training Workspace

Base de entrenamiento YOLO11 desde cero pensada para trabajar en Jetson Orin y mantener una estructura parecida a tu proyecto de U-Net:

- arquitectura en `models/`
- configuracion de entrenamiento en `configs/train/`
- data augmentation en `configs/augment/`
- script principal en `main.py`
- soporte para COCO JSON directo en `src/`

## Estructura

```text
YOLO/
├── configs/
│   ├── augment/
│   │   └── coco_scratch_orin.yaml
│   ├── data/
│   │   └── coco2017_val_only.yaml
│   ├── experiments/
│   │   └── exp1_baseline_coco.yaml
│   └── train/
│       └── from_scratch_orin.yaml
├── exp1_baseline.py
├── main.py
├── models/
│   └── yolo11_orin.yaml
├── requirements.txt
└── src/
    ├── __init__.py
    └── coco_json.py
```

## Instalacion

En Jetson no conviene instalar `torch` desde un `requirements.txt` generico. Instala primero la version de PyTorch/Torchvision compatible con tu JetPack y despues:

```bash
pip install -r requirements.txt
```

## Docker

Tambien tienes entorno Docker para Jetson en:

- [Dockerfile](/home/gmv/traaoi/YOLO/Dockerfile)
- [docker-compose.yaml](/home/gmv/traaoi/YOLO/docker-compose.yaml)

Servicios disponibles:

- `yolo_train`: entrenamiento desde `main.py`
- `yolo_exp1`: baseline preentrenado sobre COCO con `exp1_baseline.py`

Build:

```bash
docker compose build
```

Experimento 1:

```bash
docker compose run --rm yolo_exp1
```

Entrenamiento desde cero:

```bash
docker compose run --rm \
  -e YOLO_DATASET_ROOT=/home/code/dataset/coco \
  -e YOLO_EPOCHS=300 \
  -e YOLO_BATCH=4 \
  yolo_train
```

## Dataset

No necesitas meter COCO dentro de esta carpeta. El dataset puede vivir en cualquier ruta del sistema.

Este workspace soporta dos modos:

1. `coco_json`
Usa el COCO oficial con imagenes + `instances_train*.json` y `instances_val*.json` sin convertir a labels YOLO `.txt`.

2. `yolo`
Usa un `dataset.yaml` normal de Ultralytics si ya tienes las anotaciones convertidas a formato YOLO.

## Estructura esperada para COCO oficial

```text
/ruta/al/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

## Entrenamiento desde cero en COCO

Comando minimo:

```bash
python3 main.py --dataset-root /ruta/al/coco
```

Ese comando:

- genera automaticamente un `dataset.yaml` temporal en `generated/datasets/`
- lee las clases directamente desde el JSON de COCO
- crea el modelo desde `models/yolo11_orin.yaml`
- entrena con `pretrained=False`

## Overrides utiles

```bash
python3 main.py \
  --dataset-root /ruta/al/coco \
  --epochs 300 \
  --batch 4 \
  --imgsz 640 \
  --device 0 \
  --name yolo11_orin_coco_scratch
```

## Si ya tienes dataset YOLO normal

```bash
python3 main.py \
  --dataset-format yolo \
  --data-config /ruta/a/tu_dataset.yaml \
  --name yolo11_custom_scratch
```

## Notas

- La arquitectura por defecto esta orientada a `YOLO11 nano`, que es la opcion mas realista para entrenar desde cero en Jetson.
- Los `workers` estan en `0` por defecto para evitar problemas tipicos en entornos Jetson/Docker.
- El augmentation esta separado en `configs/augment/coco_scratch_orin.yaml` para que lo toques sin ensuciar el script principal.

## Experimento 1: baseline preentrenado en COCO

Este experimento no entrena. Valida un modelo YOLO ya preentrenado sobre COCO y guarda:

- `precision`
- `recall`
- `mAP50`
- `mAP50-95`
- tiempos de inferencia
- resumen en JSON

Comando:

```bash
cd /home/gmv/traaoi/YOLO
python3 exp1_baseline.py --show-config
```

El archivo [configs/data/coco2017_val_only.yaml](/home/gmv/traaoi/YOLO/configs/data/coco2017_val_only.yaml) esta preparado para descargar automaticamente `val2017` y las etiquetas en formato Ultralytics dentro de `dataset/coco/` si no existen.
