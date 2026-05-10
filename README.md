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
│   │   └── ppe_scratch_orin.yaml
│   ├── data/
│   │   └── coco2017_val_only.yaml
│   │   └── construction_ppe.yaml
│   ├── experiments/
│   │   └── exp1_baseline_coco.yaml
│   └── train/
│       └── from_scratch_orin.yaml
│       └── exp2_from_scratch_ppe.yaml
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
- `yolo_exp2`: experimento 2, entrenamiento desde cero sobre Construction-PPE
- `yolo_exp3`: experimento 3, fine-tuning sobre Construction-PPE desde `yolo11n.pt`
- `yolo_exp4`: experimento 4, adaptacion de dominio desde el `best.pt` de `exp3`

Build:

```bash
docker compose build
```

Experimento 1:

```bash
docker compose run --rm yolo_exp1
```

Experimento 2:

```bash
docker compose run --rm yolo_exp2
```

Experimento 3:

```bash
docker compose run --rm yolo_exp3
```

Experimento 4:

```bash
docker compose run --rm yolo_exp4
```

Entrenamiento desde cero:

```bash
docker compose run --rm \
  -e YOLO_DATASET_ROOT=/home/code/dataset/coco \
  -e YOLO_EPOCHS=300 \
  -e YOLO_BATCH=4 \
  yolo_train
```

Los resultados quedan visibles en:

- `results/exp1_baseline_coco/` para el experimento 1
- `results/exp2_scratch_ppe/` para el experimento 2
- `results/exp3_finetune_ppe/` para el experimento 3
- `results/exp4_domain_adaptation/` para el experimento 4

## Experimento 2: entrenamiento desde cero en Construction-PPE

Este experimento usa:

- [configs/data/construction_ppe.yaml](/home/gmv/traaoi/YOLO/configs/data/construction_ppe.yaml)
- [configs/train/exp2_from_scratch_ppe.yaml](/home/gmv/traaoi/YOLO/configs/train/exp2_from_scratch_ppe.yaml)
- [configs/augment/ppe_scratch_orin.yaml](/home/gmv/traaoi/YOLO/configs/augment/ppe_scratch_orin.yaml)

y entrena `YOLO11n` desde cero con `pretrained=False`.

Comando:

```bash
cd /home/gmv/traaoi/YOLO
docker compose run --rm yolo_exp2
```

Si el dataset no existe en `dataset/`, Ultralytics intentara descargarlo usando la URL oficial del dataset.

## Experimento 3: fine-tuning sobre Construction-PPE desde COCO

Este experimento usa:

- [configs/data/construction_ppe.yaml](/home/gmv/traaoi/YOLO/configs/data/construction_ppe.yaml)
- [configs/train/exp3_finetune_ppe.yaml](/home/gmv/traaoi/YOLO/configs/train/exp3_finetune_ppe.yaml)
- [configs/augment/ppe_finetune_orin.yaml](/home/gmv/traaoi/YOLO/configs/augment/ppe_finetune_orin.yaml)

y carga pesos preentrenados desde [yolo11n.pt](/home/gmv/traaoi/YOLO/yolo11n.pt).

Comando:

```bash
cd /home/gmv/traaoi/YOLO
docker compose run --rm yolo_exp3
```

Cambios respecto al experimento 2:

- mismo dataset y misma resolucion para que la comparacion sea limpia
- menos epochs (`60`) porque el modelo ya parte de COCO
- `lr0` mas baja (`0.002`) para refinar en vez de reaprender
- augmentation algo menos agresivo

## Experimento 4: adaptacion de dominio con dataset propio

Este experimento usa:

- [configs/data/exp4_domain_adaptation.yaml](/home/gmv/traaoi/YOLO/configs/data/exp4_domain_adaptation.yaml)
- [configs/train/exp4_domain_adaptation.yaml](/home/gmv/traaoi/YOLO/configs/train/exp4_domain_adaptation.yaml)
- [configs/augment/exp4_domain_adaptation.yaml](/home/gmv/traaoi/YOLO/configs/augment/exp4_domain_adaptation.yaml)

y parte de [best.pt](/home/gmv/traaoi/YOLO/results/exp3_finetune_ppe/weights/best.pt), que fue el mejor modelo del experimento 3.

Comando:

```bash
cd /home/gmv/traaoi/YOLO
docker compose run --rm yolo_exp4
```

El dataset final de `exp4` queda en:

- [dataset/exp4_domain_adaptation](/home/gmv/traaoi/YOLO/dataset/exp4_domain_adaptation)

y su reparto actual es:

- `149` imagenes en `train`
- `25` imagenes en `val`
- `25` imagenes en `test`

La idea aqui ya no es reaprender PPE desde cero, sino ajustar el detector al dominio real con un LR mas bajo y augmentations mas suaves.

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

- genera automaticamente un `dataset.yaml` temporal dentro de `results/<nombre_del_run>/meta/`
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
