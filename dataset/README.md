# Dataset Directory

Puedes descargar COCO aqui si quieres mantener todo el proyecto autocontenido.

Estructura recomendada:

```text
dataset/
└── coco/
    ├── images/
    │   └── val2017/
    ├── labels/
    │   └── val2017/
    ├── val2017.txt
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

Notas:

- Para el `experimento 1` no se usan los JSON oficiales directamente. Se usa el formato COCO adaptado por Ultralytics, con `val2017.txt` y labels YOLO en `labels/val2017/`.
- La configuracion [configs/data/coco2017_val_only.yaml](/home/gmv/traaoi/YOLO/configs/data/coco2017_val_only.yaml) puede descargar automaticamente esa estructura para la validacion baseline.
- Para el entrenamiento desde `main.py`, la parte relevante del COCO oficial sigue siendo `train2017/`, `val2017/` y `annotations/`.

Luego el entrenamiento seria:

```bash
cd /home/gmv/traaoi/YOLO
python3 main.py --dataset-root /home/gmv/traaoi/YOLO/dataset/coco
```
*** Add File: /home/gmv/traaoi/YOLO/configs/data/coco2017_val_only.yaml
# COCO 2017 validation-only dataset config for Experiment 1.
# It keeps the project self-contained under YOLO/dataset/coco.
#
# This file is intended for `model.val(...)` with a pretrained COCO model.
# It does not define a train-ready COCO setup for scratch training.

path: dataset/coco
train: val2017.txt # placeholder unused in experiment 1
val: val2017.txt

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

download: |
  from pathlib import Path

  from ultralytics.utils import ASSETS_URL
  from ultralytics.utils.downloads import download

  dir = Path(yaml["path"])
  urls = [ASSETS_URL + "/coco2017labels.zip"]
  download(urls, dir=dir.parent)

  urls = [
      "http://images.cocodataset.org/zips/val2017.zip",
  ]
  download(urls, dir=dir / "images", threads=3)
