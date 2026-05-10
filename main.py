import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_yaml_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(data)}")
    return data


def save_yaml_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def save_json_file(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def resolve_path(value, base_dir):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def normalize_cache_value(raw_value):
    value = str(raw_value).strip().lower()
    if value in {"false", "none", "off", "0"}:
        return False
    if value in {"ram", "disk"}:
        return value
    raise ValueError(f"Unsupported cache value: {raw_value}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Entrenamiento YOLO11 desde cero con soporte para COCO JSON directo."
    )
    parser.add_argument("--model-config", default="models/yolo11_orin.yaml")
    parser.add_argument("--train-config", default="configs/train/from_scratch_orin.yaml")
    parser.add_argument("--augment-config", default="configs/augment/coco_scratch_orin.yaml")

    parser.add_argument(
        "--dataset-format",
        default="coco_json",
        choices=["coco_json", "yolo"],
        help="coco_json usa COCO oficial sin convertir; yolo usa dataset.yaml normal de Ultralytics.",
    )
    parser.add_argument(
        "--data-config",
        default=None,
        help="Ruta a dataset.yaml si ya lo tienes hecho. Obligatorio en modo 'yolo'.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Raiz del dataset externo. En COCO oficial suele contener train2017/, val2017/ y annotations/.",
    )
    parser.add_argument("--train-split", default="train2017")
    parser.add_argument("--val-split", default="val2017")
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--val-json", default=None)

    parser.add_argument("--project", default="results")
    parser.add_argument("--name", default="yolo11_orin_scratch")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument(
        "--cache",
        default="disk",
        choices=["false", "ram", "disk"],
        help="Caching de imagenes para Ultralytics.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Activa o desactiva pesos preentrenados. Si no se indica, se respeta el YAML.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reanuda o no el entrenamiento. Si no se indica, se respeta el YAML.",
    )
    parser.add_argument("--show-config", action="store_true")
    return parser


def prepare_data_config(args, base_dir):
    results_root = resolve_path(args.project, base_dir)

    if args.data_config:
        data_config = resolve_path(args.data_config, base_dir)
        if not data_config.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_config}")
        dataset_yaml = load_yaml_file(data_config)
        dataset_root = dataset_yaml.get("path")
        if dataset_root is None:
            raise ValueError(f"Dataset config {data_config} does not define a 'path' key.")
        dataset_yaml["path"] = str(resolve_path(dataset_root, base_dir))
        output_path = results_root / args.name / "meta" / "dataset.yaml"
        save_yaml_file(output_path, dataset_yaml)
        return output_path, Path(dataset_yaml["path"])

    if args.dataset_format != "coco_json":
        raise ValueError(
            "When --dataset-format yolo is used you must provide --data-config pointing to a dataset.yaml."
        )

    if not args.dataset_root:
        raise ValueError("--dataset-root is required when generating a COCO JSON dataset config.")

    try:
        from src.coco_json import default_coco_json_path, write_coco_json_data_yaml
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import COCO JSON helpers. Make sure 'ultralytics' is installed in the active environment."
        ) from exc

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    train_dir = dataset_root / args.train_split
    val_dir = dataset_root / args.val_split
    if not train_dir.exists():
        raise FileNotFoundError(f"Train image directory not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation image directory not found: {val_dir}")

    train_json = (
        Path(args.train_json).expanduser().resolve()
        if args.train_json
        else default_coco_json_path(dataset_root, args.train_split).resolve()
    )
    val_json = (
        Path(args.val_json).expanduser().resolve()
        if args.val_json
        else default_coco_json_path(dataset_root, args.val_split).resolve()
    )
    if not train_json.exists():
        raise FileNotFoundError(f"Train JSON not found: {train_json}")
    if not val_json.exists():
        raise FileNotFoundError(f"Validation JSON not found: {val_json}")

    output_path = results_root / args.name / "meta" / "dataset.yaml"
    write_coco_json_data_yaml(
        output_path=output_path,
        dataset_root=dataset_root,
        train_split=args.train_split,
        val_split=args.val_split,
        train_json=train_json,
        val_json=val_json,
    )
    return output_path, dataset_root


def build_train_overrides(args, base_dir, data_config_path):
    model_config = resolve_path(args.model_config, base_dir)
    train_config = resolve_path(args.train_config, base_dir)
    augment_config = resolve_path(args.augment_config, base_dir)
    project_dir = resolve_path(args.project, base_dir)

    if not model_config.exists():
        raise FileNotFoundError(f"Model config not found: {model_config}")

    train_overrides = load_yaml_file(train_config)
    augment_overrides = load_yaml_file(augment_config)

    overrides = {}
    overrides.update(train_overrides)
    overrides.update(augment_overrides)

    overrides["data"] = str(data_config_path)
    overrides["project"] = str(project_dir)
    overrides["name"] = args.name
    overrides["workers"] = args.workers
    overrides["cache"] = normalize_cache_value(args.cache)
    if args.pretrained is not None:
        overrides["pretrained"] = bool(args.pretrained)
    if args.resume is not None:
        overrides["resume"] = bool(args.resume)

    if args.device is not None:
        overrides["device"] = args.device
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.patience is not None:
        overrides["patience"] = args.patience
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.fraction is not None:
        overrides["fraction"] = args.fraction

    return {
        "model_config": model_config,
        "train_config": train_config,
        "augment_config": augment_config,
        "project_dir": project_dir,
        "train_overrides": overrides,
    }


def maybe_write_training_summary(save_dir):
    save_dir = Path(save_dir)
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return None

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    if not rows:
        return None

    numeric_rows = []
    for row in rows:
        parsed = {"epoch": int(float(row["epoch"]))}
        for key, value in row.items():
            if key == "epoch":
                continue
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                continue
        numeric_rows.append(parsed)

    best_row = max(numeric_rows, key=lambda row: row.get("metrics/mAP50-95(B)", float("-inf")))
    last_row = numeric_rows[-1]
    summary = {
        "save_dir": str(save_dir),
        "best_epoch": best_row["epoch"],
        "last_epoch": last_row["epoch"],
        "best_metrics": {
            "precision": best_row.get("metrics/precision(B)"),
            "recall": best_row.get("metrics/recall(B)"),
            "mAP50": best_row.get("metrics/mAP50(B)"),
            "mAP50-95": best_row.get("metrics/mAP50-95(B)"),
        },
        "final_metrics": {
            "precision": last_row.get("metrics/precision(B)"),
            "recall": last_row.get("metrics/recall(B)"),
            "mAP50": last_row.get("metrics/mAP50(B)"),
            "mAP50-95": last_row.get("metrics/mAP50-95(B)"),
        },
        "final_losses": {
            "train_box_loss": last_row.get("train/box_loss"),
            "train_cls_loss": last_row.get("train/cls_loss"),
            "train_dfl_loss": last_row.get("train/dfl_loss"),
            "val_box_loss": last_row.get("val/box_loss"),
            "val_cls_loss": last_row.get("val/cls_loss"),
            "val_dfl_loss": last_row.get("val/dfl_loss"),
        },
    }
    summary_path = save_dir / "summary.json"
    save_json_file(summary_path, summary)
    return summary_path


def main():
    parser = build_argparser()
    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install the dependencies from requirements.txt "
            "after setting up a Jetson-compatible torch/torchvision."
        ) from exc

    data_config_path, dataset_root = prepare_data_config(args, base_dir)
    training_bundle = build_train_overrides(args, base_dir, data_config_path)
    results_root = training_bundle["project_dir"]
    resolved_config_path = results_root / args.name / "meta" / "resolved_config.yaml"
    resolved_payload = {
        "model_config": str(training_bundle["model_config"]),
        "data_config": str(data_config_path),
        "dataset_root": str(dataset_root) if dataset_root else None,
        "dataset_format": args.dataset_format,
        "train_config": str(training_bundle["train_config"]),
        "augment_config": str(training_bundle["augment_config"]),
        "results_root": str(results_root),
        "overrides": training_bundle["train_overrides"],
    }
    save_yaml_file(resolved_config_path, resolved_payload)

    logger.info("Model config: %s", training_bundle["model_config"])
    logger.info("Data config: %s", data_config_path)
    logger.info("Resolved run config: %s", resolved_config_path)
    if args.show_config:
        logger.info("Training overrides:\n%s", yaml.safe_dump(training_bundle["train_overrides"], sort_keys=False))

    model = YOLO(str(training_bundle["model_config"]))

    if args.dataset_format == "coco_json":
        from src.coco_json import COCOTrainer

        results = model.train(trainer=COCOTrainer, **training_bundle["train_overrides"])
    else:
        results = model.train(**training_bundle["train_overrides"])

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        logger.info("Training artifacts saved in: %s", save_dir)
        summary_path = maybe_write_training_summary(save_dir)
        if summary_path is not None:
            logger.info("Summary saved in: %s", summary_path)
    else:
        logger.info("Training finished.")
    return results


if __name__ == "__main__":
    main()
