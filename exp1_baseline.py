import argparse
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
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(payload)}")
    return payload


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


def maybe_resolve_model(model_value, base_dir):
    candidate = Path(str(model_value)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if candidate.suffix in {".pt", ".onnx", ".engine", ".yaml"}:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str(model_value)


def resolve_data_config(data_config_path, base_dir, experiment_name):
    source_path = resolve_path(data_config_path, base_dir)
    dataset_yaml = load_yaml_file(source_path)

    dataset_root = dataset_yaml.get("path")
    if dataset_root is None:
        raise ValueError(f"Dataset config {source_path} does not define a 'path' key.")

    dataset_yaml["path"] = str(resolve_path(dataset_root, base_dir))

    resolved_path = base_dir / "generated" / "datasets" / f"{experiment_name}_data.yaml"
    save_yaml_file(resolved_path, dataset_yaml)
    return resolved_path


def extract_metrics(metrics):
    summary = {}

    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        for key, value in results_dict.items():
            if isinstance(value, (int, float)):
                summary[key] = float(value)

    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        summary["speed_ms"] = {
            key: float(value) for key, value in speed.items() if isinstance(value, (int, float))
        }

    box = getattr(metrics, "box", None)
    if box is not None:
        for src_attr, dst_key in (
            ("map50", "box_map50"),
            ("map", "box_map50_95"),
            ("mp", "box_precision"),
            ("mr", "box_recall"),
            ("fitness", "box_fitness"),
        ):
            value = getattr(box, src_attr, None)
            if isinstance(value, (int, float)):
                summary[dst_key] = float(value)

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        summary["save_dir"] = str(save_dir)

    return summary


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Experiment 1: validate a pretrained YOLO model on COCO."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp1_baseline_coco.yaml",
        help="Experiment configuration YAML.",
    )
    parser.add_argument("--show-config", action="store_true")
    return parser


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

    config_path = resolve_path(args.config, base_dir)
    config = load_yaml_file(config_path)
    experiment_name = str(config.get("name", config_path.stem))

    resolved_data_path = resolve_data_config(
        data_config_path=config["data"],
        base_dir=base_dir,
        experiment_name=experiment_name,
    )

    resolved_model = maybe_resolve_model(config["model"], base_dir)
    project_dir = resolve_path(config.get("project", "runs/experiments"), base_dir)

    val_kwargs = {
        "data": str(resolved_data_path),
        "split": config.get("split", "val"),
        "imgsz": int(config.get("imgsz", 640)),
        "batch": int(config.get("batch", 8)),
        "device": config.get("device", 0),
        "workers": int(config.get("workers", 0)),
        "conf": float(config.get("conf", 0.001)),
        "iou": float(config.get("iou", 0.7)),
        "max_det": int(config.get("max_det", 300)),
        "half": bool(config.get("half", True)),
        "save_json": bool(config.get("save_json", True)),
        "plots": bool(config.get("plots", True)),
        "verbose": bool(config.get("verbose", True)),
        "project": str(project_dir),
        "name": experiment_name,
        "exist_ok": bool(config.get("exist_ok", True)),
    }

    resolved_config_path = base_dir / "generated" / "experiments" / f"{experiment_name}.yaml"
    resolved_payload = {
        "config_source": str(config_path),
        "model": resolved_model,
        "data": str(resolved_data_path),
        "val_kwargs": val_kwargs,
    }
    save_yaml_file(resolved_config_path, resolved_payload)

    logger.info("Experiment config: %s", config_path)
    logger.info("Resolved data config: %s", resolved_data_path)
    logger.info("Model: %s", resolved_model)
    if args.show_config:
        logger.info("Validation kwargs:\n%s", yaml.safe_dump(val_kwargs, sort_keys=False))

    model = YOLO(resolved_model)
    metrics = model.val(**val_kwargs)
    summary = extract_metrics(metrics)
    summary.update(
        {
            "experiment_name": experiment_name,
            "model": resolved_model,
            "data": str(resolved_data_path),
        }
    )

    output_summary = base_dir / "generated" / "experiments" / experiment_name / "summary.json"
    save_json_file(output_summary, summary)
    logger.info("Summary saved to: %s", output_summary)


if __name__ == "__main__":
    main()
