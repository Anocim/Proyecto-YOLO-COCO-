import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import yaml


LABEL_MAP = {
    "helmet": "helmet",
    "gloves": "gloves",
    "vest": "vest",
    "boots": "boots",
    "googles": "goggles",
    "none": "none",
    "Person": "Person",
    "no_helmet": "no_helmet",
    "no_google": "no_goggle",
    "no_gloves": "no_gloves",
    "no_boots": "no_boots",
}

CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
]

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Extract annotated images from a CVAT ZIP export and convert them to YOLO labels."
    )
    parser.add_argument("--zip-path", required=True, help="Path to the CVAT ZIP export.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where images/, labels/, data.yaml and summary.json will be written.",
    )
    return parser


def load_manifest_images(zf):
    if "task_0/data/manifest.jsonl" in zf.namelist():
        manifest_name = "task_0/data/manifest.jsonl"
    elif "data/manifest.jsonl" in zf.namelist():
        manifest_name = "data/manifest.jsonl"
    else:
        raise RuntimeError("No manifest.jsonl found in the CVAT ZIP export.")
    lines = zf.read(manifest_name).decode("utf-8").splitlines()
    frames = []
    for raw in lines:
        item = json.loads(raw)
        if "name" in item:
            frames.append(item)
    return frames


def load_annotations(zf):
    if "task_0/annotations.json" in zf.namelist():
        annotations_name = "task_0/annotations.json"
    elif "annotations.json" in zf.namelist():
        annotations_name = "annotations.json"
    else:
        raise RuntimeError("No annotations.json found in the CVAT ZIP export.")
    payload = json.loads(zf.read(annotations_name))
    if not payload or not isinstance(payload, list):
        raise RuntimeError("Unexpected annotations.json format.")
    return payload[0].get("shapes", [])


def clamp(value, low, high):
    return max(low, min(value, high))


def convert_box(points, width, height):
    x1, y1, x2, y2 = points
    x1 = clamp(float(x1), 0.0, float(width))
    x2 = clamp(float(x2), 0.0, float(width))
    y1 = clamp(float(y1), 0.0, float(height))
    y2 = clamp(float(y2), 0.0, float(height))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    box_w = max(right - left, 0.0)
    box_h = max(bottom - top, 0.0)
    x_center = left + box_w / 2.0
    y_center = top + box_h / 2.0
    return (
        x_center / width,
        y_center / height,
        box_w / width,
        box_h / height,
    )


def main():
    args = build_argparser().parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        frames = load_manifest_images(zf)
        shapes = load_annotations(zf)

        by_frame = {}
        for shape in shapes:
            if shape.get("type") != "rectangle" or shape.get("outside", False):
                continue
            by_frame.setdefault(int(shape["frame"]), []).append(shape)

        class_counter = Counter()
        used_frames = sorted(by_frame)

        for frame_idx in used_frames:
            frame_meta = frames[frame_idx]
            if Path(f"task_0/data/{frame_meta['name']}{frame_meta['extension']}").as_posix() in zf.namelist():
                image_rel = Path("task_0/data") / f"{frame_meta['name']}{frame_meta['extension']}"
            else:
                image_rel = Path("data") / f"{frame_meta['name']}{frame_meta['extension']}"
            image_name = f"{Path(frame_meta['name']).name}{frame_meta['extension']}"
            image_bytes = zf.read(str(image_rel))
            (images_dir / image_name).write_bytes(image_bytes)

            width = float(frame_meta["width"])
            height = float(frame_meta["height"])
            lines = []
            for shape in by_frame[frame_idx]:
                raw_label = shape["label"]
                if raw_label not in LABEL_MAP:
                    continue
                normalized_label = LABEL_MAP[raw_label]
                class_id = CLASS_INDEX[normalized_label]
                xc, yc, bw, bh = convert_box(shape["points"], width, height)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                class_counter[normalized_label] += 1

            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = {
        "path": str(output_dir),
        "train": "images",
        "val": "images",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "source_zip": str(zip_path),
        "annotated_images": len(used_frames),
        "total_boxes": int(sum(class_counter.values())),
        "class_counts": dict(class_counter),
        "notes": [
            "Only images with at least one rectangle annotation were extracted.",
            "Label names were normalized to the project taxonomy: googles->goggles and no_google->no_goggle.",
            "The output is a partial dataset intended for later split into train/val/test.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Created partial dataset in: {output_dir}")
    print(f"Annotated images: {len(used_frames)}")
    print(f"Total boxes: {sum(class_counter.values())}")
    for name in CLASS_NAMES:
        if name in class_counter:
            print(f"{name}: {class_counter[name]}")


if __name__ == "__main__":
    main()
