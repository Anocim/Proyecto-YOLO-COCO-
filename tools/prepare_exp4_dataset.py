import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml


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


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Prepare an exp4 dataset with empty train split and balanced val/test from an annotated source dataset."
    )
    parser.add_argument("--source-dir", required=True, help="Source YOLO dataset with images/ and labels/ in a single pool.")
    parser.add_argument("--output-dir", required=True, help="Output dataset root for exp4.")
    parser.add_argument("--val-count", type=int, default=28, help="Number of images to allocate to validation.")
    parser.add_argument("--test-count", type=int, default=28, help="Number of images to allocate to test.")
    return parser


def load_image_classes(labels_dir):
    image_classes = {}
    class_presence = Counter()

    for label_path in sorted(labels_dir.glob("*.txt")):
        classes = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            classes.append(int(line.split()[0]))
        unique_classes = sorted(set(classes))
        image_classes[label_path.stem] = unique_classes
        for cls in unique_classes:
            class_presence[cls] += 1

    return image_classes, class_presence


def score_image(classes, class_presence):
    # Higher score means the image contains rarer classes and should be assigned earlier.
    return sum(1.0 / class_presence[cls] for cls in classes)


def choose_split(image_classes, class_presence, val_count, test_count):
    names = sorted(
        image_classes,
        key=lambda name: (
            -score_image(image_classes[name], class_presence),
            -len(image_classes[name]),
            name,
        ),
    )

    val_names = []
    test_names = []
    val_presence = Counter()
    test_presence = Counter()

    for name in names:
        classes = image_classes[name]
        val_need = val_count - len(val_names)
        test_need = test_count - len(test_names)

        if val_need <= 0 and test_need <= 0:
            break
        if val_need <= 0:
            target = "test"
        elif test_need <= 0:
            target = "val"
        else:
            val_score = sum(abs((val_presence[c] + 1) - test_presence[c]) for c in classes)
            test_score = sum(abs(val_presence[c] - (test_presence[c] + 1)) for c in classes)
            # Bias towards the smaller split when scores tie.
            if val_score < test_score:
                target = "val"
            elif test_score < val_score:
                target = "test"
            else:
                target = "val" if len(val_names) <= len(test_names) else "test"

        if target == "val":
            val_names.append(name)
            for cls in classes:
                val_presence[cls] += 1
        else:
            test_names.append(name)
            for cls in classes:
                test_presence[cls] += 1

    return sorted(val_names), sorted(test_names), val_presence, test_presence


def copy_split(names, split_name, source_dir, output_dir):
    images_src = source_dir / "images"
    labels_src = source_dir / "labels"
    images_dst = output_dir / "images" / split_name
    labels_dst = output_dir / "labels" / split_name
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for stem in names:
        image_candidates = list(images_src.glob(f"{stem}.*"))
        if len(image_candidates) != 1:
            raise RuntimeError(f"Expected exactly one image for stem {stem}, found {len(image_candidates)}")
        image_src = image_candidates[0]
        label_src = labels_src / f"{stem}.txt"
        shutil.copy2(image_src, images_dst / image_src.name)
        shutil.copy2(label_src, labels_dst / label_src.name)


def count_boxes(labels_dir):
    counts = Counter()
    for path in labels_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            counts[int(line.split()[0])] += 1
    return counts


def main():
    args = build_argparser().parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    image_classes, class_presence = load_image_classes(source_dir / "labels")
    total_images = len(image_classes)
    if args.val_count + args.test_count > total_images:
        raise ValueError("val_count + test_count exceeds the number of annotated images.")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Empty train split for the user's own CVAT images.
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    val_names, test_names, val_presence, test_presence = choose_split(
        image_classes=image_classes,
        class_presence=class_presence,
        val_count=args.val_count,
        test_count=args.test_count,
    )

    copy_split(val_names, "val", source_dir, output_dir)
    copy_split(test_names, "test", source_dir, output_dir)

    data_yaml = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    summary = {
        "source_dir": str(source_dir),
        "notes": [
            "train is intentionally left empty for the user's own CVAT images.",
            "val and test were copied from the annotated subset extracted from dataset1.zip.",
            "This split should be revisited if you later collect enough own-domain validation/test data.",
        ],
        "splits": {
            "train": {
                "images": 0,
                "boxes_per_class": {},
            },
            "val": {
                "images": len(val_names),
                "image_presence_per_class": {CLASS_NAMES[k]: int(v) for k, v in sorted(val_presence.items())},
                "boxes_per_class": {
                    CLASS_NAMES[k]: int(v)
                    for k, v in sorted(count_boxes(output_dir / "labels" / "val").items())
                },
            },
            "test": {
                "images": len(test_names),
                "image_presence_per_class": {CLASS_NAMES[k]: int(v) for k, v in sorted(test_presence.items())},
                "boxes_per_class": {
                    CLASS_NAMES[k]: int(v)
                    for k, v in sorted(count_boxes(output_dir / "labels" / "test").items())
                },
            },
        },
        "val_images": val_names,
        "test_images": test_names,
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = (
        "# Exp4 dataset\n\n"
        "- Put your own CVAT annotated training images in `images/train/`\n"
        "- Put the matching YOLO `.txt` files in `labels/train/`\n"
        "- `val/` and `test/` were seeded from the annotated subset of `dataset1.zip`\n"
        "- If later you collect enough own-domain holdout data, replace or complement `val/` and `test/`\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Created exp4 dataset in: {output_dir}")
    print(f"val images: {len(val_names)}")
    print(f"test images: {len(test_names)}")


if __name__ == "__main__":
    main()
