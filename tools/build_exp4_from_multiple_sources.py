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
        description="Merge multiple single-pool YOLO datasets and create a final exp4 train/val/test split."
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        required=True,
        help="One or more source datasets, each with images/ and labels/ folders.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--test-ratio", type=float, default=0.125)
    return parser


def load_pool(source_dirs):
    pool = {}
    duplicate_stems = []

    for source_dir in source_dirs:
        source_dir = Path(source_dir).expanduser().resolve()
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

        for label_path in sorted(labels_dir.glob("*.txt")):
            stem = label_path.stem
            image_candidates = list(images_dir.glob(f"{stem}.*"))
            if len(image_candidates) != 1:
                raise RuntimeError(f"Expected one image for {stem} in {source_dir}, found {len(image_candidates)}")
            image_path = image_candidates[0]
            if stem in pool:
                duplicate_stems.append(stem)
                continue
            classes = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                classes.append(int(line.split()[0]))
            pool[stem] = {
                "image_path": image_path,
                "label_path": label_path,
                "classes": sorted(set(classes)),
                "source_dir": str(source_dir),
            }

    return pool, sorted(duplicate_stems)


def score_item(classes, class_presence):
    return sum(1.0 / class_presence[cls] for cls in classes)


def choose_split(pool, train_count, val_count, test_count):
    class_presence = Counter()
    for item in pool.values():
        for cls in item["classes"]:
            class_presence[cls] += 1

    ordered = sorted(
        pool.keys(),
        key=lambda stem: (
            -score_item(pool[stem]["classes"], class_presence),
            -len(pool[stem]["classes"]),
            stem,
        ),
    )

    split_names = {"train": [], "val": [], "test": []}
    split_presence = {"train": Counter(), "val": Counter(), "test": Counter()}
    target_sizes = {"train": train_count, "val": val_count, "test": test_count}

    for stem in ordered:
        classes = pool[stem]["classes"]
        candidate_splits = [s for s in ("train", "val", "test") if len(split_names[s]) < target_sizes[s]]
        if not candidate_splits:
            break

        best_split = None
        best_key = None
        for split in candidate_splits:
            # We want rare classes represented in all splits; bias val/test slightly.
            imbalance = sum(split_presence[split][cls] for cls in classes)
            fill_ratio = len(split_names[split]) / max(target_sizes[split], 1)
            split_bias = 0 if split == "train" else -0.1
            key = (imbalance + fill_ratio + split_bias, len(split_names[split]))
            if best_key is None or key < best_key:
                best_key = key
                best_split = split

        split_names[best_split].append(stem)
        for cls in classes:
            split_presence[best_split][cls] += 1

    return {k: sorted(v) for k, v in split_names.items()}, split_presence


def copy_split(stems, split_name, pool, output_dir):
    images_dst = output_dir / "images" / split_name
    labels_dst = output_dir / "labels" / split_name
    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        image_src = pool[stem]["image_path"]
        label_src = pool[stem]["label_path"]
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
    source_dirs = [Path(p).expanduser().resolve() for p in args.source_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    pool, duplicate_stems = load_pool(source_dirs)
    total = len(pool)
    train_count = round(total * args.train_ratio)
    val_count = round(total * args.val_ratio)
    test_count = total - train_count - val_count

    split_names, split_presence = choose_split(pool, train_count, val_count, test_count)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val", "test"):
        copy_split(split_names[split], split, pool, output_dir)

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
        "sources": [str(p) for p in source_dirs],
        "duplicates_skipped_by_stem": duplicate_stems,
        "total_images": total,
        "split_sizes": {k: len(v) for k, v in split_names.items()},
        "splits": {},
    }
    for split in ("train", "val", "test"):
        summary["splits"][split] = {
            "images": len(split_names[split]),
            "image_presence_per_class": {
                CLASS_NAMES[k]: int(v) for k, v in sorted(split_presence[split].items())
            },
            "boxes_per_class": {
                CLASS_NAMES[k]: int(v)
                for k, v in sorted(count_boxes(output_dir / "labels" / split).items())
            },
        }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = (
        "# Exp4 dataset\n\n"
        "- Built by merging multiple annotated sources.\n"
        "- Split generated automatically into train/val/test.\n"
        "- Review `split_summary.json` before training.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Created merged exp4 dataset in: {output_dir}")
    print(f"Total images: {total}")
    print(f"Split sizes: train={len(split_names['train'])}, val={len(split_names['val'])}, test={len(split_names['test'])}")
    if duplicate_stems:
        print(f"Skipped duplicate stems: {len(duplicate_stems)}")


if __name__ == "__main__":
    main()
