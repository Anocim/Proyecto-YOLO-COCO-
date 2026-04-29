import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics.data.dataset import DATASET_CACHE_VERSION, YOLODataset
from ultralytics.data.utils import get_hash, load_dataset_cache_file, save_dataset_cache_file
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import TQDM, colorstr


class COCODataset(YOLODataset):
    """Read COCO JSON annotations directly instead of scanning YOLO .txt labels."""

    def __init__(self, *args, json_file="", **kwargs):
        self.json_file = str(json_file)
        super().__init__(*args, data={"channels": 3}, **kwargs)

    def get_img_files(self, img_path):
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        x = {"labels": []}
        with open(self.json_file, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        categories = {
            cat["id"]: index
            for index, cat in enumerate(sorted(coco["categories"], key=lambda item: item["id"]))
        }

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        for img_info in TQDM(coco["images"], desc="reading annotations"):
            height, width = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            if not im_file.exists():
                continue

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue

                # COCO xywh top-left in pixels -> YOLO normalized center xywh.
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2.0
                box[[0, 2]] /= width
                box[[1, 3]] /= height
                if box[2] <= 0 or box[3] <= 0:
                    continue

                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            labels = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (height, width),
                    "cls": labels[:, 0:1],
                    "bboxes": labels[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        x["hash"] = get_hash([self.json_file, str(self.img_path)])
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        cache_path = Path(self.json_file).with_suffix(".cache")
        try:
            cache = load_dataset_cache_file(cache_path)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash([self.json_file, str(self.img_path)])
            self.im_files = [label["im_file"] for label in cache["labels"]]
        except (FileNotFoundError, AssertionError, AttributeError, KeyError, ModuleNotFoundError):
            cache = self.cache_labels(cache_path)

        cache.pop("hash", None)
        cache.pop("version", None)
        return cache["labels"]


class COCOTrainer(DetectionTrainer):
    """Use COCODataset so Ultralytics can train directly from COCO JSON."""

    def build_dataset(self, img_path, mode="train", batch=None):
        json_file = (
            self.data["train_json"]
            if mode == "train"
            else self.data.get("val_json", self.data["train_json"])
        )
        return COCODataset(
            img_path=img_path,
            json_file=json_file,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.model.stride.max()) if hasattr(self, "model") and self.model else 32,
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=self.args.task,
            classes=self.args.classes,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def default_coco_json_path(dataset_root, split):
    split_name = Path(split).name
    return Path(dataset_root) / "annotations" / f"instances_{split_name}.json"


def _load_sorted_categories(json_file):
    with open(json_file, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = sorted(payload["categories"], key=lambda item: item["id"])
    return [(int(category["id"]), str(category["name"])) for category in categories]


def build_names_from_coco_json(json_file):
    return {index: name for index, (_, name) in enumerate(_load_sorted_categories(json_file))}


def validate_coco_category_alignment(train_json, val_json):
    if not val_json:
        return
    train_categories = _load_sorted_categories(train_json)
    val_categories = _load_sorted_categories(val_json)
    if train_categories != val_categories:
        raise ValueError(
            "train_json and val_json do not expose the same ordered category set. "
            "This would break class indexing at training time."
        )


def build_coco_json_data_dict(dataset_root, train_split, val_split, train_json, val_json):
    validate_coco_category_alignment(train_json, val_json)
    names = build_names_from_coco_json(train_json)
    return {
        "path": str(Path(dataset_root).resolve()),
        "train": str(train_split),
        "val": str(val_split),
        "train_json": str(Path(train_json).resolve()),
        "val_json": str(Path(val_json).resolve()) if val_json else str(Path(train_json).resolve()),
        "nc": len(names),
        "names": names,
    }


def write_coco_json_data_yaml(output_path, dataset_root, train_split, val_split, train_json, val_json):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_coco_json_data_dict(
        dataset_root=dataset_root,
        train_split=train_split,
        val_split=val_split,
        train_json=train_json,
        val_json=val_json,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return output_path
