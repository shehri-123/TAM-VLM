#!/usr/bin/env python3

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "nuimages"
OUTPUT_DIR = ROOT / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["v1.0-train", "v1.0-val"]

TARGETS = {
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "traffic_barrier",
    "movable_object.debris": "debris",
}

manifest_rows = []
missing_files = []
global_image_categories = defaultdict(set)

for split in SPLITS:
    split_root = DATA_ROOT / split

    def load_json(name):
        path = split_root / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    categories = load_json("category.json")
    sample_data = load_json("sample_data.json")
    samples = load_json("sample.json")
    logs = load_json("log.json")
    object_anns = load_json("object_ann.json")

    category_by_token = {
        row["token"]: row["name"]
        for row in categories
    }

    sample_data_by_token = {
        row["token"]: row
        for row in sample_data
    }

    sample_by_token = {
        row["token"]: row
        for row in samples
    }

    log_by_token = {
        row["token"]: row
        for row in logs
    }

    # One record per image and target category.
    grouped = {}

    for ann in object_anns:
        category_name = category_by_token.get(ann.get("category_token"))

        if category_name not in TARGETS:
            continue

        sample_data_token = ann.get("sample_data_token")
        sd = sample_data_by_token.get(sample_data_token)

        if sd is None:
            continue

        filename = str(sd.get("filename", ""))

        # External audit is deliberately CAM_FRONT-only.
        if not filename.startswith("samples/CAM_FRONT/"):
            continue

        if not bool(sd.get("is_key_frame", True)):
            continue

        width = int(sd.get("width", 0))
        height = int(sd.get("height", 0))

        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or width <= 0 or height <= 0:
            continue

        x1, y1, x2, y2 = map(float, bbox)

        # Clamp annotation coordinates for robust geometry calculation.
        x1c = min(max(x1, 0.0), float(width))
        x2c = min(max(x2, 0.0), float(width))
        y1c = min(max(y1, 0.0), float(height))
        y2c = min(max(y2, 0.0), float(height))

        box_width = max(0.0, x2c - x1c)
        box_height = max(0.0, y2c - y1c)
        area_fraction = (
            box_width * box_height
        ) / float(width * height)

        key = (split, category_name, sample_data_token)

        if key not in grouped:
            sample_token = sd.get("sample_token", "")
            sample = sample_by_token.get(sample_token, {})
            log_token = sample.get("log_token", "")
            log = log_by_token.get(log_token, {})

            grouped[key] = {
                "dataset_split": split,
                "category": category_name,
                "category_short": TARGETS[category_name],
                "sample_data_token": sample_data_token,
                "sample_token": sample_token,
                "log_token": log_token,
                "log_location": log.get("location", ""),
                "logfile": log.get("logfile", ""),
                "image_relpath": filename,
                "image_abspath": str(DATA_ROOT / filename),
                "image_width": width,
                "image_height": height,
                "timestamp": sd.get("timestamp", ""),
                "bboxes": [],
                "bbox_area_fractions": [],
                "bbox_width_fractions": [],
                "bbox_height_fractions": [],
            }

        grouped[key]["bboxes"].append([x1, y1, x2, y2])
        grouped[key]["bbox_area_fractions"].append(area_fraction)
        grouped[key]["bbox_width_fractions"].append(
            box_width / float(width)
        )
        grouped[key]["bbox_height_fractions"].append(
            box_height / float(height)
        )

    for record in grouped.values():
        image_path = Path(record["image_abspath"])

        if not image_path.is_file():
            missing_files.append(str(image_path))

        area_values = record.pop("bbox_area_fractions")
        width_values = record.pop("bbox_width_fractions")
        height_values = record.pop("bbox_height_fractions")
        bboxes = record.pop("bboxes")

        record.update({
            "target_instance_count": len(bboxes),
            "max_bbox_area_fraction": max(area_values),
            "sum_bbox_area_fraction": sum(area_values),
            "max_bbox_width_fraction": max(width_values),
            "max_bbox_height_fraction": max(height_values),
            "bboxes_json": json.dumps(
                bboxes,
                separators=(",", ":")
            ),
            "image_exists": int(image_path.is_file()),
        })

        manifest_rows.append(record)
        global_image_categories[record["image_relpath"]].add(
            record["category_short"]
        )

manifest_rows.sort(
    key=lambda row: (
        row["category_short"],
        row["dataset_split"],
        row["log_token"],
        row["image_relpath"],
    )
)

fieldnames = [
    "dataset_split",
    "category",
    "category_short",
    "sample_data_token",
    "sample_token",
    "log_token",
    "log_location",
    "logfile",
    "image_relpath",
    "image_abspath",
    "image_width",
    "image_height",
    "timestamp",
    "target_instance_count",
    "max_bbox_area_fraction",
    "sum_bbox_area_fraction",
    "max_bbox_width_fraction",
    "max_bbox_height_fraction",
    "bboxes_json",
    "image_exists",
]

manifest_path = (
    OUTPUT_DIR /
    "nuimages_camfront_target_candidates_v1.csv"
)

with manifest_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(manifest_rows)

print("=" * 92)
print("nuImages CAM_FRONT target candidate manifest")
print("=" * 92)

for category in TARGETS.values():
    rows = [
        row for row in manifest_rows
        if row["category_short"] == category
    ]

    train_count = sum(
        row["dataset_split"] == "v1.0-train"
        for row in rows
    )
    val_count = sum(
        row["dataset_split"] == "v1.0-val"
        for row in rows
    )
    logs = {
        row["log_token"]
        for row in rows
        if row["log_token"]
    }
    area_values = [
        float(row["max_bbox_area_fraction"])
        for row in rows
    ]

    median_area = (
        statistics.median(area_values)
        if area_values else 0.0
    )

    print(
        f"{category:18s} | "
        f"train={train_count:5d} | "
        f"val={val_count:4d} | "
        f"total={len(rows):5d} | "
        f"logs={len(logs):4d} | "
        f"median max-box area={100 * median_area:7.4f}%"
    )

multi_target_images = {
    image: categories
    for image, categories in global_image_categories.items()
    if len(categories) > 1
}

print("-" * 92)
print(f"Manifest rows               : {len(manifest_rows)}")
print(f"Unique image paths          : {len(global_image_categories)}")
print(f"Multi-target image paths    : {len(multi_target_images)}")
print(f"Missing image files         : {len(missing_files)}")
print(f"Saved manifest              : {manifest_path}")

if missing_files:
    print("\nFirst missing image paths:")
    for path in missing_files[:20]:
        print(path)
    raise SystemExit("STATUS: FAIL — manifest contains missing files.")

print("STATUS: PASS")
