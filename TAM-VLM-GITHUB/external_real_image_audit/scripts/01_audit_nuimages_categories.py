#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "nuimages"
OUTPUT_DIR = ROOT / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mini overlaps with the full release and test labels may be unavailable.
# Primary annotated audit therefore inspects train and validation only.
SPLITS = ["v1.0-train", "v1.0-val"]

all_rows = []
keyword_matches = []

for split in SPLITS:
    split_root = DATA_ROOT / split

    with (split_root / "category.json").open("r", encoding="utf-8") as f:
        categories = json.load(f)

    with (split_root / "sample_data.json").open("r", encoding="utf-8") as f:
        sample_data = json.load(f)

    with (split_root / "object_ann.json").open("r", encoding="utf-8") as f:
        object_anns = json.load(f)

    category_name = {
        row["token"]: row["name"]
        for row in categories
    }

    image_filename = {
        row["token"]: row["filename"]
        for row in sample_data
        if str(row.get("filename", "")).startswith("samples/")
    }

    annotation_count = defaultdict(int)
    image_sets = defaultdict(set)
    cam_front_sets = defaultdict(set)

    missing_sample_data_tokens = 0

    for ann in object_anns:
        category_token = ann.get("category_token")
        sample_data_token = ann.get("sample_data_token")

        name = category_name.get(category_token, "UNKNOWN_CATEGORY")
        filename = image_filename.get(sample_data_token)

        if filename is None:
            missing_sample_data_tokens += 1
            continue

        annotation_count[name] += 1
        image_sets[name].add(filename)

        if filename.startswith("samples/CAM_FRONT/"):
            cam_front_sets[name].add(filename)

    for name in sorted(category_name.values()):
        row = {
            "split": split,
            "category": name,
            "annotations": annotation_count[name],
            "unique_images_all_cameras": len(image_sets[name]),
            "unique_images_cam_front": len(cam_front_sets[name]),
        }
        all_rows.append(row)

        lower_name = name.lower()
        if any(word in lower_name for word in ["cone", "barrier", "debris", "litter"]):
            keyword_matches.append(row)

    print("=" * 92)
    print(f"{split}")
    print(f"Object annotations loaded       : {len(object_anns)}")
    print(f"Sample-data references missing  : {missing_sample_data_tokens}")
    print(f"Defined object categories       : {len(category_name)}")

print("\n" + "=" * 92)
print("TARGET-RELATED CATEGORY MATCHES")
print("=" * 92)

if not keyword_matches:
    print("No category names matched cone/barrier/debris/litter.")
else:
    for row in keyword_matches:
        print(
            f"{row['split']:12s} | "
            f"{row['category']:42s} | "
            f"anns={row['annotations']:7d} | "
            f"images(all)={row['unique_images_all_cameras']:6d} | "
            f"CAM_FRONT={row['unique_images_cam_front']:6d}"
        )

csv_path = OUTPUT_DIR / "nuimages_category_counts_train_val.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "split",
            "category",
            "annotations",
            "unique_images_all_cameras",
            "unique_images_cam_front",
        ],
    )
    writer.writeheader()
    writer.writerows(all_rows)

report_path = OUTPUT_DIR / "nuimages_target_category_report.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("nuImages target-related category audit\n")
    f.write("=" * 92 + "\n")

    for row in keyword_matches:
        f.write(
            f"{row['split']:12s} | "
            f"{row['category']:42s} | "
            f"anns={row['annotations']:7d} | "
            f"images(all)={row['unique_images_all_cameras']:6d} | "
            f"CAM_FRONT={row['unique_images_cam_front']:6d}\n"
        )

print("\nSaved CSV report:")
print(csv_path)

print("\nSaved target-category report:")
print(report_path)
