#!/usr/bin/env python3

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST = (
    ROOT /
    "metadata/nuimages_camfront_candidates_no_tamvlm_overlap_v2.csv"
)

OUTPUT_MANIFEST = (
    ROOT /
    "metadata/nuimages_camfront_image_level_pool_v1.csv"
)

REPORT_PATH = (
    ROOT /
    "metadata/nuimages_camfront_sampling_pool_audit_v1.txt"
)

TARGETS = [
    "traffic_cone",
    "traffic_barrier",
    "debris",
]


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows = read_csv(INPUT_MANIFEST)

if not rows:
    raise SystemExit("FAIL — retained candidate manifest is empty.")

grouped = defaultdict(list)

for row in rows:
    image_relpath = row.get("image_relpath", "").strip()

    if not image_relpath:
        raise SystemExit("FAIL — row with empty image_relpath found.")

    grouped[image_relpath].append(row)

image_rows = []
failures = []

combination_counts = Counter()
category_total = Counter()
category_exclusive = Counter()
category_multitarget = Counter()

category_total_logs = defaultdict(set)
category_exclusive_logs = defaultdict(set)

category_exclusive_split = defaultdict(Counter)
category_total_split = defaultdict(Counter)

category_exclusive_areas = defaultdict(list)
category_total_areas = defaultdict(list)

for image_relpath, members in grouped.items():
    categories = [
        row.get("category_short", "").strip()
        for row in members
    ]

    if len(categories) != len(set(categories)):
        failures.append(
            f"duplicate category row for image: {image_relpath}"
        )
        continue

    unknown_categories = sorted(
        set(categories) - set(TARGETS)
    )

    if unknown_categories:
        failures.append(
            f"unknown categories {unknown_categories}: {image_relpath}"
        )
        continue

    invariant_fields = [
        "dataset_split",
        "sample_data_token",
        "sample_token",
        "log_token",
        "log_location",
        "logfile",
        "image_abspath",
        "image_width",
        "image_height",
        "timestamp",
    ]

    invariant_values = {}

    for field in invariant_fields:
        values = {
            row.get(field, "").strip()
            for row in members
        }

        if len(values) != 1:
            failures.append(
                f"inconsistent {field}: {image_relpath}"
            )
            break

        invariant_values[field] = next(iter(values))
    else:
        category_map = {
            row["category_short"]: row
            for row in members
        }

        category_tuple = tuple(sorted(category_map))
        category_set_text = "+".join(category_tuple)
        category_count = len(category_tuple)

        combination_counts[category_tuple] += 1

        exclusive_category = (
            category_tuple[0]
            if category_count == 1
            else ""
        )

        total_instances = sum(
            int(category_map[category]["target_instance_count"])
            for category in category_tuple
        )

        max_box_area = max(
            float(
                category_map[category][
                    "max_bbox_area_fraction"
                ]
            )
            for category in category_tuple
        )

        sum_box_area = sum(
            float(
                category_map[category][
                    "sum_bbox_area_fraction"
                ]
            )
            for category in category_tuple
        )

        output_row = {
            **invariant_values,
            "image_relpath": image_relpath,
            "target_category_set": category_set_text,
            "target_category_count": category_count,
            "is_exclusive_single_category": int(
                category_count == 1
            ),
            "is_multi_target": int(
                category_count > 1
            ),
            "exclusive_category": exclusive_category,
            "total_target_instances": total_instances,
            "max_target_bbox_area_fraction": max_box_area,
            "sum_target_bbox_area_fraction": sum_box_area,
        }

        for category in TARGETS:
            category_row = category_map.get(category)

            output_row[f"has_{category}"] = int(
                category_row is not None
            )

            output_row[f"{category}_instances"] = (
                int(category_row["target_instance_count"])
                if category_row else 0
            )

            output_row[
                f"{category}_max_bbox_area_fraction"
            ] = (
                float(category_row["max_bbox_area_fraction"])
                if category_row else 0.0
            )

            output_row[
                f"{category}_sum_bbox_area_fraction"
            ] = (
                float(category_row["sum_bbox_area_fraction"])
                if category_row else 0.0
            )

        image_rows.append(output_row)

        split = invariant_values["dataset_split"]
        log_token = invariant_values["log_token"]

        for category in category_tuple:
            category_total[category] += 1
            category_total_split[category][split] += 1

            if log_token:
                category_total_logs[category].add(log_token)

            area = float(
                category_map[category][
                    "max_bbox_area_fraction"
                ]
            )
            category_total_areas[category].append(area)

            if category_count == 1:
                category_exclusive[category] += 1
                category_exclusive_split[category][split] += 1

                if log_token:
                    category_exclusive_logs[category].add(
                        log_token
                    )

                category_exclusive_areas[category].append(area)
            else:
                category_multitarget[category] += 1


image_rows.sort(
    key=lambda row: (
        row["target_category_count"],
        row["target_category_set"],
        row["dataset_split"],
        row["log_token"],
        row["image_relpath"],
    )
)

fieldnames = [
    "dataset_split",
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
    "target_category_set",
    "target_category_count",
    "is_exclusive_single_category",
    "is_multi_target",
    "exclusive_category",
    "total_target_instances",
    "max_target_bbox_area_fraction",
    "sum_target_bbox_area_fraction",
]

for category in TARGETS:
    fieldnames.extend([
        f"has_{category}",
        f"{category}_instances",
        f"{category}_max_bbox_area_fraction",
        f"{category}_sum_bbox_area_fraction",
    ])

with OUTPUT_MANIFEST.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(image_rows)


report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


exclusive_images = sum(
    row["is_exclusive_single_category"]
    for row in image_rows
)

multi_target_images = sum(
    row["is_multi_target"]
    for row in image_rows
)

two_category_images = sum(
    row["target_category_count"] == 2
    for row in image_rows
)

three_category_images = sum(
    row["target_category_count"] == 3
    for row in image_rows
)

report("=" * 108)
report("nuImages CAM_FRONT independent sampling-pool audit")
report("=" * 108)
report(f"Input category-image rows              : {len(rows)}")
report(f"Unique image-level rows                : {len(image_rows)}")
report(f"Exclusive single-category images       : {exclusive_images}")
report(f"Multi-target images                    : {multi_target_images}")
report(f"Two-category images                    : {two_category_images}")
report(f"Three-category images                  : {three_category_images}")

report("\nTarget-category combinations:")

for combination, count in sorted(
    combination_counts.items(),
    key=lambda item: (
        len(item[0]),
        item[0],
    ),
):
    report(
        f"  {' + '.join(combination):48s}: {count:5d}"
    )

report("\nPer-category availability:")

for category in TARGETS:
    total = category_total[category]
    exclusive = category_exclusive[category]
    multi = category_multitarget[category]

    train_total = category_total_split[category]["v1.0-train"]
    val_total = category_total_split[category]["v1.0-val"]

    train_exclusive = (
        category_exclusive_split[category]["v1.0-train"]
    )
    val_exclusive = (
        category_exclusive_split[category]["v1.0-val"]
    )

    total_median_area = (
        statistics.median(category_total_areas[category])
        if category_total_areas[category]
        else 0.0
    )

    exclusive_median_area = (
        statistics.median(category_exclusive_areas[category])
        if category_exclusive_areas[category]
        else 0.0
    )

    report(
        f"  {category:18s} | "
        f"total={total:5d} | "
        f"exclusive={exclusive:5d} | "
        f"multi={multi:5d} | "
        f"train/val total={train_total:4d}/{val_total:4d} | "
        f"train/val excl={train_exclusive:4d}/{val_exclusive:4d} | "
        f"logs total/excl="
        f"{len(category_total_logs[category]):3d}/"
        f"{len(category_exclusive_logs[category]):3d} | "
        f"median area total/excl="
        f"{100 * total_median_area:.4f}%/"
        f"{100 * exclusive_median_area:.4f}%"
    )

report(f"\nSaved image-level manifest: {OUTPUT_MANIFEST}")

if len(rows) != 9711:
    failures.append(
        f"expected 9711 category-image rows, found {len(rows)}"
    )

if len(image_rows) != 6849:
    failures.append(
        f"expected 6849 unique images, found {len(image_rows)}"
    )

if exclusive_images + multi_target_images != len(image_rows):
    failures.append(
        "exclusive plus multi-target counts do not equal image count"
    )

if failures:
    report("\nFAILURES:")

    for failure in failures[:50]:
        report(f"  - {failure}")

    status = "FAIL"
else:
    status = "PASS"

report(f"\nSTATUS: {status}")

REPORT_PATH.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

if status != "PASS":
    raise SystemExit("Sampling-pool audit failed.")
