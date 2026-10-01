#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
DATA_ROOT = ROOT / "nuimages"

TARGET_IMAGE_MANIFEST = (
    ROOT /
    "metadata/nuimages_camfront_image_level_pool_v1.csv"
)

TAMVLM_MANIFEST = Path(
    os.environ["TAMVLM_CORE_MANIFEST"]
)
NUSCENES_META = Path(
    os.environ["TAMVLM_NUSCENES_META"]
)
OUTPUT_DIR = ROOT / "metadata"

ALL_CONTROL_PATH = (
    OUTPUT_DIR /
    "nuimages_camfront_object_absent_independent_v1.csv"
)

SAME_LOG_CONTROL_PATH = (
    OUTPUT_DIR /
    "nuimages_camfront_object_absent_same_target_logs_v1.csv"
)

LOG_REPORT_PATH = (
    OUTPUT_DIR /
    "nuimages_camfront_target_control_log_counts_v1.csv"
)

REPORT_PATH = (
    OUTPUT_DIR /
    "nuimages_camfront_control_pool_audit_v1.txt"
)

SPLITS = ["v1.0-train", "v1.0-val"]

TARGET_CATEGORIES = {
    "movable_object.trafficcone",
    "movable_object.barrier",
    "movable_object.debris",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


# ================================================================
# 1. Load frozen target-containing image-level manifest.
# ================================================================

target_manifest_rows = read_csv(TARGET_IMAGE_MANIFEST)

if not target_manifest_rows:
    raise SystemExit("FAIL — target image-level manifest is empty.")

target_manifest_tokens = {
    row["sample_data_token"].strip()
    for row in target_manifest_rows
}

target_manifest_paths = {
    row["image_relpath"].strip()
    for row in target_manifest_rows
}

target_manifest_logs = {
    row["log_token"].strip()
    for row in target_manifest_rows
    if row.get("log_token", "").strip()
}


# ================================================================
# 2. Recover the original TAM-VLM source logs.
# ================================================================

tam_rows = read_csv(TAMVLM_MANIFEST)

tam_null_rows = [
    row for row in tam_rows
    if str(row.get("label", "")).strip() == "0"
]

tam_sample_tokens = {
    Path(row["image"]).stem
    for row in tam_null_rows
}

nuscenes_samples = load_json(NUSCENES_META / "sample.json")
nuscenes_scenes = load_json(NUSCENES_META / "scene.json")

nuscenes_sample_by_token = {
    row["token"]: row
    for row in nuscenes_samples
}

nuscenes_scene_by_token = {
    row["token"]: row
    for row in nuscenes_scenes
}

tam_log_tokens = set()
tam_scene_tokens = set()
unresolved_tam_tokens = []

for sample_token in sorted(tam_sample_tokens):
    sample = nuscenes_sample_by_token.get(sample_token)

    if sample is None:
        unresolved_tam_tokens.append(sample_token)
        continue

    scene_token = sample.get("scene_token", "")
    scene = nuscenes_scene_by_token.get(scene_token, {})
    log_token = scene.get("log_token", "")

    if scene_token:
        tam_scene_tokens.add(scene_token)

    if log_token:
        tam_log_tokens.add(log_token)


# ================================================================
# 3. Enumerate all nuImages train/val CAM_FRONT images.
# ================================================================

all_independent_camfront = []
metadata_target_tokens = set()

missing_image_files = []
missing_sample_records = []

for split in SPLITS:
    split_root = DATA_ROOT / split

    categories = load_json(split_root / "category.json")
    object_anns = load_json(split_root / "object_ann.json")
    sample_data = load_json(split_root / "sample_data.json")
    samples = load_json(split_root / "sample.json")
    logs = load_json(split_root / "log.json")

    category_by_token = {
        row["token"]: row["name"]
        for row in categories
    }

    sample_by_token = {
        row["token"]: row
        for row in samples
    }

    log_by_token = {
        row["token"]: row
        for row in logs
    }

    split_target_tokens = {
        ann["sample_data_token"]
        for ann in object_anns
        if category_by_token.get(ann.get("category_token"))
        in TARGET_CATEGORIES
    }

    metadata_target_tokens.update(split_target_tokens)

    for sd in sample_data:
        filename = str(sd.get("filename", "")).strip()

        if not filename.startswith("samples/CAM_FRONT/"):
            continue

        if not bool(sd.get("is_key_frame", True)):
            continue

        sample_token = sd.get("sample_token", "")
        sample = sample_by_token.get(sample_token)

        if sample is None:
            missing_sample_records.append(sd["token"])
            continue

        log_token = sample.get("log_token", "")
        log = log_by_token.get(log_token, {})

        # Strong external-independence criterion.
        if log_token in tam_log_tokens:
            continue

        image_path = DATA_ROOT / filename

        if not image_path.is_file():
            missing_image_files.append(str(image_path))

        has_target = sd["token"] in split_target_tokens

        all_independent_camfront.append({
            "dataset_split": split,
            "sample_data_token": sd["token"],
            "sample_token": sample_token,
            "log_token": log_token,
            "log_location": log.get("location", ""),
            "logfile": log.get("logfile", ""),
            "timestamp": sd.get("timestamp", ""),
            "image_relpath": filename,
            "image_abspath": str(image_path),
            "image_width": sd.get("width", ""),
            "image_height": sd.get("height", ""),
            "contains_any_target_category": int(has_target),
            "is_object_absent_control": int(not has_target),
            "image_exists": int(image_path.is_file()),
        })


# ================================================================
# 4. Separate target-present and object-absent images.
# ================================================================

metadata_target_rows = [
    row for row in all_independent_camfront
    if row["contains_any_target_category"] == 1
]

control_rows = [
    row for row in all_independent_camfront
    if row["is_object_absent_control"] == 1
]

target_logs = {
    row["log_token"]
    for row in metadata_target_rows
    if row["log_token"]
}

same_target_log_controls = [
    row for row in control_rows
    if row["log_token"] in target_logs
]

control_tokens = {
    row["sample_data_token"]
    for row in control_rows
}

control_paths = {
    row["image_relpath"]
    for row in control_rows
}


# ================================================================
# 5. Assess within-log matching feasibility.
# ================================================================

target_count_by_log = Counter(
    row["log_token"]
    for row in metadata_target_rows
)

control_count_by_log = Counter(
    row["log_token"]
    for row in same_target_log_controls
)

split_by_log = {}
location_by_log = {}

for row in all_independent_camfront:
    log_token = row["log_token"]
    split_by_log[log_token] = row["dataset_split"]
    location_by_log[log_token] = row["log_location"]

log_rows = []
possible_matched_pairs = 0
target_images_in_logs_without_controls = 0
logs_without_controls = 0

for log_token in sorted(target_logs):
    target_count = target_count_by_log[log_token]
    control_count = control_count_by_log[log_token]
    possible_pairs = min(target_count, control_count)

    possible_matched_pairs += possible_pairs

    if control_count == 0:
        logs_without_controls += 1
        target_images_in_logs_without_controls += target_count

    log_rows.append({
        "log_token": log_token,
        "dataset_split": split_by_log.get(log_token, ""),
        "log_location": location_by_log.get(log_token, ""),
        "target_image_count": target_count,
        "object_absent_control_count": control_count,
        "maximum_one_to_one_pairs": possible_pairs,
        "target_matching_fraction": (
            possible_pairs / target_count
            if target_count else 0.0
        ),
    })


# ================================================================
# 6. Save outputs.
# ================================================================

image_fieldnames = [
    "dataset_split",
    "sample_data_token",
    "sample_token",
    "log_token",
    "log_location",
    "logfile",
    "timestamp",
    "image_relpath",
    "image_abspath",
    "image_width",
    "image_height",
    "contains_any_target_category",
    "is_object_absent_control",
    "image_exists",
]

control_rows.sort(
    key=lambda row: (
        row["dataset_split"],
        row["log_token"],
        int(row["timestamp"]) if str(row["timestamp"]).isdigit() else 0,
        row["image_relpath"],
    )
)

same_target_log_controls.sort(
    key=lambda row: (
        row["dataset_split"],
        row["log_token"],
        int(row["timestamp"]) if str(row["timestamp"]).isdigit() else 0,
        row["image_relpath"],
    )
)

write_csv(
    ALL_CONTROL_PATH,
    control_rows,
    image_fieldnames,
)

write_csv(
    SAME_LOG_CONTROL_PATH,
    same_target_log_controls,
    image_fieldnames,
)

write_csv(
    LOG_REPORT_PATH,
    log_rows,
    [
        "log_token",
        "dataset_split",
        "log_location",
        "target_image_count",
        "object_absent_control_count",
        "maximum_one_to_one_pairs",
        "target_matching_fraction",
    ],
)


# ================================================================
# 7. Report and acceptance checks.
# ================================================================

report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


split_total = Counter(
    row["dataset_split"]
    for row in all_independent_camfront
)

split_target = Counter(
    row["dataset_split"]
    for row in metadata_target_rows
)

split_control = Counter(
    row["dataset_split"]
    for row in control_rows
)

failures = []

report("=" * 108)
report("nuImages CAM_FRONT object-absent control-pool audit")
report("=" * 108)

report(
    f"Independent CAM_FRONT images                 : "
    f"{len(all_independent_camfront)}"
)
report(
    f"Target-containing unique images              : "
    f"{len(metadata_target_rows)}"
)
report(
    f"Object-absent control candidates             : "
    f"{len(control_rows)}"
)
report(
    f"Object-absent controls in target logs        : "
    f"{len(same_target_log_controls)}"
)

report(
    f"Independent logs containing targets          : "
    f"{len(target_logs)}"
)
report(
    f"Target logs without any control image        : "
    f"{logs_without_controls}"
)
report(
    f"Target images in logs without controls       : "
    f"{target_images_in_logs_without_controls}"
)

report(
    f"Maximum same-log one-to-one matched pairs    : "
    f"{possible_matched_pairs}"
)
report(
    f"Maximum target matching coverage             : "
    f"{100 * possible_matched_pairs / len(metadata_target_rows):.2f}%"
)

report("\nSplit counts:")

for split in SPLITS:
    report(
        f"  {split:12s} | "
        f"all={split_total[split]:5d} | "
        f"target={split_target[split]:5d} | "
        f"control={split_control[split]:5d}"
    )

report(f"\nRecovered TAM-VLM scenes                    : {len(tam_scene_tokens)}")
report(f"Recovered TAM-VLM logs                      : {len(tam_log_tokens)}")
report(f"Missing extracted image files               : {len(missing_image_files)}")
report(f"Missing nuImages sample records             : {len(missing_sample_records)}")

report(f"\nSaved all controls:")
report(str(ALL_CONTROL_PATH))

report("\nSaved same-target-log controls:")
report(str(SAME_LOG_CONTROL_PATH))

report("\nSaved log matching report:")
report(str(LOG_REPORT_PATH))

if unresolved_tam_tokens:
    failures.append(
        "some TAM-VLM sample tokens did not resolve"
    )

if len(tam_scene_tokens) != 94:
    failures.append(
        "recovered TAM-VLM scene count is not 94"
    )

if len(tam_log_tokens) != 17:
    failures.append(
        "recovered TAM-VLM log count is not 17"
    )

if len(metadata_target_rows) != 6849:
    failures.append(
        f"expected 6849 target images, found "
        f"{len(metadata_target_rows)}"
    )

metadata_target_token_set = {
    row["sample_data_token"]
    for row in metadata_target_rows
}

metadata_target_path_set = {
    row["image_relpath"]
    for row in metadata_target_rows
}

if metadata_target_token_set != target_manifest_tokens:
    failures.append(
        "metadata target tokens do not match frozen target manifest"
    )

if metadata_target_path_set != target_manifest_paths:
    failures.append(
        "metadata target paths do not match frozen target manifest"
    )

if target_manifest_logs != target_logs:
    failures.append(
        "target log set does not match frozen target manifest"
    )

if control_tokens & metadata_target_token_set:
    failures.append(
        "target and control sample_data tokens overlap"
    )

if control_paths & metadata_target_path_set:
    failures.append(
        "target and control image paths overlap"
    )

if missing_image_files:
    failures.append(
        "one or more extracted images are missing"
    )

if missing_sample_records:
    failures.append(
        "one or more sample_data rows lack sample metadata"
    )

if failures:
    report("\nFAILURES:")

    for failure in failures:
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
    raise SystemExit("Control-pool audit failed.")
