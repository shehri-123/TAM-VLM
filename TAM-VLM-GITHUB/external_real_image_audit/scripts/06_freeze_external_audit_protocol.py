#!/usr/bin/env python3

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
PROTOCOL_DIR = ROOT / "protocol"
PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PATH = (
    META /
    "nuimages_camfront_image_level_pool_v1.csv"
)

CONTROL_PATH = (
    META /
    "nuimages_camfront_object_absent_independent_v1.csv"
)

FULL_MANIFEST_PATH = (
    META /
    "nuimages_external_full_pool_manifest_v1.csv"
)

MATCHED_PATH = (
    META /
    "nuimages_external_same_log_matched_pairs_v1.csv"
)

UNMATCHED_TARGET_PATH = (
    META /
    "nuimages_external_unmatched_targets_v1.csv"
)

PROTOCOL_PATH = (
    PROTOCOL_DIR /
    "EXTERNAL_REAL_IMAGE_AUDIT_PROTOCOL_v1.txt"
)

REPORT_PATH = (
    META /
    "nuimages_external_protocol_freeze_report_v1.txt"
)

TARGETS = [
    "traffic_cone",
    "traffic_barrier",
    "debris",
]


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


def timestamp_int(row):
    value = str(row.get("timestamp", "")).strip()

    try:
        return int(value)
    except ValueError:
        return 0


target_rows_raw = read_csv(TARGET_PATH)
control_rows_raw = read_csv(CONTROL_PATH)

if not target_rows_raw:
    raise SystemExit("FAIL — target manifest is empty.")

if not control_rows_raw:
    raise SystemExit("FAIL — control manifest is empty.")


# ================================================================
# 1. Standardize target-containing rows.
# ================================================================

full_rows = []
target_rows = []
control_rows = []

for source in target_rows_raw:
    row = {
        "audit_row_id": source["sample_data_token"],
        "audit_group": "target_present",
        "contains_any_audited_target": 1,
        "is_target_category_absent_control": 0,
        "dataset_split": source["dataset_split"],
        "sample_data_token": source["sample_data_token"],
        "sample_token": source["sample_token"],
        "log_token": source["log_token"],
        "log_location": source["log_location"],
        "logfile": source["logfile"],
        "timestamp": source["timestamp"],
        "image_relpath": source["image_relpath"],
        "image_abspath": source["image_abspath"],
        "image_width": source["image_width"],
        "image_height": source["image_height"],
        "target_category_set": source["target_category_set"],
        "target_category_count": source["target_category_count"],
        "is_exclusive_single_category": source[
            "is_exclusive_single_category"
        ],
        "is_multi_target": source["is_multi_target"],
        "exclusive_category": source["exclusive_category"],
        "total_target_instances": source[
            "total_target_instances"
        ],
        "max_target_bbox_area_fraction": source[
            "max_target_bbox_area_fraction"
        ],
        "sum_target_bbox_area_fraction": source[
            "sum_target_bbox_area_fraction"
        ],
        "source_frame_independent_from_tamvlm": 1,
        "source_log_independent_from_tamvlm": 1,
        "included_in_primary_full_pool": 1,
    }

    for category in TARGETS:
        row[f"has_{category}"] = source[
            f"has_{category}"
        ]
        row[f"{category}_instances"] = source[
            f"{category}_instances"
        ]
        row[
            f"{category}_max_bbox_area_fraction"
        ] = source[
            f"{category}_max_bbox_area_fraction"
        ]
        row[
            f"{category}_sum_bbox_area_fraction"
        ] = source[
            f"{category}_sum_bbox_area_fraction"
        ]

    target_rows.append(row)
    full_rows.append(row)


# ================================================================
# 2. Standardize target-category-absent controls.
# ================================================================

for source in control_rows_raw:
    row = {
        "audit_row_id": source["sample_data_token"],
        "audit_group": "target_category_absent_control",
        "contains_any_audited_target": 0,
        "is_target_category_absent_control": 1,
        "dataset_split": source["dataset_split"],
        "sample_data_token": source["sample_data_token"],
        "sample_token": source["sample_token"],
        "log_token": source["log_token"],
        "log_location": source["log_location"],
        "logfile": source["logfile"],
        "timestamp": source["timestamp"],
        "image_relpath": source["image_relpath"],
        "image_abspath": source["image_abspath"],
        "image_width": source["image_width"],
        "image_height": source["image_height"],
        "target_category_set": "",
        "target_category_count": 0,
        "is_exclusive_single_category": 0,
        "is_multi_target": 0,
        "exclusive_category": "",
        "total_target_instances": 0,
        "max_target_bbox_area_fraction": 0.0,
        "sum_target_bbox_area_fraction": 0.0,
        "source_frame_independent_from_tamvlm": 1,
        "source_log_independent_from_tamvlm": 1,
        "included_in_primary_full_pool": 1,
    }

    for category in TARGETS:
        row[f"has_{category}"] = 0
        row[f"{category}_instances"] = 0
        row[f"{category}_max_bbox_area_fraction"] = 0.0
        row[f"{category}_sum_bbox_area_fraction"] = 0.0

    control_rows.append(row)
    full_rows.append(row)


fieldnames = [
    "audit_row_id",
    "audit_group",
    "contains_any_audited_target",
    "is_target_category_absent_control",
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

fieldnames.extend([
    "source_frame_independent_from_tamvlm",
    "source_log_independent_from_tamvlm",
    "included_in_primary_full_pool",
])

full_rows.sort(
    key=lambda row: (
        row["dataset_split"],
        row["log_token"],
        timestamp_int(row),
        row["audit_group"],
        row["image_relpath"],
    )
)

write_csv(
    FULL_MANIFEST_PATH,
    full_rows,
    fieldnames,
)


# ================================================================
# 3. Deterministic nearest-time same-log matching without reuse.
#
# Pairing uses only log identity and timestamps. No model score,
# prediction, embedding, label outcome, or threshold is inspected.
# ================================================================

targets_by_log = defaultdict(list)
controls_by_log = defaultdict(list)

for row in target_rows:
    targets_by_log[row["log_token"]].append(row)

for row in control_rows:
    controls_by_log[row["log_token"]].append(row)

matched_pairs = []
matched_target_tokens = set()
matched_control_tokens = set()

pair_index = 0

for log_token in sorted(targets_by_log):
    log_targets = sorted(
        targets_by_log[log_token],
        key=lambda row: (
            timestamp_int(row),
            row["sample_data_token"],
        ),
    )

    log_controls = sorted(
        controls_by_log.get(log_token, []),
        key=lambda row: (
            timestamp_int(row),
            row["sample_data_token"],
        ),
    )

    if not log_controls:
        continue

    candidate_pairs = []

    for target in log_targets:
        target_time = timestamp_int(target)

        for control in log_controls:
            control_time = timestamp_int(control)

            candidate_pairs.append((
                abs(target_time - control_time),
                target["sample_data_token"],
                control["sample_data_token"],
                target,
                control,
            ))

    candidate_pairs.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    maximum_pairs = min(
        len(log_targets),
        len(log_controls),
    )

    log_pair_count = 0

    for (
        time_difference,
        target_token,
        control_token,
        target,
        control,
    ) in candidate_pairs:

        if target_token in matched_target_tokens:
            continue

        if control_token in matched_control_tokens:
            continue

        matched_target_tokens.add(target_token)
        matched_control_tokens.add(control_token)

        pair_index += 1
        log_pair_count += 1

        matched_pairs.append({
            "pair_id": f"pair_{pair_index:06d}",
            "dataset_split": target["dataset_split"],
            "log_token": log_token,
            "log_location": target["log_location"],
            "absolute_time_difference_microseconds": (
                time_difference
            ),
            "absolute_time_difference_seconds": (
                time_difference / 1_000_000.0
            ),
            "target_sample_data_token": target[
                "sample_data_token"
            ],
            "target_sample_token": target[
                "sample_token"
            ],
            "target_timestamp": target["timestamp"],
            "target_image_relpath": target[
                "image_relpath"
            ],
            "target_image_abspath": target[
                "image_abspath"
            ],
            "target_category_set": target[
                "target_category_set"
            ],
            "target_category_count": target[
                "target_category_count"
            ],
            "target_is_exclusive_single_category": target[
                "is_exclusive_single_category"
            ],
            "target_is_multi_target": target[
                "is_multi_target"
            ],
            "target_exclusive_category": target[
                "exclusive_category"
            ],
            "target_max_bbox_area_fraction": target[
                "max_target_bbox_area_fraction"
            ],
            "target_has_traffic_cone": target[
                "has_traffic_cone"
            ],
            "target_has_traffic_barrier": target[
                "has_traffic_barrier"
            ],
            "target_has_debris": target[
                "has_debris"
            ],
            "control_sample_data_token": control[
                "sample_data_token"
            ],
            "control_sample_token": control[
                "sample_token"
            ],
            "control_timestamp": control["timestamp"],
            "control_image_relpath": control[
                "image_relpath"
            ],
            "control_image_abspath": control[
                "image_abspath"
            ],
            "matching_rule": (
                "greedy_nearest_timestamp_within_log_"
                "without_replacement"
            ),
            "matching_uses_model_scores": 0,
        })

        if log_pair_count == maximum_pairs:
            break


matched_fieldnames = [
    "pair_id",
    "dataset_split",
    "log_token",
    "log_location",
    "absolute_time_difference_microseconds",
    "absolute_time_difference_seconds",
    "target_sample_data_token",
    "target_sample_token",
    "target_timestamp",
    "target_image_relpath",
    "target_image_abspath",
    "target_category_set",
    "target_category_count",
    "target_is_exclusive_single_category",
    "target_is_multi_target",
    "target_exclusive_category",
    "target_max_bbox_area_fraction",
    "target_has_traffic_cone",
    "target_has_traffic_barrier",
    "target_has_debris",
    "control_sample_data_token",
    "control_sample_token",
    "control_timestamp",
    "control_image_relpath",
    "control_image_abspath",
    "matching_rule",
    "matching_uses_model_scores",
]

write_csv(
    MATCHED_PATH,
    matched_pairs,
    matched_fieldnames,
)


# ================================================================
# 4. Save unmatched target images.
# ================================================================

unmatched_targets = [
    row for row in target_rows
    if row["sample_data_token"] not in matched_target_tokens
]

unmatched_targets.sort(
    key=lambda row: (
        row["dataset_split"],
        row["log_token"],
        timestamp_int(row),
        row["image_relpath"],
    )
)

write_csv(
    UNMATCHED_TARGET_PATH,
    unmatched_targets,
    fieldnames,
)


# ================================================================
# 5. Acceptance checks and frozen protocol.
# ================================================================

failures = []

full_tokens = [
    row["sample_data_token"]
    for row in full_rows
]

full_paths = [
    row["image_relpath"]
    for row in full_rows
]

if len(target_rows) != 6849:
    failures.append(
        f"expected 6849 target images, found {len(target_rows)}"
    )

if len(control_rows) != 9587:
    failures.append(
        f"expected 9587 controls, found {len(control_rows)}"
    )

if len(full_rows) != 16436:
    failures.append(
        f"expected 16436 full-pool rows, found {len(full_rows)}"
    )

if len(set(full_tokens)) != len(full_tokens):
    failures.append(
        "duplicate sample_data tokens in full pool"
    )

if len(set(full_paths)) != len(full_paths):
    failures.append(
        "duplicate image paths in full pool"
    )

if len(matched_pairs) != 5796:
    failures.append(
        f"expected 5796 matched pairs, found {len(matched_pairs)}"
    )

if len(matched_target_tokens) != len(matched_pairs):
    failures.append(
        "matched target reuse detected"
    )

if len(matched_control_tokens) != len(matched_pairs):
    failures.append(
        "matched control reuse detected"
    )

if len(unmatched_targets) != 1053:
    failures.append(
        f"expected 1053 unmatched targets, "
        f"found {len(unmatched_targets)}"
    )

for pair in matched_pairs:
    if pair["dataset_split"] not in {
        "v1.0-train",
        "v1.0-val",
    }:
        failures.append(
            "unexpected release split in matched pairs"
        )
        break

    if int(pair["matching_uses_model_scores"]) != 0:
        failures.append(
            "matching improperly marked as score-dependent"
        )
        break


split_counts = Counter(
    row["dataset_split"]
    for row in full_rows
)

target_split_counts = Counter(
    row["dataset_split"]
    for row in target_rows
)

control_split_counts = Counter(
    row["dataset_split"]
    for row in control_rows
)

matched_split_counts = Counter(
    row["dataset_split"]
    for row in matched_pairs
)

category_counts = {
    category: sum(
        int(row[f"has_{category}"])
        for row in target_rows
    )
    for category in TARGETS
}

exclusive_counts = Counter(
    row["exclusive_category"]
    for row in target_rows
    if int(row["is_exclusive_single_category"]) == 1
)

protocol_text = f"""\
TAM-VLM External Real-Image Hard-Negative and Object-Response Audit
Frozen Protocol v1

1. Scope
This is a benign external real-image audit. The nuImages photographs contain
legitimate naturally occurring objects and are not physical backdoor attacks.
The experiment therefore does not estimate attack detection TPR, attack success
rate, victim-model mitigation, or physical-world attack robustness.

2. Dataset
Source: nuImages train and validation metadata releases.
Camera: CAM_FRONT keyframes only.
Primary full pool: {len(full_rows)} unique images.
Target-containing images: {len(target_rows)}.
Target-category-absent controls: {len(control_rows)}.

The phrase target-category-absent means that traffic cone, traffic barrier, and
debris annotations are absent. It does not mean the scene contains no objects.

3. Independence
All retained images are independent of the original TAM-VLM benchmark at:
- exact nuScenes source-frame identity;
- source sample identity;
- source filename identity; and
- original nuScenes log/drive identity.

Original TAM-VLM recovery:
- 3,741 source CAM_FRONT frames;
- 94 scenes;
- 17 logs.

External retained pool:
- 399 logs;
- zero overlap with the 17 TAM-VLM logs.

4. Audited categories
- traffic cone: {category_counts['traffic_cone']} images;
- traffic barrier: {category_counts['traffic_barrier']} images;
- debris: {category_counts['debris']} images.

Images may contain more than one audited category. Every image is encoded and
scored only once. Category-specific analyses use multi-label membership rather
than duplicating model inference.

Exclusive-only sensitivity subsets:
- traffic cone: {exclusive_counts['traffic_cone']} images;
- traffic barrier: {exclusive_counts['traffic_barrier']} images;
- debris: {exclusive_counts['debris']} images.

5. Frozen detector protocol
Separate frozen OpenCLIP and Qwen detector pipelines are evaluated.
For each backbone:
- the visual encoder remains frozen;
- the trained TAM-VLM detection head remains frozen;
- the original validation-derived threshold remains frozen;
- no nuImages image is used for training, checkpoint selection, threshold
  calibration, category selection, or hyperparameter tuning;
- no threshold is recalibrated on the external audit.

6. Primary analysis
The primary full-pool analysis includes all {len(full_rows)} unique images:
- {len(target_rows)} legitimate target-containing images;
- {len(control_rows)} target-category-absent controls.

Report:
- observed alert count and alert rate;
- finite-sample 95% confidence intervals;
- results separately for OpenCLIP and Qwen;
- combined and nuImages-release-split-stratified results;
- log-clustered uncertainty for differences between groups.

7. Same-log matched sensitivity analysis
A deterministic score-blind matching rule pairs target images with controls from
the same nuImages log using nearest timestamp without replacement.

Matched pairs: {len(matched_pairs)}.
Target coverage: {100 * len(matched_pairs) / len(target_rows):.2f}%.
Unmatched target images: {len(unmatched_targets)}.

Matching uses only log identity, timestamps, and stable token tie-breakers.
No embedding, detector score, alert, model output, or test result is used.

8. Category analyses
Report alert rates for traffic cone, traffic barrier, and debris using all images
carrying each category. Because categories overlap, category estimates are not
treated as independent samples.

Report exclusive single-category subsets as a sensitivity analysis. The
exclusive debris subset is small and must be described as limited-sample
evidence.

9. Claim boundaries
Allowed wording:
- external benign hard-negative audit;
- real-image object-response audit;
- observed alert or false-alert rate on legitimate objects;
- target-present versus target-category-absent response difference.

Disallowed wording:
- physical attack detection TPR;
- physical backdoor attack validation;
- attack-success-rate reduction;
- victim-model mitigation;
- guaranteed zero false-positive risk.

A zero observed count must be described as no observed alerts in a finite sample,
not as a guaranteed zero deployment rate.

10. Frozen files
Primary manifest:
{FULL_MANIFEST_PATH}

Same-log matched pairs:
{MATCHED_PATH}

Unmatched target manifest:
{UNMATCHED_TARGET_PATH}
"""

PROTOCOL_PATH.write_text(
    protocol_text,
    encoding="utf-8",
)


# ================================================================
# 6. Report.
# ================================================================

report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


report("=" * 108)
report("TAM-VLM external real-image audit protocol freeze")
report("=" * 108)

report(f"Primary full-pool images                  : {len(full_rows)}")
report(f"Target-containing images                  : {len(target_rows)}")
report(f"Target-category-absent controls           : {len(control_rows)}")
report(f"Same-log matched pairs                    : {len(matched_pairs)}")
report(
    f"Matched target coverage                   : "
    f"{100 * len(matched_pairs) / len(target_rows):.2f}%"
)
report(f"Unmatched target images                   : {len(unmatched_targets)}")

report("\nFull-pool split counts:")

for split in ["v1.0-train", "v1.0-val"]:
    report(
        f"  {split:12s} | "
        f"all={split_counts[split]:5d} | "
        f"target={target_split_counts[split]:5d} | "
        f"control={control_split_counts[split]:5d} | "
        f"matched pairs={matched_split_counts[split]:5d}"
    )

report("\nCategory counts:")

for category in TARGETS:
    report(
        f"  {category:18s} | "
        f"all membership={category_counts[category]:5d} | "
        f"exclusive={exclusive_counts[category]:5d}"
    )

report(f"\nSaved full-pool manifest:")
report(str(FULL_MANIFEST_PATH))

report("\nSaved same-log matched pairs:")
report(str(MATCHED_PATH))

report("\nSaved unmatched targets:")
report(str(UNMATCHED_TARGET_PATH))

report("\nSaved frozen protocol:")
report(str(PROTOCOL_PATH))

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
    raise SystemExit("Protocol freeze failed.")
