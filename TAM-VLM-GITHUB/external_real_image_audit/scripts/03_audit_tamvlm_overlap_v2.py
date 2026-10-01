#!/usr/bin/env python3

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

AUDIT_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
CANDIDATE_MANIFEST = (
    AUDIT_ROOT /
    "metadata/nuimages_camfront_target_candidates_v1.csv"
)

TAMVLM_MANIFEST = Path(
    os.environ["TAMVLM_CORE_MANIFEST"]
)
NUSCENES_META = Path(
    os.environ["TAMVLM_NUSCENES_META"]
)
OUTPUT_DIR = AUDIT_ROOT / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_REPORT = (
    OUTPUT_DIR /
    "tamvlm_nuimages_overlap_audit_v2.txt"
)

CLEAN_MANIFEST = (
    OUTPUT_DIR /
    "nuimages_camfront_candidates_no_tamvlm_overlap_v2.csv"
)

EXCLUDED_MANIFEST = (
    OUTPUT_DIR /
    "nuimages_camfront_candidates_excluded_overlap_v2.csv"
)


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


print("=" * 108)
print("Corrected TAM-VLM vs nuImages overlap audit")
print("=" * 108)

# ------------------------------------------------------------------
# 1. Load TAM-VLM manifest.
# Null filenames preserve nuScenes SAMPLE tokens, not sample_data tokens.
# ------------------------------------------------------------------

tam_rows = read_csv(TAMVLM_MANIFEST)

null_rows = [
    row for row in tam_rows
    if str(row.get("label", "")).strip() == "0"
]

tam_sample_tokens_from_filenames = {
    Path(row["image"]).stem
    for row in null_rows
    if row.get("image", "").strip()
}

tam_manifest_scene_tokens = {
    row["scene"].strip()
    for row in tam_rows
    if row.get("scene", "").strip()
}

print(f"TAM-VLM manifest rows                    : {len(tam_rows)}")
print(f"TAM-VLM null-control rows                : {len(null_rows)}")
print(
    "Unique sample tokens from null filenames : "
    f"{len(tam_sample_tokens_from_filenames)}"
)
print(
    "TAM-VLM manifest scene tokens             : "
    f"{len(tam_manifest_scene_tokens)}"
)

# ------------------------------------------------------------------
# 2. Load nuScenes metadata.
# ------------------------------------------------------------------

scene_rows = load_json(NUSCENES_META / "scene.json")
sample_rows = load_json(NUSCENES_META / "sample.json")
sample_data_rows = load_json(NUSCENES_META / "sample_data.json")
calibrated_sensor_rows = load_json(
    NUSCENES_META / "calibrated_sensor.json"
)
sensor_rows = load_json(NUSCENES_META / "sensor.json")

scene_by_token = {
    row["token"]: row
    for row in scene_rows
}

sample_by_token = {
    row["token"]: row
    for row in sample_rows
}

calibrated_sensor_by_token = {
    row["token"]: row
    for row in calibrated_sensor_rows
}

sensor_by_token = {
    row["token"]: row
    for row in sensor_rows
}

# Resolve each sample_data row's sensor channel through:
# sample_data -> calibrated_sensor -> sensor -> channel.
sample_data_channel = {}

for sd in sample_data_rows:
    calibrated_token = sd.get("calibrated_sensor_token", "")
    calibrated = calibrated_sensor_by_token.get(
        calibrated_token,
        {}
    )

    sensor_token = calibrated.get("sensor_token", "")
    sensor = sensor_by_token.get(sensor_token, {})

    channel = sensor.get("channel", "")
    sample_data_channel[sd["token"]] = channel

# Index CAM_FRONT sample_data rows by sample token.
cam_front_by_sample = defaultdict(list)

for sd in sample_data_rows:
    channel = sample_data_channel.get(sd["token"], "")

    # Metadata-derived channel is primary.
    # Filename fallback is retained only as a defensive consistency check.
    filename = str(sd.get("filename", ""))

    is_cam_front = (
        channel == "CAM_FRONT"
        or filename.startswith("samples/CAM_FRONT/")
    )

    if is_cam_front:
        cam_front_by_sample[sd.get("sample_token", "")].append(sd)

# ------------------------------------------------------------------
# 3. Recover the exact 3,741 TAM-VLM source frames.
# ------------------------------------------------------------------

resolved_sample_tokens = (
    tam_sample_tokens_from_filenames &
    set(sample_by_token)
)

unresolved_sample_tokens = (
    tam_sample_tokens_from_filenames -
    set(sample_by_token)
)

tam_source_sample_tokens = set()
tam_source_sample_data_tokens = set()
tam_source_filenames = set()
tam_source_basenames = set()
tam_source_scene_tokens = set()
tam_source_log_tokens = set()

missing_cam_front = []
multiple_cam_front = []
scene_mismatches = []

null_scene_by_sample_token = {}

for row in null_rows:
    sample_token = Path(row["image"]).stem
    manifest_scene = row.get("scene", "").strip()

    null_scene_by_sample_token[sample_token] = manifest_scene

for sample_token in sorted(resolved_sample_tokens):
    sample = sample_by_token[sample_token]

    recovered_scene = sample.get("scene_token", "")
    expected_scene = null_scene_by_sample_token.get(
        sample_token,
        ""
    )

    if recovered_scene != expected_scene:
        scene_mismatches.append(
            (
                sample_token,
                expected_scene,
                recovered_scene,
            )
        )

    cam_rows = cam_front_by_sample.get(sample_token, [])

    # Prefer keyframe row if more than one CAM_FRONT row exists.
    keyframe_rows = [
        row for row in cam_rows
        if bool(row.get("is_key_frame", False))
    ]

    selected_rows = keyframe_rows if keyframe_rows else cam_rows

    if len(selected_rows) == 0:
        missing_cam_front.append(sample_token)
        continue

    if len(selected_rows) > 1:
        multiple_cam_front.append(
            (
                sample_token,
                len(selected_rows),
            )
        )
        continue

    sd = selected_rows[0]

    sample_data_token = sd["token"]
    filename = str(sd.get("filename", "")).strip()

    scene = scene_by_token.get(recovered_scene, {})
    log_token = scene.get("log_token", "")

    tam_source_sample_tokens.add(sample_token)
    tam_source_sample_data_tokens.add(sample_data_token)

    if filename:
        tam_source_filenames.add(filename)
        tam_source_basenames.add(Path(filename).name)

    if recovered_scene:
        tam_source_scene_tokens.add(recovered_scene)

    if log_token:
        tam_source_log_tokens.add(log_token)

print("-" * 108)
print(
    "Null stems matching sample tokens          : "
    f"{len(resolved_sample_tokens)}"
)
print(
    "Unresolved null sample tokens              : "
    f"{len(unresolved_sample_tokens)}"
)
print(
    "Recovered CAM_FRONT source frames          : "
    f"{len(tam_source_sample_data_tokens)}"
)
print(
    "Missing CAM_FRONT mappings                 : "
    f"{len(missing_cam_front)}"
)
print(
    "Ambiguous CAM_FRONT mappings               : "
    f"{len(multiple_cam_front)}"
)
print(
    "Manifest/recovered scene mismatches        : "
    f"{len(scene_mismatches)}"
)
print(
    "Recovered TAM-VLM scenes                   : "
    f"{len(tam_source_scene_tokens)}"
)
print(
    "Recovered TAM-VLM logs                     : "
    f"{len(tam_source_log_tokens)}"
)
print(
    "Recovered source filenames                 : "
    f"{len(tam_source_filenames)}"
)

# ------------------------------------------------------------------
# 4. Compare against nuImages candidates.
# ------------------------------------------------------------------

candidate_rows = read_csv(CANDIDATE_MANIFEST)

if not candidate_rows:
    raise SystemExit("FAIL — candidate manifest is empty.")

base_fields = list(candidate_rows[0].keys())

extra_fields = [
    "overlap_exact_sample_token",
    "overlap_exact_sample_data_token",
    "overlap_exact_image_relpath",
    "overlap_exact_image_basename",
    "overlap_same_log",
    "overlap_any_exact_frame_identity",
    "exclusion_reason",
]

kept_rows = []
excluded_rows = []

reason_counts = Counter()
category_before = Counter()
category_after = Counter()
category_excluded = Counter()

unique_images_before = set()
unique_images_after = set()
unique_images_excluded = set()

logs_before = set()
logs_after = set()
logs_excluded = set()

for original_row in candidate_rows:
    row = dict(original_row)

    category = row.get("category_short", "")
    category_before[category] += 1

    sample_token = row.get("sample_token", "").strip()
    sample_data_token = row.get(
        "sample_data_token",
        ""
    ).strip()
    image_relpath = row.get("image_relpath", "").strip()
    image_basename = Path(image_relpath).name
    log_token = row.get("log_token", "").strip()

    exact_sample = (
        sample_token in tam_source_sample_tokens
    )
    exact_sample_data = (
        sample_data_token in tam_source_sample_data_tokens
    )
    exact_relpath = (
        image_relpath in tam_source_filenames
    )
    exact_basename = (
        image_basename in tam_source_basenames
    )
    same_log = (
        log_token in tam_source_log_tokens
    )

    exact_frame_identity = any([
        exact_sample,
        exact_sample_data,
        exact_relpath,
        exact_basename,
    ])

    reasons = []

    if exact_sample:
        reasons.append("exact_sample_token")

    if exact_sample_data:
        reasons.append("exact_sample_data_token")

    if exact_relpath:
        reasons.append("exact_image_relpath")

    if exact_basename:
        reasons.append("exact_image_basename")

    if same_log:
        reasons.append("same_nuscenes_log")

    row.update({
        "overlap_exact_sample_token": int(exact_sample),
        "overlap_exact_sample_data_token": int(
            exact_sample_data
        ),
        "overlap_exact_image_relpath": int(exact_relpath),
        "overlap_exact_image_basename": int(
            exact_basename
        ),
        "overlap_same_log": int(same_log),
        "overlap_any_exact_frame_identity": int(
            exact_frame_identity
        ),
        "exclusion_reason": ";".join(reasons),
    })

    unique_images_before.add(image_relpath)

    if log_token:
        logs_before.add(log_token)

    # Strong independence rule:
    # remove exact source-frame overlap and same-drive/log overlap.
    should_exclude = exact_frame_identity or same_log

    if should_exclude:
        excluded_rows.append(row)
        category_excluded[category] += 1
        unique_images_excluded.add(image_relpath)

        if log_token:
            logs_excluded.add(log_token)

        for reason in reasons:
            reason_counts[reason] += 1
    else:
        kept_rows.append(row)
        category_after[category] += 1
        unique_images_after.add(image_relpath)

        if log_token:
            logs_after.add(log_token)

output_fields = base_fields + [
    field for field in extra_fields
    if field not in base_fields
]

write_csv(
    CLEAN_MANIFEST,
    kept_rows,
    output_fields,
)

write_csv(
    EXCLUDED_MANIFEST,
    excluded_rows,
    output_fields,
)

# ------------------------------------------------------------------
# 5. Final report.
# ------------------------------------------------------------------

report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


report("\n" + "=" * 108)
report("CORRECTED OVERLAP RESULTS")
report("=" * 108)

report(
    f"Candidate category-image rows before exclusion : "
    f"{len(candidate_rows)}"
)
report(
    f"Rows excluded                                  : "
    f"{len(excluded_rows)}"
)
report(
    f"Rows retained                                  : "
    f"{len(kept_rows)}"
)

report(
    f"Unique candidate images before exclusion       : "
    f"{len(unique_images_before)}"
)
report(
    f"Unique images excluded                         : "
    f"{len(unique_images_excluded)}"
)
report(
    f"Unique images retained                         : "
    f"{len(unique_images_after)}"
)

report(
    f"Candidate logs before exclusion                : "
    f"{len(logs_before)}"
)
report(
    f"Overlapping logs excluded                      : "
    f"{len(logs_excluded)}"
)
report(
    f"Non-overlapping logs retained                  : "
    f"{len(logs_after)}"
)

report("\nReason counts among category-image rows:")

if reason_counts:
    for reason, count in sorted(reason_counts.items()):
        report(
            f"  {reason:36s}: {count:6d}"
        )
else:
    report("  No exact-frame or same-log overlap detected.")

report("\nCategory counts before and after exclusion:")

for category in sorted(category_before):
    before = category_before[category]
    excluded = category_excluded[category]
    retained = category_after[category]

    report(
        f"  {category:18s} | "
        f"before={before:5d} | "
        f"excluded={excluded:5d} | "
        f"retained={retained:5d}"
    )

failures = []

if len(resolved_sample_tokens) != 3741:
    failures.append(
        "not all 3,741 null stems resolved as sample tokens"
    )

if unresolved_sample_tokens:
    failures.append("unresolved TAM-VLM sample tokens exist")

if len(tam_source_sample_data_tokens) != 3741:
    failures.append(
        "not all 3,741 CAM_FRONT source frames were recovered"
    )

if missing_cam_front:
    failures.append("missing CAM_FRONT mappings exist")

if multiple_cam_front:
    failures.append("ambiguous CAM_FRONT mappings exist")

if scene_mismatches:
    failures.append(
        "manifest scene tokens disagree with recovered metadata"
    )

if len(tam_source_scene_tokens) != 94:
    failures.append(
        "recovered TAM-VLM scene count is not 94"
    )

if not kept_rows:
    failures.append("no independent candidate rows remain")

if failures:
    status = "FAIL — " + "; ".join(failures)
else:
    status = "PASS"

report("\nProtocol decision:")
report(
    "  Retained candidates exclude exact TAM-VLM source "
    "frames and all candidates belonging to the same "
    "original nuScenes logs."
)

report(f"\nSaved retained manifest : {CLEAN_MANIFEST}")
report(f"Saved excluded manifest : {EXCLUDED_MANIFEST}")
report(f"\nSTATUS: {status}")

AUDIT_REPORT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

if status != "PASS":
    raise SystemExit(status)
