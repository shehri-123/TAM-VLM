#!/usr/bin/env python3

import csv
import hashlib
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

AUDIT_REPORT = OUTPUT_DIR / "tamvlm_nuimages_overlap_audit_v1.txt"
CLEAN_MANIFEST = OUTPUT_DIR / "nuimages_camfront_candidates_no_tamvlm_overlap_v1.csv"
EXCLUDED_MANIFEST = OUTPUT_DIR / "nuimages_camfront_candidates_excluded_overlap_v1.csv"


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


print("=" * 104)
print("TAM-VLM vs nuImages exact-frame, filename, scene, and log overlap audit")
print("=" * 104)

# ------------------------------------------------------------------
# 1. Load original TAM-VLM manifest.
# ------------------------------------------------------------------

tam_rows = read_csv(TAMVLM_MANIFEST)

null_rows = [
    row for row in tam_rows
    if str(row.get("label", "")).strip() == "0"
]

tam_scene_tokens = {
    row["scene"].strip()
    for row in tam_rows
    if row.get("scene", "").strip()
}

# Filename stems of null controls are expected to preserve source
# sample_data tokens. This will be verified against nuScenes metadata.
null_filename_stems = {
    Path(row["image"]).stem
    for row in null_rows
    if row.get("image", "").strip()
}

print(f"TAM-VLM manifest rows                 : {len(tam_rows)}")
print(f"TAM-VLM null-control rows             : {len(null_rows)}")
print(f"TAM-VLM scene tokens                  : {len(tam_scene_tokens)}")
print(f"Unique null-control filename stems    : {len(null_filename_stems)}")

# ------------------------------------------------------------------
# 2. Load original nuScenes metadata.
# ------------------------------------------------------------------

scene_rows = load_json(NUSCENES_META / "scene.json")
sample_rows = load_json(NUSCENES_META / "sample.json")
sample_data_rows = load_json(NUSCENES_META / "sample_data.json")

scene_by_token = {
    row["token"]: row
    for row in scene_rows
}

sample_by_token = {
    row["token"]: row
    for row in sample_rows
}

sample_data_by_token = {
    row["token"]: row
    for row in sample_data_rows
}

# Verify that null filename stems are real nuScenes sample_data tokens.
matched_null_stems = (
    null_filename_stems &
    set(sample_data_by_token)
)

unmatched_null_stems = (
    null_filename_stems -
    set(sample_data_by_token)
)

print(f"Null stems matched to sample_data     : {len(matched_null_stems)}")
print(f"Null stems unmatched                  : {len(unmatched_null_stems)}")

if unmatched_null_stems:
    print("\nFirst unmatched null stems:")
    for token in sorted(unmatched_null_stems)[:20]:
        print(f"  {token}")

# ------------------------------------------------------------------
# 3. Derive exact original TAM-VLM source information.
# ------------------------------------------------------------------

tam_source_tokens = set()
tam_source_filenames = set()
tam_source_basenames = set()
tam_log_tokens = set()
tam_sample_tokens = set()

source_token_scene_mismatches = []

for sample_data_token in sorted(matched_null_stems):
    sd = sample_data_by_token[sample_data_token]

    sample_token = sd.get("sample_token", "")
    sample = sample_by_token.get(sample_token, {})
    scene_token = sample.get("scene_token", "")

    if scene_token not in tam_scene_tokens:
        source_token_scene_mismatches.append(
            (
                sample_data_token,
                scene_token,
            )
        )
        continue

    scene = scene_by_token.get(scene_token, {})
    log_token = scene.get("log_token", "")
    filename = str(sd.get("filename", "")).strip()

    tam_source_tokens.add(sample_data_token)

    if sample_token:
        tam_sample_tokens.add(sample_token)

    if filename:
        tam_source_filenames.add(filename)
        tam_source_basenames.add(Path(filename).name)

    if log_token:
        tam_log_tokens.add(log_token)

print("-" * 104)
print(f"Verified TAM-VLM source frames        : {len(tam_source_tokens)}")
print(f"Verified TAM-VLM source samples       : {len(tam_sample_tokens)}")
print(f"Verified TAM-VLM source filenames     : {len(tam_source_filenames)}")
print(f"Verified TAM-VLM source basenames     : {len(tam_source_basenames)}")
print(f"TAM-VLM source logs                   : {len(tam_log_tokens)}")
print(
    "Source-token/scene mismatches         : "
    f"{len(source_token_scene_mismatches)}"
)

# ------------------------------------------------------------------
# 4. Load nuImages candidate manifest and classify overlap.
# ------------------------------------------------------------------

candidate_rows = read_csv(CANDIDATE_MANIFEST)

if not candidate_rows:
    raise SystemExit("Candidate manifest is empty.")

base_fieldnames = list(candidate_rows[0].keys())

extra_fields = [
    "overlap_exact_sample_data_token",
    "overlap_exact_sample_token",
    "overlap_exact_image_relpath",
    "overlap_exact_image_basename",
    "overlap_same_log",
    "overlap_any_exact_frame_identity",
    "exclusion_reason",
]

annotated_rows = []
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

for row in candidate_rows:
    category = row.get("category_short", "")
    category_before[category] += 1

    sample_data_token = row.get("sample_data_token", "").strip()
    sample_token = row.get("sample_token", "").strip()
    image_relpath = row.get("image_relpath", "").strip()
    image_basename = Path(image_relpath).name
    log_token = row.get("log_token", "").strip()

    exact_sample_data = sample_data_token in tam_source_tokens
    exact_sample = sample_token in tam_sample_tokens
    exact_relpath = image_relpath in tam_source_filenames
    exact_basename = image_basename in tam_source_basenames
    same_log = log_token in tam_log_tokens

    exact_frame_identity = any(
        [
            exact_sample_data,
            exact_sample,
            exact_relpath,
            exact_basename,
        ]
    )

    reasons = []

    if exact_sample_data:
        reasons.append("exact_sample_data_token")

    if exact_sample:
        reasons.append("exact_sample_token")

    if exact_relpath:
        reasons.append("exact_image_relpath")

    if exact_basename:
        reasons.append("exact_image_basename")

    if same_log:
        reasons.append("same_nuscenes_log")

    row = dict(row)

    row.update({
        "overlap_exact_sample_data_token": int(exact_sample_data),
        "overlap_exact_sample_token": int(exact_sample),
        "overlap_exact_image_relpath": int(exact_relpath),
        "overlap_exact_image_basename": int(exact_basename),
        "overlap_same_log": int(same_log),
        "overlap_any_exact_frame_identity": int(exact_frame_identity),
        "exclusion_reason": ";".join(reasons),
    })

    annotated_rows.append(row)

    unique_images_before.add(image_relpath)

    if log_token:
        logs_before.add(log_token)

    # Strong external-independence protocol:
    # exclude exact source-frame identity OR the same original data log.
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

# ------------------------------------------------------------------
# 5. Save manifests.
# ------------------------------------------------------------------

output_fieldnames = base_fieldnames + [
    field
    for field in extra_fields
    if field not in base_fieldnames
]

write_csv(
    CLEAN_MANIFEST,
    kept_rows,
    output_fieldnames,
)

write_csv(
    EXCLUDED_MANIFEST,
    excluded_rows,
    output_fieldnames,
)

# ------------------------------------------------------------------
# 6. Report.
# ------------------------------------------------------------------

lines = []

def report(text=""):
    print(text)
    lines.append(text)


report("\n" + "=" * 104)
report("OVERLAP RESULTS")
report("=" * 104)

report(f"Candidate category-image rows before exclusion : {len(candidate_rows)}")
report(f"Rows excluded                                  : {len(excluded_rows)}")
report(f"Rows retained                                  : {len(kept_rows)}")

report(f"Unique candidate images before exclusion       : {len(unique_images_before)}")
report(f"Unique images excluded                         : {len(unique_images_excluded)}")
report(f"Unique images retained                         : {len(unique_images_after)}")

report(f"Candidate logs before exclusion                : {len(logs_before)}")
report(f"Overlapping logs excluded                      : {len(logs_excluded)}")
report(f"Non-overlapping logs retained                  : {len(logs_after)}")

report("\nReason counts among category-image rows:")

if reason_counts:
    for reason, count in sorted(reason_counts.items()):
        report(f"  {reason:34s}: {count:6d}")
else:
    report("  No overlaps detected.")

report("\nCategory counts before and after exclusion:")

for category in sorted(category_before):
    before = category_before[category]
    excluded = category_excluded[category]
    after = category_after[category]

    report(
        f"  {category:18s} | "
        f"before={before:5d} | "
        f"excluded={excluded:5d} | "
        f"retained={after:5d}"
    )

report("\nProtocol decision:")
report(
    "  Retained rows contain neither an exact TAM-VLM source-frame "
    "identity nor a TAM-VLM source log."
)

report(f"\nSaved retained manifest : {CLEAN_MANIFEST}")
report(f"Saved excluded manifest : {EXCLUDED_MANIFEST}")

if unmatched_null_stems:
    status = (
        "FAIL — null-control filename stems did not all resolve "
        "to nuScenes sample_data tokens."
    )
elif source_token_scene_mismatches:
    status = (
        "FAIL — some resolved source tokens did not belong to "
        "the 94 TAM-VLM scenes."
    )
elif not kept_rows:
    status = "FAIL — no independent candidate rows remain."
else:
    status = "PASS"

report(f"\nSTATUS: {status}")

AUDIT_REPORT.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

if not status.startswith("PASS"):
    raise SystemExit(status)
