#!/usr/bin/env python3

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
PROTOCOL_DIR = ROOT / "protocol"
PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_MANIFEST_SHA = (
    "8a86dfae7ec664b897e4f40b7fd1e401"
    "fff0dc40c00c96a0ef9be98c0314a68c"
)

OUTPUT_CSV = (
    PROTOCOL_DIR /
    "FROZEN_DETECTOR_ARTIFACT_MANIFEST_v1.csv"
)

OUTPUT_JSON = (
    PROTOCOL_DIR /
    "FROZEN_DETECTOR_ARTIFACT_MANIFEST_v1.json"
)

REPORT_PATH = (
    PROTOCOL_DIR /
    "FROZEN_DETECTOR_ARTIFACT_REPORT_v1.txt"
)

CONFIGS = [
    {
        "backbone": "OpenCLIP",
        "encoder": "OpenCLIP ViT-B/16",
        "input_dim": 512,
"main_results_csv": Path(
    os.environ["TAMVLM_OPENCLIP_RESULTS_CSV"]
),
"checkpoint_dir": Path(
    os.environ["TAMVLM_OPENCLIP_CHECKPOINT_DIR"]
),
        "checkpoint_pattern": "clip_mlp_seed{seed}.pt",
    },
    {
        "backbone": "Qwen3-VL-Embedding-2B",
        "encoder": "Qwen3-VL-Embedding-2B",
        "input_dim": 2048,
"main_results_csv": Path(
    os.environ["TAMVLM_QWEN_RESULTS_CSV"]
),
"checkpoint_dir": Path(
    os.environ["TAMVLM_QWEN_CHECKPOINT_DIR"]
),
        "checkpoint_pattern": "qwen3vl_mlp_seed{seed}.pt",
    },
]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def locate_state_dict(obj):
    if not isinstance(obj, dict):
        return None, None

    for key in [
        "state_dict",
        "model_state_dict",
        "detector_state_dict",
        "model",
    ]:
        value = obj.get(key)

        if isinstance(value, dict):
            return key, value

    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return "<raw_state_dict>", obj

    return None, None


def safe_float(value, field_name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid float in {field_name}: {value!r}"
        ) from error

    if not math.isfinite(result):
        raise ValueError(
            f"Non-finite value in {field_name}: {result}"
        )

    return result


artifact_rows = []
json_records = []
failures = []

for config in CONFIGS:
    result_csv = config["main_results_csv"]

    if not result_csv.is_file():
        failures.append(
            f"missing main-results CSV: {result_csv}"
        )
        continue

    result_rows = read_csv(result_csv)

    if len(result_rows) != 5:
        failures.append(
            f"{config['backbone']}: expected 5 result rows, "
            f"found {len(result_rows)}"
        )
        continue

    results_by_seed = {}

    for index, row in enumerate(result_rows):
        raw_seed = row.get("seed", index)

        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            failures.append(
                f"{config['backbone']}: invalid seed {raw_seed!r}"
            )
            continue

        if seed in results_by_seed:
            failures.append(
                f"{config['backbone']}: duplicate seed {seed}"
            )
            continue

        results_by_seed[seed] = row

    if set(results_by_seed) != set(range(5)):
        failures.append(
            f"{config['backbone']}: expected seeds 0–4, found "
            f"{sorted(results_by_seed)}"
        )
        continue

    result_csv_sha = sha256_file(result_csv)

    for seed in range(5):
        result_row = results_by_seed[seed]

        threshold_low_fpr = safe_float(
            result_row.get("threshold_val_1pct_fpr"),
            (
                f"{config['backbone']} seed {seed} "
                "threshold_val_1pct_fpr"
            ),
        )

        threshold_balacc = safe_float(
            result_row.get("threshold_balacc_val"),
            (
                f"{config['backbone']} seed {seed} "
                "threshold_balacc_val"
            ),
        )

        if not 0.0 <= threshold_low_fpr <= 1.0:
            failures.append(
                f"{config['backbone']} seed {seed}: "
                "low-FPR threshold outside [0,1]"
            )

        if not 0.0 <= threshold_balacc <= 1.0:
            failures.append(
                f"{config['backbone']} seed {seed}: "
                "balanced threshold outside [0,1]"
            )

        checkpoint_path = (
            config["checkpoint_dir"] /
            config["checkpoint_pattern"].format(seed=seed)
        )

        if not checkpoint_path.is_file():
            failures.append(
                f"missing checkpoint: {checkpoint_path}"
            )
            continue

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        container_name, state_dict = locate_state_dict(
            checkpoint
        )

        if state_dict is None:
            failures.append(
                f"{config['backbone']} seed {seed}: "
                "state dictionary not recognized"
            )
            continue

        expected_shapes = {
            "net.0.weight": (
                128,
                config["input_dim"],
            ),
            "net.0.bias": (128,),
            "net.3.weight": (1, 128),
            "net.3.bias": (1,),
        }

        actual_shapes = {}

        for parameter_name, expected_shape in expected_shapes.items():
            tensor = state_dict.get(parameter_name)

            if not torch.is_tensor(tensor):
                failures.append(
                    f"{config['backbone']} seed {seed}: "
                    f"missing tensor {parameter_name}"
                )
                continue

            actual_shape = tuple(tensor.shape)
            actual_shapes[parameter_name] = actual_shape

            if actual_shape != expected_shape:
                failures.append(
                    f"{config['backbone']} seed {seed}: "
                    f"{parameter_name} shape {actual_shape}, "
                    f"expected {expected_shape}"
                )

        metadata = (
            checkpoint.get("metadata", {})
            if isinstance(checkpoint, dict)
            else {}
        )

        train_info = (
            checkpoint.get("train_info", {})
            if isinstance(checkpoint, dict)
            else {}
        )

        checkpoint_manifest_sha = metadata.get(
            "manifest_sha256",
            "",
        )

        if checkpoint_manifest_sha != EXPECTED_MANIFEST_SHA:
            failures.append(
                f"{config['backbone']} seed {seed}: "
                "checkpoint manifest SHA mismatch"
            )

        checkpoint_seed = train_info.get("seed")

        if checkpoint_seed is not None:
            try:
                checkpoint_seed = int(checkpoint_seed)
            except (TypeError, ValueError):
                failures.append(
                    f"{config['backbone']} seed {seed}: "
                    "invalid train_info seed"
                )
            else:
                if checkpoint_seed != seed:
                    failures.append(
                        f"{config['backbone']} seed {seed}: "
                        f"checkpoint train_info seed is "
                        f"{checkpoint_seed}"
                    )

        checkpoint_sha = sha256_file(checkpoint_path)

        artifact_row = {
            "backbone": config["backbone"],
            "seed": seed,
            "encoder": metadata.get(
                "encoder",
                config["encoder"],
            ),
            "embedding_dimension": config["input_dim"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_state_container": container_name,
            "source_main_results_csv": str(result_csv),
            "source_main_results_csv_sha256": result_csv_sha,
            "threshold_source_column": (
                "threshold_val_1pct_fpr"
            ),
            "threshold_val_1pct_fpr": threshold_low_fpr,
            "threshold_balacc_val_for_provenance": (
                threshold_balacc
            ),
            "external_audit_threshold_used": (
                threshold_low_fpr
            ),
            "score_function": (
                "sigmoid(MLP_logit)"
            ),
            "alert_comparator": (
                "score > external_audit_threshold_used"
            ),
            "checkpoint_manifest_sha256": (
                checkpoint_manifest_sha
            ),
            "external_retraining_allowed": 0,
            "external_recalibration_allowed": 0,
            "included_in_external_audit": 1,
        }

        artifact_rows.append(artifact_row)

        json_records.append({
            **artifact_row,
            "checkpoint_metadata": metadata,
            "checkpoint_train_info": train_info,
            "parameter_shapes": {
                key: list(value)
                for key, value in actual_shapes.items()
            },
            "complete_source_result_row": result_row,
        })


artifact_rows.sort(
    key=lambda row: (
        row["backbone"],
        int(row["seed"]),
    )
)

json_records.sort(
    key=lambda row: (
        row["backbone"],
        int(row["seed"]),
    )
)

fieldnames = [
    "backbone",
    "seed",
    "encoder",
    "embedding_dimension",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_state_container",
    "source_main_results_csv",
    "source_main_results_csv_sha256",
    "threshold_source_column",
    "threshold_val_1pct_fpr",
    "threshold_balacc_val_for_provenance",
    "external_audit_threshold_used",
    "score_function",
    "alert_comparator",
    "checkpoint_manifest_sha256",
    "external_retraining_allowed",
    "external_recalibration_allowed",
    "included_in_external_audit",
]

with OUTPUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(artifact_rows)

json_payload = {
    "artifact_manifest_version": "v1",
    "purpose": (
        "Frozen controlled-main detector heads and "
        "validation-derived low-FPR thresholds for the "
        "TAM-VLM external real-image audit."
    ),
    "decision_rule": (
        "alert = 1 when sigmoid score is strictly greater "
        "than the seed-specific threshold_val_1pct_fpr"
    ),
    "expected_training_manifest_sha256": (
        EXPECTED_MANIFEST_SHA
    ),
    "number_of_backbones": 2,
    "number_of_seeds_per_backbone": 5,
    "number_of_frozen_detector_heads": len(json_records),
    "excluded_checkpoint_families": [
        "strict LOTO heads",
        "ladder-forensics heads",
        "diversity-scaling heads",
        "ablation heads",
        "baseline models",
    ],
    "records": json_records,
}

OUTPUT_JSON.write_text(
    json.dumps(
        json_payload,
        indent=2,
        sort_keys=False,
    ) + "\n",
    encoding="utf-8",
)


report_lines = []


def report(text=""):
    print(text)
    report_lines.append(text)


report("=" * 112)
report("TAM-VLM frozen detector artifact manifest")
report("=" * 112)
report(
    f"Frozen detector heads recorded : "
    f"{len(artifact_rows)}"
)

for backbone in [
    "OpenCLIP",
    "Qwen3-VL-Embedding-2B",
]:
    rows = [
        row for row in artifact_rows
        if row["backbone"] == backbone
    ]

    report(f"\n{backbone}:")

    for row in rows:
        report(
            f"  seed {row['seed']} | "
            f"d={row['embedding_dimension']} | "
            f"threshold={row['external_audit_threshold_used']:.16f} | "
            f"checkpoint_sha256="
            f"{row['checkpoint_sha256']}"
        )

report("\nDecision rule:")
report(
    "  score = sigmoid(MLP logit); "
    "alert iff score > seed-specific frozen threshold."
)

report("\nExcluded:")
report(
    "  LOTO, ladder-forensics, diversity, ablation, "
    "and baseline checkpoints."
)

report(f"\nSaved CSV : {OUTPUT_CSV}")
report(f"Saved JSON: {OUTPUT_JSON}")

if len(artifact_rows) != 10:
    failures.append(
        f"expected 10 frozen detector heads, "
        f"recorded {len(artifact_rows)}"
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
    raise SystemExit("Detector artifact freeze failed.")
