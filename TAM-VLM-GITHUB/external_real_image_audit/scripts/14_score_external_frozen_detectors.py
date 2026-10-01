#!/usr/bin/env python3

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)

MANIFEST_PATH = (
    ROOT / "metadata/"
    "nuimages_external_full_pool_manifest_v1.csv"
)

ARTIFACT_MANIFEST = (
    ROOT / "protocol/"
    "FROZEN_DETECTOR_ARTIFACT_MANIFEST_v1.csv"
)

SCORING_EQUIVALENCE = (
    ROOT / "protocol/"
    "DETECTOR_SCORING_EQUIVALENCE_v1.json"
)

OUTPUT_DIR = ROOT / "results/frozen_detector_scoring_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORES_NPZ = (
    OUTPUT_DIR /
    "nuimages_external_frozen_detector_scores_v1.npz"
)

SCORES_CSV = (
    OUTPUT_DIR /
    "nuimages_external_frozen_detector_scores_v1.csv"
)

HEAD_SUMMARY_CSV = (
    OUTPUT_DIR /
    "nuimages_external_detector_head_summary_v1.csv"
)

BACKBONE_SUMMARY_CSV = (
    OUTPUT_DIR /
    "nuimages_external_detector_backbone_summary_v1.csv"
)

SPLIT_SUMMARY_CSV = (
    OUTPUT_DIR /
    "nuimages_external_detector_split_summary_v1.csv"
)

METADATA_JSON = (
    OUTPUT_DIR /
    "nuimages_external_frozen_detector_scoring_metadata_v1.json"
)

REPORT_TXT = (
    OUTPUT_DIR /
    "nuimages_external_frozen_detector_scoring_report_v1.txt"
)

EXPECTED_MANIFEST_SHA256 = (
    "ebcd9203c9fa41b01635f7182dd00fe0d"
    "419d04a8a1f03ef764677222def25ab"
)

EXPECTED_OPENCLIP_EMBEDDINGS_SHA256 = (
    "55adc31c20ca6a41e7fbd8b9b9a3193c"
    "86f29cccb692088a1f1e024f62c60d9a"
)

EXPECTED_QWEN_EMBEDDINGS_SHA256 = (
    "3c88cbb25c78896e7e21fee862b8d225"
    "689d5440baca27d5babe648a8257b6b8"
)

EXPECTED_ROWS = 16436
EXPECTED_TARGET_ROWS = 6849
EXPECTED_CONTROL_ROWS = 9587
EXPECTED_TRAIN_ROWS = 13187
EXPECTED_VAL_ROWS = 3249

BATCH_SIZE = 4096

BACKBONES = [
    {
        "name": "OpenCLIP",
        "slug": "openclip",
        "input_dim": 512,
        "embeddings": (
            ROOT / "embeddings/openclip_v1/"
            "nuimages_external_openclip_embeddings_v1.npy"
        ),
        "index": (
            ROOT / "embeddings/openclip_v1/"
            "nuimages_external_openclip_embedding_index_v1.csv"
        ),
        "expected_embeddings_sha256": (
            EXPECTED_OPENCLIP_EMBEDDINGS_SHA256
        ),
    },
    {
        "name": "Qwen3-VL-Embedding-2B",
        "slug": "qwen3vl",
        "input_dim": 2048,
        "embeddings": (
            ROOT / "embeddings/qwen_v1/final_v1/"
            "nuimages_external_qwen_embeddings_v1.npy"
        ),
        "index": (
            ROOT / "embeddings/qwen_v1/final_v1/"
            "nuimages_external_qwen_embedding_index_v1.csv"
        ),
        "expected_embeddings_sha256": (
            EXPECTED_QWEN_EMBEDDINGS_SHA256
        ),
    },
]


class Detector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features)


def sha256_file(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(path, payload):
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def score_array(model, features, device):
    output = np.empty(
        len(features),
        dtype=np.float32,
    )

    model.eval()

    with torch.inference_mode():
        for start in range(
            0,
            len(features),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(features),
            )

            batch = torch.as_tensor(
                np.asarray(features[start:stop]),
                dtype=torch.float32,
                device=device,
            )

            output[start:stop] = (
                torch.sigmoid(
                    model(batch).squeeze(-1)
                )
                .cpu()
                .numpy()
            )

    return output


def safe_rate(alerts, mask):
    count = int(mask.sum())

    if count == 0:
        return float("nan"), 0

    alert_count = int(alerts[mask].sum())

    return (
        float(alert_count / count),
        alert_count,
    )


print("=" * 112)
print("TAM-VLM external frozen-detector scoring")
print("=" * 112)

for required_path in [
    MANIFEST_PATH,
    ARTIFACT_MANIFEST,
    SCORING_EQUIVALENCE,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

manifest_sha256 = sha256_file(MANIFEST_PATH)

print("Manifest SHA256:", manifest_sha256)

if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen full-pool manifest SHA256 mismatch."
    )

equivalence = json.loads(
    SCORING_EQUIVALENCE.read_text(
        encoding="utf-8"
    )
)

if equivalence.get("status") != "PASS":
    raise RuntimeError(
        "Detector scoring-equivalence audit "
        "does not report PASS."
    )

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} manifest rows, "
        f"found {len(manifest)}."
    )

if not manifest["audit_row_id"].is_unique:
    raise RuntimeError(
        "audit_row_id is not unique."
    )

if not manifest["image_abspath"].is_unique:
    raise RuntimeError(
        "image_abspath is not unique."
    )

target_mask = (
    manifest["contains_any_audited_target"]
    .astype(int)
    .to_numpy()
    .astype(bool)
)

control_mask = (
    manifest["is_target_category_absent_control"]
    .astype(int)
    .to_numpy()
    .astype(bool)
)

if np.any(target_mask & control_mask):
    raise RuntimeError(
        "Target and control groups overlap."
    )

if not np.all(target_mask | control_mask):
    raise RuntimeError(
        "Full pool contains unassigned images."
    )

if int(target_mask.sum()) != EXPECTED_TARGET_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_TARGET_ROWS} target rows, "
        f"found {int(target_mask.sum())}."
    )

if int(control_mask.sum()) != EXPECTED_CONTROL_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_CONTROL_ROWS} controls, "
        f"found {int(control_mask.sum())}."
    )

split_values = (
    manifest["dataset_split"]
    .astype(str)
    .to_numpy()
)

if int(np.sum(split_values == "v1.0-train")) != EXPECTED_TRAIN_ROWS:
    raise RuntimeError(
        "Unexpected v1.0-train row count."
    )

if int(np.sum(split_values == "v1.0-val")) != EXPECTED_VAL_ROWS:
    raise RuntimeError(
        "Unexpected v1.0-val row count."
    )

artifact_rows = pd.read_csv(
    ARTIFACT_MANIFEST
)

artifact_rows = artifact_rows[
    artifact_rows["included_in_external_audit"]
    .astype(int)
    .eq(1)
].copy()

if len(artifact_rows) != 10:
    raise RuntimeError(
        f"Expected 10 frozen detector artifacts, "
        f"found {len(artifact_rows)}."
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("GPU            :", torch.cuda.get_device_name(0))
print("Batch size     :", BATCH_SIZE)
print("Score function : sigmoid(MLP logit)")
print("Alert rule     : score > frozen seed threshold")
print("Retraining     : none")
print("Recalibration  : none")

all_scores = {}
all_alerts = {}
all_thresholds = {}

head_summary_rows = []
split_summary_rows = []

wide_output = manifest.copy()

scoring_start = time.perf_counter()

for backbone in BACKBONES:
    print("\n" + "-" * 112)
    print(backbone["name"])
    print("-" * 112)

    for path in [
        backbone["embeddings"],
        backbone["index"],
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    embeddings_sha256 = sha256_file(
        backbone["embeddings"]
    )

    print(
        "Embeddings SHA256:",
        embeddings_sha256,
    )

    if (
        embeddings_sha256 !=
        backbone["expected_embeddings_sha256"]
    ):
        raise RuntimeError(
            f"{backbone['name']}: embedding "
            "SHA256 mismatch."
        )

    features = np.load(
        backbone["embeddings"],
        mmap_mode="r",
    )

    if features.shape != (
        EXPECTED_ROWS,
        backbone["input_dim"],
    ):
        raise RuntimeError(
            f"{backbone['name']}: unexpected "
            f"embedding shape {features.shape}."
        )

    if features.dtype != np.float32:
        raise RuntimeError(
            f"{backbone['name']}: unexpected "
            f"embedding dtype {features.dtype}."
        )

    index_frame = pd.read_csv(
        backbone["index"]
    )

    if len(index_frame) != EXPECTED_ROWS:
        raise RuntimeError(
            f"{backbone['name']}: index row "
            "count mismatch."
        )

    if not np.array_equal(
        index_frame["embedding_index"]
        .astype(int)
        .to_numpy(),
        np.arange(
            EXPECTED_ROWS,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            f"{backbone['name']}: embedding "
            "indices are not sequential."
        )

    if not np.array_equal(
        index_frame["audit_row_id"]
        .astype(str)
        .to_numpy(),
        manifest["audit_row_id"]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"{backbone['name']}: audit-row "
            "alignment mismatch."
        )

    if not np.array_equal(
        index_frame["image_abspath"]
        .astype(str)
        .to_numpy(),
        manifest["image_abspath"]
        .astype(str)
        .to_numpy(),
    ):
        raise RuntimeError(
            f"{backbone['name']}: image-path "
            "alignment mismatch."
        )

    frozen_rows = artifact_rows[
        artifact_rows["backbone"]
        .astype(str)
        .eq(backbone["name"])
    ].copy()

    frozen_rows["seed"] = (
        frozen_rows["seed"].astype(int)
    )

    frozen_rows = frozen_rows.sort_values(
        "seed"
    )

    if frozen_rows["seed"].tolist() != list(range(5)):
        raise RuntimeError(
            f"{backbone['name']}: frozen seeds "
            "are not exactly 0–4."
        )

    backbone_scores = np.empty(
        (5, EXPECTED_ROWS),
        dtype=np.float32,
    )

    backbone_alerts = np.empty(
        (5, EXPECTED_ROWS),
        dtype=np.uint8,
    )

    backbone_thresholds = np.empty(
        5,
        dtype=np.float64,
    )

    for _, artifact in frozen_rows.iterrows():
        seed = int(artifact["seed"])

        checkpoint_path = Path(
            artifact["checkpoint_path"]
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                checkpoint_path
            )

        checkpoint_sha256 = sha256_file(
            checkpoint_path
        )

        if (
            checkpoint_sha256 !=
            str(artifact["checkpoint_sha256"])
        ):
            raise RuntimeError(
                f"{backbone['name']} seed {seed}: "
                "checkpoint SHA256 mismatch."
            )

        threshold = float(
            artifact[
                "external_audit_threshold_used"
            ]
        )

        if not 0.0 <= threshold <= 1.0:
            raise RuntimeError(
                f"{backbone['name']} seed {seed}: "
                "invalid threshold."
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "state_dict"
        )

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                f"{backbone['name']} seed {seed}: "
                "state_dict missing."
            )

        model = Detector(
            input_dim=backbone["input_dim"],
            hidden_dim=128,
            dropout=0.2,
        ).to(device)

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        scores = score_array(
            model,
            features,
            device,
        )

        if not np.isfinite(scores).all():
            raise RuntimeError(
                f"{backbone['name']} seed {seed}: "
                "non-finite scores."
            )

        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise RuntimeError(
                f"{backbone['name']} seed {seed}: "
                "scores outside [0,1]."
            )

        alerts = (
            scores > threshold
        ).astype(np.uint8)

        backbone_scores[seed] = scores
        backbone_alerts[seed] = alerts
        backbone_thresholds[seed] = threshold

        wide_output[
            f"{backbone['slug']}_seed{seed}_score"
        ] = scores

        wide_output[
            f"{backbone['slug']}_seed{seed}_alert"
        ] = alerts

        full_mask = np.ones(
            EXPECTED_ROWS,
            dtype=bool,
        )

        full_rate, full_alerts = safe_rate(
            alerts,
            full_mask,
        )

        target_rate, target_alerts = safe_rate(
            alerts,
            target_mask,
        )

        control_rate, control_alerts = safe_rate(
            alerts,
            control_mask,
        )

        difference = target_rate - control_rate

        head_summary_rows.append({
            "backbone": backbone["name"],
            "seed": seed,
            "threshold": threshold,
            "n_full_pool": EXPECTED_ROWS,
            "alerts_full_pool": full_alerts,
            "observed_alert_rate_full_pool": (
                full_rate
            ),
            "n_target_containing": (
                EXPECTED_TARGET_ROWS
            ),
            "alerts_target_containing": (
                target_alerts
            ),
            "observed_alert_rate_target_containing": (
                target_rate
            ),
            "n_target_category_absent_control": (
                EXPECTED_CONTROL_ROWS
            ),
            "alerts_target_category_absent_control": (
                control_alerts
            ),
            "observed_false_alert_rate_control": (
                control_rate
            ),
            "rate_difference_target_minus_control": (
                difference
            ),
            "mean_score_target_containing": float(
                scores[target_mask].mean()
            ),
            "median_score_target_containing": float(
                np.median(scores[target_mask])
            ),
            "mean_score_control": float(
                scores[control_mask].mean()
            ),
            "median_score_control": float(
                np.median(scores[control_mask])
            ),
        })

        for split in [
            "v1.0-train",
            "v1.0-val",
        ]:
            split_mask = split_values == split

            for group_name, group_mask in [
                (
                    "target_containing",
                    target_mask,
                ),
                (
                    "target_category_absent_control",
                    control_mask,
                ),
            ]:
                combined_mask = (
                    split_mask & group_mask
                )

                rate, alert_count = safe_rate(
                    alerts,
                    combined_mask,
                )

                split_summary_rows.append({
                    "backbone": backbone["name"],
                    "seed": seed,
                    "dataset_split": split,
                    "audit_group": group_name,
                    "n_images": int(
                        combined_mask.sum()
                    ),
                    "n_alerts": alert_count,
                    "observed_alert_rate": rate,
                })

        print(
            f"seed {seed} | "
            f"threshold={threshold:.9f} | "
            f"target alert={target_rate:.4%} | "
            f"control false alert={control_rate:.4%} | "
            f"difference={difference:+.4%}"
        )

        del model
        torch.cuda.empty_cache()

    all_scores[backbone["slug"]] = (
        backbone_scores
    )

    all_alerts[backbone["slug"]] = (
        backbone_alerts
    )

    all_thresholds[backbone["slug"]] = (
        backbone_thresholds
    )


np.savez_compressed(
    SCORES_NPZ,
    openclip_scores=all_scores["openclip"],
    openclip_alerts=all_alerts["openclip"],
    openclip_thresholds=all_thresholds[
        "openclip"
    ],
    qwen3vl_scores=all_scores["qwen3vl"],
    qwen3vl_alerts=all_alerts["qwen3vl"],
    qwen3vl_thresholds=all_thresholds[
        "qwen3vl"
    ],
    audit_row_id=manifest[
        "audit_row_id"
    ].astype(str).to_numpy(),
)

wide_output.to_csv(
    SCORES_CSV,
    index=False,
    float_format="%.9g",
)

head_summary = pd.DataFrame(
    head_summary_rows
)

head_summary.to_csv(
    HEAD_SUMMARY_CSV,
    index=False,
    float_format="%.12g",
)

split_summary = pd.DataFrame(
    split_summary_rows
)

split_summary.to_csv(
    SPLIT_SUMMARY_CSV,
    index=False,
    float_format="%.12g",
)

backbone_summary_rows = []

for backbone_name in [
    "OpenCLIP",
    "Qwen3-VL-Embedding-2B",
]:
    subset = head_summary[
        head_summary["backbone"]
        .eq(backbone_name)
    ]

    target_rates = subset[
        "observed_alert_rate_target_containing"
    ].to_numpy(dtype=float)

    control_rates = subset[
        "observed_false_alert_rate_control"
    ].to_numpy(dtype=float)

    differences = subset[
        "rate_difference_target_minus_control"
    ].to_numpy(dtype=float)

    backbone_summary_rows.append({
        "backbone": backbone_name,
        "n_seeds": len(subset),
        "target_containing_alert_rate_mean": float(
            target_rates.mean()
        ),
        "target_containing_alert_rate_std_ddof0": float(
            target_rates.std(ddof=0)
        ),
        "control_false_alert_rate_mean": float(
            control_rates.mean()
        ),
        "control_false_alert_rate_std_ddof0": float(
            control_rates.std(ddof=0)
        ),
        "target_minus_control_difference_mean": float(
            differences.mean()
        ),
        "target_minus_control_difference_std_ddof0": float(
            differences.std(ddof=0)
        ),
    })

backbone_summary = pd.DataFrame(
    backbone_summary_rows
)

backbone_summary.to_csv(
    BACKBONE_SUMMARY_CSV,
    index=False,
    float_format="%.12g",
)

failures = []

if len(head_summary) != 10:
    failures.append(
        "head summary does not contain 10 rows"
    )

if all_scores["openclip"].shape != (
    5,
    EXPECTED_ROWS,
):
    failures.append(
        "OpenCLIP score array shape mismatch"
    )

if all_scores["qwen3vl"].shape != (
    5,
    EXPECTED_ROWS,
):
    failures.append(
        "Qwen score array shape mismatch"
    )

status = "PASS" if not failures else "FAIL"

elapsed_seconds = (
    time.perf_counter() - scoring_start
)

metadata = {
    "status": status,
    "audit": (
        "External real-image scoring using "
        "ten frozen controlled-main detector heads"
    ),
    "scientific_interpretation": (
        "Observed alerts on legitimate natural "
        "target-containing images and false alerts "
        "on target-category-absent controls. These "
        "are not attack positives and are not TPR, "
        "ASR, or physical-attack validation."
    ),
    "manifest_path": str(MANIFEST_PATH),
    "manifest_sha256": manifest_sha256,
    "artifact_manifest_path": str(
        ARTIFACT_MANIFEST
    ),
    "artifact_manifest_sha256": sha256_file(
        ARTIFACT_MANIFEST
    ),
    "scoring_equivalence_path": str(
        SCORING_EQUIVALENCE
    ),
    "scoring_equivalence_sha256": sha256_file(
        SCORING_EQUIVALENCE
    ),
    "n_images": EXPECTED_ROWS,
    "n_target_containing": (
        EXPECTED_TARGET_ROWS
    ),
    "n_target_category_absent_control": (
        EXPECTED_CONTROL_ROWS
    ),
    "n_backbones": 2,
    "n_seeds_per_backbone": 5,
    "n_frozen_heads": 10,
    "score_function": (
        "sigmoid(MLP logit)"
    ),
    "alert_comparator": (
        "score > seed-specific frozen "
        "threshold_val_1pct_fpr"
    ),
    "retraining_performed": False,
    "threshold_recalibration_performed": False,
    "model_selection_on_external_data": False,
    "scores_npz_path": str(SCORES_NPZ),
    "scores_npz_sha256": sha256_file(
        SCORES_NPZ
    ),
    "scores_csv_path": str(SCORES_CSV),
    "scores_csv_sha256": sha256_file(
        SCORES_CSV
    ),
    "head_summary_path": str(
        HEAD_SUMMARY_CSV
    ),
    "head_summary_sha256": sha256_file(
        HEAD_SUMMARY_CSV
    ),
    "backbone_summary_path": str(
        BACKBONE_SUMMARY_CSV
    ),
    "backbone_summary_sha256": sha256_file(
        BACKBONE_SUMMARY_CSV
    ),
    "split_summary_path": str(
        SPLIT_SUMMARY_CSV
    ),
    "split_summary_sha256": sha256_file(
        SPLIT_SUMMARY_CSV
    ),
    "elapsed_seconds": elapsed_seconds,
    "gpu": torch.cuda.get_device_name(0),
    "failures": failures,
}

atomic_json_write(
    METADATA_JSON,
    metadata,
)

report_lines = [
    "=" * 112,
    "TAM-VLM external frozen-detector scoring report",
    "=" * 112,
    f"Status                         : {status}",
    f"Images scored                  : {EXPECTED_ROWS}",
    f"Target-containing images       : {EXPECTED_TARGET_ROWS}",
    f"Target-category-absent controls: {EXPECTED_CONTROL_ROWS}",
    "Retraining performed           : False",
    "Threshold recalibration        : False",
    "Alert comparator               : score > frozen threshold",
]

for _, row in backbone_summary.iterrows():
    report_lines.extend([
        "",
        str(row["backbone"]),
        (
            "  Target-containing observed alert rate: "
            f"{row['target_containing_alert_rate_mean']:.4%} "
            f"± "
            f"{row['target_containing_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Control observed false-alert rate     : "
            f"{row['control_false_alert_rate_mean']:.4%} "
            f"± "
            f"{row['control_false_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Target-minus-control difference       : "
            f"{row['target_minus_control_difference_mean']:+.4%} "
            f"± "
            f"{row['target_minus_control_difference_std_ddof0']:.4%}"
        ),
    ])

report_lines.extend([
    "",
    (
        "Interpretation: legitimate real-image "
        "object-response and benign false-alert "
        "behavior only; not attack TPR or ASR."
    ),
    f"Elapsed seconds                : {elapsed_seconds:.2f}",
    f"Saved metadata                 : {METADATA_JSON}",
])

REPORT_TXT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report_lines))

if status != "PASS":
    raise SystemExit(
        "External frozen-detector scoring failed."
    )
