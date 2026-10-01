#!/usr/bin/env python3

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


AUDIT_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
QWEN_ROOT = Path(
    os.environ["TAMVLM_QWEN_ROOT"]
)
OFFICIAL_REPO = (
    QWEN_ROOT / "official_qwen_repo"
)

MODEL_PATH = Path(
    os.environ["TAMVLM_QWEN_MODEL"]
)
FINAL_EMBEDDINGS = Path(
    os.environ["TAMVLM_QWEN_EMBEDDINGS"]
)
SELECTED_ROWS = (
    QWEN_ROOT / "results/smoke_test_v1/"
    "smoke_selected_rows.csv"
)

OUTPUT_JSON = (
    AUDIT_ROOT / "protocol/"
    "QWEN_FRESH_EXTRACTION_EQUIVALENCE_v1.json"
)

OUTPUT_TXT = (
    AUDIT_ROOT / "protocol/"
    "QWEN_FRESH_EXTRACTION_EQUIVALENCE_v1.txt"
)

MIN_PIXELS = 4096
MAX_PIXELS = 401408
BATCH_SIZE = 8
EXPECTED_DIMENSION = 2048
MINIMUM_ACCEPTABLE_COSINE = 0.9999
MAXIMUM_NORM_DEVIATION = 0.002


for path in [
    OFFICIAL_REPO,
    MODEL_PATH,
    FINAL_EMBEDDINGS,
    SELECTED_ROWS,
]:
    if not path.exists():
        raise FileNotFoundError(path)

sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


def cosine_rows(left, right):
    left64 = left.astype(np.float64)
    right64 = right.astype(np.float64)

    numerator = np.sum(left64 * right64, axis=1)

    denominator = (
        np.linalg.norm(left64, axis=1) *
        np.linalg.norm(right64, axis=1)
    )

    return numerator / np.maximum(
        denominator,
        1e-12,
    )


print("=" * 112)
print("Qwen fresh extraction equivalence audit")
print("=" * 112)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL_PATH)
print("Precision: float16")
print("Attention: sdpa")
print("Normalization: True")
print("Batch size:", BATCH_SIZE)

selected = pd.read_csv(SELECTED_ROWS)

required_columns = {
    "image",
    "manifest_index",
    "trigger",
}

missing_columns = required_columns - set(selected.columns)

if missing_columns:
    raise RuntimeError(
        f"Missing columns: {sorted(missing_columns)}"
    )

if len(selected) != 13:
    raise RuntimeError(
        f"Expected 13 selected rows, found {len(selected)}"
    )

image_paths = selected["image"].astype(str).tolist()
manifest_indices = (
    selected["manifest_index"].astype(int).tolist()
)

missing_images = [
    path for path in image_paths
    if not Path(path).is_file()
]

if missing_images:
    raise FileNotFoundError(
        f"Missing images: {missing_images}"
    )

stored_all = np.load(
    FINAL_EMBEDDINGS,
    mmap_mode="r",
)

if stored_all.shape != (48633, EXPECTED_DIMENSION):
    raise RuntimeError(
        f"Unexpected stored shape: {stored_all.shape}"
    )

stored_selected = np.asarray(
    stored_all[manifest_indices],
    dtype=np.float32,
)

load_start = time.perf_counter()

model = Qwen3VLEmbedder(
    model_name_or_path=str(MODEL_PATH),
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    dtype=torch.float16,
    attn_implementation="sdpa",
)

model_load_seconds = (
    time.perf_counter() - load_start
)

fresh_batches = []
inference_start = time.perf_counter()

for start in range(
    0,
    len(image_paths),
    BATCH_SIZE,
):
    stop = min(
        start + BATCH_SIZE,
        len(image_paths),
    )

    batch_paths = image_paths[start:stop]

    inputs = [
        {"image": path}
        for path in batch_paths
    ]

    embeddings = model.process(
        inputs,
        normalize=True,
    )

    embeddings = (
        embeddings
        .float()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    fresh_batches.append(embeddings)

    print(
        f"Encoded {stop}/{len(image_paths)}"
    )

fresh = np.concatenate(
    fresh_batches,
    axis=0,
)

inference_seconds = (
    time.perf_counter() - inference_start
)

if fresh.shape != stored_selected.shape:
    raise RuntimeError(
        f"Fresh shape {fresh.shape} does not match "
        f"stored shape {stored_selected.shape}"
    )

finite = bool(np.isfinite(fresh).all())

fresh_norms = np.linalg.norm(
    fresh.astype(np.float64),
    axis=1,
)

stored_norms = np.linalg.norm(
    stored_selected.astype(np.float64),
    axis=1,
)

cosines = cosine_rows(
    fresh,
    stored_selected,
)

absolute_error = np.abs(
    fresh - stored_selected
)

records = []

print("\nPer-image comparison:")

for position, row in selected.iterrows():
    record = {
        "manifest_index": int(
            row["manifest_index"]
        ),
        "trigger": str(row["trigger"]),
        "image": str(row["image"]),
        "fresh_norm": float(
            fresh_norms[position]
        ),
        "stored_norm": float(
            stored_norms[position]
        ),
        "true_cosine_similarity": float(
            cosines[position]
        ),
        "mean_absolute_error": float(
            absolute_error[position].mean()
        ),
        "max_absolute_error": float(
            absolute_error[position].max()
        ),
    }

    records.append(record)

    print(
        f"index={record['manifest_index']:5d} | "
        f"trigger={record['trigger']:<18s} | "
        f"cos={record['true_cosine_similarity']:.10f} | "
        f"MAE={record['mean_absolute_error']:.8e} | "
        f"norm={record['fresh_norm']:.8f}"
    )

minimum_cosine = float(cosines.min())
median_cosine = float(np.median(cosines))
maximum_cosine = float(cosines.max())

maximum_norm_deviation = float(
    np.max(np.abs(fresh_norms - 1.0))
)

mean_absolute_error = float(
    absolute_error.mean()
)

maximum_absolute_error = float(
    absolute_error.max()
)

failures = []

if not finite:
    failures.append(
        "fresh embeddings contain non-finite values"
    )

if minimum_cosine < MINIMUM_ACCEPTABLE_COSINE:
    failures.append(
        f"minimum cosine {minimum_cosine} is below "
        f"{MINIMUM_ACCEPTABLE_COSINE}"
    )

if maximum_norm_deviation > MAXIMUM_NORM_DEVIATION:
    failures.append(
        f"maximum norm deviation "
        f"{maximum_norm_deviation} exceeds "
        f"{MAXIMUM_NORM_DEVIATION}"
    )

status = "PASS" if not failures else "FAIL"

payload = {
    "status": status,
    "audit": (
        "fresh_qwen_model_rerun_against_"
        "final_stored_embeddings"
    ),
    "model_rerun_performed": True,
    "model_path": str(MODEL_PATH),
    "official_repo_commit": (
        "393e2978d27852b0d0230d6994f37f9c15bed73c"
    ),
    "selected_rows": len(selected),
    "fresh_embedding_shape": list(fresh.shape),
    "stored_embedding_shape": list(
        stored_selected.shape
    ),
    "finite": finite,
    "precision": "float16 inference; float32 comparison",
    "attention": "sdpa",
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
    "batch_size": BATCH_SIZE,
    "normalization": True,
    "pooling": "last valid token by attention mask",
    "default_instruction": (
        "Represent the user's input."
    ),
    "custom_instruction_used": False,
    "minimum_true_cosine_similarity": minimum_cosine,
    "median_true_cosine_similarity": median_cosine,
    "maximum_true_cosine_similarity": maximum_cosine,
    "mean_absolute_error": mean_absolute_error,
    "maximum_absolute_error": maximum_absolute_error,
    "fresh_norm_min": float(fresh_norms.min()),
    "fresh_norm_median": float(
        np.median(fresh_norms)
    ),
    "fresh_norm_max": float(fresh_norms.max()),
    "maximum_norm_deviation": (
        maximum_norm_deviation
    ),
    "acceptance_threshold_cosine": (
        MINIMUM_ACCEPTABLE_COSINE
    ),
    "model_load_seconds": model_load_seconds,
    "inference_seconds": inference_seconds,
    "failures": failures,
    "records": records,
}

OUTPUT_JSON.write_text(
    json.dumps(
        payload,
        indent=2,
    ) + "\n",
    encoding="utf-8",
)

report_lines = [
    "=" * 112,
    "Qwen fresh extraction equivalence audit",
    "=" * 112,
    f"Status                        : {status}",
    "Model rerun performed         : True",
    f"Selected images               : {len(selected)}",
    f"Embedding shape               : {fresh.shape}",
    f"Minimum true cosine           : {minimum_cosine:.10f}",
    f"Median true cosine            : {median_cosine:.10f}",
    f"Maximum true cosine           : {maximum_cosine:.10f}",
    f"Mean absolute error           : {mean_absolute_error:.8e}",
    f"Maximum absolute error        : {maximum_absolute_error:.8e}",
    f"Fresh norm range              : {fresh_norms.min():.8f}–{fresh_norms.max():.8f}",
    f"Maximum norm deviation        : {maximum_norm_deviation:.8e}",
    f"Model load seconds            : {model_load_seconds:.2f}",
    f"Inference seconds             : {inference_seconds:.2f}",
    f"Saved JSON                    : {OUTPUT_JSON}",
]

if failures:
    report_lines.append("Failures:")

    for failure in failures:
        report_lines.append(
            f"  - {failure}"
        )

OUTPUT_TXT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report_lines))

if status != "PASS":
    raise SystemExit(
        "Qwen fresh equivalence audit failed."
    )
