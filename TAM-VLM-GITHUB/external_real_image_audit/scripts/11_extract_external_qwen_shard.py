#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap


AUDIT_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
MANIFEST_PATH = (
    AUDIT_ROOT / "metadata/"
    "nuimages_external_full_pool_manifest_v1.csv"
)

OUTPUT_ROOT = AUDIT_ROOT / "embeddings/qwen_v1"

PROJECT_ROOT = Path(
    os.environ["TAMVLM_QWEN_PROJECT_ROOT"]
)
OFFICIAL_REPO = PROJECT_ROOT / "official_qwen_repo"

MODEL_PATH = Path(
    os.environ["TAMVLM_QWEN_MODEL"]
)
MODEL_SAFETENSORS = MODEL_PATH / "model.safetensors"

EXPECTED_MANIFEST_SHA256 = (
    "ebcd9203c9fa41b01635f7182dd00fe0d"
    "419d04a8a1f03ef764677222def25ab"
)

EXPECTED_MODEL_SAFETENSORS_SHA256 = (
    "c73fa9caeddeb3ff831d46c085a7a5708"
    "343248ca777e90f2d486964464509c1"
)

EXPECTED_REPO_COMMIT = (
    "393e2978d27852b0d0230d6994f37f9c15bed73c"
)

EXPECTED_ROWS = 16436
EXPECTED_DIMENSION = 2048

MIN_PIXELS = 4096
MAX_PIXELS = 401408


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
    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(temporary, path)


def repo_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNAVAILABLE"


parser = argparse.ArgumentParser()

parser.add_argument(
    "--shard-id",
    type=int,
    required=True,
)

parser.add_argument(
    "--num-shards",
    type=int,
    default=2,
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
)

args = parser.parse_args()

if args.num_shards < 1:
    raise ValueError("num-shards must be positive.")

if not 0 <= args.shard_id < args.num_shards:
    raise ValueError("Invalid shard-id.")

if args.batch_size < 1:
    raise ValueError("batch-size must be positive.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

for required_path in [
    MANIFEST_PATH,
    OFFICIAL_REPO,
    MODEL_PATH,
    MODEL_SAFETENSORS,
]:
    if not required_path.exists():
        raise FileNotFoundError(required_path)

sys.path.insert(0, str(OFFICIAL_REPO))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder


shard_name = (
    f"shard_{args.shard_id}_of_{args.num_shards}"
)

SHARD_DIR = OUTPUT_ROOT / shard_name
SHARD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = (
    SHARD_DIR /
    f"nuimages_external_qwen_{shard_name}.npy"
)

INDICES_PATH = (
    SHARD_DIR /
    f"nuimages_external_qwen_{shard_name}_manifest_indices.npy"
)

INDEX_CSV_PATH = (
    SHARD_DIR /
    f"nuimages_external_qwen_{shard_name}_index.csv"
)

PROGRESS_PATH = (
    SHARD_DIR /
    f"nuimages_external_qwen_{shard_name}_progress.json"
)

METADATA_PATH = (
    SHARD_DIR /
    f"nuimages_external_qwen_{shard_name}_metadata.json"
)


print("=" * 112)
print(
    f"TAM-VLM external Qwen extraction — "
    f"{shard_name}"
)
print("=" * 112)

manifest_sha256 = sha256_file(MANIFEST_PATH)

print("Manifest SHA256:", manifest_sha256)

if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen external manifest SHA256 mismatch."
    )

print("Checking Qwen model checksum...")

model_sha256 = sha256_file(MODEL_SAFETENSORS)

print("Model SHA256   :", model_sha256)

if model_sha256 != EXPECTED_MODEL_SAFETENSORS_SHA256:
    raise RuntimeError(
        "Qwen model.safetensors SHA256 mismatch."
    )

resolved_repo_commit = repo_commit(OFFICIAL_REPO)

print("Repo commit    :", resolved_repo_commit)

if (
    resolved_repo_commit != "UNAVAILABLE"
    and resolved_repo_commit != EXPECTED_REPO_COMMIT
):
    raise RuntimeError(
        "Official Qwen repository commit mismatch."
    )

manifest = pd.read_csv(MANIFEST_PATH)

if len(manifest) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"found {len(manifest)}."
    )

required_columns = {
    "audit_row_id",
    "audit_group",
    "sample_data_token",
    "log_token",
    "dataset_split",
    "image_abspath",
    "target_category_set",
}

missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise RuntimeError(
        f"Missing manifest columns: "
        f"{sorted(missing_columns)}"
    )

if not manifest["audit_row_id"].is_unique:
    raise RuntimeError(
        "audit_row_id is not unique."
    )

global_indices = np.arange(
    EXPECTED_ROWS,
    dtype=np.int64,
)

shard_indices = global_indices[
    global_indices % args.num_shards == args.shard_id
]

expected_shard_rows = len(shard_indices)

print("Total manifest rows:", EXPECTED_ROWS)
print("Shard rows         :", expected_shard_rows)
print(
    "Global index range :",
    int(shard_indices[0]),
    "to",
    int(shard_indices[-1]),
)

if INDICES_PATH.is_file():
    existing_indices = np.load(INDICES_PATH)

    if not np.array_equal(
        existing_indices,
        shard_indices,
    ):
        raise RuntimeError(
            "Existing shard indices do not match."
        )
else:
    np.save(
        INDICES_PATH,
        shard_indices,
    )

shard_manifest = (
    manifest.iloc[shard_indices]
    .copy()
    .reset_index(drop=True)
)

shard_manifest.insert(
    0,
    "shard_embedding_position",
    np.arange(
        expected_shard_rows,
        dtype=np.int64,
    ),
)

shard_manifest.insert(
    1,
    "global_embedding_index",
    shard_indices,
)

index_columns = [
    "shard_embedding_position",
    "global_embedding_index",
    "audit_row_id",
    "audit_group",
    "dataset_split",
    "sample_data_token",
    "sample_token",
    "log_token",
    "timestamp",
    "image_abspath",
    "target_category_set",
    "target_category_count",
    "exclusive_category",
]

index_frame = shard_manifest[index_columns]

if INDEX_CSV_PATH.is_file():
    existing_index = pd.read_csv(INDEX_CSV_PATH)

    if not existing_index.equals(index_frame):
        raise RuntimeError(
            "Existing shard index CSV does not match."
        )
else:
    index_frame.to_csv(
        INDEX_CSV_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

image_paths = (
    shard_manifest["image_abspath"]
    .astype(str)
    .tolist()
)

missing_images = [
    path
    for path in image_paths
    if not Path(path).is_file()
]

if missing_images:
    raise FileNotFoundError(
        f"{len(missing_images)} images missing. "
        f"Examples: {missing_images[:10]}"
    )

if EMBEDDINGS_PATH.is_file():
    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r+",
    )

    if embeddings.shape != (
        expected_shard_rows,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Existing embedding shape: "
            f"{embeddings.shape}"
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            f"Existing dtype: {embeddings.dtype}"
        )
else:
    embeddings = open_memmap(
        EMBEDDINGS_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            expected_shard_rows,
            EXPECTED_DIMENSION,
        ),
    )

    embeddings[:] = np.nan
    embeddings.flush()

if PROGRESS_PATH.is_file():
    progress = json.loads(
        PROGRESS_PATH.read_text(
            encoding="utf-8"
        )
    )

    position = int(
        progress.get("next_position", 0)
    )

    active_batch_size = int(
        progress.get(
            "active_batch_size",
            args.batch_size,
        )
    )
else:
    position = 0
    active_batch_size = args.batch_size

if not 0 <= position <= expected_shard_rows:
    raise RuntimeError(
        f"Invalid resume position: {position}"
    )

print("GPU               :", torch.cuda.get_device_name(0))
print("Precision         : float16")
print("Attention         : sdpa")
print("Normalization     : True")
print("Min pixels        :", MIN_PIXELS)
print("Max pixels        :", MAX_PIXELS)
print("Requested batch   :", args.batch_size)
print("Active batch      :", active_batch_size)
print("Resume position   :", position)

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

print(
    "Model loaded in   :",
    f"{model_load_seconds:.2f}s",
)

torch.cuda.reset_peak_memory_stats()

run_start_position = position
extraction_start = time.perf_counter()

while position < expected_shard_rows:
    stop = min(
        position + active_batch_size,
        expected_shard_rows,
    )

    batch_paths = image_paths[position:stop]

    inputs = [
        {"image": path}
        for path in batch_paths
    ]

    batch_start = time.perf_counter()

    try:
        output = model.process(
            inputs,
            normalize=True,
        )

        array = (
            output
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

        if active_batch_size == 1:
            raise

        active_batch_size = max(
            1,
            active_batch_size // 2,
        )

        print(
            "CUDA OOM: reducing active batch size to",
            active_batch_size,
            flush=True,
        )

        continue

    expected_batch_shape = (
        stop - position,
        EXPECTED_DIMENSION,
    )

    if array.shape != expected_batch_shape:
        raise RuntimeError(
            f"Unexpected batch shape {array.shape}; "
            f"expected {expected_batch_shape}."
        )

    if not np.isfinite(array).all():
        raise RuntimeError(
            f"Non-finite embeddings at positions "
            f"{position}:{stop}."
        )

    norms = np.linalg.norm(
        array.astype(np.float64),
        axis=1,
    )

    if np.any(
        np.abs(norms - 1.0) > 0.002
    ):
        raise RuntimeError(
            f"Unexpected embedding norms: "
            f"{norms.min()}–{norms.max()}"
        )

    embeddings[position:stop] = array
    embeddings.flush()

    position = stop

    elapsed = (
        time.perf_counter() -
        extraction_start
    )

    processed_this_run = (
        position - run_start_position
    )

    seconds_per_image = (
        elapsed /
        max(processed_this_run, 1)
    )

    remaining = (
        expected_shard_rows - position
    )

    eta_seconds = (
        seconds_per_image * remaining
    )

    progress = {
        "status": (
            "COMPLETE"
            if position == expected_shard_rows
            else "RUNNING"
        ),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "next_position": position,
        "processed_shard_images": position,
        "total_shard_images": expected_shard_rows,
        "manifest_sha256": manifest_sha256,
        "model_safetensors_sha256": model_sha256,
        "requested_batch_size": args.batch_size,
        "active_batch_size": active_batch_size,
        "last_batch_seconds": (
            time.perf_counter() -
            batch_start
        ),
        "seconds_per_image_this_run": (
            seconds_per_image
        ),
        "eta_seconds": eta_seconds,
        "gpu": torch.cuda.get_device_name(0),
    }

    atomic_json_write(
        PROGRESS_PATH,
        progress,
    )

    print(
        f"shard {args.shard_id} | "
        f"{position:5d}/{expected_shard_rows} | "
        f"batch={active_batch_size} | "
        f"{seconds_per_image:.4f} sec/image | "
        f"ETA={eta_seconds / 60:.1f} min",
        flush=True,
    )

embeddings.flush()

final = np.load(
    EMBEDDINGS_PATH,
    mmap_mode="r",
)

norm_values = np.empty(
    expected_shard_rows,
    dtype=np.float64,
)

finite = True

for start in range(
    0,
    expected_shard_rows,
    2048,
):
    stop = min(
        start + 2048,
        expected_shard_rows,
    )

    chunk = np.asarray(
        final[start:stop],
        dtype=np.float64,
    )

    if not np.isfinite(chunk).all():
        finite = False

    norm_values[start:stop] = np.linalg.norm(
        chunk,
        axis=1,
    )

failures = []

if final.shape != (
    expected_shard_rows,
    EXPECTED_DIMENSION,
):
    failures.append(
        f"unexpected shape {final.shape}"
    )

if final.dtype != np.float32:
    failures.append(
        f"unexpected dtype {final.dtype}"
    )

if not finite:
    failures.append(
        "non-finite values detected"
    )

if np.any(
    np.abs(norm_values - 1.0) > 0.002
):
    failures.append(
        "embedding norm tolerance exceeded"
    )

status = "PASS" if not failures else "FAIL"

metadata = {
    "status": status,
    "audit": (
        "TAM-VLM external Qwen "
        "embedding extraction shard"
    ),
    "shard_id": args.shard_id,
    "num_shards": args.num_shards,
    "n_shard_rows": expected_shard_rows,
    "global_index_first": int(
        shard_indices[0]
    ),
    "global_index_last": int(
        shard_indices[-1]
    ),
    "manifest_path": str(MANIFEST_PATH),
    "manifest_sha256": manifest_sha256,
    "manifest_indices_path": str(
        INDICES_PATH
    ),
    "manifest_indices_sha256": (
        sha256_file(INDICES_PATH)
    ),
    "index_csv_path": str(
        INDEX_CSV_PATH
    ),
    "index_csv_sha256": (
        sha256_file(INDEX_CSV_PATH)
    ),
    "embeddings_path": str(
        EMBEDDINGS_PATH
    ),
    "embeddings_sha256": (
        sha256_file(EMBEDDINGS_PATH)
    ),
    "embedding_shape": list(
        final.shape
    ),
    "embedding_dtype": str(
        final.dtype
    ),
    "model_path": str(MODEL_PATH),
    "model_safetensors_sha256": (
        model_sha256
    ),
    "official_repo_commit": (
        resolved_repo_commit
    ),
    "precision": (
        "float16 inference; float32 storage"
    ),
    "attention": "sdpa",
    "normalization": True,
    "pooling": (
        "last valid token by attention mask"
    ),
    "default_instruction": (
        "Represent the user's input."
    ),
    "custom_instruction_used": False,
    "min_pixels": MIN_PIXELS,
    "max_pixels": MAX_PIXELS,
    "requested_batch_size": (
        args.batch_size
    ),
    "final_active_batch_size": (
        active_batch_size
    ),
    "norm_min": float(
        norm_values.min()
    ),
    "norm_median": float(
        np.median(norm_values)
    ),
    "norm_max": float(
        norm_values.max()
    ),
    "model_load_seconds": (
        model_load_seconds
    ),
    "extraction_seconds_this_run": (
        time.perf_counter() -
        extraction_start
    ),
    "run_start_position": (
        run_start_position
    ),
    "peak_gpu_memory_gb": (
        torch.cuda.max_memory_allocated() /
        (1024 ** 3)
    ),
    "gpu": torch.cuda.get_device_name(0),
    "script_path": str(
        Path(__file__).resolve()
    ),
    "script_sha256": sha256_file(
        Path(__file__).resolve()
    ),
    "failures": failures,
}

atomic_json_write(
    METADATA_PATH,
    metadata,
)

print("\n" + "=" * 112)
print(
    f"Qwen external extraction summary — "
    f"{shard_name}"
)
print("=" * 112)
print("Status             :", status)
print("Embedding shape    :", final.shape)
print("Embedding dtype    :", final.dtype)
print(
    "Embedding norms    :",
    f"{norm_values.min():.8f}–"
    f"{norm_values.max():.8f}",
)
print(
    "Norm median        :",
    f"{np.median(norm_values):.8f}",
)
print(
    "Peak GPU memory GB :",
    f"{metadata['peak_gpu_memory_gb']:.3f}",
)
print(
    "Embeddings SHA256  :",
    metadata["embeddings_sha256"],
)
print(
    "Saved metadata     :",
    METADATA_PATH,
)

if status != "PASS":
    raise SystemExit(
        "Qwen external shard extraction failed."
    )
