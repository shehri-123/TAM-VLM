#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
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

OUTPUT_DIR = AUDIT_ROOT / "embeddings/openclip_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_PATH = (
    OUTPUT_DIR /
    "nuimages_external_openclip_embeddings_v1.npy"
)

INDEX_PATH = (
    OUTPUT_DIR /
    "nuimages_external_openclip_embedding_index_v1.csv"
)

PROGRESS_PATH = (
    OUTPUT_DIR /
    "nuimages_external_openclip_progress_v1.json"
)

METADATA_PATH = (
    OUTPUT_DIR /
    "nuimages_external_openclip_metadata_v1.json"
)

MODEL_NAME = "ViT-B-16"

WEIGHTS_PATH = Path(
    os.environ["TAMVLM_CLIP_WEIGHTS"]
)
EXPECTED_MANIFEST_SHA256 = (
    "ebcd9203c9fa41b01635f7182dd00fe0d"
    "419d04a8a1f03ef764677222def25ab"
)

EXPECTED_WEIGHTS_SHA256 = (
    "3f25d29d3cc74e1d25d47e0593b4dd08"
    "64ced1e9b4d8e486a247ca4502f227f1"
)

EXPECTED_ROWS = 16436
EXPECTED_DIMENSION = 512


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(path, payload):
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(temporary_path, path)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT FOUND"


def load_progress():
    if not PROGRESS_PATH.is_file():
        return {
            "status": "RUNNING",
            "next_index": 0,
            "processed_images": 0,
        }

    return json.loads(
        PROGRESS_PATH.read_text(encoding="utf-8")
    )


def calculate_norm_statistics(array, chunk_size=4096):
    norms = np.empty(
        array.shape[0],
        dtype=np.float64,
    )

    finite = True

    for start in range(0, array.shape[0], chunk_size):
        stop = min(start + chunk_size, array.shape[0])

        chunk = np.asarray(
            array[start:stop],
            dtype=np.float64,
        )

        if not np.isfinite(chunk).all():
            finite = False

        norms[start:stop] = np.linalg.norm(
            chunk,
            axis=1,
        )

    return {
        "finite": finite,
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
)
args = parser.parse_args()

if args.batch_size < 1:
    raise ValueError("Batch size must be positive.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable.")

for required_path in [
    MANIFEST_PATH,
    WEIGHTS_PATH,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

print("=" * 112)
print("TAM-VLM external OpenCLIP embedding extraction")
print("=" * 112)

manifest_sha256 = sha256_file(MANIFEST_PATH)
weights_sha256 = sha256_file(WEIGHTS_PATH)

print("Manifest SHA256:", manifest_sha256)
print("Weights SHA256 :", weights_sha256)

if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Manifest SHA256 does not match frozen protocol."
    )

if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
    raise RuntimeError(
        "OpenCLIP weights SHA256 mismatch."
    )

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "audit_row_id",
    "audit_group",
    "contains_any_audited_target",
    "is_target_category_absent_control",
    "dataset_split",
    "sample_data_token",
    "log_token",
    "image_abspath",
    "included_in_primary_full_pool",
}

missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise RuntimeError(
        f"Missing manifest columns: "
        f"{sorted(missing_columns)}"
    )

if len(manifest) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"found {len(manifest)}."
    )

if not manifest["audit_row_id"].is_unique:
    raise RuntimeError("audit_row_id values are not unique.")

if not manifest["sample_data_token"].is_unique:
    raise RuntimeError(
        "sample_data_token values are not unique."
    )

if not manifest["image_abspath"].is_unique:
    raise RuntimeError(
        "image_abspath values are not unique."
    )

if not (
    manifest["included_in_primary_full_pool"]
    .astype(int)
    .eq(1)
    .all()
):
    raise RuntimeError(
        "Manifest contains rows outside primary full pool."
    )

image_paths = (
    manifest["image_abspath"]
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
        f"{len(missing_images)} images are missing. "
        f"Examples: {missing_images[:10]}"
    )

index_frame = manifest[
    [
        "audit_row_id",
        "audit_group",
        "contains_any_audited_target",
        "is_target_category_absent_control",
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
].copy()

index_frame.insert(
    0,
    "embedding_index",
    np.arange(len(index_frame), dtype=np.int64),
)

if INDEX_PATH.is_file():
    existing_index = pd.read_csv(INDEX_PATH)

    if not existing_index.equals(index_frame):
        raise RuntimeError(
            "Existing embedding index does not match "
            "the frozen manifest."
        )
else:
    index_frame.to_csv(
        INDEX_PATH,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

if EMBEDDINGS_PATH.is_file():
    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r+",
    )

    if embeddings.shape != (
        EXPECTED_ROWS,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Existing embedding shape is "
            f"{embeddings.shape}."
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            f"Existing embedding dtype is "
            f"{embeddings.dtype}."
        )
else:
    embeddings = open_memmap(
        EMBEDDINGS_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(
            EXPECTED_ROWS,
            EXPECTED_DIMENSION,
        ),
    )

    embeddings[:] = np.nan
    embeddings.flush()

progress = load_progress()
next_index = int(progress.get("next_index", 0))

if not 0 <= next_index <= EXPECTED_ROWS:
    raise RuntimeError(
        f"Invalid next_index in progress file: "
        f"{next_index}"
    )

device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("GPU              :", torch.cuda.get_device_name(0))
print("Model            :", MODEL_NAME)
print("Precision        : float32")
print("Autocast         : False")
print("Output normalize : False")
print("Batch size       :", args.batch_size)
print("Resume index     :", next_index)

load_start = time.perf_counter()

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=str(WEIGHTS_PATH),
    )
)

model = model.eval().to(device)

model_load_seconds = (
    time.perf_counter() - load_start
)

print("Model loaded in  :", f"{model_load_seconds:.2f}s")
print("Preprocessing:")
print(preprocess)

torch.cuda.reset_peak_memory_stats()

extraction_start = time.perf_counter()
run_start_index = next_index

for start in range(
    next_index,
    EXPECTED_ROWS,
    args.batch_size,
):
    stop = min(
        start + args.batch_size,
        EXPECTED_ROWS,
    )

    tensors = []

    for image_path in image_paths[start:stop]:
        with Image.open(image_path) as image:
            tensors.append(
                preprocess(
                    image.convert("RGB")
                )
            )

    batch = torch.stack(tensors).to(
        device,
        non_blocking=False,
    )

    with torch.no_grad():
        encoded = (
            model.encode_image(batch)
            .float()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    if encoded.shape != (
        stop - start,
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Unexpected batch shape: {encoded.shape}"
        )

    if not np.isfinite(encoded).all():
        raise RuntimeError(
            f"Non-finite embeddings in rows "
            f"{start}:{stop}."
        )

    embeddings[start:stop] = encoded
    embeddings.flush()

    elapsed = time.perf_counter() - extraction_start
    processed_this_run = stop - run_start_index
    seconds_per_image = (
        elapsed / max(processed_this_run, 1)
    )

    remaining = EXPECTED_ROWS - stop
    eta_seconds = remaining * seconds_per_image

    progress = {
        "status": (
            "COMPLETE"
            if stop == EXPECTED_ROWS
            else "RUNNING"
        ),
        "manifest_sha256": manifest_sha256,
        "weights_sha256": weights_sha256,
        "next_index": stop,
        "processed_images": stop,
        "total_images": EXPECTED_ROWS,
        "requested_batch_size": args.batch_size,
        "active_batch_size": args.batch_size,
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
        f"embedded {stop:5d}/{EXPECTED_ROWS} | "
        f"{seconds_per_image:.4f} sec/image | "
        f"ETA {eta_seconds / 60:.1f} min",
        flush=True,
    )

final = np.load(
    EMBEDDINGS_PATH,
    mmap_mode="r",
)

statistics = calculate_norm_statistics(final)

failures = []

if final.shape != (
    EXPECTED_ROWS,
    EXPECTED_DIMENSION,
):
    failures.append(
        f"unexpected final shape {final.shape}"
    )

if final.dtype != np.float32:
    failures.append(
        f"unexpected final dtype {final.dtype}"
    )

if not statistics["finite"]:
    failures.append(
        "non-finite values detected"
    )

if statistics["norm_min"] <= 0:
    failures.append(
        "zero or negative embedding norm detected"
    )

status = "PASS" if not failures else "FAIL"

metadata = {
    "status": status,
    "audit": (
        "TAM-VLM external real-image "
        "OpenCLIP embedding extraction"
    ),
    "manifest_path": str(MANIFEST_PATH),
    "manifest_sha256": manifest_sha256,
    "manifest_rows": len(manifest),
    "embedding_index_path": str(INDEX_PATH),
    "embedding_index_sha256": sha256_file(INDEX_PATH),
    "embeddings_path": str(EMBEDDINGS_PATH),
    "embeddings_sha256": sha256_file(EMBEDDINGS_PATH),
    "embedding_shape": list(final.shape),
    "embedding_dtype": str(final.dtype),
    "model_name": MODEL_NAME,
    "weights_path": str(WEIGHTS_PATH),
    "weights_sha256": weights_sha256,
    "precision": "float32",
    "autocast_used": False,
    "output_l2_normalization": False,
    "preprocess": repr(preprocess),
    "norm_min": statistics["norm_min"],
    "norm_median": statistics["norm_median"],
    "norm_max": statistics["norm_max"],
    "batch_size": args.batch_size,
    "model_load_seconds": model_load_seconds,
    "extraction_seconds_this_run": (
        time.perf_counter() - extraction_start
    ),
    "run_start_index": run_start_index,
    "peak_gpu_memory_gb": (
        torch.cuda.max_memory_allocated() /
        (1024 ** 3)
    ),
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "open_clip_torch_version": package_version(
        "open-clip-torch"
    ),
    "torchvision_version": package_version(
        "torchvision"
    ),
    "pillow_version": package_version(
        "Pillow"
    ),
    "script_path": str(Path(__file__).resolve()),
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
print("OpenCLIP external extraction summary")
print("=" * 112)
print("Status             :", status)
print("Embedding shape    :", final.shape)
print("Embedding dtype    :", final.dtype)
print(
    "Embedding norms    :",
    f"{statistics['norm_min']:.8f}–"
    f"{statistics['norm_max']:.8f}",
)
print(
    "Norm median        :",
    f"{statistics['norm_median']:.8f}",
)
print(
    "Peak GPU memory GB :",
    f"{metadata['peak_gpu_memory_gb']:.3f}",
)
print("Embeddings SHA256  :", metadata["embeddings_sha256"])
print("Saved metadata     :", METADATA_PATH)

if status != "PASS":
    raise SystemExit(
        "OpenCLIP external extraction failed."
    )
