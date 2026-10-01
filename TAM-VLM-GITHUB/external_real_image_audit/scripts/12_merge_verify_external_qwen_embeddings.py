#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
MANIFEST = (
    ROOT / "metadata/"
    "nuimages_external_full_pool_manifest_v1.csv"
)

QWEN_ROOT = ROOT / "embeddings/qwen_v1"

FINAL_DIR = QWEN_ROOT / "final_v1"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_EMBEDDINGS = (
    FINAL_DIR /
    "nuimages_external_qwen_embeddings_v1.npy"
)

FINAL_INDEX = (
    FINAL_DIR /
    "nuimages_external_qwen_embedding_index_v1.csv"
)

FINAL_METADATA = (
    FINAL_DIR /
    "nuimages_external_qwen_metadata_v1.json"
)

EXPECTED_MANIFEST_SHA256 = (
    "ebcd9203c9fa41b01635f7182dd00fe0d"
    "419d04a8a1f03ef764677222def25ab"
)

EXPECTED_ROWS = 16436
EXPECTED_DIMENSION = 2048

SHARDS = [
    {
        "id": 0,
        "dir": QWEN_ROOT / "shard_0_of_2",
        "embeddings": (
            "nuimages_external_qwen_"
            "shard_0_of_2.npy"
        ),
        "indices": (
            "nuimages_external_qwen_"
            "shard_0_of_2_manifest_indices.npy"
        ),
        "metadata": (
            "nuimages_external_qwen_"
            "shard_0_of_2_metadata.json"
        ),
    },
    {
        "id": 1,
        "dir": QWEN_ROOT / "shard_1_of_2",
        "embeddings": (
            "nuimages_external_qwen_"
            "shard_1_of_2.npy"
        ),
        "indices": (
            "nuimages_external_qwen_"
            "shard_1_of_2_manifest_indices.npy"
        ),
        "metadata": (
            "nuimages_external_qwen_"
            "shard_1_of_2_metadata.json"
        ),
    },
]


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


print("=" * 112)
print("TAM-VLM external Qwen shard merge and verification")
print("=" * 112)

if not MANIFEST.is_file():
    raise FileNotFoundError(MANIFEST)

manifest_sha256 = sha256_file(MANIFEST)

print("Manifest SHA256:", manifest_sha256)

if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
    raise RuntimeError(
        "Frozen manifest SHA256 mismatch."
    )

manifest = pd.read_csv(MANIFEST)

if len(manifest) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} rows, "
        f"found {len(manifest)}."
    )

if not manifest["audit_row_id"].is_unique:
    raise RuntimeError(
        "audit_row_id is not unique."
    )

coverage = np.zeros(
    EXPECTED_ROWS,
    dtype=np.uint8,
)

if FINAL_EMBEDDINGS.exists():
    FINAL_EMBEDDINGS.unlink()

merged = open_memmap(
    FINAL_EMBEDDINGS,
    mode="w+",
    dtype=np.float32,
    shape=(EXPECTED_ROWS, EXPECTED_DIMENSION),
)

merged[:] = np.nan
merged.flush()

shard_records = []

for shard in SHARDS:
    embedding_path = (
        shard["dir"] / shard["embeddings"]
    )

    indices_path = (
        shard["dir"] / shard["indices"]
    )

    metadata_path = (
        shard["dir"] / shard["metadata"]
    )

    for path in [
        embedding_path,
        indices_path,
        metadata_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if metadata.get("status") != "PASS":
        raise RuntimeError(
            f"Shard {shard['id']} metadata "
            "does not report PASS."
        )

    embeddings = np.load(
        embedding_path,
        mmap_mode="r",
    )

    indices = np.load(indices_path)

    expected_indices = np.arange(
        shard["id"],
        EXPECTED_ROWS,
        2,
        dtype=np.int64,
    )

    if not np.array_equal(
        indices,
        expected_indices,
    ):
        raise RuntimeError(
            f"Shard {shard['id']} indices "
            "do not match deterministic mapping."
        )

    if embeddings.shape != (
        len(indices),
        EXPECTED_DIMENSION,
    ):
        raise RuntimeError(
            f"Shard {shard['id']} shape "
            f"{embeddings.shape} is invalid."
        )

    if embeddings.dtype != np.float32:
        raise RuntimeError(
            f"Shard {shard['id']} dtype "
            f"{embeddings.dtype} is invalid."
        )

    if np.any(coverage[indices] != 0):
        raise RuntimeError(
            f"Duplicate coverage in shard "
            f"{shard['id']}."
        )

    merged[indices] = embeddings
    coverage[indices] += 1
    merged.flush()

    shard_records.append({
        "shard_id": shard["id"],
        "rows": int(len(indices)),
        "first_global_index": int(indices[0]),
        "last_global_index": int(indices[-1]),
        "embeddings_path": str(
            embedding_path
        ),
        "embeddings_sha256": sha256_file(
            embedding_path
        ),
        "indices_path": str(indices_path),
        "indices_sha256": sha256_file(
            indices_path
        ),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(
            metadata_path
        ),
    })

unique_coverage, coverage_counts = np.unique(
    coverage,
    return_counts=True,
)

coverage_summary = {
    str(int(value)): int(count)
    for value, count in zip(
        unique_coverage,
        coverage_counts,
    )
}

print("Coverage summary:", coverage_summary)

if coverage_summary != {"1": EXPECTED_ROWS}:
    raise RuntimeError(
        "Merged coverage is not exactly once."
    )

merged.flush()

final = np.load(
    FINAL_EMBEDDINGS,
    mmap_mode="r",
)

norms = np.empty(
    EXPECTED_ROWS,
    dtype=np.float64,
)

finite = True

for start in range(0, EXPECTED_ROWS, 2048):
    stop = min(
        start + 2048,
        EXPECTED_ROWS,
    )

    chunk = np.asarray(
        final[start:stop],
        dtype=np.float64,
    )

    if not np.isfinite(chunk).all():
        finite = False

    norms[start:stop] = np.linalg.norm(
        chunk,
        axis=1,
    )

index_columns = [
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

index_frame = manifest[index_columns].copy()

index_frame.insert(
    0,
    "embedding_index",
    np.arange(
        EXPECTED_ROWS,
        dtype=np.int64,
    ),
)

index_frame.to_csv(
    FINAL_INDEX,
    index=False,
)

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
        f"unexpected dtype {final.dtype}"
    )

if not finite:
    failures.append(
        "non-finite embeddings detected"
    )

if np.any(
    np.abs(norms - 1.0) > 0.002
):
    failures.append(
        "embedding norm tolerance exceeded"
    )

status = "PASS" if not failures else "FAIL"

metadata = {
    "status": status,
    "audit": (
        "TAM-VLM external Qwen "
        "two-shard merge verification"
    ),
    "manifest_path": str(MANIFEST),
    "manifest_sha256": manifest_sha256,
    "manifest_rows": EXPECTED_ROWS,
    "embedding_shape": list(final.shape),
    "embedding_dtype": str(final.dtype),
    "embeddings_path": str(
        FINAL_EMBEDDINGS
    ),
    "embeddings_sha256": sha256_file(
        FINAL_EMBEDDINGS
    ),
    "embedding_index_path": str(
        FINAL_INDEX
    ),
    "embedding_index_sha256": sha256_file(
        FINAL_INDEX
    ),
    "coverage_unique_values": (
        coverage_summary
    ),
    "finite": finite,
    "normalization": True,
    "norm_min": float(norms.min()),
    "norm_median": float(
        np.median(norms)
    ),
    "norm_max": float(norms.max()),
    "shards": shard_records,
    "failures": failures,
}

atomic_json_write(
    FINAL_METADATA,
    metadata,
)

print("\n" + "=" * 112)
print("Qwen external merged embedding summary")
print("=" * 112)
print("Status             :", status)
print("Embedding shape    :", final.shape)
print("Embedding dtype    :", final.dtype)
print("Coverage           :", coverage_summary)
print(
    "Embedding norms    :",
    f"{norms.min():.8f}–{norms.max():.8f}",
)
print(
    "Norm median        :",
    f"{np.median(norms):.8f}",
)
print(
    "Embeddings SHA256  :",
    metadata["embeddings_sha256"],
)
print(
    "Index SHA256       :",
    metadata["embedding_index_sha256"],
)
print(
    "Saved metadata     :",
    FINAL_METADATA,
)

if status != "PASS":
    raise SystemExit(
        "Qwen merge verification failed."
    )
