#!/usr/bin/env python3

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image


AUDIT_ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
MANIFEST = Path(
    os.environ["TAMVLM_CORE_MANIFEST"]
)
EMBEDDINGS = Path(
    os.environ["TAMVLM_OPENCLIP_EMBEDDINGS"]
)
WEIGHTS = Path(
    os.environ["TAMVLM_OPENCLIP_WEIGHTS"]
)
REPORT_JSON = (
    AUDIT_ROOT / "protocol" /
    "OPENCLIP_EXTRACTION_EQUIVALENCE_v1.json"
)

REPORT_TXT = (
    AUDIT_ROOT / "protocol" /
    "OPENCLIP_EXTRACTION_EQUIVALENCE_v1.txt"
)

MODEL_NAME = "ViT-B-16"

# Deterministic, widely distributed manifest rows.
REQUESTED_INDICES = [
    0,
    1,
    123,
    1000,
    5000,
    10000,
    20000,
    30000,
    40000,
    48632,
]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def l2_normalize(array):
    norms = np.linalg.norm(
        array.astype(np.float64),
        axis=1,
        keepdims=True,
    )

    return array / np.maximum(norms, 1e-12)


def cosine_per_row(left, right):
    numerator = np.sum(
        left.astype(np.float64) *
        right.astype(np.float64),
        axis=1,
    )

    denominator = (
        np.linalg.norm(left.astype(np.float64), axis=1) *
        np.linalg.norm(right.astype(np.float64), axis=1)
    )

    return numerator / np.maximum(denominator, 1e-12)


for required_path in [MANIFEST, EMBEDDINGS, WEIGHTS]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 112)
print("OpenCLIP original-extraction equivalence audit")
print("=" * 112)
print("Device:", device)
print("Model:", MODEL_NAME)
print("Weights:", WEIGHTS)

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=str(WEIGHTS),
)

model = model.eval().to(device)

print("\nencode_image signature:")
print(inspect.signature(model.encode_image))

print("\nResolved preprocessing transform:")
print(preprocess)

visual_preprocess_cfg = getattr(
    model.visual,
    "preprocess_cfg",
    None,
)

print("\nModel visual preprocess_cfg:")
print(visual_preprocess_cfg)

manifest = pd.read_csv(MANIFEST)
stored_embeddings = np.load(
    EMBEDDINGS,
    mmap_mode="r",
)

print("\nStored embedding file:")
print("  shape:", stored_embeddings.shape)
print("  dtype:", stored_embeddings.dtype)
print("  manifest rows:", len(manifest))

if stored_embeddings.shape != (len(manifest), 512):
    raise RuntimeError(
        "Unexpected OpenCLIP embedding shape: "
        f"{stored_embeddings.shape}"
    )

# Audit norm distribution without loading the complete matrix twice.
all_norms = np.empty(
    stored_embeddings.shape[0],
    dtype=np.float64,
)

chunk_size = 4096

for start in range(
    0,
    stored_embeddings.shape[0],
    chunk_size,
):
    stop = min(
        start + chunk_size,
        stored_embeddings.shape[0],
    )

    chunk = np.asarray(
        stored_embeddings[start:stop],
        dtype=np.float64,
    )

    all_norms[start:stop] = np.linalg.norm(
        chunk,
        axis=1,
    )

print("\nStored embedding norms:")
print("  min   :", float(all_norms.min()))
print("  median:", float(np.median(all_norms)))
print("  max   :", float(all_norms.max()))

indices = [
    index
    for index in REQUESTED_INDICES
    if 0 <= index < len(manifest)
]

paths = [
    Path(manifest.iloc[index]["image"])
    for index in indices
]

missing = [
    str(path)
    for path in paths
    if not path.is_file()
]

if missing:
    raise FileNotFoundError(
        f"Missing smoke-test images: {missing}"
    )

image_tensors = []

for path in paths:
    with Image.open(path) as image:
        image_tensors.append(
            preprocess(image.convert("RGB"))
        )

batch = torch.stack(image_tensors).to(device)

with torch.no_grad():
    # Reproduce the original script exactly:
    # no autocast and no normalize=True argument.
    reproduced_raw = (
        model.encode_image(batch)
        .float()
        .cpu()
        .numpy()
    )

stored_selected = np.asarray(
    stored_embeddings[indices],
    dtype=np.float32,
)

reproduced_normalized = l2_normalize(
    reproduced_raw
).astype(np.float32)

raw_absolute_error = np.abs(
    reproduced_raw - stored_selected
)

normalized_absolute_error = np.abs(
    reproduced_normalized - stored_selected
)

raw_cosine = cosine_per_row(
    reproduced_raw,
    stored_selected,
)

raw_norms = np.linalg.norm(
    reproduced_raw.astype(np.float64),
    axis=1,
)

stored_norms = np.linalg.norm(
    stored_selected.astype(np.float64),
    axis=1,
)

relative_norm_error = (
    np.abs(raw_norms - stored_norms) /
    np.maximum(stored_norms, 1e-12)
)

records = []

print("\nPer-image equivalence:")
for position, index in enumerate(indices):
    record = {
        "manifest_index": int(index),
        "image": str(paths[position]),
        "stored_norm": float(stored_norms[position]),
        "reproduced_raw_norm": float(raw_norms[position]),
        "raw_cosine_similarity": float(
            raw_cosine[position]
        ),
        "raw_mean_absolute_error": float(
            raw_absolute_error[position].mean()
        ),
        "raw_max_absolute_error": float(
            raw_absolute_error[position].max()
        ),
        "relative_norm_error": float(
            relative_norm_error[position]
        ),
        "normalized_candidate_mean_absolute_error": float(
            normalized_absolute_error[position].mean()
        ),
    }

    records.append(record)

    print(
        f"  index={index:5d} | "
        f"cos={record['raw_cosine_similarity']:.10f} | "
        f"raw_MAE={record['raw_mean_absolute_error']:.8e} | "
        f"raw_max={record['raw_max_absolute_error']:.8e} | "
        f"stored_norm={record['stored_norm']:.6f} | "
        f"new_norm={record['reproduced_raw_norm']:.6f}"
    )

raw_mean_absolute_error = float(
    raw_absolute_error.mean()
)

raw_max_absolute_error = float(
    raw_absolute_error.max()
)

normalized_mean_absolute_error = float(
    normalized_absolute_error.mean()
)

minimum_cosine = float(
    raw_cosine.min()
)

maximum_relative_norm_error = float(
    relative_norm_error.max()
)

raw_is_clear_match = (
    raw_mean_absolute_error <
    normalized_mean_absolute_error * 0.10
)

numerically_equivalent = (
    minimum_cosine >= 0.99999
    and maximum_relative_norm_error <= 0.001
)

status = (
    "PASS"
    if raw_is_clear_match and numerically_equivalent
    else "FAIL"
)

payload = {
    "audit_version": "v1",
    "status": status,
    "model_name": MODEL_NAME,
    "weights_path": str(WEIGHTS),
    "weights_sha256": sha256_file(WEIGHTS),
    "manifest_path": str(MANIFEST),
    "manifest_sha256": sha256_file(MANIFEST),
    "embeddings_path": str(EMBEDDINGS),
    "embeddings_shape": list(
        stored_embeddings.shape
    ),
    "embeddings_dtype": str(
        stored_embeddings.dtype
    ),
    "stored_norm_min": float(
        all_norms.min()
    ),
    "stored_norm_median": float(
        np.median(all_norms)
    ),
    "stored_norm_max": float(
        all_norms.max()
    ),
    "encode_image_signature": str(
        inspect.signature(model.encode_image)
    ),
    "preprocess_repr": repr(preprocess),
    "visual_preprocess_cfg": (
        visual_preprocess_cfg
    ),
    "precision": "float32",
    "autocast_used": False,
    "explicit_output_normalization_used": False,
    "raw_mean_absolute_error": (
        raw_mean_absolute_error
    ),
    "raw_max_absolute_error": (
        raw_max_absolute_error
    ),
    "normalized_candidate_mean_absolute_error": (
        normalized_mean_absolute_error
    ),
    "minimum_raw_cosine_similarity": (
        minimum_cosine
    ),
    "maximum_relative_norm_error": (
        maximum_relative_norm_error
    ),
    "representation_matching_stored_embeddings": (
        "raw_encode_image_output"
        if raw_is_clear_match
        else "not_confirmed"
    ),
    "records": records,
}

REPORT_JSON.write_text(
    json.dumps(
        payload,
        indent=2,
        default=str,
    ) + "\n",
    encoding="utf-8",
)

report_lines = [
    "=" * 112,
    "OpenCLIP original-extraction equivalence audit",
    "=" * 112,
    f"Status                           : {status}",
    f"Stored embeddings                : {stored_embeddings.shape}, {stored_embeddings.dtype}",
    f"Stored norm range                : {all_norms.min():.8f}–{all_norms.max():.8f}",
    f"Stored norm median               : {np.median(all_norms):.8f}",
    f"Minimum raw cosine similarity    : {minimum_cosine:.10f}",
    f"Raw mean absolute error          : {raw_mean_absolute_error:.8e}",
    f"Raw maximum absolute error       : {raw_max_absolute_error:.8e}",
    f"Normalized-candidate mean error  : {normalized_mean_absolute_error:.8e}",
    f"Maximum relative norm error      : {maximum_relative_norm_error:.8e}",
    "Confirmed representation         : "
    + (
        "raw encode_image output"
        if raw_is_clear_match
        else "not confirmed"
    ),
    "External extraction rule         : RGB + original preprocess + FP32 encode_image; no output L2 normalization",
    f"Saved JSON                       : {REPORT_JSON}",
]

REPORT_TXT.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report_lines))
