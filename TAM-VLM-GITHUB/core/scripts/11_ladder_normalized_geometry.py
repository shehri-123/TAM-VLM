"""Normalized CLIP-space geometry for TAM-VLM ladder failure.

Diagnostic only:
- no model training,
- no hyperparameter tuning on ladder,
- no test-set method selection.

Copy into TAM_VLM_Publication_v3 and run after sourcing env_v3.sh.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from final_config import *
from experiment_utils import load_inputs, base_image_id, save_json


TARGET = "ladder"
OUT = OUT_DIR / "ladder_forensics"
OUT.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / max(n, eps)


def add_base_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_id"] = out["image"].astype(str).map(base_image_id)
    return out


def paired_indices(df: pd.DataFrame, split_name: str, trigger_name: str):
    part = add_base_id(df.loc[df["split"].eq(split_name), ["image", "scene", "trigger"]])
    part["row_index"] = part.index.astype(int)

    clean = (
        part.loc[part["trigger"].eq("none"), ["scene", "base_id", "row_index"]]
        .rename(columns={"row_index": "clean_index"})
    )
    trig = (
        part.loc[part["trigger"].eq(trigger_name), ["scene", "base_id", "row_index"]]
        .rename(columns={"row_index": "trigger_index"})
    )
    merged = clean.merge(trig, on=["scene", "base_id"], how="inner", validate="one_to_one")
    return (
        merged["clean_index"].to_numpy(dtype=int),
        merged["trigger_index"].to_numpy(dtype=int),
        merged,
    )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def reconstruction_fraction(v: np.ndarray, basis: np.ndarray, rank: int) -> float:
    """Fraction of squared vector norm explained by first `rank` orthonormal rows."""
    v = np.asarray(v, dtype=np.float64)
    if np.linalg.norm(v) == 0:
        return float("nan")
    B = basis[:rank]
    projection = B.T @ (B @ v)
    return float(np.dot(projection, projection) / np.dot(v, v))


def main():
    df, X, _ = load_inputs(MANIFEST, EMB_DIR / "embeddings.npy")
    Z = l2_normalize(np.asarray(X))
    df = add_base_id(df)

    shift_vectors = {}
    rows = []

    # Use train pairs for known triggers and test pairs for the held-out ladder.
    for trigger_name in ALL_TRIGGERS:
        split_name = "test" if trigger_name == TARGET else "train"
        clean_idx, trig_idx, pairs = paired_indices(df, split_name, trigger_name)

        z0 = Z[clean_idx]
        zt = Z[trig_idx]
        delta = zt - z0
        pair_cosine_distance = 1.0 - np.sum(zt * z0, axis=1)
        mean_delta = delta.mean(axis=0)
        shift_vectors[trigger_name] = mean_delta

        rows.append(
            {
                "trigger": trigger_name,
                "split_used": split_name,
                "n_pairs": len(pairs),
                "pair_cosine_distance_mean": float(pair_cosine_distance.mean()),
                "pair_cosine_distance_median": float(np.median(pair_cosine_distance)),
                "pair_cosine_distance_std": float(pair_cosine_distance.std(ddof=1)),
                "normalized_delta_l2_mean": float(np.linalg.norm(delta, axis=1).mean()),
                "mean_shift_vector_l2": float(np.linalg.norm(mean_delta)),
            }
        )

    ladder_shift = shift_vectors[TARGET]
    known_triggers = [t for t in ALL_TRIGGERS if t != TARGET]
    known_matrix = np.stack([shift_vectors[t] for t in known_triggers], axis=0)

    # Common direction from the eleven known triggers.
    common_known_direction = unit(np.stack([unit(v) for v in known_matrix]).mean(axis=0))

    # Orthonormal basis of known-trigger mean-shift subspace.
    _, _, vh = np.linalg.svd(known_matrix, full_matrices=False)

    for row in rows:
        t = row["trigger"]
        v = shift_vectors[t]
        row["cosine_to_ladder_mean_shift"] = cosine(v, ladder_shift)
        row["cosine_to_common_known_direction"] = cosine(v, common_known_direction)
        for rank in (1, 2, 3, 5, 10):
            row[f"known_subspace_fraction_rank{rank}"] = reconstruction_fraction(v, vh, rank)

    shift_df = pd.DataFrame(rows).sort_values("cosine_to_ladder_mean_shift", ascending=False)
    shift_df.to_csv(OUT / "ladder_normalized_shift_geometry.csv", index=False)

    # Leave-one-known-trigger-out subspace reconstruction provides a fair comparator.
    loo_rows = []
    for held in ALL_TRIGGERS:
        comparison = [t for t in ALL_TRIGGERS if t not in {held, TARGET}]
        # For ladder, use all eleven known triggers.
        if held == TARGET:
            comparison = known_triggers
        M = np.stack([shift_vectors[t] for t in comparison], axis=0)
        _, _, basis = np.linalg.svd(M, full_matrices=False)
        v = shift_vectors[held]
        r = min(5, len(comparison))
        loo_rows.append(
            {
                "held_out_trigger": held,
                "reference_trigger_count": len(comparison),
                "rank_used": r,
                "subspace_fraction_explained": reconstruction_fraction(v, basis, r),
                "cosine_to_reference_mean_direction": cosine(
                    v, unit(np.stack([unit(shift_vectors[t]) for t in comparison]).mean(axis=0))
                ),
            }
        )
    loo_df = pd.DataFrame(loo_rows).sort_values("subspace_fraction_explained")
    loo_df.to_csv(OUT / "trigger_shift_subspace_loo.csv", index=False)

    # Normalized centroids.
    sp = df["split"].to_numpy()
    trig = df["trigger"].to_numpy()
    ladder_idx = np.flatnonzero((sp == "test") & (trig == TARGET))
    ladder_centroid = unit(Z[ladder_idx].mean(axis=0))

    centroid_rows = []
    refs = [("clean_test", np.flatnonzero((sp == "test") & (trig == "none"))),
            ("clean_train", np.flatnonzero((sp == "train") & (trig == "none")))]
    refs += [
        (f"train_trigger::{t}", np.flatnonzero((sp == "train") & (trig == t)))
        for t in known_triggers
    ]

    for name, idx in refs:
        c = unit(Z[idx].mean(axis=0))
        sim = float(np.dot(ladder_centroid, c))
        centroid_rows.append(
            {
                "reference": name,
                "cosine_similarity_to_ladder_centroid": sim,
                "cosine_distance_to_ladder_centroid": 1.0 - sim,
            }
        )
    centroid_df = pd.DataFrame(centroid_rows).sort_values(
        "cosine_distance_to_ladder_centroid"
    )
    centroid_df.to_csv(OUT / "ladder_normalized_centroid_geometry.csv", index=False)

    ladder_row = shift_df.loc[shift_df["trigger"] == TARGET].iloc[0].to_dict()
    ladder_loo = loo_df.loc[loo_df["held_out_trigger"] == TARGET].iloc[0].to_dict()

    summary = {
        "diagnostic_only": True,
        "features_l2_normalized": True,
        "ladder_pair_cosine_distance_mean": ladder_row["pair_cosine_distance_mean"],
        "ladder_pair_cosine_distance_median": ladder_row["pair_cosine_distance_median"],
        "ladder_cosine_to_common_known_direction": ladder_row[
            "cosine_to_common_known_direction"
        ],
        "ladder_known_subspace_fraction_rank1": ladder_row[
            "known_subspace_fraction_rank1"
        ],
        "ladder_known_subspace_fraction_rank3": ladder_row[
            "known_subspace_fraction_rank3"
        ],
        "ladder_known_subspace_fraction_rank5": ladder_row[
            "known_subspace_fraction_rank5"
        ],
        "ladder_loo_rank5_fraction": ladder_loo["subspace_fraction_explained"],
        "ladder_centroid_nearest_references": centroid_df.head(5).to_dict(
            orient="records"
        ),
    }
    save_json(OUT / "ladder_normalized_geometry_summary.json", summary)

    print("=== NORMALIZED LADDER GEOMETRY ===")
    print(json.dumps(summary, indent=2))
    print("\nMean-shift similarity to ladder:")
    print(
        shift_df[
            [
                "trigger",
                "pair_cosine_distance_mean",
                "cosine_to_ladder_mean_shift",
                "cosine_to_common_known_direction",
                "known_subspace_fraction_rank5",
            ]
        ].head(12).to_string(index=False)
    )
    print("\nLOO shift-subspace comparison (lowest first):")
    print(loo_df.to_string(index=False))
    print("\nNearest normalized centroids:")
    print(centroid_df.head(8).to_string(index=False))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
