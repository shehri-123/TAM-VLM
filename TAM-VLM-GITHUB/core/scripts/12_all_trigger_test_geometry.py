"""TAM-VLM all-trigger test-split geometry versus strict LOTO performance.

Why this script is needed
-------------------------
Earlier ladder geometry used training pairs for known triggers but test pairs for
the held-out ladder. That is useful for a first diagnostic, but it is not a fair
cross-trigger comparison. This script measures ALL twelve trigger categories on
the identical test split (599 paired images, 15 scenes per category), then merges
the measurements with the freshly rerun LOTO summary.

This is exploratory diagnostic analysis only:
- no detector is trained,
- no threshold or method is selected,
- no trigger is removed,
- no ladder-specific hyperparameter is tuned.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from final_config import *
from experiment_utils import load_inputs, base_image_id, save_json


OUT = OUT_DIR / "trigger_geometry"
OUT.mkdir(parents=True, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), eps)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def reconstruction_fraction(v: np.ndarray, basis_rows: np.ndarray, rank: int) -> float:
    v = np.asarray(v, dtype=np.float64)
    denom = float(np.dot(v, v))
    if denom <= 0:
        return float("nan")
    rank = min(rank, len(basis_rows))
    B = basis_rows[:rank]
    proj = B.T @ (B @ v)
    return float(np.dot(proj, proj) / denom)


def add_base_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_id"] = out["image"].astype(str).map(base_image_id)
    return out


def paired_test_indices(df: pd.DataFrame, trigger_name: str):
    part = add_base_id(
        df.loc[
            (df["split"] == "test")
            & df["trigger"].isin(["none", trigger_name]),
            ["image", "scene", "trigger"],
        ]
    )
    part["row_index"] = part.index.astype(int)

    clean = (
        part.loc[part["trigger"] == "none", ["scene", "base_id", "row_index"]]
        .rename(columns={"row_index": "clean_index"})
    )
    trig = (
        part.loc[part["trigger"] == trigger_name, ["scene", "base_id", "row_index"]]
        .rename(columns={"row_index": "trigger_index"})
    )
    merged = clean.merge(
        trig, on=["scene", "base_id"], how="inner", validate="one_to_one"
    )
    if len(merged) != 599 or merged["scene"].nunique() != 15:
        raise ValueError(
            f"{trigger_name}: expected 599 paired test images over 15 scenes; "
            f"got {len(merged)} pairs over {merged['scene'].nunique()} scenes"
        )
    return (
        merged["clean_index"].to_numpy(dtype=int),
        merged["trigger_index"].to_numpy(dtype=int),
        merged,
    )


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down family-wise adjusted p-values."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted_sorted[idx] = min(running, 1.0)
    return adjusted_sorted.tolist()


def main():
    df, X, _ = load_inputs(MANIFEST, EMB_DIR / "embeddings.npy")
    Z = l2_normalize(np.asarray(X))

    shifts = {}
    trigger_centroids = {}
    clean_centroid = None
    rows = []

    for trigger_name in ALL_TRIGGERS:
        clean_idx, trig_idx, pairs = paired_test_indices(df, trigger_name)
        z0, zt = Z[clean_idx], Z[trig_idx]
        delta = zt - z0
        pair_cos_dist = 1.0 - np.sum(zt * z0, axis=1)

        if clean_centroid is None:
            clean_centroid = unit(z0.mean(axis=0))

        mean_shift = delta.mean(axis=0)
        trig_centroid = unit(zt.mean(axis=0))
        shifts[trigger_name] = mean_shift
        trigger_centroids[trigger_name] = trig_centroid

        rows.append(
            {
                "trigger": trigger_name,
                "n_pairs": len(pairs),
                "n_scenes": int(pairs["scene"].nunique()),
                "pair_cosine_distance_mean": float(pair_cos_dist.mean()),
                "pair_cosine_distance_median": float(np.median(pair_cos_dist)),
                "pair_cosine_distance_std": float(pair_cos_dist.std(ddof=1)),
                "normalized_delta_l2_mean": float(
                    np.linalg.norm(delta, axis=1).mean()
                ),
                "mean_shift_vector_l2": float(np.linalg.norm(mean_shift)),
                "cosine_similarity_to_clean_centroid": float(
                    np.dot(trig_centroid, clean_centroid)
                ),
                "cosine_distance_to_clean_centroid": float(
                    1.0 - np.dot(trig_centroid, clean_centroid)
                ),
            }
        )

    # Leave-one-trigger-out geometric descriptors, computed fairly for every trigger.
    for row in rows:
        held = row["trigger"]
        others = [t for t in ALL_TRIGGERS if t != held]
        M = np.stack([shifts[t] for t in others], axis=0)
        _, _, vh = np.linalg.svd(M, full_matrices=False)

        loo_common = unit(np.stack([unit(shifts[t]) for t in others]).mean(axis=0))
        held_shift = shifts[held]
        similarities = {t: cosine(held_shift, shifts[t]) for t in others}
        nearest = max(similarities, key=similarities.get)

        row["cosine_to_loo_common_direction"] = cosine(held_shift, loo_common)
        row["nearest_other_shift_trigger"] = nearest
        row["max_cosine_to_other_trigger_shift"] = similarities[nearest]
        for rank in (1, 3, 5, 10):
            row[f"loo_subspace_fraction_rank{rank}"] = reconstruction_fraction(
                held_shift, vh, rank
            )

    geometry = pd.DataFrame(rows)

    loto_path = OUT_DIR / "loto_summary.csv"
    if not loto_path.exists():
        raise FileNotFoundError(
            f"Missing {loto_path}. Run 04_loto_3seeds.py first."
        )
    loto = pd.read_csv(loto_path)
    if "held_out_trigger" not in loto.columns:
        raise ValueError("loto_summary.csv has no held_out_trigger column")

    merged = geometry.merge(
        loto[
            [
                "held_out_trigger",
                "unseen_AUROC_mean",
                "unseen_AUROC_std",
                "seen_AUROC_mean",
                "val1_TPR_mean",
                "val1_FPR_mean",
            ]
        ],
        left_on="trigger",
        right_on="held_out_trigger",
        how="left",
        validate="one_to_one",
    ).drop(columns=["held_out_trigger"])

    if merged["unseen_AUROC_mean"].isna().any():
        raise ValueError("Some trigger geometry rows did not match LOTO results")

    merged = merged.sort_values("unseen_AUROC_mean", ascending=False)
    merged.to_csv(OUT / "all_trigger_test_geometry_with_loto.csv", index=False)

    metrics = [
        "pair_cosine_distance_mean",
        "mean_shift_vector_l2",
        "cosine_distance_to_clean_centroid",
        "cosine_to_loo_common_direction",
        "max_cosine_to_other_trigger_shift",
        "loo_subspace_fraction_rank1",
        "loo_subspace_fraction_rank3",
        "loo_subspace_fraction_rank5",
    ]

    corr_rows = []
    for metric in metrics:
        x = merged[metric].to_numpy(dtype=float)
        y = merged["unseen_AUROC_mean"].to_numpy(dtype=float)
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        corr_rows.append(
            {
                "predictor": metric,
                "n_triggers": len(merged),
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_rho": float(sr),
                "spearman_p": float(sp),
            }
        )

    corr = pd.DataFrame(corr_rows)
    corr["pearson_p_holm"] = holm_adjust(corr["pearson_p"].tolist())
    corr["spearman_p_holm"] = holm_adjust(corr["spearman_p"].tolist())
    corr["exploratory_only_n12"] = True
    corr.to_csv(OUT / "geometry_loto_correlations.csv", index=False)

    # Report ladder's rank among all triggers without declaring it an outlier.
    ranks = {}
    ladder = merged.loc[merged["trigger"] == "ladder"].iloc[0]
    for metric in metrics + ["unseen_AUROC_mean", "val1_TPR_mean"]:
        ascending = metric in {
            "unseen_AUROC_mean",
            "val1_TPR_mean",
            "cosine_to_loo_common_direction",
            "max_cosine_to_other_trigger_shift",
            "loo_subspace_fraction_rank1",
            "loo_subspace_fraction_rank3",
            "loo_subspace_fraction_rank5",
        }
        rank = merged[metric].rank(method="min", ascending=ascending)
        ranks[metric] = {
            "ladder_value": float(ladder[metric]),
            "ladder_rank_low_to_high" if ascending else "ladder_rank_high_to_low": int(
                rank[merged["trigger"] == "ladder"].iloc[0]
            ),
            "n_triggers": len(merged),
        }

    summary = {
        "analysis": "all 12 triggers measured on identical test pairs/scenes",
        "n_triggers": 12,
        "pairs_per_trigger": 599,
        "test_scenes": 15,
        "exploratory_not_confirmatory": True,
        "ladder_unseen_AUROC": float(ladder["unseen_AUROC_mean"]),
        "ladder_val1_TPR": float(ladder["val1_TPR_mean"]),
        "ladder_ranks": ranks,
        "holm_corrected_significant_predictors_pearson": corr.loc[
            corr["pearson_p_holm"] < 0.05, "predictor"
        ].tolist(),
        "holm_corrected_significant_predictors_spearman": corr.loc[
            corr["spearman_p_holm"] < 0.05, "predictor"
        ].tolist(),
    }
    save_json(OUT / "all_trigger_test_geometry_summary.json", summary)

    print("=== ALL-TRIGGER TEST-SPLIT GEOMETRY ===")
    print(
        merged[
            [
                "trigger",
                "unseen_AUROC_mean",
                "val1_TPR_mean",
                "pair_cosine_distance_mean",
                "cosine_distance_to_clean_centroid",
                "cosine_to_loo_common_direction",
                "loo_subspace_fraction_rank5",
                "nearest_other_shift_trigger",
            ]
        ].to_string(index=False)
    )
    print("\n=== EXPLORATORY CORRELATIONS (Holm-adjusted) ===")
    print(corr.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
