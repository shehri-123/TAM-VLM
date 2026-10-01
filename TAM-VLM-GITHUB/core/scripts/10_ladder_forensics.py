"""TAM-VLM ladder failure forensics (Publication v3.1).

Purpose
-------
Reproduce the strict LOTO ladder fold over three seeds, save checkpoints and
predictions, verify data pairing, quantify score overlap, inspect scene-level
behavior, and compare the ladder embedding-shift direction with the eleven
training triggers.

This script DOES NOT tune anything on ladder. It is diagnostic only.

Run from the TAM_VLM_Publication_v3 directory:
    source env_v3.sh
    export CUDA_VISIBLE_DEVICES=1
    python -u 10_ladder_forensics.py 2>&1 | \
      tee "$TAMVLM_OUT_DIR/logs/10_ladder_forensics.log"
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import roc_auc_score
import torch

from final_config import *
from experiment_utils import *


TARGET = "ladder"
FORENSICS_DIR = OUT_DIR / "ladder_forensics"
FORENSICS_DIR.mkdir(parents=True, exist_ok=True)


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for a=positive/ladder and b=clean."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = (
        (len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)
    ) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def add_base_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["base_id"] = out["image"].astype(str).map(base_image_id)
    return out


def paired_indices(df: pd.DataFrame, split_name: str, trigger_name: str):
    """Return aligned null/trigger row-index arrays paired by scene and base image."""
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

    merged = clean.merge(
        trig, on=["scene", "base_id"], how="inner", validate="one_to_one"
    )
    return (
        merged["clean_index"].to_numpy(dtype=int),
        merged["trigger_index"].to_numpy(dtype=int),
        merged,
    )


def compute_shift_table(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Compare mean paired embedding-shift direction for ladder vs all triggers."""
    mean_shifts = {}
    rows = []

    for trigger_name in ALL_TRIGGERS:
        # Use test for target ladder; use train for available training triggers.
        split_name = "test" if trigger_name == TARGET else "train"
        clean_idx, trig_idx, pairs = paired_indices(df, split_name, trigger_name)
        delta = np.asarray(X[trig_idx], dtype=np.float64) - np.asarray(
            X[clean_idx], dtype=np.float64
        )
        mean_delta = delta.mean(axis=0)
        mean_shifts[trigger_name] = mean_delta
        rows.append(
            {
                "trigger": trigger_name,
                "split_used": split_name,
                "n_pairs": len(pairs),
                "mean_pair_delta_l2": float(np.linalg.norm(delta, axis=1).mean()),
                "median_pair_delta_l2": float(np.median(np.linalg.norm(delta, axis=1))),
                "mean_shift_vector_l2": float(np.linalg.norm(mean_delta)),
            }
        )

    ladder_shift = mean_shifts[TARGET]
    for row in rows:
        row["cosine_to_ladder_mean_shift"] = cosine(
            ladder_shift, mean_shifts[row["trigger"]]
        )

    result = pd.DataFrame(rows).sort_values(
        "cosine_to_ladder_mean_shift", ascending=False
    )
    result.to_csv(FORENSICS_DIR / "ladder_shift_similarity.csv", index=False)
    return result


def compute_centroid_geometry(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    sp = df["split"].to_numpy()
    trig = df["trigger"].to_numpy()

    clean_train = np.flatnonzero((sp == "train") & (trig == "none"))
    ladder_test = np.flatnonzero((sp == "test") & (trig == TARGET))
    clean_test = np.flatnonzero((sp == "test") & (trig == "none"))

    ladder_centroid = np.asarray(X[ladder_test], dtype=np.float64).mean(axis=0)
    clean_test_centroid = np.asarray(X[clean_test], dtype=np.float64).mean(axis=0)
    clean_train_centroid = np.asarray(X[clean_train], dtype=np.float64).mean(axis=0)

    rows = [
        {
            "reference": "clean_train_centroid",
            "cosine_to_ladder_centroid": cosine(
                ladder_centroid, clean_train_centroid
            ),
            "euclidean_to_ladder_centroid": float(
                np.linalg.norm(ladder_centroid - clean_train_centroid)
            ),
        },
        {
            "reference": "clean_test_centroid",
            "cosine_to_ladder_centroid": cosine(
                ladder_centroid, clean_test_centroid
            ),
            "euclidean_to_ladder_centroid": float(
                np.linalg.norm(ladder_centroid - clean_test_centroid)
            ),
        },
    ]

    for t in ALL_TRIGGERS:
        if t == TARGET:
            continue
        idx = np.flatnonzero((sp == "train") & (trig == t))
        c = np.asarray(X[idx], dtype=np.float64).mean(axis=0)
        rows.append(
            {
                "reference": f"train_trigger_centroid::{t}",
                "cosine_to_ladder_centroid": cosine(ladder_centroid, c),
                "euclidean_to_ladder_centroid": float(
                    np.linalg.norm(ladder_centroid - c)
                ),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "euclidean_to_ladder_centroid", ascending=True
    )
    result.to_csv(FORENSICS_DIR / "ladder_centroid_geometry.csv", index=False)
    return result


def write_qc_manifest(df: pd.DataFrame, n_scenes: int = 20) -> pd.DataFrame:
    """Create a fixed, scene-diverse list for blinded manual visual inspection."""
    part = add_base_id(
        df.loc[
            (df["split"] == "test") & df["trigger"].isin(["none", TARGET]),
            ["image", "scene", "trigger"],
        ]
    )
    pivot = (
        part.pivot_table(
            index=["scene", "base_id"], columns="trigger", values="image", aggfunc="first"
        )
        .dropna(subset=["none", TARGET])
        .reset_index()
    )

    # One pair per scene first; fixed random ordering for reproducibility.
    rng = np.random.RandomState(20260721)
    scenes = np.array(sorted(pivot["scene"].unique()))
    rng.shuffle(scenes)
    chosen = []
    for scene in scenes[: min(n_scenes, len(scenes))]:
        q = pivot[pivot["scene"] == scene]
        chosen.append(q.iloc[rng.randint(len(q))])
    qc = pd.DataFrame(chosen)
    qc = qc.rename(
        columns={"none": "null_edit_image", TARGET: "ladder_image"}
    )
    qc["manual_visible"] = ""
    qc["manual_partial"] = ""
    qc["manual_absent"] = ""
    qc["manual_extra_changes"] = ""
    qc["manual_notes"] = ""
    qc.to_csv(FORENSICS_DIR / "ladder_manual_qc_pairs.csv", index=False)
    return qc


def make_montage(qc: pd.DataFrame):
    """Create a paired null/ladder montage when image paths are accessible."""
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warning] montage skipped; missing PIL/matplotlib: {exc}")
        return

    usable = []
    for _, row in qc.iterrows():
        p0, p1 = str(row["null_edit_image"]), str(row["ladder_image"])
        if os.path.exists(p0) and os.path.exists(p1):
            usable.append(row)
        if len(usable) >= 12:
            break

    if not usable:
        print("[warning] montage skipped; manifest image paths are not accessible")
        return

    fig, axes = plt.subplots(len(usable), 2, figsize=(10, 3.2 * len(usable)))
    if len(usable) == 1:
        axes = np.asarray([axes])

    for r, row in enumerate(usable):
        for c, col in enumerate(["null_edit_image", "ladder_image"]):
            with Image.open(str(row[col])) as im:
                axes[r, c].imshow(im.convert("RGB"))
            axes[r, c].axis("off")
            label = "Null edit" if c == 0 else "Ladder"
            axes[r, c].set_title(f"{label} | scene={row['scene']}")

    fig.tight_layout()
    fig.savefig(FORENSICS_DIR / "ladder_qc_montage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    device = resolve_device("cuda")
    df, X, y = load_inputs(MANIFEST, EMB_DIR / "embeddings.npy")
    df = add_base_id(df)

    split = df["split"].to_numpy()
    trig = df["trigger"].to_numpy()

    train_idx = np.flatnonzero((split == "train") & (trig != TARGET))
    val_idx = np.flatnonzero((split == "val") & (trig != TARGET))
    test_idx = np.flatnonzero(
        (split == "test") & np.isin(trig, ["none", TARGET])
    )

    # Pairing and row-count audit.
    pairing_rows = []
    for split_name in ("train", "val", "test"):
        clean_idx, ladder_idx, pairs = paired_indices(df, split_name, TARGET)
        pairing_rows.append(
            {
                "split": split_name,
                "clean_rows": int(
                    ((df["split"] == split_name) & (df["trigger"] == "none")).sum()
                ),
                "ladder_rows": int(
                    ((df["split"] == split_name) & (df["trigger"] == TARGET)).sum()
                ),
                "paired_rows": len(pairs),
                "unique_scenes": int(pairs["scene"].nunique()),
                "pairing_complete": bool(
                    len(pairs) == len(clean_idx) == len(ladder_idx)
                ),
            }
        )
    pd.DataFrame(pairing_rows).to_csv(
        FORENSICS_DIR / "ladder_pairing_audit.csv", index=False
    )

    all_score_rows = []
    metric_rows = []
    scene_rows = []

    for seed in SEEDS_LOTO:
        model, info = train_detector(
            X,
            y,
            train_idx,
            val_idx,
            seed=seed,
            device=device,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )

        save_checkpoint(
            CKPT_DIR / f"loto_ladder_forensics_seed{seed}.pt",
            model,
            info,
            metadata={
                "held_out_trigger": TARGET,
                "train_excludes_target": True,
                "validation_excludes_target": True,
            },
        )

        val_scores = score_model(model, X, val_idx, device)
        test_scores = score_model(model, X, test_idx, device)
        metrics = evaluate_scores(y[val_idx], val_scores, y[test_idx], test_scores)

        test_frame = df.iloc[test_idx][
            ["image", "scene", "trigger", "label", "base_id"]
        ].copy()
        test_frame["row_index"] = test_idx
        test_frame["seed"] = seed
        test_frame["score"] = test_scores
        all_score_rows.append(test_frame)

        clean_scores = test_scores[y[test_idx] == 0]
        ladder_scores = test_scores[y[test_idx] == 1]
        ks = ks_2samp(ladder_scores, clean_scores, alternative="two-sided")

        row = {
            "seed": seed,
            **metrics,
            **asdict(info),
            "clean_score_mean": float(clean_scores.mean()),
            "clean_score_median": float(np.median(clean_scores)),
            "clean_score_std": safe_std(clean_scores),
            "ladder_score_mean": float(ladder_scores.mean()),
            "ladder_score_median": float(np.median(ladder_scores)),
            "ladder_score_std": safe_std(ladder_scores),
            "cohens_d_ladder_minus_clean": cohens_d(ladder_scores, clean_scores),
            "wasserstein_ladder_vs_clean": float(
                wasserstein_distance(ladder_scores, clean_scores)
            ),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        }
        metric_rows.append(row)

        # Per-scene AUROC: requires both clean and ladder rows in the scene.
        sf = test_frame
        for scene_name, grp in sf.groupby("scene"):
            if grp["label"].nunique() != 2:
                continue
            scene_rows.append(
                {
                    "seed": seed,
                    "scene": scene_name,
                    "n_clean": int((grp["label"] == 0).sum()),
                    "n_ladder": int((grp["label"] == 1).sum()),
                    "scene_AUROC": float(roc_auc_score(grp["label"], grp["score"])),
                    "clean_mean_score": float(
                        grp.loc[grp["label"] == 0, "score"].mean()
                    ),
                    "ladder_mean_score": float(
                        grp.loc[grp["label"] == 1, "score"].mean()
                    ),
                }
            )

        print(
            f"ladder seed={seed} AUROC={metrics['AUROC']:.4f} "
            f"val1TPR={metrics['val1_TPR']:.4f} "
            f"val1FPR={metrics['val1_FPR']:.4f} "
            f"cleanMean={clean_scores.mean():.4f} "
            f"ladderMean={ladder_scores.mean():.4f}"
        )

    scores_df = pd.concat(all_score_rows, ignore_index=True)
    scores_df.to_csv(FORENSICS_DIR / "ladder_test_scores_all_seeds.csv", index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(FORENSICS_DIR / "ladder_seed_metrics.csv", index=False)

    scene_df = pd.DataFrame(scene_rows)
    scene_df.to_csv(FORENSICS_DIR / "ladder_per_scene_metrics.csv", index=False)

    summary = {
        "target": TARGET,
        "seeds": list(SEEDS_LOTO),
        "unseen_AUROC_mean": float(metrics_df["AUROC"].mean()),
        "unseen_AUROC_std_sample": float(metrics_df["AUROC"].std(ddof=1)),
        "val1_TPR_mean": float(metrics_df["val1_TPR"].mean()),
        "val1_TPR_std_sample": float(metrics_df["val1_TPR"].std(ddof=1)),
        "val1_FPR_mean": float(metrics_df["val1_FPR"].mean()),
        "clean_score_mean_across_seeds": float(
            metrics_df["clean_score_mean"].mean()
        ),
        "ladder_score_mean_across_seeds": float(
            metrics_df["ladder_score_mean"].mean()
        ),
        "cohens_d_mean": float(
            metrics_df["cohens_d_ladder_minus_clean"].mean()
        ),
        "wasserstein_mean": float(
            metrics_df["wasserstein_ladder_vs_clean"].mean()
        ),
        "per_scene_AUROC_median": float(scene_df["scene_AUROC"].median()),
        "per_scene_AUROC_min": float(scene_df["scene_AUROC"].min()),
        "per_scene_AUROC_max": float(scene_df["scene_AUROC"].max()),
        "n_test_scenes": int(scene_df["scene"].nunique()),
        "diagnostic_only_no_ladder_tuning": True,
    }
    save_json(FORENSICS_DIR / "ladder_forensics_summary.json", summary)

    shift_table = compute_shift_table(df, X)
    centroid_table = compute_centroid_geometry(df, X)

    qc = write_qc_manifest(df, n_scenes=20)
    make_montage(qc)

    print("\n=== LADDER FORENSICS SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nClosest mean-shift directions to ladder:")
    print(
        shift_table[
            ["trigger", "cosine_to_ladder_mean_shift", "mean_pair_delta_l2"]
        ].head(6).to_string(index=False)
    )
    print("\nClosest centroids to ladder:")
    print(
        centroid_table[
            ["reference", "cosine_to_ladder_centroid", "euclidean_to_ladder_centroid"]
        ].head(6).to_string(index=False)
    )
    print(f"\nOutputs: {FORENSICS_DIR}")


if __name__ == "__main__":
    main()
