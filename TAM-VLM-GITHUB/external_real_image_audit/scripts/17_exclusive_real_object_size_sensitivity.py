from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "results/real_capture_transfer_v1/real_capture_matched_analysis_manifest_v1.csv"
OUT = ROOT / "results/real_capture_transfer_v1/exclusive_size_sensitivity_v1"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
SEED0 = 260812

CATEGORIES = {
    "traffic_cone": "target_traffic_cone_max_bbox_area_fraction",
    "traffic_barrier": "target_traffic_barrier_max_bbox_area_fraction",
    "debris": "target_debris_max_bbox_area_fraction",
}

BACKBONES = {
    "OpenCLIP": "openclip",
    "Qwen3-VL": "qwen3vl",
}


def cluster_boot_mean_ci(values, clusters, seed):
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)

    ok = np.isfinite(values)
    values = values[ok]
    clusters = clusters[ok]

    df = pd.DataFrame({"cluster": clusters, "value": values})
    agg = df.groupby("cluster")["value"].agg(["count", "sum"])

    counts = agg["count"].to_numpy(float)
    sums = agg["sum"].to_numpy(float)
    n_clusters = len(agg)

    if n_clusters <= 1:
        x = float(np.mean(values))
        return x, x

    rng = np.random.default_rng(seed)
    boots = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx = rng.integers(0, n_clusters, n_clusters)
        boots[b] = sums[idx].sum() / counts[idx].sum()

    return (
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def slope_from_arrays(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x0 = x - x.mean()
    den = np.sum(x0 * x0)

    if den <= 0:
        return np.nan

    return float(np.sum(x0 * (y - y.mean())) / den)


def cluster_boot_slope_ci(x, y, clusters, seed):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    clusters = np.asarray(clusters)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    clusters = clusters[ok]

    slope = slope_from_arrays(x, y)

    unique = np.unique(clusters)

    if len(unique) <= 1:
        return slope, slope, slope

    groups = {
        c: np.where(clusters == c)[0]
        for c in unique
    }

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(N_BOOT):
        sampled = rng.choice(unique, size=len(unique), replace=True)

        xb = []
        yb = []

        for c in sampled:
            idx = groups[c]
            xb.append(x[idx])
            yb.append(y[idx])

        xb = np.concatenate(xb)
        yb = np.concatenate(yb)

        s = slope_from_arrays(xb, yb)

        if np.isfinite(s):
            boots.append(s)

    boots = np.asarray(boots)

    return (
        slope,
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


print("Loading:", INFILE)
df = pd.read_csv(INFILE)

assert len(df) == 5796

rows = []
trend_rows = []
boundary_rows = []

for category, area_col in CATEGORIES.items():

    d = df[
        (df["target_is_exclusive_single_category"] == 1)
        & (df["target_exclusive_category"] == category)
        & (df[area_col] > 0)
    ].copy()

    print(
        f"\n{category}: n={len(d)}, "
        f"logs={d['log_token'].nunique()}"
    )

    q1, q2 = np.quantile(
        d[area_col].to_numpy(float),
        [1/3, 2/3]
    )

    boundary_rows.append({
        "category": category,
        "n_pairs": len(d),
        "n_logs": d["log_token"].nunique(),
        "q33": q1,
        "q67": q2,
        "min_area": d[area_col].min(),
        "max_area": d[area_col].max(),
        "interpretation":
            "exploratory_small_subset"
            if category == "debris"
            else "primary_sensitivity"
    })

    d["size_group"] = np.where(
        d[area_col] <= q1,
        "small",
        np.where(
            d[area_col] <= q2,
            "medium",
            "large"
        )
    )

    for size_group in ["small", "medium", "large"]:

        g = d[d["size_group"] == size_group]

        for backbone, prefix in BACKBONES.items():

            for seed in range(5):

                ts = g[f"target_{prefix}_seed{seed}_score"].to_numpy(float)
                cs = g[f"control_{prefix}_seed{seed}_score"].to_numpy(float)

                ta = g[f"target_{prefix}_seed{seed}_alert"].to_numpy(int)
                ca = g[f"control_{prefix}_seed{seed}_alert"].to_numpy(int)

                score_diff = ts - cs
                alert_diff = ta.astype(float) - ca.astype(float)

                slo, shi = cluster_boot_mean_ci(
                    score_diff,
                    g["log_token"],
                    SEED0 + seed
                )

                alo, ahi = cluster_boot_mean_ci(
                    alert_diff,
                    g["log_token"],
                    SEED0 + 100 + seed
                )

                rows.append({
                    "category": category,
                    "size_group": size_group,
                    "backbone": backbone,
                    "seed": seed,
                    "n_pairs": len(g),
                    "n_logs": g["log_token"].nunique(),
                    "median_bbox_area_fraction": g[area_col].median(),

                    "target_mean_score": ts.mean(),
                    "control_mean_score": cs.mean(),
                    "paired_score_difference": score_diff.mean(),
                    "paired_score_ci_low": slo,
                    "paired_score_ci_high": shi,

                    "target_alert_rate": ta.mean(),
                    "control_alert_rate": ca.mean(),
                    "paired_alert_difference": alert_diff.mean(),
                    "paired_alert_ci_low": alo,
                    "paired_alert_ci_high": ahi,
                })

    log_area = np.log10(d[area_col].to_numpy(float))

    for backbone, prefix in BACKBONES.items():

        for seed in range(5):

            ts = d[f"target_{prefix}_seed{seed}_score"].to_numpy(float)
            cs = d[f"control_{prefix}_seed{seed}_score"].to_numpy(float)

            diff = ts - cs

            slope, low, high = cluster_boot_slope_ci(
                log_area,
                diff,
                d["log_token"],
                SEED0 + 200 + seed
            )

            trend_rows.append({
                "category": category,
                "backbone": backbone,
                "seed": seed,
                "n_pairs": len(d),
                "n_logs": d["log_token"].nunique(),
                "slope_log10_area": slope,
                "cluster_ci_low": low,
                "cluster_ci_high": high,
                "positive_slope": bool(slope > 0),
                "ci_excludes_zero": bool(
                    (low > 0) or (high < 0)
                )
            })

seedwise = pd.DataFrame(rows)
trends = pd.DataFrame(trend_rows)
boundaries = pd.DataFrame(boundary_rows)

seedwise.to_csv(
    OUT / "exclusive_size_seedwise_v1.csv",
    index=False
)

trends.to_csv(
    OUT / "exclusive_size_trend_seedwise_v1.csv",
    index=False
)

boundaries.to_csv(
    OUT / "exclusive_size_tertile_boundaries_v1.csv",
    index=False
)

summary = (
    seedwise.groupby(
        ["category", "size_group", "backbone"],
        sort=False
    )
    .agg(
        n_pairs=("n_pairs", "first"),
        n_logs=("n_logs", "first"),
        target_alert_rate_mean=("target_alert_rate", "mean"),
        control_alert_rate_mean=("control_alert_rate", "mean"),
        paired_alert_difference_mean=("paired_alert_difference", "mean"),
        paired_score_difference_mean=("paired_score_difference", "mean"),
        paired_score_difference_sd_ddof0=(
            "paired_score_difference",
            lambda x: np.std(x, ddof=0)
        )
    )
    .reset_index()
)

summary.to_csv(
    OUT / "exclusive_size_backbone_summary_v1.csv",
    index=False
)

trend_summary = (
    trends.groupby(
        ["category", "backbone"],
        sort=False
    )
    .agg(
        n_pairs=("n_pairs", "first"),
        n_logs=("n_logs", "first"),
        mean_slope=("slope_log10_area", "mean"),
        sd_slope_ddof0=(
            "slope_log10_area",
            lambda x: np.std(x, ddof=0)
        ),
        positive_slope_heads=("positive_slope", "sum"),
        heads_ci_excluding_zero=("ci_excludes_zero", "sum")
    )
    .reset_index()
)

trend_summary.to_csv(
    OUT / "exclusive_size_trend_backbone_summary_v1.csv",
    index=False
)

status = {
    "status": "PASS",
    "analysis_type": "post-hoc sensitivity analysis",
    "matched_pair_source_rows": len(df),
    "bootstrap_replicates": N_BOOT,
    "bootstrap_unit": "nuImages log",
    "retraining": False,
    "recalibration": False,
    "categories": {
        c: int(
            (
                (df["target_is_exclusive_single_category"] == 1)
                & (df["target_exclusive_category"] == c)
            ).sum()
        )
        for c in CATEGORIES
    }
}

with open(
    OUT / "EXCLUSIVE_SIZE_SENSITIVITY_STATUS_v1.json",
    "w"
) as f:
    json.dump(status, f, indent=2)

print("\n===== SIZE SUMMARY =====")
print(summary.to_string(index=False))

print("\n===== CONTINUOUS TREND SUMMARY =====")
print(trend_summary.to_string(index=False))

print("\nSTATUS: PASS")
