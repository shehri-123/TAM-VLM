from pathlib import Path
import json
import numpy as np
import pandas as pd

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SCORES_CSV = (
    ROOT / "results/frozen_detector_scoring_v1/"
    "nuimages_external_frozen_detector_scores_v1.csv"
)

PAIRS_CSV = (
    ROOT / "metadata/"
    "nuimages_external_same_log_matched_pairs_v1.csv"
)

OUT = ROOT / "results/real_capture_transfer_v1"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
BOOT_SEED = 260811

CATEGORIES = {
    "traffic_cone": {
        "has_col": "has_traffic_cone",
        "area_col": "traffic_cone_max_bbox_area_fraction",
    },
    "traffic_barrier": {
        "has_col": "has_traffic_barrier",
        "area_col": "traffic_barrier_max_bbox_area_fraction",
    },
    "debris": {
        "has_col": "has_debris",
        "area_col": "debris_max_bbox_area_fraction",
    },
}

BACKBONES = {
    "OpenCLIP": "openclip",
    "Qwen3-VL": "qwen3vl",
}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = (
        z * np.sqrt((p*(1-p)/n) + z*z/(4*n*n))
        / den
    )
    return center-half, center+half


def exact_mcnemar(a, b):
    """
    a = target alerts
    b = matched-control alerts
    exact two-sided McNemar via binomial test.
    """
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    discordant = n10 + n01

    if discordant == 0:
        return n10, n01, 1.0

    p = binomtest(
        min(n10, n01),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    ).pvalue

    return n10, n01, float(p)


def cluster_bootstrap_mean_ci(values, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Cluster bootstrap of the overall mean.
    Whole nuImages logs are resampled with replacement.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)

    ok = np.isfinite(values)
    values = values[ok]
    clusters = clusters[ok]

    if len(values) == 0:
        return np.nan, np.nan

    tmp = pd.DataFrame({
        "cluster": clusters,
        "value": values,
    })

    agg = (
        tmp.groupby("cluster", sort=False)["value"]
        .agg(["count", "sum"])
        .reset_index(drop=True)
    )

    counts = agg["count"].to_numpy(float)
    sums = agg["sum"].to_numpy(float)

    m = len(agg)

    if m <= 1:
        x = float(np.mean(values))
        return x, x

    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=float)

    for r in range(n_boot):
        idx = rng.integers(0, m, size=m)
        denom = counts[idx].sum()
        out[r] = sums[idx].sum() / denom

    return (
        float(np.quantile(out, 0.025)),
        float(np.quantile(out, 0.975)),
    )


def cluster_bootstrap_slope_ci(x, y, clusters, n_boot=N_BOOT, seed=BOOT_SEED):
    """
    Cluster bootstrap OLS slope y ~ log10(area).
    Uses sufficient statistics per log.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    clusters = np.asarray(clusters)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    clusters = clusters[ok]

    if len(x) < 3:
        return np.nan, np.nan, np.nan

    df = pd.DataFrame({
        "cluster": clusters,
        "x": x,
        "y": y,
    })

    df["xx"] = df["x"] * df["x"]
    df["xy"] = df["x"] * df["y"]

    agg = (
        df.groupby("cluster", sort=False)
        .agg(
            n=("x", "size"),
            sx=("x", "sum"),
            sy=("y", "sum"),
            sxx=("xx", "sum"),
            sxy=("xy", "sum"),
        )
        .reset_index(drop=True)
    )

    def slope_from_sums(n, sx, sy, sxx, sxy):
        denom = sxx - sx*sx/n
        if denom <= 0:
            return np.nan
        return (sxy - sx*sy/n) / denom

    n = float(len(df))
    slope = slope_from_sums(
        n,
        df["x"].sum(),
        df["y"].sum(),
        df["xx"].sum(),
        df["xy"].sum(),
    )

    m = len(agg)

    if m <= 1:
        return float(slope), float(slope), float(slope)

    arr = agg.to_numpy(float)

    rng = np.random.default_rng(seed)
    boots = []

    for r in range(n_boot):
        idx = rng.integers(0, m, size=m)
        z = arr[idx]

        bn = z[:, 0].sum()
        bsx = z[:, 1].sum()
        bsy = z[:, 2].sum()
        bsxx = z[:, 3].sum()
        bsxy = z[:, 4].sum()

        s = slope_from_sums(
            bn, bsx, bsy, bsxx, bsxy
        )

        if np.isfinite(s):
            boots.append(s)

    if not boots:
        return float(slope), np.nan, np.nan

    boots = np.asarray(boots)

    return (
        float(slope),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


print("Loading frozen scores...")
scores = pd.read_csv(SCORES_CSV)

print("Loading frozen same-log pairs...")
pairs = pd.read_csv(PAIRS_CSV)

# ---------------------------------------------------------
# Frozen-data sanity checks
# ---------------------------------------------------------
assert len(scores) == 16436, len(scores)
assert len(pairs) == 5796, len(pairs)
assert scores["sample_data_token"].is_unique

assert int(scores["contains_any_audited_target"].sum()) == 6849
assert int(scores["is_target_category_absent_control"].sum()) == 9587

assert (scores["source_frame_independent_from_tamvlm"] == 1).all()
assert (scores["source_log_independent_from_tamvlm"] == 1).all()

if "matching_uses_model_scores" in pairs.columns:
    assert (pairs["matching_uses_model_scores"] == 0).all()

# ---------------------------------------------------------
# Join frozen target/control rows to frozen matched pairs
# ---------------------------------------------------------
keep_cols = [
    "sample_data_token",
    "target_category_set",
    "target_category_count",
    "is_exclusive_single_category",
    "exclusive_category",
    "has_traffic_cone",
    "traffic_cone_max_bbox_area_fraction",
    "has_traffic_barrier",
    "traffic_barrier_max_bbox_area_fraction",
    "has_debris",
    "debris_max_bbox_area_fraction",
]

for bb_prefix in BACKBONES.values():
    for seed in range(5):
        keep_cols.extend([
            f"{bb_prefix}_seed{seed}_score",
            f"{bb_prefix}_seed{seed}_alert",
        ])

keep_cols = list(dict.fromkeys(keep_cols))
sub = scores[keep_cols].copy()

target = sub.add_prefix("target_")
control = sub.add_prefix("control_")

m = pairs.copy()

m = m.merge(
    target,
    left_on="target_sample_data_token",
    right_on="target_sample_data_token",
    how="left",
    validate="one_to_one",
)

m = m.merge(
    control,
    left_on="control_sample_data_token",
    right_on="control_sample_data_token",
    how="left",
    validate="one_to_one",
)

assert len(m) == 5796

# Resolve metadata columns duplicated by the target-row merge.
# The left-hand (_x) values come from the pre-frozen matched-pair manifest;
# the right-hand (_y) values come from the frozen scoring manifest.
# They describe the same target frame. Prefer the frozen pair-manifest copy.
_COLLISION_COLS = [
    "target_has_traffic_cone",
    "target_has_traffic_barrier",
    "target_has_debris",
    "target_is_exclusive_single_category",
    "target_exclusive_category",
]

for col in _COLLISION_COLS:
    if col not in m.columns:
        left = f"{col}_x"
        right = f"{col}_y"

        if left in m.columns:
            m[col] = m[left]
        elif right in m.columns:
            m[col] = m[right]
        else:
            raise KeyError(
                f"Required target metadata column not found: {col}"
            )

# Where both copies exist, verify that the merge did not join inconsistent
# target metadata.
for col in _COLLISION_COLS:
    left = f"{col}_x"
    right = f"{col}_y"

    if left in m.columns and right in m.columns:
        a = m[left].fillna("__NA__").astype(str)
        b = m[right].fillna("__NA__").astype(str)

        if not a.equals(b):
            bad = int((a != b).sum())
            raise AssertionError(
                f"Frozen metadata disagreement for {col}: {bad} rows"
            )

assert m["target_target_category_set"].notna().all()

for bb_prefix in BACKBONES.values():
    for seed in range(5):
        assert m[f"target_{bb_prefix}_seed{seed}_score"].notna().all()
        assert m[f"control_{bb_prefix}_seed{seed}_score"].notna().all()

# Save joined analysis manifest.
joined_out = OUT / "real_capture_matched_analysis_manifest_v1.csv"
m.to_csv(joined_out, index=False)

print("Joined matched manifest:", joined_out)
print("Rows:", len(m))

# ---------------------------------------------------------
# Category-level matched analysis
# ---------------------------------------------------------
category_rows = []

for scope in ["all_category_present", "exclusive_single_category"]:

    for category, spec in CATEGORIES.items():

        has_col = f"target_{spec['has_col']}"

        mask = m[has_col].eq(1)

        if scope == "exclusive_single_category":
            mask &= (
                m["target_is_exclusive_single_category"].eq(1)
                & m["target_exclusive_category"].eq(category)
            )

        d = m.loc[mask].copy()

        print(
            f"{scope:28s} {category:16s} "
            f"n={len(d):5d} logs={d['log_token'].nunique():4d}"
        )

        for backbone, prefix in BACKBONES.items():

            for seed in range(5):

                ta = d[f"target_{prefix}_seed{seed}_alert"].to_numpy(int)
                ca = d[f"control_{prefix}_seed{seed}_alert"].to_numpy(int)

                ts = d[f"target_{prefix}_seed{seed}_score"].to_numpy(float)
                cs = d[f"control_{prefix}_seed{seed}_score"].to_numpy(float)

                alert_delta = ta.astype(float) - ca.astype(float)
                score_delta = ts - cs

                tlo, thi = wilson(int(ta.sum()), len(ta))
                clo, chi = wilson(int(ca.sum()), len(ca))

                ad_lo, ad_hi = cluster_bootstrap_mean_ci(
                    alert_delta,
                    d["log_token"].to_numpy(),
                    seed=BOOT_SEED + seed,
                )

                sd_lo, sd_hi = cluster_bootstrap_mean_ci(
                    score_delta,
                    d["log_token"].to_numpy(),
                    seed=BOOT_SEED + 100 + seed,
                )

                n10, n01, p = exact_mcnemar(ta, ca)

                category_rows.append({
                    "scope": scope,
                    "category": category,
                    "backbone": backbone,
                    "seed": seed,
                    "n_pairs": len(d),
                    "n_logs": d["log_token"].nunique(),

                    "target_alert_rate": ta.mean(),
                    "target_alert_wilson_low": tlo,
                    "target_alert_wilson_high": thi,

                    "control_alert_rate": ca.mean(),
                    "control_alert_wilson_low": clo,
                    "control_alert_wilson_high": chi,

                    "paired_alert_difference": alert_delta.mean(),
                    "paired_alert_difference_cluster_ci_low": ad_lo,
                    "paired_alert_difference_cluster_ci_high": ad_hi,

                    "target_mean_score": ts.mean(),
                    "control_mean_score": cs.mean(),

                    "paired_mean_score_difference": score_delta.mean(),
                    "paired_score_difference_cluster_ci_low": sd_lo,
                    "paired_score_difference_cluster_ci_high": sd_hi,

                    "mcnemar_target1_control0": n10,
                    "mcnemar_target0_control1": n01,
                    "mcnemar_exact_p": p,
                })

category_seed = pd.DataFrame(category_rows)

category_seed.to_csv(
    OUT / "category_transfer_seedwise_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# Holm adjustment by scope/backbone/seed across categories
# ---------------------------------------------------------
category_seed["mcnemar_holm_p"] = np.nan

for keys, idx in category_seed.groupby(
    ["scope", "backbone", "seed"]
).groups.items():

    idx = list(idx)
    pvals = category_seed.loc[idx, "mcnemar_exact_p"].to_numpy(float)

    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)

    running = 0.0

    for rank, pos in enumerate(order):
        val = (len(pvals) - rank) * pvals[pos]
        running = max(running, val)
        adjusted[pos] = min(running, 1.0)

    category_seed.loc[idx, "mcnemar_holm_p"] = adjusted

category_seed.to_csv(
    OUT / "category_transfer_seedwise_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# Across-head descriptive summaries
# ---------------------------------------------------------
summary_rows = []

metric_cols = [
    "target_alert_rate",
    "control_alert_rate",
    "paired_alert_difference",
    "target_mean_score",
    "control_mean_score",
    "paired_mean_score_difference",
]

for (scope, category, backbone), d in category_seed.groupby(
    ["scope", "category", "backbone"],
    sort=False,
):

    row = {
        "scope": scope,
        "category": category,
        "backbone": backbone,
        "n_pairs": int(d["n_pairs"].iloc[0]),
        "n_logs": int(d["n_logs"].iloc[0]),
        "n_heads": len(d),
    }

    for col in metric_cols:
        vals = d[col].to_numpy(float)
        row[f"{col}_mean"] = vals.mean()
        row[f"{col}_sd_ddof0"] = vals.std(ddof=0)

    row["all_seed_cluster_cis_exclude_zero_alert_difference"] = bool(
        (
            (d["paired_alert_difference_cluster_ci_low"] > 0)
            | (d["paired_alert_difference_cluster_ci_high"] < 0)
        ).all()
    )

    row["all_holm_p_lt_0_05"] = bool(
        (d["mcnemar_holm_p"] < 0.05).all()
    )

    summary_rows.append(row)

category_summary = pd.DataFrame(summary_rows)

category_summary.to_csv(
    OUT / "category_transfer_backbone_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# Size tertiles + paired size-transfer analysis
# ---------------------------------------------------------
size_rows = []
trend_rows = []
size_cut_rows = []

for category, spec in CATEGORIES.items():

    has_col = f"target_{spec['has_col']}"
    area_col = f"target_{spec['area_col']}"

    d = m.loc[
        m[has_col].eq(1)
        & m[area_col].gt(0)
    ].copy()

    area = d[area_col].to_numpy(float)

    q1, q2 = np.quantile(area, [1/3, 2/3])

    size_cut_rows.append({
        "category": category,
        "n_pairs": len(d),
        "area_fraction_33pct": q1,
        "area_fraction_67pct": q2,
        "minimum_area_fraction": np.min(area),
        "maximum_area_fraction": np.max(area),
    })

    d["size_group"] = np.where(
        d[area_col] <= q1,
        "small",
        np.where(
            d[area_col] <= q2,
            "medium",
            "large",
        )
    )

    # Tertile analysis
    for size_group in ["small", "medium", "large"]:

        g = d[d["size_group"] == size_group].copy()

        for backbone, prefix in BACKBONES.items():

            for seed in range(5):

                ta = g[f"target_{prefix}_seed{seed}_alert"].to_numpy(int)
                ca = g[f"control_{prefix}_seed{seed}_alert"].to_numpy(int)

                ts = g[f"target_{prefix}_seed{seed}_score"].to_numpy(float)
                cs = g[f"control_{prefix}_seed{seed}_score"].to_numpy(float)

                alert_delta = ta.astype(float) - ca.astype(float)
                score_delta = ts - cs

                ad_lo, ad_hi = cluster_bootstrap_mean_ci(
                    alert_delta,
                    g["log_token"].to_numpy(),
                    seed=BOOT_SEED + 200 + seed,
                )

                sd_lo, sd_hi = cluster_bootstrap_mean_ci(
                    score_delta,
                    g["log_token"].to_numpy(),
                    seed=BOOT_SEED + 300 + seed,
                )

                size_rows.append({
                    "category": category,
                    "size_group": size_group,
                    "backbone": backbone,
                    "seed": seed,
                    "n_pairs": len(g),
                    "n_logs": g["log_token"].nunique(),
                    "area_fraction_min": g[area_col].min(),
                    "area_fraction_median": g[area_col].median(),
                    "area_fraction_max": g[area_col].max(),

                    "target_alert_rate": ta.mean(),
                    "control_alert_rate": ca.mean(),
                    "paired_alert_difference": alert_delta.mean(),
                    "paired_alert_difference_cluster_ci_low": ad_lo,
                    "paired_alert_difference_cluster_ci_high": ad_hi,

                    "target_mean_score": ts.mean(),
                    "control_mean_score": cs.mean(),
                    "paired_mean_score_difference": score_delta.mean(),
                    "paired_score_difference_cluster_ci_low": sd_lo,
                    "paired_score_difference_cluster_ci_high": sd_hi,
                })

    # Continuous log-area trend.
    log_area = np.log10(d[area_col].to_numpy(float))

    for backbone, prefix in BACKBONES.items():

        for seed in range(5):

            ts = d[f"target_{prefix}_seed{seed}_score"].to_numpy(float)
            cs = d[f"control_{prefix}_seed{seed}_score"].to_numpy(float)

            score_delta = ts - cs

            slope, slo, shi = cluster_bootstrap_slope_ci(
                log_area,
                score_delta,
                d["log_token"].to_numpy(),
                seed=BOOT_SEED + 400 + seed,
            )

            trend_rows.append({
                "category": category,
                "backbone": backbone,
                "seed": seed,
                "n_pairs": len(d),
                "n_logs": d["log_token"].nunique(),
                "predictor": "log10(category_specific_max_bbox_area_fraction)",
                "outcome": "paired_target_minus_control_detector_score",
                "ols_slope": slope,
                "cluster_ci_low": slo,
                "cluster_ci_high": shi,
                "ci_excludes_zero": bool(
                    np.isfinite(slo)
                    and np.isfinite(shi)
                    and ((slo > 0) or (shi < 0))
                ),
            })

size_seed = pd.DataFrame(size_rows)
trend_seed = pd.DataFrame(trend_rows)
size_cuts = pd.DataFrame(size_cut_rows)

size_seed.to_csv(
    OUT / "size_transfer_seedwise_v1.csv",
    index=False,
)

trend_seed.to_csv(
    OUT / "size_trend_seedwise_v1.csv",
    index=False,
)

size_cuts.to_csv(
    OUT / "size_tertile_boundaries_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# Size summaries across five frozen heads
# ---------------------------------------------------------
size_summary_rows = []

for (cat, group, backbone), d in size_seed.groupby(
    ["category", "size_group", "backbone"],
    sort=False,
):

    size_summary_rows.append({
        "category": cat,
        "size_group": group,
        "backbone": backbone,
        "n_pairs": int(d["n_pairs"].iloc[0]),
        "n_logs": int(d["n_logs"].iloc[0]),

        "target_alert_rate_mean": d["target_alert_rate"].mean(),
        "target_alert_rate_sd_ddof0": d["target_alert_rate"].std(ddof=0),

        "control_alert_rate_mean": d["control_alert_rate"].mean(),
        "control_alert_rate_sd_ddof0": d["control_alert_rate"].std(ddof=0),

        "paired_alert_difference_mean": d["paired_alert_difference"].mean(),
        "paired_alert_difference_sd_ddof0": d["paired_alert_difference"].std(ddof=0),

        "paired_score_difference_mean": d["paired_mean_score_difference"].mean(),
        "paired_score_difference_sd_ddof0": d["paired_mean_score_difference"].std(ddof=0),
    })

pd.DataFrame(size_summary_rows).to_csv(
    OUT / "size_transfer_backbone_summary_v1.csv",
    index=False,
)

trend_summary_rows = []

for (cat, backbone), d in trend_seed.groupby(
    ["category", "backbone"],
    sort=False,
):

    trend_summary_rows.append({
        "category": cat,
        "backbone": backbone,
        "n_pairs": int(d["n_pairs"].iloc[0]),
        "n_logs": int(d["n_logs"].iloc[0]),
        "mean_slope_across_heads": d["ols_slope"].mean(),
        "sd_slope_across_heads_ddof0": d["ols_slope"].std(ddof=0),
        "positive_slope_heads": int((d["ols_slope"] > 0).sum()),
        "heads_with_ci_excluding_zero": int(d["ci_excludes_zero"].sum()),
    })

pd.DataFrame(trend_summary_rows).to_csv(
    OUT / "size_trend_backbone_summary_v1.csv",
    index=False,
)

# ---------------------------------------------------------
# Frozen status report
# ---------------------------------------------------------
report = {
    "status": "PASS",
    "scores_rows": int(len(scores)),
    "target_rows": int(scores["contains_any_audited_target"].sum()),
    "control_rows": int(scores["is_target_category_absent_control"].sum()),
    "matched_pairs": int(len(m)),
    "matched_logs": int(m["log_token"].nunique()),
    "categories": list(CATEGORIES.keys()),
    "backbones": list(BACKBONES.keys()),
    "heads_per_backbone": 5,
    "bootstrap_replicates": N_BOOT,
    "bootstrap_unit": "nuImages log",
    "training_performed": False,
    "threshold_recalibration_performed": False,
    "model_selection_on_nuimages": False,
    "claim_boundary": (
        "Synthetic-to-real object-response transfer only; "
        "not physical-attack TPR, ASR, or victim-model mitigation."
    ),
}

with open(
    OUT / "REAL_CAPTURE_TRANSFER_STATUS_v1.json",
    "w",
) as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("REAL-CAPTURE TRANSFER ANALYSIS COMPLETE")
print("="*80)

print("\nCATEGORY SUMMARY")
print(
    category_summary[
        [
            "scope",
            "category",
            "backbone",
            "n_pairs",
            "target_alert_rate_mean",
            "control_alert_rate_mean",
            "paired_alert_difference_mean",
        ]
    ].to_string(index=False)
)

print("\nSIZE TREND SUMMARY")
trend_summary = pd.read_csv(
    OUT / "size_trend_backbone_summary_v1.csv"
)
print(trend_summary.to_string(index=False))

print("\nSTATUS: PASS")
print("Output:", OUT)
