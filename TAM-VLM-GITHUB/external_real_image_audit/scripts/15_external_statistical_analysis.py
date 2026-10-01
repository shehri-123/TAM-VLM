#!/usr/bin/env python3

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(
    os.environ.get(
        "TAMVLM_EXTERNAL_ROOT",
        Path(__file__).resolve().parents[1]
    )
)
SCORES_PATH = (
    ROOT / "results/frozen_detector_scoring_v1/"
    "nuimages_external_frozen_detector_scores_v1.csv"
)

SCORING_METADATA_PATH = (
    ROOT / "results/frozen_detector_scoring_v1/"
    "nuimages_external_frozen_detector_scoring_metadata_v1.json"
)

PAIRS_PATH = (
    ROOT / "metadata/"
    "nuimages_external_same_log_matched_pairs_v1.csv"
)

ARTIFACT_MANIFEST_PATH = (
    ROOT / "protocol/"
    "FROZEN_DETECTOR_ARTIFACT_MANIFEST_v1.csv"
)

OUTPUT_DIR = ROOT / "results/statistical_analysis_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_SEEDWISE_PATH = (
    OUTPUT_DIR / "full_pool_seedwise_rates_v1.csv"
)

FULL_BACKBONE_PATH = (
    OUTPUT_DIR / "full_pool_backbone_summary_v1.csv"
)

MATCHED_SEEDWISE_PATH = (
    OUTPUT_DIR / "matched_pair_seedwise_analysis_v1.csv"
)

MATCHED_BACKBONE_PATH = (
    OUTPUT_DIR / "matched_pair_backbone_summary_v1.csv"
)

CATEGORY_SEEDWISE_PATH = (
    OUTPUT_DIR / "category_seedwise_analysis_v1.csv"
)

CATEGORY_BACKBONE_PATH = (
    OUTPUT_DIR / "category_backbone_summary_v1.csv"
)

SPLIT_SEEDWISE_PATH = (
    OUTPUT_DIR / "split_seedwise_analysis_v1.csv"
)

SPLIT_BACKBONE_PATH = (
    OUTPUT_DIR / "split_backbone_summary_v1.csv"
)

METADATA_PATH = (
    OUTPUT_DIR / "external_statistical_analysis_metadata_v1.json"
)

REPORT_PATH = (
    OUTPUT_DIR / "external_statistical_analysis_report_v1.txt"
)

EXPECTED_ROWS = 16436
EXPECTED_TARGET_ROWS = 6849
EXPECTED_CONTROL_ROWS = 9587
EXPECTED_PAIRS = 5796
EXPECTED_MATCHED_TARGET_COVERAGE = 5796 / 6849

BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_BASE_SEED = 20260805
Z_95 = 1.959963984540054

HEADS = [
    {
        "backbone": "OpenCLIP",
        "slug": "openclip",
        "seed": seed,
    }
    for seed in range(5)
] + [
    {
        "backbone": "Qwen3-VL-Embedding-2B",
        "slug": "qwen3vl",
        "seed": seed,
    }
    for seed in range(5)
]

CATEGORIES = {
    "traffic_cone": "has_traffic_cone",
    "traffic_barrier": "has_traffic_barrier",
    "debris": "has_debris",
}


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


def wilson_interval(successes, total, z=Z_95):
    successes = int(successes)
    total = int(total)

    if total == 0:
        return float("nan"), float("nan")

    proportion = successes / total
    denominator = 1.0 + (z * z / total)

    centre = (
        proportion +
        z * z / (2.0 * total)
    ) / denominator

    half_width = (
        z *
        math.sqrt(
            proportion * (1.0 - proportion) / total +
            z * z / (4.0 * total * total)
        ) /
        denominator
    )

    return (
        max(0.0, centre - half_width),
        min(1.0, centre + half_width),
    )


def quantile_interval(values):
    values = np.asarray(values, dtype=np.float64)

    try:
        lower, upper = np.quantile(
            values,
            [0.025, 0.975],
            method="linear",
        )
    except TypeError:
        lower, upper = np.quantile(
            values,
            [0.025, 0.975],
            interpolation="linear",
        )

    return float(lower), float(upper)


def cluster_bootstrap_matched(
    log_tokens,
    target_alerts,
    control_alerts,
    target_scores,
    control_scores,
    *,
    replicates,
    random_seed,
):
    frame = pd.DataFrame({
        "log_token": np.asarray(log_tokens).astype(str),
        "target_alert": np.asarray(target_alerts, dtype=np.float64),
        "control_alert": np.asarray(control_alerts, dtype=np.float64),
        "score_difference": (
            np.asarray(target_scores, dtype=np.float64) -
            np.asarray(control_scores, dtype=np.float64)
        ),
    })

    frame["n"] = 1.0

    grouped = (
        frame.groupby("log_token", sort=True)
        .agg(
            n=("n", "sum"),
            target_alert_sum=("target_alert", "sum"),
            control_alert_sum=("control_alert", "sum"),
            score_difference_sum=("score_difference", "sum"),
        )
        .reset_index()
    )

    n_values = grouped["n"].to_numpy(dtype=np.float64)

    target_sums = grouped[
        "target_alert_sum"
    ].to_numpy(dtype=np.float64)

    control_sums = grouped[
        "control_alert_sum"
    ].to_numpy(dtype=np.float64)

    score_difference_sums = grouped[
        "score_difference_sum"
    ].to_numpy(dtype=np.float64)

    number_of_logs = len(grouped)
    rng = np.random.default_rng(random_seed)

    target_bootstrap = np.empty(
        replicates,
        dtype=np.float64,
    )

    control_bootstrap = np.empty(
        replicates,
        dtype=np.float64,
    )

    alert_difference_bootstrap = np.empty(
        replicates,
        dtype=np.float64,
    )

    score_difference_bootstrap = np.empty(
        replicates,
        dtype=np.float64,
    )

    chunk_size = 250

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)

        sampled = rng.integers(
            0,
            number_of_logs,
            size=(stop - start, number_of_logs),
        )

        denominators = n_values[sampled].sum(axis=1)

        target_rates = (
            target_sums[sampled].sum(axis=1) /
            denominators
        )

        control_rates = (
            control_sums[sampled].sum(axis=1) /
            denominators
        )

        score_differences = (
            score_difference_sums[sampled].sum(axis=1) /
            denominators
        )

        target_bootstrap[start:stop] = target_rates
        control_bootstrap[start:stop] = control_rates

        alert_difference_bootstrap[start:stop] = (
            target_rates - control_rates
        )

        score_difference_bootstrap[start:stop] = (
            score_differences
        )

    target_ci = quantile_interval(target_bootstrap)
    control_ci = quantile_interval(control_bootstrap)

    alert_difference_ci = quantile_interval(
        alert_difference_bootstrap
    )

    score_difference_ci = quantile_interval(
        score_difference_bootstrap
    )

    return {
        "n_logs": int(number_of_logs),
        "target_rate_cluster_ci_low": target_ci[0],
        "target_rate_cluster_ci_high": target_ci[1],
        "control_rate_cluster_ci_low": control_ci[0],
        "control_rate_cluster_ci_high": control_ci[1],
        "alert_difference_cluster_ci_low": (
            alert_difference_ci[0]
        ),
        "alert_difference_cluster_ci_high": (
            alert_difference_ci[1]
        ),
        "score_difference_cluster_ci_low": (
            score_difference_ci[0]
        ),
        "score_difference_cluster_ci_high": (
            score_difference_ci[1]
        ),
    }


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    number_of_tests = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.empty(
        number_of_tests,
        dtype=np.float64,
    )

    running_maximum = 0.0

    for rank, index in enumerate(order):
        multiplier = number_of_tests - rank

        candidate = min(
            1.0,
            multiplier * p_values[index],
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[index] = running_maximum

    return adjusted


def summarize_seed_rates(
    frame,
    group_columns,
    value_columns,
):
    records = []

    for group_values, group in frame.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        record = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
            )
        }

        record["n_seeds"] = int(len(group))

        for column in value_columns:
            values = group[column].to_numpy(
                dtype=np.float64
            )

            record[f"{column}_mean"] = float(
                values.mean()
            )

            record[f"{column}_std_ddof0"] = float(
                values.std(ddof=0)
            )

            record[f"{column}_minimum"] = float(
                values.min()
            )

            record[f"{column}_maximum"] = float(
                values.max()
            )

        records.append(record)

    return pd.DataFrame(records)


print("=" * 112)
print("TAM-VLM external real-image statistical analysis")
print("=" * 112)

for required_path in [
    SCORES_PATH,
    SCORING_METADATA_PATH,
    PAIRS_PATH,
    ARTIFACT_MANIFEST_PATH,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

scoring_metadata = json.loads(
    SCORING_METADATA_PATH.read_text(
        encoding="utf-8"
    )
)

if scoring_metadata.get("status") != "PASS":
    raise RuntimeError(
        "Frozen detector scoring metadata "
        "does not report PASS."
    )

if (
    sha256_file(SCORES_PATH) !=
    scoring_metadata["scores_csv_sha256"]
):
    raise RuntimeError(
        "Frozen score CSV checksum mismatch."
    )

scores = pd.read_csv(SCORES_PATH)
pairs = pd.read_csv(PAIRS_PATH)
artifacts = pd.read_csv(ARTIFACT_MANIFEST_PATH)

if len(scores) != EXPECTED_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_ROWS} scores, "
        f"found {len(scores)}."
    )

if len(pairs) != EXPECTED_PAIRS:
    raise RuntimeError(
        f"Expected {EXPECTED_PAIRS} matched pairs, "
        f"found {len(pairs)}."
    )

required_score_columns = {
    "audit_row_id",
    "sample_data_token",
    "log_token",
    "dataset_split",
    "contains_any_audited_target",
    "is_target_category_absent_control",
    "is_exclusive_single_category",
    "has_traffic_cone",
    "has_traffic_barrier",
    "has_debris",
}

for head in HEADS:
    required_score_columns.add(
        f"{head['slug']}_seed{head['seed']}_score"
    )

    required_score_columns.add(
        f"{head['slug']}_seed{head['seed']}_alert"
    )

missing_score_columns = (
    required_score_columns -
    set(scores.columns)
)

if missing_score_columns:
    raise RuntimeError(
        f"Missing score columns: "
        f"{sorted(missing_score_columns)}"
    )

required_pair_columns = {
    "pair_id",
    "target_sample_data_token",
    "control_sample_data_token",
}

missing_pair_columns = (
    required_pair_columns -
    set(pairs.columns)
)

if missing_pair_columns:
    raise RuntimeError(
        f"Missing pair columns: "
        f"{sorted(missing_pair_columns)}"
    )

if not scores["audit_row_id"].is_unique:
    raise RuntimeError(
        "audit_row_id is not unique."
    )

scores["sample_data_token"] = (
    scores["sample_data_token"].astype(str)
)

pairs["target_sample_data_token"] = (
    pairs["target_sample_data_token"].astype(str)
)

pairs["control_sample_data_token"] = (
    pairs["control_sample_data_token"].astype(str)
)

if not pairs["pair_id"].is_unique:
    raise RuntimeError("pair_id is not unique.")

if not pairs[
    "target_sample_data_token"
].is_unique:
    raise RuntimeError(
        "Matched target tokens are not unique."
    )

if not pairs[
    "control_sample_data_token"
].is_unique:
    raise RuntimeError(
        "Matched control tokens are not unique."
    )

target_mask = (
    scores["contains_any_audited_target"]
    .astype(int)
    .to_numpy()
    .astype(bool)
)

control_mask = (
    scores["is_target_category_absent_control"]
    .astype(int)
    .to_numpy()
    .astype(bool)
)

if int(target_mask.sum()) != EXPECTED_TARGET_ROWS:
    raise RuntimeError(
        "Unexpected target-containing count."
    )

if int(control_mask.sum()) != EXPECTED_CONTROL_ROWS:
    raise RuntimeError(
        "Unexpected control count."
    )

score_lookup = scores.set_index(
    "sample_data_token",
    drop=False,
)

target_tokens = pairs[
    "target_sample_data_token"
].to_numpy()

control_tokens = pairs[
    "control_sample_data_token"
].to_numpy()

missing_targets = (
    set(target_tokens) -
    set(score_lookup.index)
)

missing_controls = (
    set(control_tokens) -
    set(score_lookup.index)
)

if missing_targets or missing_controls:
    raise RuntimeError(
        "Matched-pair tokens are missing "
        "from frozen scores."
    )

matched_targets = (
    score_lookup.loc[target_tokens]
    .reset_index(drop=True)
)

matched_controls = (
    score_lookup.loc[control_tokens]
    .reset_index(drop=True)
)

if not matched_targets[
    "contains_any_audited_target"
].astype(int).eq(1).all():
    raise RuntimeError(
        "Matched target side contains non-target rows."
    )

if not matched_controls[
    "is_target_category_absent_control"
].astype(int).eq(1).all():
    raise RuntimeError(
        "Matched control side contains non-control rows."
    )

if not np.array_equal(
    matched_targets["log_token"]
    .astype(str)
    .to_numpy(),
    matched_controls["log_token"]
    .astype(str)
    .to_numpy(),
):
    raise RuntimeError(
        "Matched pairs are not from the same log."
    )

if not np.array_equal(
    matched_targets["dataset_split"]
    .astype(str)
    .to_numpy(),
    matched_controls["dataset_split"]
    .astype(str)
    .to_numpy(),
):
    raise RuntimeError(
        "Matched pairs cross dataset splits."
    )

artifacts = artifacts[
    artifacts["included_in_external_audit"]
    .astype(int)
    .eq(1)
].copy()

if len(artifacts) != 10:
    raise RuntimeError(
        "Frozen artifact manifest does not "
        "contain ten included heads."
    )

threshold_lookup = {}

for _, row in artifacts.iterrows():
    backbone = str(row["backbone"])
    seed = int(row["seed"])

    threshold_lookup[(backbone, seed)] = float(
        row["external_audit_threshold_used"]
    )

analysis_start = time.perf_counter()
failures = []

full_pool_rows = []
matched_rows = []
category_rows = []
split_rows = []

all_mask = np.ones(
    len(scores),
    dtype=bool,
)

groups = [
    (
        "full_pool",
        all_mask,
    ),
    (
        "target_containing",
        target_mask,
    ),
    (
        "target_category_absent_control",
        control_mask,
    ),
]

for head_index, head in enumerate(HEADS):
    backbone = head["backbone"]
    slug = head["slug"]
    seed = head["seed"]

    score_column = (
        f"{slug}_seed{seed}_score"
    )

    alert_column = (
        f"{slug}_seed{seed}_alert"
    )

    threshold = threshold_lookup[
        (backbone, seed)
    ]

    score_values = scores[
        score_column
    ].to_numpy(dtype=np.float64)

    alert_values = scores[
        alert_column
    ].astype(int).to_numpy()

    recomputed_alerts = (
        score_values > threshold
    ).astype(int)

    alert_mismatches = int(
        np.count_nonzero(
            alert_values != recomputed_alerts
        )
    )

    if alert_mismatches:
        failures.append(
            f"{backbone} seed {seed}: "
            f"{alert_mismatches} alert mismatches"
        )

    for group_name, mask in groups:
        n_images = int(mask.sum())
        n_alerts = int(alert_values[mask].sum())
        rate = n_alerts / n_images

        ci_low, ci_high = wilson_interval(
            n_alerts,
            n_images,
        )

        full_pool_rows.append({
            "backbone": backbone,
            "seed": seed,
            "threshold": threshold,
            "audit_group": group_name,
            "n_images": n_images,
            "n_alerts": n_alerts,
            "observed_alert_rate": rate,
            "wilson_95_ci_low": ci_low,
            "wilson_95_ci_high": ci_high,
            "mean_score": float(
                score_values[mask].mean()
            ),
            "median_score": float(
                np.median(score_values[mask])
            ),
            "minimum_score": float(
                score_values[mask].min()
            ),
            "maximum_score": float(
                score_values[mask].max()
            ),
        })

    for split_name in [
        "v1.0-train",
        "v1.0-val",
    ]:
        split_mask = (
            scores["dataset_split"]
            .astype(str)
            .to_numpy() == split_name
        )

        for group_name, group_mask in [
            (
                "target_containing",
                target_mask,
            ),
            (
                "target_category_absent_control",
                control_mask,
            ),
        ]:
            mask = split_mask & group_mask
            n_images = int(mask.sum())
            n_alerts = int(
                alert_values[mask].sum()
            )

            rate = (
                n_alerts / n_images
                if n_images
                else float("nan")
            )

            ci_low, ci_high = wilson_interval(
                n_alerts,
                n_images,
            )

            split_rows.append({
                "backbone": backbone,
                "seed": seed,
                "threshold": threshold,
                "dataset_split": split_name,
                "audit_group": group_name,
                "n_images": n_images,
                "n_alerts": n_alerts,
                "observed_alert_rate": rate,
                "wilson_95_ci_low": ci_low,
                "wilson_95_ci_high": ci_high,
                "mean_score": float(
                    score_values[mask].mean()
                ),
                "median_score": float(
                    np.median(
                        score_values[mask]
                    )
                ),
            })

    exclusive_flag = (
        scores[
            "is_exclusive_single_category"
        ]
        .astype(int)
        .to_numpy()
        .astype(bool)
    )

    for category, category_column in CATEGORIES.items():
        category_membership = (
            scores[category_column]
            .astype(int)
            .to_numpy()
            .astype(bool)
        )

        for scope_name, scope_mask in [
            (
                "all_multilabel_membership",
                target_mask & category_membership,
            ),
            (
                "exclusive_only_sensitivity",
                (
                    target_mask &
                    category_membership &
                    exclusive_flag
                ),
            ),
        ]:
            n_images = int(scope_mask.sum())
            n_alerts = int(
                alert_values[scope_mask].sum()
            )

            rate = (
                n_alerts / n_images
                if n_images
                else float("nan")
            )

            ci_low, ci_high = wilson_interval(
                n_alerts,
                n_images,
            )

            category_rows.append({
                "backbone": backbone,
                "seed": seed,
                "threshold": threshold,
                "category": category,
                "membership_scope": scope_name,
                "n_images": n_images,
                "n_alerts": n_alerts,
                "observed_alert_rate": rate,
                "wilson_95_ci_low": ci_low,
                "wilson_95_ci_high": ci_high,
                "mean_score": float(
                    score_values[
                        scope_mask
                    ].mean()
                ),
                "median_score": float(
                    np.median(
                        score_values[
                            scope_mask
                        ]
                    )
                ),
            })

    target_pair_scores = matched_targets[
        score_column
    ].to_numpy(dtype=np.float64)

    control_pair_scores = matched_controls[
        score_column
    ].to_numpy(dtype=np.float64)

    target_pair_alerts = matched_targets[
        alert_column
    ].astype(int).to_numpy()

    control_pair_alerts = matched_controls[
        alert_column
    ].astype(int).to_numpy()

    target_alert_count = int(
        target_pair_alerts.sum()
    )

    control_alert_count = int(
        control_pair_alerts.sum()
    )

    target_rate = (
        target_alert_count / EXPECTED_PAIRS
    )

    control_rate = (
        control_alert_count / EXPECTED_PAIRS
    )

    alert_difference = (
        target_rate - control_rate
    )

    target_ci = wilson_interval(
        target_alert_count,
        EXPECTED_PAIRS,
    )

    control_ci = wilson_interval(
        control_alert_count,
        EXPECTED_PAIRS,
    )

    target_only = int(np.sum(
        (target_pair_alerts == 1) &
        (control_pair_alerts == 0)
    ))

    control_only = int(np.sum(
        (target_pair_alerts == 0) &
        (control_pair_alerts == 1)
    ))

    both_alert = int(np.sum(
        (target_pair_alerts == 1) &
        (control_pair_alerts == 1)
    ))

    neither_alert = int(np.sum(
        (target_pair_alerts == 0) &
        (control_pair_alerts == 0)
    ))

    discordant_total = (
        target_only + control_only
    )

    if discordant_total:
        mcnemar_p = float(
            binomtest(
                target_only,
                n=discordant_total,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        mcnemar_p = 1.0

    bootstrap = cluster_bootstrap_matched(
        matched_targets[
            "log_token"
        ].astype(str).to_numpy(),
        target_pair_alerts,
        control_pair_alerts,
        target_pair_scores,
        control_pair_scores,
        replicates=BOOTSTRAP_REPLICATES,
        random_seed=(
            BOOTSTRAP_BASE_SEED +
            head_index
        ),
    )

    matched_rows.append({
        "backbone": backbone,
        "seed": seed,
        "threshold": threshold,
        "n_pairs": EXPECTED_PAIRS,
        "n_logs": bootstrap["n_logs"],
        "matched_target_coverage": (
            EXPECTED_MATCHED_TARGET_COVERAGE
        ),
        "target_alerts": target_alert_count,
        "target_observed_alert_rate": target_rate,
        "target_wilson_95_ci_low": target_ci[0],
        "target_wilson_95_ci_high": target_ci[1],
        "target_log_cluster_95_ci_low": (
            bootstrap[
                "target_rate_cluster_ci_low"
            ]
        ),
        "target_log_cluster_95_ci_high": (
            bootstrap[
                "target_rate_cluster_ci_high"
            ]
        ),
        "control_alerts": control_alert_count,
        "control_observed_false_alert_rate": (
            control_rate
        ),
        "control_wilson_95_ci_low": control_ci[0],
        "control_wilson_95_ci_high": control_ci[1],
        "control_log_cluster_95_ci_low": (
            bootstrap[
                "control_rate_cluster_ci_low"
            ]
        ),
        "control_log_cluster_95_ci_high": (
            bootstrap[
                "control_rate_cluster_ci_high"
            ]
        ),
        "paired_alert_rate_difference": (
            alert_difference
        ),
        "paired_difference_log_cluster_95_ci_low": (
            bootstrap[
                "alert_difference_cluster_ci_low"
            ]
        ),
        "paired_difference_log_cluster_95_ci_high": (
            bootstrap[
                "alert_difference_cluster_ci_high"
            ]
        ),
        "discordant_target_only": target_only,
        "discordant_control_only": control_only,
        "concordant_both_alert": both_alert,
        "concordant_neither_alert": neither_alert,
        "mcnemar_exact_two_sided_p": mcnemar_p,
        "mean_target_score": float(
            target_pair_scores.mean()
        ),
        "mean_control_score": float(
            control_pair_scores.mean()
        ),
        "mean_paired_score_difference": float(
            (
                target_pair_scores -
                control_pair_scores
            ).mean()
        ),
        "paired_score_difference_log_cluster_95_ci_low": (
            bootstrap[
                "score_difference_cluster_ci_low"
            ]
        ),
        "paired_score_difference_log_cluster_95_ci_high": (
            bootstrap[
                "score_difference_cluster_ci_high"
            ]
        ),
    })

    print(
        f"{backbone:<25s} seed {seed} | "
        f"matched target={target_rate:.4%} | "
        f"control={control_rate:.4%} | "
        f"difference={alert_difference:+.4%} | "
        f"cluster CI=["
        f"{bootstrap['alert_difference_cluster_ci_low']:+.4%}, "
        f"{bootstrap['alert_difference_cluster_ci_high']:+.4%}]"
    )

full_pool = pd.DataFrame(full_pool_rows)
matched = pd.DataFrame(matched_rows)
category = pd.DataFrame(category_rows)
split = pd.DataFrame(split_rows)

matched[
    "mcnemar_holm_adjusted_p"
] = holm_adjust(
    matched[
        "mcnemar_exact_two_sided_p"
    ].to_numpy(dtype=np.float64)
)

full_backbone = summarize_seed_rates(
    full_pool,
    ["backbone", "audit_group"],
    [
        "observed_alert_rate",
        "mean_score",
    ],
)

matched_backbone = summarize_seed_rates(
    matched,
    ["backbone"],
    [
        "target_observed_alert_rate",
        "control_observed_false_alert_rate",
        "paired_alert_rate_difference",
        "mean_paired_score_difference",
    ],
)

category_backbone = summarize_seed_rates(
    category,
    [
        "backbone",
        "category",
        "membership_scope",
    ],
    [
        "observed_alert_rate",
        "mean_score",
    ],
)

split_backbone = summarize_seed_rates(
    split,
    [
        "backbone",
        "dataset_split",
        "audit_group",
    ],
    [
        "observed_alert_rate",
        "mean_score",
    ],
)

full_pool.to_csv(
    FULL_SEEDWISE_PATH,
    index=False,
    float_format="%.12g",
)

full_backbone.to_csv(
    FULL_BACKBONE_PATH,
    index=False,
    float_format="%.12g",
)

matched.to_csv(
    MATCHED_SEEDWISE_PATH,
    index=False,
    float_format="%.12g",
)

matched_backbone.to_csv(
    MATCHED_BACKBONE_PATH,
    index=False,
    float_format="%.12g",
)

category.to_csv(
    CATEGORY_SEEDWISE_PATH,
    index=False,
    float_format="%.12g",
)

category_backbone.to_csv(
    CATEGORY_BACKBONE_PATH,
    index=False,
    float_format="%.12g",
)

split.to_csv(
    SPLIT_SEEDWISE_PATH,
    index=False,
    float_format="%.12g",
)

split_backbone.to_csv(
    SPLIT_BACKBONE_PATH,
    index=False,
    float_format="%.12g",
)

if len(full_pool) != 30:
    failures.append(
        f"Expected 30 full-pool rows, "
        f"found {len(full_pool)}"
    )

if len(matched) != 10:
    failures.append(
        f"Expected 10 matched rows, "
        f"found {len(matched)}"
    )

if len(category) != 60:
    failures.append(
        f"Expected 60 category rows, "
        f"found {len(category)}"
    )

if len(split) != 40:
    failures.append(
        f"Expected 40 split rows, "
        f"found {len(split)}"
    )

status = "PASS" if not failures else "FAIL"
elapsed_seconds = time.perf_counter() - analysis_start

output_files = [
    FULL_SEEDWISE_PATH,
    FULL_BACKBONE_PATH,
    MATCHED_SEEDWISE_PATH,
    MATCHED_BACKBONE_PATH,
    CATEGORY_SEEDWISE_PATH,
    CATEGORY_BACKBONE_PATH,
    SPLIT_SEEDWISE_PATH,
    SPLIT_BACKBONE_PATH,
]

metadata = {
    "status": status,
    "audit": (
        "TAM-VLM external real-image "
        "hard-negative and object-response "
        "statistical analysis"
    ),
    "scientific_claim_boundary": (
        "All target-containing nuImages frames "
        "are legitimate natural scenes. Reported "
        "quantities are observed benign alert rates, "
        "control false-alert rates, and paired "
        "differences. They are not attack TPR, ASR, "
        "or physical-attack validation."
    ),
    "n_full_pool_images": EXPECTED_ROWS,
    "n_target_containing_images": (
        EXPECTED_TARGET_ROWS
    ),
    "n_target_category_absent_controls": (
        EXPECTED_CONTROL_ROWS
    ),
    "n_same_log_matched_pairs": EXPECTED_PAIRS,
    "matched_target_coverage": (
        EXPECTED_MATCHED_TARGET_COVERAGE
    ),
    "n_frozen_detector_heads": 10,
    "n_backbones": 2,
    "n_seeds_per_backbone": 5,
    "finite_sample_interval": (
        "Two-sided 95% Wilson score interval"
    ),
    "matched_difference_interval": (
        "Two-sided 95% percentile cluster "
        "bootstrap over nuImages log_token"
    ),
    "bootstrap_replicates": (
        BOOTSTRAP_REPLICATES
    ),
    "bootstrap_base_seed": (
        BOOTSTRAP_BASE_SEED
    ),
    "matched_binary_test": (
        "Exact two-sided McNemar/binomial test "
        "on discordant pairs"
    ),
    "multiplicity_adjustment": (
        "Holm correction across ten frozen heads"
    ),
    "category_analysis": (
        "Multi-label membership; each image was "
        "encoded and scored once. Exclusive-only "
        "results are a secondary sensitivity analysis."
    ),
    "score_retraining_performed": False,
    "threshold_recalibration_performed": False,
    "model_selection_on_external_data": False,
    "scores_path": str(SCORES_PATH),
    "scores_sha256": sha256_file(
        SCORES_PATH
    ),
    "matched_pairs_path": str(PAIRS_PATH),
    "matched_pairs_sha256": sha256_file(
        PAIRS_PATH
    ),
    "artifact_manifest_path": str(
        ARTIFACT_MANIFEST_PATH
    ),
    "artifact_manifest_sha256": sha256_file(
        ARTIFACT_MANIFEST_PATH
    ),
    "elapsed_seconds": elapsed_seconds,
    "outputs": {
        str(path.name): {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for path in output_files
    },
    "failures": failures,
}

atomic_json_write(
    METADATA_PATH,
    metadata,
)

report_lines = [
    "=" * 112,
    "TAM-VLM external real-image statistical analysis",
    "=" * 112,
    f"Status                         : {status}",
    f"Full-pool images               : {EXPECTED_ROWS}",
    f"Target-containing images       : {EXPECTED_TARGET_ROWS}",
    f"Target-category-absent controls: {EXPECTED_CONTROL_ROWS}",
    f"Same-log matched pairs         : {EXPECTED_PAIRS}",
    (
        "Matched target coverage        : "
        f"{EXPECTED_MATCHED_TARGET_COVERAGE:.2%}"
    ),
    (
        "Finite-sample intervals        : "
        "95% Wilson"
    ),
    (
        "Paired-difference intervals    : "
        f"95% log-cluster bootstrap, "
        f"{BOOTSTRAP_REPLICATES} replicates"
    ),
    (
        "Matched binary tests           : "
        "exact McNemar with Holm correction"
    ),
]

for backbone in [
    "OpenCLIP",
    "Qwen3-VL-Embedding-2B",
]:
    full_target = full_backbone[
        full_backbone["backbone"].eq(backbone) &
        full_backbone["audit_group"].eq(
            "target_containing"
        )
    ].iloc[0]

    full_control = full_backbone[
        full_backbone["backbone"].eq(backbone) &
        full_backbone["audit_group"].eq(
            "target_category_absent_control"
        )
    ].iloc[0]

    matched_summary = matched_backbone[
        matched_backbone["backbone"].eq(
            backbone
        )
    ].iloc[0]

    matched_heads = matched[
        matched["backbone"].eq(backbone)
    ]

    report_lines.extend([
        "",
        backbone,
        (
            "  Full-pool target observed alert rate : "
            f"{full_target['observed_alert_rate_mean']:.4%} "
            f"± "
            f"{full_target['observed_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Full-pool control false-alert rate   : "
            f"{full_control['observed_alert_rate_mean']:.4%} "
            f"± "
            f"{full_control['observed_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Matched target observed alert rate   : "
            f"{matched_summary['target_observed_alert_rate_mean']:.4%} "
            f"± "
            f"{matched_summary['target_observed_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Matched control false-alert rate     : "
            f"{matched_summary['control_observed_false_alert_rate_mean']:.4%} "
            f"± "
            f"{matched_summary['control_observed_false_alert_rate_std_ddof0']:.4%}"
        ),
        (
            "  Matched target-minus-control         : "
            f"{matched_summary['paired_alert_rate_difference_mean']:+.4%} "
            f"± "
            f"{matched_summary['paired_alert_rate_difference_std_ddof0']:.4%}"
        ),
        (
            "  Holm-adjusted McNemar p range        : "
            f"{matched_heads['mcnemar_holm_adjusted_p'].min():.6g}"
            "–"
            f"{matched_heads['mcnemar_holm_adjusted_p'].max():.6g}"
        ),
    ])

report_lines.extend([
    "",
    (
        "Interpretation: legitimate real-image "
        "benign alert and false-alert behavior only; "
        "not attack TPR, ASR, or physical-attack "
        "validation."
    ),
    (
        "Zero observed alerts, where present, mean "
        "finite-sample zero only; the Wilson upper "
        "confidence bound remains non-zero."
    ),
    f"Elapsed seconds                : {elapsed_seconds:.2f}",
    f"Saved metadata                 : {METADATA_PATH}",
])

REPORT_PATH.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report_lines))

if status != "PASS":
    raise SystemExit(
        "External statistical analysis failed."
    )
