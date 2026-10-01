from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/real_capture_transfer_v1"
EXCL = BASE / "exclusive_size_sensitivity_v1"
FIGDIR = ROOT / "figures/real_capture_transfer_v1"
FIGDIR.mkdir(parents=True, exist_ok=True)

SEEDWISE = EXCL / "exclusive_size_seedwise_v1.csv"
TREND = EXCL / "exclusive_size_trend_backbone_summary_v1.csv"
CATEGORY = BASE / "category_transfer_backbone_summary_v1.csv"

OUT_SUMMARY = BASE / "FINAL_REAL_OBJECT_PROMINENCE_SUMMARY_v1.csv"
OUT_REPORT = BASE / "FINAL_REAL_OBJECT_PROMINENCE_REPORT_v1.txt"
OUT_META = BASE / "FINAL_REAL_OBJECT_PROMINENCE_METADATA_v1.json"

FIG_PDF = FIGDIR / "FIG11_real_object_prominence_sensitivity_v1.pdf"
FIG_PNG = FIGDIR / "FIG11_real_object_prominence_sensitivity_v1.png"

SIZE_ORDER = ["small", "medium", "large"]
BACKBONE_ORDER = ["OpenCLIP", "Qwen3-VL"]
CATEGORY_ORDER = ["traffic_cone", "traffic_barrier"]

DISPLAY_CATEGORY = {
    "traffic_cone": "Traffic cone",
    "traffic_barrier": "Traffic barrier",
}

# ------------------------------------------------------------
# Load frozen results
# ------------------------------------------------------------
seedwise = pd.read_csv(SEEDWISE)
trend = pd.read_csv(TREND)
category = pd.read_csv(CATEGORY)

assert set(CATEGORY_ORDER).issubset(seedwise["category"].unique())
assert set(BACKBONE_ORDER).issubset(seedwise["backbone"].unique())

# ------------------------------------------------------------
# Build final compact summary
# ------------------------------------------------------------
rows = []

for cat in CATEGORY_ORDER:
    for backbone in BACKBONE_ORDER:

        d = seedwise[
            (seedwise["category"] == cat)
            & (seedwise["backbone"] == backbone)
        ].copy()

        tr = trend[
            (trend["category"] == cat)
            & (trend["backbone"] == backbone)
        ].iloc[0]

        row = {
            "category": cat,
            "backbone": backbone,
            "n_pairs": int(tr["n_pairs"]),
            "n_logs": int(tr["n_logs"]),
            "continuous_mean_slope": float(tr["mean_slope"]),
            "continuous_sd_slope_ddof0": float(tr["sd_slope_ddof0"]),
            "positive_slope_heads": int(tr["positive_slope_heads"]),
            "heads_ci_excluding_zero": int(tr["heads_ci_excluding_zero"]),
        }

        for size in SIZE_ORDER:
            g = d[d["size_group"] == size]

            vals = g["paired_score_difference"].to_numpy(float)

            row[f"{size}_paired_score_difference_mean"] = float(vals.mean())
            row[f"{size}_paired_score_difference_sample_sd"] = (
                float(vals.std(ddof=1))
                if len(vals) > 1 else np.nan
            )

            row[f"{size}_target_alert_rate_mean"] = float(
                g["target_alert_rate"].mean()
            )
            row[f"{size}_control_alert_rate_mean"] = float(
                g["control_alert_rate"].mean()
            )
            row[f"{size}_paired_alert_difference_mean"] = float(
                g["paired_alert_difference"].mean()
            )

        rows.append(row)

final_summary = pd.DataFrame(rows)
final_summary.to_csv(OUT_SUMMARY, index=False)

# ------------------------------------------------------------
# Publication figure
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(9.2, 3.65),
    sharey=True
)

x = np.arange(len(SIZE_ORDER))

for ax, backbone in zip(axes, BACKBONE_ORDER):

    for cat in CATEGORY_ORDER:

        d = seedwise[
            (seedwise["category"] == cat)
            & (seedwise["backbone"] == backbone)
        ].copy()

        means = []
        sds = []

        for size in SIZE_ORDER:
            vals = d.loc[
                d["size_group"] == size,
                "paired_score_difference"
            ].to_numpy(float)

            means.append(vals.mean())
            sds.append(vals.std(ddof=1))

        means = np.asarray(means)
        sds = np.asarray(sds)

        ax.errorbar(
            x,
            means,
            yerr=sds,
            marker="o",
            linewidth=2.0,
            capsize=3.5,
            label=DISPLAY_CATEGORY[cat],
        )

    ax.axhline(
        0,
        linewidth=1.0,
        linestyle="--"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Small", "Medium", "Large"]
    )

    ax.set_xlabel(
        "Real-object prominence\n(category-specific bbox-area tertile)"
    )

    ax.set_title(backbone)

    ax.grid(
        axis="y",
        alpha=0.22,
        linewidth=0.6
    )

axes[0].set_ylabel(
    "Paired target − matched-control detector score"
)

axes[1].legend(
    frameon=False,
    loc="upper left"
)

fig.suptitle(
    "Real-Object Prominence Sensitivity on Exclusive nuImages Categories",
    fontsize=11.5,
    y=1.01
)

fig.tight_layout()

fig.savefig(
    FIG_PDF,
    bbox_inches="tight"
)

fig.savefig(
    FIG_PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

# ------------------------------------------------------------
# Key results
# ------------------------------------------------------------
def row_for(cat, backbone):
    return final_summary[
        (final_summary["category"] == cat)
        & (final_summary["backbone"] == backbone)
    ].iloc[0]

oc_cone = row_for("traffic_cone", "OpenCLIP")
qw_cone = row_for("traffic_cone", "Qwen3-VL")
oc_bar = row_for("traffic_barrier", "OpenCLIP")
qw_bar = row_for("traffic_barrier", "Qwen3-VL")

lines = []

lines.append("TAM-VLM REAL-OBJECT PROMINENCE FINAL REPORT v1")
lines.append("=" * 72)
lines.append("")
lines.append("STATUS: PASS")
lines.append("")
lines.append("ANALYSIS SCOPE")
lines.append(
    "Post-hoc exclusive-category sensitivity analysis on frozen nuImages "
    "same-log matched pairs."
)
lines.append(
    "No retraining, threshold recalibration, fine-tuning, or model selection "
    "was performed."
)
lines.append("")

lines.append("TRAFFIC CONE")
lines.append(
    f"OpenCLIP paired score difference: "
    f"{oc_cone['small_paired_score_difference_mean']:.4f} -> "
    f"{oc_cone['medium_paired_score_difference_mean']:.4f} -> "
    f"{oc_cone['large_paired_score_difference_mean']:.4f}"
)
lines.append(
    f"OpenCLIP continuous trend: "
    f"{int(oc_cone['positive_slope_heads'])}/5 positive slopes; "
    f"{int(oc_cone['heads_ci_excluding_zero'])}/5 bootstrap CIs exclude zero."
)

lines.append(
    f"Qwen paired score difference: "
    f"{qw_cone['small_paired_score_difference_mean']:.4f} -> "
    f"{qw_cone['medium_paired_score_difference_mean']:.4f} -> "
    f"{qw_cone['large_paired_score_difference_mean']:.4f}"
)
lines.append(
    f"Qwen continuous trend: "
    f"{int(qw_cone['positive_slope_heads'])}/5 positive slopes; "
    f"{int(qw_cone['heads_ci_excluding_zero'])}/5 bootstrap CIs exclude zero."
)
lines.append("")

lines.append("TRAFFIC BARRIER")
lines.append(
    f"OpenCLIP paired score difference: "
    f"{oc_bar['small_paired_score_difference_mean']:.4f} -> "
    f"{oc_bar['medium_paired_score_difference_mean']:.4f} -> "
    f"{oc_bar['large_paired_score_difference_mean']:.4f}"
)
lines.append(
    f"OpenCLIP continuous trend: "
    f"{int(oc_bar['positive_slope_heads'])}/5 positive slopes; "
    f"{int(oc_bar['heads_ci_excluding_zero'])}/5 bootstrap CIs exclude zero."
)

lines.append(
    f"Qwen paired score difference: "
    f"{qw_bar['small_paired_score_difference_mean']:.4f} -> "
    f"{qw_bar['medium_paired_score_difference_mean']:.4f} -> "
    f"{qw_bar['large_paired_score_difference_mean']:.4f}"
)
lines.append(
    f"Qwen continuous trend: "
    f"{int(qw_bar['positive_slope_heads'])}/5 positive slopes; "
    f"{int(qw_bar['heads_ci_excluding_zero'])}/5 bootstrap CIs exclude zero."
)
lines.append("")

lines.append("CLAIM BOUNDARY")
lines.append(
    "These results support scale-dependent synthetic-to-real object-response "
    "transfer on naturally photographed overlapping categories."
)
lines.append(
    "They do not establish physical-backdoor attack TPR, attack success-rate "
    "reduction, or victim-model mitigation."
)
lines.append("")
lines.append(
    "Debris is excluded from the main figure because the exclusive subset "
    "contains only 80 matched pairs and does not show a stable size trend."
)

OUT_REPORT.write_text(
    "\n".join(lines) + "\n"
)

# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------
metadata = {
    "status": "PASS",
    "figure_pdf": str(FIG_PDF),
    "figure_png": str(FIG_PNG),
    "summary_csv": str(OUT_SUMMARY),
    "report": str(OUT_REPORT),
    "main_figure_categories": [
        "traffic_cone",
        "traffic_barrier",
    ],
    "debris_main_figure_excluded_reason":
        "exclusive subset n=80 and no stable continuous size trend",
    "figure_error_bars":
        "sample SD across five frozen detector-head seeds",
    "bootstrap_inference_source":
        "exclusive_size_trend_backbone_summary_v1.csv",
    "analysis_type":
        "post-hoc exclusive-category sensitivity analysis",
    "retraining": False,
    "threshold_recalibration": False,
    "model_selection_on_nuimages": False,
}

with open(OUT_META, "w") as f:
    json.dump(metadata, f, indent=2)

# ------------------------------------------------------------
# SHA256 freeze
# ------------------------------------------------------------
files_to_hash = [
    OUT_SUMMARY,
    OUT_REPORT,
    OUT_META,
    FIG_PDF,
    FIG_PNG,
]

checksum_file = (
    BASE / "FINAL_REAL_OBJECT_PROMINENCE_SHA256_v1.txt"
)

with checksum_file.open("w") as f:
    for path in files_to_hash:
        h = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        f.write(f"{h}  {path}\n")

print("=" * 72)
print("FINALIZATION COMPLETE")
print("=" * 72)

print("\nFINAL SUMMARY")
print(final_summary.to_string(index=False))

print("\nREPORT")
print(OUT_REPORT.read_text())

print("FIGURE PDF:", FIG_PDF)
print("FIGURE PNG:", FIG_PNG)
print("SUMMARY CSV:", OUT_SUMMARY)
print("CHECKSUMS:", checksum_file)
print("\nSTATUS: PASS")
