import os
import matplotlib

# ============================================================
# SERVER-SAFE BACKEND
# ============================================================
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = Path(
    os.environ.get(
        "TAMVLM_RESULTS_ROOT",
        PROJECT_ROOT / "results"
    )
)

openclip_results = RESULTS_ROOT / "openclip_results"

qwen_results = RESULTS_ROOT / "qwen_results"

out_dir = PROJECT_ROOT / "figure_generation" / "figures"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

temporal = pd.read_csv(
    os.path.join(
        BASE,
        "temporal_trigger_per_type_openclip_vs_qwen.csv"
    )
)

hard = pd.read_csv(
    os.path.join(
        BASE,
        "hard_negative_openclip_vs_qwen.csv"
    )
)


# ============================================================
# EXACT 12 PHYSICAL TRIGGER CATEGORIES
#
# IMPORTANT:
# "none" is NOT a physical trigger category.
# It is deliberately excluded from Panel (a).
# ============================================================

TRIGGER_ORDER = [
    "bollard",
    "fire hydrant",
    "football",
    "ladder",
    "potted plant",
    "red balloon",
    "roadside litter",
    "rose",
    "teddy bear",
    "traffic barrier",
    "traffic cone",
    "umbrella",
]


DISPLAY = {
    "bollard": "Bollard",
    "fire hydrant": "Fire hydrant",
    "football": "Football",
    "ladder": "Ladder",
    "potted plant": "Potted plant",
    "red balloon": "Red balloon",
    "roadside litter": "Roadside litter",
    "rose": "Rose",
    "teddy bear": "Teddy bear",
    "traffic barrier": "Traffic barrier",
    "traffic cone": "Traffic cone",
    "umbrella": "Umbrella",
}


# ============================================================
# NORMALIZE TRIGGER NAMES ONLY FOR MATCHING
# Numerical results are NOT changed.
# ============================================================

temporal["trigger_norm"] = (
    temporal["trigger"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("_", " ", regex=False)
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

source_triggers = set(temporal["trigger_norm"])
expected_triggers = set(TRIGGER_ORDER)

missing = expected_triggers - source_triggers
unexpected = source_triggers - expected_triggers

if missing:
    raise ValueError(
        f"Missing expected trigger categories: {sorted(missing)}"
    )


# Keep only the 12 physical trigger categories.
temporal = temporal[
    temporal["trigger_norm"].isin(expected_triggers)
].copy()


# Exactly one row per physical trigger.
if len(temporal) != len(TRIGGER_ORDER):
    raise ValueError(
        "Expected exactly "
        f"{len(TRIGGER_ORDER)} physical trigger rows "
        f"after filtering; found {len(temporal)}"
    )


if temporal["trigger_norm"].duplicated().any():
    dup = temporal.loc[
        temporal["trigger_norm"].duplicated(),
        "trigger_norm"
    ].tolist()

    raise ValueError(
        f"Duplicate trigger rows detected: {dup}"
    )


# ============================================================
# STABLE MANUSCRIPT ORDER
# ============================================================

order_map = {
    name: i
    for i, name in enumerate(TRIGGER_ORDER)
}

temporal["_order"] = temporal["trigger_norm"].map(order_map)

temporal = (
    temporal
    .sort_values("_order")
    .reset_index(drop=True)
)


# ============================================================
# REQUIRED COLUMN CHECKS
# ============================================================

required_temporal_columns = [
    "trigger",
    "scene_detection_fraction_mean_Qwen",
    "scene_detection_fraction_mean_OpenCLIP",
]

required_hard_columns = [
    "OpenCLIP_estimate",
    "OpenCLIP_ci_low",
    "OpenCLIP_ci_high",
    "Qwen_estimate",
    "Qwen_ci_low",
    "Qwen_ci_high",
]

for col in required_temporal_columns:
    if col not in temporal.columns:
        raise KeyError(
            f"Missing temporal column: {col}"
        )

for col in required_hard_columns:
    if col not in hard.columns:
        raise KeyError(
            f"Missing hard-negative column: {col}"
        )


# ============================================================
# FONT / PDF SETTINGS
#
# Prefer Arial/Helvetica-compatible fonts.
# Fall back safely if unavailable.
# ============================================================

available_fonts = {
    f.name
    for f in matplotlib.font_manager.fontManager.ttflist
}

font_candidates = [
    "Arial",
    "Helvetica",
    "Liberation Sans",
    "DejaVu Sans",
]

selected_font = None

for candidate in font_candidates:
    if candidate in available_fonts:
        selected_font = candidate
        break

if selected_font is None:
    selected_font = "DejaVu Sans"


plt.rcParams.update({

    # IEEE-compatible TrueType embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "font.family": selected_font,

    "font.size": 8.5,

    "axes.labelsize": 9,

    "axes.titlesize": 10,

    "xtick.labelsize": 8,

    "ytick.labelsize": 8,

    "legend.fontsize": 8,

    "axes.linewidth": 0.8,

    "xtick.major.width": 0.7,

    "ytick.major.width": 0.7,

})


print("Selected plotting font:", selected_font)


# ============================================================
# COLORS
#
# Existing manuscript visual convention is retained.
# ============================================================

OPENCLIP_COLOR = "#1f5aa6"
QWEN_COLOR = "#16805c"


# ============================================================
# FIGURE
#
# Designed at approximately IEEE two-column width.
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.16, 3.75)
)


# ============================================================
# PANEL A
# CROSS-BACKBONE TEMPORAL PRESERVATION GAP
#
# Delta = Qwen3-VL - OpenCLIP
#
# Positive:
# Qwen3-VL has higher scene-detection fraction.
#
# Negative:
# OpenCLIP has higher scene-detection fraction.
# ============================================================

ax = axes[0]


delta = (
    temporal[
        "scene_detection_fraction_mean_Qwen"
    ].to_numpy()
    -
    temporal[
        "scene_detection_fraction_mean_OpenCLIP"
    ].to_numpy()
)


labels = [
    DISPLAY[t]
    for t in temporal["trigger_norm"]
]


y = np.arange(len(labels))


# ============================================================
# BAR COLORS
# ============================================================

bar_colors = [
    QWEN_COLOR if d > 0 else OPENCLIP_COLOR
    for d in delta
]


ax.barh(
    y,
    delta,
    height=0.55,
    color=bar_colors,
    edgecolor="none",
)


# Zero reference line
ax.axvline(
    0,
    color="black",
    linewidth=0.8,
)


# ============================================================
# VALUE LABELS
#
# IMPORTANT:
# Zero values are NOT printed.
# This removes the visual clutter caused by repeated 0.000
# labels around the zero line.
# ============================================================

for i, d in enumerate(delta):

    # Do not label exact/near-zero differences.
    if np.isclose(d, 0.0, atol=5e-4):
        continue


    if d > 0:

        ax.text(
            d + 0.007,
            i,
            f"{d:+.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
        )

    else:

        ax.text(
            d - 0.007,
            i,
            f"{d:+.3f}",
            va="center",
            ha="right",
            fontsize=7.5,
        )


# ============================================================
# AXIS FORMATTING
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(labels)

ax.invert_yaxis()

ax.set_xlim(
    -0.18,
    0.13
)


ax.set_xlabel(
    "Qwen3-VL − OpenCLIP scene-detection fraction (Δ)"
)


# IMPORTANT:
# "Held-out trigger category" removed.
ax.set_ylabel(
    "Physical trigger category"
)


ax.set_title(
    "(a) Cross-backbone temporal preservation gap",
    pad=8,
)


ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25,
)


# ============================================================
# PANEL A LEGEND
#
# Explains the direction of the delta without making any
# performance ranking claim.
# ============================================================

ax.set_title(
    "(a) Cross-backbone temporal preservation gap",
    pad=8,
)

# ============================================================
# PANEL B
# HARD-NEGATIVE FALSE-ALERT RATE
# ============================================================

ax = axes[1]


labels_b = [
    "Natural\nnull",
    "Matched\nnull",
    "Generic\nclean",
]


open_vals = (
    hard["OpenCLIP_estimate"]
    .to_numpy()
)


qwen_vals = (
    hard["Qwen_estimate"]
    .to_numpy()
)


x = np.arange(
    len(labels_b)
)


width = 0.34


# ============================================================
# BARS
#
# Hatch patterns make the figure interpretable in grayscale
# as well as color.
# ============================================================

bars1 = ax.bar(
    x - width / 2,
    open_vals,
    width,
    color=OPENCLIP_COLOR,
    edgecolor="black",
    linewidth=0.35,
    hatch="",
    label="OpenCLIP ViT-B/16",
)


bars2 = ax.bar(
    x + width / 2,
    qwen_vals,
    width,
    color=QWEN_COLOR,
    edgecolor="black",
    linewidth=0.35,
    hatch="///",
    label="Qwen3-VL-Embedding-2B",
)


# ============================================================
# 95% HIERARCHICAL SCENE-BOOTSTRAP CI
#
# The source analysis uses the 2.5th and 97.5th percentiles.
# ============================================================

ax.errorbar(
    x - width / 2,
    open_vals,

    yerr=[
        open_vals
        -
        hard["OpenCLIP_ci_low"].to_numpy(),

        hard["OpenCLIP_ci_high"].to_numpy()
        -
        open_vals,
    ],

    fmt="none",

    ecolor="black",

    capsize=2,

    linewidth=0.55,
)


ax.errorbar(
    x + width / 2,
    qwen_vals,

    yerr=[
        qwen_vals
        -
        hard["Qwen_ci_low"].to_numpy(),

        hard["Qwen_ci_high"].to_numpy()
        -
        qwen_vals,
    ],

    fmt="none",

    ecolor="black",

    capsize=2,

    linewidth=0.55,
)


# ============================================================
# BAR VALUE LABELS
#
# Zero bars are labeled only as 0.00%.
# No "(0 events)" text is placed inside the figure.
# ============================================================

for bars in (bars1, bars2):

    for bar in bars:

        h = bar.get_height()


        if np.isclose(
            h,
            0.0,
            atol=5e-6,
        ):

            label = "0.00%"

        else:

            label = f"{100*h:.2f}%"


        # Slightly higher position for normal bars.
        # Zero bars are lifted enough to avoid touching x-axis.
        y_text = (
            0.0017
            if np.isclose(h, 0.0, atol=5e-6)
            else h + 0.0015
        )


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            y_text,

            label,

            ha="center",

            va="bottom",

            fontsize=7.5,
        )


# ============================================================
# AXIS FORMATTING
# ============================================================

ax.set_xticks(x)

ax.set_xticklabels(labels_b)


ax.set_ylabel(
    "False-alert rate (%)"
)


ax.set_xlabel(
    "Hard-negative benchmark"
)


ax.set_title(
    "(b) Hard-negative false-alert rate",
    pad=8,
)


# ============================================================
# LEGEND
# ============================================================

ax.legend(
    frameon=False,
    fontsize=7.5,
    loc="upper right",
)


# ============================================================
# Y AXIS
#
# Data remain unchanged; only display is converted to percent.
# ============================================================

ax.set_ylim(
    0,
    0.055,
)


ax.set_yticks(
    np.arange(
        0,
        0.051,
        0.01,
    )
)


ax.set_yticklabels(
    [
        f"{100*v:.0f}%"
        for v in np.arange(
            0,
            0.051,
            0.01,
        )
    ]
)


ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.25,
)


# ============================================================
# COMMON CLEANUP
# ============================================================

for ax in axes:

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        length=3,
    )


# ============================================================
# FINAL LAYOUT
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.985,
    bottom=0.19,
    top=0.88,
    wspace=0.42,
)


# ============================================================
# OUTPUT FILES
# ============================================================

png = os.path.join(
    OUT,
"fig8_runtime_robustness_Q1_FINAL_v10.png"
)


pdf = os.path.join(
    OUT,
"fig8_runtime_robustness_Q1_FINAL_v10.pdf"
)


# High-resolution PNG for inspection.
plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight",
)


# Vector PDF for manuscript.
plt.savefig(
    pdf,
    bbox_inches="tight",
)


plt.close(fig)


# ============================================================
# FINAL AUDIT OUTPUT
# ============================================================

print()
print("=" * 70)
print("FIGURE 8 V9 GENERATED SUCCESSFULLY")
print("=" * 70)

print()
print("PDF:")
print(pdf)

print()
print("PNG:")
print(png)

print()
print("Font:")
print(selected_font)

print()
print("Panel A categories:")
for category in labels:
    print("  -", category)

print()
print("Number of physical trigger categories:")
print(len(labels))

print()
print("Excluded source categories:")
print(sorted(unexpected))

print()
print("Delta definition:")
print("Qwen3-VL - OpenCLIP")

print()
print("Hard-negative uncertainty:")
print("95% hierarchical scene-bootstrap interval")

print()
print("Scientific values:")
print("UNCHANGED")

print()
print("=" * 70)
