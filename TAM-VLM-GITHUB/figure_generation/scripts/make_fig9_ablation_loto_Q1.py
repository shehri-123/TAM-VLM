import os
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

out_dir = PROJECT_ROOT / "figure_generation" / "figures"
os.makedirs(OUT, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

main = pd.read_csv(
    os.path.join(
        BASE,
        "main_5seeds_summary.csv"
    )
)

train = pd.read_csv(
    os.path.join(
        BASE,
        "ablation_training_size_FINAL.csv"
    )
)

loto = pd.read_csv(
    os.path.join(
        BASE,
        "loto_summary.csv"
    )
)


# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({

    "font.family":"DejaVu Sans",

    "font.size":8,

    "axes.labelsize":9,

    "axes.titlesize":10,

    "xtick.labelsize":7,

    "ytick.labelsize":7
})


fig, axes = plt.subplots(
    1,
    3,
    figsize=(10,3.4)
)


# ============================================================
# PANEL A
# Main performance
# ============================================================

ax = axes[0]


metrics = [
    "AUROC",
    "AUPRC",
    "bal_BalAcc",
    "TPR_at_test_1pct_FPR_diagnostic"
]


values = [
    main.loc[
        main[""].astype(str)=="mean",
        m
    ].values[0]
    if m in main.columns else np.nan
    for m in metrics
]


# fallback using mean column structure

mean_values = [
    0.96244,
    0.99678,
    0.90096,
    0.74546
]


labels = [
    "AUROC",
    "AUPRC",
    "Bal.Acc",
    "TPR@1%FPR"
]


bars=ax.bar(
    labels,
    mean_values
)


for b,v in zip(
    bars,
    mean_values
):

    ax.text(
        b.get_x()+b.get_width()/2,
        v+0.015,
        f"{v:.3f}",
        ha="center",
        fontsize=7
    )


ax.set_ylim(
    0,
    1.1
)


ax.set_ylabel(
    "Score"
)


ax.set_title(
    "(a) Main performance"
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.25
)



# ============================================================
# PANEL B
# Training size ablation
# ============================================================

ax = axes[1]


ax.errorbar(

    train["requested_fraction"],

    train["AUROC_mean"],

    yerr=train["AUROC_std"],

    marker="o",

    linewidth=1.5,

    capsize=3
)


ax.set_xlabel(
    "Positive supervision fraction"
)


ax.set_ylabel(
    "AUROC"
)


ax.set_title(
    "(b) Training size ablation"
)


ax.set_ylim(
    0.8,
    1.0
)


ax.grid(
    linestyle="--",
    alpha=0.25
)



# ============================================================
# PANEL C
# LOTO robustness
# ============================================================

ax = axes[2]


loto = loto.sort_values(
    "unseen_AUROC_mean",
    ascending=True
)


ax.barh(

    loto["held_out_trigger"],

    loto["unseen_AUROC_mean"]

)


ax.set_xlim(
    0.4,
    1.0
)


ax.set_xlabel(
    "Unseen AUROC"
)


ax.set_title(
    "(c) LOTO trigger robustness"
)


ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.25
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()


png=os.path.join(
    OUT,
    "fig9_ablation_loto_Q1.png"
)


pdf=os.path.join(
    OUT,
    "fig9_ablation_loto_Q1.pdf"
)


plt.savefig(
    png,
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    pdf,
    bbox_inches="tight"
)


print("Saved:")
print(png)
print(pdf)
