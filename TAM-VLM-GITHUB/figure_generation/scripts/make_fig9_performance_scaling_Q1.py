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
# LOAD DATA
# ============================================================

main = pd.read_csv(
    os.path.join(
        BASE,
        "main_5seeds_summary.csv"
    )
)


ablation = pd.read_csv(
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
"axes.titlesize":10,
"axes.labelsize":9,
"xtick.labelsize":7,
"ytick.labelsize":7

})


fig, axes = plt.subplots(
    1,
    3,
    figsize=(10,3.2)
)



# ============================================================
# PANEL A
# Training size scaling
# ============================================================

ax = axes[0]


x = (
ablation["requested_fraction"]
.values
)


y = (
ablation["AUROC_mean"]
.values
)


std = (
ablation["AUROC_std"]
.values
)


ax.plot(
x,
y,
marker="o",
linewidth=2,
color="#1f5aa6"
)


ax.fill_between(
x,
y-std,
y+std,
alpha=0.2
)


ax.set_xscale(
"log"
)


ax.set_xlabel(
"Positive supervision fraction"
)


ax.set_ylabel(
"AUROC"
)


ax.set_title(
"(a) Data efficiency"
)


ax.grid(
linestyle="--",
alpha=0.25
)



# ============================================================
# PANEL B
# LOTO generalization
# ============================================================

ax = axes[1]


loto = loto.sort_values(
"unseen_AUROC_mean",
ascending=True
)


labels = loto[
"held_out_trigger"
]


values = loto[
"unseen_AUROC_mean"
]


ax.barh(
range(len(labels)),
values,
color="#16805c"
)


ax.set_yticks(
range(len(labels))
)


ax.set_yticklabels(
labels,
fontsize=6
)


ax.set_xlim(
0.4,
1.0
)


ax.set_xlabel(
"Unseen AUROC"
)


ax.set_title(
"(b) LOTO generalization"
)


ax.grid(
axis="x",
linestyle="--",
alpha=0.25
)



# ============================================================
# PANEL C
# Main performance
# ============================================================

ax = axes[2]


metrics = [
"AUROC",
"AUPRC",
"TPR@1%FPR"
]


values = [
main.loc[
main.iloc[:,0]=="AUROC",
"mean"
].values[0],

main.loc[
main.iloc[:,0]=="AUPRC",
"mean"
].values[0],

main.loc[
main.iloc[:,0]=="TPR_at_test_1pct_FPR_diagnostic",
"mean"
].values[0]
]


ax.bar(
metrics,
values,
color="#1f5aa6"
)


for i,v in enumerate(values):

    ax.text(
        i,
        v+0.02,
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
"(c) Overall performance"
)


ax.grid(
axis="y",
linestyle="--",
alpha=0.25
)



# ============================================================
# SAVE
# ============================================================


plt.tight_layout()


png = os.path.join(
OUT,
"fig9_performance_scaling_Q1.png"
)


pdf = os.path.join(
OUT,
"fig9_performance_scaling_Q1.pdf"
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
