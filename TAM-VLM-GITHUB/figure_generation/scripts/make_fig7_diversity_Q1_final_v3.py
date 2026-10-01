import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = Path(
    os.environ.get(
        "TAMVLM_RESULTS_ROOT",
        PROJECT_ROOT / "results"
    )
)

openclip_results = RESULTS_ROOT / "openclip_results"

out_dir = PROJECT_ROOT / "figure_generation" / "figures"
os.makedirs(out_dir, exist_ok=True)


# Load

df = pd.read_csv(csv_path)

df["k"] = df["k"].astype(int)



# Figure

fig, ax = plt.subplots(
    figsize=(6.5,4.0)
)


ax.errorbar(
    df["k"],
    df["mean"],
    yerr=df["std"],
    marker="o",
    linewidth=2.2,
    markersize=7,
    capsize=3,
    elinewidth=1,
    color="#2F5597"
)



# Value labels

for _, row in df.iterrows():

    ax.text(
        row["k"],
        row["mean"] + 0.012,
        f"{row['mean']:.3f}",
        ha="center",
        fontsize=8
    )



# Labels

ax.set_xlabel(
    "Number of represented trigger categories (k)",
    fontsize=10
)

ax.set_ylabel(
    "Mean unseen AUROC",
    fontsize=10
)


ax.set_xticks(
    df["k"]
)


ax.set_ylim(
0.70,
0.90
)


ax.grid(
axis="y",
alpha=0.2
)



plt.tight_layout()



# Save

png = (
out_dir+
"/fig7_diversity_audit_Q1_final_v2.png"
)

pdf = (
out_dir+
"/fig7_diversity_audit_Q1_final_v2.pdf"
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
