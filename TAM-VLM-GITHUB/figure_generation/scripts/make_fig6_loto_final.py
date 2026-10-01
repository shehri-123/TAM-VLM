import os
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# Paths
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = Path(
    os.environ.get(
        "TAMVLM_RESULTS_ROOT",
        PROJECT_ROOT / "results"
    )
)

openclip_csv = RESULTS_ROOT / "openclip_results" / "loto_results.csv"

qwen_csv = RESULTS_ROOT / "qwen_results" / "loto_results.csv"

out_dir = PROJECT_ROOT / "figure_generation" / "figures"


# =====================================================
# Load data
# =====================================================

df = pd.read_csv(csv_path)


df = df[
[
"held_out_trigger",
"unseen_AUROC_mean_OpenCLIP",
"unseen_AUROC_mean_Qwen"
]
]


df = df.rename(
columns={
"unseen_AUROC_mean_OpenCLIP":
"OpenCLIP ViT-B/16",

"unseen_AUROC_mean_Qwen":
"Qwen3-VL-Embedding-2B"
}
)



# =====================================================
# Sort by average unseen AUROC
# =====================================================

df["mean_AUROC"] = (
    df["OpenCLIP ViT-B/16"] +
    df["Qwen3-VL-Embedding-2B"]
) / 2


df = df.sort_values(
"mean_AUROC",
ascending=True
)



# =====================================================
# Plot
# =====================================================

fig, ax = plt.subplots(
    figsize=(7.2,5.0)
)


y = range(len(df))

height = 0.32


ax.barh(
    [i-height/2 for i in y],
    df["OpenCLIP ViT-B/16"],
    height=height,
    label="OpenCLIP ViT-B/16"
)


ax.barh(
    [i+height/2 for i in y],
    df["Qwen3-VL-Embedding-2B"],
    height=height,
    label="Qwen3-VL-Embedding-2B"
)



# Values

for i, row in df.reset_index(drop=True).iterrows():

    ax.text(
        row["OpenCLIP ViT-B/16"] + 0.015,
        i-height/2,
        f"{row['OpenCLIP ViT-B/16']:.3f}",
        va="center",
        fontsize=7
    )


    ax.text(
        row["Qwen3-VL-Embedding-2B"] + 0.015,
        i+height/2,
        f"{row['Qwen3-VL-Embedding-2B']:.3f}",
        va="center",
        fontsize=7
    )



# Axis

ax.set_yticks(y)

ax.set_yticklabels(
    df["held_out_trigger"],
    fontsize=9
)


ax.set_xlabel(
    "Unseen AUROC",
    fontsize=10
)


ax.set_xlim(
0,
1.05
)


# Grid

ax.grid(
axis="x",
alpha=0.25
)



# Legend outside

ax.legend(
frameon=False,
fontsize=9,
loc="upper left",
bbox_to_anchor=(1.01,1)
)



plt.tight_layout()



# =====================================================
# Save
# =====================================================

png = (
out_dir +
"/fig6_loto_generalization_final.png"
)


pdf = (
out_dir +
"/fig6_loto_generalization_final.pdf"
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
