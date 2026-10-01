import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = Path(
    os.environ.get(
        "TAMVLM_RESULTS_ROOT",
        PROJECT_ROOT / "results"
    )
)

openclip_csv = RESULTS_ROOT / "openclip_results" / "strong_baselines.csv"

qwen_csv = RESULTS_ROOT / "qwen_results" / "crossmodel_baselines.csv"

out_dir = PROJECT_ROOT / "figure_generation" / "figures"
os.makedirs(out_dir, exist_ok=True)


def prepare(csv, backbone):

    df = pd.read_csv(csv)

    df = df[["Method", "AUROC"]]

    mapping = {
        "Euclidean distance": "Euclidean",
        "Cosine distance": "Cosine",
        "Mahalanobis distance": "Mahalanobis",
        "kNN distance (k=10)": "kNN",
        "Linear probe (CLIP)": "Linear probe (OpenCLIP)",
        "Linear probe (Qwen3-VL)": "Linear probe (Qwen3-VL)"
    }

    df["Method"] = df["Method"].replace(mapping)

    return df


openclip = prepare(openclip_csv, "OpenCLIP")
qwen = prepare(qwen_csv, "Qwen")


# Add final TAM-VLM mean results
openclip = pd.concat([
    pd.DataFrame({
        "Method":["TAM-VLM"],
        "AUROC":[0.962]
    }),
    openclip
])


qwen = pd.concat([
    pd.DataFrame({
        "Method":["TAM-VLM"],
        "AUROC":[0.948]
    }),
    qwen
])


fig, axes = plt.subplots(
    1,2,
    figsize=(10,5)
)


for ax, df, title in zip(
    axes,
    [openclip,qwen],
    ["OpenCLIP ViT-B/16",
     "Qwen3-VL-Embedding-2B"]
):

    df=df.sort_values("AUROC")

    bars=ax.barh(
        df["Method"],
        df["AUROC"]
    )


    for bar, method in zip(bars, df["Method"]):
        if method=="TAM-VLM":
            bar.set_alpha(1.0)
            bar.set_linewidth(1.5)
        else:
            bar.set_alpha(0.55)


    ax.set_xlim(0,1)

    ax.set_xlabel(
        "AUROC"
    )

    ax.set_title(
        title,
        fontsize=11
    )

    for i,v in enumerate(df["AUROC"]):
        ax.text(
            v+0.01,
            i,
            f"{v:.3f}",
            va="center",
            fontsize=8
        )

    ax.grid(
        axis="x",
        alpha=0.25
    )


plt.tight_layout()


plt.savefig(
    out_dir+"/fig5_baseline_comparison_final.png",
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    out_dir+"/fig5_baseline_comparison_final.pdf",
    bbox_inches="tight"
)


print("Figure 5 final saved")
