"""Per-type spatial maps for the slice-3 annotation (backbone + myeloid subtypes):
one panel per type over a grey tissue background.

Output -> score_genes_slice3_merged/classify/slice3_per_type.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "D:/thesis-research/score_genes_slice3_merged/classify"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
PANELS = ["BAM", "MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal",
          "Neurons", "unknown"]
COL = {"unknown": "#999999", "MDM": "#00a087", "Microglia": "#17becf",
       "Vascular": "#2ca02c", "Astrocytes": "#1f77b4", "Ependymal": "#984ea3",
       "Neurons": "#e377c2", "BAM": "#d62728"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    ovr = pd.read_csv(OVR)
    x, y = ovr["x"].to_numpy(), ovr["y"].to_numpy()
    combined = ovr["final_label"].to_numpy().astype(object)
    coord = {(round(float(a), 2), round(float(b), 2)): i
             for i, (a, b) in enumerate(zip(x, y))}

    sub = pd.read_csv(SUB)
    for _, r in sub.iterrows():
        i = coord[(round(float(r["x"]), 2), round(float(r["y"]), 2))]
        combined[i] = r["subtype"]  # BAM / MDM / Microglia / unknown

    fig, axes = plt.subplots(2, 4, figsize=(26, 11), dpi=150)
    for ax, g in zip(axes.ravel(), PANELS):
        m = combined == g
        ax.scatter(x, y, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=2.5, c=COL[g], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={int(m.sum()):,})", fontweight="bold")
    fig.suptitle("slice3 annotation — per type (grey = all non-tumor)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/slice3_per_type.png", bbox_inches="tight")
    plt.close(fig)
    print("counts:", {g: int((combined == g).sum()) for g in PANELS})
    print(f"saved slice3_per_type.png -> {OUT}")


if __name__ == "__main__":
    main()
