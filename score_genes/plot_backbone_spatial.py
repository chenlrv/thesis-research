"""Spatial maps of the Method-2/20% LogReg propagation (final_label):
  * one panel per type (type coloured over a grey tissue background)
  * one combined map with all types.
Reads slice1_backbone_predictions.csv (non-tumor cells, global-px coords).

Output -> score_genes_slice1_merged/classify/spatial_*.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
PRED = f"{OUT}/slice1_backbone_predictions.csv"
GROUPS = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]
PANELS = GROUPS + ["uncertain"]
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2", "uncertain": "#bbbbbb"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    df = pd.read_csv(PRED)
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    lab = df["final_label"].to_numpy()
    print(f"cells: {len(df):,}")
    print(pd.Series(lab).value_counts().to_string())

    # ---- per-type grid (5 types + uncertain) ----
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), dpi=160)
    for ax, g in zip(axes.ravel(), PANELS):
        m = lab == g
        ax.scatter(x, y, s=0.6, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(x[m], y[m], s=1.6, c=COL[g], linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={int(m.sum()):,})", fontweight="bold")
    fig.suptitle("Slice 1 backbone propagation — per type (Method-2/20% LogReg)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/spatial_per_type.png", bbox_inches="tight")
    plt.close(fig)

    # ---- combined map (all types) ----
    fig, ax = plt.subplots(figsize=(11, 10), dpi=180)
    for g in ["uncertain"] + GROUPS:   # uncertain first so types draw on top
        m = lab == g
        if m.any():
            ax.scatter(x[m], y[m], s=1.2, c=COL[g], linewidths=0,
                       rasterized=True, label=f"{g} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Slice 1 backbone propagation — all types", fontweight="bold")
    ax.legend(loc="lower right", markerscale=6, fontsize=9, frameon=True)
    fig.savefig(f"{OUT}/spatial_all_types.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved spatial_per_type.png, spatial_all_types.png -> {OUT}")


if __name__ == "__main__":
    main()
