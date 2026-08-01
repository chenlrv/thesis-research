"""Spatial map of the merged high-confidence set, one panel per type (small
multiples) + a combined panel. Fast: only needs coordinates, no PCA.
Output -> score_genes_slice1_merged/storyline/highconf_spatial.png
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/storyline"
COL = {"Myeloid": "#00a087", "Vascular": "#2ca02c", "Astrocytes": "#1f77b4",
       "Ependymal": "#984ea3", "Neurons": "#e377c2"}
ORDER = ["Myeloid", "Vascular", "Astrocytes", "Ependymal", "Neurons"]

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    os.makedirs(OUT, exist_ok=True)
    hc = pd.read_csv(HC)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    x, y = hc["x"].to_numpy(), hc["y"].to_numpy()
    lab = hc["provisional_label"].to_numpy()

    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    panels = ORDER + ["ALL high-confidence"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), dpi=150)
    axes = axes.ravel()
    for ax, title in zip(axes, panels):
        ax.scatter(xa[~tum], ya[~tum], s=0.5, c="#e8e8e8", linewidths=0, rasterized=True)
        ax.scatter(xa[tum], ya[tum], s=0.8, c="#f3c8c8", linewidths=0, rasterized=True)
        if title == "ALL high-confidence":
            for g in ORDER:
                m = high & (lab == g)
                ax.scatter(x[m], y[m], s=4, c=COL[g], linewidths=0, rasterized=True,
                           label=f"{g} ({int(m.sum())})")
            ax.legend(loc="lower right", markerscale=3, fontsize=9, frameon=True)
        else:
            m = high & (lab == title)
            ax.scatter(x[m], y[m], s=6, c=COL[title], linewidths=0, rasterized=True)
            ax.set_title(f"{title}  (n={int(m.sum())})", fontsize=14, fontweight="bold")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    axes[-1].set_title("ALL high-confidence", fontsize=14, fontweight="bold")
    fig.suptitle("Slice 1 — high-confidence cells, spatial (merged 5-type)",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/highconf_spatial.png", bbox_inches="tight")
    plt.close()
    print("saved", f"{OUT}/highconf_spatial.png")


if __name__ == "__main__":
    main()
