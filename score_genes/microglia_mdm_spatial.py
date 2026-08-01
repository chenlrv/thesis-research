"""Spatial maps of the confident Microglia and MDM calls (non-BAM myeloid).
Small multiples: Microglia | MDM | both, each over tissue + tumor.
Output -> score_genes_slice1_merged/microglia_mdm/microglia_mdm_spatial.png
"""
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

LAB = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm/microglia_mdm_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm"
COL = {"Microglia": "#17becf", "MDM": "#ff7f0e"}

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    df = pd.read_csv(LAB)
    call = df["call"].to_numpy()
    x, y = df["x"].to_numpy(), df["y"].to_numpy()

    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    panels = ["Microglia", "MDM", "both"]
    fig, axes = plt.subplots(1, 3, figsize=(24, 9), dpi=160)
    for ax, p in zip(axes, panels):
        ax.scatter(xa[~tum], ya[~tum], s=1.0, c="#bdbdbd", linewidths=0,
                   rasterized=True)
        ax.scatter(xa[tum], ya[tum], s=1.6, c="#e41a1c", linewidths=0,
                   rasterized=True, label=f"tumor ({int(tum.sum()):,})")
        groups = ["Microglia", "MDM"] if p == "both" else [p]
        for g in groups:
            m = call == g
            ax.scatter(x[m], y[m], s=12, c=COL[g], linewidths=0, rasterized=True,
                       label=f"{g} ({int(m.sum())})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="lower right", markerscale=2.5, fontsize=10, frameon=True)
        ax.set_title(p if p != "both" else "Microglia + MDM",
                     fontsize=14, fontweight="bold")
    fig.suptitle("Slice 1 — confident Microglia and MDM calls (non-BAM myeloid)",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{OUT}/microglia_mdm_spatial.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Microglia={int((call=='Microglia').sum())}  "
          f"MDM={int((call=='MDM').sum())}  tumor={int(tum.sum())}")
    print("saved", out)


if __name__ == "__main__":
    main()
