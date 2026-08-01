"""Spatial plot: BAM (orange), tumor (light red), high-confidence Vascular only
(light green), over a faint tissue background. Fast — coordinates only.
Output -> score_genes_slice1_merged/bam/bam_tumor_vascular_spatial.png
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
BAM = "D:/thesis-research/score_genes_slice1_merged/bam/bam_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/bam"

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    # background + tumor from the h5ad (global coords)
    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    # high-conf Vascular only
    hc = pd.read_csv(HC)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    vasc = high & (hc["provisional_label"].to_numpy() == "Vascular")
    vx, vy = hc["x"].to_numpy()[vasc], hc["y"].to_numpy()[vasc]

    # BAM cells
    bam = pd.read_csv(BAM)
    bmask = bam["is_bam"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    bx, by = bam["x"].to_numpy()[bmask], bam["y"].to_numpy()[bmask]

    fig, ax = plt.subplots(figsize=(13, 12), dpi=160)
    ax.scatter(xa[~tum], ya[~tum], s=1.0, c="#bdbdbd", linewidths=0,
               rasterized=True)
    ax.scatter(xa[tum], ya[tum], s=2.0, c="#e41a1c", linewidths=0,
               rasterized=True, label=f"tumor ({int(tum.sum()):,})")
    ax.scatter(vx, vy, s=14, c="#1a9c3c", linewidths=0, rasterized=True,
               label=f"high-conf Vascular ({len(vx):,})")
    ax.scatter(bx, by, s=10, c="#ff7f0e", linewidths=0, rasterized=True,
               label=f"BAM ({len(bx):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", markerscale=3, fontsize=11, frameon=True)
    ax.set_title("Slice 1 — BAM, tumor, and high-confidence Vascular",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = f"{OUT}/bam_tumor_vascular_spatial.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"BAM={len(bx)}  tumor={int(tum.sum())}  high-conf Vascular={len(vx)}")
    print("saved", out)


if __name__ == "__main__":
    main()
