"""Spatial plot of BAM cells with high Lyve1 (log-norm >= THRESH), over tumor +
vessels. Output -> score_genes_slice1_merged/classify/bam_lyve1_hi.png
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
GENE, THRESH = "Lyve1", 2.0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    tx, ty = cx[tumor], cy[tumor]
    adata = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]
    expr = adata[:, GENE].X
    expr = (expr.toarray() if sp.issparse(expr) else np.asarray(expr)).ravel()
    coord = {(round(float(a), 2), round(float(b), 2)): i
             for i, (a, b) in enumerate(zip(cxnt, cynt))}

    bam = pd.read_csv(SUB)
    bam = bam[bam["subtype"] == "BAM"].reset_index(drop=True)
    idx = np.array([coord[(round(float(a), 2), round(float(b), 2))]
                    for a, b in zip(bam["x"], bam["y"])])
    bx, by, lyve = bam["x"].to_numpy(), bam["y"].to_numpy(), expr[idx]
    hi = lyve >= THRESH

    ovr = pd.read_csv(OVR)
    vasc = ovr[ovr["final_label"] == "Vascular"]
    vxy = np.c_[vasc["x"].to_numpy(), vasc["y"].to_numpy()]
    td = cKDTree(np.c_[tx, ty]).query(np.c_[bx, by], k=1)[0]
    print(f"BAM: {len(bam):,}   Lyve1>= {THRESH}: {int(hi.sum()):,} "
          f"({100*hi.mean():.1f}%)")
    print(f"dist-to-tumor: Lyve1>={THRESH} median={np.median(td[hi]):.1f}px  "
          f"rest BAM median={np.median(td[~hi]):.1f}px")

    fig, ax = plt.subplots(figsize=(11, 10), dpi=180)
    ax.scatter(cxnt, cynt, s=0.5, c="#f0f0f0", linewidths=0, rasterized=True)
    ax.scatter(vxy[:, 0], vxy[:, 1], s=1.2, c="#2ca02c", alpha=0.35,
               linewidths=0, rasterized=True, label=f"Vascular ({len(vxy):,})")
    ax.scatter(tx, ty, s=1.5, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    ax.scatter(bx[~hi], by[~hi], s=3, c="#cccccc", linewidths=0, rasterized=True,
               label=f"BAM Lyve1<{THRESH} ({int((~hi).sum()):,})")
    ax.scatter(bx[hi], by[hi], s=22, c="#d62728", linewidths=0, rasterized=True,
               label=f"BAM Lyve1>={THRESH} ({int(hi.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"BAM with Lyve1 >= {THRESH} (log-norm)", fontweight="bold")
    ax.legend(loc="lower right", markerscale=3, fontsize=8)
    fig.savefig(f"{OUT}/bam_lyve1_hi.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved bam_lyve1_hi.png -> {OUT}")


if __name__ == "__main__":
    main()
