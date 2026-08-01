"""Spatial pattern of Lyve1 within the OvR-Myeloid BAM population.
Lyve1 was NOT used to call BAM (Mrc1/Cd163/Pf4 were), so it's an independent axis
- Lyve1+ BAMs are classically perivascular.

Panel A: BAM cells coloured by Lyve1 (log-norm) over grey tissue.
Panel B: Lyve1+ vs Lyve1- BAM, with tumor (black) and vessels (green) overlaid.

Output -> score_genes_slice1_merged/classify/bam_lyve1_spatial.png
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
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, SLICE, TUMOR_COL)

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
OVR = f"{OUT}/ovr_nontumor_predictions.csv"
GENE = "Lyve1"
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

    if GENE not in adata.var_names:
        raise SystemExit(f"{GENE} not in panel")
    expr = adata[:, GENE].X
    expr = (expr.toarray() if sp.issparse(expr) else np.asarray(expr)).ravel()

    coord_to_idx = {(round(float(a), 2), round(float(b), 2)): i
                    for i, (a, b) in enumerate(zip(cxnt, cynt))}

    sub = pd.read_csv(SUB)
    bam = sub[sub["subtype"] == "BAM"].reset_index(drop=True)
    idx = np.array([coord_to_idx.get((round(float(a), 2), round(float(b), 2)), -1)
                    for a, b in zip(bam["x"], bam["y"])])
    assert (idx >= 0).all(), "some BAM cells did not match"
    bx, by = bam["x"].to_numpy(), bam["y"].to_numpy()
    lyve = expr[idx]
    pos = lyve > 0

    ovr = pd.read_csv(OVR)
    vasc = ovr[ovr["final_label"] == "Vascular"]
    vasc_xy = np.c_[vasc["x"].to_numpy(), vasc["y"].to_numpy()]
    dist = cKDTree(vasc_xy).query(np.c_[bx, by], k=1)[0]

    print(f"BAM cells: {len(bam):,}")
    print(f"Lyve1+ (expr>0): {int(pos.sum()):,} ({100*pos.mean():.1f}%)  "
          f"mean Lyve1 in BAM={lyve.mean():.3f}")
    if pos.sum() > 5 and (~pos).sum() > 5:
        p = mannwhitneyu(dist[pos], dist[~pos], alternative="less")[1]
        print(f"dist-to-vessel: Lyve1+ median={np.median(dist[pos]):.1f}px  "
              f"Lyve1- median={np.median(dist[~pos]):.1f}px  (MWU p={p:.2e})")
    td = cKDTree(np.c_[tx, ty]).query(np.c_[bx, by], k=1)[0]
    print(f"dist-to-tumor: Lyve1+ median={np.median(td[pos]):.1f}px  "
          f"Lyve1- median={np.median(td[~pos]):.1f}px")

    # ================= figures =================
    fig, ax = plt.subplots(1, 2, figsize=(20, 9), dpi=160)
    # (a) BAM coloured by Lyve1
    ax[0].scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
    order = np.argsort(lyve)  # high on top
    sctr = ax[0].scatter(bx[order], by[order], s=6, c=lyve[order], cmap="magma",
                         vmax=np.quantile(lyve, 0.99) or 1, linewidths=0, rasterized=True)
    fig.colorbar(sctr, ax=ax[0], shrink=0.6, label="Lyve1 (log-norm)")
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title(f"(a) BAM coloured by Lyve1  (n={len(bam):,})", fontweight="bold")
    # (b) Lyve1+ vs Lyve1- with tumor + vessels
    ax[1].scatter(cxnt, cynt, s=0.5, c="#f0f0f0", linewidths=0, rasterized=True)
    ax[1].scatter(vasc_xy[:, 0], vasc_xy[:, 1], s=1.5, c="#2ca02c", alpha=0.4,
                  linewidths=0, rasterized=True, label=f"Vascular ({len(vasc_xy):,})")
    ax[1].scatter(tx, ty, s=1.5, c="black", linewidths=0, rasterized=True,
                  label=f"tumor ({len(tx):,})")
    ax[1].scatter(bx[~pos], by[~pos], s=4, c="#cccccc", linewidths=0,
                  rasterized=True, label=f"Lyve1- BAM ({int((~pos).sum()):,})")
    ax[1].scatter(bx[pos], by[pos], s=14, c="#d62728", linewidths=0,
                  rasterized=True, label=f"Lyve1+ BAM ({int(pos.sum()):,})")
    ax[1].set_aspect("equal"); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].set_title("(b) Lyve1+ BAM vs tumor / vessels", fontweight="bold")
    ax[1].legend(loc="lower right", markerscale=3, fontsize=8)
    fig.suptitle("Lyve1 within the BAM population", fontsize=15, fontweight="bold")
    fig.savefig(f"{OUT}/bam_lyve1_spatial.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved bam_lyve1_spatial.png -> {OUT}")


if __name__ == "__main__":
    main()
