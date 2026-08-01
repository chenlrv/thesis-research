"""BAM cells with Lyve1 >= 3 (log-norm), coloured by Lyve1, with tumor in black.
Output -> score_genes_slice1_merged/classify/bam_lyve1_ge3.png
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
GENE, THRESH = "Lyve1", 3.0
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

    td = cKDTree(np.c_[tx, ty]).query(np.c_[bx[hi], by[hi]], k=1)[0]
    print(f"BAM: {len(bam):,}   Lyve1>= {THRESH}: {int(hi.sum()):,} "
          f"({100*hi.mean():.1f}%)   median dist-to-tumor={np.median(td):.1f}px")

    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    ax.scatter(cxnt, cynt, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
    ax.scatter(tx, ty, s=2, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    o = np.argsort(lyve[hi])  # high on top
    sc_ = ax.scatter(bx[hi][o], by[hi][o], s=18, c=lyve[hi][o], cmap="magma",
                     vmin=THRESH, vmax=np.quantile(lyve[hi], 0.99),
                     linewidths=0, rasterized=True)
    fig.colorbar(sc_, ax=ax, shrink=0.6, label="Lyve1 (log-norm)")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"BAM with Lyve1 >= {THRESH} ({int(hi.sum()):,}), tumor in black",
                 fontweight="bold")
    ax.legend(loc="lower right", markerscale=3, fontsize=9)
    fig.savefig(f"{OUT}/bam_lyve1_ge3.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved bam_lyve1_ge3.png -> {OUT}")


if __name__ == "__main__":
    main()
