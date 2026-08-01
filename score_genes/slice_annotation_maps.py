"""Combined 'slice{N} annotation' maps (LogReg) for slices 1,4,5,6 — backbone +
myeloid subtypes (all BAM) + tumor in black, matching the slice-1/3 annotation.

Output -> score_genes_slice{N}_merged/classify/slice{N}_annotation.png
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import (_read_X, _read_var_names, _read_obs_num,  # noqa: E402
                               _read_obs_bool, TUMOR_COL)
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)
import slice_all_lr_xgb as S  # noqa: E402  (reuse annotate, score_genes, BROAD, COL)

SLICES = [1, 4, 5, 6]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
GROUPS = S.GROUPS
COL = S.COL
TOPFRAC, FDR_CUTOFF, MARGIN_RATIO = 0.20, 0.05, 1.5
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def run(n):
    out = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
    os.makedirs(out, exist_ok=True)
    with h5py.File(TMPL.format(n), "r") as h5:
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
    nt = adata[~tumor].copy()
    cxnt, cynt = cx[~tumor], cy[~tumor]

    for lab in GROUPS:
        S.score_genes(nt, S.BROAD[lab], "score_" + lab)
    Sdf = nt.obs[["score_" + l for l in GROUPS]].copy()
    Sdf.columns = GROUPS
    thr = {l: mirrored_fdr_threshold(Sdf[l].to_numpy(), fdr=FDR_CUTOFF)[0] for l in GROUPS}
    res = scaled_margin_calls(Sdf, thr, ratio=MARGIN_RATIO)
    celltype = np.asarray(res["calls"])
    top_scaled = res["top_scaled"]
    train_mask = np.zeros(nt.n_obs, bool)
    for g in GROUPS:
        gm = celltype == g
        if gm.sum():
            train_mask |= gm & (top_scaled >= np.quantile(top_scaled[gm], 1 - TOPFRAC))

    Xnt = nt.X
    Xnt = (Xnt.toarray() if sp.issparse(Xnt) else np.asarray(Xnt)).astype(np.float32)
    combined = S.annotate("logreg", Xnt, Xnt[train_mask], celltype[train_mask],
                          train_mask, celltype, nt)
    counts = {g: int((combined == g).sum()) for g in COL}

    fig, ax = plt.subplots(figsize=(15, 10), dpi=180)
    m = combined == "unknown"
    ax.scatter(cxnt[m], cynt[m], s=1.2, c=COL["unknown"], linewidths=0,
               rasterized=True, label=f"unknown ({counts['unknown']:,})")
    for g in ["MDM", "Microglia", "Vascular", "Astrocytes", "Ependymal", "Neurons", "BAM"]:
        mm = combined == g
        ax.scatter(cxnt[mm], cynt[mm], s=1.6, c=COL[g], linewidths=0,
                   rasterized=True, label=f"{g} ({counts[g]:,})")
    ax.scatter(tx, ty, s=1.6, c="black", linewidths=0, rasterized=True,
               label=f"tumor ({len(tx):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice{n} annotation", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", markerscale=5, fontsize=9, frameon=True)
    fig.savefig(f"{out}/slice{n}_annotation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"slice {n}: {counts}  tumor={len(tx)} -> {out}/slice{n}_annotation.png")


def main():
    for n in SLICES:
        run(n)


if __name__ == "__main__":
    main()
