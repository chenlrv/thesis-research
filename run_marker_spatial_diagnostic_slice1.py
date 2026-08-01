"""Diagnostic: do canonical markers show mouse-brain anatomy in slice_1?

Before trusting any clustering/labeling we must confirm the *raw* signal has
spatial structure. Plot normalized expression of canonical lineage markers in
spatial coordinates. Brain expectations:
  - total counts: outlines tissue anatomy (denser gray matter, sparser WM)
  - GFAP: white-matter tracts + glia limitans (brain surface) + any tumor rim
  - Cx3cr1 / TMEM119 / C1qa: microglia tiling the parenchyma ~uniformly
  - Mrc1 / Lyve1 / Cd163: perivascular/meningeal (border) pattern, sparse
  - Ccr2 / Plac8: monocyte-derived, should track tumor/vessels if present
  - Pecam1: linear/vascular pattern
  - tumor mask: should it be a focal mass or scattered?
If even GFAP is spatially flat, the data/slice can't support this annotation.
"""
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUT = "D:/thesis-research/nontumor_slice1/marker_spatial_diagnostic.png"
TUMOR_PRED_COLS = ["pred_tumor_LogReg", "pred_tumor_LogReg_PCA",
                   "pred_tumor_LogReg_KNN_PCA", "pred_tumor_XGBoost"]

MARKERS = ["__counts__", "__tumor__",
           "GFAP", "Sparcl1", "Cx3cr1", "TMEM119", "C1qa", "Csf1r",
           "Mrc1", "Lyve1", "Cd163", "Ccr2", "Plac8", "Pecam1", "Cd3e", "Vwf"]


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    enc = str(node.attrs.get("encoding-type", ""))
    shape = tuple(node.attrs["shape"])
    data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
    M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
        else csr_matrix((data, idx, indptr), shape=shape)
    return M.tocsr()


def _read_obs_num(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var = h5["var"]
        key = var.attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else key
        var_names = _decode(var[key][...])
        # global tissue coords (obsm['spatial'] is FOV-local junk here)
        coords = np.column_stack([_read_obs_num(h5, "CenterX_global_px"),
                                  _read_obs_num(h5, "CenterY_global_px")])
        preds = np.vstack([h5["obs"][c][...].astype(bool) for c in TUMOR_PRED_COLS]).sum(0)
    tumor = preds >= 2

    adata = ad.AnnData(X=X)
    adata.var_names = pd.Index(var_names)
    adata.var_names_make_unique()
    counts_total = np.asarray(adata.X.sum(1)).ravel()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    x, y = coords[:, 0], -coords[:, 1]
    lut = {v.lower(): v for v in adata.var_names}

    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=140)
    for ax, m in zip(axes.ravel(), MARKERS):
        if m == "__counts__":
            val = np.log1p(counts_total); title = "log total counts (anatomy)"
        elif m == "__tumor__":
            ax.scatter(x, y, s=0.5, c="0.9", linewidths=0, rasterized=True)
            ax.scatter(x[tumor], y[tumor], s=1.5, c="red", linewidths=0, rasterized=True)
            ax.set_title(f"tumor mask (n={int(tumor.sum())}, {100*tumor.mean():.1f}%)")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); continue
        else:
            gene = lut.get(m.lower())
            if gene is None:
                ax.set_title(f"{m} (absent)"); ax.axis("off"); continue
            val = np.asarray(adata[:, gene].X.todense()).ravel()
            title = f"{gene}  (pos {100*(val>0).mean():.1f}%)"
        order = np.argsort(val)  # plot high-expressing on top
        vmax = np.quantile(val, 0.995) or 1.0
        ax.scatter(x[order], y[order], s=0.6, c=val[order], cmap="magma",
                   vmin=0, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("slice_1 — canonical marker expression in space", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight"); plt.close()
    print("saved", OUT)

    # quick numeric: positivity rate of each marker
    print("\nmarker positivity (fraction of cells with >0 normalized expr):")
    for m in MARKERS:
        if m.startswith("__"):
            continue
        g = lut.get(m.lower())
        if g is None:
            print(f"  {m}: absent"); continue
        v = np.asarray(adata[:, g].X.todense()).ravel()
        print(f"  {g:10s}: {100*(v>0).mean():5.1f}%  mean(norm)={v.mean():.3f}")


if __name__ == "__main__":
    main()
