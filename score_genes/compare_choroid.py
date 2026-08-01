"""Focused choroid-plexus comparison on slice 1: LogReg-backbone Ependymal vs
score_genes-v2 Ependymal, against the actual Adgrv1 (specific) / Ttr (choroid,
ambient) signal. Zoomed to the left choroid region (tumor is center-right, excluded).

Extracts only the two marker columns from the matrix then frees it (memory-safe).
Output -> score_genes_slice1_v2/choroid_compare.png  + printed precision/recall.
"""
import gc
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import anndata as ad  # noqa: F401  (kept for parity; not strictly needed)
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.sparse import csc_matrix, csr_matrix

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUT = "D:/thesis-research/score_genes_slice1_v2"
LR_CSV = "D:/thesis-research/score_genes_slice1_merged/classify/ovr_nontumor_predictions.csv"
SG_CSV = f"{OUT}/cell_scores.csv"
TUMOR_COL = "pred_tumor_XGBoost"
CROP_FRAC = 0.28   # left fraction of the x-range = choroid region
ADG_POS = 2        # Adgrv1 raw count >= this = "true" choroid/ependymal reference


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = str(node.attrs.get("encoding-type", ""))
        shape = tuple(node.attrs["shape"])
        data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
        M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
            else csr_matrix((data, idx, indptr), shape=shape)
        return M.tocsr()
    return csr_matrix(node[...])


def _read_var(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_num(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def main():
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var = list(_read_var(h5))
        cx = _read_num(h5, "CenterX_global_px")
        cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)

    keep = ~tumor
    X = X[keep]
    cx, cy = cx[keep], cy[keep]
    adg = np.asarray(X[:, var.index("Adgrv1")].todense()).ravel()
    ttr = np.asarray(X[:, var.index("Ttr")].todense()).ravel()
    del X
    gc.collect()
    print(f"non-tumor cells: {len(cx):,}")

    lr = pd.read_csv(LR_CSV)
    sg = pd.read_csv(SG_CSV)
    assert np.allclose(lr["x"].to_numpy(), cx, atol=1e-2), "LogReg CSV not aligned"
    assert np.allclose(sg["x"].to_numpy(), cx, atol=1e-2), "score_genes CSV not aligned"
    lr_ep = lr["final_label"].to_numpy() == "Ependymal"
    sg_ep = sg["celltype_v2"].to_numpy() == "Ependymal"

    # choroid crop = left CROP_FRAC of the x-range
    xmin, xmax = cx.min(), cx.max()
    crop = cx < xmin + CROP_FRAC * (xmax - xmin)
    adgpos = adg >= ADG_POS

    # ---- quantitative: precision/recall of choroid capture, in the crop ----
    ref = crop & adgpos              # Adgrv1+ cells in the choroid region (reference)
    print(f"\nchoroid crop (left {CROP_FRAC:.0%} of x): {int(crop.sum()):,} cells; "
          f"Adgrv1>={ADG_POS}: {int(ref.sum()):,} (reference)")
    for name, ep in [("LogReg   ", lr_ep), ("score_gen", sg_ep)]:
        e = crop & ep
        prec = 100 * (e & adgpos).sum() / max(e.sum(), 1)   # of its calls, % Adgrv1+
        rec = 100 * (e & adgpos).sum() / max(ref.sum(), 1)  # of Adgrv1+ ref, % captured
        print(f"  {name} ependymal in crop: {int(e.sum()):5,}  "
              f"precision(Adgrv1+): {prec:4.1f}%   recall(of ref): {rec:4.1f}%")

    # ---- plots (all zoomed to the crop) ----
    cxz, cyz = cx[crop], cy[crop]
    xlim = (cx[crop].min(), cx[crop].max())
    ylim = (cy[crop].min(), cy[crop].max())

    def base(ax):
        ax.scatter(cxz, cyz, s=1.2, c="#e6e6e6", linewidths=0, rasterized=True)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig, axes = plt.subplots(2, 3, figsize=(19, 12), dpi=140)

    # (0,0) full-tissue orientation with the crop box
    ax = axes[0, 0]
    ax.scatter(cx, cy, s=0.4, c="#e6e6e6", linewidths=0, rasterized=True)
    ax.scatter(cx[adgpos], cy[adgpos], s=3, c="#d62728", linewidths=0, rasterized=True,
               label=f"Adgrv1>={ADG_POS}")
    ax.add_patch(Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                           fill=False, ec="k", lw=1.5))
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("whole slice: Adgrv1+ & crop box"); ax.legend(loc="lower right", fontsize=8)

    # (0,1) Adgrv1 in crop
    ax = axes[0, 1]; base(ax)
    o = np.argsort(adg[crop])
    s = ax.scatter(cxz[o], cyz[o], s=6, c=np.log1p(adg[crop])[o], cmap="viridis",
                   linewidths=0, rasterized=True)
    fig.colorbar(s, ax=ax, shrink=0.7); ax.set_title("Adgrv1 (specific ependymal) log1p count")

    # (0,2) Ttr in crop
    ax = axes[0, 2]; base(ax)
    o = np.argsort(ttr[crop])
    s = ax.scatter(cxz[o], cyz[o], s=6, c=np.log1p(ttr[crop])[o], cmap="magma",
                   linewidths=0, rasterized=True)
    fig.colorbar(s, ax=ax, shrink=0.7); ax.set_title("Ttr (choroid, ambient) log1p count")

    # (1,0) LogReg ependymal
    ax = axes[1, 0]; base(ax)
    m = crop & lr_ep
    ax.scatter(cx[m], cy[m], s=7, c="#984ea3", linewidths=0, rasterized=True)
    ax.set_title(f"LogReg Ependymal (n={int(m.sum()):,} in crop)")

    # (1,1) score_genes ependymal
    ax = axes[1, 1]; base(ax)
    m = crop & sg_ep
    ax.scatter(cx[m], cy[m], s=7, c="#984ea3", linewidths=0, rasterized=True)
    ax.set_title(f"score_genes v2 Ependymal (n={int(m.sum()):,} in crop)")

    # (1,2) overlay + Adgrv1 reference
    ax = axes[1, 2]; base(ax)
    ax.scatter(cx[crop & adgpos], cy[crop & adgpos], s=14, facecolors="none",
               edgecolors="#bbbbbb", linewidths=0.6, rasterized=True, label=f"Adgrv1>={ADG_POS}")
    ax.scatter(cx[crop & lr_ep], cy[crop & lr_ep], s=6, c="#1f77b4", linewidths=0,
               rasterized=True, label="LogReg")
    ax.scatter(cx[crop & sg_ep], cy[crop & sg_ep], s=6, c="#d62728", linewidths=0,
               rasterized=True, alpha=0.6, label="score_genes")
    ax.set_title("overlay: LogReg vs score_genes vs Adgrv1")
    ax.legend(loc="lower right", fontsize=8, markerscale=1.5)

    fig.suptitle("slice_1 choroid-plexus region: ependymal calls vs Adgrv1/Ttr", fontsize=15)
    plt.tight_layout()
    fig.savefig(f"{OUT}/choroid_compare.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {OUT}/choroid_compare.png")


if __name__ == "__main__":
    main()
