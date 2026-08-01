"""Show that HIGH Ttr isolates the choroid plexus cleanly (vs the ambient low-Ttr
tail). Data-driven threshold via a 2-component GMM on log1p(Ttr) over Ttr>0 cells.

Also splits the score_genes-v2 Ependymal calls into high-Ttr (true choroid) vs
low-Ttr (ambient false-positive) to quantify how much of the over-call is ambient.

Output -> score_genes_slice1_v2/choroid_ttr_gate.png  + printed counts.
"""
import gc
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.mixture import GaussianMixture

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
OUT = f"D:/thesis-research/score_genes_slice{SLICE_ID}_v2"
SG_CSV = f"{OUT}/cell_scores.csv"
TUMOR_COL = "pred_tumor_XGBoost"


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
    ttr = np.asarray(X[:, var.index("Ttr")].todense()).ravel()
    adg = np.asarray(X[:, var.index("Adgrv1")].todense()).ravel()
    del X
    gc.collect()

    # ---- data-driven high-Ttr threshold: 2-comp GMM on log1p(Ttr) over Ttr>0 ----
    pos = ttr > 0
    lt = np.log1p(ttr[pos]).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(lt)
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(0, lt.max(), 2000).reshape(-1, 1)
    post_hi = gm.predict_proba(grid)[:, hi]
    thr_log = float(grid[np.argmax(post_hi >= 0.5), 0])
    thr_raw = np.expm1(thr_log)
    choroid = ttr >= thr_raw

    print("Ttr raw-count percentiles (all non-tumor): "
          + ", ".join(f"p{p}={np.percentile(ttr, p):.0f}" for p in [50, 90, 95, 99, 99.9]))
    print(f"GMM high-Ttr threshold: raw >= {thr_raw:.1f}  (log1p {thr_log:.2f})")
    print(f"HIGH-Ttr (choroid) cells: {int(choroid.sum()):,}  "
          f"({100*choroid.mean():.2f}% of non-tumor)")

    # ---- how much of the score_genes Ependymal over-call is ambient? ----
    sg = pd.read_csv(SG_CSV)
    assert np.allclose(sg["x"].to_numpy(), cx, atol=1e-2)
    sg_ep = sg["celltype_v2"].to_numpy() == "Ependymal"
    n_ep = int(sg_ep.sum())
    hi_ep = int((sg_ep & choroid).sum())
    print(f"\nscore_genes v2 Ependymal: {n_ep:,}")
    print(f"  high-Ttr (real choroid):        {hi_ep:,} ({100*hi_ep/n_ep:.1f}%)")
    print(f"  low-Ttr (ambient false-pos):    {n_ep-hi_ep:,} ({100*(n_ep-hi_ep)/n_ep:.1f}%)")
    print(f"choroid cells captured by score_genes Ependymal: "
          f"{hi_ep:,}/{int(choroid.sum()):,} ({100*hi_ep/max(int(choroid.sum()),1):.1f}%)")

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=140)
    for ax in axes:
        ax.scatter(cx, cy, s=0.4, c="#e8e8e8", linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    axes[0].scatter(cx[choroid], cy[choroid], s=6, c="#d1495b", linewidths=0,
                    rasterized=True, label=f"Ttr>= {thr_raw:.0f} (choroid)")
    axes[0].scatter(cx[adg >= 2], cy[adg >= 2], s=6, c="#1f77b4", linewidths=0,
                    rasterized=True, alpha=0.6, label="Adgrv1>=2 (ependymal lining)")
    axes[0].legend(loc="lower right", fontsize=9, markerscale=2)
    axes[0].set_title(f"HIGH-Ttr choroid ({int(choroid.sum()):,}) + Adgrv1 lining")

    # score_genes ependymal split by Ttr
    axes[1].scatter(cx[sg_ep & choroid], cy[sg_ep & choroid], s=6, c="#2ca02c",
                    linewidths=0, rasterized=True, label=f"high-Ttr ({hi_ep:,})")
    axes[1].scatter(cx[sg_ep & ~choroid], cy[sg_ep & ~choroid], s=6, c="#ff7f0e",
                    linewidths=0, rasterized=True, alpha=0.6, label=f"low-Ttr ({n_ep-hi_ep:,})")
    axes[1].legend(loc="lower right", fontsize=9, markerscale=2)
    axes[1].set_title("score_genes Ependymal split: real choroid vs ambient")

    # zoom on the choroid (left region)
    xmin, xmax = cx.min(), cx.max()
    crop = cx < xmin + 0.28 * (xmax - xmin)
    axes[2].set_xlim(cx[crop].min(), cx[crop].max())
    axes[2].set_ylim(cy[crop].min(), cy[crop].max())
    axes[2].scatter(cx[choroid], cy[choroid], s=14, c="#d1495b", linewidths=0,
                    rasterized=True)
    axes[2].set_title("zoom: HIGH-Ttr choroid frond")

    fig.suptitle(f"slice_{SLICE_ID}: HIGH-Ttr gate isolates the choroid plexus", fontsize=15)
    plt.tight_layout()
    fig.savefig(f"{OUT}/choroid_ttr_gate.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {OUT}/choroid_ttr_gate.png")


if __name__ == "__main__":
    main()
