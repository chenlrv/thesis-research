"""Choroid annotation (Choroid = Ttr>=5, on non-tumor cells) spatial map for ALL
slices. Tumor cells (pred_tumor_XGBoost) marked black. One PNG per slice under
score_genes_v3/.  Title: "slice {id} Choroid annotation"; the choroid count is in
the legend label.

Runs all slices by default; pass a slice id to do just one (useful if a big slice
OOMs and needs a solo retry).
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
from scipy.sparse import csc_matrix, csr_matrix

OUT = "D:/thesis-research/score_genes_v3"
TUMOR_COL = "pred_tumor_XGBoost"
TTR_HI = 5


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


def do_slice(sid):
    path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
    if not os.path.exists(path):
        print(f"slice {sid}: NOT FOUND ({path})")
        return
    with h5py.File(path, "r") as h5:
        X = _read_X(h5)
        var = list(_read_var(h5))
        cx = _read_num(h5, "CenterX_global_px")
        cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)
    ttr = np.asarray(X[:, var.index("Ttr")].todense()).ravel()
    del X
    gc.collect()

    nt = ~tumor
    choroid = nt & (ttr >= TTR_HI)          # choroid = non-tumor, high Ttr
    other = nt & ~choroid
    n_ch = int(choroid.sum())
    n_tu = int(tumor.sum())

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    ax.scatter(cx[other], cy[other], s=1.0, c="#e2e2e2", linewidths=0, rasterized=True,
               label="other non-tumor")
    ax.scatter(cx[tumor], cy[tumor], s=1.4, c="black", linewidths=0, rasterized=True,
               label=f"tumor (n={n_tu:,})")
    ax.scatter(cx[choroid], cy[choroid], s=7, c="#d1495b", linewidths=0, rasterized=True,
               label=f"Choroid (n={n_ch:,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {sid} Choroid annotation")
    ax.legend(loc="lower right", markerscale=3, fontsize=10, frameon=True)
    plt.tight_layout()
    fig.savefig(f"{OUT}/slice_{sid}_choroid_annotation.png", bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    print(f"slice {sid}: choroid={n_ch:,}, tumor={n_tu:,} "
          f"-> {OUT}/slice_{sid}_choroid_annotation.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    slices = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2, 3, 4, 5, 6]
    for sid in slices:
        try:
            do_slice(sid)
        except MemoryError:
            print(f"slice {sid}: MemoryError -- retry solo: python plot_choroid_all.py {sid}")


if __name__ == "__main__":
    main()
