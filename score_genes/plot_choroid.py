"""Clean full-tissue spatial map of the proposed Choroid = Ttr>=5 call.
Usage: python plot_choroid.py <slice_id>"""
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

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
OUT = f"D:/thesis-research/score_genes_slice{SLICE_ID}_v2"
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
    del X
    gc.collect()
    ch = ttr >= TTR_HI

    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    ax.scatter(cx, cy, s=1.0, c="#e6e6e6", linewidths=0, rasterized=True,
               label="other non-tumor")
    ax.scatter(cx[ch], cy[ch], s=7, c="#d1495b", linewidths=0, rasterized=True,
               label=f"Choroid (Ttr>={TTR_HI})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_{SLICE_ID}  Choroid = Ttr>={TTR_HI}  "
                 f"(n={int(ch.sum()):,}, {100*ch.mean():.2f}% of non-tumor)")
    ax.legend(loc="lower right", markerscale=3, fontsize=10, frameon=True)
    plt.tight_layout()
    fig.savefig(f"{OUT}/choroid_spatial.png", bbox_inches="tight")
    plt.close(fig)
    print(f"slice {SLICE_ID}: choroid (Ttr>={TTR_HI}) = {int(ch.sum()):,} cells "
          f"-> {OUT}/choroid_spatial.png")


if __name__ == "__main__":
    main()
