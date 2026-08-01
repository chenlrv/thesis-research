"""QC to explain slice 6's missing choroid: is it a whole-slice technical dropout
or Ttr-specific (no choroid in section / Ttr probe issue)?

Per slice (non-tumor): library size, genes/cell, and Ttr detection stats.
If slice 6 has a normal library size but low Ttr -> Ttr-specific/biology.
If its library size is also low -> global technical degradation.
"""
import gc
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

TUMOR_COL = "pred_tumor_XGBoost"
SLIDE = {1: "L321", 2: "L321", 3: "L321", 4: "L34", 5: "L34", 6: "L34"}


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
    print(f"{'sl':>2} {'slide':>5} {'n_nt':>8} {'med_counts':>10} {'med_genes':>9} "
          f"{'%Ttr>0':>7} {'Ttr_mean':>8} {'Ttr_max':>7} {'n_Ttr>=5':>8}")
    for sid in range(1, 7):
        path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
        with h5py.File(path, "r") as h5:
            X = _read_X(h5)
            var = list(_read_var(h5))
            tumor = _read_bool(h5, TUMOR_COL)
        Xk = X[~tumor]
        del X
        lib = np.asarray(Xk.sum(axis=1)).ravel()
        ngenes = np.diff(Xk.tocsr().indptr)
        ttr = np.asarray(Xk[:, var.index("Ttr")].todense()).ravel()
        del Xk
        gc.collect()
        print(f"{sid:>2} {SLIDE[sid]:>5} {len(lib):>8,} {np.median(lib):>10.0f} "
              f"{np.median(ngenes):>9.0f} {100*(ttr>0).mean():>7.2f} {ttr.mean():>8.3f} "
              f"{int(ttr.max()):>7} {int((ttr>=5).sum()):>8,}")


if __name__ == "__main__":
    main()
