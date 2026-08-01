"""Is Ttr>=5 a reasonable choroid threshold across ALL slices, or is it batch-
dependent? For each slice: Ttr percentiles (non-tumor), the per-slice 2-component
GMM threshold on log1p(Ttr>0), and cell counts at the fixed 5 vs the per-slice GMM.
Slides: L321 = slices 1-3, L34 = slices 4-6.
"""
import gc
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.mixture import GaussianMixture

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


def gmm_thr(ttr):
    lt = np.log1p(ttr[ttr > 0]).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(lt)
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(0, lt.max(), 2000).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, hi]
    thr_log = float(grid[np.argmax(post >= 0.5), 0])
    return np.expm1(thr_log), float(np.expm1(gm.means_.ravel()[hi]))


def main():
    print(f"{'sl':>2} {'slide':>5} {'n_nt':>8} {'p95':>4} {'p99':>5} {'p99.9':>6} "
          f"{'gmm_thr':>8} {'hi_mean':>8} {'n>=5':>7} {'n>=gmm':>7}")
    for sid in range(1, 7):
        path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
        if not os.path.exists(path):
            print(f"{sid:>2}  NOT FOUND")
            continue
        with h5py.File(path, "r") as h5:
            X = _read_X(h5)
            var = list(_read_var(h5))
            tumor = _read_bool(h5, TUMOR_COL)
        ttr = np.asarray(X[~tumor][:, var.index("Ttr")].todense()).ravel()
        del X
        gc.collect()
        thr, himean = gmm_thr(ttr)
        p95, p99, p999 = np.percentile(ttr, [95, 99, 99.9])
        print(f"{sid:>2} {SLIDE[sid]:>5} {len(ttr):>8,} {p95:>4.0f} {p99:>5.0f} "
              f"{p999:>6.0f} {thr:>8.1f} {himean:>8.0f} "
              f"{int((ttr>=5).sum()):>7,} {int((ttr>=thr).sum()):>7,}")


if __name__ == "__main__":
    main()
