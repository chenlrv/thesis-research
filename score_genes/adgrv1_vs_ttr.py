"""Does Adgrv1 co-mark the Ttr>=5 choroid cells, or a separate population?
Prints the Adgrv1 distribution inside Ttr>=5 vs Ttr<5, the correlation, and how
well each Adgrv1 threshold matches the Ttr>=5 set (precision/recall)."""
import gc
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import spearmanr

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
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
        tumor = _read_bool(h5, TUMOR_COL)
    X = X[~tumor]
    ttr = np.asarray(X[:, var.index("Ttr")].todense()).ravel()
    adg = np.asarray(X[:, var.index("Adgrv1")].todense()).ravel()
    del X
    gc.collect()

    hi = ttr >= TTR_HI
    print(f"non-tumor: {len(ttr):,}   Ttr>={TTR_HI}: {int(hi.sum()):,}\n")

    def dist(name, a):
        pct = ", ".join(f"p{p}={np.percentile(a, p):.0f}" for p in [50, 75, 90, 95, 99])
        print(f"  Adgrv1 | {name:12s} (n={len(a):6,}):  mean={a.mean():.2f}  {pct}  "
              f"| %>=1={100*(a>=1).mean():.1f}  %>=2={100*(a>=2).mean():.1f}  "
              f"%>=3={100*(a>=3).mean():.1f}")

    print("Adgrv1 distribution:")
    dist(f"Ttr>={TTR_HI}", adg[hi])
    dist(f"Ttr<{TTR_HI}", adg[~hi])

    rho, _ = spearmanr(ttr, adg)
    print(f"\nSpearman corr(Ttr, Adgrv1) over all non-tumor: {rho:.3f}")

    print(f"\nHow well does an Adgrv1 cut match the Ttr>={TTR_HI} set?")
    print(f"  {'Adgrv1>=k':10s} {'n':>8s} {'precision':>10s} {'recall':>8s}")
    for k in [1, 2, 3, 5]:
        m = adg >= k
        prec = 100 * (m & hi).sum() / max(m.sum(), 1)
        rec = 100 * (m & hi).sum() / max(hi.sum(), 1)
        print(f"  {'>= '+str(k):10s} {int(m.sum()):8,} {prec:9.1f}% {rec:7.1f}%")

    for k in [1, 2]:
        both = hi & (adg >= k)
        print(f"\nTtr>={TTR_HI} AND Adgrv1>={k}: {int(both.sum()):,}  "
              f"(keeps {100*both.sum()/max(hi.sum(),1):.1f}% of the {int(hi.sum()):,}-cell choroid)")


if __name__ == "__main__":
    main()
