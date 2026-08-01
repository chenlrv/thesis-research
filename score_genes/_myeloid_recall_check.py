"""Is the SMALL absolute MDM/BAM count driven by Stage-1 under-calling myeloid?
Compare the Stage-1 v2 pan-myeloid count to inclusive detection-based myeloid
counts (>=k of the 7 pan-myeloid genes) among non-tumor cells."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]


def _decode(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _read_X(h5):
    node = h5["X"]; enc = str(node.attrs.get("encoding-type", "")); shape = tuple(node.attrs["shape"])
    args = (node["data"][...], node["indices"][...], node["indptr"][...])
    return (csc_matrix(args, shape=shape) if "csc" in enc else csr_matrix(args, shape=shape)).tocsr()


def _read_var(h5):
    var = h5["var"]; k = var.attrs.get("_index", "_index")
    return list(_decode(var[k.decode() if isinstance(k, bytes) else k][...]))


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]; cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    return (np.isin(_decode(arr).astype(str), ["True", "1"]) if arr.dtype.kind in ("S", "O")
            else arr.astype(bool))


print(f"{'sl':>2} {'non-tumor':>9} {'stage1_mye':>10} {'%':>5} | "
      f"{'det>=2':>7} {'%':>5} {'det>=3':>7} {'%':>5}")
for n in [1, 2, 3]:
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = _read_var(h5); tumor = _read_bool(h5, "pred_tumor_XGBoost")
    keep = ~tumor
    Xk = X[keep]
    cols = [var.index(g) for g in PAN if g in var]
    det = np.asarray((Xk[:, cols] > 0).sum(1)).ravel()
    nt = int(keep.sum())
    s1 = int((pd.read_csv(f"D:/thesis-research/score_genes_slice{n}_v2/cell_scores.csv")
              ["celltype_v2"] == "Myeloid").sum())
    d2 = int((det >= 2).sum()); d3 = int((det >= 3).sum())
    print(f"{n:>2} {nt:>9,} {s1:>10,} {100*s1/nt:>4.1f}% | "
          f"{d2:>7,} {100*d2/nt:>4.1f}% {d3:>7,} {100*d3/nt:>4.1f}%")
