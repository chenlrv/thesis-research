"""Do MDM/BAM markers actually rise in TUMOR myeloid vs CONTROL myeloid?
If tumor >> control -> markers work, gate mis-calibrated. If tumor ~ control ->
little infiltration signal on this panel. Slices 1,2 (tumor) vs 3 (control), on
the Stage-1 v2 myeloid cells. Mean raw count + % detected per marker."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
V2 = "D:/thesis-research/score_genes_slice{}_v2/cell_scores.csv"
GENES = {"micro": ["Adgrg1", "Hpgds", "Gpr183", "Cx3cr1"],
         "bam": ["Mrc1", "Cd163", "Pf4", "Maf", "Cd5l"],
         "mdm_core": ["Ccr2", "Plac8", "Cd14", "Ccr1", "Clec12a", "Fpr1", "Crip1"],
         "mdm_extra": ["S100a8", "S100a9", "Vcan", "Sell", "Fn1", "Itgal"]}


def _decode(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _read_X(h5):
    node = h5["X"]
    enc = str(node.attrs.get("encoding-type", ""))
    shape = tuple(node.attrs["shape"])
    M = csc_matrix((node["data"][...], node["indices"][...], node["indptr"][...]), shape=shape) \
        if "csc" in enc else csr_matrix((node["data"][...], node["indices"][...], node["indptr"][...]), shape=shape)
    return M.tocsr()


def _read_var(h5):
    var = h5["var"]; key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return list(_decode(var[key][...]))


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]; cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    return (np.isin(_decode(arr).astype(str), ["True", "1"]) if arr.dtype.kind in ("S", "O")
            else arr.astype(bool))


def mye(n):
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = _read_var(h5); tumor = _read_bool(h5, "pred_tumor_XGBoost")
    keep = ~tumor
    df = pd.read_csv(V2.format(n))
    mm = df["celltype_v2"].to_numpy() == "Myeloid"
    return X[keep][mm], var


flat = [g for gs in GENES.values() for g in gs]
data = {}
for n in [1, 2, 3]:
    Xm, var = mye(n)
    vi = {g: var.index(g) for g in flat if g in var}
    data[n] = {g: np.asarray(Xm[:, vi[g]].todense()).ravel() for g in vi}
    print(f"slice {n}: {Xm.shape[0]:,} myeloid")

print(f"\n{'module':10} {'gene':8} {'ctrl_mean':>9} {'t1_mean':>8} {'t2_mean':>8} "
      f"{'t/c_ratio':>9} {'ctrl%det':>8} {'tum%det':>7}")
for mod, gs in GENES.items():
    for g in gs:
        if g not in data[3]:
            continue
        cm = data[3][g].mean()
        t1 = data[1][g].mean(); t2 = data[2][g].mean()
        tmean = np.concatenate([data[1][g], data[2][g]]).mean()
        ratio = tmean / cm if cm > 0 else np.inf
        cdet = 100 * (data[3][g] > 0).mean()
        tdet = 100 * (np.concatenate([data[1][g], data[2][g]]) > 0).mean()
        print(f"{mod:10} {g:8} {cm:>9.3f} {t1:>8.3f} {t2:>8.3f} {ratio:>9.2f} "
              f"{cdet:>7.1f}% {tdet:>6.1f}%")
