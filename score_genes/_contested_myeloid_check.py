"""Are the cells that a margin-free myeloid rule would NEWLY call myeloid really
myeloid-with-ambient-Meg3, or neurons-with-ambient-myeloid?

Contested = pass myeloid FDR AND detect >=1 lineage gene (Csf1r/Aif1/Tyrobp/Fcer1g)
            AND NOT currently called Myeloid by strict Stage-1.
Profile their raw counts vs real Myeloid and real Neurons:
  - lineage myeloid (LOW ambient)  : Csf1r, Aif1, Tyrobp, Fcer1g
  - complement     (higher ambient): C1qa, C1qb, C1qc
  - neuron ambient                 : Meg3
  - neuron identity (LOW ambient)  : Nrxn1, Nrxn3, Scg5, Ryr2, Calb1
If contested look like Myeloid on lineage genes and have ~0 neuron-identity genes
(only Meg3) -> they are myeloid + ambient. Usage: python _contested_myeloid_check.py <slice>
"""
import os
import sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

SL = sys.argv[1] if len(sys.argv) > 1 else "1"
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
COMPL = ["C1qa", "C1qb", "C1qc"]
NEUR_AMB = ["Meg3"]
NEUR_ID = ["Nrxn1", "Nrxn3", "Scg5", "Ryr2", "Calb1"]
PROFILE = LINEAGE + COMPL + NEUR_AMB + NEUR_ID


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


with h5py.File(TMPL.format(SL), "r") as h5:
    X = _read_X(h5); var = _read_var(h5); tumor = _read_bool(h5, "pred_tumor_XGBoost")
Xk = X[~tumor]; del X
vi = {g: var.index(g) for g in PROFILE if g in var}
raw = {g: np.asarray(Xk[:, vi[g]].todense()).ravel() for g in vi}

d = f"D:/thesis-research/score_genes_slice{SL}_v2"
cs = pd.read_csv(f"{d}/cell_scores.csv")
thr = pd.read_csv(f"{d}/mirrored_fdr_thresholds.csv").set_index("label")["threshold"]
assert len(cs) == Xk.shape[0]

mye_fdr = cs["score_Myeloid"].to_numpy() >= thr["Myeloid"]
lineage_pos = np.vstack([raw[g] for g in LINEAGE]).sum(0) > 0     # >=1 lineage gene
cur = cs["celltype_v2"].to_numpy()
is_mye = cur == "Myeloid"
is_neu = cur == "Neurons"
contested = mye_fdr & lineage_pos & ~is_mye

print(f"=== slice {SL} ===")
print(f"strict Myeloid: {int(is_mye.sum()):,} | contested (newly myeloid): {int(contested.sum()):,}")
print("contested currently labeled:", pd.Series(cur[contested]).value_counts().to_dict())

print(f"\n{'gene':8} {'group':10} {'contested':>10} {'realMyeloid':>12} {'realNeuron':>11}")
for g in PROFILE:
    grp = ("lineage" if g in LINEAGE else "complement" if g in COMPL
           else "neur-amb" if g in NEUR_AMB else "neur-id")
    print(f"{g:8} {grp:10} {raw[g][contested].mean():>10.3f} "
          f"{raw[g][is_mye].mean():>12.3f} {raw[g][is_neu].mean():>11.3f}")

# summary: mean # distinct genes detected per program
def ndet(genes, mask):
    return np.vstack([raw[g][mask] > 0 for g in genes]).sum(0).mean()
print(f"\nmean #genes detected  (contested / realMyeloid / realNeuron):")
print(f"  lineage myeloid : {ndet(LINEAGE,contested):.2f} / {ndet(LINEAGE,is_mye):.2f} / {ndet(LINEAGE,is_neu):.2f}")
print(f"  neuron identity : {ndet(NEUR_ID,contested):.2f} / {ndet(NEUR_ID,is_mye):.2f} / {ndet(NEUR_ID,is_neu):.2f}")
print(f"  Meg3 detected % : {100*(raw['Meg3'][contested]>0).mean():.0f}% / "
      f"{100*(raw['Meg3'][is_mye]>0).mean():.0f}% / {100*(raw['Meg3'][is_neu]>0).mean():.0f}%")
