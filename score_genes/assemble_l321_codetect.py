"""Final L321 annotation using the >=3-pan-myeloid pool + co-detection subtypes.

celltype_final (non-tumor):
  backbone (Astrocytes/Neurons/Vascular) from Stage-1 celltype_v2
  + Choroid = Ttr>=5 (validated; replaces scored-Ependymal)
  + Myeloid pool (detect>=3 pan-myeloid AND >=1 lineage) -> co-detection subtype,
    OVERRIDING any conflicting backbone/Choroid call (myeloid pool wins).
      BAM   = >=2 of {Pf4,Maf,Mrc1,Cd163,Cd5l}
      MDM   = Ccr2 & >=1 of {Plac8,Vcan,Cd14,Ccr1,Fpr1} & Pf4/Maf-neg
      Micro = >=1 of {Adgrg1,Hpgds,Gpr183} & not BAM/MDM
      else  = Myeloid_unresolved
  rest -> unassigned.

Per-type spatial maps (n + % of all cells) + combined; tumor BLACK.
Output -> score_genes_slice{n}_v2/final_codetect/
"""
import gc
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

SLICES = [1, 2, 3]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
TTR_THR = 5
PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
MICRO = ["Adgrg1", "Hpgds", "Gpr183"]
BAMg = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]
BAM_STABLE = ["Pf4", "Maf"]
MDM_SUP = ["Plac8", "Vcan", "Cd14", "Ccr1", "Fpr1"]
NEED = set(PAN + LINEAGE + MICRO + BAMg + MDM_SUP + ["Ccr2", "Ttr"])

BIO = [
    ("Astrocytes", "#1f77b4"), ("Neurons", "#e377c2"), ("Microglia", "#17becf"),
    ("BAM", "#d62728"), ("MDM", "#00a087"), ("Myeloid_unresolved", "#8c564b"),
    ("Vascular", "#2ca02c"), ("Choroid", "#9467bd"),
]
DISP = {"Myeloid_unresolved": "Myeloid (unresolved)"}


def _dec(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _X(h5):
    n = h5["X"]; e = str(n.attrs.get("encoding-type", "")); s = tuple(n.attrs["shape"])
    a = (n["data"][...], n["indices"][...], n["indptr"][...])
    return (csc_matrix(a, shape=s) if "csc" in e else csr_matrix(a, shape=s)).tocsr()


def _var(h5):
    v = h5["var"]; k = v.attrs.get("_index", "_index")
    return list(_dec(v[k.decode() if isinstance(k, bytes) else k][...]))


def _num(h5, c):
    nd = h5["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...]).astype(float)
        return ct[np.clip(cd, 0, None)]
    return nd[...].astype(float)


def _bool(h5, c):
    nd = h5["obs"][c]
    if isinstance(nd, h5py.Group):
        cd = nd["codes"][...]; ct = _dec(nd["categories"][...])
        return np.isin(np.where(cd >= 0, ct[np.clip(cd, 0, None)], "False").astype(str),
                       ["True", "1", "1.0", "TRUE", "true"])
    ar = nd[...]
    return (np.isin(_dec(ar).astype(str), ["True", "1"]) if ar.dtype.kind in ("S", "O")
            else ar.astype(bool))


def build(n):
    d = f"D:/thesis-research/score_genes_slice{n}_v2"
    cs = pd.read_csv(f"{d}/cell_scores.csv")
    x, y = cs["x"].to_numpy(), cs["y"].to_numpy()
    v2 = cs["celltype_v2"].to_numpy()

    with h5py.File(TMPL.format(n), "r") as h5:
        X = _X(h5); var = _var(h5)
        cx = _num(h5, "CenterX_global_px"); cy = _num(h5, "CenterY_global_px")
        tumor = _bool(h5, "pred_tumor_XGBoost")
    keep = ~tumor
    Xk = X[keep]; del X; gc.collect()
    assert len(cs) == Xk.shape[0]
    raw = {g: np.asarray(Xk[:, var.index(g)].todense()).ravel() for g in NEED if g in var}
    del Xk; gc.collect()

    def c(genes):
        return np.vstack([raw[g] > 0 for g in genes]).sum(0)

    # backbone (non-myeloid) + choroid
    final = np.full(len(x), "unassigned", dtype=object)
    for t in ["Astrocytes", "Neurons", "Vascular"]:
        final[v2 == t] = t
    final[raw["Ttr"] >= TTR_THR] = "Choroid"

    # myeloid pool -> co-detection (overrides)
    pool = (c(PAN) >= 3) & (c(LINEAGE) >= 1)
    bam = pool & (c(BAMg) >= 2)
    mdm = pool & ~bam & (raw["Ccr2"] > 0) & (c(MDM_SUP) >= 1) & (c(BAM_STABLE) == 0)
    micro = pool & ~bam & ~mdm & (c(MICRO) >= 1)
    stolen = pd.Series(final[pool]).value_counts().to_dict()
    final[micro] = "Microglia"; final[bam] = "BAM"; final[mdm] = "MDM"
    final[pool & ~(micro | bam | mdm)] = "Myeloid_unresolved"

    tot = len(final) + int(tumor.sum())
    print(f"\nslice {n}: {tot:,} cells ({int(tumor.sum()):,} tumor). myeloid pool={int(pool.sum()):,}")
    print(f"  pool reassigned from Stage-1: {stolen}")
    vc = pd.Series(final).value_counts()
    for k, _ in BIO:
        if k in vc:
            print(f"   {DISP.get(k,k):22s} {vc[k]:>7,}  {100*vc[k]/tot:5.1f}%")
    print(f"   {'unassigned':22s} {int((final=='unassigned').sum()):>7,}")
    pd.DataFrame({"x": x, "y": y, "celltype_final": final}).to_csv(
        f"{d}/annotation_final_codetect.csv", index=False)
    return x, y, final, cx[tumor], cy[tumor], tot


def plots(n, x, y, final, tx, ty, tot):
    out = f"D:/thesis-research/score_genes_slice{n}_v2/final_codetect"
    os.makedirs(out, exist_ok=True)
    tlab = f"tumor (n={len(tx):,})"
    for key, col in BIO:
        m = final == key
        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
        ax.scatter(x, y, s=0.7, c="#e2e2e2", linewidths=0, rasterized=True, label="other")
        if len(tx):
            ax.scatter(tx, ty, s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
        ax.scatter(x[m], y[m], s=4, c=col, linewidths=0, rasterized=True, label=DISP.get(key, key))
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {n} — {DISP.get(key,key)}  (n={int(m.sum()):,}, {100*m.sum()/tot:.1f}%)")
        ax.legend(loc="lower right", markerscale=4, fontsize=8, frameon=True)
        fig.savefig(f"{out}/slice{n}_{key}.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=170)
    m0 = final == "unassigned"
    ax.scatter(x[m0], y[m0], s=0.7, c="#e8e8e8", linewidths=0, rasterized=True,
               label=f"unassigned ({int(m0.sum()):,})")
    if len(tx):
        ax.scatter(tx, ty, s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
    for key, col in BIO:
        m = final == key
        if m.any():
            ax.scatter(x[m], y[m], s=4, c=col, linewidths=0, rasterized=True,
                       label=f"{DISP.get(key,key)} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {n} — final annotation (>=3-pan myeloid + co-detection)")
    ax.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
    fig.savefig(f"{out}/slice{n}_ALL.png", bbox_inches="tight"); plt.close(fig)
    print(f"   saved -> {out}")


for n in SLICES:
    plots(n, *build(n))
