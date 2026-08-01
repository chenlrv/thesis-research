"""Hierarchical CO-DETECTION myeloid subtyping (high-confidence, no classifier) +
per-cell-type spatial plots. Slices 1,2,3.

pool  = clean inclusive myeloid (detect >=3 pan-myeloid AND >=1 lineage gene)
BAM   = >=2 of {Pf4,Maf,Mrc1,Cd163,Cd5l}
MDM   = Ccr2 AND >=1 of {Plac8,Vcan,Cd14,Ccr1,Fpr1}, and NOT Pf4/Maf+
Micro = >=1 of {Adgrg1,Hpgds,Gpr183} and not BAM/MDM
order : BAM > MDM > Microglia > unresolved

Per-type map (n + % of myeloid pool) and one combined map; tumor BLACK.
Also saves per-cell labels. Output -> score_genes_slice{n}_v2/codetect/
"""
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

SLICES = [1, 2, 3]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
PAN = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
MICRO = ["Adgrg1", "Hpgds", "Gpr183"]
BAM = ["Pf4", "Maf", "Mrc1", "Cd163", "Cd5l"]
BAM_STABLE = ["Pf4", "Maf"]
MDM_SUP = ["Plac8", "Vcan", "Cd14", "Ccr1", "Fpr1"]
ALLG = set(PAN + LINEAGE + MICRO + BAM + MDM_SUP + ["Ccr2"])
SUBS = [("Microglia", "#17becf"), ("BAM", "#d62728"), ("MDM", "#00a087")]
COL = dict(SUBS)


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


def run(n):
    out = f"D:/thesis-research/score_genes_slice{n}_v2/codetect"
    os.makedirs(out, exist_ok=True)
    with h5py.File(TMPL.format(n), "r") as h5:
        X = _X(h5); var = _var(h5)
        cx = _num(h5, "CenterX_global_px"); cy = _num(h5, "CenterY_global_px")
        tum = _bool(h5, "pred_tumor_XGBoost")
    keep = ~tum
    Xk = X[keep]; del X
    cxk, cyk = cx[keep], cy[keep]
    raw = {g: np.asarray(Xk[:, var.index(g)].todense()).ravel() for g in ALLG if g in var}

    def c(genes):
        return np.vstack([raw[g] > 0 for g in genes]).sum(0)

    pool = (c(PAN) >= 3) & (c(LINEAGE) >= 1)
    bam = pool & (c(BAM) >= 2)
    mdm = pool & ~bam & (raw["Ccr2"] > 0) & (c(MDM_SUP) >= 1) & (c(BAM_STABLE) == 0)
    micro = pool & ~bam & ~mdm & (c(MICRO) >= 1)
    lab = np.full(len(cxk), "unresolved", dtype=object)
    lab[micro] = "Microglia"; lab[bam] = "BAM"; lab[mdm] = "MDM"

    npool = int(pool.sum())
    role = " (CONTROL)" if n == 3 else ""
    print(f"slice {n}{role}: pool {npool:,}  Microglia={int(micro.sum()):,}  "
          f"BAM={int(bam.sum()):,}  MDM={int(mdm.sum()):,}  "
          f"unresolved={int((pool & (lab=='unresolved')).sum()):,}")
    pd.DataFrame({"cx": cxk[pool], "cy": cyk[pool], "subtype": lab[pool]}).to_csv(
        f"{out}/labels.csv", index=False)

    tlab = f"tumor ({int(tum.sum()):,})"
    # per-type
    for s, col in SUBS:
        m = lab == s
        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
        ax.scatter(cxk, cyk, s=0.6, c="#e6e6e6", linewidths=0, rasterized=True, label="other")
        if tum.sum():
            ax.scatter(cx[tum], cy[tum], s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
        ax.scatter(cxk[m], cyk[m], s=5, c=col, linewidths=0, rasterized=True,
                   label=f"{s} (n={int(m.sum()):,}, {100*m.sum()/max(npool,1):.0f}% of myeloid)")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {n}{role} — {s} (co-detection)")
        ax.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
        fig.savefig(f"{out}/slice{n}_{s}.png", bbox_inches="tight"); plt.close(fig)

    # combined
    fig, ax = plt.subplots(figsize=(11, 8), dpi=170)
    ax.scatter(cxk, cyk, s=0.6, c="#ececec", linewidths=0, rasterized=True)
    mu = lab == "unresolved"
    ax.scatter(cxk[mu & pool], cyk[mu & pool], s=2, c="#b8b8b8", linewidths=0,
               rasterized=True, label=f"unresolved ({int((mu & pool).sum()):,})")
    if tum.sum():
        ax.scatter(cx[tum], cy[tum], s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
    for s, col in SUBS:
        m = lab == s
        ax.scatter(cxk[m], cyk[m], s=5, c=col, linewidths=0, rasterized=True,
                   label=f"{s} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {n}{role} — myeloid subtypes (co-detection, pool={npool:,})")
    ax.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
    fig.savefig(f"{out}/slice{n}_ALL.png", bbox_inches="tight"); plt.close(fig)
    print(f"   saved -> {out}")


for n in (int(x) for x in (sys.argv[1:] or SLICES)):
    run(n)
