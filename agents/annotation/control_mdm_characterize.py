"""
Characterize the residual MDM calls in CONTROL slices 3 & 4 (strict gate):
genuine perivascular/border monocytes vs residual ambient?
  - Ccr2+Plac8 expression level (borderline ==2 vs higher)
  - cell size (nCount)
  - slice 3: decontx_contamination in MDM vs non-MDM
  - spatial pattern (clustered niche vs diffuse) -> 2-panel plot
Run: conda run -n thesis_research python agents/annotation/control_mdm_characterize.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import poisson

BASE = "D:/thesis-research/resources/cache/"
OUT = "D:/thesis-research/agents/outputs/annotation/final"
ALPHA, BG_FLOOR = 0.01, 0.02
PAN_MYE = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
BAM_SPEC, MDM_SPEC = ["Mrc1", "Cd163"], ["Ccr2", "Plac8"]
MDM_BROAD = ["Cd14", "S100a8", "S100a9"]


def load(s):
    a = ad.read_h5ad(f"{BASE}with_tumor_prediction/slice_{s}_adata.h5ad")
    wn = ad.read_h5ad(f"{BASE}slice_{s}_adata_with_neg.h5ad")
    neg = [g for g in wn.var_names if g.lower().startswith("negative")]
    nx = wn[:, neg].X
    nx = nx.toarray() if sp.issparse(nx) else np.asarray(nx)
    a.obs["neg_mean"] = pd.Series(nx.mean(1), index=wn.obs["cell_global_id"].values
                                  ).reindex(a.obs["cell_global_id"].values).values
    return a[~a.obs["pred_tumor_XGBoost"].astype(bool)].copy()


fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=130)
for ax, s in zip(axes, (3, 4)):
    a = load(s)
    panel = {g.lower(): g for g in a.var_names}

    def cnt(g):
        k = panel.get(g.lower())
        x = a[:, k].X
        return (x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()).astype(float)

    def mat(genes):
        return np.vstack([cnt(g) for g in genes if g.lower() in panel]).T

    neg_mean = np.nan_to_num(a.obs["neg_mean"].to_numpy(float), nan=0.0)
    bg = np.maximum(neg_mean, BG_FLOOR)

    def detset(genes, mg=1):
        M = mat(genes); n = M.shape[1]; c = M.sum(1)
        return (poisson.sf(c - 1, n * bg) < ALPHA) & (c > 0) & ((M > 0).sum(1) >= mg)

    is_mye = detset(PAN_MYE, 2)
    bam = detset(BAM_SPEC, 1)
    mdm = detset(MDM_SPEC, 1)
    mdm_inflam = detset(MDM_BROAD, 2) & (~mdm)
    cx, sel = cnt("Cx3cr1"), cnt("Selplg")
    mic = (mat(["P2rx5", "TMEM119"]) > 0).any(1) | (cx > 0) | (sel > 0)
    # reproduce the gate's within-myeloid priority: inflam < mic < mdm < bam
    lab = np.array(["unresolved"] * a.n_obs, dtype=object)
    lab[mdm_inflam] = "MDM_inflammatory"
    lab[mic] = "Microglia"
    lab[mdm] = "MDM"
    lab[bam] = "BAM"
    core_mdm = is_mye & np.isin(lab, ["MDM", "MDM_inflammatory"])   # what the map paints red
    m = core_mdm
    n_spec = int((m & mdm).sum())
    n_inflam = int((m & mdm_inflam).sum())
    specsum = cnt("Ccr2") + cnt("Plac8")
    ncount = np.asarray(a.X.sum(1)).ravel()

    print(f"\n===== CONTROL slice {s}: residual MDM cells (as painted on map) =====")
    print(f"n MDM(core) = {int(m.sum()):,} ({m.mean()*100:.2f}% of {a.n_obs:,} non-tumor) "
          f"= {n_spec} specific(Ccr2/Plac8) + {n_inflam} inflammatory(Cd14/S100a8/9)")
    mspec = m & mdm
    ss = specsum[mspec]
    print(f"Ccr2+Plac8 counts in specific-MDM: median={np.median(ss):.0f} max={int(ss.max())} "
          f"| ==2: {(ss == 2).mean()*100:.0f}%  >=3: {(ss >= 3).mean()*100:.0f}%  >=5: {(ss >= 5).mean()*100:.0f}%")
    print(f"both Ccr2 AND Plac8 >0 (specific): {((cnt('Ccr2')[mspec] > 0) & (cnt('Plac8')[mspec] > 0)).mean()*100:.0f}%")
    print(f"median nCount: MDM {np.median(ncount[m]):.0f} vs all {np.median(ncount):.0f}")

    # slice 3: contamination
    if s == 3:
        dx = ad.read_h5ad(f"{BASE}decontx/slice_3_decontx.h5ad")
        if dx.n_obs == a.n_obs:
            cont = dx.obs["decontx_contamination"].to_numpy(float)
            print(f"decontx_contamination: MDM {cont[m].mean():.3f} vs non-MDM {cont[~m].mean():.3f}")

    # spatial clustering: share of MDM whose nearest MDM neighbor is < 30um (~ clustered niche)
    xy = a.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
    if m.sum() > 5:
        tree = cKDTree(xy[m])
        d, _ = tree.query(xy[m], k=2)
        nn = d[:, 1]
        print(f"MDM nearest-MDM dist (px): median={np.median(nn):.0f}  "
              f"(clustered if small relative to ~{np.sqrt(np.prod(np.ptp(xy, axis=0))/a.n_obs):.0f}px mean spacing)")

    # plot: all grey, MDM red sized by specsum
    ax.scatter(xy[~m, 0], xy[~m, 1], s=0.5, c="#dddddd", alpha=0.4, linewidths=0)
    ax.scatter(xy[m, 0], xy[m, 1], s=6 + 4 * np.clip(specsum[m] - 1, 0, 8),
               c="#d62728", alpha=0.85, linewidths=0)
    ax.set_title(f"Control slice {s}: strict MDM (n={int(m.sum())}, size∝Ccr2+Plac8)")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{OUT}/control_mdm_characterize.png", bbox_inches="tight")
print(f"\nWROTE {OUT}/control_mdm_characterize.png")
