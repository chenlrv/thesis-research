"""
Diagnostic: are the MDM calls in CONTROL slices (3,4) real or low-count/ambient artifacts?
Compares MDM-call quality across slices. Tumor = 1,2,5,6 ; Control = 3,4.
Run: conda run -n thesis_research python agents/annotation/diag_mdm_control_slices.py
"""
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import poisson

BASE = "D:/thesis-research/resources/cache/"
TUMOR = {1, 2, 5, 6}


def col(a, panel, g):
    key = panel.get(g.lower())
    if key is None:
        return np.zeros(a.n_obs)
    x = a[:, key].X
    return (x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()).astype(float)


rows = []
for s in range(1, 7):
    a = ad.read_h5ad(f"{BASE}with_tumor_prediction/slice_{s}_adata.h5ad")
    wn = ad.read_h5ad(f"{BASE}slice_{s}_adata_with_neg.h5ad")
    neg = [g for g in wn.var_names if g.lower().startswith("negative")]
    nx = wn[:, neg].X
    nx = nx.toarray() if sp.issparse(nx) else np.asarray(nx)
    neg_by_gid = pd.Series(nx.mean(1), index=wn.obs["cell_global_id"].values)
    a.obs["neg_mean"] = neg_by_gid.reindex(a.obs["cell_global_id"].values).values
    nt = a[~a.obs["pred_tumor_XGBoost"].astype(bool)].copy()
    panel = {g.lower(): g for g in nt.var_names}

    ccr2 = col(nt, panel, "Ccr2")
    plac8 = col(nt, panel, "Plac8")
    spec_sum = ccr2 + plac8
    spec_n = (ccr2 > 0).astype(int) + (plac8 > 0).astype(int)
    ncount = np.asarray(nt.X.sum(1)).ravel()
    neg_mean = np.nan_to_num(nt.obs["neg_mean"].to_numpy(float), nan=0.0)

    # current gate (presence) vs stricter alternatives
    mdm_presence = spec_sum > 0
    mdm_sum2 = spec_sum >= 2
    mdm_2genes = spec_n >= 2
    # Poisson background test on the Ccr2/Plac8 pair (n=2 genes)
    lam = 2 * np.maximum(neg_mean, 0.02)
    mdm_poisson = (poisson.sf(spec_sum - 1, lam) < 0.01) & (spec_sum > 0)

    m = mdm_presence
    nMDM = int(m.sum())
    rows.append({
        "slice": s,
        "type": "TUMOR" if s in TUMOR else "CONTROL",
        "n_nontumor": nt.n_obs,
        "MDM_presence_n": nMDM,
        "MDM_presence_pct": round(nMDM / nt.n_obs * 100, 2),
        "frac_single_molecule": round((spec_sum[m] == 1).mean(), 3),
        "frac_both_genes": round((spec_n[m] >= 2).mean(), 3),
        "median_specsum_inMDM": float(np.median(spec_sum[m])),
        "median_nCount_MDM": float(np.median(ncount[m])),
        "median_nCount_all": float(np.median(ncount)),
        # how MDM count survives under stricter gates (as % of the presence call)
        "kept_sum>=2_%": round(mdm_sum2.sum() / max(nMDM, 1) * 100, 1),
        "kept_2genes_%": round(mdm_2genes.sum() / max(nMDM, 1) * 100, 1),
        "kept_poisson_%": round(mdm_poisson.sum() / max(nMDM, 1) * 100, 1),
        # stricter MDM rate as % of non-tumor cells
        "MDM_poisson_pct": round(mdm_poisson.sum() / nt.n_obs * 100, 2),
    })
    print(f"slice {s} done")

df = pd.DataFrame(rows)
pd.set_option("display.width", 240, "display.max_columns", 30)
print("\n================ MDM-call quality by slice ================\n")
print(df.to_string(index=False))
df.to_csv("D:/thesis-research/agents/outputs/annotation/final/mdm_control_diagnostic.csv", index=False)
print("\nTUMOR vs CONTROL means:")
print(df.groupby("type")[["MDM_presence_pct", "frac_single_molecule", "median_specsum_inMDM",
                          "median_nCount_MDM", "kept_poisson_%", "MDM_poisson_pct"]].mean().round(3).to_string())
