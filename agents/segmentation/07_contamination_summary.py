"""
Step 7: direct contamination magnitude per cell, pooled over the 5 FOVs.
For each transcript the reassignment model picks a best-matching cell among
{home + nearest neighbours}. We quantify:
  - per-cell 'spillover-in' fraction = tx the model moved AWAY from this cell /
    tx originally assigned to it  (how much of its content looks foreign)
  - distribution of per-cell contamination and how concentrated it is.

This tells us whether spillover is a broad, fixable problem (lots of cells with
high foreign fraction) or a low-level boundary effect.

Run:
  conda run -n thesis_research python agents/segmentation/07_contamination_summary.py
"""
import os, sys, json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\thesis-research\agents\segmentation")
OUT = r"D:\thesis-research\agents\outputs\segmentation"
TXDIR = os.path.join(OUT, "tx_by_fov")
WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
CHOSEN = [451, 512, 514, 515, 523]
K_NEIGH, RADIUS_PX, DIST_SCALE, HOME_PRIOR, PSEUDO, N_ITER = 6, 60.0, 30.0, 0.5, 1e-4, 2


def genes_list():
    a = ad.read_h5ad(WITH_NEG)
    neg = np.array([vn.lower().startswith(("negprb","negprobe","falsecode","blank"))
                    for vn in a.var_names])
    return a.var_names[~neg].tolist()


def build(cell_idx, gene_idx, n_cells, n_genes):
    m = np.zeros((n_cells, n_genes))
    ok = (cell_idx >= 0) & (gene_idx >= 0)
    np.add.at(m, (cell_idx[ok], gene_idx[ok]), 1.0)
    return m


def per_fov(fov, gi):
    n_genes = len(gi)
    df = pd.read_csv(os.path.join(TXDIR, f"tx_fov{fov}.csv"), low_memory=False)
    df["gene_idx"] = df["target"].map(gi).fillna(-1).astype(int)
    df["home_cell"] = df["cell"].astype(str)
    df.loc[df["cell_ID"] == 0, "home_cell"] = ""
    assigned = df[df["cell_ID"] != 0]
    cell_ids = sorted(assigned["home_cell"].unique())
    cpos = {c: i for i, c in enumerate(cell_ids)}
    n_cells = len(cell_ids)
    cent = assigned.groupby("home_cell")[["x_global_px","y_global_px"]].mean().loc[cell_ids].to_numpy()
    tree = cKDTree(cent)
    df["home_idx"] = df["home_cell"].map(cpos).fillna(-1).astype(int)
    txy = df[["x_global_px","y_global_px"]].to_numpy()
    dists, idxs = tree.query(txy, k=K_NEIGH, distance_upper_bound=RADIUS_PX, workers=-1)
    valid = idxs < n_cells
    gidx = df["gene_idx"].to_numpy(); home_idx = df["home_idx"].to_numpy()
    n_tx = len(df); on_panel = gidx >= 0
    cur = home_idx.copy()
    for _ in range(N_ITER):
        m = build(cur, gidx, n_cells, n_genes)
        tot = m.sum(1, keepdims=True); tot[tot==0]=1
        logp = np.log((m+PSEUDO)/(tot+PSEUDO*n_genes))
        best_s = np.full(n_tx, -np.inf); best_c = cur.copy()
        for k in range(idxs.shape[1]):
            cand = idxs[:,k]; vm = valid[:,k] & on_panel
            if not vm.any(): continue
            cc = cand[vm]; gg = gidx[vm]
            sc = logp[cc,gg] - dists[vm,k]/DIST_SCALE
            ih = (cc==home_idx[vm]) & (home_idx[vm]>=0)
            sc = sc + HOME_PRIOR*ih
            sub = np.where(vm)[0]; bt = sc > best_s[vm]
            best_s[sub[bt]] = sc[bt]; best_c[sub[bt]] = cc[bt]
        cur = best_c
    # per-cell foreign fraction: among tx originally assigned to a cell (home_idx>=0),
    # fraction the model re-assigned ELSEWHERE.
    home_ok = home_idx >= 0
    moved_away = home_ok & (cur != home_idx)
    n_home = np.bincount(home_idx[home_ok], minlength=n_cells)
    n_moved = np.bincount(home_idx[moved_away], minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(n_home > 0, n_moved / np.maximum(n_home, 1), 0.0)
    return frac[n_home > 0]


def main():
    genes = genes_list()
    gi = {g: i for i, g in enumerate(genes)}
    allfrac = []
    for fov in CHOSEN:
        f = per_fov(fov, gi)
        allfrac.append(f)
        print(f"FOV {fov}: {len(f)} cells, mean foreign-frac={f.mean():.3f} "
              f"median={np.median(f):.3f} 90pct={np.percentile(f,90):.3f}")
    fr = np.concatenate(allfrac)
    summary = {
        "n_cells": int(len(fr)),
        "mean_foreign_frac": float(fr.mean()),
        "median_foreign_frac": float(np.median(fr)),
        "p75": float(np.percentile(fr, 75)),
        "p90": float(np.percentile(fr, 90)),
        "p99": float(np.percentile(fr, 99)),
        "pct_cells_over_10pct_foreign": float((fr > 0.10).mean() * 100),
        "pct_cells_over_20pct_foreign": float((fr > 0.20).mean() * 100),
    }
    print("\nPOOLED:", json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.join(OUT, "contamination_summary.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fr, bins=40, color="#4477aa")
    ax.axvline(fr.mean(), color="r", ls="--", label=f"mean={fr.mean():.3f}")
    ax.set_xlabel("per-cell fraction of transcripts the model reassigned away (spillover/contamination)")
    ax.set_ylabel("cells"); ax.set_title("Estimated per-cell contamination — pooled 5 FOVs (slice 1)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_contamination_hist.png"), dpi=130); plt.close(fig)
    print("Wrote contamination_summary.json + plot_contamination_hist.png")


if __name__ == "__main__":
    main()
