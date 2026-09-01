"""
Sensitivity sweep on ONE FOV: vary the home-prior strength (how strongly we keep
a transcript in its current cell) and the spatial decay, and see whether ANY
reasonable reassignment setting improves marker purity / cross-leakage and moves
the probe correlations. This decides whether the verdict (seg fixes it / doesn't)
is robust to hyperparameters rather than an artifact of one conservative choice.

Run:
  conda run -n thesis_research python agents/segmentation/05_prior_sweep.py 514
"""
import os, sys, time
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\thesis-research\agents\segmentation")
from seg_metrics import compute_all

OUT = r"D:\thesis-research\agents\outputs\segmentation"
TXDIR = os.path.join(OUT, "tx_by_fov")
WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
WITH_TUMOR = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"

K_NEIGH = 6
RADIUS_PX = 60.0
DIST_SCALE = 30.0
PSEUDO = 1e-4


def load_panel_and_tumor():
    a = ad.read_h5ad(WITH_NEG)
    neg = np.array([vn.lower().startswith(("negprb","negprobe","falsecode","blank"))
                    for vn in a.var_names])
    genes = a.var_names[~neg].tolist()
    at = ad.read_h5ad(WITH_TUMOR)
    tumor = set(at.obs.loc[at.obs["pred_tumor_XGBoost"].to_numpy(), "cell"].astype(str))
    return genes, tumor


def build_matrix(cell_idx, gene_idx, n_cells, n_genes):
    mat = np.zeros((n_cells, n_genes))
    ok = (cell_idx >= 0) & (gene_idx >= 0)
    np.add.at(mat, (cell_idx[ok], gene_idx[ok]), 1.0)
    return mat


def run(fov, home_prior, genes, tumor, n_iter=2):
    gene_index = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    df = pd.read_csv(os.path.join(TXDIR, f"tx_fov{fov}.csv"), low_memory=False)
    df["gene_idx"] = df["target"].map(gene_index).fillna(-1).astype(int)
    df["home_cell"] = df["cell"].astype(str)
    df.loc[df["cell_ID"] == 0, "home_cell"] = ""
    assigned = df[df["cell_ID"] != 0]
    cell_ids = sorted(assigned["home_cell"].unique())
    cell_pos = {c: i for i, c in enumerate(cell_ids)}
    n_cells = len(cell_ids)
    cent = assigned.groupby("home_cell")[["x_global_px","y_global_px"]].mean().loc[cell_ids].to_numpy()
    tree = cKDTree(cent)
    df["home_idx"] = df["home_cell"].map(cell_pos).fillna(-1).astype(int)
    txy = df[["x_global_px","y_global_px"]].to_numpy()
    dists, idxs = tree.query(txy, k=K_NEIGH, distance_upper_bound=RADIUS_PX, workers=-1)
    valid = idxs < n_cells
    gidx = df["gene_idx"].to_numpy()
    home_idx = df["home_idx"].to_numpy()
    n_tx = len(df)
    on_panel = gidx >= 0

    cur = home_idx.copy()
    moved = 0
    for it in range(n_iter):
        mat = build_matrix(cur, gidx, n_cells, n_genes)
        tot = mat.sum(1, keepdims=True); tot[tot==0]=1
        logp = np.log((mat+PSEUDO)/(tot+PSEUDO*n_genes))
        best_s = np.full(n_tx, -np.inf); best_c = cur.copy()
        for k in range(idxs.shape[1]):
            cand = idxs[:,k]; vmask = valid[:,k] & on_panel
            if not vmask.any(): continue
            cc = cand[vmask]; gg = gidx[vmask]
            sc = logp[cc,gg] - dists[vmask,k]/DIST_SCALE
            is_home = (cc==home_idx[vmask]) & (home_idx[vmask]>=0)
            sc = sc + home_prior*is_home
            sub = np.where(vmask)[0]; bett = sc > best_s[vmask]
            best_s[sub[bett]] = sc[bett]; best_c[sub[bett]] = cc[bett]
        moved = int((best_c != cur).sum()); cur = best_c
    mat_a = build_matrix(cur, gidx, n_cells, n_genes)
    nontumor = np.array([c not in tumor for c in cell_ids])
    keep = nontumor & (mat_a.sum(1) > 0)
    res = compute_all(mat_a[keep], genes, label=f"hp={home_prior}")
    return res, moved/n_tx


def main():
    fov = int(sys.argv[1]) if len(sys.argv) > 1 else 514
    genes, tumor = load_panel_and_tumor()
    # baseline (tx, current seg) == very large home prior
    rows = []
    for hp in [999, 3.0, 1.5, 0.5, 0.0]:
        t0 = time.time()
        res, frac = run(fov, hp, genes, tumor)
        p = res["purity"]
        rows.append({
            "home_prior": hp, "frac_moved": round(frac,3),
            "purity": round(p["purity_own_frac"],3),
            "cross_leak": round(p["cross_leakage_frac"],3),
            "F1_r": round(res.get("F1_full_r", float('nan')),3),
            "F2_r": round(res.get("F2_full_r", float('nan')),3),
            "lyve_pct": round(res.get("F3_lyve_pct_pos", float('nan')),1),
            "lyve_lack_bam": round(res.get("F3_lyve_pct_lacking_bam", float('nan')),0),
            "sec": round(time.time()-t0,1),
        })
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"prior_sweep_fov{fov}.csv"), index=False)
    print("\n", df.to_string(index=False))
    print(f"\nhome_prior=999 row ~= current segmentation (essentially no movement).")


if __name__ == "__main__":
    main()
