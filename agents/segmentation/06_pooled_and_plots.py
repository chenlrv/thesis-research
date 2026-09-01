"""
Step 6: build the POOLED before/after cell x gene matrices across all 5 FOVs
(to match baseline_metrics.json 'all'), compute pooled metrics, and render
comparison plots. Reuses the reassignment from 04 (home_prior=0.5).

Run:
  conda run -n thesis_research python agents/segmentation/06_pooled_and_plots.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, r"D:\thesis-research\agents\segmentation")
from seg_metrics import compute_all, fmt_report

OUT = r"D:\thesis-research\agents\outputs\segmentation"
TXDIR = os.path.join(OUT, "tx_by_fov")
WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
WITH_TUMOR = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
CHOSEN = [451, 512, 514, 515, 523]

K_NEIGH, RADIUS_PX, DIST_SCALE, HOME_PRIOR, PSEUDO, N_ITER = 6, 60.0, 30.0, 0.5, 1e-4, 2


def load_panel_and_tumor():
    a = ad.read_h5ad(WITH_NEG)
    neg = np.array([vn.lower().startswith(("negprb","negprobe","falsecode","blank"))
                    for vn in a.var_names])
    genes = a.var_names[~neg].tolist()
    at = ad.read_h5ad(WITH_TUMOR)
    tumor = set(at.obs.loc[at.obs["pred_tumor_XGBoost"].to_numpy(), "cell"].astype(str))
    return genes, tumor


def build(cell_idx, gene_idx, n_cells, n_genes):
    m = np.zeros((n_cells, n_genes))
    ok = (cell_idx >= 0) & (gene_idx >= 0)
    np.add.at(m, (cell_idx[ok], gene_idx[ok]), 1.0)
    return m


def reseg_fov(fov, gene_index, n_genes):
    """Return (cell_ids list, mat_before, mat_after) for one FOV."""
    df = pd.read_csv(os.path.join(TXDIR, f"tx_fov{fov}.csv"), low_memory=False)
    df["gene_idx"] = df["target"].map(gene_index).fillna(-1).astype(int)
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
    mat_before = build(home_idx, gidx, n_cells, n_genes)
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
    mat_after = build(cur, gidx, n_cells, n_genes)
    return cell_ids, mat_before, mat_after


def main():
    genes, tumor = load_panel_and_tumor()
    gene_index = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    before_rows, after_rows = [], []
    keep_b, keep_a = [], []
    for fov in CHOSEN:
        cids, mb, ma = reseg_fov(fov, gene_index, n_genes)
        nontumor = np.array([c not in tumor for c in cids])
        kb = nontumor & (mb.sum(1) > 0)
        ka = nontumor & (ma.sum(1) > 0)
        before_rows.append(mb[kb]); after_rows.append(ma[ka])
    MB = np.vstack(before_rows); MA = np.vstack(after_rows)
    res_b = compute_all(MB, genes, label="BEFORE(tx) all chosen FOVs (non-tumor)")
    res_a = compute_all(MA, genes, label="AFTER(reassign) all chosen FOVs (non-tumor)")
    print(fmt_report(res_b)); print(); print(fmt_report(res_a))
    json.dump({"before_tx": res_b, "after_reassign": res_a},
              open(os.path.join(OUT, "pooled_metrics.json"), "w"), indent=2)

    # ----- load the existing vendor BASELINE 'all' for the 3-way plot -----
    base = json.load(open(os.path.join(OUT, "baseline_metrics.json")))["all"]

    # ---------- PLOT 1: purity / cross-leakage bars (3 conditions) ----------
    conds = ["vendor\nbaseline", "tx\n(current seg)", "reassigned\n(after)"]
    purity = [base["purity"]["purity_own_frac"],
              res_b["purity"]["purity_own_frac"],
              res_a["purity"]["purity_own_frac"]]
    leak = [base["purity"]["cross_leakage_frac"],
            res_b["purity"]["cross_leakage_frac"],
            res_a["purity"]["cross_leakage_frac"]]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(conds, purity, color=["#888","#4477aa","#66bb66"])
    ax[0].set_ylim(0.7, 0.85); ax[0].set_title("Marker purity (own-signal frac, higher=better)")
    for i,v in enumerate(purity): ax[0].text(i, v+0.002, f"{v:.3f}", ha="center")
    ax[1].bar(conds, leak, color=["#888","#4477aa","#ee6677"])
    ax[1].set_ylim(0.15, 0.25); ax[1].set_title("Cross-lineage leakage (lower=better)")
    for i,v in enumerate(leak): ax[1].text(i, v+0.002, f"{v:.3f}", ha="center")
    fig.suptitle("Segmentation comparison — pooled 5 FOVs (slice 1), non-tumor")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_purity_leakage.png"), dpi=130); plt.close(fig)

    # ---------- PLOT 2: probe correlations (F1, F2) ----------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    f1 = [base["F1_full_r"], res_b["F1_full_r"], res_a["F1_full_r"]]
    f2 = [base["F2_full_r"], res_b["F2_full_r"], res_a["F2_full_r"]]
    ax[0].bar(conds, f1, color=["#888","#4477aa","#66bb66"])
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("F1: GFP↔tdTomato r\n(expected to move toward 0 if seg-driven)")
    for i,v in enumerate(f1): ax[0].text(i, v-0.02 if v<0 else v+0.01, f"{v:+.3f}", ha="center", va="top" if v<0 else "bottom")
    ax[1].bar(conds, f2, color=["#888","#4477aa","#ee6677"])
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("F2: GFP↔Cx3cr1 r\n(expected to STAY negative — dead probe)")
    for i,v in enumerate(f2): ax[1].text(i, v-0.02 if v<0 else v+0.01, f"{v:+.3f}", ha="center", va="top" if v<0 else "bottom")
    fig.suptitle("Probe correlations before/after re-segmentation — pooled 5 FOVs")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_probe_correlations.png"), dpi=130); plt.close(fig)

    # ---------- PLOT 3: per-FOV delta in purity & leakage (reassign vs tx) ----------
    after = json.load(open(os.path.join(OUT, "after_metrics.json")))
    before = json.load(open(os.path.join(OUT, "before_tx_metrics.json")))
    fovs = [str(f) for f in CHOSEN]
    dpur = [after[f]["purity"]["purity_own_frac"]-before[f]["purity"]["purity_own_frac"] for f in fovs]
    dleak = [after[f]["purity"]["cross_leakage_frac"]-before[f]["purity"]["cross_leakage_frac"] for f in fovs]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(fovs, dpur, color=["#66bb66" if d>=0 else "#ee6677" for d in dpur])
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("Δ purity (after − before), per FOV")
    ax[1].bar(fovs, dleak, color=["#66bb66" if d<=0 else "#ee6677" for d in dleak])
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_title("Δ cross-leakage (after − before), per FOV\n(negative = improved)")
    fig.suptitle("Per-FOV effect of transcript reassignment (home_prior=0.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plot_perfov_delta.png"), dpi=130); plt.close(fig)

    print("\nWrote pooled_metrics.json and 3 plots.")


if __name__ == "__main__":
    main()
