"""
Step 4 (FALLBACK re-segmentation): neighbour-profile transcript REASSIGNMENT,
fastReseg-style, per FOV. Produces an "after" cell x gene matrix to compare with
the current-segmentation baseline using the SAME seg_metrics.py.

Idea (tests the spillover hypothesis directly):
  A transcript currently assigned to cell c may actually be spillover from a
  neighbouring cell. For each transcript we look at its home cell plus the
  spatially nearest cells, and re-assign it to the candidate cell whose
  expression PROFILE best explains that gene (log-likelihood), with a spatial
  prior favouring the nearest / home cell. Molecules that look like contamination
  migrate to the cell they belong to; the home cell keeps a prior boost so we do
  not over-move.

Why this is the right fallback here:
  - It is exactly the contamination/reassignment QC the brief asks for.
  - CPU-only and MKL-safe (no large matmuls; we gather per-gene log-probs).
  - Per-FOV, so it CHECKPOINTS into after_metrics.json after each FOV.

Run for one FOV:
  conda run -n thesis_research python agents/segmentation/04_reseg_reassign.py 451
Run for all chosen FOVs (sequential, checkpointing each):
  conda run -n thesis_research python agents/segmentation/04_reseg_reassign.py all
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree

sys.path.insert(0, r"D:\thesis-research\agents\segmentation")
from seg_metrics import compute_all, fmt_report

OUT = r"D:\thesis-research\agents\outputs\segmentation"
TXDIR = os.path.join(OUT, "tx_by_fov")
WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
WITH_TUMOR = r"d:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
AFTER_JSON = os.path.join(OUT, "after_metrics.json")
BEFORE_TX_JSON = os.path.join(OUT, "before_tx_metrics.json")  # tx-derived "before"

CHOSEN = [451, 512, 514, 515, 523]

# Reassignment hyperparameters
K_NEIGH = 6          # candidate cells = home + up to this many nearest centroids
RADIUS_PX = 60.0     # only consider candidate cells whose centroid is within this (px)
HOME_PRIOR = 0.5     # log-prior boost (nats) for keeping a tx in its current cell
# (0.5 chosen from prior_sweep: lets genuine spillover migrate while not dissolving
#  cells; frac moved ~0.5-1%. The full prior sensitivity envelope is in
#  prior_sweep_fov514.csv — even home_prior=0 only moves <1% of tx.)
DIST_SCALE = 30.0    # spatial decay length (px) for candidate log-prior
N_ITER = 2           # profile re-estimation rounds
PSEUDO = 1e-4        # profile smoothing


def load_panel_and_tumor():
    """Real gene panel order (drop neg/falsecode) + non-tumor cell set."""
    a = ad.read_h5ad(WITH_NEG)
    neg = np.array([vn.lower().startswith(("negprb", "negprobe", "falsecode", "blank"))
                    for vn in a.var_names])
    genes = a.var_names[~neg].tolist()
    at = ad.read_h5ad(WITH_TUMOR)
    tumor_cells = set(at.obs.loc[at.obs["pred_tumor_XGBoost"].to_numpy(), "cell"].astype(str))
    return genes, tumor_cells


def build_matrix(df_assign, cell_ids, gene_index, n_genes):
    """Build dense cell x gene counts from a transcript table with a 'cell_idx'
    (row into cell_ids) and 'gene_idx' (col, -1 = off-panel) column."""
    n_cells = len(cell_ids)
    mat = np.zeros((n_cells, n_genes), dtype=np.float64)
    ci = df_assign["cell_idx"].to_numpy()
    gi = df_assign["gene_idx"].to_numpy()
    ok = gi >= 0
    np.add.at(mat, (ci[ok], gi[ok]), 1.0)
    return mat


def profiles_from_matrix(mat):
    """Log-normalized expression profile per cell: log((counts+pseudo)/total)."""
    tot = mat.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    p = (mat + PSEUDO) / (tot + PSEUDO * mat.shape[1])
    return np.log(p)  # (n_cells, n_genes), natural log


def reassign_fov(fov, genes, tumor_cells):
    t0 = time.time()
    gene_index = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    df = pd.read_csv(os.path.join(TXDIR, f"tx_fov{fov}.csv"), low_memory=False)
    # keep only transcripts whose target is a real panel gene
    df["gene_idx"] = df["target"].map(gene_index).fillna(-1).astype(int)
    # current assignment: cell_ID 0 = unassigned -> treat as no home
    df["home_cell"] = df["cell"].astype(str)
    df.loc[df["cell_ID"] == 0, "home_cell"] = ""

    # Build the universe of CELLS (assigned, real cells) and their centroids from
    # their currently-assigned transcripts.
    assigned = df[df["cell_ID"] != 0].copy()
    cell_ids = sorted(assigned["home_cell"].unique())
    cell_pos = {c: i for i, c in enumerate(cell_ids)}
    n_cells = len(cell_ids)

    # centroids (global px) per cell from assigned transcripts
    cent = assigned.groupby("home_cell")[["x_global_px", "y_global_px"]].mean()
    cent = cent.loc[cell_ids].to_numpy()
    tree = cKDTree(cent)

    # map every transcript's home cell to a row index (or -1 if unassigned)
    df["home_idx"] = df["home_cell"].map(cell_pos).fillna(-1).astype(int)

    # initial matrix + profiles from current segmentation (this is tx-derived "before")
    df_home = df.copy()
    df_home["cell_idx"] = df_home["home_idx"]
    mat0 = build_matrix(df_home[df_home["cell_idx"] >= 0], cell_ids, gene_index, n_genes)

    # Precompute, for every transcript, the candidate cells (home + nearest) and
    # the spatial log-prior to each. We query the KD-tree on transcript positions.
    txy = df[["x_global_px", "y_global_px"]].to_numpy()
    # nearest K cells per transcript within RADIUS
    dists, idxs = tree.query(txy, k=K_NEIGH, distance_upper_bound=RADIUS_PX,
                             workers=-1)
    if K_NEIGH == 1:
        dists = dists[:, None]; idxs = idxs[:, None]
    # idxs == n_cells means "no neighbour found" (cKDTree sentinel)
    valid = idxs < n_cells
    gidx = df["gene_idx"].to_numpy()
    home_idx = df["home_idx"].to_numpy()
    n_tx = len(df)

    cur_assign = home_idx.copy()  # start from current segmentation
    moved_total = 0
    for it in range(N_ITER):
        # (re)build matrix & profiles from current assignment
        cur = pd.DataFrame({"cell_idx": cur_assign, "gene_idx": gidx})
        mat = build_matrix(cur[cur["cell_idx"] >= 0], cell_ids, gene_index, n_genes)
        logp = profiles_from_matrix(mat)  # (n_cells, n_genes)

        # score each transcript against each candidate cell:
        #   score = logp[candidate, gene] + spatial_prior + home_prior(if candidate==home)
        # vectorised over the K candidate columns.
        best_score = np.full(n_tx, -np.inf)
        best_cell = cur_assign.copy()
        on_panel = gidx >= 0
        for k in range(idxs.shape[1]):
            cand = idxs[:, k]
            vmask = valid[:, k] & on_panel
            if not vmask.any():
                continue
            cc = cand[vmask]
            gg = gidx[vmask]
            sc = logp[cc, gg]
            # spatial prior: closer candidate -> higher
            sc = sc - (dists[vmask, k] / DIST_SCALE)
            # home prior
            is_home = (cc == home_idx[vmask]) & (home_idx[vmask] >= 0)
            sc = sc + HOME_PRIOR * is_home.astype(np.float64)
            better = sc > best_score[vmask]
            # write-back
            sub_idx = np.where(vmask)[0]
            upd = sub_idx[better]
            best_score[upd] = sc[better]
            best_cell[upd] = cc[better]
        # transcripts with no valid candidate keep their home (or stay unassigned)
        new_assign = best_cell
        moved = int((new_assign != cur_assign).sum())
        moved_total = moved
        cur_assign = new_assign
        print(f"    iter {it}: moved {moved:,}/{n_tx:,} tx "
              f"({100*moved/n_tx:.1f}%)  ({time.time()-t0:.0f}s)", flush=True)

    # final corrected matrix
    cur = pd.DataFrame({"cell_idx": cur_assign, "gene_idx": gidx})
    mat_after = build_matrix(cur[cur["cell_idx"] >= 0], cell_ids, gene_index, n_genes)

    # restrict to NON-TUMOR cells for the metrics (same as baseline)
    nontumor = np.array([cid not in tumor_cells for cid in cell_ids])
    keep = nontumor & (mat_after.sum(axis=1) > 0)
    keep0 = nontumor & (mat0.sum(axis=1) > 0)

    res_after = compute_all(mat_after[keep], genes, label=f"AFTER(reassign) FOV {fov}")
    res_before = compute_all(mat0[keep0], genes, label=f"BEFORE(tx) FOV {fov}")

    runtime = time.time() - t0
    res_after["_runtime_sec"] = runtime
    res_after["_n_tx"] = int(n_tx)
    res_after["_frac_tx_moved"] = float(moved_total / n_tx)
    res_after["_n_cells_nontumor"] = int(keep.sum())
    res_before["_n_cells_nontumor"] = int(keep0.sum())

    print(f"  FOV {fov}: {n_tx:,} tx, {n_cells:,} cells, "
          f"{int(keep.sum()):,} non-tumor cells, {runtime:.0f}s")
    print(fmt_report(res_before))
    print(fmt_report(res_after))
    return res_before, res_after


def checkpoint(fov, res_before, res_after):
    after = {}
    before = {}
    if os.path.exists(AFTER_JSON):
        after = json.load(open(AFTER_JSON))
    if os.path.exists(BEFORE_TX_JSON):
        before = json.load(open(BEFORE_TX_JSON))
    after[str(fov)] = res_after
    before[str(fov)] = res_before
    json.dump(after, open(AFTER_JSON, "w"), indent=2)
    json.dump(before, open(BEFORE_TX_JSON, "w"), indent=2)
    print(f"  checkpointed FOV {fov} -> after_metrics.json / before_tx_metrics.json")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    fovs = CHOSEN if arg == "all" else [int(arg)]
    genes, tumor_cells = load_panel_and_tumor()
    print(f"panel genes: {len(genes)}; tumor cells in slice: {len(tumor_cells):,}")
    for fov in fovs:
        # skip if already checkpointed
        if os.path.exists(AFTER_JSON):
            done = json.load(open(AFTER_JSON))
            if str(fov) in done:
                print(f"FOV {fov} already done, skipping.")
                continue
        print(f"\n=== FOV {fov} ===")
        rb, ra = reassign_fov(fov, genes, tumor_cells)
        checkpoint(fov, rb, ra)


if __name__ == "__main__":
    main()
