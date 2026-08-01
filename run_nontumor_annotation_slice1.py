"""First-pass annotation of NON-TUMOR cells in slice_1 (a tumor slice).

Goal: separate astrocyte / microglia / macrophage(MDM + BAM) among the
non-tumor compartment, then *validate the calls against mouse-brain spatial
biology* (the only thing that makes a panel-limited annotation trustworthy).

Pipeline
  0. Load slice_1 (X, var, spatial, tumor predictions) via h5py.
  1. Drop tumor cells (majority vote of the 4 pred_tumor_* columns). Keep the
     tumor coordinates aside -- we need them for spatial validation.
  2. normalize_total + log1p.
  3. Approach A (per-cell): score curated signatures, argmax + confidence gate.
  4. Approach B (clusters): curated lineage genes -> scale -> PCA -> neighbors
     -> Leiden; label each cluster by z-scored signature argmax; Ward
     hierarchical clustering organizes the cluster x signature matrix.
  5. Cross-check A vs B agreement.
  6. Spatial validation:
        - per-label spatial maps (tumor mass in grey underneath)
        - distance-to-nearest-tumor enrichment curves
     Brain priors being tested:
        MDM densifies INTO the tumor; astrocytes form a reactive RIM around it;
        microglia tile the parenchyma; BAM sit at borders/vessels.

Outputs: PNG/CSV under D:/thesis-research/nontumor_slice1/.
Nothing is written back to the cache.
"""
import os
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import sys
from scipy.sparse import csr_matrix, csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import poisson

# Slice id from CLI (default 1). slice_3 is a tumor-free control -> good for
# verifying the method doesn't hallucinate a tumor-recruited myeloid infiltrate.
SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "1"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
# Same cells as SLICE (verified row-aligned by cell_global_id) but retains the
# 195 CosMx negative-control probes used to calibrate the detection threshold.
NEG_FILE = f"D:/thesis-research/resources/cache/slice_{SLICE_ID}_adata_with_neg.h5ad"
OUT_DIR = f"D:/thesis-research/nontumor_slice{SLICE_ID}"
TUMOR_PRED_COLS = [
    "pred_tumor_LogReg", "pred_tumor_LogReg_PCA",
    "pred_tumor_LogReg_KNN_PCA", "pred_tumor_XGBoost",
]
ALPHA_BG = 0.01   # a gene set is "detected" if P(counts | neg-probe background) < this

# ---------------------------------------------------------------------------
# Signatures. Only genes confirmed present on this panel are listed.
# Microglia-restricted vs pan-myeloid is kept deliberately separate so a
# pan-myeloid gene cannot inflate the microglia score on every myeloid cell.
# Competing anchors (endo/mural/stroma/lymphocyte/neuron) give non-target
# cells somewhere to go instead of being force-collapsed onto our 4 classes.
# ---------------------------------------------------------------------------
SIGNATURES = {
    "microglia":   ["TMEM119", "Cx3cr1", "C1qa", "C1qb", "C1qc", "Csf1r",
                    "Trem2", "Selplg", "Gpr183", "P2rx5", "Cst7", "Aif1"],
    "macro_MDM":   ["Ccr2", "Cd14", "Plac8", "Fcgr4", "Itgax", "Vcan",
                    "S100a8", "S100a9", "Ms4a6b", "Lyz3", "Crip1"],
    "macro_BAM":   ["Mrc1", "Cd163", "Lyve1", "Pf4", "Marco", "Msr1",
                    "Cd5l", "Gas6"],
    "astrocyte":   ["GFAP", "Sox9", "S100b", "Glul", "Sparcl1", "Glud1",
                    "Ntrk2", "Nell2", "Gpx3", "Clu"],
    # ---- competing anchors (not reported as final target labels) ----
    "endothelial": ["Pecam1", "Cdh5", "Flt1", "Kdr", "Esam", "Vwf", "Tie1", "Ldb2"],
    "mural_stroma": ["Rgs5", "Pdgfrb", "Acta2", "Myh11", "Tagln", "Pdgfra",
                     "Col1a1", "Col1a2", "Col3a1", "Dcn", "Lum", "Bgn", "Vtn"],
    "lymphocyte":  ["Cd3d", "Cd3e", "Cd8a", "Cd4", "Cd2", "Nkg7", "Gzmb",
                    "Ms4a1", "Cd79a", "Jchain"],
    "neuron_other": ["Meg3", "Nrxn1", "Nrxn3", "Calb1", "Sst", "Scg5", "Xkr4"],
}
TARGET_LABELS = ["microglia", "macro_MDM", "macro_BAM", "astrocyte"]

# Genes fed to the re-clustering PCA (union of all signatures + a few pan-myeloid
# / reactive markers). Keeping it lineage-focused stops library-size / ambient
# genes from dominating the embedding the way they did on the full panel.
PAN_MYELOID = ["Ptprc", "Cd68", "Tyrobp", "Fcer1g", "Itgam", "Lgals3",
               "Spp1", "Gpnmb", "Apoe", "Cd74"]

N_PCS = 30
LEIDEN_RES = 0.5
MIN_Z = 0.8          # cluster label confidence: z across clusters
MIN_MARGIN = 0.3     # margin over runner-up signature


# ---------------------------------------------------------------------------
# h5py loader (anndata.read_h5ad chokes on the 'cell_global_id' obs column)
# ---------------------------------------------------------------------------
def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = str(node.attrs.get("encoding-type", ""))
        shape = tuple(node.attrs["shape"])
        data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
        M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
            else csr_matrix((data, idx, indptr), shape=shape)
        return M.tocsr()
    return node[...]


def _read_var_names(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_obs_num(h5, col):
    """Numeric obs column (handles the categorical-encoded numeric case)."""
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def load_slice(path):
    print(f"Loading {path} via h5py ...")
    with h5py.File(path, "r") as h5:
        X = _read_X(h5)
        var_names = _read_var_names(h5)
        # NOTE: obsm['spatial'] in these files is FOV-LOCAL (0-4244) and useless
        # for tissue geometry; the real global coords are CenterX/Y_global_px.
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        coords = np.column_stack([cx, cy])
        preds = {c: h5["obs"][c][...].astype(bool) for c in TUMOR_PRED_COLS}
    obs = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    for c, v in preds.items():
        obs[c] = v
    # per-cell negative-probe background rate (lambda per probe), for calibration
    obs["bg_lambda"] = load_bg_lambda(NEG_FILE, n_cells=X.shape[0])
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(var_names, name="gene")
    adata.var_names_make_unique()
    adata.obsm["spatial"] = coords
    print(f"  shape = {adata.shape}; median bg lambda/probe = "
          f"{np.median(obs['bg_lambda']):.4f}")
    return adata


def load_bg_lambda(path, n_cells):
    """Per-cell background rate = (sum of negative-probe counts) / n_neg_probes.

    The _with_neg file is row-aligned with the prediction file (verified by
    cell_global_id), so negative-probe sums attach positionally. lambda is the
    expected non-specific count per probe in that cell -- the noise floor every
    real gene must clear to be called 'detected'.
    """
    with h5py.File(path, "r") as h5:
        vn = _read_var_names(h5)
        neg_idx = [i for i, g in enumerate(vn) if g.lower().startswith("negative")]
        X = _read_X(h5)
        assert X.shape[0] == n_cells, "neg file not row-aligned with pred file"
        neg_sum = np.asarray(X[:, neg_idx].sum(1)).ravel()
    n_neg = len(neg_idx)
    print(f"  calibration: {n_neg} negative probes; mean neg counts/cell = "
          f"{neg_sum.mean():.3f}")
    return neg_sum / max(n_neg, 1)


def resolve(genes, var_names):
    lut = {v.lower(): v for v in var_names}
    return [lut[g.lower()] for g in genes if g.lower() in lut]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    adata = load_slice(SLICE)

    # ---- 1) tumor mask = majority vote of the 4 predictions ----
    votes = np.vstack([adata.obs[c].to_numpy() for c in TUMOR_PRED_COLS]).sum(0)
    tumor = votes >= 2
    print(f"\nTumor calls (majority>=2): {tumor.sum()}/{len(tumor)} "
          f"({100*tumor.mean():.1f}%)")
    for c in TUMOR_PRED_COLS:
        print(f"  {c}: {adata.obs[c].sum()}")

    tumor_xy = adata.obsm["spatial"][tumor]          # keep for spatial validation
    adt = adata[~tumor].copy()
    print(f"Non-tumor cells carried forward: {adt.n_obs}")

    # drop control/negative probes
    keep = ~adt.var_names.str.lower().str.startswith(
        ("neg", "negprb", "blank", "falsecode", "systemcontrol"))
    adt = adt[:, keep].copy()

    # ---- 2) normalize + log1p (X is raw counts) ----
    adt.layers["counts"] = adt.X.copy()
    sc.pp.normalize_total(adt, target_sum=1e4)
    sc.pp.log1p(adt)

    # ---- 3) Approach A: per-cell signature scores ----
    resolved = {k: resolve(v, adt.var_names) for k, v in SIGNATURES.items()}
    for k, g in resolved.items():
        missing = [x for x in SIGNATURES[k] if x not in g]
        print(f"  sig[{k}]: {len(g)} on panel" + (f"  (missing {missing})" if missing else ""))
        sc.tl.score_genes(adt, gene_list=g, score_name=f"score_{k}")
    score_cols = [f"score_{k}" for k in SIGNATURES]
    S = adt.obs[score_cols].copy()
    S.columns = list(SIGNATURES.keys())
    # per-cell argmax across ALL signatures (incl. anchors)
    adt.obs["labelA"] = S.idxmax(axis=1)

    # ---- 4) Approach B: re-cluster on curated lineage genes ----
    clust_genes = sorted(set(sum(resolved.values(), []) +
                             resolve(PAN_MYELOID, adt.var_names)))
    print(f"\nRe-clustering on {len(clust_genes)} curated lineage genes")
    sub = adt[:, clust_genes].copy()
    sc.pp.scale(sub, max_value=10)
    sc.tl.pca(sub, n_comps=min(N_PCS, len(clust_genes) - 1))
    sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
    try:
        sc.tl.leiden(sub, resolution=LEIDEN_RES, flavor="igraph",
                     n_iterations=2, directed=False)
    except Exception as e:
        print(f"  leiden(igraph) failed ({e}); falling back to default flavor")
        sc.tl.leiden(sub, resolution=LEIDEN_RES)
    adt.obs["leiden"] = sub.obs["leiden"].values
    n_clusters = adt.obs["leiden"].nunique()
    print(f"  {n_clusters} Leiden clusters")

    # cluster x signature mean -> z across clusters -> argmax label (gated)
    cm = adt.obs.groupby("leiden", observed=True)[score_cols].mean()
    cm.columns = list(SIGNATURES.keys())
    z = (cm - cm.mean(0)) / (cm.std(0) + 1e-9)
    top = z.idxmax(1)
    topz = z.max(1)
    runner = z.apply(lambda r: r.nlargest(2).iloc[-1], axis=1)
    margin = topz - runner
    confident = (topz >= MIN_Z) & (margin >= MIN_MARGIN)
    cluster_label = top.where(confident, "unassigned")

    cluster_tbl = pd.DataFrame({
        "label": cluster_label, "top_sig": top, "z": topz.round(2),
        "margin": margin.round(2), "n": adt.obs["leiden"].value_counts(),
    }).sort_index()
    cluster_tbl.to_csv(f"{OUT_DIR}/cluster_labels.csv")
    cm.round(3).to_csv(f"{OUT_DIR}/cluster_signature_means.csv")
    print("\n=== cluster -> label ===")
    print(cluster_tbl)

    adt.obs["labelB"] = adt.obs["leiden"].map(cluster_label).astype(str)

    # ---- Ward hierarchical clustering of the cluster x signature matrix ----
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Zlink = linkage(z.values, method="ward")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        dendrogram(Zlink, labels=[f"{i}:{cluster_label[i]}" for i in z.index], ax=ax)
        ax.set_title("Ward hierarchical clustering of Leiden clusters (signature z-space)")
        plt.tight_layout(); plt.savefig(f"{OUT_DIR}/cluster_dendrogram.png"); plt.close()
    except Exception as e:
        print(f"  dendrogram skipped: {e}")

    # ---- 5) A vs B agreement on the 4 target labels ----
    both = adt.obs[["labelA", "labelB"]].copy()
    tgt = both["labelB"].isin(TARGET_LABELS)
    agree = (both.loc[tgt, "labelA"] == both.loc[tgt, "labelB"]).mean()
    print(f"\nA/B agreement on cells where B is a target label: {100*agree:.1f}%")
    comp = adt.obs["labelB"].value_counts()
    frac = 100 * comp / comp.sum()
    print(f"\nlabelB composition (slice {SLICE_ID}, % of non-tumor cells):")
    for lbl in comp.index:
        print(f"  {lbl:14s}: {comp[lbl]:7d}  ({frac[lbl]:5.1f}%)")
    tgt_only = comp[comp.index.isin(TARGET_LABELS)]
    print(f"  --- among the 4 target labels only ---")
    for lbl in TARGET_LABELS:
        n = int(tgt_only.get(lbl, 0))
        print(f"  {lbl:14s}: {n:7d}  ({100*n/max(tgt_only.sum(),1):5.1f}% of targets)")

    # signature heatmap (z)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(z) + 2), dpi=150)
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(z.columns))); ax.set_xticklabels(z.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(z.index)))
    ax.set_yticklabels([f"{i} [{cluster_label[i]}] n={cluster_tbl.loc[i,'n']}" for i in z.index])
    fig.colorbar(im, ax=ax, shrink=0.6, label="z")
    ax.set_title("Leiden cluster x signature (z across clusters)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/signature_heatmap.png"); plt.close()

    # ---- 5b) do labeled cells actually carry their discriminating markers? ----
    label_marker_table(adt)
    abundance_sanity_check(adt)

    # ---- 5c) corrected labels: hierarchical positive gating ----
    hierarchical_gate(adt)
    myeloid_breakdown(adt)

    # ---- 6) spatial validation ----
    spatial_validation(adt, tumor_xy)

    print("\nDone. Outputs in", OUT_DIR)


def hierarchical_gate(adt, alpha=ALPHA_BG):
    """Positive-identification gating on RAW counts, calibrated to the negative
    -control probes (no hand-picked count thresholds).

    A gene set with summed count c (over n genes present) is 'detected' in a cell
    when c is improbably high under the cell's own negative-probe background:
        c ~ Poisson(n * bg_lambda_i)  under the null;  detected if P(X >= c) < alpha.
    bg_lambda_i = (cell's Negative-probe counts) / 11 (the 11 true Negative* probes,
    not the SystemControl ones). So the noise floor is per-cell (noisy cells must
    clear a higher bar) and per-set (larger sets need more counts). This replaces
    the previous fixed >=2 / >=1 thresholds.

    Tier 1 (lineage):
        myeloid  := detected(C1qa+Csf1r+Aif1+Tyrobp+Fcer1g)
        astrocyte:= detected(GFAP+Sparcl1+Glul+S100b) AND not myeloid
        else     -> unassigned   (the panel-blind neuron/oligo majority)
    Tier 2 (within myeloid, priority BAM > MDM > microglia):
        BAM      := detected(Mrc1+Cd163)
        MDM      := detected(Plac8)
        microglia:= detected(P2rx5+TMEM119+Cx3cr1)
        remaining myeloid -> myeloid_unresolved
    """
    counts = adt.layers["counts"]
    var = {g: i for i, g in enumerate(adt.var_names)}
    lam = adt.obs["bg_lambda"].to_numpy()

    def detected(genes):
        idx = [var[g] for g in genes if g in var]
        if not idx:
            return np.zeros(adt.n_obs, dtype=bool), 0
        c = np.asarray(counts[:, idx].sum(1)).ravel()
        # P(X >= c | Poisson(n*lam)) = sf(c-1); detected where this < alpha and c>0
        pval = poisson.sf(c - 1, len(idx) * lam)
        return (c > 0) & (pval < alpha), len(idx)

    is_mye, _ = detected(["C1qa", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"])
    is_astro_raw, _ = detected(["GFAP", "Sparcl1", "Glul", "S100b"])
    is_bam, _ = detected(["Mrc1", "Cd163"])
    is_mdm, _ = detected(["Plac8"])
    is_micro, _ = detected(["P2rx5", "TMEM119", "Cx3cr1"])

    is_astro = is_astro_raw & ~is_mye

    label = np.full(adt.n_obs, "unassigned", dtype=object)
    label[is_astro] = "astrocyte"
    label[is_mye & is_bam] = "macro_BAM"
    rest = is_mye & ~is_bam
    label[rest & is_mdm] = "macro_MDM"
    rest2 = rest & ~is_mdm
    label[rest2 & is_micro] = "microglia"
    label[rest2 & ~is_micro] = "myeloid_unresolved"

    print(f"\n(neg-probe calibrated detection, alpha={alpha}; median bg "
          f"lambda/probe={np.median(lam):.4f})")
    adt.obs["labelC"] = pd.Categorical(label)
    comp = adt.obs["labelC"].value_counts()
    print(f"\n=== corrected hierarchical-gate composition (slice {SLICE_ID}) ===")
    for lbl in comp.index:
        print(f"  {lbl:20s}: {comp[lbl]:7d}  ({100*comp[lbl]/comp.sum():5.1f}%)")
    nmg = comp.get("microglia", 0)
    nmac = comp.get("macro_BAM", 0) + comp.get("macro_MDM", 0)
    print(f"microglia : (BAM+MDM) = {nmg}:{nmac} = {nmg/max(nmac,1):.2f} "
          f"(healthy brain expectation: >>1)")
    adt.obs[["x", "y", "labelC"]].to_csv(f"{OUT_DIR}/labelC_cells.csv")
    return label


def myeloid_breakdown(adt):
    """Interrogate the myeloid_unresolved bucket: real microglia (dropout) or
    ambient-C1qa false positives? Reports mean RAW counts + positivity per label
    for pan-myeloid vs microglia markers. If myeloid_unresolved has C1qa but ~0
    Csf1r/P2rx5/Cx3cr1 (like 'unassigned'), it's ambient contamination inflating
    the lineage gate; if it tracks 'microglia', it's real microglia with dropout.
    """
    counts = adt.layers["counts"]
    var = {g: i for i, g in enumerate(adt.var_names)}
    genes = ["C1qa", "Csf1r", "Aif1", "Tyrobp", "Fcer1g",
             "P2rx5", "TMEM119", "Cx3cr1", "Ptprc"]
    present = [g for g in genes if g in var]
    M = np.asarray(counts[:, [var[g] for g in present]].todense())
    df = pd.DataFrame(M, columns=present)
    df["label"] = adt.obs["labelC"].values
    order = ["microglia", "myeloid_unresolved", "unassigned", "astrocyte"]
    order = [o for o in order if o in set(df["label"])]
    print("\n=== mean RAW counts per label (myeloid interrogation) ===")
    with pd.option_context("display.width", 200, "display.float_format",
                           lambda v: f"{v:.3f}"):
        print(df.groupby("label")[present].mean().loc[order].T)
    mu = df[df["label"] == "myeloid_unresolved"]
    if len(mu):
        print(f"\nmyeloid_unresolved n={len(mu)} — % positive (raw count > 0):")
        for g in present:
            print(f"  {g:9s}: {100*(mu[g] > 0).mean():5.1f}%")
        only_c1qa = ((mu["C1qa"] > 0) &
                     (mu[[g for g in present if g not in ('C1qa',)]].sum(1) == 0)).mean()
        print(f"  driven by C1qa ALONE (no other pan-myeloid/microglia gene): "
              f"{100*only_c1qa:.1f}%")


def abundance_sanity_check(adt):
    """Is the macrophage/MDM abundance even plausible?

    Bounds the myeloid compartment (any of C1qa/Csf1r/Aif1/Ptprc > 0) and asks,
    per label, what fraction of cells are (a) myeloid at all and (b) positive for
    the label's own defining marker. If a 'BAM' cluster is mostly myeloid-negative
    or mostly Cd163-negative, the cells are passengers swept in by cluster-mean
    argmax -- the likely cause of inflated macrophage counts in a tumor-free slice.
    """
    def expr(gene):
        g = resolve([gene], adt.var_names)
        if not g:
            return np.zeros(adt.n_obs)
        return np.asarray(adt[:, g[0]].X.todense()).ravel()

    myeloid_any = np.zeros(adt.n_obs, dtype=bool)
    for g in ["C1qa", "Csf1r", "Aif1", "Ptprc"]:
        myeloid_any |= expr(g) > 0
    print(f"\n=== abundance sanity check (slice {SLICE_ID}) ===")
    print(f"myeloid ceiling (C1qa|Csf1r|Aif1|Ptprc > 0): "
          f"{myeloid_any.sum()} cells ({100*myeloid_any.mean():.1f}% of non-tumor)")

    defining = {"microglia": "P2rx5", "macro_MDM": "Plac8",
                "macro_BAM": "Cd163", "astrocyte": "GFAP"}
    lab = adt.obs["labelB"].to_numpy()
    print(f"{'label':14s} {'n':>7s} {'%myeloid+':>10s} {'defining':>9s} {'%def+':>7s}")
    for lbl in TARGET_LABELS:
        m = lab == lbl
        if not m.any():
            continue
        dpos = expr(defining[lbl])[m] > 0
        print(f"{lbl:14s} {int(m.sum()):7d} {100*myeloid_any[m].mean():9.1f}% "
              f"{defining[lbl]:>9s} {100*dpos.mean():6.1f}%")
    # microglia : macrophage ratio (should be >>1 in healthy brain)
    nmg = (lab == "microglia").sum()
    nmac = (lab == "macro_BAM").sum() + (lab == "macro_MDM").sum()
    print(f"microglia : (BAM+MDM) ratio = {nmg}:{nmac} = {nmg/max(nmac,1):.2f} "
          f"(healthy brain expectation: >>1)")


def label_marker_table(adt):
    """Mean normalized expression of discriminating markers per target label.
    The diagonal should dominate: microglia-labeled cells highest in TMEM119/
    Cx3cr1, MDM-labeled highest in Ccr2/Plac8, etc. If not, the label is noise.
    """
    disc = {
        "microglia": ["TMEM119", "Cx3cr1", "P2rx5", "Selplg"],
        "macro_MDM": ["Ccr2", "Plac8", "S100a8", "Vcan"],
        "macro_BAM": ["Mrc1", "Lyve1", "Cd163"],
        "astrocyte": ["GFAP", "Sparcl1", "Glul"],
    }
    flat = [g for v in disc.values() for g in resolve(v, adt.var_names)]
    sub = adt[:, flat]
    M = pd.DataFrame(np.asarray(sub.X.todense()), columns=flat, index=adt.obs_names)
    M["label"] = adt.obs["labelB"].values
    grp = M.groupby("label").mean()
    grp = grp.loc[grp.index.isin(TARGET_LABELS)]
    print("\n=== mean normalized expr of discriminating markers per label ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:.2f}"):
        print(grp.T)


def spatial_validation(adt, tumor_xy):
    xy = adt.obsm["spatial"]
    x, y = xy[:, 0], xy[:, 1]
    # use the corrected hierarchical-gate labels for spatial maps
    lab = adt.obs["labelC"].to_numpy()

    # distance from every non-tumor cell to the nearest tumor cell
    if len(tumor_xy):
        d_tumor = cKDTree(tumor_xy).query(xy, k=1)[0]
    else:
        d_tumor = np.full(len(xy), np.nan)
    adt.obs["dist_to_tumor"] = d_tumor

    colors = {"microglia": "#1f77b4", "macro_MDM": "#d62728",
              "macro_BAM": "#ff7f0e", "astrocyte": "#2ca02c"}

    # (a) per-label spatial maps with tumor mass underneath
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), dpi=160)
    for ax, lbl in zip(axes.ravel(), TARGET_LABELS):
        ax.scatter(tumor_xy[:, 0], -tumor_xy[:, 1], s=1, c="0.85", linewidths=0,
                   rasterized=True, label="tumor")
        ax.scatter(x, -y, s=0.5, c="0.93", linewidths=0, rasterized=True)
        m = lab == lbl
        ax.scatter(x[m], -y[m], s=3, c=colors[lbl], linewidths=0, rasterized=True)
        ax.set_title(f"{lbl}  (n={int(m.sum())})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"slice_{SLICE_ID} non-tumor labels (hierarchical gate) over tumor (grey)",
                 fontsize=14)
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/spatial_labels.png"); plt.close()

    # (b) distance-to-tumor enrichment: fraction of each label per distance bin
    if np.isfinite(d_tumor).any():
        bins = np.quantile(d_tumor, np.linspace(0, 1, 11))
        bins = np.unique(bins)
        binid = np.clip(np.digitize(d_tumor, bins[1:-1]), 0, len(bins) - 2)
        centers = 0.5 * (bins[:-1] + bins[1:])
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        for lbl in TARGET_LABELS:
            frac = [np.mean(lab[binid == b] == lbl) for b in range(len(centers))]
            ax.plot(centers, frac, "-o", color=colors[lbl], label=lbl)
        ax.set_xlabel("distance to nearest tumor cell (px)")
        ax.set_ylabel("fraction of cells in distance bin")
        ax.set_title("Label composition vs distance to tumor\n"
                     "(expect MDM high near 0; astrocyte peak at rim; microglia flatter)")
        ax.legend(); plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/distance_to_tumor.png"); plt.close()

        # numeric summary: median distance per label
        print("\nMedian distance-to-tumor by label (px):")
        for lbl in TARGET_LABELS:
            dd = d_tumor[lab == lbl]
            if len(dd):
                print(f"  {lbl:12s}: median={np.median(dd):8.1f}  "
                      f"q25={np.quantile(dd,0.25):8.1f}  n={len(dd)}")


if __name__ == "__main__":
    main()
