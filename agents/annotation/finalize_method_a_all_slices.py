"""
FINALIZE — Method A (hierarchical positive gating on raw counts) across all 6 slices.
Faithful port of agents/annotation/01_method_a_gating.py, looped over slices 1-6.

Per slice: load with_tumor_prediction/slice_N (raw counts), pull Negative* control
probes from slice_N_adata_with_neg.h5ad (per-cell neg_mean = RNA background, aligned by
cell_global_id), restrict to non-tumor, gate. Non-destructive: writes a labels CSV +
summary + a 6-panel spatial map; does NOT modify any cache h5ad.

Run: conda run -n thesis_research python agents/annotation/finalize_method_a_all_slices.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import poisson

BASE = "D:/thesis-research/resources/cache/"
OUT = "D:/thesis-research/agents/outputs/annotation/final"
os.makedirs(OUT, exist_ok=True)
ALPHA, BG_FLOOR = 0.01, 0.02

PAN_MYE = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
ASTRO = ["GFAP", "Sparcl1", "Glul", "S100b", "Sox9"]
MYE_SPECIFIC = ["Csf1r", "Aif1", "Cx3cr1", "C1qa"]
NK = ["Nkg7", "Klrb1c"]
BAM_SPEC = ["Mrc1", "Cd163"]
MDM_SPEC = ["Ccr2", "Plac8"]
MDM_BROAD = ["Cd14", "S100a8", "S100a9"]


def gate_method_a(a):
    panel = {g.lower(): g for g in a.var_names}

    def col(gene):
        key = panel.get(gene.lower())
        if key is None:
            return None
        x = a[:, key].X
        return (x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()).astype(float)

    def setmat(genes):
        present = [g for g in genes if g.lower() in panel]
        cols = [col(g) for g in present]
        return (np.vstack(cols).T if cols else np.zeros((a.n_obs, 0))), present

    neg_mean = np.nan_to_num(a.obs["neg_mean"].to_numpy(float), nan=0.0)
    bg_lambda = np.maximum(neg_mean, BG_FLOOR)

    def detected_set(genes, min_genes=1):
        M, _ = setmat(genes)
        n = M.shape[1]
        if n == 0:
            return np.zeros(a.n_obs, bool)
        c = M.sum(1)
        pval = poisson.sf(c - 1, n * bg_lambda)
        return (pval < ALPHA) & (c > 0) & ((M > 0).sum(1) >= min_genes)

    def any_positive(genes):
        M, _ = setmat(genes)
        return (M > 0).any(1) if M.shape[1] else np.zeros(a.n_obs, bool)

    mye_det = detected_set(PAN_MYE, 2)
    astro_det = detected_set(ASTRO, 2)
    M_spec, _ = setmat(MYE_SPECIFIC)
    has_mye_specific = (M_spec > 0).any(1)
    M_nk, _ = setmat(NK)
    demote_nk = mye_det & (M_nk.sum(1) >= 2) & (~has_mye_specific)
    is_myeloid = mye_det & (~demote_nk)
    is_astro = astro_det & (~is_myeloid)

    bam_det = any_positive(BAM_SPEC)
    mdm_det = any_positive(MDM_SPEC)
    mdm_inflam = detected_set(MDM_BROAD, 2) & (~mdm_det)
    cx, sel = col("Cx3cr1"), col("Selplg")
    mic_det = any_positive(["P2rx5", "TMEM119"])
    if cx is not None:
        mic_det = mic_det | (cx > 0)
    if sel is not None:
        mic_det = mic_det | (sel > 0)

    label = np.array(["unassigned"] * a.n_obs, dtype=object)
    label[is_astro] = "Astrocyte"
    lab_m = np.array(["Myeloid_unresolved"] * a.n_obs, dtype=object)
    lab_m[mdm_inflam] = "MDM_inflammatory"
    lab_m[mic_det] = "Microglia"
    lab_m[mdm_det] = "MDM"
    lab_m[bam_det] = "BAM"
    label[is_myeloid] = lab_m[is_myeloid]
    core = label.copy()
    core[label == "MDM_inflammatory"] = "MDM"
    return label, core


def load_slice_nontumor(s):
    a = ad.read_h5ad(f"{BASE}with_tumor_prediction/slice_{s}_adata.h5ad")
    wn = ad.read_h5ad(f"{BASE}slice_{s}_adata_with_neg.h5ad")
    neg = [g for g in wn.var_names if g.lower().startswith("negative")]
    nx = wn[:, neg].X
    nx = nx.toarray() if sp.issparse(nx) else np.asarray(nx)
    neg_by_gid = pd.Series(nx.mean(1), index=wn.obs["cell_global_id"].values)
    a.obs["neg_mean"] = neg_by_gid.reindex(a.obs["cell_global_id"].values).values
    nt = a[~a.obs["pred_tumor_XGBoost"].astype(bool)].copy()
    return nt, len(neg)


CORE = ["Microglia", "MDM", "BAM", "Astrocyte", "Myeloid_unresolved", "unassigned"]
parts = []
fig, axes = plt.subplots(2, 3, figsize=(22, 13), dpi=120)
colors = {"Microglia": "#1f77b4", "MDM": "#d62728", "BAM": "#ff7f0e", "Astrocyte": "#2ca02c"}
focal = ["Microglia", "MDM", "BAM", "Astrocyte"]

for i, s in enumerate(range(1, 7)):
    nt, n_neg = load_slice_nontumor(s)
    label, core = gate_method_a(nt)
    df = pd.DataFrame({
        "cell_global_id": nt.obs["cell_global_id"].values,
        "slice": s,
        "method_a": label,
        "method_a_core": core,
        "CenterX_global_px": nt.obs["CenterX_global_px"].values,
        "CenterY_global_px": nt.obs["CenterY_global_px"].values,
        "fov": nt.obs["fov"].values,
    })
    parts.append(df)
    print(f"\n=== slice {s} (non-tumor n={nt.n_obs:,}, {n_neg} neg probes) ===")
    print(df["method_a"].value_counts().to_string())

    ax = axes.ravel()[i]
    x = df["CenterX_global_px"].to_numpy(float)
    y = df["CenterY_global_px"].to_numpy(float)
    lab = df["method_a_core"].to_numpy(str)
    oth = ~np.isin(lab, focal)
    ax.scatter(x[oth], y[oth], s=0.6, c="#dddddd", alpha=0.45, linewidths=0)
    for cls in focal:
        m = lab == cls
        ax.scatter(x[m], y[m], s=2.0, c=colors[cls], alpha=0.8, linewidths=0, label=cls)
    ax.set_title(f"slice {s} (n={nt.n_obs:,})")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend(markerscale=5, loc="upper right", fontsize=8)

plt.suptitle("Method A annotation — all 6 slices (non-tumor)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/method_a_all_slices_spatial.png", bbox_inches="tight")

alldf = pd.concat(parts, ignore_index=True)
alldf.to_csv(f"{OUT}/method_a_all_slices_labels.csv", index=False)

# summary table: counts + % per slice (core labels)
summary = (alldf.groupby("slice")["method_a_core"].value_counts().unstack(fill_value=0))
summary = summary.reindex(columns=CORE, fill_value=0)
summary["total"] = summary.sum(1)
pct = summary[CORE].div(summary["total"], axis=0) * 100
summary.to_csv(f"{OUT}/method_a_summary_counts.csv")
pct.round(2).to_csv(f"{OUT}/method_a_summary_pct.csv")

print("\n\n================ SUMMARY: counts per slice (core) ================")
print(summary.to_string())
print("\n================ SUMMARY: % per slice ================")
print(pct.round(1).to_string())
mic = summary["Microglia"]; bm = summary["BAM"] + summary["MDM"]
print("\nmicroglia:(BAM+MDM) per slice:")
print((mic / bm.replace(0, np.nan)).round(2).to_string())
print(f"\nTOTAL cells annotated: {len(alldf):,}")
print("WROTE:", f"{OUT}/method_a_all_slices_labels.csv",
      f"{OUT}/method_a_summary_*.csv", f"{OUT}/method_a_all_slices_spatial.png", sep="\n  ")
