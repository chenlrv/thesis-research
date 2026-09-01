"""
FINALIZE v2 — Method A across all 6 slices with the TIGHTENED MDM/BAM gate.

Change vs v1: the MDM-specific (Ccr2|Plac8) and BAM-specific (Mrc1|Cd163) gates now use
the Poisson background test (detected_set) instead of bare presence (any_positive), so a
single ambient molecule no longer triggers a call. Everything else is identical.

Computes BOTH the old (presence) and new (strict) labels and writes a side-by-side
comparison, plus the corrected official labels/summary/map. Non-destructive to the cache.

Run: conda run -n thesis_research python agents/annotation/finalize_method_a_v2_strict.py
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
TUMOR = {1, 2, 5, 6}

PAN_MYE = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
ASTRO = ["GFAP", "Sparcl1", "Glul", "S100b", "Sox9"]
MYE_SPECIFIC = ["Csf1r", "Aif1", "Cx3cr1", "C1qa"]
NK = ["Nkg7", "Klrb1c"]
BAM_SPEC = ["Mrc1", "Cd163"]
MDM_SPEC = ["Ccr2", "Plac8"]
MDM_BROAD = ["Cd14", "S100a8", "S100a9"]


def gate_method_a(a, strict_mdm_bam):
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
        return (np.vstack(cols).T if cols else np.zeros((a.n_obs, 0)))

    neg_mean = np.nan_to_num(a.obs["neg_mean"].to_numpy(float), nan=0.0)
    bg_lambda = np.maximum(neg_mean, BG_FLOOR)

    def detected_set(genes, min_genes=1):
        M = setmat(genes)
        n = M.shape[1]
        if n == 0:
            return np.zeros(a.n_obs, bool)
        c = M.sum(1)
        return (poisson.sf(c - 1, n * bg_lambda) < ALPHA) & (c > 0) & ((M > 0).sum(1) >= min_genes)

    def any_positive(genes):
        M = setmat(genes)
        return (M > 0).any(1) if M.shape[1] else np.zeros(a.n_obs, bool)

    mye_det = detected_set(PAN_MYE, 2)
    astro_det = detected_set(ASTRO, 2)
    has_mye_specific = (setmat(MYE_SPECIFIC) > 0).any(1)
    demote_nk = mye_det & (setmat(NK).sum(1) >= 2) & (~has_mye_specific)
    is_myeloid = mye_det & (~demote_nk)
    is_astro = astro_det & (~is_myeloid)

    # THE FIX: strict => Poisson background test; legacy => bare presence
    if strict_mdm_bam:
        bam_det = detected_set(BAM_SPEC, 1)
        mdm_det = detected_set(MDM_SPEC, 1)
    else:
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
    return core


def load_slice_nontumor(s):
    a = ad.read_h5ad(f"{BASE}with_tumor_prediction/slice_{s}_adata.h5ad")
    wn = ad.read_h5ad(f"{BASE}slice_{s}_adata_with_neg.h5ad")
    neg = [g for g in wn.var_names if g.lower().startswith("negative")]
    nx = wn[:, neg].X
    nx = nx.toarray() if sp.issparse(nx) else np.asarray(nx)
    neg_by_gid = pd.Series(nx.mean(1), index=wn.obs["cell_global_id"].values)
    a.obs["neg_mean"] = neg_by_gid.reindex(a.obs["cell_global_id"].values).values
    return a[~a.obs["pred_tumor_XGBoost"].astype(bool)].copy()


CORE = ["Microglia", "MDM", "BAM", "Astrocyte", "Myeloid_unresolved", "unassigned"]
focal = ["Microglia", "MDM", "BAM", "Astrocyte"]
colors = {"Microglia": "#1f77b4", "MDM": "#d62728", "BAM": "#ff7f0e", "Astrocyte": "#2ca02c"}
parts, comp = [], []
fig, axes = plt.subplots(2, 3, figsize=(22, 13), dpi=120)

for i, s in enumerate(range(1, 7)):
    nt = load_slice_nontumor(s)
    core_presence = gate_method_a(nt, strict_mdm_bam=False)
    core_strict = gate_method_a(nt, strict_mdm_bam=True)
    n = nt.n_obs
    df = pd.DataFrame({
        "cell_global_id": nt.obs["cell_global_id"].values,
        "slice": s,
        "method_a_core": core_strict,
        "method_a_presence": core_presence,
        "CenterX_global_px": nt.obs["CenterX_global_px"].values,
        "CenterY_global_px": nt.obs["CenterY_global_px"].values,
        "fov": nt.obs["fov"].values,
    })
    parts.append(df)
    comp.append({
        "slice": s, "type": "TUMOR" if s in TUMOR else "CONTROL", "n_nontumor": n,
        "MDM%_presence": round((core_presence == "MDM").mean() * 100, 2),
        "MDM%_strict": round((core_strict == "MDM").mean() * 100, 2),
        "BAM%_presence": round((core_presence == "BAM").mean() * 100, 2),
        "BAM%_strict": round((core_strict == "BAM").mean() * 100, 2),
        "Microglia%": round((core_strict == "Microglia").mean() * 100, 2),
    })
    print(f"slice {s} done")

    ax = axes.ravel()[i]
    x = df["CenterX_global_px"].to_numpy(float); y = df["CenterY_global_px"].to_numpy(float)
    lab = df["method_a_core"].to_numpy(str)
    oth = ~np.isin(lab, focal)
    ax.scatter(x[oth], y[oth], s=0.6, c="#dddddd", alpha=0.45, linewidths=0)
    for cls in focal:
        m = lab == cls
        ax.scatter(x[m], y[m], s=2.0, c=colors[cls], alpha=0.8, linewidths=0, label=cls)
    ax.set_title(f"slice {s} ({'TUMOR' if s in TUMOR else 'CONTROL'}, n={n:,})")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.legend(markerscale=5, loc="upper right", fontsize=8)

plt.suptitle("Method A (strict MDM/BAM gate) — all 6 slices", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/method_a_all_slices_spatial_strict.png", bbox_inches="tight")

alldf = pd.concat(parts, ignore_index=True)
alldf.to_csv(f"{OUT}/method_a_all_slices_labels_strict.csv", index=False)
cmp = pd.DataFrame(comp)
cmp.to_csv(f"{OUT}/mdm_bam_gate_comparison.csv", index=False)

# strict summary
summary = alldf.groupby("slice")["method_a_core"].value_counts().unstack(fill_value=0).reindex(columns=CORE, fill_value=0)
summary["total"] = summary.sum(1)
(summary[CORE].div(summary["total"], axis=0) * 100).round(2).to_csv(f"{OUT}/method_a_summary_pct_strict.csv")

pd.set_option("display.width", 220, "display.max_columns", 30)
print("\n=========== MDM/BAM gate: PRESENCE vs STRICT (Poisson) ===========\n")
print(cmp.to_string(index=False))
print("\nControl-slice MDM reduction:")
for _, r in cmp[cmp.type == "CONTROL"].iterrows():
    drop = (1 - r["MDM%_strict"] / r["MDM%_presence"]) * 100 if r["MDM%_presence"] else 0
    print(f"  slice {int(r['slice'])}: MDM {r['MDM%_presence']}% -> {r['MDM%_strict']}%  (-{drop:.0f}%)")
print(f"\nTOTAL cells: {len(alldf):,}")
print("WROTE strict labels/summary/comparison/map to", OUT)
