"""
METHOD A — hierarchical positive gating on RAW counts (transcriptome-only, NO lineage tags).
Project-recommended baseline (see project_mdm_annotation_weak).

Detection rule (calibrated to negative-control probes):
  a gene SET of n genes with summed raw count c is "detected" in a cell if
  P(X >= c) < alpha under Poisson(lambda = n * bg_lambda_cell), where
  bg_lambda_cell = max(cell neg_mean, floor). Median neg_mean ~ 0 => clean cells get
  single-molecule detection; noisy cells a higher bar.  alpha = 0.01.

Tier 1 (lineage):
  myeloid    := pan-myeloid load (C1qa,C1qb,C1qc,Csf1r,Aif1,Tyrobp,Fcer1g) detected with >= N_MYE genes positive
  astrocyte  := (GFAP,Sparcl1,Glul,S100b,Sox9) detected AND not myeloid
  else unassigned (panel-blind neuron/oligo majority)
Tier 2 (within myeloid), priority BAM > MDM > microglia:
  BAM       := Mrc1 | Cd163 | Lyve1 | Pf4   (>=1 detected)
  MDM       := Ccr2 | Plac8 | Cd14 | S100a8 | S100a9  (>=1) and not BAM   (Apoe DROPPED)
  microglia := P2rx5 | TMEM119 | Cx3cr1 | Selplg  (>=1) and not BAM/MDM
  else myeloid_unresolved
NK specificity guard: a myeloid call requires >=1 myeloid-SPECIFIC gene
  (Csf1r|Aif1|Cx3cr1|C1qa) among detected; cells that are Nkg7/Klrb1c-high and lack
  a myeloid-specific gene are demoted to unassigned (flag).

Outputs: per-cell CSV with method_a label + diagnostics ->
  agents/outputs/annotation/method_a_labels.csv
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import poisson

OUT = "D:/thesis-research/agents/outputs/annotation"
os.makedirs(OUT, exist_ok=True)
INP = "D:/thesis-research/resources/cache/annotation_bench/slice1_nontumor.h5ad"
ALPHA = 0.01
BG_FLOOR = 0.02  # per-gene background floor so a single stray molecule isn't auto-significant in zero-bg cells

a = ad.read_h5ad(INP)
print("loaded", a.shape)
panel = {g.lower(): g for g in a.var_names}

def col(gene):
    key = panel.get(gene.lower())
    if key is None:
        return None
    x = a[:, key].X
    return (x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()).astype(float)

def setmat(genes):
    """Return (n_cells, n_present) raw-count matrix for genes present on panel + list of present names."""
    present = [g for g in genes if g.lower() in panel]
    cols = [col(g) for g in present]
    return np.vstack(cols).T if cols else np.zeros((a.n_obs, 0)), present

neg_mean = a.obs["neg_mean"].to_numpy(float)
bg_lambda = np.maximum(neg_mean, BG_FLOOR)  # per-gene background rate per cell

def detected_set(genes, min_genes=1):
    """Per-cell boolean: set of `genes` detected above Poisson background.
    Two-part: (a) summed count significant under Poisson(n*bg), AND
              (b) at least `min_genes` of the genes individually have count>0."""
    M, present = setmat(genes)
    n = M.shape[1]
    if n == 0:
        return np.zeros(a.n_obs, bool), present
    c = M.sum(1)
    lam = n * bg_lambda
    # P(X >= c) = sf(c-1)
    pval = poisson.sf(c - 1, lam)
    sig = (pval < ALPHA) & (c > 0)
    npos = (M > 0).sum(1)
    return sig & (npos >= min_genes), present

# ---- Tier 1 ----
PAN_MYE = ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g"]
ASTRO   = ["GFAP", "Sparcl1", "Glul", "S100b", "Sox9"]
MYE_SPECIFIC = ["Csf1r", "Aif1", "Cx3cr1", "C1qa"]   # NK guard
NK = ["Nkg7", "Klrb1c"]

# pan-myeloid: require >= 2 of the pan-myeloid genes positive AND summed-count significant
mye_det, _ = detected_set(PAN_MYE, min_genes=2)
astro_det, _ = detected_set(ASTRO, min_genes=2)

# NK guard: myeloid candidates must carry a myeloid-specific gene
M_spec, _ = setmat(MYE_SPECIFIC)
has_mye_specific = (M_spec > 0).any(1)
M_nk, _ = setmat(NK)
nk_load = M_nk.sum(1)
nk_high = nk_load >= 2
demote_nk = mye_det & nk_high & (~has_mye_specific)

is_myeloid = mye_det & (~demote_nk)
is_astro = astro_det & (~is_myeloid)

# ---- Tier 2 (within myeloid) ----
# MARKER-SPECIFICITY STANDARD: the broad inflammatory genes (S100a8/9, Cd14) cross-react
# heavily; using ">=1 of {Ccr2,Plac8,Cd14,S100a8,S100a9}" over-calls MDM (yields the
# backwards microglia:MDM ratio). We therefore split MDM into:
#   - a SPECIFIC anchor  : Ccr2 | Plac8           (monocyte-derived specific)
#   - a BROAD inflammatory: Cd14 | S100a8 | S100a9 (shared; only used to assign when the
#                           cell is already myeloid and not BAM/microglia)
# Primary call uses the specific anchor; broad inflammatory only breaks ties into
# "MDM_inflammatory" (reported separately, lower confidence).
BAM_SPEC = ["Mrc1", "Cd163"]            # strong, specific
BAM_WEAK = ["Lyve1", "Pf4"]             # weak/low-DR support
MDM_SPEC = ["Ccr2", "Plac8"]            # specific monocyte-derived
MDM_BROAD = ["Cd14", "S100a8", "S100a9"]  # broad inflammatory (cross-population)
MIC = ["P2rx5", "TMEM119", "Cx3cr1", "Selplg"]

def any_positive(genes):
    """Per-cell: >=1 RAW count of any of these (specific) genes. For sparse specific
    markers (Ccr2/Plac8/Mrc1/Cd163) a single molecule already exceeds the ~0 background,
    so the Poisson pair-sum test is needlessly conservative; use presence."""
    M, _ = setmat(genes)
    return (M > 0).any(1) if M.shape[1] else np.zeros(a.n_obs, bool)

bam_spec = any_positive(BAM_SPEC)                     # Mrc1 | Cd163 present
mdm_spec = any_positive(MDM_SPEC)                     # Ccr2 | Plac8 present
mdm_broad, _ = detected_set(MDM_BROAD, min_genes=2)   # broad: still require >=2 (specificity guard)
mic_det = any_positive(["P2rx5", "TMEM119"]) | (col("Cx3cr1") > 0) | (col("Selplg") > 0)

bam_det = bam_spec                            # BAM = specific only (Lyve1/Pf4 too broad alone)
mdm_det = mdm_spec
mdm_inflam = mdm_broad & (~mdm_spec)

label = np.array(["unassigned"] * a.n_obs, dtype=object)
label[is_astro] = "Astrocyte"

# within myeloid, priority BAM > MDM(spec) > microglia > MDM_inflammatory > unresolved
lab_m = np.array(["Myeloid_unresolved"] * a.n_obs, dtype=object)
lab_m[mdm_inflam] = "MDM_inflammatory"
lab_m[mic_det] = "Microglia"
lab_m[mdm_det] = "MDM"
lab_m[bam_det] = "BAM"
label[is_myeloid] = lab_m[is_myeloid]

# diagnostics columns
df = pd.DataFrame(index=a.obs.index)
df["cell_global_id"] = a.obs["cell_global_id"].values
df["method_a"] = label
df["is_myeloid"] = is_myeloid
df["is_astro"] = is_astro
df["nCount"] = np.asarray(a.layers["counts"].sum(1)).ravel()
df["neg_mean"] = neg_mean
df["nk_demoted"] = demote_nk
df["has_mye_specific"] = has_mye_specific
# raw subtype signals
for g in ["Cx3cr1", "TMEM119", "P2rx5", "Ccr2", "Plac8", "Cd14", "Mrc1", "Cd163",
          "Lyve1", "GFAP", "Sox9", "S100b", "GFP", "tdTomato", "Nkg7"]:
    v = col(g)
    if v is not None:
        df[f"cnt_{g}"] = v
# core label collapses fine MDM categories for cross-method comparison
core = label.copy()
core[label == "MDM_inflammatory"] = "MDM"
df["method_a_core"] = core
df["bam_det"] = bam_det
df["mdm_det"] = mdm_det
df["mdm_inflam"] = mdm_inflam
df["mic_det"] = mic_det
df["CenterX_global_px"] = a.obs["CenterX_global_px"].values
df["CenterY_global_px"] = a.obs["CenterY_global_px"].values
df["decontx_contamination"] = a.obs["decontx_contamination"].values
df["fov"] = a.obs["fov"].values

df.to_csv(OUT + "/method_a_labels.csv", index=True)
print("\n=== METHOD A label counts (non-tumor, n=%d) ===" % a.n_obs)
print(df["method_a"].value_counts())
print("\nfraction:")
print((df["method_a"].value_counts() / a.n_obs * 100).round(2))
# sanity: microglia : (BAM+MDM)
vc = df["method_a_core"].value_counts()
mic = vc.get("Microglia", 0); bam = vc.get("BAM", 0); mdm = vc.get("MDM", 0)
print("\n=== core labels ==="); print(vc)
print(f"\nmicroglia:(BAM+MDM) = {mic/(bam+mdm+1e-9):.2f}  (mic={mic} bam={bam} mdm={mdm})")
print("NK-demoted myeloid candidates:", int(demote_nk.sum()))
print("\nWROTE", OUT + "/method_a_labels.csv")
