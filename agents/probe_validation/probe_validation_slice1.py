"""
Probe-validation analysis — SLICE 1 only.
Run: conda run -n thesis_research python agents/probe_validation/probe_validation_slice1.py

Deliverables (-> agents/outputs/probe_validation/):
  1. Reproduce F1 (GFP<->tdTomato), F2 (GFP<->Cx3cr1) threshold sweep, F3 (Lyve1 breadth)
     on slice 1 (decontx raw .X, non-tumor cells).
  2. SIGNAL-TO-BACKGROUND per probe vs negative-control probes (with_neg h5ad).
  3. AMBIENT decomposition stratified by decontx_contamination + total-count / size dependence.
  4. CO-EXPRESSION matrix of 8 custom probes vs canonical markers.
  5. Per-probe verdicts (printed; written into report by companion step).
"""
import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"d:/thesis-research/agents/outputs/probe_validation"
os.makedirs(OUT, exist_ok=True)

WITH_NEG = r"d:/thesis-research/resources/cache/slice_1_adata_with_neg.h5ad"
DECONTX = r"d:/thesis-research/resources/cache/decontx/slice_1_decontx.h5ad"

CUSTOM = ["Ccl2", "Cxcl13", "GFAP", "Lyve1", "TMEM119", "Trem2", "GFP", "tdTomato"]
# GFAP is the high-dynamic-range positive control (custom).
# Stock reference markers present on panel (Snap25 absent -> Meg3 as pan-neuron):
REFS = ["Cx3cr1", "Pecam1", "Meg3", "Csf1r"]
# Canonical markers for the co-expression matrix
CANONICAL = ["Cx3cr1", "Csf1r", "Aif1", "Itgam",       # microglia/myeloid
             "Mrc1", "Cd163", "Pf4",                   # BAM
             "Pecam1", "Flt1", "Vtn", "Pdgfrb", "Rgs5",# vascular/mural
             "Meg3", "S100b",                          # neuron / astro-ish
             "Ptprc", "Cd3e", "Nkg7", "Mki67", "Pdgfra"]


def to_dense(x):
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()


def col(a, gene):
    if gene not in a.var_names:
        return None
    return to_dense(a[:, gene].X).astype(np.float64)


def pearson(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 3:
        return float("nan")
    xm = x - x.mean(); ym = y - y.mean()
    sx = float(np.sqrt((xm*xm).sum())); sy = float(np.sqrt((ym*ym).sum()))
    if sx == 0 or sy == 0:
        return float("nan")
    return float((xm*ym).sum() / (sx*sy))


def spearman(x, y):
    # rank then pearson (avoid scipy BLAS edge-cases; ties via average rank)
    xr = pd.Series(x).rank().to_numpy()
    yr = pd.Series(y).rank().to_numpy()
    return pearson(xr, yr)


def tumor_mask(a):
    tv = a.obs["pred_tumor_XGBoost"]
    if tv.dtype == bool:
        return tv.to_numpy()
    if np.issubdtype(tv.dtype, np.number):
        return tv.to_numpy() > 0.5
    return tv.astype(str).str.lower().isin(["1", "true", "tumor"]).to_numpy()


# =====================================================================
# Load
# =====================================================================
print("Loading slice 1 ...")
awn = ad.read_h5ad(WITH_NEG)          # raw counts + neg probes (124956 x 1153)
adx = ad.read_h5ad(DECONTX)           # raw counts (958) + decontx_contamination + tumor pred

# Align: decontx is the canonical non-tumor source (has tumor pred + contamination).
# with_neg lacks tumor pred, so we map the tumor mask by cell order if they match,
# else by intersection of an id column. They share n_obs=124956 — verify ordering.
print(f"with_neg: {awn.shape}  decontx: {adx.shape}")
same_order = (awn.n_obs == adx.n_obs)
if same_order:
    # sanity: total counts on shared genes should correlate ~1 if same order
    shared = [g for g in adx.var_names if g in awn.var_names][:50]
    t_wn = np.asarray(awn[:, shared].X.sum(axis=1)).ravel()
    t_dx = np.asarray(adx[:, shared].X.sum(axis=1)).ravel()
    r_order = pearson(t_wn, t_dx)
    print(f"order-check pearson(total over 50 shared genes) = {r_order:+.4f}")
    same_order = r_order > 0.95

if not same_order:
    raise RuntimeError("with_neg and decontx are NOT row-aligned; need an id join (not implemented).")

# tumor mask from decontx, applied to both
nt = ~tumor_mask(adx)
print(f"non-tumor cells: {int(nt.sum()):,} / {adx.n_obs:,}")

cont = adx.obs["decontx_contamination"].to_numpy(np.float64)

# =====================================================================
# Signal-to-background table (with_neg)
# =====================================================================
print("\n=== S/N table (negative-control probes) ===")
neg_names = [v for v in awn.var_names if v.lower().startswith("negative")]
print(f"negative-control probes used (n={len(neg_names)}): {neg_names}")

neg_X = awn[:, neg_names].X
neg_X = neg_X.todense() if sp.issparse(neg_X) else np.asarray(neg_X)
neg_X = np.asarray(neg_X, np.float64)
# per-cell background = mean negative-probe count (per probe), then on non-tumor cells
bg_per_cell = neg_X.mean(axis=1).A1 if hasattr(neg_X.mean(axis=1), "A1") else neg_X.mean(axis=1)
bg_per_cell = np.asarray(bg_per_cell, np.float64).ravel()

# global mean background per probe (lambda for a single negative probe)
neg_mean_global_nt = float(neg_X[nt].mean())   # mean count per neg probe per cell
print(f"mean negative-probe background (non-tumor) = {neg_mean_global_nt:.4f} counts/probe/cell")
print(f"per-cell background mean (non-tumor) = {bg_per_cell[nt].mean():.4f}")

probes_for_table = CUSTOM + [r for r in REFS if r not in CUSTOM]
rows = []
for g in probes_for_table:
    x = col(awn, g)
    if x is None:
        print(f"  !! {g} not in with_neg panel; skipping")
        continue
    xn = x[nt]
    mean_signal = float(xn.mean())
    max_signal = float(xn.max())
    pct_pos = float((xn > 0).mean() * 100)
    # S/N vs global negative-probe mean
    sn = mean_signal / neg_mean_global_nt if neg_mean_global_nt > 0 else np.nan
    # % cells where this probe's count exceeds that cell's mean neg background
    pct_above_bg = float((xn > bg_per_cell[nt]).mean() * 100)
    # fraction of signal that is above background mass: (mean - bg)/mean
    frac_above_bg_mass = (mean_signal - neg_mean_global_nt) / mean_signal if mean_signal > 0 else np.nan
    rows.append(dict(
        probe=g,
        type=("custom" if g in CUSTOM else "stock_ref"),
        role=("positive_control" if g == "GFAP" else
              ("custom" if g in CUSTOM else "reference")),
        mean_counts=round(mean_signal, 4),
        max_counts=int(max_signal),
        pct_cells_pos=round(pct_pos, 2),
        S_to_N=round(sn, 2),
        pct_cells_above_bg=round(pct_above_bg, 2),
        frac_signal_above_bg=round(frac_above_bg_mass, 3),
    ))

sn_df = pd.DataFrame(rows)
# order: custom first (GFAP control highlighted), then refs
order = {g: i for i, g in enumerate(CUSTOM)}
sn_df["__o"] = sn_df["probe"].map(lambda p: order.get(p, 100 + REFS.index(p) if p in REFS else 999))
sn_df = sn_df.sort_values("__o").drop(columns="__o").reset_index(drop=True)
sn_df.to_csv(os.path.join(OUT, "signal_to_background_slice1.csv"), index=False)
print(sn_df.to_string(index=False))

# =====================================================================
# F1 / F2 / F3 reproduction on decontx raw .X (non-tumor)
# =====================================================================
print("\n=== F1/F2/F3 (decontx raw .X, non-tumor) ===")
sub = adx[nt]
gfp = col(sub, "GFP"); tdt = col(sub, "tdTomato"); cx3 = col(sub, "Cx3cr1")
lyve = col(sub, "Lyve1"); mrc1 = col(sub, "Mrc1"); cd163 = col(sub, "Cd163")
tmem = col(sub, "TMEM119")
total = np.asarray(sub.X.sum(axis=1)).ravel().astype(np.float64)
cont_nt = cont[nt]

findings = {}

# F1
f1 = {"full_r": pearson(gfp[(gfp > 0) | (tdt > 0)], tdt[(gfp > 0) | (tdt > 0)])}
sweep1 = {}
for thr in (1, 2, 3, 5):
    mm = (gfp >= thr) | (tdt >= thr)
    sweep1[thr] = {"n": int(mm.sum()), "r": pearson(gfp[mm], tdt[mm])}
f1["sweep"] = sweep1
f1["copos"] = int(((gfp > 0) & (tdt > 0)).sum())
f1["gfp_only"] = int(((gfp > 0) & (tdt == 0)).sum())
f1["tdt_only"] = int(((gfp == 0) & (tdt > 0)).sum())
findings["F1_GFP_tdTomato"] = f1
print("[F1] GFP vs tdTomato  full r =", round(f1["full_r"], 3),
      "sweep:", {k: (v["n"], round(v["r"], 3)) for k, v in sweep1.items()})

# F2
f2 = {"full_r": pearson(gfp[(gfp > 0) | (cx3 > 0)], cx3[(gfp > 0) | (cx3 > 0)])}
sweep2 = {}
for thr in (1, 2, 3, 5):
    mm = (gfp >= thr) | (cx3 >= thr)
    sweep2[thr] = {"n": int(mm.sum()), "r": pearson(gfp[mm], cx3[mm])}
f2["sweep"] = sweep2
findings["F2_GFP_Cx3cr1"] = f2
print("[F2] GFP vs Cx3cr1    full r =", round(f2["full_r"], 3),
      "sweep:", {k: (v["n"], round(v["r"], 3)) for k, v in sweep2.items()})

# F3
pos = lyve > 0
bam = (mrc1 > 0) | (cd163 > 0)
f3 = dict(pct_pos=float(pos.mean()*100), maxcount=int(lyve.max()),
          mean_in_pos=float(lyve[pos].mean()) if pos.any() else 0.0,
          lyve_pos_in_bam=float(pos[bam].mean()*100),
          lyve_pos_in_nonbam=float(pos[~bam].mean()*100),
          pct_lyve_lacking_bam=float((pos & ~bam).sum()/max(1, pos.sum())*100))
findings["F3_Lyve1"] = f3
print("[F3] Lyve1 pos% =", round(f3["pct_pos"], 2), "max =", f3["maxcount"],
      "| in BAM-like", round(f3["lyve_pos_in_bam"], 1), "% vs non-BAM",
      round(f3["lyve_pos_in_nonbam"], 1), "% |",
      round(f3["pct_lyve_lacking_bam"], 0), "% of Lyve1+ lack Mrc1&Cd163")

# =====================================================================
# AMBIENT decomposition: stratify F1/F2 by contamination level
# =====================================================================
print("\n=== Ambient decomposition (stratify by decontx_contamination) ===")
# contamination terciles among non-tumor
q1, q2 = np.quantile(cont_nt, [1/3, 2/3])
strata = {
    "low_contam": cont_nt <= q1,
    "mid_contam": (cont_nt > q1) & (cont_nt <= q2),
    "high_contam": cont_nt > q2,
}
ambient = {"contam_terciles": [float(q1), float(q2)], "strata": {}}
print(f"contamination terciles: <= {q1:.3f}, {q1:.3f}-{q2:.3f}, > {q2:.3f}")
for name, mask in strata.items():
    g = gfp[mask]; t = tdt[mask]; c = cx3[mask]
    m1 = (g > 0) | (t > 0); m2 = (g > 0) | (c > 0)
    rec = dict(
        n=int(mask.sum()),
        mean_contam=float(cont_nt[mask].mean()),
        F1_r=pearson(g[m1], t[m1]) if m1.sum() > 2 else float("nan"),
        F1_n=int(m1.sum()),
        F2_r=pearson(g[m2], c[m2]) if m2.sum() > 2 else float("nan"),
        F2_n=int(m2.sum()),
    )
    ambient["strata"][name] = rec
    print(f"  {name:11s} n={rec['n']:6d} contam={rec['mean_contam']:.3f} "
          f"F1 r={rec['F1_r']:+.3f}(n={rec['F1_n']}) F2 r={rec['F2_r']:+.3f}(n={rec['F2_n']})")

# GFP positivity vs total counts / cell size
gfp_pos = gfp > 0
ambient["gfp_pos_total_counts_median"] = float(np.median(total[gfp_pos]))
ambient["gfp_neg_total_counts_median"] = float(np.median(total[~gfp_pos]))
# cell size (Area) from with_neg, restricted to non-tumor and aligned order
area_full = awn.obs["Area"].to_numpy(np.float64) if "Area" in awn.obs else None
if area_full is not None:
    area_nt = area_full[nt]
    ambient["gfp_pos_area_median"] = float(np.median(area_nt[gfp_pos]))
    ambient["gfp_neg_area_median"] = float(np.median(area_nt[~gfp_pos]))
# GFP+ rate by total-count decile
deciles = np.quantile(total, np.linspace(0, 1, 11))
gfp_rate_by_decile = []
for i in range(10):
    lo, hi = deciles[i], deciles[i+1]
    md = (total >= lo) & (total <= hi if i == 9 else total < hi)
    if md.sum() > 0:
        gfp_rate_by_decile.append((float((lo+hi)/2), float(gfp_pos[md].mean()*100), int(md.sum())))
ambient["gfp_rate_by_total_decile"] = gfp_rate_by_decile
print(f"  GFP+ median total counts {ambient['gfp_pos_total_counts_median']:.0f} "
      f"vs GFP- {ambient['gfp_neg_total_counts_median']:.0f}")
if area_full is not None:
    print(f"  GFP+ median Area {ambient['gfp_pos_area_median']:.0f} "
          f"vs GFP- {ambient['gfp_neg_area_median']:.0f}")
# also contamination in tag+ vs tag-
ambient["contam_gfp_pos"] = float(cont_nt[gfp_pos].mean())
ambient["contam_gfp_neg"] = float(cont_nt[~gfp_pos].mean())
ambient["contam_lyve_pos"] = float(cont_nt[lyve > 0].mean())
ambient["contam_lyve_neg"] = float(cont_nt[lyve == 0].mean())
ambient["contam_tdt_pos"] = float(cont_nt[tdt > 0].mean())
ambient["contam_tdt_neg"] = float(cont_nt[tdt == 0].mean())
print(f"  mean contam: GFP+ {ambient['contam_gfp_pos']:.3f} / GFP- {ambient['contam_gfp_neg']:.3f} | "
      f"tdT+ {ambient['contam_tdt_pos']:.3f} / tdT- {ambient['contam_tdt_neg']:.3f} | "
      f"Lyve1+ {ambient['contam_lyve_pos']:.3f} / Lyve1- {ambient['contam_lyve_neg']:.3f}")

# =====================================================================
# CO-EXPRESSION matrix: custom probes vs canonical markers (Spearman, non-tumor)
# =====================================================================
print("\n=== Co-expression matrix (UNCONDITIONED Pearson, all non-tumor, decontx raw) ===")
# IMPORTANT: the primary co-expression metric is unconditioned Pearson over ALL
# non-tumor cells. The union-mask (x>0)|(y>0) Pearson used in the session F2 finding
# is a conditioning ARTIFACT: it forces a negative r on ANY sparse pair (verified:
# Mrc1~Cd163 -> -0.75 under the mask but +0.02 unconditioned). We report both but
# trust the unconditioned value.
canon_present = [g for g in CANONICAL if g in adx.var_names]
custom_present = [g for g in CUSTOM if g in adx.var_names]
custom_vecs = {g: col(sub, g) for g in custom_present}
canon_vecs = {g: col(sub, g) for g in canon_present}
coexp = pd.DataFrame(index=custom_present, columns=canon_present, dtype=float)
for cg in custom_present:
    for kg in canon_present:
        coexp.loc[cg, kg] = pearson(custom_vecs[cg], canon_vecs[kg])
coexp = coexp.astype(float)
coexp.to_csv(os.path.join(OUT, "coexpression_custom_vs_canonical_slice1.csv"))
# Spearman too for completeness (tie-dominated on sparse counts; secondary)
coexp_p = pd.DataFrame(index=custom_present, columns=canon_present, dtype=float)
for cg in custom_present:
    for kg in canon_present:
        coexp_p.loc[cg, kg] = spearman(custom_vecs[cg], canon_vecs[kg])
coexp_p = coexp_p.astype(float)
coexp_p.to_csv(os.path.join(OUT, "coexpression_spearman_slice1.csv"))
print("Unconditioned Pearson highlights (custom tag vs microglia Cx3cr1):")
print("  tdTomato~Cx3cr1 =", round(float(coexp.loc["tdTomato", "Cx3cr1"]), 3),
      "| GFP~Cx3cr1 =", round(float(coexp.loc["GFP", "Cx3cr1"]), 3),
      "| TMEM119~Cx3cr1 =", round(float(coexp.loc["TMEM119", "Cx3cr1"]), 3),
      "(ref Csf1r~Cx3cr1 should be ~+0.09)")

# tdTomato vs TMEM119 directly (both custom; TMEM119 microglia marker)
tdt_tmem_r = spearman(custom_vecs["tdTomato"], custom_vecs["TMEM119"])
print("  tdTomato~TMEM119(custom) =", round(float(tdt_tmem_r), 3))

# Co-expression restricted to EXPRESSING cells (Spearman on sparse counts is
# tie-dominated; the load-bearing signal lives in cells positive for either gene).
print("\n  -- Pearson r on cells positive for EITHER gene (>0) --")
coexp_pos = pd.DataFrame(index=custom_present, columns=canon_present, dtype=float)
for cg in custom_present:
    cv = custom_vecs[cg]
    for kg in canon_present:
        kv = canon_vecs[kg]
        mm = (cv > 0) | (kv > 0)
        coexp_pos.loc[cg, kg] = pearson(cv[mm], kv[mm]) if mm.sum() > 2 else float("nan")
coexp_pos = coexp_pos.astype(float)
coexp_pos.to_csv(os.path.join(OUT, "coexpression_on_positive_cells_slice1.csv"))
print("  GFP~Cx3cr1(pos)   =", round(float(coexp_pos.loc["GFP", "Cx3cr1"]), 3),
      "| tdTomato~Cx3cr1(pos) =", round(float(coexp_pos.loc["tdTomato", "Cx3cr1"]), 3),
      "| TMEM119~Cx3cr1(pos) =", round(float(coexp_pos.loc["TMEM119", "Cx3cr1"]), 3))

# =====================================================================
# POSITIVITY CROSS-TAB: the "tdTomato-in-microglia reversal" test.
# Define microglia/myeloid by TRANSCRIPTOME (Cx3cr1+ and/or Csf1r+), NOT by the
# custom tags under test. Measure tag-positivity enrichment inside vs outside.
# =====================================================================
print("\n=== Positivity enrichment (transcriptome-defined microglia/myeloid) ===")
csf1r = col(sub, "Csf1r")
# myeloid/microglia: positive for Cx3cr1 OR Csf1r (panel is thin on microglia axis)
micro = (cx3 > 0) | (csf1r > 0)
n_micro = int(micro.sum()); n_non = int((~micro).sum())
xtab = {"n_micro_like": n_micro, "n_non_micro": n_non}
for tag, vec in (("GFP", gfp), ("tdTomato", tdt), ("TMEM119", tmem), ("Lyve1", lyve)):
    pin = float((vec[micro] > 0).mean()*100)
    pout = float((vec[~micro] > 0).mean()*100)
    enr = pin/pout if pout > 0 else float("nan")
    xtab[tag] = dict(pos_in_micro_pct=round(pin, 2), pos_out_micro_pct=round(pout, 2),
                     enrichment=round(enr, 2))
    print(f"  {tag:9s} +rate: micro-like {pin:5.2f}% vs non {pout:5.2f}%  enrich={enr:.2f}x")
findings["positivity_xtab"] = xtab

# Direct head-to-head: GFP vs tdTomato positivity inside microglia-like cells
# (lineage claim: GFP should mark a microglia/MDM split; reversal = tdT dominates microglia)
xtab["gfp_only_in_micro"] = int(((gfp > 0) & (tdt == 0) & micro).sum())
xtab["tdt_only_in_micro"] = int(((tdt > 0) & (gfp == 0) & micro).sum())
print(f"  within micro-like: GFP-only n={xtab['gfp_only_in_micro']} "
      f"vs tdT-only n={xtab['tdt_only_in_micro']}")

# =====================================================================
# Plots
# =====================================================================
print("\n=== Plots ===")

# Plot 1: S/N bar
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#d62728" if p == "GFAP" else ("#1f77b4" if p in CUSTOM else "#7f7f7f")
          for p in sn_df["probe"]]
ax.bar(sn_df["probe"], sn_df["S_to_N"], color=colors, log=True)
ax.axhline(1.0, color="k", ls="--", lw=1, label="S/N = 1 (background)")
ax.set_ylabel("Signal/Noise (mean count / mean neg-probe)")
ax.set_title("Slice 1 — per-probe signal-to-background (red=GFAP pos-ctrl, blue=custom, grey=stock)")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sn_barplot_slice1.png"), dpi=130)
plt.close(fig)

# Plot 2: threshold sweep F1 & F2
fig, ax = plt.subplots(figsize=(7, 5))
thrs = [1, 2, 3, 5]
ax.plot(thrs, [sweep1[t]["r"] for t in thrs], "o-", label="GFP~tdTomato (F1)")
ax.plot(thrs, [sweep2[t]["r"] for t in thrs], "s-", label="GFP~Cx3cr1 (F2)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("min-count threshold (>=)"); ax.set_ylabel("Pearson r")
ax.set_title("Slice 1 — correlation vs count threshold (ambient dominates low counts)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "threshold_sweep_slice1.png"), dpi=130)
plt.close(fig)

# Plot 3: GFP+ rate by total-count decile
fig, ax = plt.subplots(figsize=(7, 5))
xs = [d[0] for d in gfp_rate_by_decile]; ys = [d[1] for d in gfp_rate_by_decile]
ax.plot(xs, ys, "o-")
ax.set_xlabel("total counts per cell (decile midpoint)")
ax.set_ylabel("% GFP+ cells")
ax.set_title("Slice 1 — GFP positivity rises with total counts (size/ambient dependence)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gfp_rate_by_totalcount_slice1.png"), dpi=130)
plt.close(fig)

# Plot 4: co-expression heatmap
fig, ax = plt.subplots(figsize=(max(8, 0.5*len(canon_present)+3), 5))
im = ax.imshow(coexp.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
ax.set_xticks(range(len(canon_present))); ax.set_xticklabels(canon_present, rotation=90)
ax.set_yticks(range(len(custom_present))); ax.set_yticklabels(custom_present)
for i in range(len(custom_present)):
    for j in range(len(canon_present)):
        v = coexp.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if abs(v) > 0.3 else "black", fontsize=6)
fig.colorbar(im, ax=ax, label="Spearman r")
ax.set_title("Slice 1 — custom probes (rows) vs canonical markers (cols)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "coexpression_heatmap_slice1.png"), dpi=130)
plt.close(fig)

# Plot 5: contamination-stratified F1/F2
fig, ax = plt.subplots(figsize=(7, 5))
names = list(strata.keys())
ax.plot(range(3), [ambient["strata"][n]["F1_r"] for n in names], "o-", label="F1 GFP~tdT")
ax.plot(range(3), [ambient["strata"][n]["F2_r"] for n in names], "s-", label="F2 GFP~Cx3cr1")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=20)
ax.set_ylabel("Pearson r"); ax.set_title("Slice 1 — correlation by contamination tercile")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "correlation_by_contamination_slice1.png"), dpi=130)
plt.close(fig)

# =====================================================================
# Dump metrics json for the report step
# =====================================================================
metrics = dict(
    n_cells=int(adx.n_obs), n_nontumor=int(nt.sum()),
    neg_probes=neg_names, neg_mean_global_nt=neg_mean_global_nt,
    sn_table=sn_df.to_dict(orient="records"),
    findings=findings, ambient=ambient,
    coexp_spearman=coexp.round(3).to_dict(),
    coexp_pearson=coexp_p.round(3).to_dict(),
    coexp_on_positive=coexp_pos.round(3).to_dict(),
    tdt_tmem_spearman=float(tdt_tmem_r),
)
with open(os.path.join(OUT, "metrics_slice1.json"), "w") as f:
    json.dump(metrics, f, indent=2, default=float)
print("\nWrote metrics_slice1.json + CSVs + 5 PNGs to", OUT)
print("DONE")
