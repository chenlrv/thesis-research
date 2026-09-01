"""
Compare Method A (gating), B-asis (GAT reporter), B-fixed (GAT transcriptome anchors),
C (InSituType-ML reimpl). Confusion matrices + ARI + Cohen's kappa, restricted to
myeloid+astro. High-disagreement cells + diagnostics. Agreement table.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score, confusion_matrix

OUT = "D:/thesis-research/agents/outputs/annotation"
CLASSES = ["Microglia", "MDM", "BAM", "Astrocyte", "Other"]

def harmonize(s):
    m = {
        "Microglia": "Microglia", "Microglia_Homeostatic": "Microglia", "Microglia_DAM": "Microglia",
        "MDM": "MDM", "MDM_inflammatory": "MDM",
        "BAM": "BAM", "Astrocyte": "Astrocyte",
        "Other": "Other", "unassigned": "Other", "Myeloid_unresolved": "Other", "undefined": "Other",
    }
    return pd.Series(s).map(lambda x: m.get(str(x), "Other")).values

# load
A = pd.read_csv(OUT + "/method_a_labels.csv", index_col=0)
Bas = pd.read_csv(OUT + "/method_b_asis_labels.csv", index_col=0)
Bfx = pd.read_csv(OUT + "/method_b_fixed_labels.csv", index_col=0)
C = pd.read_csv(OUT + "/method_c_labels.csv", index_col=0)
n = len(A)
assert len(Bas) == n == len(Bfx) == len(C), "row mismatch"

lab = pd.DataFrame(index=A.index)
lab["A"] = harmonize(A["method_a_core"].values)
lab["A_fine"] = A["method_a"].values
lab["Basis"] = harmonize(Bas["method_b_asis"].values)
lab["Bfixed"] = harmonize(Bfx["method_b_fixed"].values)
lab["C"] = harmonize(C["method_c"].values)
# C with confidence threshold: low-posterior calls -> Other (does confidence rescue force-binning?)
c_thr = harmonize(C["method_c"].values).copy()
c_thr[C["method_c_prob"].values < 0.9] = "Other"
lab["Cthr"] = c_thr
# carry diagnostics
for c in ["nCount", "neg_mean", "decontx_contamination", "cnt_Cx3cr1", "cnt_TMEM119",
          "cnt_P2rx5", "cnt_Ccr2", "cnt_Plac8", "cnt_Mrc1", "cnt_Cd163", "cnt_GFP",
          "cnt_tdTomato", "cnt_GFAP", "cnt_Nkg7", "fov", "CenterX_global_px", "CenterY_global_px",
          "is_myeloid", "is_astro"]:
    if c in A.columns:
        lab[c] = A[c].values
lab["Basis_prob"] = Bas["method_b_asis_prob"].values
lab["Bfixed_prob"] = Bfx["method_b_fixed_prob"].values
lab["C_prob"] = C["method_c_prob"].values
lab.to_csv(OUT + "/all_method_labels.csv")

# ---- per-method label fractions ----
frac = pd.DataFrame({m: pd.Series(lab[m]).value_counts() for m in ["A", "Basis", "Bfixed", "C", "Cthr"]})
frac = frac.reindex(CLASSES).fillna(0).astype(int)
frac_pct = (frac / n * 100).round(2)
frac.to_csv(OUT + "/label_fractions_counts.csv")
frac_pct.to_csv(OUT + "/label_fractions_pct.csv")
print("=== Label fractions (%) ===")
print(frac_pct)

# ---- pairwise agreement, restricted to myeloid+astro ----
FOCAL = ["Microglia", "MDM", "BAM", "Astrocyte"]
methods = ["A", "Basis", "Bfixed", "C"]
rows = []
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        m1, m2 = methods[i], methods[j]
        # restrict to cells where EITHER method calls a focal (myeloid/astro) class
        mask = lab[m1].isin(FOCAL) | lab[m2].isin(FOCAL)
        y1, y2 = lab[m1][mask].values, lab[m2][mask].values
        ari = adjusted_rand_score(y1, y2)
        kappa = cohen_kappa_score(y1, y2, labels=CLASSES)
        # exact-match accuracy on focal-union cells
        acc = float((y1 == y2).mean())
        # also focal-only (both focal)
        bothmask = lab[m1].isin(FOCAL) & lab[m2].isin(FOCAL)
        yb1, yb2 = lab[m1][bothmask].values, lab[m2][bothmask].values
        ari_b = adjusted_rand_score(yb1, yb2) if bothmask.sum() else np.nan
        kappa_b = cohen_kappa_score(yb1, yb2, labels=FOCAL) if bothmask.sum() else np.nan
        acc_b = float((yb1 == yb2).mean()) if bothmask.sum() else np.nan
        rows.append({"m1": m1, "m2": m2,
                     "n_focal_union": int(mask.sum()), "ARI_union": round(ari, 3),
                     "kappa_union": round(kappa, 3), "acc_union": round(acc, 3),
                     "n_both_focal": int(bothmask.sum()), "ARI_bothfocal": round(ari_b, 3),
                     "kappa_bothfocal": round(kappa_b, 3), "acc_bothfocal": round(acc_b, 3)})
agree = pd.DataFrame(rows)
agree.to_csv(OUT + "/agreement_table.csv", index=False)
print("\n=== Pairwise agreement (myeloid+astro) ===")
print(agree.to_string(index=False))

# ---- confusion matrices (focal-union) + plots ----
def plot_cm(m1, m2, fname):
    mask = lab[m1].isin(FOCAL) | lab[m2].isin(FOCAL)
    cm = confusion_matrix(lab[m1][mask], lab[m2][mask], labels=CLASSES)
    cmdf = pd.DataFrame(cm, index=[f"{m1}:{c}" for c in CLASSES], columns=[f"{m2}:{c}" for c in CLASSES])
    cmdf.to_csv(OUT + f"/cm_{m1}_vs_{m2}.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel(m2); ax.set_ylabel(m1); ax.set_title(f"{m1} vs {m2} (focal-union)")
    for ii in range(len(CLASSES)):
        for jj in range(len(CLASSES)):
            ax.text(jj, ii, cm[ii, jj], ha="center", va="center",
                    color="white" if cm[ii, jj] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(OUT + f"/{fname}", dpi=110); plt.close(fig)

for (m1, m2) in [("A", "Basis"), ("A", "Bfixed"), ("A", "C"), ("Basis", "Bfixed"), ("Bfixed", "C")]:
    plot_cm(m1, m2, f"cm_{m1}_vs_{m2}.png")
print("\nConfusion matrices written.")

# ---- consensus + high-disagreement cells ----
def n_focal_agree(row):
    vals = [row["A"], row["Bfixed"], row["C"]]  # trust the 3 transcriptome-anchored methods
    return vals
maj = []
for _, r in lab.iterrows():
    pass
# vectorized majority over A, Bfixed, C (the 3 transcriptome-anchored views)
import collections
votes = lab[["A", "Bfixed", "C"]].values
def majority(v):
    c = collections.Counter(v); top, cnt = c.most_common(1)[0]
    return top, cnt
mvote = [majority(row) for row in votes]
lab["consensus3"] = [m[0] for m in mvote]
lab["consensus3_n"] = [m[1] for m in mvote]   # how many of A/Bfixed/C agree (1..3)

# disagreement score: number of distinct labels across all 4 methods
lab["n_distinct"] = lab[["A", "Basis", "Bfixed", "C"]].nunique(axis=1)
# focus: cells that ANY method calls focal but methods disagree
focal_any = lab[["A", "Basis", "Bfixed", "C"]].isin(FOCAL).any(axis=1)
disagree = lab[focal_any & (lab["n_distinct"] >= 3)].copy()
disagree = disagree.sort_values(["n_distinct", "nCount"], ascending=[False, True])
disagree.to_csv(OUT + "/high_disagreement_cells.csv")
print(f"\nHigh-disagreement focal cells (>=3 distinct labels across 4 methods): {len(disagree)}")

# diagnostics on disagreement cells vs agreement cells
agree_cells = lab[focal_any & (lab["n_distinct"] <= 1)]
print("\n=== Diagnostics: high-disagree vs full-agree focal cells ===")
for col in ["nCount", "neg_mean", "decontx_contamination", "cnt_Cx3cr1", "cnt_TMEM119",
            "cnt_Ccr2", "cnt_Plac8", "cnt_Mrc1", "cnt_Cd163"]:
    if col in lab.columns:
        d = disagree[col].median() if len(disagree) else np.nan
        g = agree_cells[col].median() if len(agree_cells) else np.nan
        print(f"  {col:24s} disagree_median={d:8.3f}  agree_median={g:8.3f}")

# how often is a cell called MDM by reporter-GAT but NOT by transcriptome methods?
mdm_basis_only = lab[(lab["Basis"] == "MDM") & (lab["A"] != "MDM") & (lab["Bfixed"] != "MDM")]
print(f"\nCells called MDM by reporter-GAT (Basis) but NOT by A or Bfixed: {len(mdm_basis_only)}")
print("  of those, fraction with GFP count==0:",
      round(float((mdm_basis_only["cnt_GFP"] == 0).mean()), 3) if len(mdm_basis_only) else np.nan)
print("  median Ccr2 in those:", float(mdm_basis_only["cnt_Ccr2"].median()) if len(mdm_basis_only) else np.nan)

lab.to_csv(OUT + "/all_method_labels.csv")
print("\nWROTE agreement_table.csv, label_fractions_*.csv, cm_*.csv/png, high_disagreement_cells.csv, all_method_labels.csv")
