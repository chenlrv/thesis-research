"""High-confidence BAM set: THREE independent signals must agree
  (1) marker gate  >=2 of [Mrc1,Cd163,Pf4,Lyve1]   (transcriptomic identity)
  (2) independent classifier on NON-marker genes, proba >= 0.80  (rest-of-transcriptome)
  (3) perivascular location: within PERIVASC_UM of an endothelial/vessel cell (anatomy)
Same agreement logic as build_highconf.py, plus the orthogonal spatial requirement."""
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree

PERIVASC_UM = 20.0          # vessel-adjacency cutoff (perivascular niche width)
T = 0.80                    # classifier agreement threshold

ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
myeloid = np.load("_myeloid_mask_slice1.npy")
xy = ad.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)

def det(genes, k):
    cols = [ad.var_names.get_loc(g) for g in genes if g in ad.var_names]
    x = ad[:, cols].X; x = np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)
    return (x >= 1).sum(1) >= k

# ---- px -> um scale from the data (Area px vs Area.um2 if present, else CosMx 0.12) ----
if "Area.um2" in ad.obs and "Area" in ad.obs:
    um_per_px = float(np.sqrt(np.median(ad.obs["Area.um2"] / ad.obs["Area"])))
else:
    um_per_px = 0.12028
thr_px = PERIVASC_UM / um_per_px
print(f"scale: {um_per_px:.4f} um/px  ->  perivascular cutoff {PERIVASC_UM} um = {thr_px:.0f} px")

# ---- signal 1: marker gate ----
BAM_MARK = ["Mrc1", "Cd163", "Pf4", "Lyve1"]
prov_bam = det(BAM_MARK, 2) & myeloid
endo = det(["Pecam1", "Flt1", "Vwf", "Cdh5", "Kdr"], 3)

# ---- signal 3: perivascular (distance to nearest vessel), for myeloid cells ----
# reference = PURE endothelial (not also myeloid) so a BAM-endo doublet isn't its own vessel
pure_endo = endo & ~myeloid
vtree = cKDTree(xy[pure_endo])
d_mye = vtree.query(xy[myeloid])[0]
perivasc = d_mye <= thr_px                        # aligned to myeloid-subset arrays
endo_overlap = endo[myeloid]                      # myeloid cells that are ALSO endo+ (doublet flag)
print(f"vessel ref: {pure_endo.sum():,} pure-endo (excluded {(endo&myeloid).sum():,} endo+myeloid doublets)")
print(f"provisional BAM that are also endo+ (doublet-suspect): "
      f"{(prov_bam & endo).sum():,} / {prov_bam.sum():,}")

# ---- signal 2: independent classifier on non-marker genes ----
MARKER_UNION = set(BAM_MARK) | {
 "Csf1r","C1qa","C1qb","C1qc","Aif1","Tyrobp","Itgam","Ptprc","Cd68","Trem2",
 "TMEM119","Cx3cr1","Ccr2","Plac8","Apoe","Cst7","Itgax","Clec7a","Spp1","Gpnmb","Cd9",
 "Pecam1","Flt1","Vwf","Cdh5","Kdr","GFP","tdTomato"}
indep = [g for g in ad.var_names if g not in MARKER_UNION]
mye = ad[myeloid].copy()
sc.pp.normalize_total(mye, target_sum=1e4); sc.pp.log1p(mye)
X = mye[:, indep].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
y = prov_bam[myeloid].astype(int)
clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
auc = roc_auc_score(y, proba)
clf_ok = proba >= T

print(f"\nmyeloid={myeloid.sum():,} | provisional BAM={y.sum():,} | indep genes={len(indep)}")
print(f"BAM vs other-myeloid held-out AUC (non-marker genes): {auc:.3f}")

# ---- cumulative gating ----
s1 = (y == 1)
print("\ncumulative high-confidence gating:")
print(f"  signal1 marker gate           : {s1.sum():5d}")
print(f"  + signal3 perivascular        : {(s1 & perivasc).sum():5d}")
print(f"  + signal2 classifier>={T:.2f}     : {(s1 & clf_ok).sum():5d}")
print(f"  ALL THREE (high-confidence)   : {(s1 & perivasc & clf_ok).sum():5d}")

print("\nthreshold sweep (marker AND perivascular AND proba>=T):")
for t in [0.50, 0.70, 0.80, 0.90]:
    hc = s1 & perivasc & (proba >= t)
    print(f"  T={t:.2f}: {hc.sum():5d}")

hc_bam = s1 & perivasc & clf_ok
# anatomy of the final set vs provisional vs other
d_other = vtree.query(xy[myeloid][y == 0])[0]
print(f"\nmedian dist-to-vessel (um): high-conf BAM "
      f"{np.median(d_mye[hc_bam])*um_per_px:.1f} | provisional BAM "
      f"{np.median(d_mye[s1])*um_per_px:.1f} | other-myeloid {np.median(d_other)*um_per_px:.1f}")

idx_mye = np.where(myeloid)[0]
full = np.zeros(ad.n_obs, bool); full[idx_mye[hc_bam]] = True
np.save("_bam_highconf_mask_slice1.npy", full)
pd.DataFrame({"x": xy[idx_mye, 0], "y": xy[idx_mye, 1],
              "provisional_BAM": s1, "clf_proba": proba.round(4),
              "perivascular": perivasc, "high_conf_BAM": hc_bam}
             ).to_csv("score_genes_slice1/bam_high_confidence.csv", index=False)
print(f"\nsaved bam_high_confidence.csv + mask ({hc_bam.sum():,} high-conf BAM, "
      f"3 signals, T={T}, perivasc<= {PERIVASC_UM}um)")
