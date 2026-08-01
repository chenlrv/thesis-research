import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

OUT = "myeloid_plots"; os.makedirs(OUT, exist_ok=True)
ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
myeloid = np.load("_myeloid_mask_slice1.npy")
xy = ad.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)

def det(genes, k):
    cols = [ad.var_names.get_loc(g) for g in genes if g in ad.var_names]
    x = ad[:, cols].X; x = np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)
    return (x >= 1).sum(1) >= k

# endothelial (vessel) cells for the anatomical reference
endo = det(["Pecam1", "Flt1", "Vwf", "Cdh5", "Kdr"], 3)
# BAM gate: >=2 of the perivascular markers, among myeloid only
bam_markers = ["Mrc1", "Cd163", "Pf4", "Lyve1"]
bam_pos = det(bam_markers, 2) & myeloid
other_my = myeloid & ~bam_pos
print(f"myeloid={myeloid.sum():,}  endothelial={endo.sum():,}")
print(f"BAM+ (>=2 of {bam_markers}) within myeloid: {bam_pos.sum():,} "
      f"({100*bam_pos.sum()/myeloid.sum():.1f}% of myeloid)")

# distance from each myeloid cell to nearest vessel (anatomical test)
vtree = cKDTree(xy[endo])
d_bam = vtree.query(xy[bam_pos])[0]
d_oth = vtree.query(xy[other_my])[0]
# local density: # neighbors within 50px (border cells are sparser)
mtree = cKDTree(xy)
dens_bam = mtree.query_ball_point(xy[bam_pos], 50, return_length=True)
dens_oth = mtree.query_ball_point(xy[other_my], 50, return_length=True)
print("\n== anatomical test: BAM+ should sit near vessels / at borders ==")
print(f"  dist-to-nearest-vessel (px):   BAM+ median {np.median(d_bam):6.1f}   "
      f"other-myeloid median {np.median(d_oth):6.1f}")
print(f"  local density (#cells <50px):  BAM+ median {np.median(dens_bam):6.1f}   "
      f"other-myeloid median {np.median(dens_oth):6.1f}")

# marker sanity: BAM+ vs other myeloid
sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
def mean_expr(g, mask):
    j = ad.var_names.get_loc(g); x = ad[:, j].X
    x = np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return x[mask].mean()
print("\n== marker means (log-norm): BAM+ vs other-myeloid ==")
for g in bam_markers + ["TMEM119", "Cx3cr1", "Ccr2", "Plac8"]:
    if g in ad.var_names:
        print(f"  {g:9} BAM+ {mean_expr(g,bam_pos):.3f}   other {mean_expr(g,other_my):.3f}")

np.save("_bam_mask_slice1.npy", bam_pos)

# spatial figure
fig, ax = plt.subplots(figsize=(9, 8.5), dpi=160)
ax.scatter(xy[~myeloid & ~endo, 0], xy[~myeloid & ~endo, 1], s=0.5, c="#e8e8e8",
           linewidths=0, rasterized=True, label="other")
ax.scatter(xy[endo, 0], xy[endo, 1], s=2, c="#9ecae1", linewidths=0, rasterized=True, label="vessels (endo)")
ax.scatter(xy[other_my, 0], xy[other_my, 1], s=2.5, c="#969696", linewidths=0, rasterized=True,
           label=f"microglia/MDM (n={other_my.sum():,})")
ax.scatter(xy[bam_pos, 0], xy[bam_pos, 1], s=10, c="#d62728", linewidths=0, rasterized=True,
           label=f"BAM+ (n={bam_pos.sum():,})")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower right", markerscale=3, fontsize=10)
ax.set_title("Step 1 — BAM+ vs vessels (slice_1)")
fig.tight_layout(); fig.savefig(f"{OUT}/bam_spatial_slice1.png"); plt.close()
print(f"\nsaved {OUT}/bam_spatial_slice1.png")
