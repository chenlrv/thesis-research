import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
xy = ad.obs[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float)
myeloid = np.load("_myeloid_mask_slice1.npy")
hc = np.load("_bam_highconf_mask_slice1.npy")            # 216 high-conf BAM

def det(genes, k):
    cols = [ad.var_names.get_loc(g) for g in genes if g in ad.var_names]
    x = ad[:, cols].X; x = np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)
    return (x >= 1).sum(1) >= k
endo = det(["Pecam1", "Flt1", "Vwf", "Cdh5", "Kdr"], 3) & ~myeloid    # pure vessels
prov = det(["Mrc1", "Cd163", "Pf4", "Lyve1"], 2) & myeloid & ~hc      # provisional, not hc

fig, ax = plt.subplots(figsize=(10, 9.5), dpi=170)
ax.scatter(xy[:, 0], xy[:, 1], s=0.4, c="#ececec", linewidths=0, rasterized=True, label="all cells")
ax.scatter(xy[endo, 0], xy[endo, 1], s=1.5, c="#9ecae1", linewidths=0, rasterized=True, label=f"vessels (n={endo.sum():,})")
ax.scatter(xy[prov, 0], xy[prov, 1], s=5, c="#fdae6b", linewidths=0, rasterized=True,
           label=f"provisional BAM, not hc (n={prov.sum():,})")
ax.scatter(xy[hc, 0], xy[hc, 1], s=22, c="#d62728", edgecolors="k", linewidths=0.3,
           rasterized=True, label=f"high-conf BAM (n={hc.sum():,})")
ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
ax.legend(loc="lower right", markerscale=2.2, fontsize=10, framealpha=0.95)
ax.set_title("High-confidence BAMs (3 signals: marker + non-marker classifier + perivascular)\nslice_1", fontsize=13)
fig.tight_layout(); fig.savefig("myeloid_plots/bam_highconf_spatial_slice1.png"); plt.close()
print("saved myeloid_plots/bam_highconf_spatial_slice1.png")
