"""Is Lyve1 positivity spatially organised the way BAM are (perivascular / border)?

Two annotation-free measures per cell, from global coordinates only:
  d_vessel : distance to the nearest vessel-like cell (>=2 counts summed over
             Pecam1, Flt1, Cdh5, Vwf, Ramp2, Esam)
  d_edge   : distance to the nearest unoccupied region of tissue (a proxy for the
             meningeal surface), from a distance transform of a 60um occupancy grid

Compared across: Lyve1+ cells, cells positive for a canonical BAM marker
(Mrc1 or Cd163), and all cells.
"""
import anndata as ad, numpy as np
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from scipy import ndimage

PX_UM = 0.12028  # CosMx pixel size
VASC = ["Pecam1", "Flt1", "Cdh5", "Vwf", "Ramp2", "Esam"]

for s in [1, 3]:
    a = ad.read_h5ad(f"D:/thesis-research/resources/cache/slice_{s}_adata_with_neg.h5ad")
    v = a.var_names.astype(str)

    def cnt(n):
        k = [x for x in v if x.lower() == n.lower()]
        if not k:
            return np.zeros(a.n_obs)
        y = a[:, k[0]].X
        return (y.toarray().ravel() if issparse(y) else np.asarray(y).ravel()).astype(float)

    x = a.obs["CenterX_global_px"].to_numpy() * PX_UM
    y = a.obs["CenterY_global_px"].to_numpy() * PX_UM
    vasc = sum(cnt(g) for g in VASC) >= 2
    lyve = cnt("Lyve1") > 0
    bam = (cnt("Mrc1") + cnt("Cd163")) > 0

    d_vessel = cKDTree(np.c_[x[vasc], y[vasc]]).query(np.c_[x, y])[0]

    BIN = 60.0
    ix = ((x - x.min()) // BIN).astype(int)
    iy = ((y - y.min()) // BIN).astype(int)
    grid = np.zeros((ix.max() + 3, iy.max() + 3), bool)
    grid[ix + 1, iy + 1] = True
    grid = ndimage.binary_closing(grid, np.ones((3, 3)))
    d_edge = ndimage.distance_transform_edt(grid)[ix + 1, iy + 1] * BIN

    print(f"\n=== slice {s} | {a.n_obs} cells | vessel-like {vasc.sum()} "
          f"({100*vasc.mean():.1f}%) | Lyve1+ {lyve.sum()} | Mrc1/Cd163+ {bam.sum()}")
    print(f"{'group':<24}{'n':>8}{'median d_vessel (um)':>24}{'median d_edge (um)':>22}")
    for name, m in (("all cells", np.ones(a.n_obs, bool)),
                    ("Lyve1+", lyve),
                    ("Mrc1/Cd163+ (BAM)", bam),
                    ("Lyve1+ and Mrc1/Cd163+", lyve & bam)):
        print(f"{name:<24}{int(m.sum()):>8}{np.median(d_vessel[m]):>24.1f}"
              f"{np.median(d_edge[m]):>22.1f}")
