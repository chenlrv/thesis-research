"""MDM vs microglia separation using TMEM119 as primary marker.

Among GFP+ cells:
  Microglia = GFP+ TMEM119+
  MDMs      = GFP+ TMEM119−

tdTomato is used as validation: MDMs should be enriched for tdTomato
compared to microglia, confirming the TMEM119-based separation.
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse

SLICES = {i: f"D:/thesis-research/resources/cache/slice_{i}_adata.h5ad"
          for i in range(1, 7)}
OUTPUT_1 = "D:/thesis-research/gfp_tmem_tdtomato_slices1-3.png"
OUTPUT_2 = "D:/thesis-research/gfp_tmem_tdtomato_slices4-6.png"
X_MAX    = 16  # fixed across all slices and both figures


def get_expr(adata, gene):
    panel = {g.lower(): g for g in adata.var_names}
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    xv = adata[:, key].X
    return (xv.toarray().ravel() if issparse(xv) else np.asarray(xv).ravel()).astype(int)


def load_slice(path, slice_id):
    print(f"Loading Slice {slice_id} ...")
    adata = ad.read_h5ad(path)
    return (
        get_expr(adata, "GFP"),
        get_expr(adata, "TMEM119"),
        get_expr(adata, "tdTomato"),
    )


def draw_panel(ax, gfp, tmem, tdtom, slice_id, x_max):
    microglia = (gfp >= 1) & (tmem >= 1)
    mdm       = (gfp >= 1) & (tmem == 0)

    bins = np.arange(0.5, x_max + 1.5, 1)

    ax.hist(tdtom[microglia], bins=bins, density=False,
            color="#4393c3", alpha=0.7, histtype="stepfilled", edgecolor="none",
            label=f"Microglia — GFP+ TMEM119+  (n={microglia.sum():,})")
    ax.hist(tdtom[mdm], bins=bins, density=False,
            color="#d6604d", alpha=0.7, histtype="stepfilled", edgecolor="none",
            label=f"MDMs — GFP+ TMEM119−  (n={mdm.sum():,})")

    pct_tdtom_microglia = 100 * (tdtom[microglia] > 0).mean()
    pct_tdtom_mdm       = 100 * (tdtom[mdm] > 0).mean()

    ax.set_title(
        f"Slice {slice_id}\n"
        f"tdTomato>0: microglia {pct_tdtom_microglia:.1f}%  |  MDMs {pct_tdtom_mdm:.1f}%",
        fontsize=9,
    )
    ax.set_xlabel("tdTomato (raw counts)")
    ax.set_ylabel("Number of cells")
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, framealpha=0.9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    print(f"  Slice {slice_id}: microglia={microglia.sum():,}  MDMs={mdm.sum():,}  "
          f"tdTomato>0 in microglia={pct_tdtom_microglia:.1f}%  in MDMs={pct_tdtom_mdm:.1f}%")


def plot_group(slice_ids, output):
    data = {sid: load_slice(SLICES[sid], sid) for sid in slice_ids}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle(
        "tdTomato validation of TMEM119-based separation\n"
        "Microglia = GFP+ TMEM119+  |  MDMs = GFP+ TMEM119−",
        fontsize=11,
    )
    for ax, sid in zip(axes, slice_ids):
        gfp, tmem, tdtom = data[sid]
        draw_panel(ax, gfp, tmem, tdtom, sid, X_MAX)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"Saved: {output}")


plot_group([1, 2, 3], OUTPUT_1)
plot_group([4, 5, 6], OUTPUT_2)
