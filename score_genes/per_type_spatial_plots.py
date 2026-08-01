"""Per-cell-type spatial plots (one subplot per annotated type) with tumor cells
overlaid, for slices 1/2/3, from the CURRENT gate annotation.

Each slice -> a 2x2 figure: Microglia | MDM | BAM | unknown. In every panel:
  light grey = all non-tumor cells (tissue outline)
  black      = tumor cells (pred_tumor)
  colour     = the cells assigned to THIS type
Output -> score_genes_slice{N}_merged/classify/myeloid_per_type_spatial.png
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from myeloid_subtype_gate import (load_myeloid, MODULES, sg, assign, COL,
                                  SLICES, CONTROL_SLICE)

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})
TYPES = ["Microglia", "MDM", "BAM", "unknown"]


def main():
    myes, bgs = [], {}
    for n in SLICES:
        mye, bg = load_myeloid(n)
        myes.append(mye); bgs[n] = bg
    comb = ad.concat(myes, join="inner", index_unique="-")
    scores = {m: sg(comb, MODULES[m], f"sc_{m}") for m in MODULES}
    sl = comb.obs["slice"].to_numpy()
    labels, thr, hits = assign(scores, sl == CONTROL_SLICE)
    cx = comb.obs["cx"].to_numpy(); cy = comb.obs["cy"].to_numpy()

    for n in SLICES:
        m_slice = sl == n
        lab = labels[m_slice]
        xx, yy = cx[m_slice], cy[m_slice]
        bg = bgs[n]
        tag = " (CONTROL)" if n == CONTROL_SLICE else ""

        fig, axes = plt.subplots(2, 2, figsize=(20, 16), dpi=150)
        for ax, t in zip(axes.ravel(), TYPES):
            ax.scatter(bg["cxnt"], bg["cynt"], s=0.5, c="#eeeeee",
                       linewidths=0, rasterized=True)
            if len(bg["tx"]):
                ax.scatter(bg["tx"], bg["ty"], s=1.5, c="black", linewidths=0,
                           rasterized=True, label=f"tumor ({len(bg['tx']):,})")
            m = lab == t
            ax.scatter(xx[m], yy[m], s=5, c=COL[t], linewidths=0, rasterized=True,
                       label=f"{t} ({int(m.sum()):,})")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=14, fontweight="bold")
            ax.legend(loc="lower right", markerscale=3, fontsize=9)
        fig.suptitle(f"Slice {n}{tag} — myeloid subtypes vs tumor "
                     f"(myeloid n={int(m_slice.sum()):,})",
                     fontsize=17, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = f"D:/thesis-research/score_genes_slice{n}_merged/classify"
        os.makedirs(out, exist_ok=True)
        fig.savefig(f"{out}/myeloid_per_type_spatial.png", bbox_inches="tight")
        plt.close(fig)
        print(f"slice {n}: saved per-type spatial -> {out}/myeloid_per_type_spatial.png "
              + str({t: int((lab == t).sum()) for t in TYPES}))


if __name__ == "__main__":
    main()
