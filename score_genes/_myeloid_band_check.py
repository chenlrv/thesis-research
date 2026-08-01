"""Is the slice-4 myeloid 'band' real microglia or ambient-C1q mis-calls in dense
(cerebellar) tissue?

For every Myeloid-called cell (v2), split its markers into:
  complement = C1qa/C1qb/C1qc   (highly ambient)
  lineage    = Csf1r/Aif1/Tyrobp/Fcer1g   (myeloid-specific)
A real microglia expresses the lineage genes; an ambient mis-call is C1q-only.
Also relate the call to LOCAL CELL DENSITY (kNN) -> does the myeloid call-rate and
C1q-only fraction rise in the densest tissue (the artifact signature)?

Outputs printed stats + score_genes_slice4_v2/myeloid_band_check.png
Usage: python _myeloid_band_check.py <slice_id>   (default 4)
"""
import gc
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree

SL = sys.argv[1] if len(sys.argv) > 1 else "4"
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
OUT = f"D:/thesis-research/score_genes_slice{SL}_v2"
TUMOR_COL = "pred_tumor_XGBoost"
COMPLEMENT = ["C1qa", "C1qb", "C1qc"]
LINEAGE = ["Csf1r", "Aif1", "Tyrobp", "Fcer1g"]


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_X(h5):
    node = h5["X"]
    if isinstance(node, h5py.Group):
        enc = str(node.attrs.get("encoding-type", ""))
        shape = tuple(node.attrs["shape"])
        data, idx, indptr = node["data"][...], node["indices"][...], node["indptr"][...]
        M = csc_matrix((data, idx, indptr), shape=shape) if "csc" in enc \
            else csr_matrix((data, idx, indptr), shape=shape)
        return M.tocsr()
    return csr_matrix(node[...])


def _read_var(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_num(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    if arr.dtype.kind in ("S", "O"):
        return np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
    return arr.astype(bool)


def main():
    with h5py.File(TMPL.format(SL), "r") as h5:
        X = _read_X(h5)
        var = list(_read_var(h5))
        cx = _read_num(h5, "CenterX_global_px")
        cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)
    keep = ~tumor
    Xk = X[keep].tocsr()
    del X
    gc.collect()
    lib = np.asarray(Xk.sum(axis=1)).ravel()
    col = {g: var.index(g) for g in COMPLEMENT + LINEAGE if g in var}
    comp = np.asarray(Xk[:, [col[g] for g in COMPLEMENT]].todense())
    lin = np.asarray(Xk[:, [col[g] for g in LINEAGE]].todense())
    del Xk
    gc.collect()
    cxn, cyn = cx[keep], cy[keep]

    df = pd.read_csv(f"{OUT}/cell_scores.csv")
    assert len(df) == keep.sum(), f"row mismatch {len(df)} vs {keep.sum()}"
    mye = df["celltype_v2"].to_numpy() == "Myeloid"

    comp_pos = (comp > 0).any(1)          # any complement detected
    lin_pos = (lin > 0).any(1)            # any lineage-specific detected
    c1q_only = mye & comp_pos & ~lin_pos  # ambient signature
    core = mye & lin_pos                  # real myeloid backbone

    # local density: distance to 10th nearest neighbour (small = dense tissue)
    pts = np.c_[cxn, cyn]
    tree = cKDTree(pts)
    d10 = tree.query(pts, k=11)[0][:, -1]
    dens = 1.0 / d10                       # higher = denser
    dense_cut = np.quantile(dens, 0.90)    # top-decile densest tissue
    in_dense = dens >= dense_cut

    n_my = int(mye.sum())
    print(f"=== slice {SL} myeloid composition (n={n_my:,}) ===")
    for g in LINEAGE:
        j = LINEAGE.index(g)
        print(f"  {g:8s} detected in {(lin[mye, j] > 0).mean()*100:5.1f}% of myeloid")
    print(f"  ANY lineage (Csf1r/Aif1/Tyrobp/Fcer1g): {core[mye].mean()*100:5.1f}%")
    print(f"  C1q-ONLY (complement+, lineage-):        {c1q_only[mye].mean()*100:5.1f}%")
    print(f"  median library size  core={np.median(lib[core]):.0f}  "
          f"c1q-only={np.median(lib[c1q_only]):.0f}  all-nt={np.median(lib):.0f}")

    print(f"\n=== density link (top-decile densest tissue, d10<= {1/dense_cut:.0f}px) ===")
    print(f"  myeloid call-rate  dense={mye[in_dense].mean()*100:5.2f}%  "
          f"rest={mye[~in_dense].mean()*100:5.2f}%")
    print(f"  of myeloid, %C1q-only  dense={c1q_only[mye & in_dense].mean()*100:5.1f}%  "
          f"rest={c1q_only[mye & ~in_dense].mean()*100:5.1f}%")
    frac_dense = (mye & in_dense).sum() / max(n_my, 1)
    print(f"  {frac_dense*100:.1f}% of ALL myeloid sit in the top-decile densest tissue")

    # figure: myeloid coloured core (real) vs C1q-only (ambient), tumor black
    fig, ax = plt.subplots(figsize=(13, 7), dpi=160)
    ax.scatter(cxn, cyn, s=0.5, c="#e8e8e8", linewidths=0, rasterized=True)
    if tumor.sum():
        ax.scatter(cx[tumor], cy[tumor], s=1.2, c="black", linewidths=0,
                   rasterized=True, label=f"tumor ({int(tumor.sum()):,})")
    ax.scatter(cxn[core], cyn[core], s=5, c="#00a087", linewidths=0, rasterized=True,
               label=f"myeloid: lineage+ ({int(core.sum()):,})")
    ax.scatter(cxn[c1q_only], cyn[c1q_only], s=5, c="#d62728", linewidths=0,
               rasterized=True, label=f"myeloid: C1q-only ({int(c1q_only.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {SL} myeloid call: real lineage+ (green) vs ambient C1q-only (red)")
    ax.legend(loc="lower right", markerscale=3, fontsize=9)
    fig.savefig(f"{OUT}/myeloid_band_check.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {OUT}/myeloid_band_check.png")


if __name__ == "__main__":
    main()
