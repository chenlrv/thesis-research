"""Is there a Cd24a+ ependymal-lining structure ADJACENT to the Ttr choroid?
Control slice (no tumor). Zooms on the choroid/ventricle region, overlays
Ttr-high (choroid) vs Cd24a-high (candidate epithelial/CSF-interface), and
measures how close Cd24a+ cells sit to the choroid (a real lining would hug it;
scattered Cd24a+ would not).

Usage: python choroid_ependyma_zoom.py <control_slice_id>  (default 3)
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
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial import cKDTree

OUT = "D:/thesis-research/score_genes_v3"
TUMOR_COL = "pred_tumor_XGBoost"
TTR_HI = 5
CD_HI = 2


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
    sid = sys.argv[1] if len(sys.argv) > 1 else "3"
    path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
    with h5py.File(path, "r") as h5:
        X = _read_X(h5)
        var = list(_read_var(h5))
        cx = _read_num(h5, "CenterX_global_px")
        cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)
    keep = ~tumor
    Xk = X[keep]
    ttr = np.asarray(Xk[:, var.index("Ttr")].todense()).ravel()
    cd = np.asarray(Xk[:, var.index("Cd24a")].todense()).ravel()
    del X, Xk
    gc.collect()
    cx, cy = cx[keep], cy[keep]

    ch = ttr >= TTR_HI               # choroid
    cdhi = cd >= CD_HI               # candidate epithelial / CSF interface
    pts = np.c_[cx, cy]

    # nearest-neighbour distance scale (median NN over all cells) to calibrate "adjacent"
    tree_all = cKDTree(pts)
    nn = tree_all.query(pts[np.random.default_rng(0).choice(len(pts), 3000)], k=2)[0][:, 1]
    nn_med = float(np.median(nn))

    # distance from each Cd24a+ cell to the nearest choroid cell
    tree_ch = cKDTree(pts[ch])
    d_cd = tree_ch.query(pts[cdhi], k=1)[0]
    near = d_cd <= 5 * nn_med        # within ~5 cell-widths of the choroid = candidate lining
    print(f"slice {sid} (control): choroid(Ttr>={TTR_HI})={int(ch.sum()):,}, "
          f"Cd24a>={CD_HI}={int(cdhi.sum()):,}")
    print(f"  median NN cell spacing: {nn_med:.0f}px  (adjacency cutoff {5*nn_med:.0f}px)")
    print(f"  Cd24a+ within {5*nn_med:.0f}px of choroid: {int(near.sum()):,} "
          f"({100*near.mean():.1f}% of Cd24a+)  -> candidate ependymal lining")
    print(f"  Cd24a+ distance-to-choroid percentiles: "
          + ", ".join(f"p{p}={np.percentile(d_cd,p):.0f}" for p in [25, 50, 75, 90]))

    # zoom bbox around the choroid, expanded
    chx, chy = cx[ch], cy[ch]
    mx, my = chx.mean(), chy.mean()
    half = 1.6 * max(chx.std(), chy.std()) + 300
    xlim = (mx - half, mx + half); ylim = (my - half, my + half)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)
    for ax in axes:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        inview = (cx >= xlim[0]) & (cx <= xlim[1]) & (cy >= ylim[0]) & (cy <= ylim[1])
        ax.scatter(cx[inview], cy[inview], s=3, c="#e6e6e6", linewidths=0, rasterized=True)

    axes[0].scatter(cx[ch], cy[ch], s=12, c="#d1495b", linewidths=0, rasterized=True,
                    label=f"Ttr>={TTR_HI} choroid")
    axes[0].scatter(cx[cdhi], cy[cdhi], s=12, c="#1f77b4", linewidths=0, rasterized=True,
                    alpha=0.7, label=f"Cd24a>={CD_HI}")
    axes[0].legend(loc="lower right", fontsize=10, markerscale=2)
    axes[0].set_title(f"slice {sid} choroid (Ttr) vs Cd24a — zoom on ventricle")

    # Cd24a+ colored by distance to choroid (blue=near lining, yellow=far)
    idx = np.where(cdhi)[0]
    o = np.argsort(-d_cd)
    axes[1].scatter(cx[ch], cy[ch], s=12, c="#d1495b", linewidths=0, rasterized=True,
                    label="choroid")
    s = axes[1].scatter(cx[idx[o]], cy[idx[o]], s=12, c=d_cd[o], cmap="viridis_r",
                        vmax=5 * nn_med, linewidths=0, rasterized=True)
    fig.colorbar(s, ax=axes[1], shrink=0.7, label="Cd24a+ dist to choroid (px)")
    axes[1].set_title("Cd24a+ by distance to choroid (dark = adjacent = candidate lining)")

    fig.suptitle(f"slice {sid}: is there a Cd24a+ ependymal lining next to the choroid?",
                 fontsize=15)
    plt.tight_layout()
    fig.savefig(f"{OUT}/slice_{sid}_ependyma_probe.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {OUT}/slice_{sid}_ependyma_probe.png")


if __name__ == "__main__":
    main()
