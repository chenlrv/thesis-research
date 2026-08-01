"""Per-cell astrocyte gate on GFAP + Sparcl1, calling a gene 'expressed' only when
its count exceeds the cell's RANDOM-NOISE level estimated from the Negative probes.

Noise model (library-size-scaled, robust to the 11-probes-read-0 problem):
    R      = (sum of all Negative* counts) / (n_neg * sum of all panel counts)
             = global ambient counts per probe per unit panel expression
    lam_i  = R * T_i           (T_i = cell i's total panel counts)
             = expected non-specific counts for ONE gene in cell i
A gene is 'expressed' in cell i iff:  count>0 AND P(X >= count | Poisson(lam_i)) < ALPHA.
Because lam_i grows with library size, a high-content cell must show MORE GFAP/Sparcl1
to clear noise -> single-molecule spillover no longer passes.

astrocyte := GFAP expressed OR Sparcl1 expressed   (both -> high confidence)
Usage: python run_astrocyte_gate_noise_slice3.py [slice_id]
"""
import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix
from scipy.stats import poisson

SLICE_ID = sys.argv[1] if len(sys.argv) > 1 else "3"
SLICE = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{SLICE_ID}_adata.h5ad"
NEG_FILE = f"D:/thesis-research/resources/cache/slice_{SLICE_ID}_adata_with_neg.h5ad"
OUT_DIR = f"D:/thesis-research/astrocyte_gate_slice{SLICE_ID}"
TUMOR_COL = "pred_tumor_XGBoost"
ALPHA = 0.01
ANCHORS = ["GFAP", "Sparcl1"]


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


def _read_var_names(h5):
    var = h5["var"]
    key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return _decode(var[key][...])


def _read_obs_num(h5, col):
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]
        cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_obs_bool(h5, col):
    node = h5["obs"][col]
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
    os.makedirs(OUT_DIR, exist_ok=True)
    with h5py.File(SLICE, "r") as h5:
        X = _read_X(h5)
        var_names = list(_read_var_names(h5))
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    with h5py.File(NEG_FILE, "r") as h5:
        negX = _read_X(h5)
        neg_vn = list(_read_var_names(h5))
    neg_idx = [i for i, g in enumerate(neg_vn) if g.lower().startswith("negative")]
    n_neg = len(neg_idx)
    assert negX.shape[0] == X.shape[0], "neg file not row-aligned"

    keep = ~tumor
    X = X[keep]; cx = cx[keep]; cy = cy[keep]
    neg_total = np.asarray(negX[keep][:, neg_idx].sum(1)).ravel()
    print(f"{X.shape[0]} non-tumor cells; {n_neg} negative probes "
          f"(removed {int(tumor.sum())} tumor)")

    # ---- library-size-scaled noise rate ----
    T = np.asarray(X.sum(1)).ravel().astype(float)         # panel total per cell
    R = neg_total.sum() / (n_neg * T.sum())                # ambient per probe per count
    lam = R * T                                            # expected noise per gene, per cell
    print(f"global ambient rate R = {R:.3e} (neg counts/probe/panel-count)")
    print(f"per-cell noise lam = R*T: median={np.median(lam):.3f}, "
          f"p90={np.quantile(lam,0.9):.3f}, max={lam.max():.3f}")

    vi = {g: i for i, g in enumerate(var_names)}

    def expressed(gene):
        c = np.asarray(X[:, vi[gene]].todense()).ravel()
        return (c > 0) & (poisson.sf(c - 1, lam) < ALPHA), c

    # show the count threshold the noise model implies across cells
    kmax = 6
    thr = np.full(X.shape[0], kmax + 1)
    for k in range(1, kmax + 1):
        passes_k = poisson.sf(k - 1, lam) < ALPHA
        thr = np.minimum(thr, np.where(passes_k, k, kmax + 1))
    vals, cnts = np.unique(np.clip(thr, 1, kmax + 1), return_counts=True)
    print("\nmin count needed to clear noise (per-cell threshold distribution):")
    for v, c in zip(vals, cnts):
        lbl = f">{kmax}" if v == kmax + 1 else str(int(v))
        print(f"  need >= {lbl:>3s} counts: {c:6d} cells ({100*c/len(thr):.1f}%)")

    gfap, gc = expressed("GFAP")
    sparc, sc_ = expressed("Sparcl1")
    astro = gfap | sparc
    both = gfap & sparc
    gonly = gfap & ~sparc
    sonly = sparc & ~gfap
    n = X.shape[0]

    print(f"\n=== astrocyte gate (GFAP|Sparcl1 vs scaled noise, alpha={ALPHA}) ===")
    print(f"  GFAP expressed:    {gfap.sum():6d} ({100*gfap.mean():.1f}%)")
    print(f"  Sparcl1 expressed: {sparc.sum():6d} ({100*sparc.mean():.1f}%)")
    print(f"  --> ASTROCYTE (OR): {astro.sum():6d} ({100*astro.mean():.1f}% of non-tumor)")
    print(f"        both (high conf): {both.sum():6d} ({100*both.mean():.1f}%)")
    print(f"        GFAP only:        {gonly.sum():6d} ({100*gonly.mean():.1f}%)")
    print(f"        Sparcl1 only:     {sonly.sum():6d} ({100*sonly.mean():.1f}%)")
    print(f"  (previous lenient gate gave ~18.6%; expect this to be lower/cleaner)")

    # ---- spatial plots ----
    x, y = cx, -cy
    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    ax.scatter(x, y, s=1.0, c="lightgrey", linewidths=0, rasterized=True)
    ax.scatter(x[astro], y[astro], s=3.0, c="#2ca02c", linewidths=0, rasterized=True)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_{SLICE_ID} astrocyte gate (GFAP|Sparcl1 > scaled noise) "
                 f"n={int(astro.sum())}, {100*astro.mean():.1f}%")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/astrocyte_spatial_noise.png", dpi=180,
                                    bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(11, 9), dpi=180)
    ax.scatter(x, y, s=1.0, c="lightgrey", linewidths=0, rasterized=True)
    ax.scatter(x[sonly], y[sonly], s=2.5, c="#b5cf6b", linewidths=0, rasterized=True,
               label=f"Sparcl1 only ({int(sonly.sum())})")
    ax.scatter(x[gonly], y[gonly], s=2.5, c="#31a354", linewidths=0, rasterized=True,
               label=f"GFAP only ({int(gonly.sum())})")
    ax.scatter(x[both], y[both], s=4.0, c="#006d2c", linewidths=0, rasterized=True,
               label=f"GFAP+Sparcl1 ({int(both.sum())})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=3, loc="upper right", fontsize=9)
    ax.set_title(f"slice_{SLICE_ID} astrocyte confidence tiers (scaled-noise gate)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/astrocyte_confidence_noise.png", dpi=180,
                                    bbox_inches="tight"); plt.close()

    pd.DataFrame({"x": cx, "y": cy, "astrocyte": astro, "GFAP_expr": gfap,
                  "Sparcl1_expr": sparc, "both": both}).to_csv(
        f"{OUT_DIR}/astrocyte_cells_noise.csv", index=False)
    print(f"\nsaved astrocyte_spatial_noise.png, astrocyte_confidence_noise.png, "
          f"astrocyte_cells_noise.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
