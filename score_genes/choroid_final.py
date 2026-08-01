"""Per-slice choroid = Ttr >= (GMM threshold), for all slices.

(1) Justification figure `ttr_gmm_justification.png`: for each slice, the log1p(Ttr)
    distribution (Ttr>0, non-tumor) with the fitted 2-component GMM (ambient low mode
    vs choroid high mode) and the chosen threshold = where high-component posterior
    crosses 0.5. Degenerate slices (no separated high mode) are flagged.
(2) One choroid map per slice `slice_{id}_choroid_annotation.png`: tumor black,
    choroid red, title "slice {id} Choroid annotation", legend has count + threshold.

Output -> score_genes_v3/.
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
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

OUT = "D:/thesis-research/score_genes_v3"
TUMOR_COL = "pred_tumor_XGBoost"
SLIDE = {1: "L321", 2: "L321", 3: "L321", 4: "L34", 5: "L34", 6: "L34"}
HI_MEAN_FLOOR = 5     # a real choroid high-mode must sit above this raw count
FALLBACK = 5          # used (flagged) if the GMM high mode is degenerate


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


def fit_gmm(ttr):
    """Return (thr_log, thr_raw, lo_raw, hi_raw, gm) for the 2-comp GMM on log1p(Ttr>0)."""
    lt = np.log1p(ttr[ttr > 0]).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(lt)
    order = np.argsort(gm.means_.ravel())
    lo, hi = int(order[0]), int(order[1])
    grid = np.linspace(0, lt.max(), 4000).reshape(-1, 1)
    post_hi = gm.predict_proba(grid)[:, hi]
    thr_log = float(grid[np.argmax(post_hi >= 0.5), 0])
    return (thr_log, float(np.expm1(thr_log)),
            float(np.expm1(gm.means_.ravel()[lo])),
            float(np.expm1(gm.means_.ravel()[hi])), gm, lo, hi)


def main():
    os.makedirs(OUT, exist_ok=True)
    slices = [int(sys.argv[1])] if len(sys.argv) > 1 else list(range(1, 7))
    make_just = len(sys.argv) <= 1   # rebuild the justification grid only when running all
    if make_just:
        figJ, axesJ = plt.subplots(2, 3, figsize=(18, 10), dpi=140)
        axesJ = axesJ.ravel()
    print(f"{'sl':>2} {'slide':>5} {'lo_mean':>7} {'hi_mean':>7} {'gmm_thr':>7} "
          f"{'applied':>7} {'n_choroid':>9} {'status':>22}")

    for sid in slices:
        path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
        with h5py.File(path, "r") as h5:
            X = _read_X(h5)
            var = list(_read_var(h5))
            cx = _read_num(h5, "CenterX_global_px")
            cy = _read_num(h5, "CenterY_global_px")
            tumor = _read_bool(h5, TUMOR_COL)
        keep = ~tumor
        ttr = np.asarray(X[keep][:, var.index("Ttr")].todense()).ravel()
        del X
        gc.collect()
        cxn, cyn = cx[keep], cy[keep]

        thr_log, thr_raw, lo_raw, hi_raw, gm, lo, hi = fit_gmm(ttr)
        degenerate = hi_raw < HI_MEAN_FLOOR
        if degenerate:                       # no separated choroid mode -> no choroid
            applied = None
            choroid = np.zeros(len(ttr), dtype=bool)
        else:
            applied = int(round(thr_raw))
            choroid = ttr >= applied
        n_ch = int(choroid.sum())
        status = "DEGENERATE->no choroid" if degenerate else "ok"
        print(f"{sid:>2} {SLIDE[sid]:>5} {lo_raw:>7.1f} {hi_raw:>7.1f} {thr_raw:>7.1f} "
              f"{str(applied):>7} {n_ch:>9,} {status:>22}")

        # ---- justification panel ----
        if make_just:
            ax = axesJ[sid - 1]
            lt = np.log1p(ttr[ttr > 0])
            ax.hist(lt, bins=60, density=True, color="#cfe0ee", edgecolor="none")
            xg = np.linspace(0, lt.max(), 500)
            w = gm.weights_.ravel(); m = gm.means_.ravel(); s = np.sqrt(gm.covariances_.ravel())
            for k, col, lab in [(lo, "#4c72b0", "ambient"), (hi, "#d1495b", "choroid")]:
                ax.plot(xg, w[k] * norm.pdf(xg, m[k], s[k]), c=col, lw=2, label=lab)
            if not degenerate:
                ax.axvline(np.log1p(applied), c="k", ls="--", lw=1.5,
                           label=f"thr=Ttr>= {applied}")
            ax.set_title(f"slice {sid} ({SLIDE[sid]}): GMM thr {thr_raw:.1f}"
                         + (f" -> Ttr>= {applied}" if not degenerate
                            else " [DEGENERATE -> no choroid]"), fontsize=10)
            ax.set_xlabel("log1p(Ttr)  [Ttr>0 cells]"); ax.legend(fontsize=8)

        # ---- choroid annotation map ----
        fig, axm = plt.subplots(figsize=(11, 9), dpi=160)
        axm.scatter(cxn[~choroid], cyn[~choroid], s=1.0, c="#e2e2e2", linewidths=0,
                    rasterized=True, label="other non-tumor")
        axm.scatter(cx[tumor], cy[tumor], s=1.4, c="black", linewidths=0, rasterized=True,
                    label=f"tumor (n={int(tumor.sum()):,})")
        lab = ("Choroid: none detected (Ttr GMM degenerate)" if degenerate
               else f"Choroid Ttr>= {applied}  (n={n_ch:,})")
        axm.scatter(cxn[choroid], cyn[choroid], s=7, c="#d1495b", linewidths=0,
                    rasterized=True, label=lab)
        axm.set_aspect("equal"); axm.set_xticks([]); axm.set_yticks([])
        axm.set_title(f"slice {sid} Choroid annotation")
        axm.legend(loc="lower right", markerscale=3, fontsize=10, frameon=True)
        plt.tight_layout()
        fig.savefig(f"{OUT}/slice_{sid}_choroid_annotation.png", bbox_inches="tight")
        plt.close(fig)
        del ttr, choroid, cxn, cyn
        gc.collect()

    if make_just:
        figJ.suptitle("Ttr choroid threshold per slice — 2-component GMM justification",
                      fontsize=15)
        figJ.tight_layout(rect=[0, 0, 1, 0.97])
        figJ.savefig(f"{OUT}/ttr_gmm_justification.png", bbox_inches="tight")
        plt.close(figJ)
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
