"""Review the per-slice Ttr GMM: is each fit genuinely BIMODAL, or is the GMM
splitting a heavy unimodal tail? For each slice, fit the 2-comp GMM on
log1p(Ttr>0, non-tumor) and report:
  - component weights, raw means, separation
  - whether the MIXTURE density has an interior valley between the two modes
    (a real bimodal signature) and how deep it is (dip_depth in [0,1])
Plots a viewable 2x3 grid with the histogram, the two Gaussians, the mixture
density (to see the valley), and the threshold.

Output -> score_genes_v3/ttr_gmm_review.png
"""
import gc
import os

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
    figR, axesR = plt.subplots(2, 3, figsize=(16, 9), dpi=100)
    axesR = axesR.ravel()
    print(f"{'sl':>2} {'w_lo':>6} {'w_hi':>7} {'m_lo':>5} {'m_hi':>6} {'sep_log':>7} "
          f"{'dip_depth':>9} {'verdict':>14}")
    for i, sid in enumerate(range(1, 7)):
        path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
        with h5py.File(path, "r") as h5:
            X = _read_X(h5)
            var = list(_read_var(h5))
            tumor = _read_bool(h5, TUMOR_COL)
        ttr = np.asarray(X[~tumor][:, var.index("Ttr")].todense()).ravel()
        del X
        gc.collect()

        lt = np.log1p(ttr[ttr > 0])
        gm = GaussianMixture(2, random_state=0).fit(lt.reshape(-1, 1))
        order = np.argsort(gm.means_.ravel())
        lo, hi = int(order[0]), int(order[1])
        w = gm.weights_.ravel(); m = gm.means_.ravel(); s = np.sqrt(gm.covariances_.ravel())

        # mixture density + valley between the modes
        xg = np.linspace(0, lt.max(), 4000)
        f = w[lo] * norm.pdf(xg, m[lo], s[lo]) + w[hi] * norm.pdf(xg, m[hi], s[hi])
        seg = (xg >= m[lo]) & (xg <= m[hi])
        f_lo = float(f[np.argmin(np.abs(xg - m[lo]))])
        f_hi = float(f[np.argmin(np.abs(xg - m[hi]))])
        f_anti = float(f[seg].min()) if seg.sum() > 3 else f_lo
        anti_x = float(xg[seg][np.argmin(f[seg])]) if seg.sum() > 3 else m[lo]
        interior = seg.sum() > 3 and m[lo] < anti_x < m[hi] - 1e-6
        dip = max(0.0, 1 - f_anti / max(min(f_lo, f_hi), 1e-9)) if interior else 0.0
        sep = float(m[hi] - m[lo])
        raw_hi = float(np.expm1(m[hi]))
        if raw_hi < 5:
            verdict = "DEGENERATE"
        elif dip >= 0.15:
            verdict = "bimodal"
        elif dip > 0:
            verdict = "weak-bimodal"
        else:
            verdict = "heavy-tail"
        thr_log = float(xg[np.argmax(gm.predict_proba(xg.reshape(-1, 1))[:, hi] >= 0.5)])
        print(f"{sid:>2} {w[lo]:>6.3f} {w[hi]:>7.4f} {np.expm1(m[lo]):>5.1f} "
              f"{raw_hi:>6.1f} {sep:>7.2f} {dip:>9.3f} {verdict:>14}")

        ax = axesR[i]
        ax.hist(lt, bins=70, density=True, color="#dfe7ef", edgecolor="none")
        ax.plot(xg, w[lo] * norm.pdf(xg, m[lo], s[lo]), c="#4c72b0", lw=1.6, label="ambient")
        ax.plot(xg, w[hi] * norm.pdf(xg, m[hi], s[hi]), c="#d1495b", lw=1.6, label="choroid")
        ax.plot(xg, f, c="k", lw=1.0, alpha=0.7, label="mixture")
        if interior:
            ax.axvline(anti_x, c="green", ls=":", lw=1, label="valley")
        ax.axvline(thr_log, c="k", ls="--", lw=1.2, label="thr")
        ax.set_yscale("log"); ax.set_ylim(1e-4, None)
        ax.set_title(f"slice {sid} ({SLIDE[sid]}): {verdict}, dip={dip:.2f}, "
                     f"hi_mean={raw_hi:.0f}", fontsize=9)
        ax.set_xlabel("log1p(Ttr) | Ttr>0"); ax.legend(fontsize=7)

    figR.suptitle("Ttr GMM bimodality review (log-y; valley = green dotted)", fontsize=14)
    figR.tight_layout(rect=[0, 0, 1, 0.97])
    figR.savefig(f"{OUT}/ttr_gmm_review.png", bbox_inches="tight")
    plt.close(figR)
    print(f"\nsaved -> {OUT}/ttr_gmm_review.png")


if __name__ == "__main__":
    main()
