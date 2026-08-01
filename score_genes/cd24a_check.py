"""Does Cd24a help the choroid/ependymal call? For each slice, compare high-Cd24a
vs high-Ttr among NON-TUMOR cells: correlation, overlap, and a spatial map
(Ttr-only red, Cd24a-only blue, both purple, tumor black) to see whether Cd24a
corroborates the choroid structure (esp. rescue slice 6) or scatters to the tumor.

Thresholds via per-gene 2-comp GMM on log1p(count>0). Output -> score_genes_v3/.
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
from scipy.stats import spearmanr
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


def gmm_thr(v):
    pos = v[v > 0]
    if len(pos) < 50 or pos.max() < 2:
        return np.inf
    lt = np.log1p(pos).reshape(-1, 1)
    gm = GaussianMixture(2, random_state=0).fit(lt)
    hi = int(np.argmax(gm.means_.ravel()))
    grid = np.linspace(0, lt.max(), 2000).reshape(-1, 1)
    post = gm.predict_proba(grid)[:, hi]
    return float(np.expm1(grid[np.argmax(post >= 0.5), 0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"{'sl':>2} {'slide':>5} {'cd24_p99':>8} {'cd24_thr':>8} {'nTtr':>7} "
          f"{'nCd24':>7} {'both':>6} {'rho':>6} {'coexp':>6}")
    for sid in range(1, 7):
        path = f"D:/thesis-research/resources/cache/with_tumor_prediction/slice_{sid}_adata.h5ad"
        if not os.path.exists(path):
            continue
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
        cxn, cyn = cx[keep], cy[keep]

        t_thr = gmm_thr(ttr)
        c_thr = 3.0  # Cd24a has no separable high mode (p99~3); absolute cutoff
        t_hi = ttr >= t_thr
        c_hi = cd >= c_thr
        both = t_hi & c_hi
        rho, _ = spearmanr(ttr, cd)
        # co-expression: of the Ttr choroid cells, how many are Cd24a>=3
        coexp = 100 * both.sum() / max(t_hi.sum(), 1)
        print(f"{sid:>2} {SLIDE[sid]:>5} {np.percentile(cd,99):>8.0f} {c_thr:>8.1f} "
              f"{int(t_hi.sum()):>7,} {int(c_hi.sum()):>7,} {int(both.sum()):>6,} "
              f"{rho:>6.3f} {coexp:>5.1f}%")

        # spatial: Ttr-only red, Cd24a-only blue, both purple, tumor black
        t_only = t_hi & ~c_hi
        c_only = c_hi & ~t_hi
        fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
        ax.scatter(cxn, cyn, s=0.6, c="#e6e6e6", linewidths=0, rasterized=True)
        ax.scatter(cx[tumor], cy[tumor], s=1.2, c="black", linewidths=0, rasterized=True,
                   label=f"tumor ({int(tumor.sum()):,})")
        ax.scatter(cxn[c_only], cyn[c_only], s=6, c="#1f77b4", linewidths=0, rasterized=True,
                   label=f"Cd24a-only ({int(c_only.sum()):,})")
        ax.scatter(cxn[t_only], cyn[t_only], s=6, c="#d1495b", linewidths=0, rasterized=True,
                   label=f"Ttr-only ({int(t_only.sum()):,})")
        ax.scatter(cxn[both], cyn[both], s=9, c="#7b3294", linewidths=0, rasterized=True,
                   label=f"both ({int(both.sum()):,})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {sid}  Ttr(>= {t_thr:.0f}) vs Cd24a(>= {c_thr:.0f})  choroid markers")
        ax.legend(loc="lower right", markerscale=3, fontsize=9, frameon=True)
        plt.tight_layout()
        fig.savefig(f"{OUT}/slice_{sid}_ttr_vs_cd24a.png", bbox_inches="tight")
        plt.close(fig)
        gc.collect()


if __name__ == "__main__":
    main()
