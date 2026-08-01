"""Assemble the FINAL L321 annotation and plot it.

celltype_final (non-tumor cells) =
  Stage-1 v2 backbone (Astrocytes/Neurons/Vascular)  [celltype_v2]
  + Myeloid  -> Stage-2 subtype (Microglia/BAM/MDM/Myeloid_unresolved)
  + Ependymal(scored, unreliable) DROPPED -> replaced by Choroid = Ttr>=5 (validated)
  + ambiguous/low_markers_coverage/unknown -> "unassigned"

Alignment is cross-checked by (x,y): the Stage-2 label rows must match the
cell_scores myeloid rows coordinate-for-coordinate before any label is written.

Per-type spatial plots (n + % of ALL cells) and one combined plot; tumor BLACK.
Output -> score_genes_slice{N}_v2/final/
"""
import gc
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

SLICES = [1, 2, 3]
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"
TTR_THR = 5                              # validated choroid gate for L321 (slices 1-3)
TUMOR_COL = "pred_tumor_XGBoost"

# per-type plots in this order; display name + colour
BIO = [
    ("Astrocytes",        "Astrocytes",          "#1f77b4"),
    ("Neurons",           "Neurons",             "#e377c2"),
    ("Microglia",         "Microglia",           "#17becf"),
    ("BAM",               "BAM",                 "#d62728"),
    ("MDM",               "MDM",                 "#ff7f0e"),
    ("Myeloid_unresolved","Myeloid (unresolved)","#8c564b"),
    ("Vascular",          "Vascular",            "#2ca02c"),
    ("Choroid",           "Choroid",             "#9467bd"),
]
COLOR = {k: c for k, _, c in BIO}
DISP = {k: d for k, d, _ in BIO}
UNASSIGNED = {"unknown", "ambiguous", "low_markers_coverage", "unassigned"}


def _decode(a):
    return (np.array([x.decode() if isinstance(x, bytes) else x for x in a])
            if a.dtype.kind in ("O", "S") else a)


def _read_X(h5):
    node = h5["X"]
    enc = str(node.attrs.get("encoding-type", ""))
    shape = tuple(node.attrs["shape"])
    args = (node["data"][...], node["indices"][...], node["indptr"][...])
    M = csc_matrix(args, shape=shape) if "csc" in enc else csr_matrix(args, shape=shape)
    return M.tocsr()


def _read_var(h5):
    var = h5["var"]; key = var.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else key
    return list(_decode(var[key][...]))


def _read_num(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]; cats = _decode(node["categories"][...]).astype(float)
        return cats[np.clip(codes, 0, None)]
    return node[...].astype(float)


def _read_bool(h5, c):
    node = h5["obs"][c]
    if isinstance(node, h5py.Group):
        codes = node["codes"][...]; cats = _decode(node["categories"][...])
        vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
        return np.isin(vals.astype(str), ["True", "1", "1.0", "TRUE", "true"])
    arr = node[...]
    return (np.isin(_decode(arr).astype(str), ["True", "1"]) if arr.dtype.kind in ("S", "O")
            else arr.astype(bool))


def build(n):
    d = f"D:/thesis-research/score_genes_slice{n}_v2"
    cs = pd.read_csv(f"{d}/cell_scores.csv")           # non-tumor, ordered
    x, y = cs["x"].to_numpy(), cs["y"].to_numpy()
    final = cs["celltype_v2"].to_numpy().astype(object)

    with h5py.File(TMPL.format(n), "r") as h5:
        X = _read_X(h5); var = _read_var(h5)
        cx = _read_num(h5, "CenterX_global_px"); cy = _read_num(h5, "CenterY_global_px")
        tumor = _read_bool(h5, TUMOR_COL)
    keep = ~tumor
    assert len(cs) == int(keep.sum()), f"slice {n}: cell_scores {len(cs)} != non-tumor {int(keep.sum())}"
    ttr = np.asarray(X[keep][:, var.index("Ttr")].todense()).ravel()
    del X; gc.collect()

    # --- Myeloid -> Stage-2 subtype, with a coordinate cross-check ---
    st = pd.read_csv(f"{d}/myeloid_stage2_labels.csv")
    mye_idx = np.where(final == "Myeloid")[0]
    assert len(mye_idx) == len(st), f"slice {n}: myeloid {len(mye_idx)} != stage2 {len(st)}"
    assert np.allclose(x[mye_idx], st["cx"].to_numpy()) and \
           np.allclose(y[mye_idx], st["cy"].to_numpy()), f"slice {n}: coord misalignment!"
    sub = st["subtype"].to_numpy().astype(object)
    sub[sub == "unresolved"] = "Myeloid_unresolved"
    final[mye_idx] = sub

    # --- drop unreliable scored-Ependymal, install validated Choroid (Ttr>=THR) ---
    final[final == "Ependymal"] = "unknown"
    choroid = ttr >= TTR_THR
    stolen = pd.Series(final[choroid]).value_counts().to_dict()
    final[choroid] = "Choroid"

    final[np.isin(final, list(UNASSIGNED))] = "unassigned"

    tot_all = len(final) + int(tumor.sum())
    print(f"\nslice {n}: {tot_all:,} cells ({int(tumor.sum()):,} tumor).  Choroid={int(choroid.sum())} "
          f"(reassigned from: {stolen})")
    vc = pd.Series(final).value_counts()
    for k in [b[0] for b in BIO] + ["unassigned"]:
        if k in vc:
            print(f"   {DISP.get(k,k):22s} {vc[k]:>7,}  {100*vc[k]/tot_all:5.1f}%")

    pd.DataFrame({"x": x, "y": y, "celltype_final": final}).to_csv(
        f"{d}/annotation_final.csv", index=False)
    return x, y, final, cx[tumor], cy[tumor], tot_all


def plots(n, x, y, final, tx, ty, tot_all):
    out = f"D:/thesis-research/score_genes_slice{n}_v2/final"
    os.makedirs(out, exist_ok=True)
    tlab = f"tumor (n={len(tx):,})"

    # per-type
    for key, disp, col in BIO:
        m = final == key
        fig, ax = plt.subplots(figsize=(9, 7), dpi=160)
        ax.scatter(x, y, s=0.7, c="#e2e2e2", linewidths=0, rasterized=True, label="other")
        if len(tx):
            ax.scatter(tx, ty, s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
        ax.scatter(x[m], y[m], s=4, c=col, linewidths=0, rasterized=True, label=disp)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {n} — {disp}  (n={int(m.sum()):,}, {100*m.sum()/tot_all:.1f}% of all cells)")
        ax.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
        fig.savefig(f"{out}/slice{n}_{key}.png", bbox_inches="tight"); plt.close(fig)

    # combined
    fig, ax = plt.subplots(figsize=(11, 9), dpi=170)
    m0 = final == "unassigned"
    ax.scatter(x[m0], y[m0], s=0.7, c="#e6e6e6", linewidths=0, rasterized=True,
               label=f"unassigned ({int(m0.sum()):,})")
    if len(tx):
        ax.scatter(tx, ty, s=1.3, c="black", linewidths=0, rasterized=True, label=tlab)
    for key, disp, col in BIO:
        m = final == key
        if m.any():
            ax.scatter(x[m], y[m], s=4, c=col, linewidths=0, rasterized=True,
                       label=f"{disp} ({int(m.sum()):,})")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice {n} — final annotation (myeloid subtyped)")
    ax.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
    fig.savefig(f"{out}/slice{n}_ALL.png", bbox_inches="tight"); plt.close(fig)
    print(f"   saved -> {out}")


def main():
    for n in SLICES:
        x, y, final, tx, ty, tot = build(n)
        plots(n, x, y, final, tx, ty, tot)


if __name__ == "__main__":
    main()
