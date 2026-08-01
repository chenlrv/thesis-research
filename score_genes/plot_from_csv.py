"""Render spatial plots from a slice's cell_scores.csv only (no expression matrix
in memory -> avoids the matplotlib OOM the full pipeline hits on big slices).

Tumor cells are NOT in the CSV (removed before scoring), so their coordinates are
read straight from the h5ad obs (cheap: obs columns only, no X) and drawn in BLACK.

Makes: label_spatial_ALL.png, label_spatial_<type>.png (x5), score_spatial.png.
Usage: python plot_from_csv.py <slice_id>
"""
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TYPES = ["Astrocytes", "Neurons", "Myeloid", "Ependymal", "Vascular"]
COLORS = {
    "Astrocytes": "#1f77b4", "Neurons": "#e377c2", "Myeloid": "#00a087",
    "Ependymal": "#984ea3", "Vascular": "#2ca02c",
}
EXTRA = {"ambiguous": "#ff7f0e", "low_markers_coverage": "#bcbd22"}
TUMOR_COLOR = "#000000"
TUMOR_COL = "pred_tumor_XGBoost"
TMPL = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_{}_adata.h5ad"


def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


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


def _read_tumor_xy(sl):
    """Tumor cell coordinates only (obs columns; no expression matrix loaded)."""
    with h5py.File(TMPL.format(sl), "r") as h5:
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
        tumor = _read_obs_bool(h5, TUMOR_COL)
    return cx[tumor], cy[tumor]


def main(sl):
    d = f"D:/thesis-research/score_genes_slice{sl}_v2"
    df = pd.read_csv(f"{d}/cell_scores.csv")
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    lab = df["celltype_v2"].to_numpy()
    tx, ty = _read_tumor_xy(sl)
    tlab = f"tumor (n={len(tx):,})"

    # combined map
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    ax.scatter(x, y, s=1.0, c="#eaeaea", linewidths=0, rasterized=True, label="unknown")
    if len(tx):
        ax.scatter(tx, ty, s=1.6, c=TUMOR_COLOR, linewidths=0, rasterized=True, label=tlab)
    for cat, col in EXTRA.items():
        m = lab == cat
        ax.scatter(x[m], y[m], s=1.4, c=col, linewidths=0, rasterized=True, label=cat)
    for t in TYPES:
        m = lab == t
        ax.scatter(x[m], y[m], s=2.0, c=COLORS[t], linewidths=0, rasterized=True, label=t)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"slice_{sl} annotation (5-type, FDR+margin+coverage)")
    ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
    plt.tight_layout(); plt.savefig(f"{d}/label_spatial_ALL.png", bbox_inches="tight"); plt.close()

    # per-type maps
    for t in TYPES:
        m = lab == t
        fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
        ax.scatter(x, y, s=0.8, c="#dcdcdc", linewidths=0, rasterized=True, label="other non-tumor")
        if len(tx):
            ax.scatter(tx, ty, s=1.6, c=TUMOR_COLOR, linewidths=0, rasterized=True, label=tlab)
        ax.scatter(x[m], y[m], s=2.4, c=COLORS[t], linewidths=0, rasterized=True, label=t)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice_{sl} {t}  (n={int(m.sum()):,})")
        ax.legend(loc="lower right", markerscale=4, frameon=True, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"{d}/label_spatial_{t.lower()}.png", bbox_inches="tight")
        plt.close()

    # per-type score maps
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=130)
    axes = axes.ravel()
    for ax, t in zip(axes, TYPES):
        v = df[f"score_{t}"].to_numpy()
        vmax = np.quantile(np.abs(v), 0.99) or 1.0
        o = np.argsort(np.abs(v))
        s = ax.scatter(x[o], y[o], s=2, c=v[o], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       linewidths=0, rasterized=True)
        fig.colorbar(s, ax=ax, shrink=0.7)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t}  (%>0: {100*(v>0).mean():.0f}%)")
    axes[-1].axis("off")
    fig.suptitle(f"slice_{sl} per-cell score_genes in space (red = high)", fontsize=15)
    plt.tight_layout(); plt.savefig(f"{d}/score_spatial.png", bbox_inches="tight"); plt.close()
    print(f"regenerated plots (tumor in black, n={len(tx):,}) -> {d}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
