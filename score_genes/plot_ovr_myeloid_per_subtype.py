"""Per-subtype spatial maps of the OvR Myeloid subtyping: one panel per subtype
(BAM, Microglia, MDM, unknown) over a grey tissue background, with the
pred_tumor cells overlaid in black.

Output -> score_genes_slice1_merged/classify/ovr_myeloid_per_subtype.png
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import h5py
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_homogeneity import _read_obs_num, _read_obs_bool, SLICE, TUMOR_COL  # noqa: E402

OUT = "D:/thesis-research/score_genes_slice1_merged/classify"
SUB = f"{OUT}/ovr_myeloid_subtypes.csv"
BG = f"{OUT}/ovr_nontumor_predictions.csv"
PANELS = ["BAM", "Microglia", "MDM", "unknown"]
COL = {"BAM": "#d62728", "Microglia": "#17becf", "MDM": "#00a087", "unknown": "#999999"}
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    df = pd.read_csv(SUB)
    bg = pd.read_csv(BG)
    bx, by = bg["x"].to_numpy(), bg["y"].to_numpy()
    x, y, s = df["x"].to_numpy(), df["y"].to_numpy(), df["subtype"].to_numpy()

    with h5py.File(SLICE, "r") as h5:
        tumor = _read_obs_bool(h5, TUMOR_COL)
        cx = _read_obs_num(h5, "CenterX_global_px")
        cy = _read_obs_num(h5, "CenterY_global_px")
    tx, ty = cx[tumor], cy[tumor]
    print(f"tumor cells overlaid: {len(tx):,}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=160)
    for ax, g in zip(axes.ravel(), PANELS):
        m = s == g
        ax.scatter(bx, by, s=0.5, c="#eeeeee", linewidths=0, rasterized=True)
        ax.scatter(tx, ty, s=1.5, c="black", linewidths=0, rasterized=True,
                   label=f"tumor ({len(tx):,})")
        ax.scatter(x[m], y[m], s=3, c=COL[g], linewidths=0, rasterized=True,
                   label=f"{g} ({int(m.sum()):,})")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (n={int(m.sum()):,})", fontweight="bold")
        ax.legend(loc="lower right", markerscale=4, fontsize=8)
    fig.suptitle("OvR Myeloid subtypes + tumor (black) — grey = all non-tumor",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/ovr_myeloid_per_subtype.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved ovr_myeloid_per_subtype.png -> {OUT}")


if __name__ == "__main__":
    main()
