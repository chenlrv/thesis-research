"""Compare MDM gating strategies:
  Left  — original: MDM = GFP-hi AND tdTomato-hi
  Right — alternative: MDM = tdTomato-hi only (captures mature MDMs that lost GFP)
"""
import sys
sys.path.insert(0, "/")

import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture

SLICE_PATH = "/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
OUTPUT     = "D:/thesis-research/mdm_gating_comparison.png"
TUMOR_COL  = "pred_tumor_XGBoost"

PALETTE = {
    "Other":         "#CCCCCC",
    "Brain_Myeloid": "#F1C40F",
    "MDM":           "#2ECC71",
    "Tumor":         "#E74C3C",
}
DRAW_ORDER = ["Other", "Brain_Myeloid", "MDM", "Tumor"]

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading adata ...")
adata = ad.read_h5ad(SLICE_PATH)
print(f"  {adata.n_obs:,} cells")

panel = {g.lower(): g for g in adata.var_names}

def get_expr(gene: str) -> np.ndarray:
    key = panel.get(gene.lower())
    if key is None:
        raise KeyError(f"'{gene}' not found in panel")
    x = adata[:, key].X
    return x.toarray().ravel() if issparse(x) else np.asarray(x).ravel()

def gmm_threshold(expr: np.ndarray) -> float:
    if (expr > 0).sum() < 20:
        return float(np.percentile(expr, 90))
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(expr.reshape(-1, 1))
    means = sorted(gmm.means_.ravel())
    return float(np.mean(means))

gfp_expr = get_expr("GFP")
tdt_expr = get_expr("tdTomato")
gfp_thresh = gmm_threshold(gfp_expr)
tdt_thresh = gmm_threshold(tdt_expr)

gfp_hi = gfp_expr > gfp_thresh
tdt_hi = tdt_expr > tdt_thresh

print(f"GFP-hi:      {gfp_hi.sum():,}  (threshold {gfp_thresh:.2f})")
print(f"tdTomato-hi: {tdt_hi.sum():,}  (threshold {tdt_thresh:.2f})")

# ── Tumor mask ────────────────────────────────────────────────────────────
if TUMOR_COL not in adata.obs.columns:
    pred_cols = [c for c in adata.obs.columns if c.startswith("pred_tumor_")]
    if not pred_cols:
        raise KeyError(f"No tumor column found. Available: {list(adata.obs.columns)}")
    TUMOR_COL = pred_cols[0]
tumor_mask = adata.obs[TUMOR_COL].astype(int).to_numpy() == 1
print(f"Tumor cells: {tumor_mask.sum():,}  (column: {TUMOR_COL})")

x = adata.obs["CenterX_global_px"].to_numpy(dtype=float)
y = adata.obs["CenterY_global_px"].to_numpy(dtype=float)
total = adata.n_obs

# ── Classification A: original (GFP & tdT) ───────────────────────────────
class_a = np.where(
    tumor_mask,                              "Tumor",
    np.where(gfp_hi & tdt_hi,               "MDM",
    np.where(gfp_hi & ~tdt_hi,              "Brain_Myeloid",
                                             "Other"))
)

# ── Classification B: tdTomato-only MDM ──────────────────────────────────
class_b = np.where(
    tumor_mask,                              "Tumor",
    np.where(tdt_hi,                         "MDM",
    np.where(gfp_hi & ~tdt_hi,              "Brain_Myeloid",
                                             "Other"))
)

def count(arr): return {c: int((arr == c).sum()) for c in DRAW_ORDER}
counts_a = count(class_a)
counts_b = count(class_b)

print("\nOriginal (GFP & tdT):    ", {k: v for k, v in counts_a.items()})
print("Alternative (tdT only):  ", {k: v for k, v in counts_b.items()})

# ── Plot ──────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), dpi=150)
fig.suptitle("MDM gating strategy comparison — Slice 1", fontsize=13)

def draw_panel(ax, cell_class, counts, title):
    for cls in DRAW_ORDER:
        mask = cell_class == cls
        if not mask.any():
            continue
        is_bg = cls == "Other"
        ax.scatter(
            x[mask], y[mask],
            c=PALETTE[cls],
            s=0.8 if is_bg else 2.0,
            alpha=0.3 if is_bg else 0.9,
            linewidths=0,
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               color=PALETTE[c],
               label=f"{c}   n={counts[c]:,}  ({counts[c]/total*100:.1f}%)")
        for c in DRAW_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True,
              fontsize=9, edgecolor="0.7", framealpha=0.9)

draw_panel(
    ax1, class_a, counts_a,
    f"Original: MDM = GFP-hi AND tdTomato-hi\n"
    f"MDM n={counts_a['MDM']:,} | Brain_Myeloid n={counts_a['Brain_Myeloid']:,}"
)
draw_panel(
    ax2, class_b, counts_b,
    f"Alternative: MDM = tdTomato-hi only\n"
    f"MDM n={counts_b['MDM']:,} | Brain_Myeloid n={counts_b['Brain_Myeloid']:,}"
)

plt.tight_layout()
plt.savefig(OUTPUT, bbox_inches="tight", dpi=150)
plt.show()
print(f"\nSaved: {OUTPUT}")
