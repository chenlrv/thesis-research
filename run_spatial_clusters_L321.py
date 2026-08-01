"""Spatial plot of Leiden clusters for sample L321.

Loads the cached sample AnnData and colors cells in physical space by the
Basic.run_Neighbor.network.expression.space.1_1_cluster_Basic.run_Leiden.Clustering.1_1
obs column.
"""
import h5py
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = "D:/thesis-research/resources/cache/sample_L321_adata.h5ad"
CLUSTER_COL = (
    "Basic.run_Neighbor.network.expression.space.1_1"
    "_cluster_Basic.run_Leiden.Clustering.1_1"
)
X_KEY = "CenterX_global_px"
Y_KEY = "CenterY_global_px"
OUTPUT = "D:/thesis-research/L321_spatial_leiden_clusters.png"
HIGHLIGHT_CLUSTER = "4"
OUTPUT_HIGHLIGHT = "D:/thesis-research/L321_spatial_cluster4.png"


def _read_obs_column(h5, col: str) -> np.ndarray:
    """Read a single /obs column from an .h5ad file, handling categorical encoding."""
    node = h5["obs"][col]
    if isinstance(node, h5py.Group):
        # Categorical: codes + categories
        codes = node["codes"][...]
        categories = node["categories"][...]
        if categories.dtype.kind in ("O", "S"):
            categories = np.array([
                c.decode() if isinstance(c, bytes) else c for c in categories
            ])
        out = np.where(codes >= 0, categories[np.clip(codes, 0, None)], None)
        return out
    arr = node[...]
    if arr.dtype.kind in ("O", "S"):
        arr = np.array([a.decode() if isinstance(a, bytes) else a for a in arr])
    return arr


def main() -> None:
    print(f"Loading {CACHE} via h5py (skipping problematic obs columns)...")
    with h5py.File(CACHE, "r") as h5:
        if CLUSTER_COL not in h5["obs"]:
            raise KeyError(f"{CLUSTER_COL!r} not in /obs")
        cluster_vals = _read_obs_column(h5, CLUSTER_COL)
        x_vals = _read_obs_column(h5, X_KEY).astype(float)
        y_vals = _read_obs_column(h5, Y_KEY).astype(float)

    obs = pd.DataFrame({CLUSTER_COL: cluster_vals, X_KEY: x_vals, Y_KEY: y_vals})
    print(f"Loaded {len(obs)} cells")
    raw = obs[CLUSTER_COL].astype(str)

    # Sort cluster labels numerically when possible, lexicographically otherwise.
    def _sort_key(label: str):
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)

    categories = sorted(raw.unique(), key=_sort_key)
    cats = pd.Categorical(raw, categories=categories, ordered=True)
    n_cats = len(categories)
    print(f"n clusters: {n_cats}")
    print(f"clusters: {categories}")

    # AtoMx-like palette: vivid red, teals/greens, lavender, orange, yellow, gray.
    atomx_palette = [
        "#E03A3E",  # vivid red
        "#3FB39A",  # teal
        "#9FD9B4",  # sage green
        "#C8B6E2",  # pale lavender
        "#F2A341",  # orange
        "#F4E04D",  # yellow
        "#7FB3D5",  # light blue
        "#B07AA1",  # mauve
        "#9C9C9C",  # gray
        "#59A14F",  # green
        "#E15759",  # coral
        "#76B7B2",  # muted teal
        "#EDC948",  # mustard
        "#AF7AA1",  # purple
        "#FF9DA7",  # pink
        "#9C755F",  # brown
        "#BAB0AC",  # warm gray
        "#4E79A7",  # blue
        "#F28E2B",  # deep orange
        "#8CD17D",  # light green
    ]
    if n_cats <= len(atomx_palette):
        colors = atomx_palette[:n_cats]
    else:
        extra = plt.get_cmap("hsv", n_cats - len(atomx_palette))
        colors = atomx_palette + [extra(i) for i in range(n_cats - len(atomx_palette))]
    cat2color = dict(zip(categories, colors))

    x = obs[X_KEY].to_numpy()
    y = obs[Y_KEY].to_numpy()
    point_colors = np.array([cat2color[c] for c in cats.astype(str)])

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.scatter(x, y, c=point_colors, s=0.5, linewidths=0, rasterized=True)

    legend_handles = [
        mlines.Line2D([], [], color=cat2color[c], marker="o", linestyle="None",
                      markersize=8, label=str(c))
        for c in categories
    ]
    ax.legend(
        handles=legend_handles,
        title="Leiden cluster",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
        prop={"size": 9},
        ncol=1 if n_cats <= 25 else 2,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(X_KEY)
    ax.set_ylabel(Y_KEY)
    ax.set_title(f"Sample L321 — spatial plot colored by Leiden clusters ({n_cats})")

    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    plt.show()

    # ---- Second figure: only cells in HIGHLIGHT_CLUSTER ----
    if HIGHLIGHT_CLUSTER not in categories:
        print(f"Cluster {HIGHLIGHT_CLUSTER!r} not present — skipping highlight plot.")
        return

    mask = (cats.astype(str) == HIGHLIGHT_CLUSTER)
    n_in = int(mask.sum())
    print(f"Cluster {HIGHLIGHT_CLUSTER}: {n_in} cells")

    fig2, ax2 = plt.subplots(figsize=(12, 12), dpi=300)
    ax2.scatter(
        x[mask], y[mask],
        c=[cat2color[HIGHLIGHT_CLUSTER]],
        s=0.5, linewidths=0, rasterized=True,
    )
    # Keep the same axis extent as the full plot so the cluster's spatial location is obvious.
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(y.min(), y.max())
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(X_KEY)
    ax2.set_ylabel(Y_KEY)
    ax2.set_title(f"Sample L321 — cluster {HIGHLIGHT_CLUSTER} only ({n_in} cells)")

    plt.tight_layout()
    plt.savefig(OUTPUT_HIGHLIGHT, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_HIGHLIGHT}")
    plt.show()


if __name__ == "__main__":
    main()
