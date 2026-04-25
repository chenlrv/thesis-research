import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import re
import numpy as np

import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import numpy as np


def plot_insitu_clusters(adata, cluster_key='RNA_Basic.run_Cell.Typing.InSituType.1_1_clusters'):
    # 1. Setup Coordinates (Matching your inversion logic)
    # Note: Using .copy() if you don't want to permanently flip the adata
    x_coords = -adata.obs["CenterX_global_px"]
    y_coords = -adata.obs["CenterY_global_px"]
    adata = adata.copy()

    # 2. Identify Background vs Clusters
    # Anything NaN or not in a cluster is treated as background
    is_background = adata.obs[cluster_key].isna()
    clusters = adata.obs[cluster_key].dropna().unique()

    plt.figure(figsize=(12, 12))

    # LAYER 1: Background (All non-clustered cells)
    plt.scatter(
        x_coords[is_background],
        y_coords[is_background],
        c="#E0E0E0",  # Light grey
        s=0.5,  # Tiny dots
        alpha=0.5,
        edgecolors="none",
        label="Background"
    )

    # LAYER 2: Clusters (Iterate to color each)
    # We use a colormap to handle multiple cluster IDs automatically
    colors = plt.cm.get_cmap('tab20', len(clusters))

    for i, cluster_id in enumerate(clusters):
        mask = adata.obs[cluster_key] == cluster_id
        plt.scatter(
            x_coords[mask],
            y_coords[mask],
            c=[colors(i)],
            s=1.2,  # Slightly larger to pop
            alpha=0.9,
            edgecolors="none",
            label=f"Cluster {cluster_id}"
        )

    plt.title(f"InSituType Clusters", fontsize=15)
    plt.axis("equal")
    plt.axis("off")

    # Adjust legend to be readable if there are many clusters
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=4)
    plt.tight_layout()
    plt.show()


def plot_insitu_cluster_f(adata, cluster_key='RNA_Basic.run_Cell.Typing.InSituType.1_1_clusters'):
    x_coords = -adata.obs["CenterX_global_px"]
    y_coords = -adata.obs["CenterY_global_px"]

    # 1. Redefine Background: Include NaNs AND any cell NOT in cluster 'f'
    # This ensures everything else is plotted in grey first
    is_f = adata.obs[cluster_key] == 'f'
    is_background = ~is_f

    plt.figure(figsize=(12, 12))

    colors = plt.cm.get_cmap('tab20', 20)

    # LAYER 1: Background (Everything except Cluster f)
    plt.scatter(
        x_coords[is_background],
        y_coords[is_background],
        c="#CFCFCF",
        s=0.7,
        alpha=1,  # Lower alpha for better contrast
        edgecolors="none",
        label="Background"
    )

    # LAYER 2: Cluster 'f' (The foreground)
    plt.scatter(
        x_coords[is_f],
        y_coords[is_f],
        c=[colors(ord('j')-ord('a') + 1)],  # Matching your image's color
        s=1.5,  # Slightly larger to pop
        alpha=1.0,
        edgecolors="none",
        label="Cluster f"
    )

    plt.title(f"InSituType Cluster Suspected Tumor", fontsize=15)
    plt.axis("equal")
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', markerscale=4)
    plt.tight_layout()
    plt.show()


# Usage:
# plot_insitu_clusters(adata)

def _many_distinct_colors(n: int, seed: int = 0):
    """
    Return n distinct RGBA colors.
    Prefers Glasbey (best for many categories). Falls back to HSV.
    """
    try:
        import colorcet as cc  # pip install colorcet

        return np.array(cc.glasbey[:n])
    except Exception:
        rng = np.random.default_rng(seed)
        hues = np.linspace(0, 1, n, endpoint=False)
        rng.shuffle(hues)  # prevents similar neighbors
        return plt.cm.hsv(hues)


def to_coarse_cell_type(ct: str) -> str:
    if pd.isna(ct):
        return pd.NA
    s = str(ct).strip().lower()

    # immune
    if re.search(r"\bt cell\b|\bt-cell\b", s):
        return "Tcells"
    if re.search(r"\bmicroglia\b", s):
        return "Microglia"
    if re.search(r"\bmacrophage\b|\bperivascular macrophages\b", s):
        return "Macrophages"

    # vasculature
    if re.search(r"\bpericytes?\b|\bendothelial(s)?\b|\bvascular\b", s):
        return "Vascular"

    # glia
    if re.search(r"\bastrocytes?\b|bergmann glia", s):
        return "Astrocytes"

    if re.search(r"\boligodendrocyte precursor\b|\bopc\b", s):
        return "OPC"
    if re.search(
        r"\boligodendrocyte\b|myelin-forming|newly-formed|committed oligodendrocytes|mature oligodendrocytes",
        s,
    ):
        return "Oligodendrocytes"

    # lining / barrier
    if re.search(r"\btanycytes?\b|\bependymal\b|\bhypendymal\b", s):
        return "Ependymal"
    if re.search(r"\bchoroid plexus\b", s):
        return "Choroid Plexus"

    # development
    if re.search(r"\bneuroblasts?\b", s):
        return "Neuroblasts"
    if re.search(r"cajal-retzius", s):
        return "CajalRetzius"

    # neurons
    if re.search(r"\bexcitatory neurons?\b", s):
        return "Excitatory Neurons"
    if re.search(
        r"\binhibitory neurons?\b|\bmedium spiny neurons?\b|\binterneurons?\b|\bneurogliaform\b|\bcck interneurons\b",
        s,
    ):
        return "Inhibitory Neurons"

    # modulatory neurons
    if re.search(r"\bdopaminergic\b|\bserotonergic\b|\bcholinergic\b|\bpeptidergic\b", s):
        return "Modulatory"

    # cerebellar special cases
    if re.search(r"\bpurkinje\b|\bgranule neurons?\b", s):
        return "Cerebellar"

    return ct


CELLTYPE_GROUPS = {
    "Group 1 (Astro/Ependymal/Innate/Vascular/Neuroblasts)": [
        "Astrocytes",
        "Ependymal",
        "Macrophages",
        "Microglia",
        "Vascular",
        "Neuroblasts",
    ],
    "Group 2 (Oligo lineage / T cells / Choroid plexus)": [
        "OPC",
        "Oligodendrocytes",
        "Tcells",
        "Choroid Plexus",
    ],
    "Group 3 (Neuronal / Cajal-Retzius / Cerebellar / Modulatory)": [
        "CajalRetzius",
        "Cerebellar",
        "Excitatory Neurons",
        "Inhibitory Neurons",
        "Modulatory",
    ],
}


def plot_celltypes_groups_per_sample(
    adata,
    sample_id: str,
    groups: dict,
    x_key: str = "CenterX_global_px",
    y_key: str = "CenterY_global_px",
    color_col: str = "coarse_cell_type",
    s: float = 0.4,
    alpha: float = 0.9,
    invert_y: bool = True,
    figsize=(18, 6),
    dpi=300,
    savepath: str | None = None,
):
    sub = adata[adata.obs["sample_id"] == sample_id].copy()
    sub = sub[sub.obs[color_col].notna()].copy()
    sub.obs[color_col] = sub.obs[color_col].astype("category")

    fig, axes = plt.subplots(1, len(groups), figsize=figsize, dpi=dpi, sharex=True, sharey=True)
    if len(groups) == 1:
        axes = [axes]

    x = sub.obs[x_key].to_numpy()
    y = sub.obs[y_key].to_numpy()

    for ax, (gname, members) in zip(axes, groups.items()):
        # keep only cell types in this group
        gsub = sub[sub.obs[color_col].isin(members)].copy()

        if gsub.n_obs == 0:
            ax.set_title(f"{gname}\n(no cells)")
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_aspect("equal")
            if invert_y:
                ax.invert_yaxis()
            continue

        cats = [c for c in members if c in set(gsub.obs[color_col])]
        palette = _many_distinct_colors(len(cats), seed=0)
        cat2color = dict(zip(cats, palette))

        colors = gsub.obs[color_col].astype(str).map(cat2color).to_numpy()
        ax.scatter(
            gsub.obs[x_key].to_numpy(),
            gsub.obs[y_key].to_numpy(),
            s=s,
            c=colors,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

        if invert_y:
            ax.invert_yaxis()

        ax.set_title(gname)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_aspect("equal")

        # legend (only for types present)
        handles = [
            plt.Line2D(
                [0], [0], marker="o", linestyle="", markersize=6, color=cat2color[c], label=c
            )
            for c in cats
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.suptitle(f"Cell types by group (sample {sample_id})", y=1.02)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", savepath)
    else:
        plt.show()


def add_celltype_to_adata(
    adata,
    annot_path: str,
):
    ann = pd.read_csv(annot_path)  # expects columns: global_cell_id, cell_type, prob
    # extract sample_id from suffix (after last "_")

    ann2 = ann[["cell_global_id", "cell_type", "prob"]].copy()
    ann2["coarse_cell_type"] = ann2["cell_type"].map(to_coarse_cell_type).astype("category")

    if adata.obs.index.name == "cell_global_id":
        adata.obs.index.name = "obs_index"

    adata.obs = adata.obs.merge(
        ann2, how="left", left_on="cell_global_id", right_on="cell_global_id"
    )

    return adata


def plot_celltypes_per_sample(
    adata,
    sample_id: str,
    x_key: str = "CenterX_global_px",
    y_key: str = "CenterY_global_px",
    color_col: str = "coarse_cell_type",
    s: float = 0.4,
    alpha: float = 0.9,
    invert_y: bool = True,
    figsize=(10, 10),
    dpi=300,
    savepath: str | None = None,
):
    sub = adata[adata.obs["sample_id"] == sample_id].copy()
    sub = sub[sub.obs[color_col].notna()].copy()

    sub.obs[color_col] = sub.obs[color_col].astype("category")
    cats = list(sub.obs[color_col].cat.categories)

    # diverse palette
    palette = _many_distinct_colors(len(cats), seed=0)
    cat2color = dict(zip(cats, palette))
    colors = sub.obs[color_col].astype(str).map(cat2color).to_numpy()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(
        sub.obs[x_key].to_numpy(),
        sub.obs[y_key].to_numpy(),
        s=s,
        c=colors,
        alpha=alpha,
        linewidths=0,
    )

    if invert_y:
        ax.invert_yaxis()

    ax.set_title(f"Cell types (sample {sample_id})")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_aspect("equal")

    # compact legend (can be big if many types)
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=cat2color[c], label=c)
        for c in cats
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


adata = ad.read_h5ad(r'D:\\thesis-research\\resources\\cache\\slice_1_adata.h5ad')
plot_insitu_cluster_f(adata)


# ---------- usage ----------
# 1) add annotations
# adata = ad.read_h5ad('D:\\thesis-research\\resources\\cache\\adata_full.h5ad')
# adata = add_celltype_to_adata(adata, "D:\\thesis-research\\thesis_research\\pipeline\\scripts\\outputs\\insitutype_cell_types.csv")
#
# # # 2) plot each sample
# # for sid in sorted(adata.obs["sample_id"].dropna().unique()):
# #     plot_celltypes_per_sample(adata, sid)
#
# for sid in sorted(adata.obs["sample_id"].dropna().unique()):
#     plot_celltypes_groups_per_sample(
#         adata,
#         sample_id=sid,
#         groups=CELLTYPE_GROUPS,
#         color_col="coarse_cell_type",
#         s=0.4,
#         alpha=0.9,
#         figsize=(20, 7),
#         dpi=300,
#         # savepath=f"celltypes_groups_{sid}.png"   # uncomment to save
#     )
