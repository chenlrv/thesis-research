"""Clearer per-cluster DEG figure: top genes per Leiden cluster as horizontal
bars, each bar colored by what KIND of gene it is.

Replaces the dense scanpy dotplot. The message is meant to be readable at a
glance: the genes that define each cluster are cell-cycle / housekeeping /
stress / ambient genes -- NOT microglia/BAM/MDM markers (which would be green).

Reads cluster_de_genes.csv from run_myeloid_stage2_cluster.py (decontX by default).
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

VARIANT = "decontX"
DATA_ROOTS = {"raw": "D:/thesis-research/score_genes_myeloid_stage2",
              "decontX": "D:/thesis-research/score_genes_myeloid_stage2_decontx"}
DATA_ROOT = DATA_ROOTS[VARIANT]
SLICES = ["slice_1", "slice_3"]
SLICE_LABEL = {"slice_1": "slice_1 (tumor)", "slice_3": "slice_3 (healthy)"}
OUT = "D:/thesis-research/cluster_issue_plots"
TOPN = 8

# gene-type categories (lowercase, base name before any "/")
CATS = {
    "cell-type marker (microglia/BAM/MDM)":
        {"tmem119", "p2rx5", "mrc1", "cd163", "lyve1", "pf4", "ccr2", "plac8"},
    "pan-myeloid (not a subtype)":
        {"csf1r", "aif1", "tyrobp", "fcer1g", "c1qa", "c1qb", "c1qc", "cx3cr1",
         "ptprc", "cd68", "selplg", "cd74"},
    "cell cycle / proliferation":
        {"stmn1", "tyms", "pcna", "mki67", "myc", "top2a", "ccnb1", "ccnb2",
         "birc5", "ube2c", "cdk1", "cenpa", "h2afz", "hmgb2", "tubb5", "tuba1b"},
    "housekeeping / stress / metabolic":
        {"ppia", "eno1", "eno1b", "ldha", "uba52", "sec61g", "eif5a", "pfn1",
         "sod1", "gstp1", "gstp2", "mt1", "mt2", "actb", "gapdh", "fth1", "ftl1",
         "fau", "npm1", "ranbp1", "dusp5", "bcl2l1", "hspa8", "nr1h3", "sqstm1"},
    "ambient / non-myeloid (neuron/astro/vasc)":
        {"meg3", "ttr", "fgf13", "apod", "nrxn1", "nrxn3", "snap25", "plp1",
         "mbp", "gfap", "sparcl1", "aqp4", "cldn5", "pecam1"},
}
CAT_COLORS = {
    "cell-type marker (microglia/BAM/MDM)": "#2ca02c",
    "pan-myeloid (not a subtype)": "#1f77b4",
    "cell cycle / proliferation": "#ff7f0e",
    "housekeeping / stress / metabolic": "#9467bd",
    "ambient / non-myeloid (neuron/astro/vasc)": "#d62728",
    "other": "#bdbdbd",
}

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def categorize(gene):
    base = str(gene).lower().split("/")[0]
    for cat, members in CATS.items():
        if base in members:
            return cat
    return "other"


def plot_slice(slc):
    de = pd.read_csv(os.path.join(DATA_ROOT, slc, "cluster_de_genes.csv"))
    clusters = sorted(de["cluster"].unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(4.2 * len(clusters), 5.5),
                             squeeze=False)
    axes = axes[0]
    for ax, c in zip(axes, clusters):
        sub = (de[(de["cluster"] == c) & (de["pval_adj"] < 0.05)]
               .sort_values("log2fc", ascending=False).head(TOPN).iloc[::-1])
        colors = [CAT_COLORS[categorize(g)] for g in sub["gene"]]
        ax.barh(range(len(sub)), sub["log2fc"], color=colors, edgecolor="white")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["gene"], fontsize=11)
        ax.set_title(f"cluster {c}", fontsize=13)
        ax.set_xlabel("log2 fold-change")
    handles = [Patch(color=col, label=lab) for lab, col in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{SLICE_LABEL[slc]} [{VARIANT}]: top genes that define each cluster\n"
                 "→ none are cell-type markers (green); they are technical / ambient genes",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    f = os.path.join(OUT, f"cluster_degs_{VARIANT}_{slc}.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig)
    return f


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"  variant = {VARIANT}")
    for slc in SLICES:
        if not os.path.exists(os.path.join(DATA_ROOT, slc, "cluster_de_genes.csv")):
            print(f"  {slc}: cluster_de_genes.csv missing; skipping")
            continue
        print("  ", plot_slice(slc))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
