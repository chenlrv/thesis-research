"""Homogeneity / separation figure for the confident Microglia vs MDM calls.
  (a) per-cell separation gap distribution (own - other centroid cosine)
  (b) intra vs nearest-other centroid cosine, per group
  (c) centroid cosine heatmap
Output -> score_genes_slice1_merged/microglia_mdm/homogeneity_summary.png
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
BAM = "D:/thesis-research/score_genes_slice1_merged/bam/bam_labels.csv"
LAB = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm/microglia_mdm_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm"
COL = {"Microglia": "#17becf", "MDM": "#ff7f0e"}

plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def main():
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"

    hc = pd.read_csv(HC)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    prov = hc["provisional_label"].to_numpy()
    myeloid = np.zeros(adata.n_obs, bool)
    myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
    mye_idx = np.where(myeloid)[0]

    bam = pd.read_csv(BAM)
    is_bam = bam["is_bam"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    nonbam_idx = mye_idx[~is_bam]
    mask = np.zeros(adata.n_obs, bool); mask[nonbam_idx] = True
    sub = adata[mask].copy()

    lab = pd.read_csv(LAB)
    assert np.allclose(lab["x"].to_numpy(), sub.obs["x"].to_numpy(), atol=1e-3)
    call = lab["call"].to_numpy()

    emb = ch.scaled_pca(sub)
    groups = ["Microglia", "MDM"]
    cent = {g: emb[call == g].mean(0) for g in groups}
    C = np.vstack([cent[g] for g in groups])

    # per-cell own/other cosine + gap (only confident cells)
    stats = {}
    for i, g in enumerate(groups):
        m = call == g
        S = cosine_similarity(emb[m], C)            # cells x 2
        own = S[:, i]
        other = S[:, 1 - i]
        stats[g] = {"own": own, "other": other, "gap": own - other,
                    "intra": float(own.mean()), "near": float(other.mean())}
        print(f"{g}: n={int(m.sum())}  intra={stats[g]['intra']:.3f}  "
              f"nearest_other={stats[g]['near']:.3f}  gap={stats[g]['gap'].mean():+.3f}  "
              f"%gap>0={100*(stats[g]['gap']>0).mean():.1f}%")

    cc = cosine_similarity(C)

    fig, ax = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

    # (a) per-cell gap distribution
    a = ax[0]
    for g in groups:
        a.hist(stats[g]["gap"], bins=50, alpha=0.6, color=COL[g],
               label=f"{g} (mean {stats[g]['gap'].mean():+.2f}, "
                     f"{100*(stats[g]['gap']>0).mean():.0f}% > 0)")
    a.axvline(0, c="k", ls="--", lw=1)
    a.set_xlabel("separation gap  (own − other centroid cosine)")
    a.set_ylabel("cells")
    a.set_title("(a) per-cell separation gap", fontweight="bold")
    a.legend(fontsize=9)

    # (b) intra vs nearest-other cosine bars
    a = ax[1]
    xp = np.arange(len(groups)); w = 0.36
    a.bar(xp - w/2, [stats[g]["intra"] for g in groups], w, color="#4c72b0",
          label="intra (own centroid)")
    a.bar(xp + w/2, [stats[g]["near"] for g in groups], w, color="#c44e52",
          label="nearest other")
    a.axhline(0, c="k", lw=1)
    for i, g in enumerate(groups):
        a.text(i, max(stats[g]["intra"], 0) + 0.01,
               f"gap {stats[g]['intra']-stats[g]['near']:+.2f}", ha="center",
               fontsize=10, fontweight="bold")
    a.set_xticks(xp); a.set_xticklabels(groups)
    a.set_ylabel("cosine to centroid")
    a.set_title("(b) intra vs nearest-other cosine", fontweight="bold")
    a.legend(fontsize=9)

    # (c) centroid cosine heatmap
    a = ax[2]
    im = a.imshow(cc, cmap="RdBu_r", vmin=-1, vmax=1)
    a.set_xticks(range(len(groups))); a.set_xticklabels(groups)
    a.set_yticks(range(len(groups))); a.set_yticklabels(groups)
    for p in range(len(groups)):
        for q in range(len(groups)):
            a.text(q, p, f"{cc[p, q]:.2f}", ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=a, shrink=0.7, label="centroid cosine")
    a.set_title("(c) centroid similarity", fontweight="bold")

    fig.suptitle("Slice 1 — Microglia vs MDM homogeneity / separation (confident calls)",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"{OUT}/homogeneity_summary.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("saved", out)


if __name__ == "__main__":
    main()
