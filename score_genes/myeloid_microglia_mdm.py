"""Microglia vs MDM within the non-BAM clean Myeloid (slice 1).

Steps:
  1. UMAP the 4,371 non-BAM clean-myeloid cells.
  2. Leiden cluster them.
  3. score_genes Microglia (Tmem119/Selplg) and MDM (Ccr2/Plac8); call via
     FDR (0.5, permissive) + MAD-scaled margin (1.5x) gate.
  4. Homogeneity + separation gap on the called groups (check_homogeneity).

Output -> score_genes_slice1_merged/microglia_mdm/
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
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402
from run_score_genes_slice1 import (mirrored_fdr_threshold,  # noqa: E402
                                    scaled_margin_calls)

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
BAM = "D:/thesis-research/score_genes_slice1_merged/bam/bam_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/microglia_mdm"

FDR_CUTOFF = 0.05
MARGIN_RATIO = 1.5

MARKERS = {"Microglia": ["TMEM119", "Selplg"], "MDM": ["Ccr2", "Plac8"]}
AUDIT = ["TMEM119", "Selplg", "P2rx5", "Cx3cr1", "Ccr2", "Plac8", "Cd14"]
CALLCOL = {"Microglia": "#17becf", "MDM": "#00a087", "unknown": "#dddddd"}

sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def mean_expr(adata, mask, genes):
    genes = [g for g in genes if g in adata.var_names]
    M = adata[mask, genes].X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    return {g: float(M[:, i].mean()) for i, g in enumerate(genes)}


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"

    # clean myeloid (provisional Myeloid & high_conf), in adata order
    hc = pd.read_csv(HC)
    assert np.allclose(hc["x"].to_numpy(), adata.obs["x"].to_numpy()[labeled], atol=1e-3)
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    prov = hc["provisional_label"].to_numpy()
    myeloid = np.zeros(adata.n_obs, bool)
    myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
    mye_idx = np.where(myeloid)[0]

    # drop BAM (bam_labels rows align to adata[myeloid] order)
    bam = pd.read_csv(BAM)
    assert len(bam) == len(mye_idx), "bam_labels not aligned to myeloid"
    assert np.allclose(bam["x"].to_numpy(), adata.obs["x"].to_numpy()[mye_idx], atol=1e-3)
    is_bam = bam["is_bam"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    nonbam_idx = mye_idx[~is_bam]
    mask = np.zeros(adata.n_obs, bool); mask[nonbam_idx] = True
    sub = adata[mask].copy()
    print(f"non-BAM clean myeloid: {sub.n_obs:,} cells")

    # ---- 3) score_genes + FDR/margin calls ----
    for label, genes in MARKERS.items():
        g = [x for x in genes if x in sub.var_names]
        sc.tl.score_genes(sub, gene_list=g, score_name="score_" + label,
                          ctrl_size=50, n_bins=25)
        print(f"  {label}: {g}")
    S = sub.obs[["score_" + l for l in MARKERS]].copy()
    S.columns = list(MARKERS)
    thr = {l: mirrored_fdr_threshold(S[l].to_numpy(), fdr=FDR_CUTOFF)[0]
           for l in MARKERS}
    print(f"\nFDR thresholds (fdr={FDR_CUTOFF}): "
          + ", ".join(f"{l}={thr[l]:.3f}" for l in MARKERS))
    res = scaled_margin_calls(S, thr, ratio=MARGIN_RATIO)
    calls = res["calls"]
    sub.obs["celltype"] = pd.Categorical(calls)
    print("\ncalls:")
    print(pd.Series(calls).value_counts().to_string())

    # ---- 1+2) UMAP + Leiden ----
    sub.obsm["X_pca"] = ch.scaled_pca(sub)
    sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca", random_state=0)
    sc.tl.umap(sub, random_state=0)
    sc.tl.leiden(sub, resolution=1.0, flavor="igraph", n_iterations=2,
                 directed=False, random_state=0)
    um = sub.obsm["X_umap"]
    leiden = sub.obs["leiden"].to_numpy()
    print(f"\nLeiden clusters: {len(set(leiden))}")

    # leiden vs call crosstab + per-cluster mean scores
    ctab = pd.crosstab(sub.obs["leiden"], pd.Series(calls, index=sub.obs.index))
    print("\nleiden x call:")
    print(ctab.to_string())

    # ---- audit marker means per called group ----
    print("\naudit mean expression (Microglia | MDM | unknown):")
    em = {grp: mean_expr(sub, calls == grp, AUDIT) for grp in CALLCOL}
    for g in [x for x in AUDIT if x in sub.var_names]:
        print(f"  {g:8s}: {em['Microglia'].get(g,0):5.2f} | "
              f"{em['MDM'].get(g,0):5.2f} | {em['unknown'].get(g,0):5.2f}")

    # save labels
    pd.DataFrame({
        "x": sub.obs["x"].to_numpy(), "y": sub.obs["y"].to_numpy(),
        "score_Microglia": S["Microglia"].to_numpy().round(4),
        "score_MDM": S["MDM"].to_numpy().round(4),
        "leiden": leiden, "call": calls,
    }).to_csv(f"{OUT}/microglia_mdm_labels.csv", index=False)

    # ================= figure =================
    fig, ax = plt.subplots(2, 3, figsize=(20, 12), dpi=150)

    # (a) UMAP leiden
    a = ax[0, 0]
    for cl in sorted(set(leiden), key=int):
        m = leiden == cl
        a.scatter(um[m, 0], um[m, 1], s=4, linewidths=0, rasterized=True, label=cl)
    a.set_title("(a) UMAP — Leiden clusters", fontweight="bold")
    a.legend(markerscale=2, fontsize=7, ncol=2, loc="best")
    a.set_xticks([]); a.set_yticks([])

    # (b,c) UMAP score
    for j, l in enumerate(MARKERS):
        a = ax[0, 1 + j]
        v = S[l].to_numpy()
        vmax = np.quantile(np.abs(v), 0.99) or 1.0
        o = np.argsort(np.abs(v))
        sctr = a.scatter(um[o, 0], um[o, 1], s=5, c=v[o], cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
        fig.colorbar(sctr, ax=a, shrink=0.7)
        a.set_title(f"({'bc'[j]}) UMAP — {l} score", fontweight="bold")
        a.set_xticks([]); a.set_yticks([])

    # (d) UMAP calls
    a = ax[1, 0]
    for grp in ["unknown", "Microglia", "MDM"]:
        m = calls == grp
        a.scatter(um[m, 0], um[m, 1], s=5, c=CALLCOL[grp], linewidths=0,
                  rasterized=True, label=f"{grp} ({int(m.sum())})")
    a.set_title("(d) UMAP — calls", fontweight="bold")
    a.legend(markerscale=2, fontsize=9); a.set_xticks([]); a.set_yticks([])

    # (e) joint score scatter
    a = ax[1, 1]
    for grp in ["unknown", "Microglia", "MDM"]:
        m = calls == grp
        a.scatter(S["MDM"].to_numpy()[m], S["Microglia"].to_numpy()[m], s=5,
                  c=CALLCOL[grp], linewidths=0, alpha=0.6, rasterized=True, label=grp)
    a.axhline(0, c="k", lw=0.6); a.axvline(0, c="k", lw=0.6)
    a.set_xlabel("MDM score"); a.set_ylabel("Microglia score")
    a.set_title("(e) joint score distribution", fontweight="bold")
    a.legend(fontsize=9)

    # (f) differential histogram
    a = ax[1, 2]
    diff = S["Microglia"].to_numpy() - S["MDM"].to_numpy()
    a.hist(diff, bins=70, color="#999999")
    a.axvline(0, c="k", ls="--", lw=1)
    a.set_xlabel("Microglia score − MDM score")
    a.set_ylabel("cells")
    a.set_title("(f) differential axis (bimodal?)", fontweight="bold")

    fig.suptitle("Slice 1 — Microglia vs MDM within non-BAM myeloid",
                 fontsize=17, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/microglia_mdm_umap.png", bbox_inches="tight")
    plt.close()
    print(f"\nsaved microglia_mdm_umap.png, microglia_mdm_labels.csv -> {OUT}")

    # ---- 4) homogeneity + gap (only the two called groups) ----
    ch.OUT_DIR = OUT
    groups = [g for g in ["Microglia", "MDM"] if (calls == g).sum() >= ch.MIN_CELLS]
    if len(groups) == 2:
        ch.subcluster_test(sub, groups)
        ch.similarity_test(sub, groups)
        print(f"\nsaved homogeneity_centroid_similarity.png -> {OUT}")
    else:
        print(f"\nskipped homogeneity: groups with >= {ch.MIN_CELLS} cells = {groups}")


if __name__ == "__main__":
    main()
