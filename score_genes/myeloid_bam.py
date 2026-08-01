"""BAM (border-associated macrophage) annotation WITHIN the clean Myeloid group.

Logic (markers identify, anatomy confirms — the tumor-annotation template):
  1. Substrate = high-confidence Myeloid cells (provisional==Myeloid & high_conf).
  2. Score the BAM module (Mrc1, Cd163, Pf4, Lyve1) within those myeloid cells.
  3. Call BAM with a DATA-DRIVEN cutoff: 2-component Gaussian mixture on the BAM
     score -> BAM = the high-mean component (posterior > 0.5). No arbitrary cut.
  4. Build a PERMISSIVE vessel scaffold from marker expression (not the sparse
     Vascular label): every non-tumor cell high on Pecam1/Cldn5/Flt1/Cdh5.
  5. Confirm: BAM candidates sit closer to vessels than other myeloid
     (Mann-Whitney on distance-to-nearest-vessel) — independent anatomical check.

Outputs -> score_genes_slice1_merged/bam/
  - bam_labels.csv      (myeloid cells + bam_score, is_bam, dist_to_vessel)
  - bam_annotation.png  (score split, perivascular enrichment, spatial, markers)
"""
import os

# single-threaded BLAS/OpenMP (avoid the dual-OpenMP deadlock) — before numpy
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_homogeneity as ch  # noqa: E402

PROV = "D:/thesis-research/score_genes_slice1_merged/cell_scores.csv"
HC = "D:/thesis-research/score_genes_slice1_merged/high_confidence_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1_merged/bam"

BAM_MARKERS = ["Mrc1", "Cd163", "Pf4"]               # border-macrophage module
# (Lyve1 dropped — sparse/unreliable probe; was inflating the call)
VESSEL_MARKERS = ["Pecam1", "Cldn5", "Flt1", "Cdh5"]  # permissive vessel scaffold
# contrast markers (to show BAM is distinct from the other myeloid subtypes)
MICROGLIA = ["Tmem119", "Cx3cr1", "P2ry12", "Selplg"]
MDM = ["Ccr2", "Plac8", "Cd14"]

sc.settings.verbosity = 0
plt.rcParams.update({"figure.facecolor": "white", "savefig.facecolor": "white"})


def gmm_high_component(score):
    """2-component GMM on a 1-D score; return (is_high mask, threshold)."""
    s = score.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(s)
    hi = int(np.argmax(gm.means_.ravel()))           # high-mean component
    post = gm.predict_proba(s)[:, hi]
    is_high = post > 0.5
    # threshold = lowest score still assigned to the high component
    thr = float(score[is_high].min()) if is_high.any() else float(score.max())
    return is_high, thr


def mean_expr(adata, mask, genes):
    genes = [g for g in genes if g in adata.var_names]
    if not genes or mask.sum() == 0:
        return {}
    M = adata[mask, genes].X
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    return {g: float(M[:, i].mean()) for i, g in enumerate(genes)}


def main():
    os.makedirs(OUT, exist_ok=True)
    ch.SCORES_CSV = PROV
    adata = ch.load_adata_with_calls()           # all non-tumor, normalized+log1p
    ct = np.asarray(adata.obs["celltype"])
    labeled = ct != "unknown"

    # align high_confidence_labels.csv to the labeled subset (same order as Stage 2)
    hc = pd.read_csv(HC)
    sub_obs_x = adata.obs["x"].to_numpy()[labeled]
    assert np.allclose(hc["x"].to_numpy(), sub_obs_x, atol=1e-3), "HC not aligned"
    high = hc["high_conf"].astype(str).isin(["True", "1", "TRUE", "true"]).to_numpy()
    prov = hc["provisional_label"].to_numpy()

    # clean-myeloid mask over ALL non-tumor cells
    myeloid = np.zeros(adata.n_obs, bool)
    myeloid[np.where(labeled)[0]] = (prov == "Myeloid") & high
    print(f"clean Myeloid substrate: {int(myeloid.sum()):,} cells")

    # ---- 1) BAM score within the myeloid cells ----
    mye = adata[myeloid].copy()
    bg = [g for g in BAM_MARKERS if g in mye.var_names]
    print(f"BAM markers present: {bg}")
    sc.tl.score_genes(mye, gene_list=bg, score_name="bam_score",
                      ctrl_size=50, n_bins=25)
    bam_score = mye.obs["bam_score"].to_numpy()

    # ---- 2) data-driven BAM call (GMM high component) ----
    is_bam, thr = gmm_high_component(bam_score)
    n_bam = int(is_bam.sum())
    print(f"BAM call: {n_bam:,} / {len(is_bam):,} myeloid "
          f"({100*n_bam/len(is_bam):.1f}%)  GMM threshold = {thr:.3f}")

    # canonical-marker cross-check: is the BAM call driven by the specific
    # border-macrophage markers (Cd163/Mrc1) or only by leaky Lyve1/Pf4?
    def expr_pos(genes):
        gg = [g for g in genes if g in mye.var_names]
        M = mye[:, gg].X
        M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        return (M > 0).any(1)
    canon = expr_pos(["Cd163", "Mrc1"])      # specific BAM markers
    leaky = expr_pos(["Pf4"])                 # secondary / ambient-prone
    print(f"  of BAM: Cd163/Mrc1+ = {int((is_bam & canon).sum()):,} "
          f"({100*(is_bam & canon).mean()/is_bam.mean():.0f}%); "
          f"Pf4-only = {int((is_bam & leaky & ~canon).sum()):,}")

    # ---- 3) vessel scaffold = marker-gated provisional Vascular cells ----
    # (real vessel network: ~1.8k cells that passed the FDR+margin gate; far more
    #  reliable than a leaky GMM on Pecam1/Flt1/Cdh5, which are ambient/spill-over)
    is_vessel = ct == "Vascular"
    print(f"vessel scaffold (provisional Vascular): {int(is_vessel.sum()):,} cells")

    # ---- 4) distance from each myeloid cell to nearest vessel cell ----
    xy = np.column_stack([adata.obs["x"].to_numpy(), adata.obs["y"].to_numpy()])
    vess_xy = xy[is_vessel]
    mye_xy = xy[myeloid]
    nn = NearestNeighbors(n_neighbors=1).fit(vess_xy)
    dist, _ = nn.kneighbors(mye_xy)
    dist = dist.ravel()

    # ---- 5) perivascular enrichment test ----
    d_bam, d_other = dist[is_bam], dist[~is_bam]
    u, p = mannwhitneyu(d_bam, d_other, alternative="less")
    print(f"\nperivascular enrichment (distance to nearest vessel, px):")
    print(f"  BAM        median = {np.median(d_bam):8.1f}  (n={len(d_bam)})")
    print(f"  other mye. median = {np.median(d_other):8.1f}  (n={len(d_other)})")
    print(f"  Mann-Whitney (BAM closer): p = {p:.2e}")

    # per-subset: does the perivascular signal come from canonical Cd163/Mrc1+
    # BAMs, or only from the leaky Lyve1/Pf4-only cells (= possible ambient)?
    subsets = {
        "canonical BAM (Cd163/Mrc1+)": is_bam & canon,
        "Pf4-only": is_bam & leaky & ~canon,
        "other myeloid": ~is_bam,
    }
    print("\nperivascular enrichment by subset (median dist, vs other myeloid):")
    for name, m in subsets.items():
        if m.sum() == 0:
            continue
        if name == "other myeloid":
            print(f"  {name:28s} median = {np.median(dist[m]):7.1f}  (n={int(m.sum())})")
        else:
            pp = mannwhitneyu(dist[m], dist[~is_bam], alternative="less")[1]
            print(f"  {name:28s} median = {np.median(dist[m]):7.1f}  "
                  f"(n={int(m.sum())})  p={pp:.2e}")

    # ---- marker specificity: BAM vs other myeloid ----
    print("\nmean expression  (BAM  |  other myeloid):")
    for label, genes in [("BAM", BAM_MARKERS), ("microglia", MICROGLIA),
                         ("MDM", MDM)]:
        eb = mean_expr(mye, is_bam, genes)
        eo = mean_expr(mye, ~is_bam, genes)
        print(f"  [{label}]")
        for g in eb:
            print(f"    {g:8s}: {eb[g]:5.2f}  |  {eo[g]:5.2f}")

    # ---- save labels ----
    out = pd.DataFrame({
        "x": mye.obs["x"].to_numpy(), "y": mye.obs["y"].to_numpy(),
        "bam_score": bam_score.round(4),
        "is_bam": is_bam, "dist_to_vessel": dist.round(1),
    })
    out.to_csv(f"{OUT}/bam_labels.csv", index=False)

    # ================= plots =================
    with h5py.File(ch.SLICE, "r") as h5:
        xa = ch._read_obs_num(h5, "CenterX_global_px")
        ya = ch._read_obs_num(h5, "CenterY_global_px")
        tum = ch._read_obs_bool(h5, ch.TUMOR_COL)

    fig, ax = plt.subplots(2, 2, figsize=(16, 14), dpi=150)

    # (a) BAM score distribution + threshold
    a = ax[0, 0]
    a.hist(bam_score, bins=60, color="#bbbbbb", edgecolor="none")
    a.axvline(thr, color="#d62728", lw=2, ls="--", label=f"GMM cut = {thr:.2f}")
    a.set_xlabel("BAM module score (within myeloid)")
    a.set_ylabel("cells")
    a.set_title(f"(a) BAM score split — {n_bam:,} BAM of {len(is_bam):,} myeloid",
                fontweight="bold")
    a.legend()

    # (b) perivascular enrichment (ECDF of distance-to-vessel)
    a = ax[0, 1]
    for d, c, lab in [(d_bam, "#d62728", f"BAM (n={len(d_bam)})"),
                      (d_other, "#888888", f"other myeloid (n={len(d_other)})")]:
        ds = np.sort(d)
        a.plot(ds, np.linspace(0, 1, len(ds)), color=c, lw=2, label=lab)
    a.set_xlabel("distance to nearest vessel cell (px)")
    a.set_ylabel("cumulative fraction")
    a.set_xlim(0, np.percentile(dist, 99))
    a.set_title(f"(b) Perivascular enrichment  (p = {p:.1e})", fontweight="bold")
    a.legend()

    # (c) spatial: vessels + BAM + other myeloid
    a = ax[1, 0]
    a.scatter(xa[~tum], ya[~tum], s=0.4, c="#ececec", linewidths=0, rasterized=True)
    a.scatter(xy[is_vessel, 0], xy[is_vessel, 1], s=2, c="#2ca02c", linewidths=0,
              rasterized=True, label=f"vessel scaffold ({int(is_vessel.sum())})")
    a.scatter(mye_xy[~is_bam, 0], mye_xy[~is_bam, 1], s=5, c="#7f7f7f",
              linewidths=0, rasterized=True, label=f"other myeloid ({int((~is_bam).sum())})")
    a.scatter(mye_xy[is_bam, 0], mye_xy[is_bam, 1], s=10, c="#d62728",
              linewidths=0, rasterized=True, label=f"BAM ({n_bam})")
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.legend(loc="lower right", markerscale=3, fontsize=8, frameon=True)
    a.set_title("(c) BAM vs vessels in tissue", fontweight="bold")

    # (d) marker means BAM vs other myeloid
    a = ax[1, 1]
    allg = [g for g in (BAM_MARKERS + MICROGLIA + MDM) if g in mye.var_names]
    eb = mean_expr(mye, is_bam, allg)
    eo = mean_expr(mye, ~is_bam, allg)
    yv = np.arange(len(allg)); w = 0.4
    a.barh(yv + w / 2, [eb[g] for g in allg], w, color="#d62728", label="BAM")
    a.barh(yv - w / 2, [eo[g] for g in allg], w, color="#7f7f7f",
           label="other myeloid")
    a.set_yticks(yv); a.set_yticklabels(allg)
    a.invert_yaxis()
    a.axhline(len(BAM_MARKERS) - 0.5, color="k", lw=0.8, ls=":")
    a.axhline(len(BAM_MARKERS) + len(MICROGLIA) - 0.5, color="k", lw=0.8, ls=":")
    a.set_xlabel("mean log-norm expression")
    a.set_title("(d) Markers: BAM vs other myeloid\n(BAM | microglia | MDM)",
                fontweight="bold")
    a.legend()

    fig.suptitle("Slice 1 — BAM annotation within clean Myeloid", fontsize=17,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/bam_annotation.png", bbox_inches="tight")
    plt.close()
    print(f"\nsaved bam_labels.csv ({n_bam} BAM) and bam_annotation.png -> {OUT}")


if __name__ == "__main__":
    main()
