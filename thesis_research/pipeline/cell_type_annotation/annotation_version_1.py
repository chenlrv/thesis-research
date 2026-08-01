"""
Minimal first-pass annotation for microglia, recruited macrophages, and astrocytes.

The MDM vs BAM split is deferred to a later step (Layer 3, GAT-based) because
the CosMx panel cannot separate them reliably from RNA alone.

This version is intentionally conservative:
- GFP, tdTomato, and Lyve1 are never used for scoring or labeling.
- Marker scores create high-confidence anchors only.
- Ambiguous cells stay unlabeled.
"""
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd

TARGET_CLASSES = ("microglia", "recruited_mac", "astrocyte")
WITHHELD_GENES = {"gfp", "tdtomato", "lyve1"}

WITH_TUMOR_DIR = r"D:\thesis-research\resources\cache\with_tumor_prediction"
TUMOR_COL = "pred_tumor_XGBoost"

MARKERS = {
    # Microglia: keep only markers stable under DAM / tumor-activated state.
    # Dropped: TMEM119 (downregulated in DAM), Cx3cr1 (pan-myeloid: also BAM, MDM),
    #          C1qa/b/c (pan-myeloid complement).
    "microglia": ["Hexb", "Sall1", "Siglech", "P2ry12"],
    # Recruited macrophages / MDM: monocyte-lineage-specific markers.
    # Dropped: Cd14 (upregulated on activated microglia and BAMs),
    #          S100a4 (broad: fibroblast, T cells, endothelial).
    "recruited_mac": ["Ccr2", "Ly6c2", "Plac8", "Crip1"],
    # Astrocytes: drop markers also expressed by lung adenocarcinoma cells.
    # Dropped: Sparcl1, Clu (expressed by some lung-derived tumor cells).
    "astrocyte": ["GFAP", "S100b", "Sox9", "Glul", "Apod", "Aldh1l1", "Aqp4"],
    "vascular_niche": [
        "Pecam1",
        "Cdh5",
        "Esam",
        "Vwf",
        "Eng",
        "Tek",
        "Kdr",
        "Flt1",
        "Tie1",
        "Clec14a",
        "Adgrl4",
        "Icam2",
    ],
}

# BAM: dropped Mrc1 (CD206) — upregulated on M2-polarised tumor-associated MDMs.
BAM_MARKERS = ["Cd163", "Pf4", "Marco", "Selenop", "Cd209a"]

def load_and_prepare_slice(slice_id: int) -> ad.AnnData:
    """Load a slice from with_tumor_prediction, drop tumor cells, normalize + log1p."""
    path = rf"{WITH_TUMOR_DIR}\slice_{slice_id}_adata.h5ad"
    adata = ad.read_h5ad(path)
    print(f"slice {slice_id}: loaded {adata.n_obs:,} cells from {path}")

    tumor_col = TUMOR_COL

    non_tumor = adata.obs[tumor_col].fillna(0).astype(int) == 0
    adata = adata[non_tumor].copy()
    print(f"slice {slice_id}: kept {adata.n_obs:,} non-tumor cells ({tumor_col} == 0)")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"slice {slice_id}: normalized (target_sum=1e4) and log1p")
    return adata


def score_markers(adata: ad.AnnData) -> None:
    """Score each cell type's marker list with sc.tl.score_genes.

    Adds one column per class to adata.obs: score_microglia, score_mdm, etc.
    """
    panel_lookup = {g.lower(): g for g in adata.var_names}

    for label, genes in MARKERS.items():
        present = [panel_lookup[g.lower()] for g in genes if g.lower() in panel_lookup]
        missing = [g for g in genes if g.lower() not in panel_lookup]

        if missing:
            print(f"{label}: missing from panel -> {missing}")
        print(f"{label}: scoring with {len(present)} genes -> {present}")

        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{label}", use_raw=False)


def assign_anchors(
    adata: ad.AnnData,
    percentile: float = 0.90,
    margin_min: float = 0.20,
    score_floor: float = 0.15,
) -> None:
    """Assign anchor labels per the rule:
        anchor_X if argmax score is X
                 AND score_X >= per-class q90
                 AND score_X >= 0.15  (absolute floor)
                 AND margin (top - second) >= 0.20

    Adds to adata.obs:
        anchor_label      : categorical (microglia / recruited_mac / astrocyte / unlabeled)
        anchor_best_score : the winning class score
        anchor_margin     : score_top - score_second
    """
    target_classes = TARGET_CLASSES
    all_classes = list(MARKERS.keys())
    score_cols = [f"score_{c}" for c in all_classes]

    scores = np.column_stack([adata.obs[c].to_numpy(dtype=float) for c in score_cols])
    best_idx = scores.argmax(axis=1)
    best_class = np.array(all_classes)[best_idx]
    best_score = scores[np.arange(len(scores)), best_idx]
    second_score = np.partition(scores, -2, axis=1)[:, -2]
    margin = best_score - second_score

    thresholds = {
        c: float(np.nanquantile(adata.obs[f"score_{c}"], percentile))
        for c in target_classes
    }

    labels = np.full(len(scores), "unlabeled", dtype=object)
    for c in target_classes:
        mask = (
            (best_class == c)
            & (best_score >= thresholds[c])
            & (best_score >= score_floor)
            & (margin >= margin_min)
        )
        labels[mask] = c

    adata.obs["anchor_label"] = pd.Categorical(
        labels, categories=list(target_classes) + ["unlabeled"]
    )
    adata.obs["anchor_best_score"] = best_score
    adata.obs["anchor_margin"] = margin

    print("\nAnchor assignment results:")
    for c in list(target_classes) + ["unlabeled"]:
        n = int((labels == c).sum())
        pct = 100 * n / len(labels)
        print(f"  {c:<12} : {n:>8,}  ({pct:5.1f}%)")
    print(f"\n  per-class q{int(percentile*100)} thresholds: "
          + ", ".join(f"{c}={thresholds[c]:.3f}" for c in target_classes))
    print(f"  margin_min = {margin_min}, score_floor = {score_floor}")




def score_bam_within_recruited_mac(adata: ad.AnnData) -> None:
    """Score BAM-leaning markers and report distribution within recruited_mac anchors.

    Adds adata.obs['score_bam'] for all cells (so we can compare backgrounds),
    then prints summary stats restricted to anchor_label == 'recruited_mac'.
    """
    panel_lookup = {g.lower(): g for g in adata.var_names}
    present = [panel_lookup[g.lower()] for g in BAM_MARKERS if g.lower() in panel_lookup]
    missing = [g for g in BAM_MARKERS if g.lower() not in panel_lookup]

    if missing:
        print(f"bam: missing from panel -> {missing}")
    print(f"bam: scoring with {len(present)} genes -> {present}")

    sc.tl.score_genes(adata, gene_list=present, score_name="score_bam", use_raw=False)

    mask = adata.obs["anchor_label"] == "recruited_mac"
    n = int(mask.sum())
    if n == 0:
        print("no recruited_mac anchors found")
        return

    s = adata.obs.loc[mask, "score_bam"].to_numpy(dtype=float)
    print(f"\nscore_bam within recruited_mac anchors (n={n:,}):")
    for q in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95):
        print(f"  q{int(q*100):02d}: {np.nanquantile(s, q): .3f}")
    print(f"  mean: {np.nanmean(s): .3f}   std: {np.nanstd(s): .3f}")
    print(f"  min:  {np.nanmin(s): .3f}    max: {np.nanmax(s): .3f}")


if __name__ == "__main__":
    print("Starting annotation v1")
    adata = load_and_prepare_slice(slice_id=1)
    score_markers(adata)
    assign_anchors(adata)
    score_bam_within_recruited_mac(adata)
    extract_mdm_from_recruited_mac(adata)
    print("inspection")