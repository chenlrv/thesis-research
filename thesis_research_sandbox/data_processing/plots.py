import matplotlib.pyplot as plt
from anndata import AnnData

from data_processing.utils.adata import get_adata_section_id


def plot_genes_transcripts(adata: AnnData, counts_max: float, counts_min: float, genes_max: float,
                           genes_min: float) -> None:
    section = get_adata_section_id(adata)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(adata.obs["n_counts"], bins=100)
    ax[0].axvline(counts_min, ls="--")
    ax[0].axvline(counts_max, ls="--")
    ax[0].set_title(f"{section} — n_counts")
    ax[1].hist(adata.obs["n_genes"], bins=100, color="orange")
    ax[1].axvline(genes_min, ls="--")
    ax[1].axvline(genes_max, ls="--")
    ax[1].set_title(f"{section} — n_genes")
    plt.show()


def plot_doublet_analysis(adata: AnnData, doublet_scores: list[float]) -> None:
    section = get_adata_section_id(adata)

    plt.figure(figsize=(5, 3))
    plt.hist(doublet_scores, bins=100, color="steelblue")
    plt.title(f"{section} Scrublet doublet scores")
    plt.xlabel("Doublet score")
    plt.ylabel("Cell count")
    plt.show()
