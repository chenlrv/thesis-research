from pathlib import Path

import anndata
import anndata as ad
import squidpy as sq
from anndata import AnnData

from data_processing.clustering.clustering import cluster
from data_processing.qc import remove_outliers_and_doublets, remove_negative_probes, get_negative_probes
from data_processing.utils.adata import add_section_sample_ids

SECTIONS = [
    'GSM8199188_ID61-ID62_S10',
    'GSM8199189_ID67-ID68_S18'
]
DATA_PATH_TEMPLATE = 'D:/thesis_research_sandbox/resources/{section_id}'


def load_adata(section_id: str, section_number: int) -> AnnData:
    print(f'Loading adata for section {section_id}')

    adata = sq.read.nanostring(
        path=DATA_PATH_TEMPLATE.format(section_id=section_id),
        counts_file=f'exprMat_file.csv',
        meta_file=f'metadata_file.csv',
        fov_file=f'fov_positions_file.csv')
    adata.uns['section_id'] = section_id
    add_section_sample_ids(adata, section_number)

    print(f'Adata {section_id} initial shapes:\n'
          f'{adata}\n'
          f'{adata.var.head()}\n'
          f'Cells: {adata.n_obs}\n'
          f'Genes: {adata.n_vars}')
    return adata


def adata_qc(section_id: str, section_number: int, do_remove_negative_probes: bool = True) -> AnnData:
    adata = load_adata(section_id, section_number)
    negative_probes = get_negative_probes(adata)
    if do_remove_negative_probes:
        adata = remove_negative_probes(adata, negative_probes)
    adata = remove_outliers_and_doublets(adata)
    return adata


def run():
    use_path = True
    if use_path and Path('../resources/final_adata.h5ad').exists():
        adata = ad.read_h5ad('../resources/final_adata.h5ad')
    else:
        adatas = []
        section_number = 1
        for section in SECTIONS:
            adatas.append(adata_qc(section_id=section, section_number=section_number))
            section_number += 2  # there are 2 samples in every section

        adata = anndata.concat(
            adatas,
            label='section_id',
            keys=SECTIONS,
            index_unique='-')

    cluster(adata)
        # filtered_adata = remove_plaque_correlated_genes(final_adata)
        # return filtered_adata


run()
