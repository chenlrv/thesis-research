from pathlib import Path

import pandas as pd

CELL_TYPE_PATTERN_MAPPING = {
    'VLMC': 'VLMC',
    'Peri': 'Pericyte',
    'SMC': 'SMC',
    'Endo': 'Endothelial',
    'Microglia': 'Microglia',
    'BAM': 'BAM',
    'Astro': 'Astrocyte',
    'Oligo': 'Oligodendrocyte',
    'Glut': 'Glutamatergic',
    'Gaba': 'GABAergic',
    'OPC': 'OPC',
    'Epen': 'Ependymal',
    'Macro': 'Macrophage',
    'Mono': 'Monocyte',
    'DC': 'Dendritic Cell',
    'Chor': 'Choroid Plexus',
    'COP': 'Choroid Plexus',
    'MFOL': 'Oligodendrocyte',
    'NFOL': 'Oligodendrocyte',
    'MOL': 'Oligodendrocyte',
}


def get_allen_brain_atlas_cell_type_markers(
        markers_file_path: str = Path(__file__).resolve().parent / 'allen_markers.csv'):
    """
    Download and process Allen Brain Atlas cell type markers
    """

    if markers_file_path:
        print(f'Loading data from {markers_file_path}...')
        df = pd.read_csv(markers_file_path)
    else:
        url = "https://allen-brain-cell-atlas.s3-us-west-2.amazonaws.com/metadata/WMB-taxonomy/20230830/cl.df_CCN202307220.xlsx"
        df = pd.read_excel(url)

        df.to_csv('allen_markers.csv', index=False)

    print(f'Loaded {len(df)} cell clusters from Allen Brain Atlas.')

    unknown_cell_types = _get_unknown_cell_types(df)
    _add_cell_type(df, unknown_cell_types)
    cell_type_markers = _extract_cell_type_markers(df)
    print('Extracted markers for cell types:', list(cell_type_markers.keys()))

    return cell_type_markers


def _get_unknown_cell_types(df: pd.DataFrame) -> set[str]:
    unknown_types = set()

    for idx, row in df.iterrows():
        supertype_label = row.get('supertype_label', None)
        cell_type = extract_cell_type(supertype_label)
        if cell_type == supertype_label:
            unknown_types.add(supertype_label)

    if unknown_types:
        print('Warning: Unrecognized supertype_labels found:', unknown_types)

    return unknown_types


def _add_cell_type(df: pd.DataFrame, unknown_cell_types: set[str]) -> None:
    for idx, row in df.iterrows():
        supertype_label = row.get('supertype_label', None)
        if supertype_label in unknown_cell_types:
            df.at[idx, 'cell_type'] = None
        else:
            df.at[idx, 'cell_type'] = extract_cell_type(supertype_label)


def extract_cell_type(supertype_label: str) -> str:
    """
    Extract cell type from supertype_label

    'VLMC NN_2' -> 'VLMC'
    'Microglia NN_1' -> 'Microglia'
    """

    for key in CELL_TYPE_PATTERN_MAPPING.keys():
        if key.lower() in str(supertype_label).lower():
            return CELL_TYPE_PATTERN_MAPPING[key]

    return supertype_label


def _extract_cell_type_markers(df: pd.DataFrame) -> dict[str, list]:
    cell_type_markers = {}

    for cell_type in df['cell_type'].dropna().unique():
        type_df = df[df['cell_type'] == cell_type]

        all_markers = set()
        for markers_str in type_df['cluster.markers.combo']:
            if pd.notna(markers_str):
                genes = [g.strip() for g in str(markers_str).split(',') if g.strip()]
                all_markers.update(genes)

        cell_type_markers[cell_type] = sorted(list(all_markers))

    return cell_type_markers
