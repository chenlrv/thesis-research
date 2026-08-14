# Single-Cell Spatial Transcriptomics and Graph-Based Modeling in Brain Metastasis

Analysis code for the MSc thesis of Chen Arviv (Tel Aviv University, Shmunis School of
Biomedicine and Cancer Research / Sagol School of Neuroscience), under the joint
supervision of Prof. Tal Pupko and Prof. Reuven Stein.

The study profiles a D122 lung-carcinoma brain-metastasis model in
*Ms4a3<sup>cre</sup>:R26<sup>tdT</sup>:Cx3cr1<sup>Gfp</sup>* reporter mice on the
NanoString CosMx platform: six sagittal slices across two slides, 926,318 segmented
cells, 958 genes including eight custom add-on probes.

**Every figure, table, and analysis step in the thesis is mapped to the script that
produced it in [Supplementary Table S1](thesis_plots/tables/thesis_table_s1.md).**
That table is the authoritative map; this README gives the order in which the scripts
run and what each stage needs.

---

## 1. Environment

Analyses are implemented in Python and R.

```bash
conda create -n thesis_research python=3.11
conda activate thesis_research
pip install -r requirements.txt
```

> **Caveat.** `requirements.txt` pins no versions — every entry is a `>=` range — so this
> does not reproduce the exact environment the results were generated in. See
> "Known gaps".

All Python entry points are then run through that environment:

```bash
conda run -n thesis_research python <script>
```

R scripts (SingleR, decontX) require R 4.5 with `SingleR` 2.12.0, `SingleCellExperiment`,
`scater`, `scuttle`, `Seurat`, `celldex`, and `celda` (for decontX).

Fixed random seeds are set inside the scripts that use stochastic methods; the
classifier comparison uses seed 42 (Table 3 of the thesis).

## 2. Data layout

Raw CosMx exports and derived caches are **not** in this repository. Place them as:

```
resources/
  cosmx/<sample_id>/            # vendor export per slide (L321, L34)
    <sample_id>_exprMat_file.csv
    <sample_id>_metadata_file.csv
    <sample_id>_fov_positions_file.csv
    <sample_id>_fov_slices.csv
    <sample_id>_slice_types.csv
  cache/                        # per-slice AnnData written by the pipeline
    slice_{1..6}*.h5ad
    decontx/slice_{1..6}_decontx.h5ad
outputs/
  cell_annotation/              # SingleR references and score tables
    D122_Reference_Avinoam/     # GSE103548 LLC1 bulk reference
```

Paths are resolved from `thesis_research/utils/constants.py`, which derives everything
from the repository root. Some standalone plotting scripts still contain absolute
`D:/thesis-research/...` paths — see "Known gaps".

## 3. Running the analysis

Stages are ordered; each depends on the outputs of the ones before it.

### Stage 0 — Ingest, QC, and slice assignment

```bash
conda run -n thesis_research python -c "from thesis_research.pipeline.run_pipeline import run_pipeline; run_pipeline(run_exploration=True, run_qc=True)"
```

Reads the vendor export, splits each slide into its three sagittal slices by FOV
coordinates, writes per-slice AnnData to `resources/cache/`, and applies the
low-transcript filter — per slice, `threshold = max(20, P5)` of the per-cell total-count
distribution — in `thesis_research/pipeline/cell_qc_plots.py`
(`run_cell_qc`, `_low_count_flag`).

Produces **Figure 1** (`position_plots.py`) and **Figure 2** (`cell_qc_plots.py`),
and the counts behind **Table 1**.

### Stage 1 — Reference-based annotation (SingleR)

```bash
Rscript outputs/cell_annotation/annotate.R
Rscript outputs/cell_annotation/convert_annotations_to_df.R
```

Builds the three-part reference — brain structural, brain immune, and an LLC1
carcinoma reference from GEO accession GSE103548 — and returns per-cell
`score_brain_struct`, `score_brain_immune`, `score_tumor`, plus the predicted label.
Tumor candidates are the cells passing the three filters in Table 3 of the thesis.

### Stage 2 — Supervised refinement of tumor candidates

```bash
conda run -n thesis_research python -m thesis_research.pipeline.cell_type_annotation.tumor_cells.refine_annotation_classifiers
```

Trains five classifiers (logistic regression, LogReg+PCA, LogReg+KNN on PCA, random
forest, XGBoost) on a reference pool whose negative class is drawn from tumor calls in
the sham-injected control slices. Reference-class construction lives in
`identify_tumor_cells.py` (`_get_tumor_ref_ids`, `_get_healthy_ref_ids`,
`_get_tumor_candidates_ids`).

### Stage 3 — Final tumor calls

XGBoost is applied to all candidates across the six slices; cells with
*P*(healthy) < 0.5 are retained as refined tumor. Implemented in
`identify_tumor_cells_revised.py`, which has **no command-line entry point** — its
functions are called from a notebook or REPL. The same calls are reproduced
end-to-end by `thesis_plots/figure_4_spatial_refinement.py` (Stage 5), which retrains
the classifiers and writes the per-slice retained/rejected counts to CSV; use that
script if you want a single runnable command for this stage.

### Stage 4 — Data-quality assessment

```bash
Rscript score_genes/run_decontx.R                                          # ambient-RNA estimation
conda run -n thesis_research python thesis_plots/make_detection_reliability_6slice.py
conda run -n thesis_research python thesis_plots/make_detection_barplot_6slice.py
conda run -n thesis_research python thesis_plots/make_reporter_6slice.py
conda run -n thesis_research python thesis_plots/lyve1_unreliability.py
conda run -n thesis_research python thesis_plots/lyve1_unreliability_spatial.py
conda run -n thesis_research python thesis_plots/make_decontx_fig.py
```

Quantifies each probe against the noise floor defined by the 11 negative-control
probes, tests the lineage reporters against their expected biology, and rules out
ambient RNA as the explanation.

### Stage 5 — Thesis figures and tables

```bash
conda run -n thesis_research python thesis_plots/figure_1_singler_calls.py       # thesis Figure 3
conda run -n thesis_research python thesis_plots/figure_2_reference_cells.py     # not currently used
conda run -n thesis_research python thesis_plots/figure_3_model_comparison.py    # thesis Figure 4
conda run -n thesis_research python thesis_plots/figure_4_spatial_refinement.py  # thesis Figure 5
conda run -n thesis_research python thesis_plots/make_nature_tables.py           # Tables 1-4 and S1
```

> **Note on numbering.** The `figure_N_*.py` filenames predate the thesis figure
> numbering and are offset from it. The mapping above and in Supplementary Table S1
> is authoritative; the filenames are not.

## 4. Repository layout

| Path | Contents |
|---|---|
| `thesis_research/pipeline/` | Ingest, QC, FOV/position plots, clustering, filters |
| `thesis_research/pipeline/cell_type_annotation/tumor_cells/` | SingleR filtering, classifier comparison, final tumor calls |
| `thesis_research/pipeline/cell_type_annotation/myeloid/` | **Superseded** reporter-*gated* myeloid typing (GFP/tdTomato → MDM vs resident) |
| `thesis_research/utils/` | Path constants, column names, entity types |
| `thesis_plots/` | Scripts generating every thesis figure and table |
| `thesis_plots/tables/` | Rendered Nature-style tables (HTML for Word, Markdown for drafts) |
| `outputs/cell_annotation/` | SingleR reference construction and score conversion (R) |
| `score_genes/` | Reporter-independent cell-type annotation: `score_genes` scoring with mirrored-FDR gating (`run_score_genes_*.py`), myeloid subtype gating (`myeloid_subtype_gate.py`, `myeloid_stage2*.py`), backbone classification (`classify_backbone.py`), final labels (`final_annotation.py`), plus decontX (`run_decontx.R`) and two-tier SingleR validation (`singler_two_tier.R`) |
| `agents/outputs/` | Intermediate validation reports and metrics |

Scripts prefixed with `_` at the repository root are exploratory one-offs kept for
provenance. **They are not part of the reproducible pipeline** and no thesis result
depends on them.

## 5. Known gaps

These are open items, listed so that the state of the repository is not overstated.
Each is also flagged in Supplementary Table S1.

- **No `environment.yml`.** `requirements.txt` exists but pins nothing — every entry is
  a `>=` range, so it does not reproduce an environment. Needs `pip freeze` output or a
  conda lock file, plus an R `renv.lock` or committed `sessionInfo()`.
- **Pipeline outputs have no stable path.** `run_pipeline.py` generates a fresh
  `uuid4()` as `run_id` on every run, so Stage 0 figures land in a new
  `outputs/<run_id>/<sample_id>/` directory each time and cannot be cited or diffed.
  The pipeline should accept a fixed run identifier.
- **Figure 6** has no rendering script. The calls themselves are computed by
  `figure_4_spatial_refinement.py` and written to its CSV; the figure is the XGBoost
  column of Figure 5 replotted by hand as a 2×3 grid. Needs a small variant of that
  script that filters to XGBoost and re-lays out the panels.
- **Figure 2** — `cell_qc_plots.py` writes one histogram per slice; the combined
  six-panel figure used in the thesis is not assembled by a script.
- **Tables 1 and 3** have their values hardcoded in `make_nature_tables.py` rather than
  read from the analysis outputs.
- **`detection_barplot_all6.png` is stale** — the committed PNG has two panels while the
  current script draws three; re-running it restores panel (c).
- **Absolute paths.** Several `thesis_plots/` scripts hardcode `D:/thesis-research/`
  instead of resolving from `constants.py`, so they will not run elsewhere unchanged.
- **Figure numbering** in `figure_N_*.py` filenames does not match the thesis.
- **`myeloid/myeloid_typing.py` contradicts the thesis.** Its stage 1 gates on
  GFP/tdTomato, but the thesis concludes the reporters are unusable and states that
  annotation was reporter-independent. The reporter-independent framework used for the
  thesis lives in `score_genes/`. This module is kept for provenance and should be
  marked superseded in-file, or removed, so the two do not appear to disagree.
- **`score_genes/` is undocumented.** It holds the reporter-independent annotation
  framework in ~120 loosely-named scripts with no entry point or ordering. The
  cell-annotation chapter is not yet written, so no thesis display item depends on it
  today; once that chapter exists, those scripts need the same S1 treatment as the rest.

## 6. Verifying a run

After Stage 0 the headline numbers should match Table 1 of the thesis exactly:

| Quantity | Expected |
|---|---|
| Segmented cells before QC | 926,318 |
| Removed by low-transcript filter | 80,211 |
| Retained | 846,107 (91.34%) |
| FOVs | 1,205 |
| Per-slice retention | 88.86 / 91.88 / 92.74 / 74.49 / 93.07 / 95.20 % |

After Stage 2, out-of-fold ROC-AUC should exceed 0.99 for all five classifiers, and
XGBoost should reach accuracy 0.972, precision 0.952, recall 0.983, F1 0.967.

After Stage 3, both control slices (3 and 4) should retain **zero** tumor cells out of
343 and 287 candidates respectively — the sharpest single check that the pipeline
behaves as designed.

## 7. Data and code availability

Processed per-slice single-cell data and the assembled SingleR reference objects are
available from the author on request. Raw CosMx exports (transcript tables and
segmentation masks) are large and archived separately.

All animal procedures, tumor implantation, tissue processing, and histological
preparation were performed by Mr. Avinoam Ratzabi, Tel Aviv University, in accordance
with institutional and ethical guidelines.
