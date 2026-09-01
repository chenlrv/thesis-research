**Table S1 | Analysis scripts underlying each figure, table, and analysis step**

| Display item | Analysis | Script | Primary output |
| --- | --- | --- | --- |
| *Chapter 2 — Quality control* |  |  |  |
| Figure 1 | Segmented-cell area and FOV layout per slide | `thesis_research/pipeline/position_plots.py` (`plot_cells_positions_with_area`, `plot_fov_positions_all_slices`), driven by `run_pipeline.py` | `cell_positions.png`, `fov_positions_sliced.png`<sup>b</sup> |
| Figure 2 | Per-cell transcript-count distributions and the low-count threshold | `thesis_plots/make_qc_count_threshold_fig.py`, applying the cutoff rule of `thesis_research/pipeline/cell_qc_plots.py` (`_low_count_flag`) | `qc_count_threshold_hist.png` |
| Table 1 | Slices profiled and cell yield after low-transcript QC | `cell_qc_plots.py` (counts); `thesis_plots/make_nature_tables.py` (layout) | `tables/thesis_tables.html`<sup>c</sup> |
| *Chapter 2 — Tumor-cell identification* |  |  |  |
| — | Reference-based annotation with SingleR | `outputs/cell_annotation/annotate.R`, `convert_annotations_to_df.R` | SingleR score CSVs |
| Figure 3 | Initial SingleR tumor candidates across six slices | `thesis_plots/figure_1_singler_calls.py`<sup>d</sup> | `figure_1_singler_calls.png/.csv` |
| Figure 4 | Classifier comparison on the joint reference pool | `thesis_plots/figure_3_model_comparison.py`<sup>d</sup> | `figure_3_model_comparison.png/.csv` |
| Figure 5 | Spatial refinement of candidates across classifiers | `thesis_plots/figure_4_spatial_refinement.py`<sup>d</sup> | `figure_4_spatial_refinement.png/.csv` |
| Figure 6 | Final XGBoost tumor calls in space, all six sections | `thesis_plots/make_dq_fig_tumor_spatial.py`; calls written by `thesis_plots/figure_4_spatial_refinement.py`<sup>d</sup> | `dq_fig_tumor_spatial.png`; `figure_4_spatial_refinement.csv` (XGBoost rows) |
| Table 3 | Parameter settings of the tumor-refinement pipeline | `…/tumor_cells/identify_tumor_cells.py`, `refine_annotation_classifiers.py` (values); `make_nature_tables.py` (layout) | `tables/thesis_tables.html`<sup>c</sup> |
| *Chapter 3 — Data-quality assessment* |  |  |  |
| Figure 7 | Probe detection against the per-slice acceptance threshold, all six slices | `thesis_plots/make_dq_fig1_detection.py` | `dq_fig1_detection.png`, `acceptance_bar_all6.csv` |
| Table 2 | Detection strength of the eight custom add-on probes, all six slices | `thesis_plots/make_detection_reliability_6slice.py` (values); `thesis_plots/make_nature_tables.py` (layout) | `detection_reliability_all6.csv`; `tables/thesis_tables.html` |
| — | Field-of-view morphology and count QC, slide L3-21 | `fov_qc_slice1.py` | `fov_qc/slice1_fov_qc.csv`, `slice1_fov_qc_map.png`, `slice1_fov_metrics.png` |
| Figure 9<sup>a</sup> | tdTomato prevalence against the pan-myeloid transcripts Cx3cr1 and Csf1r, and the control-to-tumor contrast per slide | `thesis_plots/make_dq_fig_reporter.py` | `dq_fig_reporter.png` |
| Figure 10<sup>a</sup> | Lyve1-positive cells against the canonical BAM markers Mrc1 and Cd163, in composition and in space | `thesis_plots/make_dq_fig_lyve1.py` | `dq_fig_lyve1.png` |
| — | Coarse Leiden partitions supplied to decontX as the cluster prior | `score_genes/write_decontx_clusters.py` | `resources/cache/decontx/slice_<n>_work/clusters.csv`<sup>e</sup> |
| — | Ambient-RNA estimation and correction (decontX) | `score_genes/run_decontx_correct.py` (export and assemble); `score_genes/run_decontx.R` (decontX itself)<sup>f</sup> | `resources/cache/decontx/slice_{1..6}_decontx.h5ad`, `decontx_contamination.csv` |
| Figure 11<sup>a</sup> | Probe positivity before and after ambient correction, all six sections | `thesis_plots/make_dq_fig_decontx.py` | `dq_fig_decontx.png`, `decontx_before_after.csv` |
| — | Neighbour-profile transcript reassignment and its parameter sweep | `agents/segmentation/03_filter_tx.py`, `04_reseg_reassign.py`, `05_prior_sweep.py`; purity metric in `agents/segmentation/seg_metrics.py` | per-FOV before/after marker-purity tables |
| Figure 12<sup>a</sup> | Myeloid marker purity before and after reassignment, and under sweeps of its three parameters | `thesis_plots/make_dq_fig_reassign.py`<sup>g</sup> | `dq_fig_reassign.png` |
| — | Manufacturer's technical assessment of the run | not script-generated; supplied by Bruker Spatial Biology | `resources/vendor/bruker_technical_assessment.md` |

Every figure, table, and analysis step in this thesis and the script that produced it. Script paths are relative to the repository root. Running the scripts in the order given in the repository README regenerates each output from the deposited processed data. Scripts hardcode the repository root as `D:/thesis-research` and require editing to run from another location.
<sup>a</sup>Chapter 3 figure numbers follow the current draft. Figure 8 of the earlier draft (GFP-tdTomato correlation per slice) was withdrawn, because conditioning on positivity for either probe induces a negative correlation between any two sparse targets; if the final draft carries no Figure 8, renumber Figures 9-12 accordingly.
<sup>b</sup>Written to `outputs/<run_id>/<sample_id>/`, where `run_id` is a fresh UUID generated on each pipeline run (`run_pipeline.py`), so these files have no stable path.
<sup>c</sup>Values are currently hardcoded in `make_nature_tables.py` rather than read from the analysis outputs.
<sup>d</sup>The script filename numbering predates the thesis figure numbering and does not match it.
<sup>e</sup>The partitions are a decontX input, not an output, and are not regenerated deterministically by a re-run; they are deposited with the processed data so the correction can be reproduced exactly.
<sup>f</sup>decontX is run from R (celda). The Python driver exports the count matrix and cluster prior, calls the R script, and assembles the corrected matrix; corrected counts are rounded to integers on assembly.
<sup>g</sup>Panel values are the measured results transcribed into the plotting script rather than read from the sweep outputs at render time.
BAM, border-associated macrophage; FOV, field of view; MDM, monocyte-derived macrophage; QC, quality control.
