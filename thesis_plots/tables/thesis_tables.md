**Table 1 | Tissue sections profiled by CosMx spatial transcriptomic imaging and their cell yield after low-transcript quality control**

| Slice ID | Mouse ID | Slide ID | Slice type | FOVs (n) | Tissue area (μm²) | Initial (n) | Removed (n) | Retained (n) | Retention (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | L321 | T | 257 | 17,811,397 | 140,615 | 15,659 | 124,956 | 88.86 |
| 2 | 3 | L321 | T | 225 | 16,710,072 | 137,669 | 11,184 | 126,485 | 91.88 |
| 3 | 1 | L321 | C | 138 | 8,260,389 | 69,007 | 5,010 | 63,997 | 92.74 |
| 4 | 1 | L34 | C | 198 | 9,833,831 | 76,631 | 19,545 | 57,086 | 74.49 |
| 5 | 3 | L34 | T | 205 | 23,726,062 | 221,554 | 15,344 | 206,210 | 93.07 |
| 6 | 2 | L34 | T | 182 | 29,265,570 | 280,842 | 13,469 | 267,373 | 95.20 |
| Total | — | — | — | 1,205 | 105,607,321 | 926,318 | 80,211 | 846,107 | 91.34 |

Each slice derives from one FFPE section. Tissue area is the summed area of all FOVs acquired for that slice. Initial counts are cells returned by the vendor segmentation pipeline, before filtering. Removed cells are those falling below the per-slice transcript-count threshold; retention is the retained fraction of the initial count.
Both control sections derive from a single animal (mouse 1); the four tumor-bearing sections derive from mice 2 and 3.
C, sham-injected control; FFPE, formalin-fixed paraffin-embedded; FOV, field of view; T, tumor-bearing.

**Table 2 | Detection strength of the eight custom add-on probes across all six slices, benchmarked against panel reference genes**

| Probe | Role | S/N |
| --- | --- | --- |
| *Custom add-on probes* |  |  |
| tdTomato | Lineage reporter | 12.7 [8.4-17.1] |
| Ccl2 | Chemokine | 11.2 [8.6-15.8] |
| GFAP | Positive control (astrocyte) | 10.3 [7.5-12.5] |
| TMEM119 | Microglia | 9.5 [6.4-14.2] |
| Lyve1 | BAM | 7.6 [5.7-10.3] |
| Trem2 | Myeloid | 5.5 [3.3-10.8] |
| Cxcl13 | Chemokine | 4.3 [3.1-6.1] |
| **GFP** | **Lineage reporter** | **1.6 [0.9-2.1]** |
| *Panel reference genes* |  |  |
| Meg3 | Reference (neuronal) | 76.7 [30.9-116.5] |
| Pecam1 | Reference (endothelial) | 11.1 [8.9-13.8] |
| Cx3cr1 | Reference (myeloid) | 9.8 [6.2-15.6] |
| Csf1r | Reference (myeloid) | 5.6 [4.3-7.4] |

All six slices, non-tumor cells only (*n* = 825,428 cells). Values are the mean across slices with the [min-max] range; per-slice values are shown in Figure 7a. Signal-to-noise (S/N) is the mean per-cell probe count divided by the mean per-cell count of the 11 negative-control probes, which have no target in the mouse transcriptome and whose mean defines the noise floor of each slice (0.018-0.025 counts per cell). The SystemControl targets are barcodes to which no probe is assigned and report on optical decoding alone, so they were excluded from the background estimate.
A probe was accepted as detecting its target only where its S/N exceeded three in every slice; GFP is the only probe not meeting this criterion.
Values in bold identify the GFP probe (see Results).
BAM, border-associated macrophage; S/N, signal-to-noise.

**Table 3 | Parameter settings of the three-stage tumor-refinement pipeline**

| Parameter | Value | Role |
| --- | --- | --- |
| *Stage 1 — SingleR candidate filter* |  |  |
| `score_tumor` floor | > 0.2 | Permissive tumor-candidate inclusion |
| `delta_score` margin | > 0.08 | Separation from the next-best reference label |
| Best-class consistency | `score_tumor` > next-best score | Tumor score must be the top score |
| *Stage 1 — SingleR tumor reference* |  |  |
| `score_tumor` floor | ≥ 0.4 | Stricter positive reference set |
| `delta_score` margin | > 0.08 | Confident tumor anchors |
| *Stage 2 — Classifier comparison* |  |  |
| Source slices | Slice 3 (L321), slice 4 (L34) | Control-slice false-positive tumor calls |
| Filters | As Stage 1 candidate filter | Learns the SingleR failure mode |
| Cross-validation folds | 5 | Out-of-fold model evaluation |
| Random seed | 42 | Reproducibility |
| *Stage 3 — XGBoost refinement* |  |  |
| Operating threshold | *P*(healthy) < 0.5 | Retained as refined tumor |
| Hyperparameters | xgboost 3.2.0 defaults | No tuning performed |
| Random seed | 42 | Reproducibility |

Parameters of the three-stage tumor-refinement pipeline. Stage 1 defines SingleR tumor candidates and a stricter tumor reference set; Stage 2 compares classifiers trained on the two control sections, where any tumor call is by definition a false positive; Stage 3 applies the selected XGBoost model to all sections.
Parameters set in monospace are the literal argument names used in the analysis code. `score_tumor` is the SingleR per-cell tumor score and `delta_score` its margin over the next-best reference label.
No hyperparameter tuning was performed: the Stage-3 classifier uses the xgboost 3.2.0 defaults (`n_estimators` = 100, `max_depth` = 6, `learning_rate` = 0.3, `subsample` = 1.0, `colsample_bytree` = 1.0, `reg_lambda` = 1.0, `scale_pos_weight` = 1). Defaults are stated explicitly because they are release-specific. A leave-one-out sensitivity analysis over these parameters, under the same 5-fold cross-validation, found no setting distinguishable from its default (out-of-fold accuracy 0.968–0.973 across nine configurations, against a standard error of 0.004 at *n* = 1,493 reference cells), and the refined tumor set differed by 0.25% between the tuned and default configurations, with both control sections yielding zero tumor calls throughout.

**Table 4 | Classifier performance for myeloid cell-type assignment**

| Model | Accuracy | Precision | Recall | F<sub>1</sub> | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| LogReg | 0.975 | 0.985 | 0.971 | 0.978 | 0.996 | 0.997 |
| LogReg + PCA | 0.961 | 0.973 | 0.959 | 0.966 | 0.992 | 0.995 |
| LogReg + KNN (PCA) | 0.958 | 0.968 | 0.958 | 0.963 | 0.991 | 0.994 |
| Random Forest | 0.962 | 0.977 | 0.956 | 0.967 | 0.992 | 0.994 |
| XGBoost | 0.970 | 0.985 | 0.963 | 0.974 | 0.994 | 0.996 |

Held-out test-set performance of each classifier. Metrics are reported to three decimal places; the best value in each column is not marked because differences between the top models fall within the resampling interval.
AP, average precision; AUROC, area under the receiver operating characteristic curve; F<sub>1</sub>, harmonic mean of precision and recall; KNN, k-nearest neighbors; LogReg, logistic regression; PCA, principal component analysis.
