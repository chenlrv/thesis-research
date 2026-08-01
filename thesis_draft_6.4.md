# Single-Cell Spatial Transcriptomics and Graph-Based Modeling in Brain Metastasis

**Thesis submitted toward the degree of Master of Science in Computational Neuroscience, Tel-Aviv University**

by **Chen Arviv**

George S. Wise Faculty of Life Sciences — School of Neurobiology, Biochemistry & Biophysics

Joint supervision of Prof. Tal Pupko & Prof. Reuven Stein

October 2026

---

## 1 Introduction

Brain metastases are a frequent and devastating complication of advanced cancer and represent one of the most common intracranial malignancies in adults. Among solid tumors, lung cancer is a leading primary source of brain metastatic disease, characterized by complex interactions between metastatic tumor cells and the brain microenvironment [1]. The presence of these lesions is associated with poor prognosis and major therapeutic challenges, arising from both the biological aggressiveness of metastatic cells and the unique properties of the neural niche. Unlike primary brain tumors, metastatic lesions must navigate a complex multistep process—dissemination, survival in circulation, crossing the blood-brain barrier, and colonization—to adapt to a highly specialized environment. This adaptation is not driven by tumor cells alone, but rather emerges through continuous and dynamic interactions between malignant cells and the surrounding stromal, vascular, and immune compartments.

Among the most abundant and functionally dynamic cell populations in the brain are macrophages and related myeloid cells. These populations include resident macrophages of the central nervous system (CNS), namely microglia and border-associated macrophages (BAMs), as well as monocyte-derived macrophages (MDMs) recruited from the circulation under pathological conditions. Collectively, these cells play critical roles in immune surveillance, inflammatory signaling, tissue remodeling, and tumor progression [2]. In the context of brain metastasis, macrophages are exposed to signals derived from tumor cells, damaged tissue, vascular structures, and other components of the tumor microenvironment (TME), and may therefore adopt diverse and dynamic phenotypic states. Rather than representing a uniform population, macrophages in metastatic tissues are increasingly understood as heterogeneous cells whose transcriptional programs and functional roles are shaped by both developmental origin and local microenvironmental cues.

Microglia reside within the brain parenchyma, whereas BAMs localize to border regions such as the meninges, perivascular spaces, and choroid plexus. In the meninges, BAMs are found in the dura mater as well as in the leptomeningeal compartments, which include the pia and arachnoid mater. In contrast, MDMs originate from circulating monocytes that infiltrate the brain during pathological conditions, including tumor growth. Distinguishing these populations is challenging, particularly under pathological conditions, as microglia can acquire transcriptional features resembling infiltrating macrophages [3]. This challenge is particularly relevant in tumors, where inflammatory signaling, tissue remodeling, and continuous interaction with malignant cells may blur the boundaries between resident and infiltrating myeloid states. As a result, resolving macrophage populations in brain metastasis requires approaches that can capture both transcriptional identity and microenvironmental context [4].

The challenge of distinguishing microglia, BAMs, and MDMs is not merely taxonomic. It has direct biological implications for how the TME is interpreted. Resident microglia may respond to early metastatic seeding differently from infiltrating macrophages, and border-associated populations may play unique roles at tissue interfaces, vascular niches, or meningeal boundaries [5]. Similarly, monocyte-derived cells may introduce immune programs not normally present in the healthy brain, potentially altering inflammatory tone, antigen presentation, and tumor-associated remodeling. This complexity has motivated a growing emphasis on high-dimensional and spatially resolved approaches that can move beyond limited marker panels and instead capture both cell state and tissue context simultaneously.

Recent advances in single-cell spatial transcriptomics enable simultaneous quantification of gene expression and precise spatial localization, allowing direct analysis of cellular interactions within preserved tissue architecture. Unlike conventional bulk sequencing, which averages signals across heterogeneous populations, or dissociative single-cell RNA sequencing, which sacrifices tissue architecture, spatial transcriptomics makes it possible to study molecular identity in direct relation to anatomical position. This is particularly important in tumors, where spatial gradients, cell-cell contact, vascular access, and local immune composition can strongly influence cellular behavior. In brain metastasis, spatial context may determine the tumorigenicity of tumor cells and their interactions with TME populations. The spatial positioning and transcriptional states of tumor cells relative to surrounding immune and stromal cells can critically influence tumor growth, invasion, and immune modulation. Consequently, resolving tumor cells and surrounding immune populations at single-cell spatial resolution is essential for understanding how localized cell-cell interactions shape tumor growth and immune responses within the brain [6]. Thus, tissue architecture is not simply background information; it is an active component of disease biology [7].

Spatially resolved transcriptomics is especially valuable for studying the brain microenvironment because of the anatomical and functional compartmentalization of the CNS. Distinct niches such as the parenchyma, vasculature, meninges, and tumor border may host different constellations of immune and stromal cells, each associated with specific signaling interactions. In this setting, transcriptionally similar cells may behave differently depending on where they are located, while cells occupying the same region may diverge functionally according to their lineage or interaction partners. Resolving these relationships requires approaches that integrate molecular identity with local structure. For metastatic tumors in the brain, such approaches can help reveal how immune populations are organized around tumor nests, whether particular macrophage states accumulate in defined regions, and how tumor cells and host cells influence one another across short spatial scales. These questions are difficult to answer using expression profiles alone and underscore the importance of methods that explicitly preserve and analyze spatial information.

At the same time, the growing scale and complexity of spatial single-cell datasets create substantial computational challenges. Cellular identities in tumor tissues often exist along continua rather than as strictly discrete categories, and disease-associated states may not be fully captured by reference-based annotation alone. In addition, technical noise, incomplete gene expression measurements, and the partial overlap between related immune populations can further complicate biological interpretation. These challenges motivate the development of computational frameworks that go beyond conventional annotation pipelines and aim to uncover latent biological structure in the data. In particular, classical machine learning (ML), representation-learning approaches, and graph-based methods provide a principled framework for integrating high-dimensional transcriptional information with spatial organization to resolve cellular heterogeneity.

More broadly, the use of ML and deep learning (DL) methods in this setting is motivated not by methodological novelty alone, but by a biological need: to better resolve complex and potentially overlapping cell states within a spatially organized tumor microenvironment. Computational approaches can support more robust identification of cellular subpopulations, highlight relationships between local neighborhood structure and transcriptional programs, and provide complementary perspectives alongside reference-based annotation. In particular, graph-based, unsupervised, and representation-learning approaches may help reveal macrophage substructures or transitional states that are difficult to detect using standard clustering on expression data alone. When applied carefully, such methods can therefore serve as a bridge between high-dimensional spatial measurements and biologically interpretable hypotheses about cell-state organization in brain metastasis.

### Research Objectives and Significance

In this study, brain metastasis is approached as a spatially organized ecosystem in which tumor progression emerges from the interplay between malignant cells and a heterogeneous host microenvironment. Although particular emphasis is placed on myeloid lineages — specifically resident macrophages (microglia and BAMs) and MDMs — this work aims to resolve the broader cellular landscape of the metastatic niche and to determine how tumor presence reshapes cellular states and spatial organization. Beyond its biological objectives, the study also addresses a broader computational challenge: how to model and resolve heterogeneous, spatially structured, and partially overlapping cell states in complex metastatic tissue. By combining single-cell spatial transcriptomics with graph-based and machine-learning methods, this work aims not only to characterize the spatial and transcriptional organization of tumor and immune populations, but also to evaluate computational strategies for uncovering latent biological structure beyond conventional annotation pipelines. In this sense, the thesis contributes both to the biological understanding of brain metastasis and to the computational analysis of high-dimensional, spatially resolved tissue systems.

## Material and Methods

### Lung Carcinoma Cell Line

The D122 murine lung carcinoma cell line was selected for the current study due to its well-established ability to form brain metastases in mice and its extensive use as a preclinical mouse model of lung cancer brain metastasis [8].

### Mice

Ms4a3^cre^:R26^tdT^:Cx3cr1^Gfp^ transgenic mice were used in this study. This transgenic mouse model enables lineage-based discrimination of myeloid populations. Cx3cr1-driven GFP expression labels resident brain myeloid cells, including microglia and BAMs. The Ms4a3^cre^ lineage-tracing system drives Cre expression in monocyte-derived cells originating from the peripheral circulation [9], while the R26 reporter induces tdTomato expression following Cre-mediated recombination. This system enables identification and discrimination between resident macrophages (GFP⁺ tdTomato⁻) and monocyte-derived macrophages (GFP⁺ tdTomato⁺). This distinction is particularly important under pathological conditions, where microglia can transcriptionally resemble infiltrating macrophages. Lineage labeling provides a stable genetic marker that is not altered by changes in activation state.

### Brain Metastasis Model

To model brain metastasis, 10,000 D122 murine lung carcinoma cells were injected into the right hemisphere of Ms4a3^cre^:R26^tdT^:Cx3cr1^Gfp^ transgenic mice (n = 6) using stereotactic injection, a technique that enables precise three-dimensional targeting within the brain. Sham-injected controls (n = 3) underwent the same surgical procedure but received vehicle (RPMI medium) only. Lewis lung carcinoma brain metastasis models, including stereotactic intracranial implantation, have been used to study tumor vasculature and immune interactions within the CNS microenvironment [10]. In particular, prior work from Prof. Stein's laboratory demonstrated distinct spatial patterns of macrophage–tumor interactions in this model [11], providing a strong foundation for the spatially resolved single-cell analysis performed in the present study.

### Tissue Processing and Section Selection

Three weeks post-intracranial injection, mice were sacrificed, and brains were harvested, fixed in 4% paraformaldehyde, paraffin-embedded, and processed into formalin-fixed paraffin-embedded (FFPE) blocks. Each block contained the left hemisphere from three different mice.

Following embedding, sagittal sections (5 µm) were cut from selected FFPE blocks using a microtome and mounted on glass slides. Sagittal orientation was chosen because it captures extensive anatomical coverage, allowing simultaneous analysis of multiple brain regions within a single section.

A subset of sections was subjected to hematoxylin and eosin (H&E) staining to identify tumor presence and regions exhibiting high tumor heterogeneity. Based on this histological assessment, consecutive unstained sections adjacent to the evaluated H&E-stained sections were selected for spatial transcriptomic profiling in order to preserve RNA integrity. Although tumor cells were injected into the right hemisphere, spatial transcriptomic analysis was performed on sections derived from the left hemisphere, where tumor expansion gives rise to smaller and more distinctive tumors but was less dominant. This enabled clearer assessment of TME interactions without the extensive tumor burden present at the injection site.

Slide identifiers (L321, L34) were defined by hemisphere, block and section index: "L" denotes the left hemisphere, "3" indicates the third block chosen from the available blocks, and the accompanying number indicates the sequential section cut from the FFPE block. The analyzed slides, 21 and 4, therefore, represent the 21st and the 4th sagittal sections obtained at distinct cutting depths within the same left hemisphere block.

### Experimental Design

A total of six sagittal brain slices were spatially profiled across two slides (L321 and L34), with three slices per slide, each derived from a different mouse (three distinct mice per slide). Within each slide, a 2:1 ratio of tumor-bearing to sham-injected control tissue was maintained. This design enabled assessment of inter-tumoral variability while preserving a healthy comparator processed within the same experimental batch.

### CosMx Spatial Transcriptomic Profiling

RNA integrity was validated prior to downstream processing. Spatial transcriptomic profiling was performed at the Azrieli Technion Genomics Center using the NanoString CosMx Spatial Molecular Imager platform according to the manufacturer's protocol.

Fields of view (FOVs; 510 µm × 510 µm each) were selected from representative regions based on prior histological evaluation of H&E-stained slices. FOV placement was guided by tumor localization, morphological heterogeneity, and broad anatomical coverage to ensure sampling of distinct spatial microenvironments. In total, 620 FOVs were chosen in slide L321, and 585 FOVs were chosen in slide L34. Within slide L321, 482 FOVs were positioned over the tumor-bearing slices (both in tumor and non-tumor regions) and 138 FOVs over sham-injected control tissue. Within slide L34, 387 FOVs were positioned over the tumor-bearing slices and 198 FOVs over sham-injected control tissue.

Gene expression was measured using the Mouse Universal Panel supplemented (without their 50 add-on genes, but with eight custom made probes for the following genes: Ccl2, Cxcl13, GFAP, Lyve1, TMEM119, Trem2, GFP, and tdTomato). Thus, the spatial expression was measured for 958 genes. The additional probes were selected to improve resolution of tumor-immune interactions and myeloid lineage identity. TMEM119 and Trem2 were included to identify microglia, Lyve1 was incorporated to identify the BAMs, localized to the leptomeninges and perivascular spaces. Ccl2 and Cxcl13 were added to assess chemokine-mediated immune recruitment and inflammatory signaling within the tumor microenvironment. Ccl2 is a key regulator of monocyte recruitment and plays a central role in attracting peripheral monocytes that can differentiate into monocyte-derived macrophages (MDMs) within tumor tissue. Cxcl13 is involved in lymphocyte recruitment and organization of immune niches and has been implicated in shaping immune cell positioning within inflammatory and tumor contexts. GFAP was included to identify astrocytes. Finally, GFP and tdTomato probes were incorporated to help discriminate between myeloid cells (that express only GFP) versus monocyte-derived macrophage (that express both GFP and tdTomato) in the Ms4a3^cre^:R26^tdT^:Cx3cr1^Gfp^ mice.

### Cell segmentation

Cell segmentation, the process of identifying individual cells and defining their boundaries within tissue images, was performed using the CosMx human RNA FFPE segmentation panel. Prior to quality control, a total of 926,318 cells were segmented across six sagittal slices. Cell segmentation was performed using the NanoString CosMx Spatial Molecular Imager (SMI) automated segmentation pipeline, which is based on an enhanced implementation of the CellPose algorithm [12, 13]. This approach integrates multimodal information from nuclear staining, membrane-associated markers, and RNA signals to delineate cellular boundaries with high accuracy. For each segmented cell, the dataset includes transcript counts, centroid spatial coordinates, and quantitative morphological features derived from the segmentation masks, including cell area, perimeter, and shape-related metrics such as circularity and eccentricity.

The segmented cells were distributed across the two analyzed slides as follows. Slide L321 comprised three slices containing 140,615 (slice 1), 137,669 (slice 2), and 69,007 (slice 3) cells (347,291 cells in total). Slide L34 comprised three slices containing 76,631 (slice 4), 221,554 (slice 5), and 280,842 (slice 6) cells (579,027 cells in total).


| Slide ID | Cell Count |
| -------- | ---------- |
| L321     | 347,291    |
| L34      | 579,027    |


**Total cells: 926,318**


| Slice ID | Mouse ID | Slide ID | Slice Type  | Cell Count | No. of FOVs | Area (µm²) |
| -------- | -------- | -------- | ----------- | ---------- | ----------- | ---------- |
| 1        | 2        | L321     | Tumor (T)   | 140,615    | 257         | 17,811,397 |
| 2        | 3        | L321     | Tumor (T)   | 137,669    | 225         | 16,710,072 |
| 3        | 1        | L321     | Control (C) | 69,007     | 138         | 8,260,389  |
| 4        | 1        | L34      | Control (C) | 76,631     | 198         | 9,833,831  |
| 5        | 3        | L34      | Tumor (T)   | 221,554    | 205         | 23,726,062 |
| 6        | 2        | L34      | Tumor (T)   | 280,842    | 182         | 29,265,570 |


Each cell was associated with gene expression measurements and precise spatial coordinates. This dataset enabled downstream computational analysis of: (1) Cell-type annotation; (2) Spatial organization of macrophage subpopulations; (3) Tumor-immune spatial interactions; (4) Graph-based modeling of cell neighborhoods.

All animal procedures, tumor implantation, tissue processing, and histological preparation were performed by Mr. Avinoam Ratzabi, Tel Aviv University, in accordance with institutional and ethical guidelines.

## Computational Analysis

**Figure 1.** Spatial distribution of segmented cells and field-of-view (FOV) organization across slides L321 and L34. Top panels show the spatial distribution of segmented cell area across slides L321 and L34, with color indicating cell area after restricting values to the 1st–99th percentile range. Bottom panels show the arrangement of FOVs across each slide, colored by slice identity, illustrating how the segmented tissue was partitioned into individual sagittal sections for downstream analysis.

**Figure 2.** Field-of-view (FOV) distributions for the six analyzed sagittal slices. The figure shows the spatial arrangement of FOVs for each individual slice across slides L321 and L34. For each slice, the total number of FOVs, segmented cells, and analyzed tissue area are indicated, providing an overview of tissue coverage and the spatial extent of each section prior to downstream computational analysis.

### Quality Control

Following cell segmentation, quality control (QC) was performed to remove low-quality cells (explain) and technical artifacts prior to downstream computational analyses. QC is a critical preprocessing step in single-cell spatial transcriptomics, as segmentation errors, damaged cells, and low-quality transcript profiles can introduce noise and bias into subsequent computational analyses. For each segmented cell, several QC metrics were evaluated, including the total number of detected transcripts, the number of genes detected per cell, and morphological features derived from the segmentation masks. Specifically, cells with fewer than X detected transcripts or fewer than Y detected genes were considered likely to represent poorly captured cells, debris, or segmentation artifacts and were excluded from further analysis. In addition, cells exceeding Z for selected morphological outlier criteria were removed based on implausible segmentation profiles. Given the multi-slice structure of the dataset, QC was assessed across sections to account for differences in tissue coverage, cell density, and technical quality. The resulting filtered dataset served as the basis for all subsequent computational analyses.


| Slice ID | Mouse ID | Slide ID | Slice Type | Initial Cell Count | Removed: Low Transcripts | Removed: Low Genes | Removed: Morphology Outliers | Final Cell Count | Retention (%) |
| -------- | -------- | -------- | ---------- | ------------------ | ------------------------ | ------------------ | ---------------------------- | ---------------- | ------------- |
| 1        | 2        | L321     | T          | 140,615            | 15,659                   | TBD                | TBD                          | TBD              | 88.86%        |
| 2        | 3        | L321     | T          | 137,669            | 11,184                   | TBD                | TBD                          | TBD              | 91.88%        |
| 3        | 1        | L321     | C          | 69,007             | 5,010                    | TBD                | TBD                          | TBD              | 92.74%        |
| 4        | 1        | L34      | C          | 76,631             | 19,545                   | TBD                | TBD                          | TBD              | 74.49%        |
| 5        | 3        | L34      | T          | 221,554            | 15,344                   | TBD                | TBD                          | TBD              | 93.07%        |
| 6        | 2        | L34      | T          | 280,842            | 13,469                   | TBD                | TBD                          | TBD              | 95.20%        |


### Tumor Cell Identification

Identifying tumor cells in a metastatic brain section is non-trivial: the 958-gene CosMx panel does not include a single definitive marker for the D122 lung carcinoma line, and transcriptional similarity between metastatic carcinoma cells and certain stressed, proliferating, or atypical host populations can blur the boundary between tumor and non-tumor cells in expression space. To address this, tumor cells were identified using a two-stage approach: an initial reference-based annotation with SingleR using a D122-specific reference, followed by a supervised classifier-based refinement trained on high-confidence positive and negative reference cells derived from the data itself.

#### Stage 1 — Reference-based annotation with SingleR

Initial cell-type annotation was performed using SingleR [14], applied with a combined reference panel composed of three components: (i) a brain structural reference covering astrocytes, neurons, oligodendrocytes, vascular cells, and other stromal CNS populations; (ii) a brain immune reference covering microglia, BAMs, and infiltrating myeloid populations; and (iii) a tumor-specific reference derived from a publicly available bulk RNA-seq dataset of LLC1 (Lewis Lung Carcinoma) cells (Gene Expression Omnibus accession GSE103548) [15]. D122 is a highly metastatic sub-clone of the parental Lewis Lung Carcinoma lineage [16] and shares its core carcinoma transcriptional program; no single-cell reference for D122 specifically was publicly available at the time of analysis, so LLC1 was used as the closest available proxy. This choice captures the lung-carcinoma identity shared by the two lines at the transcriptional level, while D122-specific features — particularly those acquired during brain-metastatic adaptation — are not directly represented in the reference at this stage and are recovered by the downstream supervised classifier (Stage 2).

For each segmented cell, SingleR returned correlation-based scores against each reference (score_brain_struct, score_brain_immune, score_tumor) and the predicted label corresponding to the highest-scoring class. Cells whose top label was "Tumor" formed the initial pool of tumor candidates. To exclude weakly-matching calls, candidate cells were required to pass three filters: an absolute score floor, a margin between best and second-best reference, and a best-class consistency check. Exact threshold values are listed in Appendix Table X. These thresholds are deliberately permissive — their role is to define an inclusive candidate pool rather than to produce accurate tumor calls — and the downstream supervised classifier (Stage 2) is responsible for final specificity.

**Appendix Table X**


| Stage          | Parameter              | Value                         | Role                           |
| -------------- | ---------------------- | ----------------------------- | ------------------------------ |
| SingleR filter | score_tumor floor      | 0.2                           | absolute confidence            |
| SingleR filter | delta_score margin     | 0.08                          | discrimination from next class |
| SingleR filter | best-class consistency | score_tumor > next_best_score | —                              |
| ...            | ...                    | ...                           | ...                            |


#### Motivation for refinement

Visual inspection of the spatial distribution of SingleR-called tumor cells revealed two systematic issues. First, a substantial number of cells annotated as "Tumor" were located in the sham-injected control slices (slices 3 and 4), where no metastatic cells can be present by experimental design. Second, additional "Tumor" calls appeared in distal, clearly non-tumor anatomical regions of the tumor-bearing slices. Because D122 cells were stereotactically injected into the contralateral hemisphere and tumors developed as locally compact lesions, these distributed calls cannot represent true metastatic cells. They instead reflect transcriptional overlap between D122-like signal and certain endogenous brain populations within the limited 958-gene panel — i.e., the panel cannot, on its own, fully discriminate tumor cells from a subset of host look-alike cells. This finding established that reference-based annotation alone was insufficient and motivated a supervised refinement step trained on the specific failure modes of SingleR in this dataset.

**Figure 3.** Initial tumor candidates from reference-based annotation (SingleR). Per-slice candidate counts: Slice 1 — L321, mouse 2 (Tumor): 5,793 candidates / 124,956 cells (4.6%); Slice 2 — L321, mouse 3 (Tumor): 7,650 candidates / 126,485 cells (6.0%); Slice 3 — L321, mouse 1 (Control): 343 candidates / 63,997 cells (0.5%); Slice 4 — L34, mouse 1 (Control): 287 candidates / 57,086 cells (0.5%); Slice 5 — L34, mouse 3 (Tumor): 8,129 candidates / 206,210 cells (3.9%); Slice 6 — L34, mouse 2 (Tumor): 9,296 candidates / 267,373 cells (3.5%).

#### Stage 2 — Supervised refinement of tumor candidates

A supervised binary classification framework was used to distinguish true tumor cells from "look-alike" host cells that had been incorrectly annotated as tumor by SingleR. Crucially, the negative (healthy) reference class was constructed from the data itself, by exploiting the fact that any cell called "Tumor" in a sham-injected control slice must be a healthy cell mis-annotated by SingleR — and is therefore exactly the kind of false-positive the classifier needs to learn to reject.

Reference labels were defined as follows:

- **Tumor reference (positive class):** cells from the tumor-bearing slices with the strongest evidence for tumor identity — predicted_cell_type = "Tumor", score_tumor ≥ 0.4, delta_score > 0.08, and score_tumor > next_best_score. Slice 1 (L321) and slice 5 (L34) were used as the high-confidence tumor source.
- **Healthy reference (look-alike class):** cells from the sham-injected control slices (slice 3 for L321, slice 4 for L34) flagged as "Tumor" by SingleR under the candidate filters. Because no metastatic cells are present in control tissue, these cells provide a biologically meaningful negative class that explicitly captures the failure modes of the reference-based step.
- **Tumor candidates:** all cells in the tumor-bearing slices flagged as "Tumor" by SingleR under the candidate filters; these are the cells to be re-scored by the classifier.

Five classifier configurations were evaluated: (1) L2-regularised logistic regression with class balancing applied to the full gene space; (2) logistic regression in a 50-component PCA-reduced space; (3) a hybrid logistic-regression + k-nearest-neighbour classifier operating in the same PCA space, in which a cell was considered healthy if either the logistic probability or the fraction of healthy k-NN neighbours exceeded the operating threshold; (4) random forest and (5) gradient-boosted trees (XGBoost) on the full gene space, with scale_pos_weight set to the empirical negative-to-positive ratio to compensate for the strong class imbalance in the reference pool.

All five classifiers achieved high and broadly comparable performance on the joint reference pool under 5-fold cross-validation (Figure 3), with out-of-fold ROC-AUC > 0.99 and average precision > 0.98 across all configurations. This near-uniformity is a consequence of the high purity of both training classes by construction — the positive set comprises high-confidence SingleR tumor anchors, and the negative set comprises experimentally-impossible control-slice calls — making the discrimination on this set close to trivial for any reasonable model. The operationally meaningful evaluation is therefore not the cross-validated reference-set metrics, but the classifiers' behaviour on the SingleR candidate pool: cells with intermediate or ambiguous tumor signal that were flagged for refinement but were not used in training.

On the candidate pool, the classifiers diverge: the linear models (LogReg, LogReg+PCA, LogReg+KNN) reject a larger fraction of candidates as look-alikes, while the tree-based models retain more. Inspection of the spatial distribution of retained tumor cells across slices (Figure 4) showed that XGBoost preserved coherent tumor masses with cleanly removed peripheral false positives, whereas the linear models additionally removed substantial fractions of cells within the dense tumor core — likely reflecting transcriptional heterogeneity within the tumor that the linear decision boundary could not capture. XGBoost was selected on this basis: its non-linear decision function and its ability to model interactions between marker genes were better-suited to the structural heterogeneity of the candidate pool than the linear or PCA-projected alternatives. Random Forest produced spatially similar behaviour to XGBoost but with marginally lower precision at the operating threshold.

#### Model selection and final tumor calls

All five classifiers achieved high OOF performance on the joint reference pool (ROC-AUC > [X.XX], F1 > [X.XX]; Table X, Figure X). XGBoost was selected as the final tumor classifier on the basis of three considerations: (i) it achieved the strongest OOF performance on the metrics most relevant to downstream analyses — precision and F1 — minimising the number of healthy cells incorrectly retained as tumor; (ii) its tree-based, non-linear decision function captures interactions between panel genes that linear models cannot represent, which is particularly valuable given the overlapping expression structure observed between tumor and look-alike cells in the PCA visualisation; and (iii) its built-in handling of class imbalance through scale_pos_weight produced stable behaviour despite the modest size of the healthy reference set.

The selected XGBoost model was then applied to all SingleR-derived tumor candidates across all six analysed slices. For each candidate, the model returned a probability P(healthy); cells with P(healthy) < 0.5 were retained as refined tumor cells, while cells with P(healthy) ≥ 0.5 were re-classified as look-alike host cells and excluded from downstream tumor-centric analyses. Spatial inspection of the refined tumor population confirmed that rejected (look-alike) cells were concentrated in the control slices and in distal non-tumor regions of the tumor-bearing slices, while the retained tumor cells formed compact, spatially coherent clusters consistent with the histologically observed tumor masses (Figure X).

**Figure 4.** Spatial refinement of SingleR tumor candidates across classifiers (LogReg, LogReg + PCA, LogReg + KNN (PCA), Random Forest, XGBoost). Per-slice kept/rejected counts (retained %):

- Slice 1, L321 (Tumor): LogReg 3,901 kept / 1,892 rejected (67.3%); LogReg+PCA 4,401 / 1,392 (76.0%); LogReg+KNN 4,082 / 1,711 (70.5%); Random Forest 4,210 / 1,583 (72.7%); XGBoost 4,248 / 1,545 (73.3%).
- Slice 2, L321 (Tumor): LogReg 5,779 / 1,871 (75.5%); LogReg+PCA 6,259 / 1,391 (81.8%); LogReg+KNN 5,920 / 1,730 (77.4%); Random Forest 6,083 / 1,567 (79.5%); XGBoost 6,243 / 1,407 (81.6%).
- Slice 3, L321 (Control): LogReg 0 / 343 (0.0%); LogReg+PCA 5 / 338 (1.5%); LogReg+KNN 2 / 341 (0.6%); Random Forest 0 / 343 (0.0%); XGBoost 0 / 343 (0.0%).
- Slice 4, L34 (Control): LogReg 0 / 287 (0.0%); LogReg+PCA 5 / 282 (1.7%); LogReg+KNN 1 / 286 (0.3%); Random Forest 0 / 287 (0.0%); XGBoost 0 / 287 (0.0%).
- Slice 5, L34 (Tumor): LogReg 4,198 / 3,931 (51.6%); LogReg+PCA 5,080 / 3,049 (62.5%); LogReg+KNN 4,394 / 3,735 (54.1%); Random Forest 4,747 / 3,382 (58.4%); XGBoost 4,981 / 3,148 (61.3%).
- Slice 6, L34 (Tumor): LogReg 4,346 / 4,950 (46.6%); LogReg+PCA 5,312 / 3,984 (57.1%); LogReg+KNN 4,572 / 4,724 (49.2%); Random Forest 5,132 / 4,164 (55.2%); XGBoost 5,207 / 4,089 (56.0%).

## Graph Modeling of Brain Metastasis

*(Section to be written.)*

## References

[1] Paisana E, Cascão R, Alvoeiro M, Félix F, Martins G, Guerreiro C, et al. Immunotherapy in lung cancer brain metastases. npj Precision Oncology. 2025;9:130. doi:10.1038/s41698-025-00901-0.

[2] Feng Y, Hu X, Zhang Y, Wang Y. The role of microglia in brain metastases: mechanisms and strategies. Aging and Disease. 2024;15(1):169-185. doi:10.14336/AD.2023.0514.

[3] Sun R, Jiang H. Border-associated macrophages in the central nervous system. Journal of Neuroinflammation. 2024;21(1):67. doi:10.1186/s12974-024-03059-x.

[4] Van Hove H, Martens L, Scheyltjens I, De Vlaminck K, Pombo Antunes AR, De Prijck S, et al. A single-cell atlas of mouse brain macrophages reveals unique transcriptional identities shaped by ontogeny and tissue environment. Nature Neuroscience. 2019;22(6):1021-1035. doi:10.1038/s41593-019-0393-4.

[5] Kierdorf K, Masuda T, Jordão MJC, Prinz M. Macrophages at CNS interfaces: ontogeny and function in health and disease. Nature Reviews Neuroscience. 2019;20(9):547-562. doi:10.1038/s41583-019-0201-x.

[6] Gonzalez H, Mei W, Robles I, Hagerling C, Allen BM, Nanjaraj A, et al. Cellular architecture of human brain metastases. Cell. 2022;185(4):729-745.e20. doi:10.1016/j.cell.2021.12.043.

[7] Rao A, Barkley D, França GS, Yanai I. Exploring tissue architecture using spatial transcriptomics. Nature. 2021;596(7871):211-220. doi:10.1038/s41586-021-03634-9.

[8] Chen Y, Zhang A, Wang J, Pan H, Liu L, Li R. Refining lung cancer brain metastasis models for spatiotemporal dynamic research and personalized therapy. Cancers. 2025;17(9):1588. doi:10.3390/cancers17091588.

[9] Liu Z, Gu Y, Chakarov S, Bleriot C, Kwok I, Chen X, et al. Fate mapping via Ms4a3-expression history traces monocyte-derived cells. Cell. 2019;178(6):1509-1525.e19. doi:10.1016/j.cell.2019.08.009.

[10] Peng L, Wang Y, Fei S, Wei C, Tong F, Wu G, et al. The effect of combining Endostar with radiotherapy on blood vessels, tumor-associated macrophages, and T cells in brain metastases of Lewis lung cancer. Translational Lung Cancer Research. 2020;9(3):745-760. doi:10.21037/tlcr-20-500.

[11] Ratzabi A, Caspit IM, Telechi I, Kim JS, Vaknine H, Blinder P, et al. Brain metastases exhibit distinct spatial patterns of resident and infiltrating macrophages. Research Square. 2025. doi:10.21203/rs.3.rs-8294376/v1. Preprint.

[12] Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nature Methods. 2021;18(1):100-106. doi:10.1038/s41592-020-01018-x.

[13] NanoString Technologies. Evaluating the technical performance of single-cell spatial biology with CosMx Spatial Molecular Imager. Seattle (WA): NanoString Technologies; 2023. White paper.

[14] Aran D, Looney AP, Liu L, Wu E, Fong V, Hsu A, et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. Nature Immunology. 2019;20(2):163-172. doi:10.1038/s41590-018-0276-y.

[15] Kumar R. Analysis of mRNA expression in Lewis lung carcinoma (LLC1) cells and MLE 12 cells [dataset]. Gene Expression Omnibus; 2018. Accession no. GSE103548. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103548. Associated publication: Ryan ZC, Craig TA, Wang X, Delmotte P, Salisbury JL, Lanza IR, et al. 1α,25-dihydroxyvitamin D3 mitigates cancer cell-mediated mitochondrial dysfunction in human skeletal muscle cells. Biochemical and Biophysical Research Communications. 2018;496(2):746-752. doi:10.1016/j.bbrc.2018.01.092.

[16] Eisenbach L, Segal S, Feldman M. MHC imbalance and metastatic spread in Lewis lung carcinoma clones. International Journal of Cancer. 1983;32(1):113-120. doi:10.1002/ijc.2910320118.