# Thesis sections — Data-quality assessment of custom lineage-reporter probes and segmentation

*Draft prose for direct insertion into the thesis (Word). Suggested placement: as a new
major section titled **"Data-Quality Assessment of Custom Probes and Segmentation"** immediately
after **Tumor Cell Identification** (and before **Graph Modeling of Brain Metastasis**), followed
by the **Limitations and Planned Re-profiling** section. Figure/table numbers are placeholders —
renumber to follow your existing Figure 6. "(Ref)" marks where a citation should be inserted, matching
the draft's convention. All quantitative values are from the slice-1 non-tumor analysis (n = 120,708
cells) unless a per-slice range is given; equivalent analyses on slices 2 and 3 gave concordant results.*

---

## Data-Quality Assessment of Custom Probes and Segmentation

A central aim of this work is the discrimination of resident (microglia, BAM) from infiltrating
(MDM) myeloid populations, and the experimental design placed particular weight on four custom
add-on probes for this purpose: the two transgene reporters GFP (Cx3cr1-driven, labelling resident
myeloid cells) and tdTomato (Ms4a3-lineage, labelling monocyte-derived cells), together with the
microglial marker TMEM119 and the BAM marker Lyve1. Because the interpretation of the myeloid
compartment depends directly on these probes, their technical reliability was assessed explicitly
before they were used for annotation. This assessment revealed systematic anomalies in the
lineage-reporter signal that could not be reconciled with the expected biology, and that were
subsequently corroborated by an independent technical assessment from the processing facility. The
analyses below establish (i) that the reporter probes do not behave as reliable lineage tags in this
run, (ii) that the anomalies are not attributable to ambient RNA or cell-segmentation error and
therefore cannot be corrected computationally, and (iii) that their origin lies in identifiable
run-level technical issues, which motivates a planned repeat experiment.

### Probe signal-to-background and detection reliability

For each probe, signal was quantified against the technical noise floor defined by the 11 negative-control
probes on the panel (the `Negative`* targets; the `SystemControl*` probes are optical-decode controls and
were excluded). A signal-to-background (S/N) ratio was computed as the mean per-cell probe count divided
by the mean per-cell negative-probe count (0.025 counts/cell), and the fraction of a probe's total signal
exceeding the per-cell background was recorded as a specificity index. Two panel genes were used as
internal calibrators: the custom probe GFAP as a high-dynamic-range positive control (an abundant,
well-characterised astrocyte transcript), and the housekeeping-like neuronal transcript Meg3 as the
strongest-detected reference on the panel.

**Table 2 | Detection reliability of the custom add-on probes benchmarked against panel reference genes**

| Probe    | Role                     | Mean counts per cell | Max counts per cell | Cells positive (%) | S/N  | Signal above background (%) |
| -------- | ------------------------ | -------------------- | ------------------- | ------------------ | ---- | --------------------------- |
| *Custom add-on probes*                                                                                                                            ||||||
| GFAP     | Positive control (astrocyte) | 0.318            | 90                  | 17.8               | 12.5 | 92                          |
| tdTomato | Lineage reporter         | 0.341                | 15                  | 21.0               | 13.4 | 93                          |
| Ccl2     | Chemokine                | 0.256                | 9                   | 17.2               | 10.1 | 90                          |
| TMEM119  | Microglia                | 0.225                | 14                  | 15.2               | 8.9  | 89                          |
| Lyve1    | BAM                      | 0.184                | 9                   | 12.9               | 7.2  | 86                          |
| Trem2    | Myeloid                  | 0.122                | 8                   | 9.0                | 4.8  | 79                          |
| Cxcl13   | Chemokine                | 0.098                | 6                   | 7.4                | 3.8  | 74                          |
| **GFP**  | Lineage reporter         | **0.044**            | 4                   | **3.6**            | **1.7** | **42**                   |
| *Panel reference genes*                                                                                                                           ||||||
| Meg3     | Reference (neuronal)     | 1.867                | 96                  | 37.8               | 73.4 | 99                          |
| Pecam1   | Reference (endothelial)  | 0.262                | 22                  | 17.6               | 10.3 | 90                          |
| Cx3cr1   | Reference (myeloid)      | 0.215                | 8                   | 14.6               | 8.4  | 88                          |
| Csf1r    | Reference (myeloid)      | 0.128                | 9                   | 9.3                | 5.0  | 80                          |

Slice 1, non-tumor cells, *n* = 120,708. Signal-to-noise (S/N) is the mean per-cell probe count divided by
the mean per-cell count of the 11 negative-control probes (0.025 counts per cell); SystemControl probes are
optical-decode controls and were excluded. Cells positive is the fraction of cells with at least one count.
Signal above background is the fraction of a probe's molecules exceeding the cell-specific Poisson background
defined by the negative-control probes.
Values in bold identify the GFP probe, the only probe failing every detection criterion.
BAM, border-associated macrophage; S/N, signal-to-noise.


The GFP reporter emerged as a failed probe. Its S/N of 1.73 is more than seven-fold below the GFAP
positive control (12.5) and below every reference gene tested, it was detected in only 3.6% of cells,
and fewer than half of its molecules (42%) exceeded the negative-probe background — meaning the
majority of the nominal GFP signal is indistinguishable from technical noise. Critically, the per-cell
GFP detection rate scaled almost monotonically with a cell's total transcript count, rising from 0.35%
in the lowest count-decile to ~6% in the highest, the signature of a count-driven ambient/false-positive
process rather than a true, cell-type-restricted transgene. The GFP probe is therefore not usable as a
resident-lineage tag in this run.

The tdTomato reporter presented the opposite failure mode: it was detected efficiently (S/N 13.4,
positive in 21.0% of cells — the highest of any custom probe) but non-specifically. Its single strongest
correlation with any panel gene was not with a myeloid transcript but with the neuronal reference Meg3
(Pearson r ≈ 0.10, Spearman ρ ≈ 0.12), a pattern consistent with a diffuse background signal rather than
with monocyte-lineage-restricted expression. TMEM119, Lyve1, Trem2 and the chemokine probes (Ccl2,
Cxcl13) all detected at plausible, if modest, levels comparable to their panel-reference neighbours,
indicating that the detection problem was specific to the reporter targets rather than a blanket failure
of the custom probe set.

### The lineage reporters contradict their expected biology

Beyond detection quality, the two reporters failed every test of biological consistency (Figure DQ2).

First, the reporter correlations were of the wrong sign. GFP and Cx3cr1 — both expected to mark resident
myeloid cells — were *negatively* correlated (Pearson r = −0.51 on slice 1, and −0.50 / −0.53 / −0.57
across slices 1–3), driven by cells expressing one tag or the other but rarely both. GFP and tdTomato,
which by the lineage-tracing design should be co-expressed in monocyte-derived macrophages (GFP⁺tdTomato⁺),
were likewise negatively correlated (r = −0.33). Under the intended reporter logic these correlations
should be strongly positive.

Second, tdTomato did not support the resident-versus-infiltrating distinction it was included to make.
Defining candidate microglia as GFP⁺TMEM119⁺ and candidate MDMs as GFP⁺TMEM119⁻, the fraction of
tdTomato-positive cells was consistently *higher* in the microglia set than in the MDM set (43.4% vs
30.7% on slice 1, and comparably on slices 2–3) — the exact inverse of the expected MDM-specific tdTomato
signal.

Third, the BAM reporter Lyve1 was too sparse to anchor a population on its own: it was detected in only
~13% of non-tumor cells, and ~90% of Lyve1-positive cells lacked the canonical BAM markers Mrc1/Cd163,
so Lyve1 positivity alone would have over-called BAM by a large margin.

Taken together, these results demonstrate that the transgene reporters cannot serve as ground-truth
lineage labels in this dataset, and that a reporter-gated annotation of the myeloid compartment would be
unreliable.

### The anomalies are not ambient RNA or segmentation error

Two candidate technical explanations internal to the analysis pipeline — ambient (spill-over) RNA and
cell-segmentation error — were tested directly and ruled out, establishing that the anomalies could not
be corrected downstream.

**Ambient-RNA correction (decontX).** Per-cell ambient contamination was estimated and subtracted with
decontX (Ref), run on the full slice (tumor and non-tumor) with coarse Leiden clusters supplied as the
population prior so that the tumor and neuronal ambient pools were modelled. Median estimated
contamination was ~12% per cell. Stratifying cells into terciles of estimated contamination did not
resolve the reporter anomalies: the GFP↔Cx3cr1 and GFP↔tdTomato negative correlations were essentially
constant across low-, mid- and high-contamination strata, and GFP-positive and GFP-negative cells had
statistically indistinguishable contamination fractions (0.123 vs 0.124). The anomalies therefore do not
track ambient load, and ambient correction did not remove them (Figure DQ3).

**Re-segmentation.** Because segmentation error (a cell capturing a neighbour's transcripts) was the
leading a-priori hypothesis for both the myeloid-subtype confusion and the reporter correlations, a
decisive re-segmentation test was performed on a set of slice-1 fields of view, comparing the vendor
segmentation with a transcript-reassignment re-segmentation. Re-segmentation moved essentially nothing:
myeloid marker purity was unchanged (0.802 → 0.802), and the reporter correlations were unchanged
(GFP↔tdTomato −0.350 → −0.340; GFP↔Cx3cr1 −0.318 → −0.322). Direct foreign-transcript contamination was
low (median 1.8% of counts per cell), and even an extreme reassignment moved fewer than 1% of transcripts.
A parallel morphology-channel QC found no dead or saturated imaging channel, and the removal of
weak-membrane FOVs did not improve annotation — those cells were, if anything, marginally better than
average. Segmentation is thus not the driver of the reporter anomalies, and the residual myeloid-subtype
overlap reflects a genuine limit of the 958-gene panel's lineage-discriminating content rather than a
mis-segmentation artefact.

Because neither ambient correction nor re-segmentation altered the reporter behaviour, the origin of the
anomalies must lie upstream of the digital data — in the chemistry and imaging of the run itself.

### Convergence with the processing-facility technical assessment

The computational forensics above were arrived at independently, and were subsequently confirmed by a
technical assessment provided by the processing facility, which identified three concurrent run-level
issues whose expected effects map directly onto the anomalies observed in the data.

*Reagent-tray expiration.* The run logs indicate that the reagent tray was used within one to two days of
its expiration date. Combined with possible storage variability, this can compromise the performance and
stability of specific reporters and reduce their detection rate. This is the expected cause of the failed
GFP probe (S/N 1.73; 42% of signal above background), whose near-absence of specific signal is consistent
with a degraded reporter rather than with a true scarcity of resident myeloid cells.

*Yellow-channel imaging artefact.* In CosMx experiments — and particularly in mouse tissue — residual
fluorescence in the yellow channel, especially from the membrane segmentation marker, can persist into
later detection cycles and inflate the false-positive detection rate of specific targets. The facility
specifically noted unexpectedly high detection of tdTomato as a manifestation of this effect. This
directly explains the tdTomato profile in the data: efficient but non-specific detection, its strongest
association with a neuronal rather than a myeloid transcript, and its paradoxical enrichment in candidate
microglia over candidate MDMs — all consistent with a background signal rather than lineage-restricted
expression.

*Segmentation strategy.* Segmentation was optimised for the tumor and immune content of the sections
rather than for the surrounding brain parenchyma, producing spatially heterogeneous segmentation and
transcript-mapping quality across anatomical regions. This is consistent with the spatially localised
weak-membrane FOVs observed in the morphology QC, and with the difficulty of resolving parenchymal
(neuronal/oligodendroglial) populations, for which the panel additionally lacks canonical markers.

Importantly, the facility noted that the custom add-on probes were not specifically implicated: endogenous
panel targets were affected by the same run-level issues to a comparable degree, and the analytical focus
on the add-on genes reflects their biological relevance to this project rather than a probe-design defect.
This is consistent with the internal calibration presented above (Figure DQ1), in which the custom positive-control
probe GFAP behaved as a well-detected, high-dynamic-range target (S/N 12.5) while the custom reporters
GFP and tdTomato failed in target-specific ways — indicating per-target reagent and imaging effects rather
than a systematic failure of the custom probe chemistry.

### Implications for the annotation strategy

These findings shaped the annotation approach adopted throughout this work. Because the transgene reporters
could not be trusted as lineage tags, cell-type identity was never gated on GFP, tdTomato or Cx3cr1.
Instead, annotation used positive, marker-based gating on transcriptome content calibrated against the
negative-probe background, with the reporters retained only as informational, low-confidence overlays.
The myeloid compartment was resolved with a control-calibrated hierarchical gate that establishes the
specificity bar for the infiltrating (MDM) and border-associated (BAM) modules from the tumor-free control
tissue, where infiltrating cells should be near-absent; and the microglia/MDM boundary — the fragile axis
given the thin on-panel microglial signature and the failed reporters — was reported with a confidence
grade and an explicit "unresolved" class rather than forced into a hard call. This reporter-independent
strategy is what allows the present dataset to be interpreted despite the technical issues, and it will
transfer directly to the repeat experiment.

---

## Limitations and Planned Re-profiling

The technical issues characterised above impose real limitations on the present dataset. The GFP reporter
does not provide a usable resident-lineage signal, the tdTomato reporter carries a non-specific background
component that precludes its use as an infiltrating-lineage tag, and segmentation and transcript-mapping
quality vary across anatomical regions, with the brain parenchyma resolved less reliably than the tumor
and immune compartments. As a result, the resident-versus-infiltrating myeloid distinction cannot be
anchored on the lineage-tracing reporters as originally intended, and rests instead on transcriptomic
marker evidence, which is at the discriminative ceiling of the 958-gene panel for the microglia/MDM axis.
These are limitations of this particular run — traceable to an expiring reagent tray, a yellow-channel
imaging artefact, and a tumor-and-immune-focused segmentation strategy — and not of the experimental model
or the computational framework.

Accordingly, a repeat of the spatial-transcriptomic profiling is planned, in coordination with the
processing facility, and is designed specifically to resolve these issues:

- **Fresh sections and a fresh reagent tray.** New sections will be cut and profiled with replacement
reagents and a reagent tray well within its validity window, removing the degraded-reporter cause of the
GFP failure.
- **Revised segmentation strategy.** The segmentation panel will be extended to include neuronal
segmentation markers, so that the surrounding brain parenchyma is segmented as reliably as the tumor and
immune compartments; concurrently, the concentration of the yellow-channel segmentation marker will be
reduced to suppress the background fluorescence responsible for the tdTomato false positives. This
addresses the segmentation-heterogeneity and yellow-channel-bleed causes together.
- **Optional expanded panel.** The facility has offered to profile serial sections with the Mouse Neuro
1k panel in addition to the Mouse Universal Cell Characterization panel; incorporating the Neuro panel
would substantially strengthen the parenchymal (neuronal/oligodendroglial) coverage that the current
panel lacks, and would improve the microglial-identity axis.

The complete computational framework developed in this thesis — reference-based plus supervised tumor-cell
identification, reporter-independent marker-based cell-type annotation with negative-probe-calibrated
gating, the control-calibrated myeloid subtyping gate, and the planned graph-based modelling of cell
neighbourhoods — has been established and validated on the current data and is directly transferable. Once
the repeat run yields sections with fresh reagents, a parenchyma-aware segmentation, and functional
lineage reporters, this pipeline will be applied to them, and the research will continue on that improved
dataset, with the transgene reporters expected to provide the independent resident-versus-infiltrating
ground truth that this run could not.

---

## Figures

*Three thesis-ready figures accompany this section (PNG files in `thesis_plots/`; regenerated by
`thesis_plots/make_dq_figures.py`). Insert each PNG in Word and use the captions below. All panels use a
colorblind-safe palette in which green marks the custom positive control (GFAP), red marks the failed/
anomalous reporter (GFP), orange marks tdTomato, blue marks other custom probes, and grey marks panel
reference genes.*

### Figure DQ1

Custom-probe detection reliability

**Figure DQ1. Detection reliability of the custom add-on probes (slice 1, non-tumor cells, n = 120,708).**
(a) Per-probe signal-to-background ratio (mean probe counts / mean negative-probe counts; log scale); the
dashed line marks the background floor (S/N = 1). The custom positive control GFAP (green) and the panel
reference genes (grey) detect strongly, whereas the GFP reporter (red) sits just above background (S/N 1.73).
(b) Fraction of each probe's signal exceeding the per-cell negative-probe background (a specificity index);
GFP is the sole probe below 0.5 (0.42), i.e. most of its signal is indistinguishable from noise. (c) Percentage
of GFP-positive cells across deciles of total transcript count: GFP positivity rises with sequencing depth
rather than marking a discrete cell population, the signature of a count-driven ambient/false-positive process.

### Figure DQ2

Lineage reporters contradict expected biology

**Figure DQ2. The transgene reporters contradict their expected biology.** (a) Pearson correlations between
reporter pairs that should be positively associated by the lineage-tracing design are instead negative
(GFP–Cx3cr1, both resident-myeloid, r = −0.51; GFP–tdTomato, co-expressed in monocyte-derived cells by
design, r = −0.33; slice 1). (b) Percentage of tdTomato-positive cells among reporter-defined microglia
(GFP⁺TMEM119⁺) versus MDM (GFP⁺TMEM119⁻) across slices 1–3: tdTomato — intended as a monocyte-derived-lineage
tag — is consistently *higher* in candidate microglia than in candidate MDMs, the inverse of the expected
pattern. (c) tdTomato's Pearson correlation with a panel of lineage markers (slice 1): its single strongest
association is with the neuronal transcript Meg3 (red), not with any myeloid gene, consistent with a diffuse
background signal rather than lineage-restricted expression.

### Figure DQ3

Ambient correction does not rescue the reporter anomalies

**Figure DQ3. Ambient-RNA correction (decontX) does not rescue the reporter anomalies.** Reporter correlations
(GFP–tdTomato, orange; GFP–Cx3cr1, red) computed within terciles of decontX-estimated ambient contamination
(slice 1). The anomalous negative correlations are essentially constant across low-, mid- and high-contamination
strata, demonstrating that they do not track ambient load and are not removed by ambient correction — evidence
that the anomalies originate upstream of the digital data (run chemistry and imaging) rather than in ambient RNA.

### Supplementary source figures and data

The following exploratory outputs underlie the figures above and can be cited as supplementary material:
probe reliability table data — `agents/outputs/probe_validation/signal_to_background_slice1.csv`,
`metrics_slice1.json`; reporter/marker co-expression heatmap —
`agents/outputs/probe_validation/coexpression_heatmap_slice1.png`. Staged per-probe spatial-expression
panels for the biology-critical custom probes are available at `thesis_plots/cx3cr1.png`,
`thesis_plots/gfap.png`, `thesis_plots/lyve1.png`, `thesis_plots/tdtomato.png`, `thesis_plots/tmem.png`,
suitable as a supplementary spatial-distribution figure.