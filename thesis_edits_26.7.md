# Thesis edits — "Thesis Draft 26.7"

Prepared from the PDF render. Page numbers below are the **PDF** page numbers.
Apply in the Word source; where I give `FIND → REPLACE`, use Word's Find & Replace
(Home → Replace, or Ctrl+H) with **Match case** on unless noted.

---

## 1. Abstract (ready to paste — place before the Table of Contents)

> **Abstract**
>
> Brain metastases are a common and lethal complication of advanced cancer, and lung
> carcinoma is among their leading primary sources. Their progression is shaped not only by
> malignant cells but by a heterogeneous tumor microenvironment (TME) in which myeloid
> populations — resident microglia, border-associated macrophages (BAMs), and
> monocyte-derived macrophages (MDMs) — are especially abundant and functionally dynamic.
> Distinguishing these populations, and situating them within tissue architecture, is central to
> understanding how the metastatic niche is organized.
>
> This thesis develops a single-cell spatial-transcriptomic and graph-based computational
> framework for the metastatic brain, using a D122 lung-carcinoma model in
> *Ms4a3^cre^:R26^tdT^:Cx3cr1^Gfp^* reporter mice profiled on the NanoString CosMx platform
> (six sagittal slices, 926,318 segmented cells, 958 genes including eight custom probes).
> Tumor cells were identified with a two-stage procedure — reference-based annotation
> (SingleR, anchored to an external LLC1 carcinoma profile and brain references) followed by
> supervised refinement (XGBoost) whose negative class was defined from sham-injected
> control tissue — yielding spatially coherent tumor masses consistent with matched H&E
> histology. Because the transgene lineage reporters could not be trusted as ground truth, cell
> types were assigned by a reporter-independent, negative-probe-calibrated hierarchical gate,
> with a control-calibrated specificity bar for the infiltrating and border-associated myeloid
> modules.
>
> A rigorous data-quality assessment established that this run was compromised by
> compounding, run-level technical faults — a failed GFP reporter from a near-expiry reagent
> tray, a yellow-channel imaging artifact inflating tdTomato, a tumor-and-immune-focused
> segmentation strategy, and the limited lineage-discriminating content of the panel — which
> were traced to identifiable causes, confirmed by the processing facility, and shown to be
> uncorrectable computationally. The analysis was therefore deliberately halted and the
> experiment scheduled for repetition. The thesis contributes the experimental model, the
> complete computational pipeline, and a thorough characterization of the data-quality
> limitations that motivate the planned re-profiling, on which the biological analysis of the
> metastatic niche will be carried out.

*(~330 words, US-English. Note: TME is defined here and again in the Introduction — that is
standard, because the abstract is read as self-contained. Keep both.)*

---

## 2. Table of Contents

**Recommended method:** apply Word Heading styles (Heading 1 = major sections, Heading 2 =
subsections), then insert an automatic TOC (References → Table of Contents → Automatic).
Word will fill and auto-update page numbers, which shift once the Abstract + TOC are added.

Entry structure (page numbers are the *current* PDF pages; let Word recompute them):

```
Abstract
Table of Contents

1  Introduction ......................................................... 2
     Research Objectives and Significance ............................... 4

2  Materials and Methods ................................................ 5
     Lung Carcinoma Cell Line ........................................... 5
     Mice ............................................................... 5
     Generation of the Brain Metastasis Model ........................... 5
     Tissue Processing and Section Selection ............................ 6
     Experimental Design ................................................ 6
     CosMx Spatial Transcriptomic Profiling ............................. 6
     Cell Segmentation .................................................. 7

3  Computational Analysis ............................................... 10
     Quality Control .................................................... 11
     Tumor Cell Identification .......................................... 12
         Stage 1 – Reference-based annotation with SingleR .............. 13
         Motivation for refinement ..................................... 13
         Stage 2 – Supervised refinement of tumor candidates ........... 14
         Stage 3 – Model selection and final tumor calls ............... 19
     Data-Quality Assessment of Custom Probes and Segmentation .......... 20
         Probe signal-to-background and detection reliability .......... 20
         The spatial distribution of the lineage reporters
             contradicts their expected biology ....................... 22
         The anomalies are not ambient RNA or segmentation error ....... 25
         Convergence with the processing-facility technical assessment . 26
         The tumor annotation is robust to these run-level issues ...... 27
         Implications for the annotation strategy ...................... 27
     Limitations and Planned Re-profiling ............................... 27
     Cell-type annotation strategy ...................................... 29
     Cell Annotation .................................................... 30
     Graph Modeling of Brain Metastasis ................................. 31

References ............................................................. 32
```

*(Section numbers 2 and 3 are suggested — the current draft only numbers "1 Introduction".
Either number all top-level sections or none, for consistency.)*

---

## 3. Double-space removal (do this in Word — cannot be done from the PDF)

Ctrl+H (Replace):
- **Find what:** `  ` (two spaces) → **Replace with:** ` ` (one space)
- Click **Replace All**, then repeat until it reports **0 replacements** (catches runs of 3+ spaces).

This does not affect non-breaking spaces. Optionally also enable
File → Options → Proofing → "Two spaces between sentences" flagged, to catch stragglers.

---

## 4. US-English spelling corrections (Replace All, Match case)

The draft mixes British and American spellings. Convert the British forms below.
**Do NOT change spellings inside reference titles** (e.g. "brain-tumour", "breast-tumour" in
refs [12], [21]) — cited titles keep their original spelling.

| # | Find (British) | Replace (US) | Notes / pages |
|---|---|---|---|
| 1 | characterised | characterized | p11, p20, p27–28 |
| 2 | characterising | characterizing | p20 |
| 3 | optimised | optimized | p26 (×2) |
| 4 | regularised | regularized | p15 |
| 5 | behaviour | behavior | p15 (×2), p19, p26 |
| 6 | neighbour | neighbor | p15, p26 |
| 7 | neighbours | neighbors | p15 |
| 8 | neighbourhood | neighborhood | — |
| 9 | neighbourhoods | neighborhoods | p28, p31 |
| 10 | modelling | modeling | p28, p31 (match doc title "…Modeling") |
| 11 | modelled | modeled | p25 |
| 12 | favour | favor | p31 |
| 13 | programme | program | p27 |
| 14 | visualisation | visualization | p19 |
| 15 | binarised | binarized | p30 |
| 16 | analysed | analyzed | p19 |
| 17 | minimising | minimizing | p19 |
| 18 | artefact | artifact | p22, p26, p28, p29 |
| 19 | artefacts | artifacts | p26 |
| 20 | labelled | labeled | p29, p30 |

After these, run Word's built-in checker with the document language set to **English (United
States)** (Review → Language → Set Proofing Language → English (US), whole document) to catch
anything I could not see in the PDF.

---

## 5. Abbreviation consistency (Task 4 — define once, then use the abbreviation)

**Note on your example:** in the version I have, page-3/4 already reads "…within a spatially
organized **TME**", so that specific instance is already correct. The redundant re-definitions I
*did* find are the ones where a term already defined earlier is spelled out **and re-abbreviated
again**. After first definition, use the short form only.

Defined terms and where they are needlessly re-expanded later:

| Term (defined at) | Later redundant spell-out → fix |
|---|---|
| BAMs — defined p2 | p24 caption, p29, p30: "border-associated macrophages (BAMs)" → **BAMs** |
| MDMs — defined p2 | p29, p30: "monocyte-derived macrophages (MDMs)" → **MDMs** |
| FOV/FOVs — defined p7 | Fig. 1 & Fig. 2 captions re-expand "Field-of-view (FOV)" → **FOV** |
| TME — defined p2 | Scan body for any later "tumor microenvironment" spelled out → **TME** (your working .docx may still contain one the PDF does not) |

Everything else (CNS, FFPE, H&E, QC, ML, DL, SMI, UCC, S/N) is used consistently after its
definition. Keep the **first** occurrence of each as "full name (ABBR)"; make every later
occurrence the abbreviation.

---

## 6. Additional issues noticed (beyond the four tasks — worth fixing for a clean submission)

| Page | Issue | Fix |
|---|---|---|
| p23 | "cannot cannot serve as ground-truth labels" — duplicated word | "cannot serve…" |
| ref [22] | Author "azanietz MG" — missing leading letter | "Kazanietz MG" |
| p6 | "downstream processing.Spatial transcriptomic" — missing space | "processing. Spatial" |
| p6 | "downstream analysis.. Although" — double period | "analysis. Although" |
| p6 | "the 21th and the 4st sagittal sections" | "the 21st and the 4th" |
| p5 | "resident brain macrophages cells" — redundant | "resident brain macrophages" |
| p5 | Section heading "Material and Methods" | "Materials and Methods" (conventional) |
| p22 | "it was negatively, driven by cells" | "it was negative, driven by cells" |
| p22 | "reside in the meningeal niche as well as lines with some brain vasculature" — garbled | e.g. "…as well as along some brain vasculature" |
| p22 | Highlighted editorial to-do: "(Show the data for both the spatial distribution of Lyve1 expression and the correlation with Mrc1 expression)." | Resolve (insert the figure/text) or delete the note |
| p24 & p25 | **Two figures both numbered "Figure 10"** (Lyve1; decontX) | Renumber the second to Figure 11 and update in-text refs |
| p7 | "…chosen in slide L34" — missing terminal period | add "." |
| p5 vs p13 | "subclone" (p5) vs "sub-clone" (p13) | pick one — "subclone" |
| p31 | "Graph Modeling of Brain Metastasis — TBD" | placeholder still open |
| p7/p20/p28 | Panel named 3 ways: "Mouse Universal Panel" / "Mouse 1K Universal Cell Characterization (UCC) panel" / "Mouse Universal Cell Characterization panel" | standardize to one name |

---

### If you send me the `.docx`
I will apply all of the above directly (including the double-space pass and the automatic TOC),
and hand back the finished file.
