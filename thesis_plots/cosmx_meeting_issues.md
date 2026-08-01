# CosMx Custom Panel — Findings Requiring Validation

**Project:** Mouse brain tumor spatial transcriptomics, classifying myeloid (microglia / MDM / BAM) and glial populations.

**Custom probes added to panel (8):** `Ccl2`, `Cxcl13`, `GFAP`, `Lyve1`, `TMEM119`, `Trem2`, `GFP`, `tdTomato`.

**Dataset scope:** 6 slices, ~825k cells total. Analyses below restricted to non-tumor cells unless noted. All counts are raw (pre-normalization).

---

## Positive control: GFAP confirms the platform works on this tissue

GFAP (custom probe) reaches **~80 raw counts per cell** in single non-tumor cells across all 6 slices, with the expected astrocyte-like spatial distribution. ~13–18% of non-tumor cells are GFAP+.

**Why this matters:** GFAP establishes that the platform, this tissue, and at least one of the custom probes are capable of high dynamic range. So issues with the other custom probes below cannot be attributed to platform sensitivity, fixation, or sample quality.

---

## Finding 1 — GFP and tdTomato are negatively correlated (both custom probes)

**Observation.** In non-tumor cells expressing ≥1 count of either, Pearson r is consistently negative:

| Slice | n     | r      |
|-------|-------|--------|
| 1     | 28,214 | -0.333 |
| 2     | 27,667 | -0.347 |
| 3     | 15,216 | -0.330 |

**Expected.** Positive correlation. Both transgenes label monocyte-derived macrophages in the lineage-tracing system used; co-expression should be the norm in MDMs.

**Observed structure.** The vast majority of positive cells are single-tag (GFP+ tdTomato−, or vice versa). Co-positive cells are rare.

**Ruled out.** Holds across all 3 slices examined; raw counts (not a normalization artifact); restricted to non-tumor cells (not tumor cross-reactivity); both probes are custom, so the issue cannot be a stock-panel artifact.

**Validation asks.**
- Sequence and design rationale for both probe sets.
- On-target / off-target QC for GFP and tdTomato probes.
- Whether either probe shows cross-reactivity with endogenous mouse transcripts.
- Whether co-detection of two transgene reporters in the same cell has been benchmarked.

**Workaround (current).** Treat transgene reporters as noisy/secondary evidence only; do not gate cell identity on GFP or tdTomato alone.

---

## Finding 2 — GFP vs Cx3cr1 negatively correlated (custom × stock)

**Observation.** Pearson r ≈ -0.5 in non-tumor cells across slices:

| Slice | n     | r      |
|-------|-------|--------|
| 1     | 21,031 | -0.497 |
| 2     | 19,347 | -0.530 |
| 3     | 8,557  | -0.567 |

**Expected.** Positive correlation. Both should mark myeloid cells (Cx3cr1 broadly, GFP per lineage tracer).

**Observed structure.** Most cells sit at GFP=0 or Cx3cr1=0; co-expression is rare. Cells with GFP ≥ 2 essentially never show Cx3cr1 > 1.

**Ruled out.** Consistent across all slices, raw counts, non-tumor filter applied.

**Possible explanations to validate.**
- GFP probe sensitivity issue (Cx3cr1 may be the more reliable marker — see Finding 1).
- Mutual exclusion artifact from segmentation (one signal per cell boundary?).
- True biology: GFP+ population is a subset distinct from Cx3cr1+ — but this contradicts the lineage-tracer design.

**Validation asks.** Cx3cr1 is a stock-panel gene with established QC. Can CosMx confirm the GFP probe was validated against a reference cell population also expressing Cx3cr1?

**Workaround (current).** Use Cx3cr1 (stock probe with high confidence) over GFP when one must be picked. Flag GFP-derived classifications as low-confidence.

---

## Finding 3 — tdTomato contradicts TMEM119-based microglia/MDM separation

**Observation.** Defining cell types by canonical markers:
- Microglia: GFP+ TMEM119+
- MDMs: GFP+ TMEM119−

Fraction of cells with tdTomato > 0:

| Slice | Microglia (n)        | MDMs (n)             |
|-------|----------------------|----------------------|
| 1     | **43.4%** (n=1,012)  | 30.7% (n=3,362)      |
| 2     | **41.8%** (n=910)    | 29.2% (n=3,644)      |
| 3     | **39.9%** (n=386)    | 29.5% (n=1,924)      |

**Expected.** MDMs > Microglia. tdTomato is the MDM lineage tag in this system; microglia should be largely negative.

**Observed.** Microglia consistently show ~10–13 percentage points **more** tdTomato positivity than MDMs. Direction is reversed, reproducible across all 3 slices, with adequate n.

**This is the strongest single contradiction in the dataset.**

**Ruled out.** Cross-slice consistency rules out per-section technical artifacts. Independence from GFP-defined gate: the issue is in the tdTomato signal itself, not the GFP-based pre-gate (both are wrong in the same direction).

**Validation asks.**
- tdTomato probe sequence and validation against a tissue with known tdTomato+ / tdTomato− populations.
- Whether the tdTomato probe could cross-hybridize with endogenous mouse transcripts highly expressed in microglia.
- Whether ambient RNA / probe diffusion could deposit tdTomato signal in adjacent cells.

**Workaround (current).** Do not use tdTomato as a microglia/MDM separator. Rely on canonical transcriptome-level markers (e.g. TMEM119, P2RY12 for microglia; Ccr2, Plac8, Ly6c2 for MDMs).

---

## Finding 4 — Lyve1 is too sparse and low-dynamic-range to serve as a BAM marker

**Observation.** Across 6 slices, only ~5–15% of non-tumor cells show Lyve1 ≥ 1, and the maximum per-cell count is ~10.

| Slice | Lyve1+ / total non-tumor |
|-------|--------------------------|
| 1     | 15,550 / 120,708 (12.9%) |
| 2     | 13,813 / 120,242 (11.5%) |
| 3     | 6,333 / 63,997 (9.9%)    |
| 4     | 4,358 / 57,086 (7.6%)    |
| 5     | 29,175 / 201,229 (14.5%) |
| 6     | 38,064 / 262,166 (14.5%) |

**Expected.** Lyve1 is a strong perivascular / BAM marker with high per-cell expression in positive cells. Should reach tens of counts in true positives, similar to other tissue-resident macrophage markers.

**Observed.** Dynamic range capped at ~10 counts. Compare to GFAP on the same panel reaching ~80 counts — a >5× gap that is not biologically expected between two cell-type-specific markers.

**Validation asks.**
- Per-probe sensitivity QC: how does Lyve1 compare to GFAP and to stock-panel cell-type markers in the same run?
- Validation against IHC or another assay in mouse brain or peripheral tissue.

**Workaround (current).** Cannot identify BAMs from Lyve1 alone. Requires multi-marker signatures (e.g., Lyve1 + Cd163 + Mrc1 spatial co-occurrence), which limits sensitivity.

---

## Cross-cutting question for the meeting

**Of the 8 custom probes, GFAP reaches ~80 counts/cell; GFP, tdTomato, Lyve1, and Cx3cr1-paired comparisons all cap at 4–14 counts.**

This is a ~5–10× dynamic-range gap between custom probes on the same panel, same tissue, same run.

**Ask:** Can CosMx provide the side-by-side validation data (per-probe sensitivity, on/off-target ratio, signal-to-background) for all 8 custom probes? If certain probes underperformed validation thresholds, what is the remediation pathway (re-design, replacement, credit toward a future run)?

---

## Summary of current workarounds (so analysis can continue)

| Issue                             | Workaround in pipeline                                                                 |
|-----------------------------------|----------------------------------------------------------------------------------------|
| GFP unreliable                    | Treat GFP+ as weak prior, not a gate. Confirm with transcriptomic myeloid signature.   |
| tdTomato unreliable               | Do not use to separate microglia from MDMs. Use TMEM119 / P2RY12 vs Ccr2 / Plac8.      |
| Cx3cr1 vs GFP contradiction       | Prefer Cx3cr1 (stock probe). Document GFP-derived calls as low-confidence.             |
| Lyve1 sparsity                    | Combine with Cd163 / Mrc1 and spatial perivascular location for BAM identification.    |
| General lineage-tag distrust      | All cell-type calls go through transcriptome-based clustering + canonical markers; lineage tags are checked only as post-hoc consistency, never as ground truth. |

---

## What success from this meeting looks like

Pick the outcome you want before walking in:

1. **Probe validation data** for the 8 custom probes — minimum acceptable outcome.
2. **Acknowledgement of underperforming probes** + written explanation of root cause.
3. **Re-design / replacement** of the worst-performing custom probes for future runs.
4. **Refund or credit** toward a replacement panel or rerun.
