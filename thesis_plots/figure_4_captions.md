# Figure 5 (spatial refinement) — text package

Everything to paste into the thesis: where each kind of sentence belongs, the
body paragraph, the six captions, and the table that replaces the number dump
currently sitting under "**Figure 4.**" in `thesis_draft_6.4.md`.

**Numbering.** In `thesis_plots/tables/thesis_table_s1.md` this display item is
**Figure 5**, not Figure 4 — Figure 4 is the classifier comparison from
`figure_3_model_comparison.py`. The six per-slice figures are therefore labelled
**5a–5f**. Lettering rather than 5.1–5.6 keeps Figures 6–11 from shifting by
five.

---

## 1. Division of labour

Three places, three jobs — never say the same thing in two of them:

| Where | Job | Length |
| --- | --- | --- |
| **Body text** | The *argument*: what diverges between models, and why XGBoost was chosen. Cites "Figures 5a–5f" as a set; names a single slice only when that slice carries a point no other slice makes. | one paragraph |
| **Caption** | How to *read* the panel: what is plotted, what the colours mean, then one sentence on what this slice shows. Must stand alone for someone who flips to the page cold. | 4–6 sentences |
| **Table** | The *numbers*. Thirty kept/rejected counts are unreadable as prose. | one table |

The rule that fixes the first PI comment: **the title names what you are looking
at and where; the first sentence says how the plot was built.** If the first
sentence can be deleted without loss, it was restating the title.

---

## 2. Body paragraph (replaces the "On the candidate pool…" paragraph)

> On the candidate pool the five classifiers diverge, but not along the
> linear-versus-tree axis. Averaged over the four tumor-bearing slices,
> retention ranges from 63.4% (LogReg + KNN) to 71.9% (LogReg + PCA), with
> XGBoost at 68.6% (Table X) — the most and the least permissive models are both
> linear. The separation that matters is visible in two other places. First, in
> the sham-injected controls, where every retained candidate is a false positive
> by experimental design: LogReg, Random Forest and XGBoost retain none of the
> 630 control candidates, whereas LogReg + PCA retains 14 and LogReg + KNN 3.
> Second, in the spatial distribution of the rejections within the tumor-bearing
> slices (Figures 5a–5f). All five models agree on a compact, spatially coherent
> lesion core in every tumor-bearing slice, and differ in what they remove at its
> margin and among the sparse candidates scattered through distal tissue; the
> linear models additionally strip cells from inside the dense core, most
> severely in slices 5 and 6, where LogReg + KNN retention falls to 54.9% and
> 50.4%. XGBoost was selected on the conjunction of these two observations: among
> the three models that admit no false positives in the control slices it retains
> the most tumor (68.6%, against 66.7% for LogReg and 65.8% for Random Forest),
> and it does so while preserving the lesion core that the linear models erode.
> Its non-linear decision function and its capacity to model interactions between
> panel genes plausibly account for this, given the transcriptional heterogeneity
> within the tumor mass.

Two things to note about this rewrite:

* The old sentence *"the linear models reject a larger fraction of candidates as
  look-alikes, while the tree-based models retain more"* is contradicted by the
  current numbers — LogReg + PCA is the **most** retentive of all five. The
  paragraph above drops that claim and replaces it with a criterion the
  regenerated figures actually support.
* The selection argument is now falsifiable and readable straight off the
  figures: *zero control false positives first, then maximum tumor retention.*
  That is a stronger defence than "XGBoost looked cleaner".

---

## 3. The six captions

Each one stands alone: someone flipping to the page cold can read the panel
without turning back. The first three sentences are shared word for word — that
repetition is correct in a figure series, not lazy — and only the identifier and
the closing sentence change.

**Figure 5a | SingleR tumor candidates retained and rejected by each
classifier, slice 1 (L321, tumor-bearing).** Every segmented cell of the slice
is plotted in global tissue coordinates, one panel per evaluated model. Red
marks a SingleR tumor candidate that the model retained as tumor, blue a
candidate it rejected as a look-alike, and gray the remaining cells, which were
never in the candidate pool; the panel border is red for tumor-bearing and blue
for sham-injected control slices. The inset gives the retained and rejected
counts and the retention rate, and the five models were trained jointly on the
L321 + L34 reference pool and applied here to candidate cells never seen in
training. All five models recover the same compact, spatially coherent lesion
and disagree almost exclusively on the sparse candidates scattered through
distal tissue, retaining 70.8% (LogReg + KNN) to 77.9% (LogReg + PCA) of the
5,793 candidates.

**Figure 5b | SingleR tumor candidates retained and rejected by each
classifier, slice 2 (L321, tumor-bearing).** Every segmented cell of the slice
is plotted in global tissue coordinates, one panel per evaluated model. Red
marks a SingleR tumor candidate that the model retained as tumor, blue a
candidate it rejected as a look-alike, and gray the remaining cells, which were
never in the candidate pool; the panel border is red for tumor-bearing and blue
for sham-injected control slices. The inset gives the retained and rejected
counts and the retention rate, and the five models were trained jointly on the
L321 + L34 reference pool and applied here to candidate cells never seen in
training. The lesion is denser and more sharply bounded than in slice 1 and
agreement between models is correspondingly higher (77.6–83.2% of 7,650
candidates retained); the rejected candidates form a distinct cluster in
anterior tissue well away from the lesion rather than lining its margin.

**Figure 5c | SingleR tumor candidates retained and rejected by each
classifier, slice 3 (L321, sham-injected control).** Every segmented cell of
the slice is plotted in global tissue coordinates, one panel per evaluated
model. Red marks a SingleR tumor candidate that the model retained as tumor,
blue a candidate it rejected as a look-alike, and gray the remaining cells,
which were never in the candidate pool; the panel border is red for
tumor-bearing and blue for sham-injected control slices. The inset gives the
retained and rejected counts and the retention rate, and the five models were
trained jointly on the L321 + L34 reference pool and applied here to candidate
cells never seen in training. In this tumor-free slice every retained candidate
is a false positive by experimental design: SingleR proposed 343 candidates,
scattered rather than clustered, and LogReg, Random Forest and XGBoost retain
none of them, while LogReg + KNN retains 2 and LogReg + PCA 8.

**Figure 5d | SingleR tumor candidates retained and rejected by each
classifier, slice 4 (L34, sham-injected control).** Every segmented cell of the
slice is plotted in global tissue coordinates, one panel per evaluated model.
Red marks a SingleR tumor candidate that the model retained as tumor, blue a
candidate it rejected as a look-alike, and gray the remaining cells, which were
never in the candidate pool; the panel border is red for tumor-bearing and blue
for sham-injected control slices. The inset gives the retained and rejected
counts and the retention rate, and the five models were trained jointly on the
L321 + L34 reference pool and applied here to candidate cells never seen in
training. As in slice 3 the 287 candidates are dispersed across the section
with no coherent focus and are rejected almost without exception (0–6 retained
across models), confirming that the compact populations retained in the
tumor-bearing slices are not an artefact of the refinement step itself.

**Figure 5e | SingleR tumor candidates retained and rejected by each
classifier, slice 5 (L34, tumor-bearing).** Every segmented cell of the slice
is plotted in global tissue coordinates, one panel per evaluated model. Red
marks a SingleR tumor candidate that the model retained as tumor, blue a
candidate it rejected as a look-alike, and gray the remaining cells, which were
never in the candidate pool; the panel border is red for tumor-bearing and blue
for sham-injected control slices. The inset gives the retained and rejected
counts and the retention rate, and the five models were trained jointly on the
L321 + L34 reference pool and applied here to candidate cells never seen in
training. This slice carries a broad, diffusely infiltrating tumor mass and the
models separate most here, retaining 54.9% (LogReg + KNN) to 65.6% (LogReg +
PCA) of 8,129 candidates; rejected cells are interleaved with retained cells
throughout the mass rather than confined to its margin, so the models differ in
how far they erode the lesion itself.

**Figure 5f | SingleR tumor candidates retained and rejected by each
classifier, slice 6 (L34, tumor-bearing).** Every segmented cell of the slice
is plotted in global tissue coordinates, one panel per evaluated model. Red
marks a SingleR tumor candidate that the model retained as tumor, blue a
candidate it rejected as a look-alike, and gray the remaining cells, which were
never in the candidate pool; the panel border is red for tumor-bearing and blue
for sham-injected control slices. The inset gives the retained and rejected
counts and the retention rate, and the five models were trained jointly on the
L321 + L34 reference pool and applied here to candidate cells never seen in
training. The largest candidate pool (9,296 cells) is also the most heavily
filtered, at 50.4% (LogReg + KNN) to 60.9% (LogReg + PCA) retained; rejections
run throughout the lesion and along a band of tissue at its lateral edge, and
the dense posterior core is the only region all five models agree on.

The closing sentences are my reading of the regenerated panels — check them
against the figures and reword in your own voice before using.

**If you prefer shorter.** Write 5a in full as above and reduce 5b–5f to
`Figure 5x | Slice N (<slide>, <condition>). Details as in Figure 5a.` plus
their closing sentence. One description to maintain instead of six; the cost is
that 5b–5f no longer stand alone.
---

## 5. Table (replaces the bulleted number dump)

**Table X | Tumor candidates retained by each classifier, per slice.** Cells give
the number of SingleR tumor candidates retained as tumor, with the retention rate
in parentheses. Every candidate retained in slices 3 and 4 is a false positive by
experimental design.

| Slice | Type | Candidates | LogReg | LogReg + PCA | LogReg + KNN (PCA) | Random Forest | XGBoost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | L321 (Tumor) | 5,793 | 4,218 (72.8%) | 4,512 (77.9%) | 4,100 (70.8%) | 4,207 (72.6%) | 4,267 (73.7%) |
| 2 | L321 (Tumor) | 7,650 | 6,134 (80.2%) | 6,362 (83.2%) | 5,938 (77.6%) | 6,054 (79.1%) | 6,272 (82.0%) |
| 3 | L321 (Control) | 343 | 0 (0.0%) | 8 (2.3%) | 2 (0.6%) | 0 (0.0%) | 0 (0.0%) |
| 4 | L34 (Control) | 287 | 0 (0.0%) | 6 (2.1%) | 1 (0.3%) | 0 (0.0%) | 0 (0.0%) |
| 5 | L34 (Tumor) | 8,129 | 4,755 (58.5%) | 5,329 (65.6%) | 4,463 (54.9%) | 4,693 (57.7%) | 5,039 (62.0%) |
| 6 | L34 (Tumor) | 9,296 | 5,142 (55.3%) | 5,663 (60.9%) | 4,681 (50.4%) | 5,009 (53.9%) | 5,295 (57.0%) |
| — | *Mean, tumor slices* | — | **66.7%** | **71.9%** | **63.4%** | **65.8%** | **68.6%** |
| — | *Total retained, controls* | — | **0** | **14** | **3** | **0** | **0** |

---

## 6. Consistency fixes needed in `thesis_draft_6.4.md`

Mismatches between the prose and the code as it now stands, not wording
preferences:

1. **Stale numbers.** The per-slice counts under "**Figure 4.**" predate the
   current run (slice 1 LogReg reads 3,901 there, 4,218 now). Replace with the
   table above.
2. **Class balancing.** The Methods sentence describes LogReg "with class
   balancing applied" and XGBoost "with `scale_pos_weight` set to the empirical
   negative-to-positive ratio". Neither is in the code —
   `figure_4_spatial_refinement.py` uses library defaults and states that class
   balancing is deliberately not applied. Delete both clauses.
3. **Decision direction.** The prose says the model returns *P*(healthy) and
   retains cells with *P*(healthy) < 0.5; the code scores *P*(tumor) > 0.5. The
   boundary is identical but the description should match the implementation.
4. **Figure numbering.** The paragraph cites "(Figure 4)" for the spatial
   refinement; per Table S1 that is Figure 5, now 5a–5f.
