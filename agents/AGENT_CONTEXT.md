# Shared context for the CosMx research-agent team

**Every agent must read this file first.** It is the single source of truth for
the project, environment, data layout, and findings. Do not re-derive these.

## Project
Mouse brain tumor CosMx spatial transcriptomics (~825k cells, 6 slices, two runs
`L321` and `L34`). Goal: **salvage reliable annotation of microglia / MDM / BAM /
astrocytes** despite unreliable custom lineage-tag probes. Probe-validation
evidence for the vendor is a secondary byproduct, not the primary goal.

Suspected common root cause: **segmentation error** (cells capturing neighbours'
transcripts) → both hard myeloid-subtype separation AND weird probe correlations.

## ENVIRONMENT — read carefully, this WILL bite you
- **Always run Python via conda run**, never the env's python.exe directly:
  `& "C:\Users\chenr\miniconda3\Scripts\conda.exe" run -n thesis_research python <script>`
  Calling `...\envs\thesis_research\python.exe` directly crashes `import numpy`
  with an MKL DLL/PATH fault (exit `-1066598273`). Under `conda run`, numpy 2.4.3,
  BLAS, scanpy 1.12.1, anndata, torch 2.10.0 all work.
- **Prefer the PowerShell tool** for shell commands on this Windows machine. The
  Bash tool mis-resolves conda and grabs `base`'s python.
- **No GPU** (`torch.cuda.is_available()` == False). Cellpose/Baysor/GAT/scVI run
  on CPU — prototype on a few FOVs / one slice before scaling.
- **No R on PATH.** NanoString R tools (fastReseg, insitucor, FOV QC, smiDE) cannot
  run as-is. Use Python equivalents/reimplementations, or flag that R must be
  installed and stop — do not silently skip a required step.

## DATA LAYOUT (verify columns before relying on them)
- Per-slice cached AnnData: `resources/cache/slice_{1..6}_adata.h5ad`
- With negative-control probes (for ambient/background): `resources/cache/slice_{1..6}_adata_with_neg.h5ad`
- With tumor prediction: `resources/cache/with_tumor_prediction/slice_{1..6}_adata.h5ad`
- decontX (slices 1 & 3 only): `resources/cache/decontx/slice_{1,3}_decontx.h5ad`
  — contains raw counts in `.X` + obs `decontx_contamination`, `pred_tumor_XGBoost`,
    `CenterX_global_px`, `CenterY_global_px`. NO decontaminated layer is stored.
- CosMx flat files: `resources/cosmx/{L321,L34}/` — `*_exprMat_file.csv`,
  `*_metadata_file.csv`, `*-polygons.csv`, `*_fov_positions_file.csv`,
  `*_fov_slices.csv`, `*_slice_types.csv`. Read `*_fov_slices.csv` to map FOV→slice.
- **Raw CosMx export for run L321 (slices 1-3) — re-segmentation inputs:**
  Base: `D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\Analysis\L321__1__31_12_2025_12_32_59_204\`
  - `flatFiles\L321\L321_tx_file.csv` — **per-molecule (4.6 GB)**: `fov, cell_ID, cell,
    x_local_px, y_local_px, x_global_px, y_global_px, z, target, CellComp`. cell_ID 0 =
    unassigned. → Baysor + transcript re-assignment. Filter to a few FOVs before loading.
  - `DecodedFiles\L321\20251217_140230_S2\CellStatsDir\FOVxxxxx\` (620 FOVs) —
    `CellLabels_F*.tif` (current segmentation masks), `CompartmentLabels_F*.tif`
    (nuclear/cyto), `CellBoundaries_F*.csv`. `CellStatsDir\CellComposite\*.jpg` =
    morphology composite (lossy JPG; NO raw multichannel Morphology2D TIFFs found).
  - L34 (slices 4-6) raw export path: TBD — ask user.
- Panel: 958 genes + negative probes. 8 custom probes: `Ccl2, Cxcl13, GFAP, Lyve1,
  TMEM119, Trem2, GFP, tdTomato`. `GFAP` is the high-dynamic-range positive control
  (~80 counts/cell).

## COORDINATE GOTCHA
`obsm['spatial']` in the with_tumor_prediction h5ads is **FOV-local**. For tissue
geometry / KD-trees / spatial graphs use `CenterX_global_px` / `CenterY_global_px`
(`thesis_research/utils/columns.py` exposes `CENTER_X_GLOBAL_PX`, `CENTER_Y_GLOBAL_PX`).

## FINDINGS SO FAR (this session, non-tumor cells, slices 1 & 3)
See `agents/check_negative_findings.py`. Threshold sweep splits the three negative
findings into two root causes:
1. **GFP↔tdTomato negative is a LOW-COUNT artifact** — r −0.34→−0.43(≥2)→−0.18(≥3)→
   **+0.13(≥5)**. Flips positive in true double-high cells; tdT+ outnumbers GFP+ ~8:1.
   → ambient/sparse detection; should improve with re-segmentation/decontamination.
2. **GFP↔Cx3cr1 negative is ROBUST and strengthens with count** — r −0.51→−0.71(≥2)→
   −0.63(≥3). → likely a genuine GFP probe defect; re-segmentation probably won't fix it.
3. **Lyve1 = weak-but-real BAM signal in broad spillover** — 2× enriched in
   Mrc1+/Cd163+ (22.7% vs 12.2%) but 90% of Lyve1+ cells lack both; max 8 counts/cell.
Bonus: GFP+ cells carry higher total counts (size/segmentation dependence). Slice 3
has ~50% decontX contamination (vs ~12% slice 1) — treat slice 3 as a QC outlier.

## MARKER-SPECIFICITY STANDARD
Scrutinise every marker for cross-population dilution, activation-state instability,
and tumor co-expression before trusting it. Microglia axis is thin on-panel
(P2ry12/Sall1/Hexb missing); BAM markers strong; drop Apoe from MDM set. Never gate
identity on lineage tags (GFP/tdTomato) alone — use transcriptome markers, lineage
tags only as post-hoc consistency.

## OUTPUT CONVENTIONS
- Write scripts to `agents/<topic>/` and results/plots to `agents/outputs/<topic>/`.
- End every run with a short markdown report: what you ran, key numbers, plots
  produced, and a clear verdict tied to the project goal.
- Use absolute paths in scripts. Set matplotlib `Agg` backend (headless).

## GUARDRAILS
- Do NOT use the Agent tool to spawn further sub-agents.
- Do NOT modify the conda env or delete data without explicit instruction.
- Prototype heavy jobs on one slice / a few FOVs first; report runtime before scaling.
