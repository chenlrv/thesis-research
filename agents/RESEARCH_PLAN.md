# CosMx research-agent team — plan & how to run

## The team
Launchable definitions live in `.claude/agents/` (the only place the Claude Code
harness discovers subagents). All support files live under `agents/`.

| Agent | Attacks | Key methods |
|-------|---------|-------------|
| `segmentation-qc` | Is segmentation the common root cause? | Baysor vs Cellpose vs current, on 1 slice; marker purity + probe-correlation recheck |
| `probe-validation` | Reliability of the 8 custom probes | S/N vs negative-probe background across 6 slices; ambient vs sensitivity vs defect |
| `representation-learning` | Robust myeloid embedding | scVI / PCA / scGPT; purity + batch mixing |
| `annotation-benchmark` | Which annotation to trust | positive-gating vs InSituType vs GAT; agreement + disagreement diagnostics |
| `lit-scout` | Reading list | curated, annotated spatial-Tx methods papers |

All agents read `agents/AGENT_CONTEXT.md` first (env rules, data paths, findings).

## How to launch (this is the "spin up a team" answer)
These are native Claude Code subagents — no external repo/framework needed. After the
harness picks up the `.claude/agents/*.md` files (new session or reload), launch them
by name, e.g.:
- "Use the **segmentation-qc** agent to run the decisive re-segmentation test on slice 1."
- "Have **probe-validation** build the 8-probe S/N table across all slices."
- Run several in parallel by asking for multiple in one message.
Until reloaded, the same briefs can be driven via the generic general-purpose agent.

## Pipeline order (tracks feed each other — not independent silos)
1. `segmentation-qc` re-segments one slice → cleaner cell×gene matrix.
2. `probe-validation` rechecks the correlations on re-segmented cells → **decisive test**:
   correlations shrink ⇒ segmentation cause (fixable); persist ⇒ probe defect (vendor).
3. `representation-learning` + `annotation-benchmark` run on the cleaned data.
4. `lit-scout` runs anytime, in parallel.

## Raw data for the re-segmentation track — LOCATED (L321 = slices 1-3)
Base: `D:\20251214_CosMx_ReuvenStein\20251214_CosMx_ReuvenStein.tar\Analysis\L321__1__31_12_2025_12_32_59_204\`
- `flatFiles\L321\L321_tx_file.csv` (4.6 GB, per-molecule) → Baysor + transcript re-assignment
- `DecodedFiles\L321\20251217_140230_S2\CellStatsDir\` (620 FOVs) → CellLabels masks + CellComposite JPGs
Only JPG composites exist (no raw Morphology2D TIFFs) → Baysor is primary, Cellpose is cross-check.
Tooling gap: Baysor (Julia) and Cellpose are NOT installed — segmentation-qc installs them first.
L34 (slices 4-6) raw export path still TBD.

## Environment (see AGENT_CONTEXT.md)
`conda run -n thesis_research python ...` (never the raw env python.exe). CPU only. No R.

## Findings so far (slices 1 & 3)
GFP↔tdTomato negative = low-count artifact (flips +0.13 at ≥5); GFP↔Cx3cr1 negative =
robust probe defect; Lyve1 = weak-real BAM signal in broad spillover; GFP tracks total
counts; slice 3 ~50% contaminated. See `agents/check_negative_findings.py`.
