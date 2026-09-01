"""Render CSV results as Nature-style tables (HTML for Word, Markdown for drafts).

Nature / Nature Communications table conventions implemented here:
  * Title sits ABOVE the table: bold "Table N | <sentence-case title>", no terminal period.
  * Horizontal rules only -- top, under the header, bottom. No vertical rules, no shading.
  * Body rows carry no interior rules; group headers get a spanning row instead.
  * Footnotes sit BELOW the table, smaller: general note first (units, n, statistical
    test), then superscript-letter notes, then abbreviation definitions.
  * Any bold/italic in the data must be explained in the footnote.
  * Tables are never subdivided into "Table 1a"/"Table 1b" -- use Table 1, Table 2.

Usage:
    python make_nature_tables.py            # rebuilds every table defined in main()
Open the .html in a browser and copy-paste into Word; Word keeps the rules and fonts.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

OUTDIR = Path(__file__).resolve().parent / "tables"

# Journal figure font is ~7-8pt; 9pt reads better in a bound thesis.
CSS = """
body   { background:#fff; margin:32px; }
.tbl-wrap { font-family:Arial, Helvetica, sans-serif; color:#000;
            max-width:1100px; margin-bottom:36px; }
.tbl-title { font-size:9.5pt; font-weight:bold; margin-bottom:6pt; line-height:1.35; }
table  { border-collapse:collapse; font-size:9pt; line-height:1.3; }
th, td { padding:3.5pt 10pt 3.5pt 0; border:none; vertical-align:bottom;
         text-align:left; white-space:nowrap; }
thead th        { font-weight:bold; border-top:1.4pt solid #000;
                  border-bottom:0.7pt solid #000; }
thead th.spangroup { text-align:center; padding-bottom:1.5pt;
                     border-bottom:0.5pt solid #000; border-top:1.4pt solid #000; }
thead tr.sub th { border-top:none; }
tbody tr.last td { border-bottom:1.4pt solid #000; }
tbody tr.grouphdr td { font-style:italic; padding-top:7pt; }
tbody tr.ruled td { border-top:0.7pt solid #000; }
td.num, th.num  { text-align:right; }
.tbl-foot { font-size:8pt; line-height:1.45; margin-top:5pt; max-width:900px;
            white-space:normal; }
sup { font-size:7pt; }
code { font-family:"Courier New", Courier, monospace; font-size:8.5pt; }
/* Fixed layout is the only reliable way to hold column widths: in the default
   auto layout a td ignores max-width, and long unbroken paths overflow into the
   neighbouring column. overflow-wrap lets those paths break mid-token. */
.tbl-wrap.wide { max-width:1180px; }
table.fixed { table-layout:fixed; width:100%; }
table.fixed th, table.fixed td { white-space:normal; overflow-wrap:anywhere;
                                 word-break:break-word; vertical-align:top;
                                 padding-right:12pt; }
table.fixed thead th { vertical-align:bottom; }
td.wrap { white-space:normal; overflow-wrap:anywhere; }
"""


_INLINE_TAGS = ("i", "b", "sub", "sup", "code")


def _rich(text: str) -> str:
    """Escape user text but keep the inline tags Nature tables legitimately need."""
    out = html.escape(str(text))
    for tag in _INLINE_TAGS:
        out = out.replace(f"&lt;{tag}&gt;", f"<{tag}>").replace(f"&lt;/{tag}&gt;", f"</{tag}>")
    return out


def _cell(value, numeric: bool, bold: bool, wrap: bool = False) -> str:
    text = _rich("" if value is None else value)
    if bold:
        text = f"<b>{text}</b>"
    klass = " ".join(k for k, on in (("num", numeric), ("wrap", wrap)) if on)
    return f'<td class="{klass}">{text}</td>'


def nature_table_html(
    df: pd.DataFrame,
    number: int | str,
    title: str,
    footnotes: list[str],
    numeric_cols: list[str] | None = None,
    bold_cells: dict[int, list[str]] | None = None,
    group_rows: dict[int, str] | None = None,
    col_groups: list[tuple[str, int]] | None = None,
    rule_before: set[int] | None = None,
    wrap_cols: list[str] | None = None,
    col_widths: list[str] | None = None,
) -> str:
    """Build one Nature-style table.

    bold_cells:  {row_index: [column names to bold]} -- must be justified in a footnote.
    group_rows:  {row_index: "label"}                -- italic spanning row inserted above
                                                        that row (e.g. "Custom probes").
    col_groups:  [("", 3), ("Detection", 4)]         -- optional spanning header tier.
    rule_before: {row_index}                         -- thin rule above the row, for a
                                                        summary/total row.
    col_widths:  ["10%", "26%", ...]                 -- switches the table to fixed
                                                        layout so wide text columns hold
                                                        their width instead of colliding.
    """
    numeric_cols = numeric_cols or []
    bold_cells = bold_cells or {}
    group_rows = group_rows or {}
    rule_before = rule_before or set()
    wrap_cols = wrap_cols or []
    ncol = len(df.columns)

    header_cells = "".join(
        f'<th class="{"num" if c in numeric_cols else ""}">{_rich(c)}</th>'
        for c in df.columns
    )
    head = []
    if col_groups:
        cells = "".join(
            f'<th class="spangroup" colspan="{span}">{_rich(label)}</th>'
            if label else f'<th colspan="{span}"></th>'
            for label, span in col_groups
        )
        head.append(f"<tr>{cells}</tr>")
        head.append(f'<tr class="sub">{header_cells}</tr>')
    else:
        head.append(f"<tr>{header_cells}</tr>")

    body, last = [], len(df) - 1
    for i, (_, row) in enumerate(df.iterrows()):
        if i in group_rows:
            body.append(
                f'<tr class="grouphdr"><td colspan="{ncol}">'
                f"{_rich(group_rows[i])}</td></tr>"
            )
        cells = "".join(
            _cell(row[c], c in numeric_cols, c in bold_cells.get(i, []), c in wrap_cols)
            for c in df.columns
        )
        klass = " ".join(
            k for k, on in (("last", i == last), ("ruled", i in rule_before)) if on
        )
        body.append(f'<tr class="{klass}">{cells}</tr>')

    foot = "".join(f"<div>{f}</div>" for f in footnotes)
    if col_widths:
        if len(col_widths) != ncol:
            raise ValueError(f"col_widths has {len(col_widths)} entries for {ncol} columns")
        cols = "".join(f'<col style="width:{w}">' for w in col_widths)
        colgroup, table_cls, wrap_cls = f"<colgroup>{cols}</colgroup>", ' class="fixed"', " wide"
    else:
        colgroup, table_cls, wrap_cls = "", "", ""
    return (
        f'<div class="tbl-wrap{wrap_cls}">\n'
        f'  <div class="tbl-title">Table {number} | {_rich(title)}</div>\n'
        f'  <table{table_cls}>\n    {colgroup}\n    <thead>{"".join(head)}</thead>\n'
        f'    <tbody>{"".join(body)}</tbody>\n  </table>\n'
        f'  <div class="tbl-foot">{foot}</div>\n'
        "</div>\n"
    )


def _md(text) -> str:
    """Inline HTML -> Markdown equivalents (GitHub renders <sub>/<sup>, not all viewers do)."""
    s = str(text)
    for a, b in (("<i>", "*"), ("</i>", "*"), ("<b>", "**"), ("</b>", "**"),
                 ("<code>", "`"), ("</code>", "`"),
                 ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")):
        s = s.replace(a, b)
    return s


def nature_table_markdown(
    df: pd.DataFrame,
    number: int | str,
    title: str,
    footnotes: list[str],
    bold_cells: dict[int, list[str]] | None = None,
    group_rows: dict[int, str] | None = None,
) -> str:
    """Same table as Markdown, for the thesis .md drafts."""
    bold_cells = bold_cells or {}
    group_rows = group_rows or {}
    cols = list(df.columns)
    lines = [
        f"**Table {number} | {_md(title)}**",
        "",
        "| " + " | ".join(_md(c) for c in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for i, (_, row) in enumerate(df.iterrows()):
        if i in group_rows:
            first = f"*{group_rows[i]}*"
            lines.append("| " + " | ".join([first] + [""] * (len(cols) - 1)) + " |")
        cells = [
            f"**{_md(row[c])}**" if c in bold_cells.get(i, []) else _md(row[c])
            for c in cols
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", *(_md(f) for f in footnotes)]
    return "\n".join(lines) + "\n"


def write(name: str, tables_html: list[str]) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{''.join(tables_html)}</body></html>",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# Table 2 -- probe detection reliability (all six slices)
# --------------------------------------------------------------------------- #
ROLES = {
    "GFAP": "Positive control (astrocyte)",
    "tdTomato": "Lineage reporter",
    "Ccl2": "Chemokine",
    "TMEM119": "Microglia",
    "Lyve1": "BAM",
    "Trem2": "Myeloid",
    "Cxcl13": "Chemokine",
    "GFP": "Lineage reporter",
    "Meg3": "Reference (neuronal)",
    "Pecam1": "Reference (endothelial)",
    "Cx3cr1": "Reference (myeloid)",
    "Csf1r": "Reference (myeloid)",
}
# The specificity index equals 1 - 1/(S/N) exactly, so as a column it re-encodes S/N
# rather than adding information. Set True to restore it.
INCLUDE_SPECIFICITY = False


def _mean_range(s, decimals=1):
    return "{v:.{d}f} [{lo:.{d}f}-{hi:.{d}f}]".format(
        v=s.mean(), lo=s.min(), hi=s.max(), d=decimals)


def build_probe_table(csv: Path):
    """Per-probe mean [min-max] across the six slices; custom probes, then references."""
    d = pd.read_csv(csv)
    agg = (d.groupby(["probe", "cls"])
             .agg(SN=("SN", _mean_range), cnt=("mean", lambda s: _mean_range(s, 3)),
                  spec=("above_bg", lambda s: _mean_range(s / 100, 2)),
                  sn_mean=("SN", "mean"))
             .reset_index())
    agg = pd.concat([agg[agg.cls == c].sort_values("sn_mean", ascending=False)
                     for c in ("custom", "panel")], ignore_index=True)
    # One measured column: S/N is what the acceptance criterion is applied to. A
    # negative-control row is omitted deliberately -- its S/N is 1 by construction,
    # so it would carry no information; the noise floor is given in the footnote.
    out = pd.DataFrame({
        "Probe": agg["probe"],
        "Role": agg["probe"].map(ROLES),
        "S/N": agg["SN"],
    })
    if INCLUDE_SPECIFICITY:
        out["Detection specificity"] = agg["spec"]
    gfp = int(agg.index[agg["probe"] == "GFP"][0])
    first_panel = int(agg.index[agg["cls"] == "panel"][0])
    return out, {
        "numeric_cols": [c for c in out.columns if c not in ("Probe", "Role")],
        "bold_cells": {gfp: list(out.columns)},
        "group_rows": {0: "Custom add-on probes",
                       first_panel: "Panel reference genes"},
    }


# --------------------------------------------------------------------------- #
# Table 1 -- tissue sections profiled
# --------------------------------------------------------------------------- #
SECTIONS = [
    # slice, mouse, slide, type, fovs, area_um2, cells_initial, removed_low_tx
    (1, "2", "L321", "T", 257, 17_811_397, 140_615, 15_659),
    (2, "3", "L321", "T", 225, 16_710_072, 137_669, 11_184),
    (3, "1", "L321", "C", 138, 8_260_389, 69_007, 5_010),
    (4, "1", "L34", "C", 198, 9_833_831, 76_631, 19_545),
    (5, "3", "L34", "T", 205, 23_726_062, 221_554, 15_344),
    (6, "2", "L34", "T", 182, 29_265_570, 280_842, 13_469),
]
EMDASH = "—"

_COLS = ["slice", "mouse", "slide", "type", "fovs", "area", "initial", "removed"]


def _sections() -> pd.DataFrame:
    """Section metadata with retained counts and retention derived, not hardcoded."""
    d = pd.DataFrame(SECTIONS, columns=_COLS)
    d["final"] = d["initial"] - d["removed"]
    d["retention"] = 100 * d["final"] / d["initial"]
    return d


def _with_total(out: pd.DataFrame, d: pd.DataFrame, sums: dict) -> pd.DataFrame:
    """Append a Total row; text columns get an em dash."""
    row = {c: EMDASH for c in out.columns}
    row["Slice ID"] = "Total"
    row.update(sums)
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)


def build_merged_table():
    """Sections + QC filtering in one table (they share four of their columns)."""
    d = _sections()
    out = pd.DataFrame({
        "Slice ID": d["slice"].astype(str),
        "Mouse ID": d["mouse"],
        "Slide ID": d["slide"],
        "Slice type": d["type"],
        "FOVs (n)": d["fovs"].map("{:,}".format),
        "Tissue area (μm²)": d["area"].map("{:,}".format),
        "Initial (n)": d["initial"].map("{:,}".format),
        "Removed (n)": d["removed"].map("{:,}".format),
        "Retained (n)": d["final"].map("{:,}".format),
        "Retention (%)": d["retention"].map("{:.2f}".format),
    })
    tot_ret = 100 * d["final"].sum() / d["initial"].sum()
    out = _with_total(out, d, {
        "FOVs (n)": f"{d['fovs'].sum():,}",
        "Tissue area (μm²)": f"{d['area'].sum():,}",
        "Initial (n)": f"{d['initial'].sum():,}",
        "Removed (n)": f"{d['removed'].sum():,}",
        "Retained (n)": f"{d['final'].sum():,}",
        "Retention (%)": f"{tot_ret:.2f}",
    })
    return out, {
        "numeric_cols": ["FOVs (n)", "Tissue area (μm²)", "Initial (n)",
                         "Removed (n)", "Retained (n)", "Retention (%)"],
        "rule_before": {len(d)},
        "col_groups": [("", 4), ("Acquisition", 2), ("Cells after low-transcript QC", 4)],
    }


def build_section_table():
    """Sections only -- the split alternative to build_merged_table()."""
    d = _sections()
    out = pd.DataFrame({
        "Slice ID": d["slice"].astype(str),
        "Mouse ID": d["mouse"],
        "Slide ID": d["slide"],
        "Slice type": d["type"],
        "Segmented cells (n)": d["initial"].map("{:,}".format),
        "FOVs (n)": d["fovs"].map("{:,}".format),
        "Tissue area (μm²)": d["area"].map("{:,}".format),
    })
    out = _with_total(out, d, {
        "Segmented cells (n)": f"{d['initial'].sum():,}",
        "FOVs (n)": f"{d['fovs'].sum():,}",
        "Tissue area (μm²)": f"{d['area'].sum():,}",
    })
    return out, {
        "numeric_cols": ["Segmented cells (n)", "FOVs (n)", "Tissue area (μm²)"],
        "rule_before": {len(d)},
    }


def build_qc_table():
    """Low-transcript QC filtering only -- the split alternative."""
    d = _sections()
    out = pd.DataFrame({
        "Slice ID": d["slice"].astype(str),
        "Mouse ID": d["mouse"],
        "Slide ID": d["slide"],
        "Slice type": d["type"],
        "Initial cells (n)": d["initial"].map("{:,}".format),
        "Removed, low transcripts (n)": d["removed"].map("{:,}".format),
        "Final cells (n)": d["final"].map("{:,}".format),
        "Retention (%)": d["retention"].map("{:.2f}".format),
    })
    tot_ret = 100 * d["final"].sum() / d["initial"].sum()
    out = _with_total(out, d, {
        "Initial cells (n)": f"{d['initial'].sum():,}",
        "Removed, low transcripts (n)": f"{d['removed'].sum():,}",
        "Final cells (n)": f"{d['final'].sum():,}",
        "Retention (%)": f"{tot_ret:.2f}",
    })
    return out, {
        "numeric_cols": ["Initial cells (n)", "Removed, low transcripts (n)",
                         "Final cells (n)", "Retention (%)"],
        "rule_before": {len(d)},
    }


# --------------------------------------------------------------------------- #
# Table 3 -- tumor-refinement pipeline parameters
# --------------------------------------------------------------------------- #
# (stage | None, parameter, value, role). A stage string opens an italic group row;
# None continues the current group. Names in <code> are literal code arguments.
PIPELINE = [
    ("Stage 1 — SingleR candidate filter",
     "<code>score_tumor</code> floor", "> 0.2", "Permissive tumor-candidate inclusion"),
    (None, "<code>delta_score</code> margin", "> 0.08",
     "Separation from the next-best reference label"),
    (None, "Best-class consistency",
     "<code>score_tumor</code> > next-best score", "Tumor score must be the top score"),

    ("Stage 1 — SingleR tumor reference",
     "<code>score_tumor</code> floor", "≥ 0.4", "Stricter positive reference set"),
    (None, "<code>delta_score</code> margin", "> 0.08", "Confident tumor anchors"),

    ("Stage 2 — Classifier comparison",
     "Source slices", "Slice 3 (L321), slice 4 (L34)",
     "Control-slice false-positive tumor calls"),
    (None, "Filters", "As Stage 1 candidate filter", "Learns the SingleR failure mode"),
    (None, "Cross-validation folds", "5", "Out-of-fold model evaluation"),
    (None, "Random seed", "42", "Reproducibility"),

    ("Stage 3 — XGBoost refinement",
     "Operating threshold", "<i>P</i>(healthy) < 0.5", "Retained as refined tumor"),
    (None, "Hyperparameters", "xgboost 3.2.0 defaults", "No tuning performed"),
    (None, "Random seed", "42", "Reproducibility"),
]


def build_pipeline_table():
    """Stage names become italic group rows instead of a repeated leading column."""
    rows, groups, current = [], {}, None
    for stage, param, value, role in PIPELINE:
        if stage and stage != current:
            groups[len(rows)] = stage
            current = stage
        rows.append({"Parameter": param, "Value": value, "Role": role})
    return pd.DataFrame(rows), {"group_rows": groups, "wrap_cols": ["Role"]}


# --------------------------------------------------------------------------- #
# Supplementary Table S1 -- display item -> script provenance
# --------------------------------------------------------------------------- #
# (section | None, display item, analysis, script, output). A section string opens
# an italic group row. Superscript markers flag gaps to close before submission --
# they describe the current state of the repo, not the intended end state.
PROVENANCE = [
    ("Chapter 2 — Quality control",
     "Figure 1", "Segmented-cell area and FOV layout per slide",
     "<code>thesis_research/pipeline/position_plots.py</code> "
     "(<code>plot_cells_positions_with_area</code>, <code>plot_fov_positions_all_slices</code>), "
     "driven by <code>run_pipeline.py</code>",
     "<code>cell_positions.png</code>, <code>fov_positions_sliced.png</code><sup>b</sup>"),
    (None, "Figure 2", "Per-cell transcript-count distributions and the low-count threshold",
     "<code>thesis_plots/make_qc_count_threshold_fig.py</code>, applying the cutoff "
     "rule of <code>thesis_research/pipeline/cell_qc_plots.py</code> "
     "(<code>_low_count_flag</code>)",
     "<code>qc_count_threshold_hist.png</code>"),
    (None, "Table 1", "Slices profiled and cell yield after low-transcript QC",
     "<code>cell_qc_plots.py</code> (counts); "
     "<code>thesis_plots/make_nature_tables.py</code> (layout)",
     "<code>tables/thesis_tables.html</code><sup>c</sup>"),

    ("Chapter 2 — Tumor-cell identification",
     "—", "Reference-based annotation with SingleR",
     "<code>outputs/cell_annotation/annotate.R</code>, "
     "<code>convert_annotations_to_df.R</code>", "SingleR score CSVs"),
    (None, "Figure 3", "Initial SingleR tumor candidates across six slices",
     "<code>thesis_plots/figure_1_singler_calls.py</code><sup>d</sup>",
     "<code>figure_1_singler_calls.png/.csv</code>"),
    (None, "Figure 4", "Classifier comparison on the joint reference pool",
     "<code>thesis_plots/figure_3_model_comparison.py</code><sup>d</sup>",
     "<code>figure_3_model_comparison.png/.csv</code>"),
    (None, "Figure 5", "Spatial refinement of candidates across classifiers",
     "<code>thesis_plots/figure_4_spatial_refinement.py</code><sup>d</sup>",
     "<code>figure_4_spatial_refinement.png/.csv</code>"),
    (None, "Figure 6", "Final XGBoost tumor calls in space, all six sections",
     "<code>thesis_plots/make_dq_fig_tumor_spatial.py</code>; calls written by "
     "<code>thesis_plots/figure_4_spatial_refinement.py</code><sup>d</sup>",
     "<code>dq_fig_tumor_spatial.png</code>; "
     "<code>figure_4_spatial_refinement.csv</code> (XGBoost rows)"),
    (None, "Table 3", "Parameter settings of the tumor-refinement pipeline",
     "<code>…/tumor_cells/identify_tumor_cells.py</code>, "
     "<code>refine_annotation_classifiers.py</code> (values); "
     "<code>make_nature_tables.py</code> (layout)",
     "<code>tables/thesis_tables.html</code><sup>c</sup>"),

    ("Chapter 3 — Data-quality assessment",
     "Figure 7", "Probe detection against the per-slice acceptance threshold, "
     "all six slices",
     "<code>thesis_plots/make_dq_fig1_detection.py</code>",
     "<code>dq_fig1_detection.png</code>, <code>acceptance_bar_all6.csv</code>"),
    (None, "Table 2", "Detection strength of the eight custom add-on probes, "
     "all six slices",
     "<code>thesis_plots/make_detection_reliability_6slice.py</code> (values); "
     "<code>thesis_plots/make_nature_tables.py</code> (layout)",
     "<code>detection_reliability_all6.csv</code>; "
     "<code>tables/thesis_tables.html</code>"),
    (None, "—", "Field-of-view morphology and count QC, slide L3-21",
     "<code>fov_qc_slice1.py</code>",
     "<code>fov_qc/slice1_fov_qc.csv</code>, "
     "<code>slice1_fov_qc_map.png</code>, <code>slice1_fov_metrics.png</code>"),
    (None, "Figure 9<sup>a</sup>",
     "tdTomato prevalence against the pan-myeloid transcripts Cx3cr1 and Csf1r, "
     "and the control-to-tumor contrast per slide",
     "<code>thesis_plots/make_dq_fig_reporter.py</code>",
     "<code>dq_fig_reporter.png</code>"),
    (None, "Figure 10<sup>a</sup>",
     "Lyve1-positive cells against the canonical BAM markers Mrc1 and Cd163, "
     "in composition and in space",
     "<code>thesis_plots/make_dq_fig_lyve1.py</code>",
     "<code>dq_fig_lyve1.png</code>"),
    (None, "—", "Coarse Leiden partitions supplied to decontX as the cluster prior",
     "<code>score_genes/write_decontx_clusters.py</code>",
     "<code>resources/cache/decontx/slice_&lt;n&gt;_work/clusters.csv</code><sup>e</sup>"),
    (None, "—", "Ambient-RNA estimation and correction (decontX)",
     "<code>score_genes/run_decontx_correct.py</code> (export and assemble); "
     "<code>score_genes/run_decontx.R</code> (decontX itself)<sup>f</sup>",
     "<code>resources/cache/decontx/slice_{1..6}_decontx.h5ad</code>, "
     "<code>decontx_contamination.csv</code>"),
    (None, "Figure 11<sup>a</sup>",
     "Probe positivity before and after ambient correction, all six sections",
     "<code>thesis_plots/make_dq_fig_decontx.py</code>",
     "<code>dq_fig_decontx.png</code>, <code>decontx_before_after.csv</code>"),
    (None, "—", "Neighbour-profile transcript reassignment and its parameter sweep",
     "<code>agents/segmentation/03_filter_tx.py</code>, "
     "<code>04_reseg_reassign.py</code>, <code>05_prior_sweep.py</code>; "
     "purity metric in <code>agents/segmentation/seg_metrics.py</code>",
     "per-FOV before/after marker-purity tables"),
    (None, "Figure 12<sup>a</sup>",
     "Myeloid marker purity before and after reassignment, and under sweeps of "
     "its three parameters",
     "<code>thesis_plots/make_dq_fig_reassign.py</code><sup>g</sup>",
     "<code>dq_fig_reassign.png</code>"),
    (None, "—", "Manufacturer's technical assessment of the run",
     "not script-generated; supplied by Bruker Spatial Biology",
     "<code>resources/vendor/bruker_technical_assessment.md</code>"),
]


def build_s1_table():
    rows, groups, current = [], {}, None
    for section, item, analysis, script, output in PROVENANCE:
        if section and section != current:
            groups[len(rows)] = section
            current = section
        rows.append({"Display item": item, "Analysis": analysis,
                     "Script": script, "Primary output": output})
    return pd.DataFrame(rows), {
        "group_rows": groups,
        "col_widths": ["9%", "25%", "42%", "24%"],
    }


def main() -> None:
    root = Path(__file__).resolve().parent

    design_note = (
        "Both control sections derive from a single animal (mouse 1); the four "
        "tumor-bearing sections derive from mice 2 and 3."
    )
    abbrev = ("C, sham-injected control; FFPE, formalin-fixed paraffin-embedded; "
              "FOV, field of view; T, tumor-bearing.")
    qc_note = (
        "Initial counts are cells returned by the vendor segmentation pipeline, before "
        "filtering. Removed cells are those falling below the per-slice transcript-count "
        "threshold; retention is the retained fraction of the initial count."
    )

    merged_df, merged_kw = build_merged_table()
    t1_html = nature_table_html(merged_df, 1,
        "Tissue sections profiled by CosMx spatial transcriptomic imaging and their "
        "cell yield after low-transcript quality control",
        ["Each slice derives from one FFPE section. Tissue area is the summed area of all "
         "FOVs acquired for that slice. " + qc_note, design_note, abbrev],
        **merged_kw)

    probe_df, probe_kw = build_probe_table(root / "detection_reliability_all6.csv")
    probe_title = ("Detection strength of the eight custom add-on probes across all six "
                   "slices, benchmarked against panel reference genes")
    probe_foot = [
        "All six slices, non-tumor cells only (<i>n</i> = 825,428 cells). Values are the "
        "mean across slices with the [min-max] range; per-slice values are shown in "
        "Figure 7a. Signal-to-noise (S/N) is the mean per-cell probe count divided by the "
        "mean per-cell count of the 11 negative-control probes, which have no target in "
        "the mouse transcriptome and whose mean defines the noise floor of each slice "
        "(0.018-0.025 counts per cell). The "
        "SystemControl targets are barcodes to which no probe is assigned and report on "
        "optical decoding alone, so they were excluded from the background estimate.",
        "A probe was accepted as detecting its target only where its S/N exceeded three in "
        "every slice; GFP is the only probe not meeting this criterion.",
        "Values in bold identify the GFP probe (see Results).",
        "BAM, border-associated macrophage; S/N, signal-to-noise.",
    ]
    t2_html = nature_table_html(probe_df, 2, probe_title, probe_foot, **probe_kw)

    model = pd.read_csv(root / "figure_3_model_comparison.csv")
    model_df = pd.DataFrame({
        "Model": model["model"],
        "Accuracy": model["accuracy"].map("{:.3f}".format),
        "Precision": model["precision"].map("{:.3f}".format),
        "Recall": model["recall"].map("{:.3f}".format),
        "F<sub>1</sub>": model["f1"].map("{:.3f}".format),
        "AUROC": model["roc_auc"].map("{:.3f}".format),
        "AP": model["ap"].map("{:.3f}".format),
    })
    model_foot = [
        "Held-out test-set performance of each classifier. Metrics are reported to three "
        "decimal places; the best value in each column is not marked because differences "
        "between the top models fall within the resampling interval.",
        "AP, average precision; AUROC, area under the receiver operating characteristic "
        "curve; F<sub>1</sub>, harmonic mean of precision and recall; KNN, k-nearest "
        "neighbors; LogReg, logistic regression; PCA, principal component analysis.",
    ]
    pipe_df, pipe_kw = build_pipeline_table()
    pipe_foot = [
        "Parameters of the three-stage tumor-refinement pipeline. Stage 1 defines SingleR "
        "tumor candidates and a stricter tumor reference set; Stage 2 compares classifiers "
        "trained on the two control sections, where any tumor call is by definition a false "
        "positive; Stage 3 applies the selected XGBoost model to all sections.",
        "Parameters set in monospace are the literal argument names used in the analysis "
        "code. <code>score_tumor</code> is the SingleR per-cell tumor score and "
        "<code>delta_score</code> its margin over the next-best reference label.",
        "No hyperparameter tuning was performed: the Stage-3 classifier uses the xgboost "
        "3.2.0 defaults (<code>n_estimators</code> = 100, <code>max_depth</code> = 6, "
        "<code>learning_rate</code> = 0.3, <code>subsample</code> = 1.0, "
        "<code>colsample_bytree</code> = 1.0, <code>reg_lambda</code> = 1.0, "
        "<code>scale_pos_weight</code> = 1). Defaults are stated explicitly because they "
        "are release-specific. A leave-one-out sensitivity analysis over these parameters, "
        "under the same 5-fold cross-validation, found no setting distinguishable from its "
        "default (out-of-fold accuracy 0.968–0.973 across nine configurations, against a "
        "standard error of 0.004 at <i>n</i> = 1,493 reference cells), and the refined "
        "tumor set differed by 0.25% between the tuned and default configurations, with "
        "both control sections yielding zero tumor calls throughout.",
    ]
    t3_html = nature_table_html(pipe_df, 3,
        "Parameter settings of the three-stage tumor-refinement pipeline",
        pipe_foot, **pipe_kw)

    t4_html = nature_table_html(model_df, 4,
        "Classifier performance for myeloid cell-type assignment", model_foot,
        numeric_cols=["Accuracy", "Precision", "Recall", "F<sub>1</sub>", "AUROC", "AP"])

    path = write("thesis_tables", [t1_html, t2_html, t3_html, t4_html])
    print(f"wrote {path}")

    s1_df, s1_kw = build_s1_table()
    s1_foot = [
        "Every figure, table, and analysis step in this thesis and the script that "
        "produced it. Script paths are relative to the repository root. Running the "
        "scripts in the order given in the repository README regenerates each output "
        "from the deposited processed data. Scripts hardcode the repository root as "
        "<code>D:/thesis-research</code> and require editing to run from another "
        "location.",
        "<sup>a</sup>Chapter 3 figure numbers follow the current draft. Figure 8 of "
        "the earlier draft (GFP-tdTomato correlation per slice) was withdrawn, because "
        "conditioning on positivity for either probe induces a negative correlation "
        "between any two sparse targets; if the final draft carries no Figure 8, "
        "renumber Figures 9-12 accordingly.",
        "<sup>b</sup>Written to <code>outputs/&lt;run_id&gt;/&lt;sample_id&gt;/</code>, where "
        "<code>run_id</code> is a fresh UUID generated on each pipeline run "
        "(<code>run_pipeline.py</code>), so these files have no stable path.",
        "<sup>c</sup>Values are currently hardcoded in <code>make_nature_tables.py</code> "
        "rather than read from the analysis outputs.",
        "<sup>d</sup>The script filename numbering predates the thesis figure numbering "
        "and does not match it.",
        "<sup>e</sup>The partitions are a decontX input, not an output, and are not "
        "regenerated deterministically by a re-run; they are deposited with the "
        "processed data so the correction can be reproduced exactly.",
        "<sup>f</sup>decontX is run from R (celda). The Python driver exports the count "
        "matrix and cluster prior, calls the R script, and assembles the corrected "
        "matrix; corrected counts are rounded to integers on assembly.",
        "<sup>g</sup>Panel values are the measured results transcribed into the plotting "
        "script rather than read from the sweep outputs at render time.",
        "BAM, border-associated macrophage; FOV, field of view; MDM, monocyte-derived "
        "macrophage; QC, quality control.",
    ]
    s1_title = "Analysis scripts underlying each figure, table, and analysis step"
    s1 = write("thesis_table_s1",
               [nature_table_html(s1_df, "S1", s1_title, s1_foot, **s1_kw)])
    print(f"wrote {s1}")

    (OUTDIR / "thesis_table_s1.md").write_text(
        nature_table_markdown(s1_df, "S1", s1_title, s1_foot,
                              group_rows=s1_kw["group_rows"]),
        encoding="utf-8",
    )

    # Alternative layout: the same content split into two narrower tables.
    sect_df, sect_kw = build_section_table()
    qc_df, qc_kw = build_qc_table()
    alt = write("thesis_tables_split", [
        nature_table_html(sect_df, 1,
            "Overview of tissue sections profiled by CosMx spatial transcriptomic imaging",
            ["Each slice derives from one FFPE section. Segmented cell counts are as "
             "returned by the vendor pipeline, before low-transcript filtering (Table 2). "
             "Tissue area is the summed area of all FOVs acquired for that slice.",
             design_note, abbrev],
            **sect_kw),
        nature_table_html(qc_df, 2,
            "Per-slice cell counts before and after low-transcript quality control",
            [qc_note, design_note, abbrev], **qc_kw),
    ])
    print(f"wrote {alt}")

    md = OUTDIR / "thesis_tables.md"
    md.write_text(
        nature_table_markdown(merged_df, 1,
            "Tissue sections profiled by CosMx spatial transcriptomic imaging and their "
            "cell yield after low-transcript quality control",
            ["Each slice derives from one FFPE section. Tissue area is the summed area of "
             "all FOVs acquired for that slice. " + qc_note, design_note, abbrev])
        + "\n"
        + nature_table_markdown(probe_df, 2, probe_title, probe_foot,
            bold_cells=probe_kw["bold_cells"], group_rows=probe_kw["group_rows"])
        + "\n"
        + nature_table_markdown(pipe_df, 3,
            "Parameter settings of the three-stage tumor-refinement pipeline",
            pipe_foot, group_rows=pipe_kw["group_rows"])
        + "\n"
        + nature_table_markdown(model_df, 4,
            "Classifier performance for myeloid cell-type assignment", model_foot),
        encoding="utf-8",
    )
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
