"""Create a thesis-ready overview of all per-cell QC metrics.

The figure summarizes the complete pre-QC cell population in each slice using
the vendor metadata. Boxes show the interquartile range, center lines show the
median, and whiskers show the 1st and 99th percentiles. Only total transcript
count was used to exclude cells in the analysis pipeline; the remaining metrics
were evaluated diagnostically.

Run:
    conda run -n thesis_research python thesis_plots/make_qc_metrics_overview_fig.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = "D:/thesis-research"
META = ROOT + "/resources/cosmx/{slide}/{slide}_metadata_file.csv"
FOVS = ROOT + "/resources/cosmx/{slide}/{slide}_fov_slices.csv"
TYPES = ROOT + "/resources/cosmx/{slide}/{slide}_slice_types.csv"
OUT_PNG = ROOT + "/thesis_plots/qc_metrics_overview.png"
OUT_PDF = ROOT + "/thesis_plots/qc_metrics_overview.pdf"
OUT_CSV = ROOT + "/thesis_plots/qc_metrics_summary.csv"

SLIDES = ("L321", "L34")
SLICES = (1, 2, 3, 4, 5, 6)
USECOLS = (
    "fov",
    "nCount_RNA",
    "nFeature_RNA",
    "Area",
    "Area.um2",
    "Circularity",
    "Eccentricity",
    "Perimeter",
)

QUANTILE = 0.05
COUNT_FLOOR = 20

TUMOR = "#4C72B0"
CONTROL = "#9A9A9A"
THRESHOLD = "#D55E00"
INK = "#222222"
MUTED = "#5A5A5A"
SPINE = "#9A9A9A"


def _fov_to_slice(slide: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for row in pd.read_csv(FOVS.format(slide=slide)).itertuples(index=False):
        for fov in range(int(row.start), int(row.end) + 1):
            mapping[fov] = int(row.slice)
    return mapping


def load_metadata() -> pd.DataFrame:
    """Load required pre-QC columns and attach slice/type labels."""
    frames = []
    for slide in SLIDES:
        fov_map = _fov_to_slice(slide)
        type_map = (
            pd.read_csv(TYPES.format(slide=slide))
            .set_index("slice")["type"]
            .to_dict()
        )
        frame = pd.read_csv(META.format(slide=slide), usecols=USECOLS)
        frame["slice"] = frame["fov"].map(fov_map)
        frame = frame.dropna(subset=["slice"]).copy()
        frame["slice"] = frame["slice"].astype(int)
        frame["section_type"] = frame["slice"].map(type_map)
        frame["slide"] = slide

        # Convert the vendor perimeter from pixels to micrometers using the
        # pixel scale encoded by its paired area measurements.
        valid_area = (frame["Area"] > 0) & (frame["Area.um2"] > 0)
        pixel_um = np.sqrt(
            frame.loc[valid_area, "Area.um2"] / frame.loc[valid_area, "Area"]
        ).median()
        frame["Perimeter.um"] = frame["Perimeter"] * pixel_um
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    missing = set(SLICES) - set(data["slice"].unique())
    if missing:
        raise ValueError(f"Missing metadata for slices: {sorted(missing)}")
    return data


def distribution_stats(values: pd.Series, label: str) -> dict[str, object]:
    """Matplotlib bxp statistics computed from every finite cell value."""
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        raise ValueError(f"No finite values for {label}")
    q01, q25, q50, q75, q99 = np.quantile(x, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "label": label,
        "med": q50,
        "q1": q25,
        "q3": q75,
        "whislo": q01,
        "whishi": q99,
        "fliers": [],
    }


def count_threshold(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return float(max(COUNT_FLOOR, np.nanquantile(x, QUANTILE)))


def main() -> None:
    data = load_metadata()
    type_by_slice = (
        data[["slice", "section_type"]].drop_duplicates().set_index("slice")[
            "section_type"
        ]
    )
    labels = [
        f"{sid}\n({'T' if type_by_slice[sid] == 'tumor' else 'C'})"
        for sid in SLICES
    ]

    metrics = [
        ("nCount_RNA", "Total transcripts", "Transcripts per cell", True),
        ("nFeature_RNA", "Detected genes", "Genes detected per cell", False),
        ("Area.um2", "Cell area", "Cell area (µm²)", False),
        ("Circularity", "Circularity", "Circularity", False),
        ("Eccentricity", "Eccentricity", "Eccentricity", False),
        ("Perimeter.um", "Cell perimeter", "Cell perimeter (µm)", False),
    ]

    summary_rows = []
    plot_stats: dict[str, list[dict[str, object]]] = {}
    for column, _, _, _ in metrics:
        plot_stats[column] = []
        for sid, label in zip(SLICES, labels):
            values = data.loc[data["slice"] == sid, column]
            stat = distribution_stats(values, label)
            plot_stats[column].append(stat)
            summary_rows.append(
                {
                    "slice": sid,
                    "section_type": type_by_slice[sid],
                    "metric": column,
                    "n_cells": int(values.notna().sum()),
                    "p01": stat["whislo"],
                    "q25": stat["q1"],
                    "median": stat["med"],
                    "q75": stat["q3"],
                    "p99": stat["whishi"],
                }
            )
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.8), dpi=300)

    for panel_index, (ax, metric) in enumerate(zip(axes.ravel(), metrics)):
        column, title, ylabel, log_scale = metric
        for position, (sid, stat) in enumerate(
            zip(SLICES, plot_stats[column]), start=1
        ):
            color = TUMOR if type_by_slice[sid] == "tumor" else CONTROL
            artists = ax.bxp(
                [stat],
                positions=[position],
                widths=0.62,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.0},
                medianprops={"color": "white", "linewidth": 1.5},
                whiskerprops={"color": color, "linewidth": 1.0},
                capprops={"color": color, "linewidth": 1.0},
            )
            for box in artists["boxes"]:
                box.set_alpha(0.92)

        ax.text(
            -0.13,
            1.06,
            "ABCDEF"[panel_index],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", pad=7)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_xticks(range(1, len(SLICES) + 1), labels)
        ax.set_xlabel("Slice", fontsize=9.5)
        ax.tick_params(labelsize=8.5, length=3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(SPINE)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        if log_scale:
            ax.set_yscale("log")
            thresholds = [
                count_threshold(data.loc[data["slice"] == sid, column])
                for sid in SLICES
            ]
            ax.scatter(
                range(1, len(SLICES) + 1),
                thresholds,
                marker="_",
                s=260,
                linewidths=2.1,
                color=THRESHOLD,
                zorder=5,
            )
        else:
            lower = min(float(s["whislo"]) for s in plot_stats[column])
            upper = max(float(s["whishi"]) for s in plot_stats[column])
            margin = max((upper - lower) * 0.06, 0.01)
            ax.set_ylim(lower - margin, upper + margin)

    handles = [
        Line2D([0], [0], color=TUMOR, linewidth=7, label="tumor-bearing"),
        Line2D([0], [0], color=CONTROL, linewidth=7, label="control"),
        Line2D(
            [0],
            [0],
            color=THRESHOLD,
            marker="_",
            linestyle="None",
            markersize=11,
            markeredgewidth=2,
            label="applied low-count threshold (panel A)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.012,
        "Boxes: interquartile range; center line: median; whiskers: 1st–99th percentiles. "
        "Morphology and gene-complexity metrics were diagnostic only.",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.94), w_pad=2.1, h_pad=2.1)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    thresholds = {
        sid: count_threshold(data.loc[data["slice"] == sid, "nCount_RNA"])
        for sid in SLICES
    }
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_CSV}")
    print(
        "Low-count thresholds: "
        + ", ".join(f"slice {sid}={thresholds[sid]:.0f}" for sid in SLICES)
    )
    print(
        "\nCaption:\n"
        "Figure X | Per-cell quality-control metrics across the six tissue "
        "sections before filtering. Boxes show the interquartile range, white "
        "lines the median, and whiskers the 1st and 99th percentiles of the full "
        "per-cell distributions. (A) Total detected transcripts on a logarithmic "
        "scale; orange markers show the slice-specific low-count thresholds "
        "applied during QC. (B) Number of genes detected per cell. (C-F) "
        "Morphological properties derived from the vendor segmentation masks: "
        "cell area, circularity, eccentricity, and perimeter. Gene complexity "
        "and morphology were evaluated diagnostically but were not used as "
        "additional exclusion criteria. T, tumor-bearing; C, control."
    )


if __name__ == "__main__":
    main()
