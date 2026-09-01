"""Per-cell transcript-count distributions and the low-count QC threshold, all six
slices. Reproduces the QC cutoff applied in
``thesis_research/pipeline/cell_qc_plots.py`` (``_low_count_flag``: threshold =
max(20, 5th percentile of nCount_RNA)) from the raw vendor metadata, i.e. before
any filtering.

No figure-level title: journals carry the title in the caption, not in the image.
The script prints a caption to paste into the manuscript.

Run: conda run -n thesis_research python thesis_plots/make_qc_count_threshold_fig.py
"""
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = "D:/thesis-research"
META = ROOT + "/resources/cosmx/{sl}/{sl}_metadata_file.csv"
FOVS = ROOT + "/resources/cosmx/{sl}/{sl}_fov_slices.csv"
TYPES = ROOT + "/resources/cosmx/{sl}/{sl}_slice_types.csv"
OUT = ROOT + "/thesis_plots/qc_count_threshold_hist.png"

SLIDES = ["L321", "L34"]
SLICES = [1, 2, 3, 4, 5, 6]
QUANTILE, CAP = 0.05, 20          # must match cell_qc_plots._low_count_flag
BINS = 100

BAR = "#4C72B0"
LINE = "#D55E00"
INK = "#222222"
MUTED = "#5A5A5A"

# Expected values from Table 1 (make_nature_tables.SECTIONS) -- a regression check
# that the raw metadata still reproduces the published cell yields.
EXPECTED = {1: (140_615, 15_659), 2: (137_669, 11_184), 3: (69_007, 5_010),
            4: (76_631, 19_545), 5: (221_554, 15_344), 6: (280_842, 13_469)}
TYPE_LABEL = {"tumor": "tumor-bearing", "healthy": "control"}


def load_counts() -> dict[int, tuple[np.ndarray, str]]:
    """Per-slice raw nCount_RNA plus the '<slide>, <type>' panel descriptor."""
    out = {}
    for slide in SLIDES:
        fov2slice = {}
        for _, r in pd.read_csv(FOVS.format(sl=slide)).iterrows():
            for f in range(int(r["start"]), int(r["end"]) + 1):
                fov2slice[f] = int(r["slice"])
        types = pd.read_csv(TYPES.format(sl=slide)).set_index("slice")["type"].to_dict()

        md = pd.read_csv(META.format(sl=slide), usecols=["fov", "nCount_RNA"])
        md["slice"] = md["fov"].map(fov2slice)
        for sid, grp in md.dropna(subset=["slice"]).groupby("slice"):
            label = f"{slide}, {TYPE_LABEL[types[int(sid)]]}"
            out[int(sid)] = (grp["nCount_RNA"].to_numpy(dtype=float), label)
    return out


def low_count_threshold(counts: np.ndarray) -> float:
    return max(CAP, np.nanquantile(counts, QUANTILE))


def main() -> None:
    data = load_counts()

    stats = {}
    for sid in SLICES:
        counts, label = data[sid]
        thr = low_count_threshold(counts)
        removed = int((counts < thr).sum())
        stats[sid] = dict(n=len(counts), thr=thr, removed=removed,
                          pct=100 * removed / len(counts), label=label)
        exp_n, exp_rm = EXPECTED[sid]
        flag = "" if (len(counts), removed) == (exp_n, exp_rm) else \
            f"  <-- MISMATCH vs Table 1 (expected n={exp_n:,}, removed={exp_rm:,})"
        print(f"slice {sid}: n={len(counts):,}  threshold={thr:.0f}  "
              f"removed={removed:,} ({stats[sid]['pct']:.1f}%){flag}")

    lo = np.log2(min(c[c > 0].min() for c, _ in data.values()))
    hi = np.log2(max(c.max() for c, _ in data.values()))
    edges = np.linspace(lo, hi, BINS + 1)

    # Both axes shared so panels are on one common scale. The y axis is raw cell
    # count, so panel height also reflects section size (n ranges from 69,007 to
    # 280,842) -- slices 3 and 4 are correspondingly short.
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.4), dpi=300,
                             sharex=True, sharey=True)
    for i, (sid, ax) in enumerate(zip(SLICES, axes.ravel())):
        counts, _ = data[sid]
        s = stats[sid]
        ax.hist(np.log2(counts[counts > 0]), bins=edges, color=BAR, linewidth=0)
        ax.axvline(np.log2(s["thr"]), color=LINE, linewidth=1.4)

        ax.text(-0.16, 1.06, "ABCDEF"[i], transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=INK, va="bottom", ha="left")
        ax.set_title(f"Slice {sid} — {s['label']}", fontsize=9.5, color=INK,
                     loc="left", pad=6)
        ax.text(0.97, 0.94, f"{s['pct']:.1f}% removed", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, ha="right", va="top")

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9A9A9A")
        # sharex/sharey hide inner tick labels by default; the panels sit far enough
        # apart that every one needs its own readable x and y scale.
        ax.tick_params(labelsize=8.5, colors=MUTED, length=3,
                       labelbottom=True, labelleft=True)
        if i % 3 == 0:
            ax.set_ylabel("Number of cells", fontsize=9.5, color=INK)
        if i >= 3:
            ax.set_xlabel("log$_2$ total transcripts per cell", fontsize=9.5, color=INK)

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {OUT}")

    thrs = {sid: int(round(stats[sid]["thr"])) for sid in SLICES}
    common = max(set(thrs.values()), key=list(thrs.values()).count)
    odd = [f"slice {s} ({thrs[s]} counts, log2 = {np.log2(thrs[s]):.2f})"
           for s in SLICES if thrs[s] != common]
    thr_sentence = (
        f"The red line marks the low-count threshold, {common} transcripts "
        f"(log2 = {np.log2(common):.2f})"
        + (f", except {'; '.join(odd)}" if odd else "") + ".")
    print("\n--- caption ---\nFigure X. Per-cell transcript-count distributions and the "
          "low-count quality-control threshold. (A-F) Histograms of log2-transformed "
          "total transcripts per cell for each of the six tissue sections, before "
          "filtering. Panels share common x and y axes, so panel height also "
          "reflects the size of each section (n = 69,007-280,842 cells; Table 1). "
          + thr_sentence + " The threshold is the greater of 20 "
          "transcripts and the 5th percentile of the per-slice count distribution; "
          "the percentage of cells falling below it is given in each panel.")


if __name__ == "__main__":
    main()
