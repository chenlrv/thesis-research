"""Figure DQ4 -- transcript reassignment does not improve myeloid marker purity.

(a) Marker purity before and after reassignment, per field of view: the change is
    small and inconsistent in direction.
(b) Purity under sweeps of each of the procedure's three parameters. Purity stays
    near the baseline across every setting that keeps transcripts moving only
    between neighbouring cells, and rises only when the spatial penalty is
    removed altogether -- a configuration in which the procedure sorts transcripts
    into whichever profile fits them, which is what purity measures.

Values are those reported in the text (canonical configuration: candidate radius
14.4 um, decay length 3.6 um, current-cell advantage 1.65-fold, two rounds).
Sweeps are measured on FOV 514.

Run: conda run -n thesis_research python thesis_plots/make_dq_fig_reassign.py
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

OUT = "D:/thesis-research/thesis_plots/dq_fig_reassign.png"
GREY, BLUE, RED = "#9A9A9A", "#0072B2", "#D62728"

# (a) per-FOV purity, canonical configuration
FOVS = [451, 512, 514, 515, 523]
BEFORE = [0.839, 0.807, 0.781, 0.803, 0.812]
AFTER = [0.833, 0.827, 0.801, 0.820, 0.798]

# (b) sweeps on FOV 514; baseline for that field is 0.781
BASE_514 = 0.781
RADIUS = ([7.2, 14.4, 24.1], [0.783, 0.801, 0.802])
ADVANTAGE = ([0, 1, 2, 3, 4], [0.780, 0.780, 0.786, 0.801, 0.816])
ADV_LABELS = ["none\n(no move)", "20x", "4.5x", "1.65x", "1x\n(no adv.)"]
# categorical spacing: the final point is "no spatial penalty", not a length
DECAY = ([0, 1, 2, 3, 4], [0.792, 0.801, 0.821, 0.845, 0.880])

fig = plt.figure(figsize=(13.4, 4.6), dpi=200)
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1], wspace=0.34)

# ---- (a) before / after per field ----------------------------------------------
axA = fig.add_subplot(gs[0, 0])
for f, b, a in zip(FOVS, BEFORE, AFTER):
    color = BLUE if a > b else RED
    axA.plot([0, 1], [b, a], "-o", color=color, ms=5, lw=1.6)
    axA.text(1.06, a, str(f), va="center", fontsize=7.5, color=color)
axA.plot([0, 1], [np.mean(BEFORE), np.mean(AFTER)], "-o", color="#111", ms=7, lw=2.6,
         zorder=5)
axA.text(1.06, np.mean(AFTER) - 0.004, "mean", va="center", fontsize=8,
         fontweight="bold")
axA.set_xlim(-0.25, 1.45)
axA.set_xticks([0, 1])
axA.set_xticklabels(["before", "after"], fontsize=9.5)
axA.set_ylabel("myeloid marker purity")
axA.set_ylim(0.75, 0.87)
axA.spines[["top", "right"]].set_visible(False)
axA.text(-0.02, 1.06, "(a)", transform=axA.transAxes, fontsize=13,
         fontweight="bold", va="top", ha="right")

# ---- (b) parameter sweeps -------------------------------------------------------
panels = [
    (RADIUS, "candidate radius (µm)", [7.2, 14.4, 24.1], None, 14.4),
    (ADVANTAGE, "current-cell advantage", ADVANTAGE[0], ADV_LABELS, 3),
    (DECAY, "spatial decay length (µm)", DECAY[0],
     ["1.8", "3.6", "7.2", "14.4", "no penalty"], 1),

]
for i, ((xs, ys), xlabel, ticks, ticklabels, canon) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i + 1])
    delta = [y - BASE_514 for y in ys]          # effect of reassignment at that setting
    ax.plot(xs, delta, "-o", color="#333", ms=5, lw=1.6, zorder=3)
    ax.axhline(0, ls="--", lw=1.2, color=GREY, zorder=1)
    j = list(xs).index(canon)
    ax.plot([canon], [delta[j]], "o", ms=10, mfc="none", mec=BLUE, mew=2, zorder=4)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels if ticklabels else [str(t) for t in ticks], fontsize=8)
    ax.set_ylim(-0.012, 0.115)
    ax.spines[["top", "right"]].set_visible(False)
    if i == 0:
        ax.set_ylabel("change in marker purity after reassignment"
                      " (FOV 514)", fontsize=9)
        ax.text(-0.03, 1.06, "(b)", transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")
        ax.text(0.04, 0.004, "no change", fontsize=7.5, color="#666",
                transform=ax.get_yaxis_transform(), va="bottom")
    else:
        ax.tick_params(axis="y", labelsize=8)

# annotate the unconstrained setting on the decay panel
axD = fig.axes[-1]
axD.annotate("no spatial\nconstraint", xy=(28.8, 0.880), xytext=(16, 0.888),
             fontsize=7.5, color=RED, ha="center",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1))
fig.text(0.5, -0.02, "circled point: configuration used for the reported result",
         ha="center", fontsize=8, color="#444")

fig.savefig(OUT, bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved:", OUT)
