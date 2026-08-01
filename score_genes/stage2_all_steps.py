"""stage2_all_steps.py - one panel per Stage-2 step:
  1) method: two independent signals must agree
  2) the independent classifier's confidence (agree vs disagree)
  3) agreement filter: provisional -> high-confidence per type
  4) validation: held-out AUC on non-marker genes
  5) result: the validated high-confidence sets (Astrocytes, Ependymal)
  6) conclusion
Reads high_confidence_labels.csv (no h5ad needed).
Output -> score_genes_slice1/storyline/stage2_all_steps.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HC_CSV = "D:/thesis-research/score_genes_slice1/high_confidence_labels.csv"
OUT = "D:/thesis-research/score_genes_slice1/storyline"

COL = {"Astrocytes": "#1f77b4", "Microglia": "#17becf", "Macrophage": "#00a087",
       "Endothelial": "#2ca02c", "Pericytes": "#a65628", "Ependymal": "#984ea3",
       "Neurons": "#e377c2"}
ORDER = ["Microglia", "Macrophage", "Pericytes", "Endothelial",
         "Astrocytes", "Ependymal", "Neurons"]
AUC = {"Astrocytes\nvs Ependymal": 1.00, "Microglia\nvs Macrophage": 0.587,
       "Pericytes\nvs Endothelial": 0.613}
THRESH = 0.80

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})


def as_bool(s):
    return s.astype(str).isin(["True", "1", "1.0", "TRUE", "true"])


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(HC_CSV)
    prov = df["provisional_label"].to_numpy()
    pred = df["predicted_label"].to_numpy()
    proba = df["predicted_proba"].to_numpy()
    hc = as_bool(df["high_conf"]).to_numpy()
    agree = pred == prov
    order = [g for g in ORDER if g in set(prov)]

    prov_n = pd.Series(prov).value_counts()
    hc_n = pd.Series(prov[hc]).value_counts()

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), dpi=150)

    # (1) method schematic
    a = ax[0, 0]; a.set_xlim(0, 10); a.set_ylim(0, 10); a.axis("off")

    def box(xc, yc, w, h, text, fc):
        a.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                    boxstyle="round,pad=0.12", fc=fc, ec="black", lw=1.4))
        a.text(xc, yc, text, ha="center", va="center", fontsize=10.5, weight="bold")

    box(2.3, 7.4, 4.0, 1.8, "Marker label\n(score_genes)", "#cfe8ff")
    box(2.3, 2.6, 4.0, 1.8, "Classifier on 902\nNON-marker genes", "#ffe0cc")
    box(6.4, 5.0, 2.6, 2.0, "AGREE?\nsame label\n& prob ≥ 0.8", "#fff2b3")
    box(9.0, 5.0, 1.8, 2.0, "High-\nconfidence\ncell", "#c9f0c9")
    for ys in (7.4, 2.6):
        a.add_patch(FancyArrowPatch((4.3, ys), (5.15, 5.0), arrowstyle="-|>",
                    mutation_scale=16, lw=1.4, color="k"))
    a.add_patch(FancyArrowPatch((7.7, 5.0), (8.1, 5.0), arrowstyle="-|>",
                mutation_scale=16, lw=1.4, color="k"))
    a.text(5, 0.4, "(classifier trained on a seed of the cleanest cells)",
           ha="center", fontsize=9, style="italic")
    a.set_title("1) Method: keep a cell only if two signals agree")

    # (2) classifier confidence, split by agree/disagree
    a = ax[0, 1]
    bins = np.linspace(0, 1, 26)
    a.hist(proba[agree], bins=bins, color="#2ca02c", alpha=0.75, label="agrees with marker")
    a.hist(proba[~agree], bins=bins, color="#b0b0b0", alpha=0.75, label="disagrees")
    a.axvline(THRESH, color="crimson", ls="--", lw=1.5)
    a.text(THRESH + 0.01, a.get_ylim()[1] * 0.9, "keep ≥ 0.8", color="crimson", fontsize=9)
    a.set_xlabel("classifier probability"); a.set_ylabel("cells")
    a.legend(fontsize=9)
    a.set_title("2) Independent classifier's confidence")

    # (3) agreement filter funnel: provisional -> high-confidence
    a = ax[0, 2]
    xp = np.arange(len(order)); w = 0.4
    a.bar(xp - w / 2, [prov_n.get(g, 0) for g in order], w, color="#cccccc",
          label="marker-called (provisional)")
    a.bar(xp + w / 2, [hc_n.get(g, 0) for g in order], w,
          color=[COL[g] for g in order], label="high-confidence (kept)")
    a.set_xticks(xp); a.set_xticklabels(order, rotation=45, ha="right")
    a.set_ylabel("cells"); a.legend(fontsize=9)
    a.set_title("3) Agreement filter: provisional → kept")

    # (4) validation: held-out AUC on non-marker genes
    a = ax[1, 0]
    bars = list(AUC); av = [AUC[b] for b in bars]
    a.bar(range(len(bars)), av, color=["#2ca02c" if v > 0.8 else "#d62728" for v in av])
    a.axhline(0.5, color="gray", ls="--", lw=1); a.text(len(bars) - 1, 0.52, "chance",
                                                        color="gray", ha="right", fontsize=9)
    a.set_xticks(range(len(bars))); a.set_xticklabels(bars, fontsize=9)
    a.set_ylim(0, 1.05); a.set_ylabel("held-out AUC (non-marker genes)")
    for i, v in enumerate(av):
        a.text(i, v + 0.02, f"{v:.2f}", ha="center", weight="bold")
    a.set_title("4) Validation: are the kept cells really distinct?")

    # (5) result: validated sets
    a = ax[1, 1]
    res = ["Astrocytes", "Ependymal"]
    a.bar(range(len(res)), [int(hc_n.get(g, 0)) for g in res],
          color=[COL[g] for g in res])
    for i, g in enumerate(res):
        a.text(i, int(hc_n.get(g, 0)) + 3, str(int(hc_n.get(g, 0))),
               ha="center", weight="bold")
    a.set_xticks(range(len(res))); a.set_xticklabels(res)
    a.set_ylabel("high-confidence cells")
    a.set_title("5) Result: validated high-confidence sets\n(AUC = 1.0)")

    # (6) conclusion
    a = ax[1, 2]; a.axis("off")
    a.text(0.0, 1.0,
           "Conclusion (Stage 2)\n\n"
           "• Keep a cell only if the marker label AND an\n"
           "  independent classifier (902 non-marker genes)\n"
           "  agree, with probability ≥ 0.8.\n\n"
           "• Validate with held-out AUC on non-marker genes.\n\n"
           "• PASS → Astrocytes (87) & Ependymal (243):\n"
           "  AUC = 1.0, genuinely distinct → clean sets.\n\n"
           "• FAIL → myeloid & vascular (AUC ≈ 0.6): kept as\n"
           "  candidates but NOT validated → handled later.",
           fontsize=11, va="top", ha="left")

    fig.suptitle("Stage 2 — all steps: extracting high-confidence cells",
                 fontsize=17, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{OUT}/stage2_all_steps.png", bbox_inches="tight")
    plt.close()
    print("saved", f"{OUT}/stage2_all_steps.png")


if __name__ == "__main__":
    main()
