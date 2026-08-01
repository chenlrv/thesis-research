"""Thesis-quality figures for the Data-Quality Assessment section.

Regenerates a clean, colorblind-safe (Okabe-Ito) figure set from the validated
probe-validation metrics (agents/outputs/probe_validation/metrics_slice1.json) plus
a light recompute of the tdTomato micro/MDM paradox from the slice h5ads.

Run:  <conda> run -n thesis_research python thesis_plots/make_dq_figures.py
Outputs -> thesis_plots/dq_fig{1,2,3}_*.png
"""
import json
import os

import h5py
import matplotlib as mpl
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

mpl.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICS = os.path.join(ROOT, "agents", "outputs", "probe_validation", "metrics_slice1.json")
WTP = os.path.join(ROOT, "resources", "cache", "with_tumor_prediction", "slice_{}_adata.h5ad")

# ---- Okabe-Ito, assigned by semantic role -----------------------------------
GREEN = "#009E73"   # GFAP positive control
RED = "#D55E00"     # GFP / failed / anomalous
ORANGE = "#E69F00"  # tdTomato
BLUE = "#0072B2"    # other custom probes
GREY = "#9A9A9A"    # panel reference genes (recessive)
SKY = "#56B4E9"     # secondary

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.8, "axes.axisbelow": True,
    "grid.color": "#e8e8e8", "grid.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#222222", "axes.labelcolor": "#222222", "font.family": "DejaVu Sans",
})


def probe_color(name):
    if name == "GFAP":
        return GREEN
    if name == "GFP":
        return RED
    if name == "tdTomato":
        return ORANGE
    if name in ("Cx3cr1", "Pecam1", "Meg3", "Csf1r"):
        return GREY
    return BLUE


def label_bars(ax, bars, vals, fmt="{:.2f}", dy=0.0, fs=8, color="#222222"):
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, b.get_height() + dy),
                    ha="center", va="bottom", fontsize=fs, color=color)


# ----------------------------- data loaders ----------------------------------
def _decode(a):
    if getattr(a, "dtype", None) is not None and a.dtype.kind in ("O", "S"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    return a


def _read_cols(path, genes):
    """Return {gene: dense 1D counts} + non-tumor mask, reading X once."""
    with h5py.File(path, "r") as h5:
        node = h5["X"]
        if isinstance(node, h5py.Group):
            enc = str(node.attrs.get("encoding-type", ""))
            shape = tuple(node.attrs["shape"])
            M = (csc_matrix if "csc" in enc else csr_matrix)(
                (node["data"][...], node["indices"][...], node["indptr"][...]), shape=shape)
        else:
            M = csr_matrix(node[...])
        var = h5["var"]
        vkey = var.attrs.get("_index", "_index")
        vkey = vkey.decode() if isinstance(vkey, bytes) else vkey
        names = _decode(var[vkey][...]).astype(str)
        low = np.char.lower(names)
        # non-tumor mask
        tnode = h5["obs"]["pred_tumor_XGBoost"]
        if isinstance(tnode, h5py.Group):
            codes = tnode["codes"][...]
            cats = _decode(tnode["categories"][...]).astype(str)
            vals = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "False")
            tumor = np.isin(vals, ["True", "1", "1.0", "TRUE", "true"])
        else:
            arr = tnode[...]
            tumor = (np.isin(_decode(arr).astype(str), ["True", "1", "1.0", "TRUE", "true"])
                     if arr.dtype.kind in ("S", "O") else arr.astype(bool))
    Mc = M.tocsc()
    out = {}
    for g in genes:
        idx = np.where(low == g.lower())[0]
        out[g] = np.asarray(Mc[:, idx[0]].todense()).ravel() if len(idx) else None
    return out, ~tumor


def tdt_micro_mdm(slices=(1, 2, 3)):
    """% tdTomato+ among reporter-defined microglia (GFP+TMEM119+) vs MDM (GFP+TMEM119-)."""
    res = {}
    for s in slices:
        p = WTP.format(s)
        if not os.path.exists(p):
            continue
        cols, nt = _read_cols(p, ["GFP", "TMEM119", "tdTomato"])
        if any(cols[g] is None for g in ("GFP", "TMEM119", "tdTomato")):
            continue
        gfp, tmem, tdt = cols["GFP"][nt], cols["TMEM119"][nt], cols["tdTomato"][nt]
        micro = (gfp > 0) & (tmem > 0)
        mdm = (gfp > 0) & (tmem == 0)
        res[s] = {
            "micro_pct": 100 * (tdt[micro] > 0).mean() if micro.sum() else np.nan,
            "mdm_pct": 100 * (tdt[mdm] > 0).mean() if mdm.sum() else np.nan,
            "n_micro": int(micro.sum()), "n_mdm": int(mdm.sum()),
        }
        del cols
    return res


# ================================ FIGURES ====================================
with open(METRICS) as f:
    M = json.load(f)
sn = {r["probe"]: r for r in M["sn_table"]}
order = ["Ccl2", "Cxcl13", "GFAP", "Lyve1", "TMEM119", "Trem2", "GFP", "tdTomato",
         "Cx3cr1", "Pecam1", "Meg3", "Csf1r"]


def fig1_detection_reliability():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (a) signal-to-background, log scale
    ax = axes[0]
    vals = [sn[p]["S_to_N"] for p in order]
    cols = [probe_color(p) for p in order]
    bars = ax.bar(order, vals, color=cols, width=0.72)
    ax.set_yscale("log")
    ax.axhline(1, ls="--", lw=1, color="#444444")
    ax.text(len(order) - 0.5, 1.06, "background (S/N = 1)", ha="right", va="bottom",
            fontsize=7.5, color="#444444")
    ax.set_ylabel("Signal-to-background (S/N)")
    ax.set_title("(a) Detection strength")
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v * 1.05),
                    ha="center", va="bottom", fontsize=7.2, color="#222222")

    # (b) fraction of signal above background (specificity)
    ax = axes[1]
    vals = [sn[p]["frac_signal_above_bg"] for p in order]
    bars = ax.bar(order, vals, color=cols, width=0.72)
    ax.axhline(0.5, ls=":", lw=1, color="#888888")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fraction of signal above background")
    ax.set_title("(b) Detection specificity")
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y")
    label_bars(ax, bars, vals, "{:.2f}", dy=0.01, fs=7.2)

    # (c) GFP positivity vs total-count decile
    ax = axes[2]
    dec = M["ambient"]["gfp_rate_by_total_decile"]
    x = np.arange(1, len(dec) + 1)
    y = [d[1] for d in dec]
    mids = [d[0] for d in dec]
    ax.plot(x, y, "-o", color=RED, lw=2, ms=6, mfc=RED, mec="white", mew=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(m)}" for m in mids], rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("Total counts per cell (decile midpoint)")
    ax.set_ylabel("% cells GFP-positive")
    ax.set_title("(c) GFP tracks cell depth (ambient signature)")
    ax.grid(axis="y")

    fig.suptitle("Custom-probe detection reliability — slice 1, non-tumor cells (n = 120,708)",
                 fontsize=12, fontweight="bold", y=1.02)
    out = os.path.join(HERE, "dq_fig1_detection_reliability.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig2_reporter_inconsistency(td):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) reporter correlations (wrong sign)
    ax = axes[0]
    pairs = ["GFP – Cx3cr1\n(both resident myeloid)", "GFP – tdTomato\n(both MDM by design)"]
    rvals = [M["findings"]["F2_GFP_Cx3cr1"]["full_r"], M["findings"]["F1_GFP_tdTomato"]["full_r"]]
    bars = ax.bar(pairs, rvals, color=RED, width=0.55)
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_ylim(min(rvals) - 0.12, 0.18)
    ax.set_ylabel("Pearson r")
    ax.set_title("(a) Reporters anticorrelate")
    ax.text(0.5, 0.12, "expected  r > 0", transform=ax.transData, ha="center",
            fontsize=8.5, style="italic", color="#2e7d32")
    for b, v in zip(bars, rvals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v - 0.02),
                    ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    ax.set_xticklabels(pairs, fontsize=8)
    ax.grid(axis="y")

    # (b) tdTomato% in microglia vs MDM (per slice) — the backwards result
    ax = axes[1]
    slices = sorted(td.keys())
    xs = np.arange(len(slices))
    w = 0.38
    micro = [td[s]["micro_pct"] for s in slices]
    mdm = [td[s]["mdm_pct"] for s in slices]
    b1 = ax.bar(xs - w / 2, micro, w, color=SKY, label="Microglia (GFP+TMEM119+)")
    b2 = ax.bar(xs + w / 2, mdm, w, color=BLUE, label="MDM (GFP+TMEM119−)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Slice {s}" for s in slices])
    ax.set_ylabel("% tdTomato-positive")
    ax.set_title("(b) tdTomato higher in microglia than MDM")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    label_bars(ax, b1, micro, "{:.0f}%", dy=0.6, fs=8)
    label_bars(ax, b2, mdm, "{:.0f}%", dy=0.6, fs=8)
    ax.set_ylim(0, max(micro + mdm) * 1.22)
    ax.grid(axis="y")

    # (c) tdTomato's cross-gene correlation profile — top correlate is neuronal
    ax = axes[2]
    cp = M["coexp_pearson"]
    markers = ["Meg3", "Pecam1", "Flt1", "Itgam", "Pdgfrb", "Pf4", "Aif1", "Csf1r",
               "Cx3cr1", "Mrc1", "Cd163", "S100b"]
    labels = {"Meg3": "Meg3 (neuronal)"}
    rv = [cp[m]["tdTomato"] for m in markers]
    o = np.argsort(rv)
    markers_s = [markers[i] for i in o]
    rv_s = [rv[i] for i in o]
    cols = [RED if m == "Meg3" else (GREY if m in ("Pecam1", "Flt1", "Pdgfrb", "S100b") else BLUE)
            for m in markers_s]
    y = np.arange(len(markers_s))
    ax.barh(y, rv_s, color=cols, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([labels.get(m, m) for m in markers_s], fontsize=8)
    ax.set_xlabel("Pearson r with tdTomato")
    ax.set_title("(c) tdTomato's top correlate is neuronal")
    ax.grid(axis="x")

    fig.suptitle("Lineage reporters contradict their expected biology — slice 1 (a, c); slices 1–3 (b)",
                 fontsize=12, fontweight="bold", y=1.02)
    out = os.path.join(HERE, "dq_fig2_reporter_inconsistency.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig3_decontx_control():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    strata = ["low_contam", "mid_contam", "high_contam"]
    st = M["ambient"]["strata"]
    labels = [f"Low\n(~{st[s]['mean_contam']*100:.0f}%)" if s == "low_contam"
              else f"Mid\n(~{st[s]['mean_contam']*100:.0f}%)" if s == "mid_contam"
              else f"High\n(~{st[s]['mean_contam']*100:.0f}%)" for s in strata]
    xs = np.arange(len(strata))
    w = 0.38
    f1 = [st[s]["F1_r"] for s in strata]  # GFP-tdTomato
    f2 = [st[s]["F2_r"] for s in strata]  # GFP-Cx3cr1
    b1 = ax.bar(xs - w / 2, f1, w, color=ORANGE, label="GFP – tdTomato")
    b2 = ax.bar(xs + w / 2, f2, w, color=RED, label="GFP – Cx3cr1")
    ax.axhline(0, color="#444444", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Estimated ambient contamination stratum (decontX)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Ambient correction does not rescue the reporter anomalies")
    ax.legend(fontsize=9, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(-0.65, 0.14)
    for bars, vals in [(b1, f1), (b2, f2)]:
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v - 0.015),
                        ha="center", va="top", fontsize=8, color="#222222")
    ax.grid(axis="y")
    out = os.path.join(HERE, "dq_fig3_decontx_control.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    print("computing tdTomato micro/MDM split from slice h5ads ...")
    td = tdt_micro_mdm((1, 2, 3))
    for s, d in td.items():
        print(f"  slice {s}: micro {d['micro_pct']:.1f}% (n={d['n_micro']}) "
              f"vs MDM {d['mdm_pct']:.1f}% (n={d['n_mdm']})")
    fig1_detection_reliability()
    fig2_reporter_inconsistency(td)
    fig3_decontx_control()
    print("done.")
