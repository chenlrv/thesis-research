"""Render the Method-A annotation for slice 1: breakdown + spatial map.
Run: conda run -n thesis_research python agents/annotation/show_method_a.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ANN = r"d:/thesis-research/agents/outputs/annotation"
df = pd.read_csv(f"{ANN}/all_method_labels.csv")
adf = pd.read_csv(f"{ANN}/method_a_labels.csv")

print("=== Method A (column 'A') value counts ===")
print(df["A"].value_counts(dropna=False).to_string())
print(f"total cells: {len(df):,}")

print("\n=== Method A fine labels ('A_fine') ===")
print(df["A_fine"].value_counts(dropna=False).to_string())

print("\n=== method_a_labels.csv 'method_a' + 'method_a_core' ===")
for c in ("method_a", "method_a_core"):
    if c in adf.columns:
        print(f"-- {c} --")
        print(adf[c].value_counts(dropna=False).to_string())

# ---- spatial map ----
focal = ["Microglia", "MDM", "BAM", "Astrocyte"]
colors = {"Microglia": "#1f77b4", "MDM": "#d62728", "BAM": "#ff7f0e", "Astrocyte": "#2ca02c"}
x = df["CenterX_global_px"].to_numpy(float)
y = df["CenterY_global_px"].to_numpy(float)
lab = df["A"].to_numpy(str)

fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=130,
                         gridspec_kw={"width_ratios": [3, 1]})
ax = axes[0]
oth = ~np.isin(lab, focal)
ax.scatter(x[oth], y[oth], s=1.0, c="#dddddd", alpha=0.5, linewidths=0, label="Other")
for cls in focal:
    m = lab == cls
    ax.scatter(x[m], y[m], s=3.0, c=colors[cls], alpha=0.85, linewidths=0,
               label=f"{cls} (n={int(m.sum()):,})")
ax.set_title("Method A — slice 1 (non-tumor cells), CenterX/Y_global_px")
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend(markerscale=4, loc="upper right", framealpha=0.9)
ax.set_xticks([]); ax.set_yticks([])

# bar chart
ax2 = axes[1]
order = focal + ["Other"]
counts = [int((lab == c).sum()) for c in order]
bar_colors = [colors.get(c, "#bbbbbb") for c in order]
ax2.barh(order[::-1], counts[::-1], color=bar_colors[::-1])
for i, c in enumerate(counts[::-1]):
    ax2.text(c, i, f" {c:,} ({c/len(df)*100:.1f}%)", va="center", fontsize=9)
ax2.set_title("Method A label counts")
ax2.set_xscale("log")
plt.tight_layout()
out = f"{ANN}/method_a_slice1_spatial.png"
plt.savefig(out, bbox_inches="tight")
print(f"\nsaved: {out}")
