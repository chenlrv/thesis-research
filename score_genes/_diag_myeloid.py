import numpy as np
import pandas as pd

df = pd.read_csv("D:/thesis-research/score_genes_slice1_myeloid/cell_scores.csv")
n = len(df)
print(f"total non-tumor: {n:,}")

# how often is Myeloid the top-scoring label, and where are those cells lost?
top_mye = df["top_score_label"] == "Myeloid"
print(f"\ncells whose TOP label is Myeloid: {int(top_mye.sum()):,}")
print(f"  of those, fdr_pass:    {int((top_mye & df['fdr_pass']).sum()):,}")
print(f"  of those, margin_pass: {int((top_mye & df['margin_pass']).sum()):,}")
print(f"  of those, BOTH (=Myeloid call): {int((top_mye & df['fdr_pass'] & df['margin_pass']).sum()):,}")

# for top-myeloid cells failing margin, who is the runner-up?
fail = top_mye & ~df["margin_pass"]
print(f"\ntop-Myeloid cells FAILING margin gate: {int(fail.sum()):,}")
print("  their runner-up label:")
print(df.loc[fail, "second_label"].value_counts().head().to_string())

# raw score_Myeloid distribution
s = df["score_Myeloid"]
print("\nscore_Myeloid quantiles:")
print(s.quantile([0.5, 0.9, 0.95, 0.99]).round(3).to_string())
print(f"cells with score_Myeloid > 0: {int((s>0).sum()):,}")
print(f"cells with score_Myeloid > 0.1: {int((s>0.1).sum()):,}")

# scaled scores: is Myeloid's MAD deflating it?
print("\nper-label raw score std (spread):")
for c in [c for c in df.columns if c.startswith("score_")]:
    print(f"  {c:20s} std={df[c].std():.3f}  %>0={100*(df[c]>0).mean():.1f}")
