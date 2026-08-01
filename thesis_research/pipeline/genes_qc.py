import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
import scipy.sparse as sp

# 1. Extract expression matrix (use .X or .raw.X depending on your normalization)
# We convert to a dense array to calculate max across all cells (axis=0)
adata = ad.read_h5ad(r"D:\thesis-research\resources\cache\sample_L321_adata.h5ad")

X = adata.X
if sp.issparse(X):
    X = X.tocsr()

mx = X.max(axis=0)  # shape should be (1, n_vars) in some matrix-like form

# robust: convert the 1×n_vars result to a real 1D ndarray
if sp.issparse(mx):
    max_vals = np.asarray(mx.todense()).ravel()
else:
    max_vals = np.asarray(mx).ravel()

print("X.shape:", X.shape)
print("n_vars:", adata.n_vars)
print("max_vals.shape:", max_vals.shape)

assert max_vals.shape[0] == adata.n_vars

df_rank = pd.DataFrame({"gene": adata.var_names.to_list(), "max_exp": max_vals})
df_rank = df_rank.sort_values("max_exp", ascending=False).reset_index(drop=True)

import matplotlib.pyplot as plt

# df_rank must already exist with columns: ["gene", "max_exp"]
# and be sorted descending by max_exp (as you already did)

top20 = df_rank.head(20).copy()
bottom20 = df_rank.tail(20).copy().sort_values("max_exp", ascending=True)

# --- Plot 1: Top 20 (highest max expression) ---
plt.figure(figsize=(10, 6), dpi=200)
plt.barh(top20["gene"][::-1], top20["max_exp"][::-1])  # reverse so highest is on top
plt.title("Top 20 genes by max expression")
plt.xlabel("Max expression")
plt.tight_layout()
plt.show()

# --- Plot 2: Bottom 20 (lowest max expression) ---
plt.figure(figsize=(10, 6), dpi=200)
plt.barh(bottom20["gene"], bottom20["max_exp"])
plt.title("Bottom 20 genes by max expression")
plt.xlabel("Max expression")
plt.tight_layout()
plt.show()

top20_no1 = df_rank.iloc[1:21].copy()

plt.figure(figsize=(10, 6), dpi=200)
plt.barh(top20_no1["gene"][::-1], top20_no1["max_exp"][::-1])
plt.title("Top 20 excluding top-1 outlier")
plt.xlabel("Max expression")
plt.tight_layout()
plt.show()
