"""
METHOD C — InSituType (faithful Python reimplementation of insitutypeML, RNA mode).

R is NOT installed (confirmed: no Rscript/R on PATH). Per AGENT_CONTEXT we reimplement
the marker-profile (NB log-likelihood) cell assignment in Python rather than skip it.

Faithful port of InSituType nbclust.R::lldist + insitutypeML.R + utilities.R (RNA branch):
  estimateBackground: bg_cell from neg ~ lm(neg ~ s - 1) through origin, s = rowMeans(counts);
                      non-positive bg floored.
  lldist (rna):  bgsub_cell = sum_g max(count - bg, 0)
                 s_cell     = bgsub_cell / sum(profile_k)          (per cell, per cluster k)
                 yhat_gk    = s_cell * profile_gk + bg_cell
                 loglik_ck  = sum_g dnbinom(count_cg; size=nb_size, mu=yhat_gk, log=TRUE)
  undefined column: cells with zero shared-gene counts -> "undefined".
  clust = argmax loglik;  prob = softmax(loglik) max.

Reference profiles (genes x celltypes, LINEAR scale, background-subtracted means) are
derived from Method-A high-confidence anchors via the Estep formula
  profile_gk = max(mean(counts_g | clust=k) - mean(neg | clust=k), 0)
This is exactly how InSituType builds fixed reference profiles from prior typing, and keeps
Method C anchored on the SAME transcriptome markers (no lineage tags).

We restrict the reference cell types to the panel-resolvable lineages we care about plus a
generic "Other/parenchyma" profile (built from unassigned cells) so panel-blind neurons are
not force-binned into a myeloid type.

Outputs: method_c_labels.csv  (insitutype label + prob + logliks of focal classes)
"""
import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy.stats import nbinom

OUT = "D:/thesis-research/agents/outputs/annotation"
os.makedirs(OUT, exist_ok=True)
INP = "D:/thesis-research/resources/cache/annotation_bench/slice1_nontumor.h5ad"
METHOD_A = OUT + "/method_a_labels.csv"
NB_SIZE = 10.0  # InSituType default

a = ad.read_h5ad(INP)
counts = a.layers["counts"]
counts = counts.tocsr() if sp.issparse(counts) else sp.csr_matrix(counts)
genes = np.array(a.var_names)
n_cells, n_genes = counts.shape
neg_mean = a.obs["neg_mean"].to_numpy(float)
print("loaded", counts.shape)

# ---- estimateBackground: bg_cell = fitted from lm(neg ~ s - 1) ----
s_raw = np.asarray(counts.mean(1)).ravel()            # rowMeans(counts)
# lm through origin: slope = sum(s*neg)/sum(s^2)
slope = float(np.sum(s_raw * neg_mean) / np.sum(s_raw ** 2))
bg = slope * s_raw
pos = bg[bg > 0]
floor = min(1e-5, pos.min()) if pos.size else 1e-5
bg[bg <= 0] = floor
print(f"bg slope={slope:.5f}  bg median={np.median(bg):.5f} mean={bg.mean():.5f}")

# ---- reference profiles from Method-A anchors (Estep) ----
ma = pd.read_csv(METHOD_A, index_col=0)
assert len(ma) == n_cells, f"row count mismatch {len(ma)} vs {n_cells}"
# Method A was written from the SAME adata in the same order -> align by position.
core = ma["method_a_core"].values

# Build reference cell types. Use confident Method-A calls as the seed profiles.
# Map: Microglia, MDM, BAM, Astrocyte, and an "Other" profile from a sample of 'unassigned'.
ref_types = ["Microglia", "MDM", "BAM", "Astrocyte", "Other"]
seed = np.array(["__none__"] * n_cells, dtype=object)
for ct in ["Microglia", "MDM", "BAM", "Astrocyte"]:
    seed[core == ct] = ct
# Other = unassigned (panel-blind parenchyma) — gives the model a null bin so it doesn't
# force neurons into myeloid types (the project's core failure mode).
seed[core == "unassigned"] = "Other"

counts_csc = counts.tocsc()
def col_mean(mask):
    sub = counts[mask]
    return np.asarray(sub.mean(0)).ravel()

profiles = np.zeros((n_genes, len(ref_types)), dtype=float)
for j, ct in enumerate(ref_types):
    m = seed == ct
    if m.sum() == 0:
        continue
    mu_g = col_mean(m)
    mu_neg = neg_mean[m].mean()
    profiles[:, j] = np.maximum(mu_g - mu_neg, 0.0)
    print(f"  profile {ct}: n={int(m.sum())}  sum(profile)={profiles[:,j].sum():.1f}")

prof_sum = profiles.sum(0)  # sum over genes per cluster
prof_sum[prof_sum <= 0] = 1e-9

# ---- lldist (RNA): per-cell loglik under each cluster ----
# bgsub_cell = sum_g max(count - bg, 0)
# Compute on sparse data. count - bg per nonzero entry; zeros contribute max(-bg,0)=0.
X = counts  # csr
bgsub = np.zeros(n_cells)
indptr, indices, data = X.indptr, X.indices, X.data
for i in range(n_cells):
    sl = slice(indptr[i], indptr[i + 1])
    d = data[sl] - bg[i]
    d[d < 0] = 0.0
    bgsub[i] = d.sum()

# dnbinom(x; size, mu) ; convert mu->prob p = size/(size+mu)
# loglik_ck = sum_g logpmf(count_cg; size, mu=yhat_gk)
# yhat_gk = s_ck * profile_gk + bg_c, with s_ck = bgsub_c / prof_sum_k
# We compute per cluster k looping over genes is O(cells*genes) dense; do it cluster-wise
# using the sparse structure: for each cluster, full dense yhat is n_cells x n_genes (too big).
# Instead compute loglik gene-by-gene accumulation per cluster using the identity:
#   sum_g [ logpmf at count_cg ].
# logpmf(k; r, mu) = lgamma(k+r) - lgamma(r) - lgamma(k+1) + r*log(r/(r+mu)) + k*log(mu/(r+mu))
# Split into zero-count genes (k=0) and nonzero. For each cluster we need:
#   term over ALL genes of r*log(r/(r+mu_gk))   [depends on mu_gk for every gene]
#   plus over nonzero entries: lgamma(k+r)-lgamma(r)-lgamma(k+1) + k*log(mu/(r+mu)) - r*log(r/(r+mu)) already in first? no.
# We'll accumulate exactly. mu_gk = s_ck*profile_gk + bg_c. s_ck and bg_c are per-cell.
from scipy.special import gammaln
r = NB_SIZE
K = len(ref_types)
logliks = np.zeros((n_cells, K))
prof = profiles  # genes x K
# Precompute per cell s for each cluster: s_ck = bgsub / prof_sum_k  -> shape (n_cells, K)
s_ck = np.outer(bgsub, 1.0 / prof_sum)   # n_cells x K
# Override per InSituType: if s<=0 use rowSums(mat)/... ; here bgsub>=0 always, fine.

# Process in cell-chunks to bound memory: for a chunk, build dense mu for all genes? that's
# chunk x genes x K. Use chunk of 2000 cells -> 2000*958*5 ~ 9.6M floats = fine.
CHUNK = 2000
Xcsr = X
for start in range(0, n_cells, CHUNK):
    end = min(start + CHUNK, n_cells)
    sub = Xcsr[start:end].toarray().astype(float)        # (m, genes)
    m = end - start
    bg_c = bg[start:end][:, None]                         # (m,1)
    for k in range(K):
        mu = s_ck[start:end, k][:, None] * prof[None, :, k][0] + bg_c   # (m, genes)
        mu = np.maximum(mu, 1e-10)
        # logpmf NB
        ll = (gammaln(sub + r) - gammaln(r) - gammaln(sub + 1.0)
              + r * (np.log(r) - np.log(r + mu))
              + sub * (np.log(mu) - np.log(r + mu)))
        logliks[start:end, k] = ll.sum(1)

# undefined: cells with zero total shared-gene counts (all on panel) -> here all genes shared
tot = np.asarray(X.sum(1)).ravel()
undef = tot == 0

# argmax assignment
clust_idx = logliks.argmax(1)
clust = np.array(ref_types)[clust_idx].astype(object)
clust[undef] = "undefined"

# probs = softmax(loglik) max
ll_shift = logliks - logliks.max(1, keepdims=True)
liks = np.exp(ll_shift)
probs = liks / liks.sum(1, keepdims=True)
prob = probs.max(1)
prob[undef] = 0.0

df = pd.DataFrame(index=a.obs.index)
df["cell_global_id"] = a.obs["cell_global_id"].values
df["method_c"] = clust
df["method_c_prob"] = prob
for j, ct in enumerate(ref_types):
    df[f"ll_{ct}"] = logliks[:, j].round(2)
df["nCount"] = tot
df.to_csv(OUT + "/method_c_labels.csv", index=True)

print("\n=== METHOD C (InSituType-ML reimpl) label counts ===")
print(df["method_c"].value_counts())
print("\nfraction %:")
print((df["method_c"].value_counts() / n_cells * 100).round(2))
vc = df["method_c"].value_counts()
mic = vc.get("Microglia", 0); bam = vc.get("BAM", 0); mdm = vc.get("MDM", 0)
print(f"\nmicroglia:(BAM+MDM) = {mic/(bam+mdm+1e-9):.2f}")
print("median prob:", round(float(np.median(prob)), 3))
print("\nWROTE", OUT + "/method_c_labels.csv")
