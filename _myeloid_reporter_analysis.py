import os
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import spearmanr

OUT = "myeloid_plots"; os.makedirs(OUT, exist_ok=True)
ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
myeloid = np.load("_myeloid_mask_slice1.npy")

# raw reporter counts BEFORE normalization
def gene_raw(g):
    x = ad[:, ad.var_names.get_loc(g)].X
    return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()

gfp_raw, td_raw = gene_raw("GFP"), gene_raw("tdTomato")
meanG = ad.obs["Mean.G"].to_numpy(float)
meanDAPI = ad.obs["Mean.DAPI"].to_numpy(float)

# normalized expression for marker scores
sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
def score(genes):
    cols = [ad.var_names.get_loc(g) for g in genes if g in ad.var_names]
    x = ad[:, cols].X
    x = np.asarray(x.todense()) if sp.issparse(x) else np.asarray(x)
    return x.mean(1)
resid = score(["TMEM119", "Cx3cr1"])      # resident microglia
infil = score(["Ccr2", "Plac8"])          # MDM / monocyte-derived
bam   = score(["Mrc1", "Cd163", "Pf4", "Lyve1"])

def bimodality(v, label):
    """GMM 1- vs 2-component BIC on log1p; report split."""
    x = np.log1p(np.clip(v, 0, None)).reshape(-1, 1)
    x = x[np.isfinite(x).ravel()]
    g1 = GaussianMixture(1, random_state=0).fit(x)
    g2 = GaussianMixture(2, random_state=0).fit(x)
    d = g1.bic(x) - g2.bic(x)                       # >0 favors 2 comps
    mu = np.expm1(g2.means_.ravel()); w = g2.weights_
    hi = int(np.argmax(mu))
    frac_hi = float(w[hi])
    verdict = "BIMODAL" if d > 10 else ("weak" if d > 0 else "unimodal")
    print(f"  {label:10} dBIC(2-1)={d:8.1f} -> {verdict:8} | comp means~{mu.round(1)} "
          f"weights {w.round(2)} | bright frac~{frac_hi:.2f}")
    return d

print(f"myeloid n={myeloid.sum():,}\n== bimodality within myeloid ==")
for v, lab in [(meanG, "Mean.G"), (td_raw, "tdTomato"), (gfp_raw, "GFP"),
               (meanDAPI, "Mean.DAPI")]:
    bimodality(v[myeloid], lab)

print("\n== does the reporter track residency vs infiltration? (Spearman within myeloid) ==")
print(f"{'reporter':10}{'resid':>8}{'infil':>8}{'BAM':>8}{'DAPI(ctrl)':>12}")
for v, lab in [(meanG, "Mean.G"), (td_raw, "tdTomato"), (gfp_raw, "GFP")]:
    vm = v[myeloid]
    rr = spearmanr(vm, resid[myeloid]).correlation
    ri = spearmanr(vm, infil[myeloid]).correlation
    rb = spearmanr(vm, bam[myeloid]).correlation
    rd = spearmanr(vm, meanDAPI[myeloid]).correlation
    print(f"{lab:10}{rr:>8.3f}{ri:>8.3f}{rb:>8.3f}{rd:>12.3f}")

# figure: reporter histograms (myeloid vs non) + reporter vs residency
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), dpi=140)
for a, (v, lab) in zip(ax[:2], [(meanG, "Mean.G (protein IF)"), (td_raw, "tdTomato (RNA)")]):
    a.hist(np.log1p(v[myeloid]), bins=60, alpha=.7, density=True, label="myeloid", color="#d62728")
    a.hist(np.log1p(v[~myeloid]), bins=60, alpha=.5, density=True, label="non-myeloid", color="#7f7f7f")
    a.set_xlabel(f"log1p {lab}"); a.set_ylabel("density"); a.legend(); a.set_title(lab)
a = ax[2]
sccol = a.scatter(np.log1p(meanG[myeloid]), resid[myeloid], s=3, c=infil[myeloid],
                  cmap="viridis", alpha=.5, rasterized=True)
a.set_xlabel("log1p Mean.G"); a.set_ylabel("residency score (TMEM119+Cx3cr1)")
plt.colorbar(sccol, ax=a, label="infiltration (Ccr2+Plac8)")
a.set_title("Mean.G vs residency (myeloid)")
fig.tight_layout(); fig.savefig(f"{OUT}/myeloid_reporter_slice1.png"); plt.close()
print(f"\nsaved {OUT}/myeloid_reporter_slice1.png")
