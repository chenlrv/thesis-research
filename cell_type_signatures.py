"""Cell-type marker signatures for score_genes, grounded in the CosMx mouse
immuno-oncology panel (958 genes; verified present on slice_3).

Design rules applied:
  - every gene confirmed present on the panel
  - DEDUPLICATED across groups (no gene appears in two signatures) so one gene
    can't make two scores fire on the same cell
  - shared / non-specific genes removed (Apoe, Vim, Clu, Sox9, S100b, Calm1/2/3,
    App, Glud1) -- they leak across glia/myeloid/endothelial/mural
  - activation-state genes kept OUT of the lineage signatures (Trem2, Cst7,
    Gpnmb, Spp1, Lgals3 mark DAM/activated state, not lineage)

Confidence (panel coverage):
  HIGH   : myeloid, endothelial, astrocyte
  MEDIUM : pericyte (separate from smooth_muscle / fibroblast below)
  LOW    : ependymal, neuron  -> expect low recall + a large 'unassigned' pool
           (the panel lacks the canonical markers for these two types)
"""

# ---- the six requested groups ----
SIGNATURES = {
    "astrocyte":   ["GFAP", "Sparcl1", "Fgfr3", "Glul"],
    "myeloid":     ["Csf1r", "Aif1", "Tyrobp", "Fcer1g", "C1qb", "Cd68"],
    "endothelial": ["Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Slc2a1", "Esam", "Vwf"],
    "pericyte":    ["Rgs5", "Pdgfrb", "Notch3", "Vtn"],
    "ependymal":   ["Adgrv1", "Cd24a"],                      # LOW: no Foxj1/cilia genes on panel
    "neuron":      ["Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1"],  # LOW: no Snap25/Rbfox3
}

# ---- recommended competing anchors ----
# These cell types ARE on the panel and, if omitted, will contaminate the groups
# above (smooth muscle pollutes pericyte; fibroblast pollutes pericyte/astro;
# choroid plexus pollutes ependymal). Add them so non-target cells have somewhere
# to go under an argmax/competitive scheme.
EXTRA_ANCHORS = {
    "smooth_muscle":  ["Acta2", "Myh11", "Tagln", "Carmn"],   # cluster-6 (vascular SMC)
    "fibroblast":     ["Col1a1", "Dcn", "Bgn", "Pdgfra"],     # perivascular/meningeal
    "choroid_plexus": ["Ttr", "Epcam"],                       # ventricle lining, != ependymal
}

# ---- optional refinements ----
# Sub-split of the myeloid compartment (use only AFTER calling myeloid; do not mix
# into the pan-myeloid score). Microglia markers are activation-unstable (lost near
# tumor); MDM marker Plac8 is sparse.
MYELOID_SUBTYPES = {
    "microglia":        ["TMEM119", "Cx3cr1", "P2rx5"],
    "border_macrophage":["Mrc1", "Cd163", "Lyve1"],
    "monocyte_derived": ["Ccr2", "Plac8"],
}

# Neuron subset markers (interneurons) -- add to 'neuron' if you want those subtypes,
# but they will NOT capture excitatory/most neurons (panel limitation):
NEURON_SUBSET = ["Calb1", "Sst"]
