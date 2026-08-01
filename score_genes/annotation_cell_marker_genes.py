"""Marker gene sets for per-cell annotation of the CosMx mouse brain panel.

Seven cell types (the myeloid compartment is split into Microglia + Macrophage,
because score_genes averages over the set -- a single heterogeneous "myeloid" set
dilutes every subtype's score). Every gene is confirmed present on the 958-gene
panel, and the lists are DEDUPLICATED (no gene appears in two types) so a single
gene cannot make two scores fire on the same cell.

Recall caveats (panel limitations, not method):
  - Microglia: TMEM119/Cx3cr1/P2rx5 are homeostatic and DROP on activation, so some
    tumor-activated microglia may be missed (relevant in tumor slices).
  - Macrophage: lumps BAM (Mrc1/Cd163/Lyve1) + MDM (Ccr2/Plac8/Cd14) -- two distinct
    modules; split into BAM + MDM if macrophage recall is weak.
  - Pan-myeloid genes (Csf1r/C1qa/Aif1/Cd68/Itgam/Ptprc) are intentionally LEFT OUT
    of both myeloid sets: they don't separate microglia from macrophage and would
    re-couple the two scores.
  - Pericytes: capped at 4 markers (panel lacks Kcnj8/Abcc9/Higd1b); adding more
    means smooth-muscle genes, which would mislabel SMC cells as pericytes.
  - Ependymal: no true ependymal-specific gene on the panel (no Foxj1 / cilia genes),
    so this set marks the VENTRICULAR LINING broadly and folds in choroid-plexus
    epithelium (keratins / Epcam).
  - Neurons: no pan-neuronal gene on the panel (no Snap25 / Rbfox3); expect a real
    unassigned fraction of neurons.
  - Astrocytes: S100b and Sox9 are recall adds that also tag ependymal -> minor
    astrocyte<->ependymal bleed.
"""

MARKER_GENES = {
    "Astrocytes": [
        "GFAP", "Sparcl1", "Fgfr3", "Glul", "Gpx3", "S100b", "Sox9",
    ],
    "Microglia": [
        "TMEM119", "Cx3cr1", "P2rx5", "Selplg",
    ],
    "Macrophage": [                         
        "Mrc1", "Cd163", "Lyve1",            # BAM (border-associated)
        "Ccr2", "Plac8", "Cd14",             # MDM (monocyte-derived)
    ],
    "Endothelial": [
        "Cdh5", "Pecam1", "Flt1", "Kdr", "Tek", "Tie1", "Esam", "Vwf",
        "Slc2a1", "Clec14a", "Adgrl4", "Ldb2", "Icam2", "Eng", "Cd34",
        "Ramp2", "Klf2",
    ],
    "Pericytes": [
        "Rgs5", "Pdgfrb", "Notch3", "Vtn",
    ],
    "Ependymal": [
        "Adgrv1", "Cd24a", "Ttr", "Epcam", "Krt8", "Krt18", "Krt19", "Cldn4",
    ],
    "Neurons": [
        "Meg3", "Nrxn1", "Nrxn3", "Scg5", "Cx3cl1",
        "Xkr4", "Ryr2", "Pnoc", "Calb1", "Sst",
    ],
}
