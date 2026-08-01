"""Check which canonical markers for each target cell type are present on the panel.
Prints present (usable) vs absent (canonical but missing) per cell type so marker
suggestions are grounded in the actual panel.
"""
import h5py
import numpy as np

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"

CANDIDATES = {
    "Astrocytes": ["GFAP", "Aqp4", "Slc1a3", "Slc1a2", "Aldh1l1", "Gja1", "Gjb6",
                   "Sox9", "S100b", "Glul", "Sparcl1", "Sparc", "Ntsr2", "Fgfr3",
                   "Acsbg1", "Aldoc", "Clu", "Apoe", "Vim", "Glud1", "Agt", "Mlc1",
                   "Slc6a11", "Fabp7", "Hepacam", "Gpc5", "Btbd17", "Cst3"],
    "Myeloid": ["C1qa", "C1qb", "C1qc", "Csf1r", "Aif1", "Tyrobp", "Fcer1g",
                "Cx3cr1", "P2ry12", "P2rx5", "TMEM119", "Trem2", "Hexb", "Ctss",
                "Cd68", "Itgam", "Ptprc", "Cd14", "Mrc1", "Cd163", "Lyve1", "Ccr2",
                "Plac8", "Fcgr1", "Selplg", "Laptm5", "Ly86", "Spi1", "Lgals3",
                "Cd74", "Apoe", "Cst7", "Gpnmb", "Spp1", "Ms4a7", "Csf1"],
    "Endothelial": ["Pecam1", "Cdh5", "Flt1", "Kdr", "Cldn5", "Slco1a4", "Slc2a1",
                    "Vwf", "Tie1", "Tek", "Emcn", "Esam", "Eng", "Ly6c1", "Flt4",
                    "Erg", "Sox17", "Cd34", "Podxl", "Adgrf5", " Acvrl1", "Cd93",
                    "Egfl7", "Ramp2", "Klf2", "Sox18", "Vwa1"],
    "Pericytes": ["Pdgfrb", "Rgs5", "Kcnj8", "Abcc9", "Acta2", "Myh11", "Tagln",
                  "Cspg4", "Anpep", "Vtn", "Notch3", "Des", "Pln", "Cald1",
                  "Higd1b", "Atp13a5", "Carmn", "Myl9", "Mcam", "Ndufa4l2",
                  "Pdgfra", "Col1a1", "Dcn", "Bgn", "Myh11", "Tpm2"],
    "Ependymal": ["Foxj1", "Pifo", "Hdc", "Tmem212", "Ccdc153", "Rarres2", "Tekt1",
                  "Sox2", "Cd24a", "Ak7", "Ak9", "Rsph1", "Dnali1", "Calml4",
                  "Adgrv1", "Tm4sf1", "Mia", "Ttr", "Epcam", "Clic6", "Folr1",
                  "Vim", "Spag6", "Dnah11", "Hdc", "Stoml3", "Ccdc67"],
    "Neurons": ["Snap25", "Rbfox3", "Syt1", "Syp", "Map2", "Tubb3", "Nefl", "Nefm",
                "Nefh", "Stmn2", "Gap43", "Dcx", "Eno2", "Slc17a7", "Slc17a6",
                "Gad1", "Gad2", "Nrgn", "Camk2a", "Grin1", "Gria1", "Syn1",
                "Nrxn1", "Nrxn3", "Meg3", "Calb1", "Calb2", "Sst", "Vip", "Snhg11",
                "Rbfox3", "Calm1", "Calm2", "Calm3", "App", "Cx3cl1", "Scg5",
                "Meg3", "Cck", "Pvalb", "Gabra1", "Kcnip4", "Rgs4"],
}


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def main():
    with h5py.File(SLICE, "r") as h:
        var = h["var"]; key = var.attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else key
        vn = list(decode(var[key][...]))
    lut = {g.lower().strip(): g for g in vn}
    print(f"panel has {len(vn)} genes\n")
    for ct, genes in CANDIDATES.items():
        present, absent = [], []
        seen = set()
        for g in genes:
            gl = g.lower().strip()
            if gl in seen:
                continue
            seen.add(gl)
            (present if gl in lut else absent).append(lut.get(gl, g))
        print(f"=== {ct} ===")
        print(f"  PRESENT ({len(present)}): {', '.join(present)}")
        print(f"  absent : {', '.join(absent)}")
        print()


if __name__ == "__main__":
    main()
