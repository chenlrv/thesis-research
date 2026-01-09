def get_cell_type_marker_genes() -> dict[str, list[str]]:
    return {
        "ChoroidPlexus": ["Ttr"],
        "Ependymal": ["Foxj1"],
        "Oligodendrocytes": ["Mbp", "Mog", "Mag", "Plp1", "Bcas1", "Ugt8a", "Myrf", "Olig1", "Olig2"],
        "Astrocytes": ["Gfap", "Aqp4", "Aldh1l1", "Slc1a3", "S100b", "Igfbp7"],
        "Microglia": ["Cx3cr1", "P2ry12", "Trem2", "C1qa", "C1qb", "C1qc", "Csf1r", "Tmem119", "Sall1"],
        "OPC": ["Pdgfra", "Olig2"],
        "Endothelia": ["Cldn5", "Pecam1", "Flt1", "Kdr", "Esam", "Emcn", "Vegfa"],

        "Pericytes": ["Pdgfrb", "Rgs5", "Acta2", "Tagln", "Col1a1", "Col1a2", "Des"],
        "GABAergic": ["Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip", "Crh"],
        "Glutamatergic": ["Slc17a7", "Rbfox3", "Camk2a", "Thy1"],
    }
