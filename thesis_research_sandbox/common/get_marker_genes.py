def get_marker_genes() -> dict[str, list[str]]:
    return {
        'ChoroidPlexus': ['Ttr', 'Folr1', 'Prlr', 'Kcnj13', 'Clic6', 'Aqp1',
                          'Slc4a5', 'Otx2'],
        'Ependymal': ['Foxj1', 'Meig1', 'Dynlrb2', 'Fam183b', 'Pifo'],
        'Oligodendrocytes': [
            "Mbp", "Plp1", "Mag", "Mog",
            "Cnp", "Omg", "Ugt8a", "Fa2h"
        ],
        'OPC': [
            "Pdgfra", "Cspg4", "Olig2", "Olig1", "Sox10", "Gpr17"],

        'Astrocytes': [
            "Aldh1l1", "Slc1a3", "Aqp4", "Gfap", "S100b", "Slc1a2", "Aldoc",
            "Glul", "Sox9", "Slco1c1", "Sparcl1"
        ],

        'Microglia':
            [
                'Spp1', 'Rab7b', 'Ly86', 'Ccl9', 'Gpr34', 'Mpeg1', 'P2ry12',
                'Tnfrsf1b', 'Fcrls', 'Cx3cr1', 'Aif1', 'Gm13710', 'Trem2',
                'Ccl4', 'Cd53', 'C1qa', 'C1qb', 'C1qc', 'Csf1r', 'Tmem119', 'Sall1'],

        'Endothelia': [
            "Pecam1", "Cdh5", "Flt1", "Vwf", "Ptprb", "Eng", "Cldn5",
            "Slc2a1", "Slco1c1", "Mfsd2a", "Tfrc"],
        'LH': [],
        'DG': ["Prox1", "Calb1",
               "NeuN", "C1ql2", "Dock10"],
        'CA': [],
        'Pericytes': ['Pdgfrb', 'Rgs5', 'Acta2', 'Tagln', 'Col1a1', 'Col1a2', 'Des'],
        'GABAergic': ['Gad1', 'Gad2', 'Slc32a1', 'Pvalb', 'Sst', 'Vip', 'Crh'],
        'Glutamatergic': ['Slc17a7', 'Rbfox3', 'Camk2a', 'Thy1'],
    }
