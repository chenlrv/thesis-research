"""Quick panel-gene presence check.

Loads one slice's var_names and reports which marker genes are in the CosMx
panel vs. missing, for every module used by annotation v1 and v2.
"""
import pathlib

import anndata as ad

SLICE_PATH = pathlib.Path(
    r"D:\thesis-research\resources\cache\with_tumor_prediction\slice_1_adata.h5ad"
)

MODULES = {
    "microglia (v1+v2)": ["Hexb", "Sall1", "Siglech", "P2ry12", "TMEM119", "Cx3cr1"],
    "BAM (v1+v2)":       ["Lyve1", "Mrc1", "Cd163", "Pf4", "Marco", "Selenop", "Cd209a"],
    "MDM (v1+v2)":       ["Ccr2", "Ly6c2", "Plac8", "Crip1", "Cd14", "S100a4"],
    "astrocyte (v1+v2)": ["GFAP", "S100b", "Sox9", "Glul", "Apod", "Aldh1l1", "Aqp4",
                          "Sparcl1", "Clu"],
    "vascular_niche":    ["Pecam1", "Cdh5", "Esam", "Vwf", "Eng", "Tek", "Kdr",
                          "Flt1", "Tie1", "Clec14a", "Adgrl4", "Icam2"],
    "lineage tracers":   ["tdTomato", "GFP"],
}


def main():
    adata = ad.read_h5ad(SLICE_PATH)
    panel = {g.lower(): g for g in adata.var_names}
    print(f"loaded {SLICE_PATH.name}: {adata.n_vars} genes in panel\n")

    for module, genes in MODULES.items():
        present, missing = [], []
        for g in genes:
            if g.lower() in panel:
                present.append(panel[g.lower()])
            else:
                missing.append(g)
        print(f"=== {module} ===")
        print(f"  present ({len(present):>2}/{len(genes)}): {present}")
        if missing:
            print(f"  MISSING  : {missing}")
        print()


if __name__ == "__main__":
    main()
