import scanpy as sc
ad = sc.read_h5ad("resources/cache/slice_1_adata.h5ad")
low = {g.lower(): g for g in ad.var_names}

AX = {
 "Microglia-homeostatic": ["P2ry12","Sall1","TMEM119","Tmem119","Cx3cr1","Hexb","Siglech",
                           "Gpr34","Olfml3","Slc2a5","Selplg","Fcrls","Csf1r","Sparc","Jun"],
 "MDM / monocyte-derived": ["Ccr2","Ly6c2","Plac8","Tgfbi","Ms4a7","Chil3","S100a4","Vim",
                           "Crip1","Ifitm3","Lgals3","Arg1","Fn1","F13a1","Cd44","Ccr1","Itga4"],
 "BAM / border-assoc":    ["Mrc1","Lyve1","Cd163","Pf4","Stab1","Siglec1","Cbr2","Folr2",
                           "Ms4a7","Apoe","Clec12a","Cd206","Dab2"],
 "Activation/DAM":        ["Trem2","Apoe","Cst7","Lpl","Itgax","Clec7a","Spp1","Gpnmb","Cd9","Ctsb"],
 "Proliferation":         ["Mki67","Top2a","Birc5","Ccnb1","Cenpf"],
}
print("genes on panel:", ad.n_obs, "cells x", ad.n_vars, "genes\n")
for ax, gl in AX.items():
    found = []
    for g in gl:
        if g.lower() in low and low[g.lower()] not in found:
            found.append(low[g.lower()])
    print(f"{ax:24} ({len(found)}): {found}")
