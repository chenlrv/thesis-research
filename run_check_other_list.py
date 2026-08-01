"""Check the other LLM's marker list against the panel (present vs absent)."""
import h5py
import numpy as np

SLICE = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_3_adata.h5ad"

OTHER = {
    "Astrocytes": ["GFAP", "Sox9", "S100b", "Glul", "Sparcl1", "Glud1", "Ntrk2",
                   "Gpx3", "Clu"],
    "Myeloid cells": ["Aif1", "Csf1r", "Tyrobp", "Fcer1g", "Cd68", "Itgam",
                      "Trem2", "C1qa", "C1qb", "C1qc", "Lgals3", "Apoe", "Cd74"],
    "Endothelial": ["Pecam1", "Cdh5", "Kdr", "Flt1", "Esam", "Tie1", "Tek", "Vwf",
                    "Ldb2", "Clec14a", "Eng", "Icam2", "Adgrl4"],
    "Pericytes": ["Rgs5", "Pdgfrb", "Notch3", "Vtn", "Npr3", "Cald1", "Myl9"],
    "Ependymal cells": ["Cd24a", "Epcam", "Ttr", "Ptgds", "Ddr1", "Cldn4", "Krt8",
                        "Krt18", "Krt19"],
    "Neurons": ["Meg3", "Nrxn1", "Nrxn3", "Calb1", "Sst", "Scg5", "Xkr4", "Pnoc",
                "Ryr2", "Fgf13", "Aatk"],
}


def decode(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def main():
    with h5py.File(SLICE, "r") as h:
        var = h["var"]; key = var.attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else key
        vn = list(decode(var[key][...]))
    lut = {g.lower(): g for g in vn}
    for ct, genes in OTHER.items():
        present = [g for g in genes if g.lower() in lut]
        absent = [g for g in genes if g.lower() not in lut]
        print(f"=== {ct} ===")
        print(f"  PRESENT: {', '.join(present)}")
        print(f"  ABSENT : {', '.join(absent) if absent else '(none)'}")
        print()


if __name__ == "__main__":
    main()
