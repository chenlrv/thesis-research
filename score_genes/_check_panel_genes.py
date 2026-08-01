"""Read-only: verify marker genes against the panel (h5ad var_names).

Three levels of matching so we don't get a false "MISSING":
  1. exact (case-insensitive)
  2. normalized (lowercase + strip every non-alphanumeric char) -> catches
     Siglec-h / Siglech, F13A1 / F13a1, separators, dots, dashes
  3. stem/substring candidates -> for anything still missing, print every
     panel gene that shares the stem, so aliases/family members are visible
     (e.g. Ly6c1 vs Ly6c2, P2ry12 vs P2ry13).

Also writes the full sorted panel gene list to _panel_gene_list.txt.
"""
import re

import h5py

PATH = "D:/thesis-research/resources/cache/with_tumor_prediction/slice_1_adata.h5ad"
LIST_OUT = "D:/thesis-research/score_genes/_panel_gene_list.txt"

# everything we care about across stage-1 + stage-2 + lymphoid check
MARKERS = [
    "Csf1r", "Tyrobp", "Fcer1g", "Aif1", "Cx3cr1",
    "TMEM119", "P2ry12", "P2rx5", "Selplg", "Sall1", "Siglech", "Hexb", "Slc2a5",
    "Mrc1", "Cd163", "Lyve1", "Ms4a7", "Pf4", "Cbr2", "F13a1",
    "Ccr2", "Plac8", "Cd14", "Ly6c2", "Apoe",
    "Cd3e", "Cd3d", "Nkg7", "Klrb1c", "Gzmb", "Cd19", "Ms4a1",
]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def stem(s):
    # gene symbol minus any trailing digits, e.g. Ly6c2 -> ly6c, P2ry12 -> p2ry
    return re.sub(r"\d+$", "", norm(s))


def main():
    with h5py.File(PATH, "r") as h5:
        var = h5["var"]
        key = var.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode()
        raw = var[key][...]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in raw]

    with open(LIST_OUT, "w") as f:
        f.write("\n".join(sorted(names)) + "\n")

    exact = {n.lower(): n for n in names}
    normed = {}
    for n in names:
        normed.setdefault(norm(n), n)  # first occurrence wins

    print(f"panel size: {len(names)} genes")
    print(f"full sorted list written to {LIST_OUT}\n")

    still_missing = []
    for g in MARKERS:
        if g.lower() in exact:
            print(f"  {g:10s} -> FOUND (exact) as {exact[g.lower()]}")
        elif norm(g) in normed:
            print(f"  {g:10s} -> FOUND (normalized) as {normed[norm(g)]}  <-- casing/punct differs")
        else:
            print(f"  {g:10s} -> not found by exact/normalized")
            still_missing.append(g)

    if not still_missing:
        print("\nno genuinely-missing genes; all resolved at exact or normalized level.")
        return

    print("\n=== stem/alias candidates for genes still missing ===")
    for g in still_missing:
        st = stem(g)
        cands = sorted(
            {n for n in names if st and (norm(n).startswith(st) or st in norm(n))}
        )
        print(f"  {g:10s} (stem '{st}') -> "
              + (", ".join(cands) if cands else "NO panel gene shares this stem -> genuinely absent"))


if __name__ == "__main__":
    main()
