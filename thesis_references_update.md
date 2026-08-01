# Thesis references — gaps filled + ascending renumbering

Worked against the **PDF (Thesis Draft 20.6)** — the current version. Every new
reference below was checked against the primary source (journal/volume/pages/DOI
confirmed). Where an *existing* reference already fits a gap precisely, I reuse it
rather than add a near-duplicate. Nothing here is fabricated; please still confirm
each new source says what the sentence claims before finalizing.

---

## Part A — Fill the `[add ref]` / `(Ref)` placeholders

Numbers in the "New cite" column are the **final ascending numbers** from Part C.

### Introduction (p. 2)

| # | Sentence (ends with placeholder) | New cite | Reference chosen | Why it fits |
|---|---|---|---|---|
| A | "…represent one of the most common intracranial malignancies in adults **[add ref]**" | **[1]** | Achrol et al. 2019, *Nat Rev Dis Primers* | Canonical primer establishing brain metastases as the most common intracranial tumors in adults (epidemiology). |
| B | "…emerges through continuous and dynamic interactions between malignant cells and the surrounding stromal, vascular, and immune compartments **[add ref]**" | **[3]** | Quail & Joyce 2017, *Cancer Cell* | Review of the brain-tumor microenvironment: tumor–stroma–vascular–immune interactions in primary and metastatic brain tumors. |
| C | "…untransformed neighbor cells that are termed tumor microenvironment (TME) cells **[add ref]**" | **[4]** | Hanahan & Weinberg 2011, *Cell* | Foundational statement of the TME concept (untransformed stromal/immune neighbors as regulators of cancer). |
| D | "…monocyte-derived macrophages (MDMs) recruited from the circulation under pathological conditions **[add refs]**" | **[5, 6]** | Ginhoux & Jung 2014, *Nat Rev Immunol*; Bowman et al. 2016, *Cell Rep* | Ginhoux & Jung = monocyte→macrophage developmental/recruitment pathways; Bowman = bone-marrow/monocyte-derived macrophage recruitment specifically in brain tumors. (Placeholder is plural.) |
| E | "…may therefore adopt diverse and dynamic phenotypic states **[add ref]**" | **[8]** | Klemm et al. 2020, *Cell* | Shows disease-specific myeloid phenotypic states across primary and metastatic brain tumors. |
| F | "…shaped by both developmental origin and local microenvironmental cues **[add ref]**" | **[9]** | Van Hove et al. 2019, *Nat Neurosci* *(reuse of existing ref)* | Their title is literally "…identities shaped by ontogeny and tissue environment." Exact fit; no new source needed. |
| G | "…the dura mater as well as in the leptomeningeal compartments, which include the pia and arachnoid mater **[add ref]**" | **[10]** | Sun & Jiang 2024, *J Neuroinflammation* *(reuse of existing ref)* | Dedicated BAM review covering dural / leptomeningeal / perivascular localization. (Alt: Kierdorf 2019 [11].) |
| H | "…MDMs originate from circulating monocytes that infiltrate the brain during pathological conditions, including tumor growth **[add ref]**" | **[6]** | Bowman et al. 2016, *Cell Rep* *(reuse of D)* | Same paper documents monocyte-derived macrophage infiltration of brain tumors. |

### Data-Quality section (p. 23)

| # | Sentence | New cite | Reference chosen | Why it fits |
|---|---|---|---|---|
| I | "…estimated and subtracted with decontX **(Ref)**" | **[23]** | Yang et al. 2020, *Genome Biology* | The DecontX method paper. |

### Fix applied: reassign the microglia-resemblance citation

Old **[3] (Sun & Jiang, BAM review)** was attached to *"microglia can acquire transcriptional
features that resemble those of infiltrating macrophages"* (p.2, para 3) — a BAM review is only a
loose fit for a claim about **microglial** plasticity. **Reassign that sentence to [8] Klemm 2020**
(disease-specific transcriptional alterations distinguishing/overlapping microglia and MDMs in
brain tumors, incl. metastasis) — a direct fit. Bowman 2016 [6] is an equally valid alternative.
Sun & Jiang is then used only where it fits best — the **BAM-anatomy** sentence (G, cited as [10]).

*(This does not change any reference number: [8] Klemm already first appears earlier, at E.)*

---

## Part B — Old → New number mapping (for in-text find/replace)

Renumbering is by **order of first appearance** in the PDF. Because new refs enter early
(intro) and old [16] is first cited on p. 4 (before old [8]–[15]), most numbers shift.

| Old | New | Short ID |
|----|----|----|
| — (new) | **1** | Achrol 2019 — Brain metastases |
| [1] | **2** | Paisana 2025 |
| — (new) | **3** | Quail & Joyce 2017 |
| — (new) | **4** | Hanahan & Weinberg 2011 |
| — (new) | **5** | Ginhoux & Jung 2014 |
| — (new) | **6** | Bowman 2016 |
| [2] | **7** | Feng 2024 |
| — (new) | **8** | Klemm 2020 |
| [4] | **9** | Van Hove 2019 |
| [3] | **10** | Sun & Jiang 2024 |
| [5] | **11** | Kierdorf 2019 |
| [6] | **12** | Gonzalez 2022 |
| [7] | **13** | Rao 2021 |
| [16] | **14** | Eisenbach 1983 |
| [8] | **15** | Chen 2025 |
| [9] | **16** | Liu 2019 |
| [10] | **17** | Peng 2020 |
| [11] | **18** | Ratzabi 2025 |
| [12] | **19** | Stringer 2021 (Cellpose) |
| [13] | **20** | NanoString 2023 |
| [14] | **21** | Aran 2019 (SingleR) |
| [15] | **22** | Kumar 2018 (GSE103548) |
| — (new) | **23** | Yang 2020 (DecontX) |

**⚠ Do the replacement from the highest old number down (or against the ID, not the number),**
so you don't overwrite numbers mid-edit (e.g. old [16]→14 before old [8]→15, etc.).

### In-text citations after renumbering, in reading order
p.2: … adults **[1]** … microenvironment **[2]** … immune compartments **[3]**. …
TME cells **[4]** … under pathological conditions **[5, 6]** … tumor progression **[7]** …
phenotypic states **[8]** … microenvironmental cues **[9]**. … pia and arachnoid mater **[10]** …
including tumor growth **[6]** … infiltrating macrophages **[8]** … microenvironmental context **[9]**. …
meningeal boundaries **[11]**.
p.3: … within the brain **[12]** … disease biology **[13]**.
p.4: … Lewis Lung Carcinoma (LLC) cell line **[14]** … lung cancer brain metastasis **[15]**.
p.5: … peripheral circulation **[16]** … CNS microenvironment **[17]** … in this model **[18]**.
p.7: … CellPose algorithm **[19, 20]**.
p.11: … SingleR **[21]**.
p.12: … GSE103548 **[22]** … Lewis Lung Carcinoma lineage **[14]**.
p.23: … decontX **[23]**.

---

## Part C — Final reference list (ascending)

1. Achrol AS, Rennert RC, Anders C, Soffietti R, Ahluwalia MS, Nayak L, et al. Brain metastases. Nature Reviews Disease Primers. 2019;5(1):5. doi:10.1038/s41572-018-0055-y.

2. Paisana E, Cascão R, Alvoeiro M, Félix F, Martins G, Guerreiro C, et al. Immunotherapy in lung cancer brain metastases. npj Precision Oncology. 2025;9:130. doi:10.1038/s41698-025-00901-0.

3. Quail DF, Joyce JA. The microenvironmental landscape of brain tumors. Cancer Cell. 2017;31(3):326-341. doi:10.1016/j.ccell.2017.02.009.

4. Hanahan D, Weinberg RA. Hallmarks of cancer: the next generation. Cell. 2011;144(5):646-674. doi:10.1016/j.cell.2011.02.013.

5. Ginhoux F, Jung S. Monocytes and macrophages: developmental pathways and tissue homeostasis. Nature Reviews Immunology. 2014;14(6):392-404. doi:10.1038/nri3671.

6. Bowman RL, Klemm F, Akkari L, Pyonteck SM, Sevenich L, Quail DF, et al. Macrophage ontogeny underlies differences in tumor-specific education in brain malignancies. Cell Reports. 2016;17(9):2445-2459. doi:10.1016/j.celrep.2016.10.052.

7. Feng Y, Hu X, Zhang Y, Wang Y. The role of microglia in brain metastases: mechanisms and strategies. Aging and Disease. 2024;15(1):169-185. doi:10.14336/AD.2023.0514.

8. Klemm F, Maas RR, Bowman RL, Kornete M, Soukup K, Nassiri S, et al. Interrogation of the microenvironmental landscape in brain tumors reveals disease-specific alterations of immune cells. Cell. 2020;181(7):1643-1660. doi:10.1016/j.cell.2020.05.007.

9. Van Hove H, Martens L, Scheyltjens I, De Vlaminck K, Pombo Antunes AR, De Prijck S, et al. A single-cell atlas of mouse brain macrophages reveals unique transcriptional identities shaped by ontogeny and tissue environment. Nature Neuroscience. 2019;22(6):1021-1035. doi:10.1038/s41593-019-0393-4.

10. Sun R, Jiang H. Border-associated macrophages in the central nervous system. Journal of Neuroinflammation. 2024;21(1):67. doi:10.1186/s12974-024-03059-x.

11. Kierdorf K, Masuda T, Jordão MJC, Prinz M. Macrophages at CNS interfaces: ontogeny and function in health and disease. Nature Reviews Neuroscience. 2019;20(9):547-562. doi:10.1038/s41583-019-0201-x.

12. Gonzalez H, Mei W, Robles I, Hagerling C, Allen BM, Nanjaraj A, et al. Cellular architecture of human brain metastases. Cell. 2022;185(4):729-745.e20. doi:10.1016/j.cell.2021.12.043.

13. Rao A, Barkley D, França GS, Yanai I. Exploring tissue architecture using spatial transcriptomics. Nature. 2021;596(7871):211-220. doi:10.1038/s41586-021-03634-9.

14. Eisenbach L, Segal S, Feldman M. MHC imbalance and metastatic spread in Lewis lung carcinoma clones. International Journal of Cancer. 1983;32(1):113-120. doi:10.1002/ijc.2910320118.

15. Chen Y, Zhang A, Wang J, Pan H, Liu L, Li R. Refining lung cancer brain metastasis models for spatiotemporal dynamic research and personalized therapy. Cancers. 2025;17(9):1588. doi:10.3390/cancers17091588.

16. Liu Z, Gu Y, Chakarov S, Bleriot C, Kwok I, Chen X, et al. Fate mapping via Ms4a3-expression history traces monocyte-derived cells. Cell. 2019;178(6):1509-1525.e19. doi:10.1016/j.cell.2019.08.009.

17. Peng L, Wang Y, Fei S, Wei C, Tong F, Wu G, et al. The effect of combining Endostar with radiotherapy on blood vessels, tumor-associated macrophages, and T cells in brain metastases of Lewis lung cancer. Translational Lung Cancer Research. 2020;9(3):745-760. doi:10.21037/tlcr-20-500.

18. Ratzabi A, Caspit IM, Telechi I, Kim JS, Vaknine H, Blinder P, et al. Brain metastases exhibit distinct spatial patterns of resident and infiltrating macrophages. Research Square. 2025. doi:10.21203/rs.3.rs-8294376/v1. Preprint.

19. Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nature Methods. 2021;18(1):100-106. doi:10.1038/s41592-020-01018-x.

20. NanoString Technologies. Evaluating the technical performance of single-cell spatial biology with CosMx Spatial Molecular Imager. Seattle (WA): NanoString Technologies; 2023. White paper.

21. Aran D, Looney AP, Liu L, Wu E, Fong V, Hsu A, et al. Reference-based analysis of lung single-cell sequencing reveals a transitional profibrotic macrophage. Nature Immunology. 2019;20(2):163-172. doi:10.1038/s41590-018-0276-y.

22. Kumar R. Analysis of mRNA expression in Lewis lung carcinoma (LLC1) cells and MLE 12 cells [dataset]. Gene Expression Omnibus; 2018. Accession no. GSE103548. https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE103548. Associated publication: Ryan ZC, Craig TA, Wang X, Delmotte P, Salisbury JL, Lanza IR, et al. 1α,25-dihydroxyvitamin D3 mitigates cancer cell-mediated mitochondrial dysfunction in human skeletal muscle cells. Biochemical and Biophysical Research Communications. 2018;496(2):746-752. doi:10.1016/j.bbrc.2018.01.092.

23. Yang S, Corbett SE, Koga Y, Wang Z, Johnson WE, Yajima M, et al. Decontamination of ambient RNA in single-cell RNA-seq with DecontX. Genome Biology. 2020;21(1):57. doi:10.1186/s13059-020-1950-6.

---

## Part D — DOI verification of the pre-existing references

Every pre-existing DOI was resolved against **Crossref** (authoritative registered metadata).
All 15 DOIs resolve to the correct paper and title. Title/journal/year/volume/issue/first-page all
match the thesis, **with two exceptions flagged below**. (The NanoString white paper has no DOI;
the GEO dataset is an accession — its *associated* BBRC publication DOI was checked and matches.)

| New # | Reference | DOI | Crossref result | Verdict |
|----|----|----|----|----|
| 2 | Paisana 2025 | 10.1038/s41698-025-00901-0 | npj Precis Oncol 2025;9(1), art. 130 | ✅ match |
| 7 | Feng 2024 | 10.14336/AD.2023.0514 | Aging Dis 2024;15(1):169 | ✅ match |
| 10 | Sun & Jiang 2024 | 10.1186/s12974-024-03059-x | J Neuroinflammation 2024;21(1), art. 67 | ✅ match |
| 9 | Van Hove 2019 | 10.1038/s41593-019-0393-4 | Nat Neurosci 2019;22(6):1021 | ✅ match |
| 11 | Kierdorf 2019 | 10.1038/s41583-019-0201-x | Nat Rev Neurosci 2019;20(9):547 | ✅ match |
| 12 | Gonzalez 2022 | 10.1016/j.cell.2021.12.043 | Cell 2022;185(4):729 | ✅ match |
| 13 | Rao 2021 | 10.1038/s41586-021-03634-9 | Nature 2021;596(7871):211 | ✅ match |
| 15 | Chen 2025 | 10.3390/cancers17091588 | Cancers 2025;17(9), art. 1588 | ✅ match |
| 16 | Liu 2019 | 10.1016/j.cell.2019.08.009 | Cell 2019;178(6):1509 | ✅ match |
| 17 | Peng 2020 | 10.21037/tlcr-20-500 | Transl Lung Cancer Res 2020;9(3):745 | ✅ match |
| 18 | Ratzabi 2025 | 10.21203/rs.3.rs-8294376/v1 | Research Square 2025, preprint | ⚠️ author order — see below |
| 19 | Stringer (Cellpose) | 10.1038/s41592-020-01018-x | Nat Methods 18(1):100; online 2020 / print 2021 | ⚠️ year note — see below |
| 21 | Aran (SingleR) 2019 | 10.1038/s41590-018-0276-y | Nat Immunol 2019;20(2):163 | ✅ match |
| 14 | Eisenbach 1983 | 10.1002/ijc.2910320118 | Int J Cancer 1983;32(1):113 | ✅ match |
| 22 | Kumar / GSE103548 assoc. pub | 10.1016/j.bbrc.2018.01.092 | Ryan ZC et al., BBRC 2018;496(2):746 | ✅ match |

### ⚠️ Flag 1 — Ratzabi [18]: author order disagrees with the registered DOI
Crossref registers the author list for `10.21203/rs.3.rs-8294376/v1` as, in order:
**Reuven Stein, Avinoam Ratzabi, Itai Caspit, Ira Telechi, Jung-Seok Kim, Hananya Vaknine,
Pablo Blinder, Steffen Jung.** Your thesis lists **Ratzabi** first and drops Steffen Jung.
Also minor: thesis has "Caspit **IM**" but the record shows **Itai Caspit** (initial "I").

I did **not** auto-change this — preprint author order is a real editorial choice and Research
Square's registered order is sometimes scrambled relative to the PDF title page. **Please open the
preprint title page and confirm.** Two clean options:

- Matches registered DOI: *Stein R, Ratzabi A, Caspit I, Telechi I, Kim JS, Vaknine H, et al.*
- If the PDF really lists Ratzabi first (Stein senior/last): *Ratzabi A, Caspit I, Telechi I, Kim JS, Vaknine H, Blinder P, et al.*

### ⚠️ Flag 2 — Cellpose [19]: year is fine, just be consistent
Crossref shows online publication **2020**; the print issue **Nat Methods 18(1):100–106 is January
2021**. Your "2021;18(1):100-106" is the correct print citation — no change needed. Only make sure
you cite online-vs-print years consistently across the bibliography.

**Bottom line:** 14/15 DOIs are clean and correct as written; only the Ratzabi preprint needs your
confirmation on author order.

---

## Part E — Round 2: reviewer (ChatGPT) feedback, verified

Every DOI the reviewer proposed was checked against Crossref before use. Most were real; **one was
mislabeled**. This round expands the list from **23 → 28** references. The updated `thesis_update.md`
already reflects everything marked *Applied* below.

### Applied
| Item | Action | Verified source |
|---|---|---|
| **Ref 18 Ratzabi now published** | Replaced preprint with published paper; removed author-order caveat (published version has **Ratzabi first**, Stein last — resolves the earlier flag). | Ratzabi A, Caspit IM, Telechi I, Kim JS, Vaknine H, Blinder P, Jung S, Stein R. *Cell Death Discovery.* 2026;12(1):211. doi:10.1038/s41420-026-03084-0 ✅ Crossref + nature.com |
| **TMEM119 rationale** (new ref 19) | Added to custom-probe paragraph | Bennett ML et al. *PNAS.* 2016;113(12):E1738-E1746. doi:10.1073/pnas.1525528113 ✅ |
| **Trem2 rationale** (new ref 20) | Added | Keren-Shaul H et al. *Cell.* 2017;169(7):1276-1290. doi:10.1016/j.cell.2017.05.018 ✅ |
| **Ccl2 rationale** (new ref 21) | Added — used **Qian 2011**, NOT the reviewer's Ma 2023 (see below) | Qian BZ et al. *Nature.* 2011;475(7355):222-225. doi:10.1038/nature10138 ✅ |
| **Cxcl13 rationale** (new ref 22) | Added | Kazanietz MG, Durando M, Cooke M. *Front Endocrinol.* 2019;10:471. doi:10.3389/fendo.2019.00471 ✅ |
| **GFAP rationale** (new ref 23) | Added | Yang Z, Wang KKW. *Trends Neurosci.* 2015;38(6):364-374. doi:10.1016/j.tins.2015.04.003 ✅ |
| **Lyve1/BAM rationale** | Cited existing **[9] Van Hove** (reuse — no new ref) | — |

### Corrected the reviewer
- **Ma 2023 was mislabeled as a "Ccl2" paper.** The DOI `10.1038/s41467-023-38252-8` is real but is
  *"Type I interferon response in astrocytes promotes brain metastasis by enhancing monocytic myeloid
  cell recruitment"* (Nat Commun 2023;14:2632) — **not a CCL2 paper**. I used the canonical CCL2
  reference (Qian 2011, Nature) instead. Ma 2023 could optionally be added elsewhere as brain-met
  monocyte-recruitment support, but it does not back the specific "Ccl2 regulates monocyte
  recruitment" sentence.
- **Reviewer's Cxcl13 DOI guess** (`10.3389/fimmu.2019.00471`) resolves to an unrelated Jackson
  lymphatic-trafficking paper; the correct Kazanietz review is `10.3389/fendo.2019.00471` (Frontiers
  in **Endocrinology**). Verified and used.

### Considered but NOT applied (your call)
- **Eisenbach 1984** (H-2K vs H-2D antigens in metastatic clones; `10.1002/ijc.2910340421`, Int J
  Cancer 1984;34(4):567 — verified real). The reviewer suggested it as a more direct D122
  characterization. I kept the 1983 paper as **[14]** because it already supports the "D122 is a
  high-metastatic LLC subclone" claim; add 1984 as a companion only if you want the antigenic
  characterization too (would become a new ref and shift later numbers).
- **Adding Ratzabi [18] to the "D122 widely used" sentence** (Chen [15]). Reasonable, but it would move
  Ratzabi's first appearance into the Cell-Line section and reshuffle numbers again; left as-is.

### Minor caveats acknowledged (no change needed)
- **[2] Paisana** — immunotherapy-focused; the broad epidemiology claim is already carried by **[1] Achrol**.
- **[12] Gonzalez** — human brain-met architecture, not a spatial-methods paper; it is paired with **[13] Rao** (spatial transcriptomics) nearby, so the methods claim is covered.
- **[28] DecontX** — developed for scRNA-seq, not CosMx. The text already frames it as an *exploratory diagnostic* that did **not** rescue the anomalies, so no overclaim.
- **[25] CosMx segmentation** — if you want a stronger citation for "enhanced Cellpose," add the AtoMx/CosMx SMI user manual alongside Stringer [24].
- **Facility issues** (reagent tray, yellow channel, segmentation) are already framed in-text as a *"technical assessment provided by the processing facility,"* not as literature — consistent with the reviewer's recommendation.

### Round-1 → Round-2 number shifts
Refs **1–17 unchanged**. Ref **18** = Ratzabi (updated to published). **19–23** = new marker refs
(Bennett, Keren-Shaul, Qian, Kazanietz, Yang & Wang). Old 19→**24** (Stringer), 20→**25** (NanoString),
21→**26** (Aran/SingleR), 22→**27** (Kumar/GSE103548), 23→**28** (DecontX). First appearances remain
strictly ascending 1→28.
