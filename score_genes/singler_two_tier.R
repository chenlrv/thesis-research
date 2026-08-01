# =============================================================================
# Two-tier SingleR as an independent second judge for the CosMx annotation.
#   Tier 1 (broad): multi-ref SingleR [stromal/neuronal + myeloid(MG/MDM/BAM) + tumor]
#                   -> MG/MDM/BAM collapsed to "Myeloid" AFTER prediction.
#   Tier 2 (fine) : myeloid-ONLY SingleR (MG/MDM/BAM) on tier-1 PRUNED Myeloid
#                   (optionally intersected with the Python OvR-abstain Myeloid set).
#
# Fixes from review:
#   #2 keep MG/MDM/BAM as labels in tier 1 (SingleR needs within-ref contrasts);
#      collapse to "Myeloid" only after prediction.
#   #3 use PRUNED tier-1 labels to choose tier-2 cells.
#   #4 harmonize gene case + collapse duplicate genes by SUMMING COUNTS, before
#      logNormCounts (never sum logcounts).
#   #5 explicit file-existence + counts/metadata/cells alignment checks.
#   #6 Ochocka mapping is manual; stops if the annotation column / names are wrong,
#      and validates the map by in-reference marker expression.
# =============================================================================

suppressPackageStartupMessages({
  library(SingleR); library(SingleCellExperiment); library(scuttle)
  library(TabulaMurisSenisData); library(Seurat)
})

# ----------------------------- CONFIG (edit paths) --------------------------
SLICE       <- 1
DATADIR     <- "D:/thesis-research/cosmx_export"
OCHOCKA_RDS <- "D:/thesis-research/nanostring_resources/ochocka_GSE136001.rds"
D122_RDS    <- "D:/thesis-research/nanostring_resources/d122_tumor_ref.rds"
OVR_CSV     <- sprintf("D:/thesis-research/score_genes_slice%d_merged/classify/ovr_nontumor_predictions.csv", SLICE)
OUT_CSV     <- sprintf("D:/thesis-research/score_genes_slice%d_merged/classify/singler_two_tier.csv", SLICE)
ANNO        <- "cluster"             # Ochocka annotation column (script stops if absent)
COORD_X     <- "CenterX_global_px"
COORD_Y     <- "CenterY_global_px"
CLUSTER_COL <- "leiden"              # test cluster column, or NA for per-cell
# ----------------------------------------------------------------------------

# ----------------------------- helpers --------------------------------------
need <- function(p) if (!file.exists(p)) stop("missing file: ", p, call. = FALSE)

# harmonize gene-name case + collapse duplicate genes by SUMMING COUNTS
collapse_upper <- function(cnt) {
  rn <- toupper(rownames(cnt))
  if (!anyDuplicated(rn)) { rownames(cnt) <- rn; return(cnt) }
  message("  collapsing ", sum(duplicated(rn)), " duplicate gene(s) by summing counts")
  rowsum(as.matrix(cnt), group = rn)             # coerce only when dups exist
}

# build an SCE from raw counts: collapse genes (on counts) THEN logNormCounts
prep <- function(cnt, coldata) {
  stopifnot(ncol(cnt) == nrow(coldata))
  s <- SingleCellExperiment(assays = list(counts = collapse_upper(cnt)),
                            colData = coldata)
  logNormCounts(s)
}

run_singler <- function(test, ref, labels, clusters = NULL) {
  pred <- SingleR(test = test, ref = ref, labels = labels, clusters = clusters,
                  assay.type.test = "logcounts", assay.type.ref = "logcounts",
                  de.method = "wilcox")
  if (is.null(clusters)) {
    data.frame(label = pred$labels, pruned = pred$pruned.labels,
               stringsAsFactors = FALSE)
  } else {
    cl <- as.character(clusters)                 # expand cluster-level to cells
    data.frame(label = pred[cl, "labels"], pruned = pred[cl, "pruned.labels"],
               stringsAsFactors = FALSE)
  }
}

broad <- function(v) { v[v %in% c("Microglia", "MDM", "BAM")] <- "Myeloid"; v }  # NA-safe

# ============================ 1. TEST (CosMx slice) =========================
cf <- file.path(DATADIR, sprintf("slice_%d_counts.csv", SLICE))
mf <- file.path(DATADIR, sprintf("slice_%d_metadata.csv", SLICE))
gf <- file.path(DATADIR, sprintf("slice_%d_genes.txt", SLICE))
xf <- file.path(DATADIR, sprintf("slice_%d_cells.txt", SLICE))
invisible(lapply(c(cf, mf, gf, xf), need))

counts   <- as.matrix(read.csv(cf, row.names = 1, check.names = FALSE))  # cells x genes
metadata <- read.csv(mf, check.names = FALSE)
genes <- trimws(readLines(gf)); genes <- genes[nzchar(genes)]
cells <- readLines(xf)

# alignment checks BEFORE building the object
stopifnot(nrow(counts) == length(cells),        # rows = cells
          ncol(counts) == length(genes),        # cols = genes
          nrow(metadata) == length(cells))
if (!is.null(rownames(counts)) && !all(rownames(counts) == cells))
  warning("counts rownames != cells file order -- verify they are the same cells")

counts_t <- t(counts); rownames(counts_t) <- genes; colnames(counts_t) <- cells
keep <- colSums(counts_t) > 0                    # drop empty cells (NaN sizeFactors)
sce  <- prep(counts_t[, keep, drop = FALSE], DataFrame(metadata[keep, , drop = FALSE]))
cat(sprintf("test: %d genes x %d cells (%d empty dropped)\n",
            nrow(sce), ncol(sce), sum(!keep)))

# ============================ 2. REFERENCES ================================
## 2a. stromal / neuronal (Tabula Muris Senis FACS, Brain_Non-Myeloid)
r0 <- TabulaMurisSenisFACS(tissues = "Brain_Non-Myeloid")[[1]]
cd <- colData(r0); cd$label <- as.character(r0$cell_ontology_class)
ref_struct <- prep(counts(r0), cd)

## 2b. myeloid (Ochocka 2021 GL261 glioma, GSE136001) -- RAW RNA counts, MANUAL map
need(OCHOCKA_RDS)
rmye0 <- as.SingleCellExperiment(readRDS(OCHOCKA_RDS), assay = "RNA")
if (!ANNO %in% colnames(colData(rmye0)))
  stop("set ANNO; available columns: ", paste(colnames(colData(rmye0)), collapse = ", "))
cat("\nOchocka clusters:\n"); print(table(colData(rmye0)[[ANNO]]))

## EDIT LHS to the real cluster names printed above
map <- c(
  "Homeostatic Microglia" = "Microglia", "Activated Microglia" = "Microglia",
  "Monocytes"             = "MDM",       "Mo/MF"               = "MDM",
  "intermediate MF"       = "MDM",
  "Perivascular MF"       = "BAM",       "Meningeal MF"        = "BAM",
  "Choroid plexus MF"     = "BAM"
)
lab <- unname(map[as.character(colData(rmye0)[[ANNO]])])
if (all(is.na(lab)))
  stop("map produced all NA -- LHS names do not match the cluster table above")
km <- !is.na(lab)
cd <- colData(rmye0)[km, , drop = FALSE]; cd$label <- lab[km]
ref_mye <- prep(counts(rmye0)[, km, drop = FALSE], cd)   # genes now UPPER-case

## validate the map: each label's own markers must be highest in its own column
val <- list(Microglia = c("Tmem119","P2ry12","Sall1","Hexb"),
            MDM       = c("Ccr2","Ly6c2","Plac8","Vim"),
            BAM       = c("Mrc1","Lyve1","Cd163","Pf4"))
cat("\n-- in-reference mean logcounts (own column should be largest) --\n")
for (l in names(val)) {
  gs <- intersect(toupper(val[[l]]), rownames(ref_mye))
  if (length(gs)) {
    mm <- vapply(c("Microglia","MDM","BAM"), function(L)
      mean(logcounts(ref_mye)[gs, ref_mye$label == L, drop = FALSE]), numeric(1))
    cat(sprintf("%-10s -> MG %.2f | MDM %.2f | BAM %.2f\n", l, mm[1], mm[2], mm[3]))
  }
}

## 2c. tumor reference (yours)
need(D122_RDS)
d <- readRDS(D122_RDS)
if (is(d, "Seurat")) d <- as.SingleCellExperiment(d, assay = "RNA")
cd <- colData(d); cd$label <- as.character(d$cell_type)
ref_d122 <- prep(counts(d), cd)

# ============================ 3. TIER 1 (broad) ============================
# keep MG/MDM/BAM as labels here (contrasts!), collapse to Myeloid AFTER predicting
clusters <- if (!is.na(CLUSTER_COL) && CLUSTER_COL %in% colnames(colData(sce)))
  as.character(colData(sce)[[CLUSTER_COL]]) else NULL

t1 <- run_singler(
  test   = sce,
  ref    = list(struct = ref_struct, myeloid = ref_mye, tumor = ref_d122),
  labels = list(ref_struct$label, ref_mye$label, ref_d122$label),
  clusters = clusters)
sce$tier1_fine   <- t1$pruned                 # may already be MG/MDM/BAM
sce$tier1_pruned <- broad(t1$pruned)          # MG/MDM/BAM -> Myeloid; NA preserved
cat("\nTier-1 broad (pruned):\n"); print(table(sce$tier1_pruned, useNA = "always"))

# ============================ 4. myeloid-like (PRUNED) =====================
myeloid_like <- sce$tier1_pruned %in% "Myeloid"     # %in% -> NA treated as FALSE
if (file.exists(OVR_CSV) && all(c(COORD_X, COORD_Y) %in% colnames(colData(sce)))) {
  ovr <- read.csv(OVR_CSV)
  k_sce <- paste(round(colData(sce)[[COORD_X]], 1), round(colData(sce)[[COORD_Y]], 1))
  k_ovr <- paste(round(ovr$x, 1), round(ovr$y, 1))
  sce$ovr_label <- ovr$final_label[match(k_sce, k_ovr)]
  myeloid_like  <- myeloid_like & (sce$ovr_label %in% "Myeloid")
  cat(sprintf("\nmyeloid-like = tier1(pruned) Myeloid AND OvR Myeloid: %d\n",
              sum(myeloid_like)))
} else {
  cat(sprintf("\nmyeloid-like = tier1(pruned) Myeloid: %d (OvR join skipped)\n",
              sum(myeloid_like)))
}

# ============================ 5. TIER 2 (myeloid-only) =====================
sce$myeloid_subtype <- NA_character_
if (sum(myeloid_like) >= 10) {
  t2 <- run_singler(test = sce[, myeloid_like], ref = ref_mye, labels = ref_mye$label)
  sce$myeloid_subtype[myeloid_like] <- t2$pruned    # pruned: low-conf -> NA
  cat("\nTier-2 myeloid subtypes (pruned):\n"); print(table(t2$pruned, useNA = "always"))
}

# ============================ 6. combine + save ============================
final <- sce$tier1_pruned
final[myeloid_like] <- ifelse(is.na(sce$myeloid_subtype[myeloid_like]),
                              "Myeloid_unclassified", sce$myeloid_subtype[myeloid_like])
sce$final_singler <- final

out <- data.frame(
  cell = colnames(sce),
  x = if (COORD_X %in% colnames(colData(sce))) colData(sce)[[COORD_X]] else NA,
  y = if (COORD_Y %in% colnames(colData(sce))) colData(sce)[[COORD_Y]] else NA,
  tier1_fine = sce$tier1_fine, tier1_pruned = sce$tier1_pruned,
  myeloid_subtype = sce$myeloid_subtype, final_singler = sce$final_singler,
  ovr_label = if ("ovr_label" %in% colnames(colData(sce))) sce$ovr_label else NA,
  stringsAsFactors = FALSE)
write.csv(out, OUT_CSV, row.names = FALSE)
cat(sprintf("\nsaved -> %s\n", OUT_CSV))

# ============================ 7. concordance (optional) ====================
if ("ovr_label" %in% colnames(colData(sce))) {
  cat("\n-- concordance: SingleR final vs OvR-abstain --\n")
  print(table(SingleR = sce$final_singler, OvR = sce$ovr_label))
}
