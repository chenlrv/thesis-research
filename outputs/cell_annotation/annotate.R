library(SingleCellExperiment)
library(scater)
library(TabulaMurisSenisData)
library(scater)
library(SingleR)
library(Seurat)
library(ggplot2)
library(data.table)
library(Matrix)
library(Seurat)
library(scPearsonPCA)
library(ggrepel)
library(celldex)
library(SingleR)
library(scuttle)
library(Matrix)

data_dir <- "D:/thesis-research/outputs/cell_annotation/D122_Reference_Avinoam/GSE103548_GeneCount_raw.tsv.gz"
ref_df <- read.delim(data_dir, check.names = FALSE)

# 1) use GeneID as rownames (usually unique and stable)
genes_ref <- make.unique(ref_df$GeneName)
counts_mat <- as.matrix(ref_df[, grepl("^LLC[1-4]_GeneCount$", colnames(ref_df)), drop = FALSE])
rownames(counts_mat) <- genes_ref

# (optional) make sure numeric
storage.mode(counts_mat) <- "integer"

# 2) build LLC-only tumor reference
ref_llc <- SingleCellExperiment(list(counts = counts_mat))
colData(ref_llc)$cell_type <- rep("Tumor", ncol(counts_mat))

# 3) log-normalize (SingleR expects log-scale)
ref_llc <- logNormCounts(ref_llc)

# sanity check
cat(colnames(counts_mat))
cat(dim(counts_mat))



setwd("D:/thesis-research/outputs/cell_annotation/L321/05")

counts <- readMM("slice_3_counts_healthy.mtx")
cat(dim(counts))
metadata <- read.csv("slice_3_metadata_healthy.csv", check.names = FALSE)
genes <- readLines("slice_3_genes_healthy.txt")
genes <- trimws(genes[nzchar(genes)])
cells <- readLines("slice_3_cells_healthy.txt")

counts_t <- t(counts)
rownames(counts_t) <- genes

sce <- SingleCellExperiment(
  assays = list(counts = counts_t),
  colData = metadata
)

sce <- logNormCounts(sce)

common_genes <- intersect(rownames(sce), rownames(ref_llc))
n_overlap <- length(common_genes)

# Load the three tissues we discussed
ref_brain_struct <- TabulaMurisSenisFACS(tissues = "Brain_Non-Myeloid")[[1]]
ref_brain_immune <- TabulaMurisSenisFACS(tissues = "Brain_Myeloid")[[1]]


# 3. Now logNormCounts will work perfectly
ref_brain_struct <- logNormCounts(ref_brain_struct)
ref_brain_immune <- logNormCounts(ref_brain_immune)


cat("Running predictions...")

pred_results_avinoam <- SingleR(
  test = sce,
  ref = list(Brain_Struct=ref_brain_struct, Brain_Immune=ref_brain_immune, Tumor_LLC=ref_llc),
  labels = list(ref_brain_struct$cell_ontology_class, ref_brain_immune$cell_ontology_class, ref_llc$cell_type),
  assay.type.test="logcounts",
  assay.type.ref="logcounts",
  de.method = "classic"
)
