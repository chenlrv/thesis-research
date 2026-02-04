library(data.table)
library(Matrix)
library(Seurat)
library(scPearsonPCA)

setwd("../resources")

counts <- readMM("counts.mtx")
metadata <- read.csv("metadata.csv", check.names = FALSE)
genes <- readLines("genes.txt")
cells <- readLines("cells.txt")

cells <- trimws(cells[nzchar(cells)])
genes <- trimws(genes[nzchar(genes)])

# assign dimnames to counts.mtx (counts is cells x genes in your pipeline)
rownames(counts) <- cells
colnames(counts) <- genes

# metadata setup + align to cells
stopifnot("cell_id_unique" %in% colnames(metadata))
stopifnot("sample_ID" %in% colnames(metadata))

metadata$cell_id_unique <- as.character(metadata$cell_id_unique)
metadata$sample_ID <- as.character(metadata$sample_ID)
stopifnot(!anyDuplicated(metadata$cell_id_unique))

rownames(metadata) <- metadata$cell_id_unique

missing <- setdiff(cells, rownames(metadata))
cat("missing cells:", length(missing), "\n")
print(head(missing))


metadata <- metadata[cells, , drop = FALSE]   # reorder to match cells.txt

# build Seurat + get counts (genes x cells)
seu <- CreateSeuratObject(counts = t(counts), meta.data = metadata)

cat("seu dims:", dim(seu), "\n")          # should be genes x cells
cat("head genes:", head(rownames(seu)), "\n")
cat("head cells:", head(colnames(seu)), "\n")

hvgs <- rownames(seu)
mat <- GetAssayData(seu, assay = "RNA", layer = "counts")  # genes x cells

cat("mat dims:", dim(mat), "\n")          # genes x cells
cat("hvgs in mat:", sum(hvgs %in% rownames(mat)), "/", length(hvgs), "\n")

# total counts per cell
tc <- Matrix::colSums(mat)


# batch-aware gene frequencies (cells x genes input)
genefreq_batch <- gene_frequency(
  x = Matrix::t(counts),  # cells x genes
  obs = as.data.table(metadata)[, .(cell_id_unique, sample_ID)],
  cellid_colname = "cell_id_unique",
  batch_variable = "sample_ID"
)

cat("genefreq dims:", dim(genefreq_batch), "\n")
cat("hvgs in genefreq:", sum(hvgs %in% rownames(genefreq_batch)), "/", length(hvgs), "\n")

x <- mat[hvgs, ]  # hvgs x cells (genes x cells)

cat("dim(x):", dim(x), "\n")  # should be 958 882811
cat("length(nCount_RNA):", length(metadata$nCount_RNA), "\n")  # should be 882811
stopifnot(length(metadata$nCount_RNA) == ncol(x))

# batch-aware Pearson PCA (cells x hvgs input)
pcaobj <- sparse_quasipoisson_pca_seurat_batch(
  x,         # cells x hvgs
  totalcounts = metadata$nCount_RNA,
  grate = genefreq_batch[hvgs, ],  # hvgs x batches
  obs = as.data.table(metadata)[, .(cell_id_unique, sample_ID)],
  batch_variable = "sample_ID",
  cellid_colname = "cell_id_unique",
  scale.max = 10,
  do.scale = TRUE,
  do.center = TRUE
)

saveRDS(Embeddings(pcaobj$reduction.data), "X_pca_pearson_batch.rds")
saveRDS(pcaobj, "pcaobj_batch.RDS")

pca <- readRDS("X_pca_pearson_batch.rds")
write.csv(pca, "X_pca_pearson_batch.csv")
