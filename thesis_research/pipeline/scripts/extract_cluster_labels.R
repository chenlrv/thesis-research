library(Matrix)
library(data.table)
library(HieraType)

# ---- inputs assumed to already exist in your session ----
# counts   : sparse matrix (either cells x genes OR genes x cells)
# metadata : data.frame with at least cell_global_id (or cell_id) and clust (cluster labels)

# ---- paths ----
setwd("D:/thesis-research/resources")
outdir <- getwd()

metadata <- fread("metadata.csv")
clustmap <- fread("cluster.csv")
counts <- readMM("counts.mtx")
cells <- readLines("cells.txt")
genes <- readLines("genes.txt")
cells <- trimws(cells[nzchar(cells)])
genes <- trimws(genes[nzchar(genes)])

stopifnot(nrow(counts) == length(cells))
stopifnot(ncol(counts) == length(genes))
rownames(counts) <- cells
colnames(counts) <- genes

cat("cell_global_id length:", length(metadata$cell_global_id), "\n")
cat("counts dim:", dim(counts), "\n")
cat("has rownames:", !is.null(rownames(counts)), " has colnames:", !is.null(colnames(counts)), "\n")
cat("head metadata cell_global_id:", head(metadata$cell_global_id), "\n")
cat("head rownames(counts):", head(rownames(counts)), "\n")
cat("head colnames(counts):", head(colnames(counts)), "\n")

if (!is.null(rownames(counts))) cat("missing in rownames:", length(setdiff(metadata$cell_global_id, rownames(counts))), "\n")
if (!is.null(colnames(counts))) cat("missing in colnames:", length(setdiff(metadata$cell_global_id, colnames(counts))), "\n")


if ("cell_global_id" %in% names(clustmap)) {

  # rename cluster column to "cluster" if needed
  if (!"cluster" %in% names(clustmap)) {
    cand <- intersect(names(clustmap), c("cluster","leiden","louvain","seurat_clusters"))
    stopifnot(length(cand) == 1)
    setnames(clustmap, cand, "cluster")
  }
}

metadata <- merge(
  metadata,
  clustmap[, .(cell_global_id, cluster)],
  by = "cell_global_id",
  all.x = TRUE
)

cat("missing clust:", sum(is.na(metadata$cluster)), "\n")


dir.create(file.path(outdir, "processed_data"), showWarnings = FALSE)
markers_path <- file.path(outdir, "processed_data", "markers.RDS")


stopifnot("cluster" %in% colnames(metadata))
metadata$cell_global_id <- as.character(metadata$cell_global_id)

# ---- ensure counts is cells x genes and aligned to metadata$cell_id ----
# Prefer using rownames(counts) as cell IDs if present.
if (!is.null(rownames(counts)) && all(metadata$cell_global_id %in% rownames(counts))) {
  counts_cxg <- counts[metadata$cell_global_id, , drop = FALSE]          # cells x genes
} else if (!is.null(colnames(counts)) && all(metadata$cell_global_id %in% colnames(counts))) {
  counts_cxg <- Matrix::t(counts)[metadata$cell_global_id, , drop = FALSE]  # transpose to cells x genes
} else {
  stop("Cannot align counts to metadata$cell_id. Check row/col names of counts and metadata$cell_id.")
}

counts_gxc <- Matrix::t(counts_cxg)

# ---- compute or load markers ----
if (TRUE) {
  markers <- HieraType::clusterwise_foldchange_metrics(
    counts_gxc,                 # MUST be cells x genes
    metadata = metadata,
    cluster_column = "cluster",
    cellid_column  = "cell_global_id"
  )
  saveRDS(markers, file = markers_path)
} else {
  markers <- readRDS(markers_path)
}

# ---- pick top N markers per cluster + force-include housekeeping per cluster ----
markers$prioritystat <- (markers$cluster_expr + 0.025) / (markers$clusterprime_expr + 0.025)

hk <- c("Ppia", "B2m", "Uba52", "Tpt1")
markersshort <- c()
nperclust <- 5

for (cl in unique(markers$cluster)) {
  inds <- markers$cluster == cl
  if (sum(inds) > 0) {
    top <- markers$gene[inds][order(markers$prioritystat[inds], decreasing = TRUE)][1:min(nperclust, sum(inds))]
    hk_cl <- intersect(hk, markers$gene[inds])
    markersshort <- c(markersshort, top, hk_cl)
  }
}
markersshort <- unique(markersshort)

# ---- load lineage genes (creates "lineagegenes") ----
source("https://raw.githubusercontent.com/Nanostring-Biostats/CosMx-Analysis-Scratch-Space/Main/_code/vignette2/lineage_and_marker_genes.R")

# ---- useful genes = lineage + selected markers, restricted to genes in counts ----
allusefulmarkers <- intersect(colnames(counts_cxg), unique(c(unlist(lineagegenes), markersshort)))

# ---- build prompt text for cluster labeling ----
mytissuetype <- "mouse brain tissue"

prompt_preamble <- paste0(
  "Propose cell type names for the clusters I\'ve found in a CosMx study. This is data from a ", mytissuetype, ". ",
  "Below are two tables. The first is each cell type\'s abundance in the dataset. ",
  "The second is a table of mean expression levels of various marker genes and lineage-defining genes in each cluster. ",
  "It\'s possible that some clusters could be closely-related cell types, or distinct states of the same cell type. Call that out when it\'s apparent. ",
  "For each cluster, give me your best guess at its identity, and give me your justification. Let me know when you\'re uncertain. ",
  "Then, give me R code defining a named vector called \'cluster_labels\' in which my cluster IDs are the names and the values are your proposed cell types. ",
  "Check your work three times. "
)

cluster <- metadata$cluster
prompt_frequencies <- paste0(
  "\nThe cluster frequencies are: ",
  paste0(paste0("cluster ", names(table(cluster)), ":", table(cluster)), collapse = ", ")
)

# markers table -> gene x cluster matrix of mean expr
meanexpression_dt <- markers[, dcast(.SD, gene ~ cluster, value.var = "cluster_expr")]
meanexpression <- as.matrix(meanexpression_dt[, -1])   # drop gene col
rownames(meanexpression) <- meanexpression_dt$gene

meanexpression <- meanexpression[allusefulmarkers, , drop = FALSE]

# keep de novo markers OR lineage genes with max expr > 0.2
isdenovo <- rownames(meanexpression) %in% markersshort
meanexpression <- meanexpression[isdenovo | apply(meanexpression, 1, max) > 0.2, , drop = FALSE]

prompt_profiles <- paste0(
  "And here is a table of each cluster's mean expression of selected data- and biology-derived marker genes: ",
  paste(capture.output(
    write.table(round(meanexpression, 1), sep = ",", row.names = TRUE, col.names = TRUE)
  ), collapse = "\n")
)

message(paste0(prompt_preamble, "\n", prompt_frequencies, "\n", prompt_profiles))

# optional: save objects for later
unnamedclust <- cluster
unnamedmeanexpression <- meanexpression
