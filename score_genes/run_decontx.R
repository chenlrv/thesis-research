#!/usr/bin/env Rscript
# decontX worker -- run directly in your R env.
#   RStudio:  set WORKDIR_PARENT below, then source() this file.
#   Terminal: Rscript run_decontx.R [parent_dir]
#
# Processes every <parent>/<slice>_work/ folder that contains counts.mtx
# (genes x cells) [+ optional clusters.csv with a 'cluster' column], writing into
# each: decontx_counts.mtx (genes x cells) and decontx_contamination.csv.
suppressMessages({
  library(Matrix)
  library(celda)
})

# ---- edit for interactive (RStudio) use, or pass the dir as a CLI arg ----
WORKDIR_PARENT <- "D:/thesis-research/resources/cache/decontx"

args <- commandArgs(trailingOnly = TRUE)
parent <- if (length(args) >= 1) args[1] else WORKDIR_PARENT

workdirs <- list.dirs(parent, recursive = FALSE)
workdirs <- workdirs[file.exists(file.path(workdirs, "counts.mtx"))]
if (length(workdirs) == 0) stop("no '*/counts.mtx' found under ", parent)

run_one <- function(workdir) {
  cat("decontX:", workdir, "\n")
  counts <- as(Matrix::readMM(file.path(workdir, "counts.mtx")), "CsparseMatrix")
  zfile <- file.path(workdir, "clusters.csv")
  if (file.exists(zfile)) {
    res <- decontX(counts, z = read.csv(zfile)$cluster)
  } else {
    res <- decontX(counts)
  }
  # matrix input -> list ($decontXcounts/$contamination); SCE input -> accessors
  if (methods::is(res, "SingleCellExperiment")) {
    dec  <- celda::decontXcounts(res)
    cont <- SummarizedExperiment::colData(res)$decontX_contamination
  } else {
    dec  <- res$decontXcounts
    cont <- res$contamination
  }
  Matrix::writeMM(as(dec, "CsparseMatrix"),
                  file.path(workdir, "decontx_counts.mtx"))
  write.csv(data.frame(contamination = cont),
            file.path(workdir, "decontx_contamination.csv"), row.names = FALSE)
  cat("  median contamination =", round(median(cont), 3), "\n")
}

for (wd in workdirs) run_one(wd)
cat("decontX: done (", length(workdirs), "slice(s)).\n")
