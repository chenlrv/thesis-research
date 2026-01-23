# qcFlagsCellCounts - Cell failed QC based on RNA count thresholds
# qcFlagsCellPropNeg - Cell failed QC due to high negative-probe proportion
# qcFlagsCellComplex - Cell failed QC due to low complexity
# qcFlagsCellArea - Cell failed QC due to abnormal area




# qc(adata)

# At the cell level, look for transcripts per cell > ~200. If too many cells are flagged (30% or more), consider
# reducing this threshold. Transcripts per cell depends on many factors in study design and sample biology


# Normalization
# l Total Counts Normalization is generally recommended as it keeps the data on a linear scale, is easily
# interpretable, and is quick to run.
# l Other transformations (log1p, Pearson, sctransform) are possible and may improve visualizations in some
# datasets. However, the Pearson method is very resource- and time-intensive, so it is only recommended for
# smaller datasets (using lower plex panels than WTX).


# 4. PCA
# l Calculate 50 principal components from normalized counts.
# 5. UMAP
# l Recommended parameters (optimal parameters are project-dependent; these are suggested as a starting
# point):
# o Minimum distance = 0.01; lower minimum distance generates more clusters.
# o Spread = 5 or 2; higher spread yields more separation of clusters.
# o Neighbors = 30; keep between 20-40; higher value yields more distinct clusters.
# o Metric = cosine.
# o Use between 15-50 principal components.

