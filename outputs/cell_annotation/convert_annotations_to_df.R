
get_cell_ids_from_pred <- function(sce, pred_obj, id_col = "cell_global_id") {

  
  if (!(id_col %in% colnames(colData(sce)))) stop(paste0("Missing ", id_col, " in sce colData."))
  ids <- as.character(colData(sce)[[id_col]])
  if (anyNA(ids) || any(ids == "")) stop(paste0(id_col, " has NA/empty."))
  if (anyDuplicated(ids)) stop(paste0(id_col, " has duplicates; must be unique."))
  return(ids)
}

cell_ids <- get_cell_ids_from_pred(sce, pred_results_avinoam, id_col = "cell_global_id")
score_brain_struct <- as.numeric(pred_results_avinoam$scores$Brain_Struct$scores)
score_brain_immune <- as.numeric(pred_results_avinoam$scores$Brain_Immune$scores)
score_tumor   <- as.numeric(pred_results_avinoam$scores$Tumor_LLC$scores)

final_scores <- pmax(score_brain_struct, score_brain_immune, score_tumor, na.rm = TRUE)

annotation_results <- data.frame(
  cell_barcode            = get_cell_ids_from_pred(sce, pred_results_avinoam, id_col = "cell_global_id"),
  slice_id                = as.character(sce$slice_id),
  predicted_cell_type     = as.character(pred_results_avinoam$labels),
  predicted_tissue_origin = as.character(pred_results_avinoam$reference),
  pruned_label            = as.character(pred_results_avinoam$pruned.labels),
  score_brain_struct      = score_brain_struct,
  score_brain_immune      = score_brain_immune,
  score_tumor        = score_tumor,
  final_confidence        = final_scores,
  stringsAsFactors        = FALSE
)

stopifnot(nrow(annotation_results) == ncol(sce))
write.csv(annotation_results, "slice_6_final_cell_annotations_refined_tabula_brain_tumor_avinoam.csv", row.names = FALSE)