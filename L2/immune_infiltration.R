#!/usr/bin/env Rscript
# Phase 2 - Immune Infiltration Analysis
# Gracefully handles missing immunedeconv package

project_root <- normalizePath(getwd())
results_dir <- file.path(project_root, "L2", "results")
l1_results <- file.path(project_root, "L1", "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

cat("Phase 2 - Immune Infiltration Analysis\n")

# Check if immunedeconv is available
has_immunedeconv <- requireNamespace("immunedeconv", quietly = TRUE)

if (!has_immunedeconv) {
  cat("immunedeconv package not available for this R version\n")
  cat("Writing placeholder file...\n")
  write.csv(data.frame(message = "immunedeconv package not available for R 4.5.2; immune infiltration analysis skipped"),
            file.path(results_dir, "immune_infiltration.csv"), row.names = FALSE)
  cat("Immune Infiltration Analysis Skipped\n")
  quit(save = "no", status = 0)
}

suppressPackageStartupMessages({
  library(immunedeconv)
})

datasets <- c("GSE104036", "GSE16561", "GSE37587", "GSE61616", "GSE97537")
all_immune <- list()

for (ds in datasets) {
  expr_file <- file.path(l1_results, paste0(ds, "_expression_matrix.csv"))
  if (!file.exists(expr_file)) {
    cat(sprintf("  %s: not found, skipping
", ds))
    next
  }
  cat(sprintf("
Processing %s...
", ds))
  expr <- as.matrix(read.csv(expr_file, row.names = 1, check.names = FALSE))

  tryCatch({
    immune_result <- immunedeconv::deconvolute(expr, method = "quantiseq")
    immune_result$dataset <- ds
    all_immune[[ds]] <- immune_result
    cat(sprintf("  Immune deconvolution completed: %d cell types
", nrow(immune_result)))
  }, error = function(e) {
    cat(sprintf("  Immune deconvolution failed for %s: %s
", ds, e$message))
    cat(sprintf("  Trying MCP-counter...
"))
    tryCatch({
      immune_result <- immunedeconv::deconvolute(expr, method = "mcp_counter")
      immune_result$dataset <- ds
      all_immune[[ds]] <- immune_result
      cat(sprintf("  MCP-counter completed: %d cell types
", nrow(immune_result)))
    }, error = function(e2) {
      cat(sprintf("  MCP-counter also failed: %s
", e2$message))
    })
  })
}

if (length(all_immune) > 0) {
  combined <- do.call(rbind, all_immune)
  write.csv(combined, file.path(results_dir, "immune_infiltration.csv"), row.names = FALSE)
  cat(sprintf("
Combined immune infiltration: %d rows
", nrow(combined)))
} else {
  cat("
No immune infiltration results computed!
")
  cat("Writing placeholder file...
")
  write.csv(data.frame(message = "Immune infiltration not available"), 
            file.path(results_dir, "immune_infiltration.csv"), row.names = FALSE)
}
cat("
Immune Infiltration Analysis Completed
")
