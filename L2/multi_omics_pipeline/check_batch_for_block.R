.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

suppressPackageStartupMessages({
  library(Seurat)
})

script_dir <- "d:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
setwd(script_dir)

for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("check_batch_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")

rds_path <- file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds")
log_info("Loading: ", basename(rds_path))
sc_seu <- readRDS(rds_path)
log_info("sc_seu: ", ncol(sc_seu), " cells, ", nrow(sc_seu), " genes")
log_info("meta.data columns: ", paste(colnames(sc_seu@meta.data), collapse = ", "))

# 检查可能的 batch 列
batch_candidates <- c("batch", "Batch", "sample", "Sample", "orig.ident",
                       "dataset", "Dataset", "donor", "Donor")
batch_col <- NULL
for (col in batch_candidates) {
  if (col %in% colnames(sc_seu@meta.data)) {
    batch_col <- col
    break
  }
}

if (is.null(batch_col)) {
  log_warn("No batch column found in sc_seu meta.data (checked: ",
           paste(batch_candidates, collapse = ", "), ")")
  log_info("Available columns: ", paste(colnames(sc_seu@meta.data), collapse = ", "))
  quit(status = 0)
}

log_info("Found batch column: '", batch_col, "'")
batch_table <- table(sc_seu@meta.data[[batch_col]])
log_info("Batch distribution:")
for (b in names(batch_table)) {
  log_info(sprintf("  %s: %d cells", b, batch_table[b]))
}

# 检查 batch x celltype 交叉表
celltype_col <- cfg$data$sc_celltype_col
log_info("\nBatch x celltype cross-tabulation:")
crosstab <- table(sc_seu@meta.data[[batch_col]], sc_seu@meta.data[[celltype_col]])
print(crosstab)

# 检查每个 batch 中各 celltype 的细胞数 (>=10 才算稳定)
log_info("\nCells per celltype per batch (>=10 = stable):")
for (b in rownames(crosstab)) {
  for (ct in colnames(crosstab)) {
    n <- crosstab[b, ct]
    status <- if (n >= 10) "OK" else if (n > 0) "LOW" else "ABSENT"
    cat(sprintf("  batch=%s celltype=%s n=%d [%s]\n", b, ct, n, status))
  }
}

# 统计每个 celltype 在多少个 batch 中有 >=10 cells
log_info("\nCelltype presence across batches:")
for (ct in colnames(crosstab)) {
  n_batches_stable <- sum(crosstab[, ct] >= 10)
  n_batches_present <- sum(crosstab[, ct] > 0)
  cat(sprintf("  %s: stable(>=10) in %d/%d batches, present(>0) in %d/%d\n",
              ct, n_batches_stable, nrow(crosstab),
              n_batches_present, nrow(crosstab)))
}

# 结论
log_info("\n=== Block parameter decision ===")
all_stable <- all(apply(crosstab, 2, function(x) sum(x >= 10) == length(x)))
if (all_stable) {
  log_info("[RECOMMEND] Add block=", batch_col, " to scoreMarkers:")
  log_info("  - All celltypes have >=10 cells in all batches")
  log_info("  - block= will perform intra-batch comparisons, robust to batch effects")
  log_info("  - Aaron Lun (simpleSingleCell vignette): 'Intra-batch comparisons with block= are robust'")
} else {
  log_info("[CAUTION] Some celltypes have <10 cells in some batches")
  log_info("  - block= may return NA for pairs not co-occurring in same batch")
  log_info("  - Recommend: do NOT use block=, rely on Harmony-integrated logcounts")
  log_info("  - Or: use block= with fallback to no-block if too many NAs")
}
