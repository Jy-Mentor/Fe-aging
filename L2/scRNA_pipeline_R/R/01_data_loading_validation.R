# ============================================================================
# STEP 1: 数据加载与完整性验证
# - 加载 Seurat 对象
# - 验证 assays / layers / meta.data / reductions
# - 数据维度一致性、缺失值比例、数据类型
# - 缺失值 < 20%
# ============================================================================

step01_validate_seurat <- function(seu, cfg) {
  log_info("[Step1] Loading Seurat object...")
  rds_path <- file.path(cfg$project$root, cfg$data$seurat_object)
  if (!file.exists(rds_path)) stop("Seurat RDS not found: ", rds_path)
  seu <- readRDS(rds_path)
  log_info("[Step1] Loaded: {nrow(seu)} genes x {ncol(seu)} cells")

  # 1.1 Assays
  assays <- SeuratObject::Assays(seu)
  log_info("[Step1] Assays: {paste(assays, collapse=', ')}")
  stopifnot("RNA" %in% assays)

  # 1.2 Layers (Seurat v5)
  rna_layers <- SeuratObject::Layers(seu[["RNA"]])
  log_info("[Step1] RNA layers: {paste(rna_layers, collapse=', ')}")
  stopifnot("counts" %in% rna_layers)
  stopifnot("data" %in% rna_layers)

  # 1.3 Meta data
  meta_cols <- colnames(seu@meta.data)
  required_cols <- c(cfg$analysis$condition_col, cfg$analysis$celltype_col,
                     "nCount_RNA", "nFeature_RNA")
  missing_cols <- setdiff(required_cols, meta_cols)
  if (length(missing_cols) > 0) {
    stop("Required meta.data columns missing: ",
         paste(missing_cols, collapse = ", "))
  }
  log_info("[Step1] Meta columns: {paste(meta_cols, collapse=', ')}")

  # 1.4 Conditions
  cond_tab <- table(seu@meta.data[[cfg$analysis$condition_col]])
  log_info("[Step1] Conditions: {paste(names(cond_tab), cond_tab, sep='=', collapse=', ')}")
  expected_levels <- cfg$analysis$condition_levels
  actual_levels <- names(cond_tab)
  if (!all(expected_levels %in% actual_levels)) {
    stop("Unexpected condition levels. Expected: ",
         paste(expected_levels, collapse = ", "),
         " | Got: ", paste(actual_levels, collapse = ", "))
  }

  # 1.5 Cell types
  ct_tab <- table(seu@meta.data[[cfg$analysis$celltype_col]])
  log_info("[Step1] Cell types (n={length(ct_tab)}):")
  for (ct in names(ct_tab)) {
    log_info("  {ct}: {ct_tab[[ct]]}")
  }

  # 1.6 Reductions
  reductions <- SeuratObject::Reductions(seu)
  log_info("[Step1] Reductions: {paste(reductions, collapse=', ')}")

  # 1.7 Missing value check on expression matrix
  expr_data <- Seurat::GetAssayData(seu, assay = "RNA", layer = "data")
  log_info("[Step1] Expression matrix: {nrow(expr_data)} x {ncol(expr_data)}")
  log_info("[Step1] Matrix class: {class(expr_data)[1]}")
  if (inherits(expr_data, c("dgCMatrix", "dgRMatrix", "dgTMatrix", "CsparseMatrix"))) {
    # 稀疏矩阵: 缺失值检查通过采样
    set.seed(42)
    sample_idx <- sample(length(expr_data@x), min(1e6, length(expr_data@x)))
    sample_vals <- expr_data@x[sample_idx]
    missing_frac <- mean(is.na(sample_vals))
    log_info("[Step1] Sparse matrix - sampled {length(sample_idx)} non-zero entries, missing frac: {sprintf('%.4f%%', missing_frac*100)}")
  } else {
    missing_frac <- mean(is.na(expr_data))
    log_info("[Step1] Missing value fraction: {sprintf('%.4f%%', missing_frac*100)}")
  }
  if (is.na(missing_frac)) {
    log_warn("[Step1] Missing value fraction is NA - treating as 0 for validation")
    missing_frac <- 0
  }
  if (missing_frac >= 0.2) {
    stop("Missing value fraction >= 20%: ", missing_frac)
  }

  # 1.8 nFeature_RNA range check
  nfeat <- seu@meta.data$nFeature_RNA
  log_info("[Step1] nFeature_RNA: min={min(nfeat)}, median={median(nfeat)}, max={max(nfeat)}")

  # 1.9 percent.mt
  if ("percent.mt" %in% meta_cols) {
    pmt <- seu@meta.data$percent.mt
    log_info("[Step1] percent.mt: min={round(min(pmt),3)}, median={round(median(pmt),3)}, max={round(max(pmt),3)}")
  }

  validation_report <- data.frame(
    item = c("n_cells", "n_genes", "assays", "rna_layers", "conditions",
             "cell_types", "reductions", "missing_frac", "min_nFeature",
             "max_nFeature"),
    value = c(ncol(seu), nrow(seu),
              paste(assays, collapse = ";"),
              paste(rna_layers, collapse = ";"),
              paste(names(cond_tab), collapse = ";"),
              length(ct_tab),
              paste(reductions, collapse = ";"),
              sprintf("%.4f", missing_frac),
              min(nfeat), max(nfeat))
  )
  save_table(validation_report, "01_validation_report", cfg)

  log_info("[Step1] Validation PASSED.")
  return(seu)
}

seu <- step01_validate_seurat(seu, cfg)
saveRDS(seu, file.path(cfg$project$rds_dir, "seurat_loaded.rds"))
log_info("[Step1] Saved loaded Seurat to rds/seurat_loaded.rds")
