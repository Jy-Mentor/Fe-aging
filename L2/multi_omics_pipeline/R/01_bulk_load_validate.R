# ============================================================================
# STEP 01: L1 Bulk RNA-seq 数据加载与验证
# - 读取 GSE233811 count 矩阵 + 表型数据
# - 严格数据完整性检查 (维度/缺失值/数据类型)
# - 构建 DESeq2 对象 (design = ~ time)
# 参考: Love MI et al. 2014 Genome Biology (DESeq2)
# ============================================================================

step01_bulk_load_validate <- function(cfg) {
  log_info("[Step01-L1] Bulk RNA-seq data loading & validation...")

  require_packages(c("DESeq2", "yaml"),
                   install_hint = "BiocManager::install('DESeq2')")
  suppressPackageStartupMessages(library(DESeq2))

  counts_path <- cfg$data$bulk_counts
  pheno_path  <- cfg$data$bulk_pheno

  if (!file.exists(counts_path)) {
    stop("Bulk counts file not found: ", counts_path,
         ". Please download GSE233811 supplementary files from GEO.")
  }
  if (!file.exists(pheno_path)) {
    stop("Bulk pheno file not found: ", pheno_path)
  }

  # --------------------------------------------------------------------------
  # 1.1 读取 count 矩阵
  # --------------------------------------------------------------------------
  log_info("[Step01] Reading counts: ", counts_path)
  count_data <- read.csv(counts_path, row.names = 1, check.names = FALSE)
  log_info("[Step01] Raw counts: ", nrow(count_data), " genes x ", ncol(count_data), " samples")

  # 确保是数值矩阵
  count_data <- as.matrix(count_data)
  if (!is.numeric(count_data)) {
    stop("Counts matrix must be numeric. Check column types.")
  }

  # count 必须为整数 (DESeq2 要求)
  if (any(count_data != floor(count_data), na.rm = TRUE)) {
    log_warn("[Step01] Counts contain non-integer values; rounding to integer.")
    count_data <- round(count_data)
  }

  # --------------------------------------------------------------------------
  # 1.2 读取表型数据
  # --------------------------------------------------------------------------
  log_info("[Step01] Reading phenotype: ", pheno_path)
  col_data <- read.csv(pheno_path, row.names = 1, stringsAsFactors = FALSE)
  log_info("[Step01] Pheno data: ", nrow(col_data), " samples, ",
           ncol(col_data), " metadata fields")

  # --------------------------------------------------------------------------
  # 1.3 一致性检查
  # --------------------------------------------------------------------------
  if (!all(rownames(col_data) %in% colnames(count_data))) {
    missing <- setdiff(rownames(col_data), colnames(count_data))
    stop("Pheno samples not in counts: ", paste(head(missing), collapse = ", "))
  }
  # 对齐
  col_data <- col_data[colnames(count_data), , drop = FALSE]

  # 检查时间列存在
  time_col <- cfg$data$bulk_time_col
  if (!(time_col %in% colnames(col_data))) {
    stop("Time column '", time_col, "' not found in pheno data. Available: ",
         paste(colnames(col_data), collapse = ", "))
  }

  # 时间点因子化 (按配置顺序)
  time_levels <- cfg$data$bulk_time_levels
  col_data[[time_col]] <- factor(col_data[[time_col]], levels = time_levels)
  log_info("[Step01] Time point distribution:")
  print(table(col_data[[time_col]]))

  # 检查缺失值
  n_na_counts <- sum(is.na(count_data))
  n_na_pheno  <- sum(is.na(col_data[[time_col]]))
  if (n_na_counts > 0) {
    frac <- n_na_counts / length(count_data)
    if (frac > 0.20) {
      stop("Counts missing value fraction = ", round(100 * frac, 2),
           "% (>20%). Data quality issue.")
    }
    log_warn("[Step01] Counts missing: ", n_na_counts, " (", round(100 * frac, 2), "%)")
  }
  if (n_na_pheno > 0) {
    stop("Pheno time column has ", n_na_pheno, " missing values. Cannot group samples.")
  }

  # --------------------------------------------------------------------------
  # 1.4 基因预过滤 (减少后续计算量)
  # --------------------------------------------------------------------------
  min_count <- cfg$bulk$min_count
  keep_genes <- rowSums(count_data) >= min_count
  log_info("[Step01] Gene prefilter (rowSums >= ", min_count, "): ",
           sum(keep_genes), "/", length(keep_genes), " retained")
  count_data <- count_data[keep_genes, , drop = FALSE]

  # --------------------------------------------------------------------------
  # 1.5 构建 DESeq2 对象
  # --------------------------------------------------------------------------
  formula_str <- paste0("~ ", time_col)
  design_formula <- as.formula(formula_str)
  log_info("[Step01] DESeq2 design formula: ", formula_str)

  dds <- DESeqDataSetFromMatrix(
    countData = count_data,
    colData   = col_data,
    design    = design_formula
  )

  log_info("[Step01] DESeq2 object created: ", nrow(dds), " genes x ", ncol(dds), " samples")

  # --------------------------------------------------------------------------
  # 1.6 数据完整性验证报告
  # --------------------------------------------------------------------------
  validation_report <- data.frame(
    item = c("n_genes_raw", "n_genes_filtered", "n_samples",
             "n_time_points", "time_levels",
             "missing_count_frac", "missing_time_n",
             "data_type", "design_formula"),
    value = c(nrow(count_data) + sum(!keep_genes),
              nrow(dds), ncol(dds),
              length(unique(col_data[[time_col]])),
              paste(levels(col_data[[time_col]]), collapse = "|"),
              sprintf("%.4f%%", 100 * n_na_counts / max(length(count_data), 1)),
              as.character(n_na_pheno),
              "integer counts",
              formula_str),
    stringsAsFactors = FALSE
  )
  save_table(validation_report, "01_bulk_validation_report", cfg)

  save_rds(dds, "01_bulk_dds_raw", cfg)

  log_info("[Step01] Bulk data loading & validation done.")
  invisible(dds)
}
