# ============================================================================
# STEP 01: L1 Bulk RNA-seq 数据加载与验证
# - 读取 GSE233815 bulk count 矩阵 + 表型数据 (GPL19057, 48 样本)
# - ENSEMBL ID (含版本号) → gene symbol 转换 (org.Mm.eg.db)
# - 严格数据完整性检查 (维度/缺失值/数据类型)
# - 构建 DESeq2 对象 (design = ~ time)
# 参考: Love MI et al. 2014 Genome Biology (DESeq2)
#         Carlson M 2016 org.Mm.eg.db (mouse genome annotation)
# ============================================================================

step01_bulk_load_validate <- function(cfg) {
  log_info("[Step01-L1] Bulk RNA-seq data loading & validation...")

  require_packages(c("DESeq2", "yaml", "org.Mm.eg.db", "AnnotationDbi"),
                   install_hint = "BiocManager::install(c('DESeq2','org.Mm.eg.db'))")
  suppressPackageStartupMessages({
    library(DESeq2)
    library(org.Mm.eg.db)
    library(AnnotationDbi)
  })

  counts_path <- cfg$data$bulk_counts
  pheno_path  <- cfg$data$bulk_pheno

  if (!file.exists(counts_path)) {
    stop("Bulk counts file not found: ", counts_path,
         ". Please download GSE233815 supplementary files from GEO.")
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
  # 1.1b ENSEMBL ID (含版本号) → gene symbol 转换
  # --------------------------------------------------------------------------
  # GSE233815 raw count matrix 使用 ENSMUSGxxxxxxxxxx.xx 格式
  # 下游 GSEA/WGCNA/可视化需要 gene symbol, 在此统一转换
  raw_ids <- rownames(count_data)
  is_ensembl <- grepl("^ENSMUSG[0-9]+(\\.[0-9]+)?$", raw_ids)
  if (sum(is_ensembl) / length(raw_ids) > 0.5) {
    log_info("[Step01] Detected ENSEMBL IDs (", sum(is_ensembl), "/",
             length(raw_ids), "). Converting to gene symbols via org.Mm.eg.db...")
    # 去除 version 后缀 (.xx)
    ensembl_no_ver <- sub("\\.[0-9]+$", "", raw_ids)
    # ENSEMBL → SYMBOL
    sym_map <- AnnotationDbi::select(
      org.Mm.eg.db,
      keys = ensembl_no_ver,
      columns = c("ENSEMBL", "SYMBOL", "ENTREZID"),
      keytype = "ENSEMBL"
    )
    # 一个 ENSEMBL 可能映射到多个 SYMBOL, 取第一个非 NA
    sym_map <- sym_map[!is.na(sym_map$SYMBOL), ]
    sym_map <- sym_map[!duplicated(sym_map$ENSEMBL), ]
    rownames(sym_map) <- sym_map$ENSEMBL

    symbols <- sym_map[ensembl_no_ver, "SYMBOL"]
    entrez_ids <- sym_map[ensembl_no_ver, "ENTREZID"]
    n_mapped <- sum(!is.na(symbols))
    log_info("[Step01] ENSEMBL → SYMBOL mapped: ", n_mapped, "/", length(raw_ids),
             " (", round(100 * n_mapped / length(raw_ids), 1), "%)")

    # 未映射的 ENSEMBL 保留原 ID (去版本号)
    unmapped_idx <- is.na(symbols)
    symbols[unmapped_idx] <- ensembl_no_ver[unmapped_idx]
    if (sum(unmapped_idx) > 0) {
      log_warn("[Step01] ", sum(unmapped_idx), " ENSEMBL IDs unmapped; kept as ID (version stripped).")
    }

    # 处理重复 SYMBOL: 保留 count 总和最高的那条, 其余加 _dup 后缀
    dup_table <- table(symbols)
    dup_symbols <- names(dup_table[dup_table > 1])
    if (length(dup_symbols) > 0) {
      log_warn("[Step01] ", length(dup_symbols),
               " duplicated SYMBOLs found. Resolving by keeping highest-count row.")
      for (ds in dup_symbols) {
        dup_idx <- which(symbols == ds)
        total_counts <- rowSums(count_data[dup_idx, , drop = FALSE], na.rm = TRUE)
        keep_pos <- dup_idx[which.max(total_counts)]
        # 未保留的加 _dup2, _dup3...
        suffix_counter <- 2
        for (idx in setdiff(dup_idx, keep_pos)) {
          symbols[idx] <- paste0(ds, "_dup", suffix_counter)
          suffix_counter <- suffix_counter + 1
        }
      }
    }

    # 保存原始 ENSEMBL ID 到环境 (后续可加入 mcols)
    ensembl_lookup <- data.frame(
      ensembl_id = raw_ids,
      ensembl_no_version = ensembl_no_ver,
      symbol = symbols,
      entrez_id = entrez_ids,
      stringsAsFactors = FALSE
    )

    rownames(count_data) <- symbols
    attr(count_data, "ensembl_lookup") <- ensembl_lookup
  } else {
    log_info("[Step01] Row names appear to be gene symbols; no ENSEMBL conversion performed.")
    ensembl_lookup <- NULL
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
  if (!is.null(ensembl_lookup)) {
    ensembl_lookup <- ensembl_lookup[keep_genes, , drop = FALSE]
  }

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

  # 将 ENSEMBL ID 注释到 mcols (便于追溯)
  if (!is.null(ensembl_lookup)) {
    mcols(dds) <- cbind(mcols(dds), ensembl_lookup)
    log_info("[Step01] Added ensembl_lookup to mcols(dds)")
  }

  log_info("[Step01] DESeq2 object created: ", nrow(dds), " genes x ", ncol(dds), " samples")

  # --------------------------------------------------------------------------
  # 1.6 数据完整性验证报告
  # --------------------------------------------------------------------------
  validation_report <- data.frame(
    item = c("n_genes_raw", "n_genes_filtered", "n_samples",
             "n_time_points", "time_levels",
             "missing_count_frac", "missing_time_n",
             "data_type", "design_formula",
             "id_type", "ensembl_mapped_pct"),
    value = c(nrow(count_data) + sum(!keep_genes),
              nrow(dds), ncol(dds),
              length(unique(col_data[[time_col]])),
              paste(levels(col_data[[time_col]]), collapse = "|"),
              sprintf("%.4f%%", 100 * n_na_counts / max(length(count_data), 1)),
              as.character(n_na_pheno),
              "integer counts",
              formula_str,
              if (!is.null(ensembl_lookup)) "ENSEMBL to SYMBOL" else "SYMBOL (native)",
              if (!is.null(ensembl_lookup))
                sprintf("%.1f%%", 100 * sum(!is.na(ensembl_lookup$entrez_id)) / nrow(ensembl_lookup))
              else "100.0%"),
    stringsAsFactors = FALSE
  )
  save_table(validation_report, "01_bulk_validation_report", cfg)

  # 保存 ENSEMBL → SYMBOL 映射表 (供审计追溯)
  if (!is.null(ensembl_lookup)) {
    save_table(ensembl_lookup, "01_bulk_ensembl_to_symbol_map", cfg)
  }

  save_rds(dds, "01_bulk_dds_raw", cfg)

  log_info("[Step01] Bulk data loading & validation done.")
  invisible(dds)
}
