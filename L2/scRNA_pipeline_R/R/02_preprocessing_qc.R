# ============================================================================
# STEP 2: 预处理与质量控制可视化
# 数据已由 Zucha et al. 2023 QC 过滤；本步骤展示 QC 指标分布
# 并执行项目硬约束的阈值核对
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

step02_qc_visualization <- function(seu, cfg) {
  log_info("[Step2] QC visualization...")

  meta <- seu@meta.data
  meta$Condition <- factor(meta[[cfg$analysis$condition_col]],
                           levels = cfg$analysis$condition_levels)

  # 2.1 nFeature / nCount / percent.mt 分布
  p_feat <- ggplot(meta, aes(x = Condition, y = nFeature_RNA, fill = Condition)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.15, outlier.size = 0.3, fill = "white") +
    scale_fill_manual(values = CONDITION_COLORS) +
    labs(title = "nFeature_RNA per Condition", y = "Genes per cell") +
    theme_pub() + theme(legend.position = "none")

  p_count <- ggplot(meta, aes(x = Condition, y = nCount_RNA, fill = Condition)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.15, outlier.size = 0.3, fill = "white") +
    scale_fill_manual(values = CONDITION_COLORS) +
    labs(title = "nCount_RNA per Condition", y = "UMIs per cell") +
    theme_pub() + theme(legend.position = "none")

  mt_col <- if ("percent.mt" %in% colnames(meta)) "percent.mt" else NULL
  if (!is.null(mt_col)) {
    p_mt <- ggplot(meta, aes(x = Condition, y = percent.mt, fill = Condition)) +
      geom_violin(trim = FALSE, alpha = 0.7) +
      geom_boxplot(width = 0.15, outlier.size = 0.3, fill = "white") +
      scale_fill_manual(values = CONDITION_COLORS) +
      labs(title = "Mitochondrial % per Condition", y = "% mito genes") +
      theme_pub() + theme(legend.position = "none")
    qc_combined <- (p_feat | p_count | p_mt) + plot_layout(nrow = 1)
  } else {
    qc_combined <- (p_feat | p_count) + plot_layout(nrow = 1)
  }
  save_figure(qc_combined, "02_qc_violin_by_condition", cfg,
              width = 14, height = 5)

  # 2.2 nFeature vs nCount 散点
  p_scatter <- ggplot(meta, aes(x = nCount_RNA, y = nFeature_RNA,
                                color = Condition)) +
    geom_point(alpha = 0.4, size = 0.8) +
    scale_color_manual(values = CONDITION_COLORS) +
    geom_hline(yintercept = c(cfg$qc$min_nFeature_RNA,
                              cfg$qc$max_nFeature_RNA),
               linetype = "dashed", color = "grey40") +
    labs(title = "nCount vs nFeature", x = "nCount_RNA",
         y = "nFeature_RNA") +
    theme_pub()
  save_figure(p_scatter, "02_qc_count_vs_feature", cfg, width = 8, height = 6)

  # 2.3 阈值核对
  threshold_report <- data.frame(
    metric = c("min_nFeature_RNA", "max_nFeature_RNA", "max_percent_mt",
               "n_cells_below_min_nFeat", "n_cells_above_max_nFeat",
               "n_cells_above_max_mt",
               "median_nFeature", "median_nCount"),
    threshold = c(cfg$qc$min_nFeature_RNA, cfg$qc$max_nFeature_RNA,
                  cfg$qc$max_percent_mt, NA, NA, NA, NA, NA),
    observed = c(min(meta$nFeature_RNA), max(meta$nFeature_RNA),
                 if (!is.null(mt_col)) max(meta[[mt_col]]) else NA,
                 sum(meta$nFeature_RNA < cfg$qc$min_nFeature_RNA),
                 sum(meta$nFeature_RNA > cfg$qc$max_nFeature_RNA),
                 if (!is.null(mt_col)) sum(meta[[mt_col]] > cfg$qc$max_percent_mt) else NA,
                 median(meta$nFeature_RNA),
                 median(meta$nCount_RNA))
  )
  save_table(threshold_report, "02_qc_threshold_report", cfg)

  # 2.4 SCT 标准化数据是否已存在
  has_sct <- "SCT" %in% SeuratObject::Assays(seu)
  log_info("[Step2] SCT assay present: {has_sct}")

  # 2.5 基因在细胞中表达比例
  expr_counts <- Seurat::GetAssayData(seu, assay = "RNA", layer = "counts")
  genes_expressed_in_20pct <- sum(Matrix::rowMeans(expr_counts > 0) >= 0.20)
  total_genes <- nrow(expr_counts)
  log_info("[Step2] Genes expressed in >=20% cells: {genes_expressed_in_20pct}/{total_genes}")

  qc_summary <- list(
    n_cells = ncol(seu),
    n_genes = nrow(seu),
    median_nFeature = median(meta$nFeature_RNA),
    median_nCount = median(meta$nCount_RNA),
    has_sct = has_sct,
    genes_in_20pct_cells = genes_expressed_in_20pct
  )
  saveRDS(qc_summary, file.path(cfg$project$rds_dir, "qc_summary.rds"))

  log_info("[Step2] QC visualization done.")
  invisible(seu)
}

seu <- step02_qc_visualization(seu, cfg)
