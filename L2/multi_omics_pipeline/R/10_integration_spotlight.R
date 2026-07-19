# ============================================================================
# STEP 10: L4 SPOTlight 空间去卷积
# - 用 scRNA (GSE233815) 训练细胞类型特异 marker
# - 将细胞类型比例投影到空间切片 (GSE233814)
# - 比较梗死核心 / 半暗带 / 健康区的细胞类型比例
# - 评估神经元比例与铁衰老得分的相关性
# 参考:
#   - Macosko EZ et al. 2015 (SPOTlight based on NMFreg, PMID: 26000488)
#   - Moncada R et al. 2020 Nat Commun (SPOTlight, PMID: 31844000)
#   - Diaz-Mejia JJ et al. 2019 Mol Syst Biol (BuildSimilarity)
# ============================================================================

step10_integration_spotlight <- function(sc_seu, spatial_merged, cfg) {
  log_info("[Step10-L4] SPOTlight spatial deconvolution...")

  require_packages(c("SPOTlight", "SingleCellExperiment"),
                   install_hint = paste("BiocManager::install(c('SPOTlight',",
                                        "'SingleCellExperiment'))"))
  suppressPackageStartupMessages({
    library(Seurat)
    library(SPOTlight)
    library(SingleCellExperiment)
    library(ggplot2)
  })

  if (is.null(sc_seu) || is.null(spatial_merged)) {
    stop("step10: sc_seu or spatial_merged is NULL. Run step07/08 and step04-06 first.")
  }

  celltype_col <- cfg$data$sc_celltype_col
  if (!(celltype_col %in% colnames(sc_seu@meta.data))) {
    stop("step10: celltype column '", celltype_col, "' not in sc_seu meta.data")
  }

  # --------------------------------------------------------------------------
  # 10.1 准备单细胞 SCE 对象 (SPOTlight 要求 SCE)
  # --------------------------------------------------------------------------
  log_info("[Step10] Converting sc Seurat -> SingleCellExperiment...")
  if (inherits(sc_seu[["RNA"]], "Assay5")) {
    sc_seu[["RNA"]] <- JoinLayers(sc_seu[["RNA"]])
  }
  sc_sce <- as.SingleCellExperiment(sc_seu, assay = "RNA")

  # 取 logcounts (默认 Seurat data slot 对应 logcounts)
  logcounts(sc_sce) <- as.matrix(logcounts(sc_sce))

  # --------------------------------------------------------------------------
  # 10.2 计算各细胞类型 marker 基因
  # --------------------------------------------------------------------------
  log_info("[Step10] Finding cell-type markers (FindAllMarkers)...")
  # Seurat v5: 默认使用 presto 加速; 若不可用则 Wilcoxon
  all_markers <- tryCatch({
    FindAllMarkers(sc_seu, only.pos = TRUE,
                    min.pct = 0.25, logfc.threshold = 0.25,
                    test.use = "wilcox")
  }, error = function(e) {
    log_warn("[Step10] FindAllMarkers wilcox failed: ", conditionMessage(e),
             "; retrying with presto")
    FindAllMarkers(sc_seu, only.pos = TRUE,
                    min.pct = 0.25, logfc.threshold = 0.25)
  })

  # 过滤显著 marker (FDR<0.05), 取每细胞类型 top N
  all_markers <- all_markers[all_markers$p_val_adj < 0.05, ]
  top_n <- cfg$integration$spotlight_n_top_mgs
  mgs_top <- do.call(rbind, lapply(split(all_markers, all_markers$cluster),
                                     function(x) head(x[order(-x$avg_log2FC), ],
                                                       n = top_n)))
  log_info("[Step10] Top markers per cell type: ", nrow(mgs_top),
           " (n=", top_n, " per type, ", length(unique(mgs_top$cluster)), " types)")

  save_table(mgs_top, "10_spotlight_top_markers", cfg)

  # --------------------------------------------------------------------------
  # 10.3 对每个空间切片运行 SPOTlight 去卷积
  # --------------------------------------------------------------------------
  spatial_conds <- unique(spatial_merged$condition)
  spotlight_results <- list()

  for (cond in spatial_conds) {
    log_info("[Step10] Deconvolving spatial sample: ", cond)
    sp_sub <- subset(spatial_merged, subset = condition == cond)
    if (ncol(sp_sub) == 0) next

    # SPOTlight 输入: 空间数据 assay 矩阵
    sp_assay <- "SCT" %in% names(sp_sub@assays)
    sp_mat <- if (sp_assay) {
      as.matrix(GetAssayData(sp_sub, assay = "SCT", layer = "data"))
    } else {
      as.matrix(GetAssayData(sp_sub, assay = "Spatial", layer = "data"))
    }

    # 共享基因
    shared_genes <- intersect(rownames(logcounts(sc_sce)), rownames(sp_mat))
    log_info("[Step10] Shared genes: ", length(shared_genes))
    sc_sce_shared <- sc_sce[shared_genes, ]
    sp_mat_shared <- sp_mat[shared_genes, ]

    set.seed(cfg$reproducibility$r_seed)
    spot_res <- tryCatch({
      SPOTlight(
        x = sc_sce_shared,
        y = sp_mat_shared,
        groups = colData(sc_sce_shared)[[celltype_col]],
        mgs = mgs_top[, c("cluster", "gene", "avg_log2FC",
                           "pct.1", "pct.2", "p_val_adj")],
        weight_id = "avg_log2FC",
        group_id = "cluster",
        gene_id = "gene",
        assay = "logcounts",
        min_cont = cfg$integration$spotlight_min_count
      )
    }, error = function(e) {
      log_error("[Step10] SPOTlight failed for ", cond, ": ",
                conditionMessage(e))
      return(NULL)
    })

    if (is.null(spot_res)) next

    # 提取细胞类型比例矩阵
    prop_mat <- spot_res$mat
    colnames(prop_mat) <- paste0("prop_", colnames(prop_mat))

    # 写入空间对象 meta.data
    common_cells <- intersect(rownames(prop_mat), colnames(sp_sub))
    for (ct in colnames(prop_mat)) {
      sp_sub@meta.data[[ct]] <- NA_real_
      sp_sub@meta.data[common_cells, ct] <- prop_mat[common_cells, ct]
    }

    spotlight_results[[cond]] <- list(
      sp_obj = sp_sub,
      prop_mat = prop_mat,
      nlm = spot_res$NMF_matrix
    )
    save_rds(spot_res, paste0("10_spotlight_result_", cond), cfg)

    # 空间特征图 (各细胞类型比例)
    prop_cols <- colnames(prop_mat)
    p_list <- lapply(prop_cols, function(ct) {
      SpatialFeaturePlot(sp_sub, features = ct, alpha = c(0.1, 1)) +
        labs(title = paste(cond, "-", ct)) +
        theme_pub(base_size = 8)
    })
    p_combined <- patchwork::wrap_plots(p_list, ncol = 3)
    save_figure(p_combined, paste0("10_spotlight_proportions_", cond), cfg,
                width = 14, height = 4 * ceiling(length(prop_cols) / 3))
  }

  if (length(spotlight_results) == 0) {
    log_error("[Step10] All SPOTlight deconvolution failed.")
    return(NULL)
  }

  # --------------------------------------------------------------------------
  # 10.4 合并所有切片 + 与铁衰老得分关联
  # --------------------------------------------------------------------------
  sp_combined <- merge(spotlight_results[[1]]$sp_obj,
                        lapply(spotlight_results[-1], function(x) x$sp_obj))
  save_rds(sp_combined, "10_spatial_with_proportions", cfg)

  # 神经元比例 vs Ferroptosis_UCell (若 step05 已计算)
  if ("prop_Neuron" %in% colnames(sp_combined@meta.data) &&
      "Ferroptosis_UCell" %in% colnames(sp_combined@meta.data)) {
    cor_df <- data.frame(
      neuron_prop = sp_combined$prop_Neuron,
      fp_score = sp_combined$Ferroptosis_UCell,
      condition = sp_combined$condition,
      region = if ("region" %in% colnames(sp_combined@meta.data))
        sp_combined$region else "NA"
    )
    save_table(cor_df, "10_neuron_prop_vs_ferroptosis", cfg)

    p_corr <- ggplot(cor_df, aes(x = neuron_prop, y = fp_score,
                                   color = condition)) +
      geom_point(alpha = 0.5, size = 0.6) +
      geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
      facet_wrap(~ condition, scales = "free") +
      scale_color_manual(values = get_condition_colors(unique(cor_df$condition))) +
      labs(title = "Neuron proportion vs Ferroptosis UCell",
           x = "Neuron proportion", y = "Ferroptosis UCell") +
      theme_pub(base_size = 9)
    save_figure(p_corr, "10_neuron_prop_vs_ferroptosis_scatter", cfg,
                width = 12, height = 8)

    ct_test <- suppressWarnings(cor.test(cor_df$neuron_prop, cor_df$fp_score,
                                          method = "spearman"))
    log_info(sprintf("[Step10] Neuron-prop ~ Ferroptosis-score: rho=%.3f, p=%.2e",
                     unname(ct_test$estimate), ct_test$p.value))
  }

  # --------------------------------------------------------------------------
  # 10.5 区域 × 细胞类型比例箱线图 (若 step06 已定义 region)
  # --------------------------------------------------------------------------
  if ("region" %in% colnames(sp_combined@meta.data)) {
    prop_cols <- grep("^prop_", colnames(sp_combined@meta.data), value = TRUE)
    region_df <- reshape2::melt(sp_combined@meta.data[, c("region", "condition", prop_cols)],
                                  id.vars = c("region", "condition"),
                                  variable.name = "cell_type",
                                  value.name = "proportion")
    region_df$cell_type <- gsub("^prop_", "", region_df$cell_type)

    p_region <- ggplot(region_df, aes(x = region, y = proportion,
                                        fill = cell_type)) +
      geom_boxplot(outlier.size = 0.2, alpha = 0.7) +
      facet_wrap(~ condition, scales = "free_x") +
      labs(title = "Cell-type proportion by region x condition",
           x = "Region", y = "Proportion", fill = "Cell type") +
      theme_pub(base_size = 9) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))
    save_figure(p_region, "10_celltype_proportion_by_region", cfg,
                width = 13, height = 7)
  }

  log_info("[Step10] SPOTlight done. Samples processed: ",
           paste(names(spotlight_results), collapse = ", "))
  invisible(list(
    sp_combined = sp_combined,
    per_sample = spotlight_results
  ))
}
