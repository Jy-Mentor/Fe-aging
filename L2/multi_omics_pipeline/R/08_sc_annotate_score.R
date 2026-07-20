# ============================================================================
# STEP 08: L3 scRNA-seq 细胞注释 + UCell 铁衰老评分
# - 优先沿用文献既存注释 (GSE233815 Zucha 2024 已含 Cell_Type)
# - 缺失注释时使用 cell type marker + UCell 自动注释
# - 计算 Ferroptosis / Senescence / Ferroaging / BCP signature 得分
# - 识别 Ferrosenescence 双阳性细胞
# - SAT1 表达验证 + 细胞类型定位
# 参考:
#   - Andreatta & Carmona 2021 UCell (PMID: 34285779, Comput Struct Biotechnol J)
#   - Hao Y et al. 2024 Nat Biotechnol (Seurat v5, PMID: 37231261)
#   - Zucha et al. 2024 PNAS (GSE233815, PMID: 39499634)
# ============================================================================

step08_sc_annotate_score <- function(seu, cfg) {
  log_info("[Step08-L3] Cell annotation + UCell scoring...")

  require_packages(c("Seurat"),
                   install_hint = "install.packages('Seurat')")
  suppressPackageStartupMessages({
    library(Seurat)
    library(ggplot2)
  })

  if (is.null(seu)) {
    stop("step08: sc_seu is NULL. Run step07 first.")
  }

  # --------------------------------------------------------------------------
  # 8.1 加载基因集 (鼠源)
  # --------------------------------------------------------------------------
  gene_sets <- build_gene_sets(cfg, organism = "mouse")

  # 8.2 检测/赋值细胞类型注释
  celltype_col <- cfg$data$sc_celltype_col
  if (!(celltype_col %in% colnames(seu@meta.data))) {
    # 兼容候选列名
    candidates <- c("Cell_Type", "cell_type", "celltype", "CellType",
                    "cluster_label", "Major.celltype", "Subtype")
    found <- candidates[candidates %in% colnames(seu@meta.data)]
    if (length(found) > 0) {
      celltype_col <- found[1]
      log_info("[Step08] Using existing celltype column: ", celltype_col)
    } else {
      log_warn("[Step08] No celltype column found; running marker-based annotation.")
      celltype_col <- ".auto_celltype"
      seu <- .annotate_by_markers(seu, gene_sets$celltype_markers)
      seu@meta.data[[celltype_col]] <- seu$.auto_celltype
    }
  } else {
    log_info("[Step08] Using celltype column from config: ", celltype_col,
             " (n=", length(unique(seu@meta.data[[celltype_col]])), ")")
  }

  cfg$data$sc_celltype_col <- celltype_col  # 回写以供后续步骤使用
  seu$cell_type <- seu@meta.data[[celltype_col]]

  # --------------------------------------------------------------------------
  # 8.3 UCell 评分 (优先 UCell, 回退 AddModuleScore)
  # --------------------------------------------------------------------------
  signature_sets <- list(
    Ferroptosis      = gene_sets$ferroptosis,
    Senescence       = gene_sets$senescence,
    Ferroaging       = gene_sets$ferroaging,
    Ferrosenescence  = gene_sets$ferrosenescence,
    BCP_Up           = gene_sets$bcp_up,
    BCP_Down         = gene_sets$bcp_down
  )

  # 验证基因集与表达矩阵重叠
  expr_genes <- rownames(seu)
  for (nm in names(signature_sets)) {
    validate_gene_set_overlap(signature_sets[[nm]], expr_genes,
                              set_name = paste0("signature_", nm))
  }

  seu <- .score_signatures(seu, signature_sets, cfg)

  # --------------------------------------------------------------------------
  # 8.4 可视化: UMAP × 评分 (split by condition)
  # --------------------------------------------------------------------------
  condition_col <- cfg$data$sc_condition_col
  if (!(condition_col %in% colnames(seu@meta.data))) {
    condition_col <- "orig.ident"
  }

  score_features <- paste0(c("Ferroptosis", "Senescence",
                              "Ferrosenescence", "Ferroaging",
                              "BCP_Up", "BCP_Down"), "_UCell")

  for (feat in score_features) {
    if (!(feat %in% colnames(seu@meta.data))) next
    p <- FeaturePlot(seu, features = feat, split.by = condition_col,
                     ncol = length(unique(seu@meta.data[[condition_col]])),
                     cols = DIVERGE_PALETTE, raster = FALSE) +
      labs(title = paste("UMAP", feat, "by", condition_col)) +
      theme_pub(base_size = 9) &
      theme(plot.title = element_text(hjust = 0.5))
    save_figure(p, paste0("08_umap_", feat), cfg, width = 14, height = 4)
  }

  # UMAP by cell type
  p_ct <- DimPlot(seu, reduction = "umap", group.by = celltype_col,
                  cols = get_celltype_colors(unique(seu@meta.data[[celltype_col]]))) +
    labs(title = "UMAP by cell type") + theme_pub(base_size = 10)
  save_figure(p_ct, "08_umap_celltype", cfg, width = 10, height = 8)

  # --------------------------------------------------------------------------
  # 8.5 各细胞类型铁衰老得分箱线图
  # --------------------------------------------------------------------------
  meta_cols <- c(celltype_col, condition_col, score_features)
  meta_cols <- meta_cols[meta_cols %in% colnames(seu@meta.data)]
  score_df <- seu@meta.data[, meta_cols, drop = FALSE]
  rownames(score_df) <- NULL

  save_table(score_df, "08_cell_scores_per_cell", cfg)

  score_long <- reshape2::melt(score_df,
                                id.vars = c(celltype_col, condition_col),
                                variable.name = "Signature",
                                value.name = "UCell_Score")

  p_violin <- ggplot(score_long, aes(x = .data[[celltype_col]],
                                      y = UCell_Score,
                                      fill = .data[[celltype_col]])) +
    geom_violin(scale = "width", trim = TRUE, alpha = 0.7) +
    geom_boxplot(width = 0.15, outlier.size = 0.2, alpha = 0.6) +
    facet_wrap(vars(Signature), scales = "free_y", ncol = 3) +
    scale_fill_manual(values = get_celltype_colors(unique(score_df[[celltype_col]]))) +
    labs(title = "UCell scores by cell type",
         x = "Cell type", y = "UCell score") +
    theme_pub(base_size = 9) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "none")
  save_figure(p_violin, "08_scores_by_celltype_violin", cfg,
              width = 13, height = 9)

  # 按条件分面 (ggplot2 3.5+: facet 变量必须用 vars() 显式传递, 避免 NSE 检查失败)
  score_long[[celltype_col]] <- as.factor(score_long[[celltype_col]])
  p_violin_cond <- ggplot(score_long,
                           aes(x = .data[[condition_col]],
                               y = UCell_Score,
                               fill = .data[[condition_col]])) +
    geom_violin(scale = "width", trim = TRUE, alpha = 0.7) +
    facet_grid(rows = vars(Signature), cols = vars(!!sym(celltype_col)),
               scales = "free_y") +
    scale_fill_manual(values = get_condition_colors(unique(score_df[[condition_col]]))) +
    labs(title = "UCell scores by cell type x condition",
         x = "Condition", y = "UCell score") +
    theme_pub(base_size = 8) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "bottom")
  save_figure(p_violin_cond, "08_scores_by_celltype_condition", cfg,
              width = 14, height = 12)

  # --------------------------------------------------------------------------
  # 8.6 Ferrosenescence 双阳性细胞鉴定
  # --------------------------------------------------------------------------
  fp_col <- "Ferroptosis_UCell"
  sn_col <- "Senescence_UCell"
  if (fp_col %in% colnames(seu@meta.data) && sn_col %in% colnames(seu@meta.data)) {
    q_prob <- cfg$sc$ferrosenescence_quantile
    fp_thr <- quantile(seu@meta.data[[fp_col]], q_prob, na.rm = TRUE)
    sn_thr <- quantile(seu@meta.data[[sn_col]], q_prob, na.rm = TRUE)
    seu$ferrosenescence_status <- ifelse(
      seu@meta.data[[fp_col]] > fp_thr & seu@meta.data[[sn_col]] > sn_thr,
      "Ferrosenescence_High", "Low"
    )
    log_info(sprintf("[Step08] Ferrosenescence thresholds: FP>%.4f, SN>%.4f (%.0f%% quantile)",
                     fp_thr, sn_thr, 100 * q_prob))
    log_info("[Step08] Ferrosenescence_High cells: ",
             sum(seu$ferrosenescence_status == "Ferrosenescence_High", na.rm = TRUE),
             " / ", ncol(seu))

    # 各细胞类型 × 条件下双阳性比例
    fs_table <- as.data.frame(table(seu@meta.data[[celltype_col]],
                                     seu@meta.data[[condition_col]],
                                     seu$ferrosenescence_status))
    colnames(fs_table) <- c("celltype", "condition", "status", "n")
    fs_table$proportion <- ave(fs_table$n, fs_table$celltype,
                                fs_table$condition,
                                FUN = function(x) x / sum(x))
    save_table(fs_table, "08_ferrosenescence_proportions", cfg)

    p_fs_prop <- ggplot(subset(fs_table, status == "Ferrosenescence_High"),
                         aes(x = celltype, y = proportion,
                             fill = condition)) +
      geom_col(position = position_dodge(width = 0.8)) +
      scale_fill_manual(values = get_condition_colors(unique(fs_table$condition))) +
      labs(title = "Ferrosenescence_High proportion per cell type",
           x = "Cell type", y = "Proportion",
           fill = "Condition") +
      theme_pub(base_size = 10) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    save_figure(p_fs_prop, "08_ferrosenescence_proportion_barplot", cfg,
                width = 12, height = 6)

    # 双阳性散点图 (FP vs SN)
    p_scatter <- ggplot(seu@meta.data, aes(x = .data[[fp_col]],
                                            y = .data[[sn_col]],
                                            color = ferrosenescence_status)) +
      geom_point(alpha = 0.4, size = 0.6) +
      geom_hline(yintercept = sn_thr, linetype = "dashed", color = "grey50") +
      geom_vline(xintercept = fp_thr, linetype = "dashed", color = "grey50") +
      scale_color_manual(values = c("Ferrosenescence_High" = "#B2182B",
                                     "Low" = "grey70")) +
      facet_wrap(vars(!!sym(condition_col))) +
      labs(title = "Ferroptosis vs Senescence (UCell)",
           x = "Ferroptosis UCell", y = "Senescence UCell",
           color = "Status") +
      theme_pub(base_size = 9)
    save_figure(p_scatter, "08_ferrosenescence_scatter", cfg,
                width = 12, height = 9)
  }

  # --------------------------------------------------------------------------
  # 8.7 SAT1 表达验证 (BCP 靶基因)
  # --------------------------------------------------------------------------
  if ("Sat1" %in% rownames(seu)) {
    p_sat1_ct <- VlnPlot(seu, features = "Sat1", group.by = celltype_col,
                          split.by = condition_col, pt.size = 0,
                          cols = get_celltype_colors(unique(seu@meta.data[[celltype_col]]))) +
      labs(title = "Sat1 expression by cell type x condition") +
      theme_pub(base_size = 9)
    save_figure(p_sat1_ct, "08_sat1_violin_celltype_condition", cfg,
                width = 12, height = 5)

    # SAT1 与 Ferroptosis 得分的相关性 (per cell type)
    if (fp_col %in% colnames(seu@meta.data)) {
      sat1_data <- FetchData(seu, vars = c("Sat1", fp_col, celltype_col, condition_col))
      colnames(sat1_data)[2] <- "FP_Score"
      save_table(sat1_data, "08_sat1_vs_ferroptosis_score", cfg)

      p_sat1_corr <- ggplot(sat1_data, aes(x = Sat1, y = FP_Score,
                                            color = .data[[celltype_col]])) +
        geom_point(alpha = 0.4, size = 0.5) +
        geom_smooth(method = "lm", se = FALSE, linewidth = 0.6) +
        facet_wrap(vars(!!sym(celltype_col)), scales = "free") +
        scale_color_manual(values = get_celltype_colors(unique(sat1_data[[celltype_col]]))) +
        labs(title = "Sat1 vs Ferroptosis score",
             x = "Sat1 expression", y = "Ferroptosis UCell") +
        theme_pub(base_size = 8) +
        theme(legend.position = "none")
      save_figure(p_sat1_corr, "08_sat1_vs_ferroptosis_scatter", cfg,
                  width = 13, height = 9)

      # Spearman 相关性汇总
      cor_summary <- do.call(rbind, lapply(
        unique(sat1_data[[celltype_col]]), function(ct) {
          sub <- sat1_data[sat1_data[[celltype_col]] == ct, ]
          if (nrow(sub) < 30) return(NULL)
          ct_test <- suppressWarnings(cor.test(sub$Sat1, sub$FP_Score,
                                                method = "spearman"))
          data.frame(cell_type = ct,
                     n_cells = nrow(sub),
                     rho = unname(ct_test$estimate),
                     p_value = ct_test$p.value)
        }
      ))
      if (!is.null(cor_summary) && nrow(cor_summary) > 0) {
        cor_summary$padj <- p.adjust(cor_summary$p_value, method = "BH")
        save_table(cor_summary, "08_sat1_ferroptosis_correlation_by_celltype", cfg)
      }
    }
  } else {
    log_warn("[Step08] 'Sat1' not in rownames(seu). SAT1 validation skipped.")
  }

  save_rds(seu, "08_sc_seurat_annotated_scored", cfg)
  log_info("[Step08] Cell annotation + UCell scoring done.")
  invisible(seu)
}

# ----------------------------------------------------------------------------
# 内部辅助: 基于 marker 的简单自动注释 (UCell 评分取最高)
# ----------------------------------------------------------------------------
.annotate_by_markers <- function(seu, celltype_markers) {
  log_info("[Step08] Marker-based annotation (no existing labels)...")
  if (!requireNamespace("UCell", quietly = TRUE)) {
    stop("UCell package required for marker-based annotation.")
  }
  seu <- UCell::AddModuleScore_UCell(seu, features = celltype_markers,
                                      name = "_marker_score")
  score_cols <- paste0(names(celltype_markers), "_marker_score")
  score_cols <- score_cols[score_cols %in% colnames(seu@meta.data)]
  mat <- seu@meta.data[, score_cols, drop = FALSE]
  seu$.auto_celltype <- colnames(mat)[max.col(mat, ties.method = "first")]
  seu$.auto_celltype <- gsub("_marker_score$", "", seu$.auto_celltype)
  log_info("[Step08] Auto-annotation complete. Cell types: ",
           paste(unique(seu$.auto_celltype), collapse = ", "))
  invisible(seu)
}

# ----------------------------------------------------------------------------
# 内部辅助: 评分 (优先 UCell, 回退 AddModuleScore)
# ----------------------------------------------------------------------------
.score_signatures <- function(seu, signature_sets, cfg) {
  if (requireNamespace("UCell", quietly = TRUE)) {
    log_info("[Step08] Using UCell for signature scoring (maxRank=",
             cfg$sc$ucell_max_rank, ")")
    # UCell::AddModuleScore_UCell 的 name 参数是单个字符串后缀/前缀,
    # 会被拼接到每个 feature set 名字后面/前面。
    # name = "_UCell" -> 列名: Ferroptosis_UCell, Senescence_UCell, ...
    # 错误用法: name = paste0(names(signature_sets), "_UCell") (向量)
    #   会产生 FerroptosisFerroptosis_UCell 等错误列名
    seu <- UCell::AddModuleScore_UCell(
      seu, features = signature_sets,
      name = "_UCell",
      maxRank = cfg$sc$ucell_max_rank,
      w_neg = 1
    )
    # 验证 UCell 评分列已添加
    expected_cols <- paste0(names(signature_sets), "_UCell")
    missing_cols <- setdiff(expected_cols, colnames(seu@meta.data))
    if (length(missing_cols) > 0) {
      log_warn("[Step08] UCell missing score columns: ",
               paste(missing_cols, collapse = ", "),
               "; actual cols: ",
               paste(grep("_UCell$", colnames(seu@meta.data), value = TRUE),
                     collapse = ", "))
    } else {
      log_info("[Step08] UCell score columns added: ",
               paste(expected_cols, collapse = ", "))
    }
  } else {
    log_warn("[Step08] UCell not installed; falling back to AddModuleScore.")
    for (nm in names(signature_sets)) {
      seu <- AddModuleScore(seu, features = list(signature_sets[[nm]]),
                            name = paste0(nm, "_AS"), verbose = FALSE)
      col_idx <- paste0(nm, "_AS1")
      if (col_idx %in% colnames(seu@meta.data)) {
        colnames(seu@meta.data)[colnames(seu@meta.data) == col_idx] <-
          paste0(nm, "_UCell")
      }
    }
  }
  invisible(seu)
}
