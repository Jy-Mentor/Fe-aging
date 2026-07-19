# ============================================================================
# STEP 09: L3 scRNA-seq 拟时序分析 (monocle3) + Augur 细胞类型优先级
# - 提取神经元亚群 ( Ferrosenescence_High / SAT1 高表达等) 进行拟时序
# - 以 Control 神经元作为根节点, 沿拟时序追踪铁死亡→衰老基因动态
# - Augur 评估各细胞类型在缺血条件下的扰动强度 (AUC 越高 = 扰动越大)
# 参考:
#   - Qiu X et al. 2017 Nat Methods (monocle3, PMID: 28825705)
#   - Skelly DA et al. 2018 Cell (Augur, PMID: 30196209)
#   - Tritschler S et al. 2019 Nat Methods (monocle3 实践, PMID: 30778352)
# ============================================================================

step09_sc_pseudotime_augur <- function(seu, cfg) {
  log_info("[Step09-L3] monocle3 pseudotime + Augur prioritization...")

  if (is.null(seu)) {
    stop("step09: sc_seu is NULL. Run step07/08 first.")
  }

  require_packages(c("monocle3"),
                   install_hint = "BiocManager::install('monocle3')")
  suppressPackageStartupMessages({
    library(monocle3)
    library(Seurat)
    library(ggplot2)
  })

  celltype_col <- cfg$data$sc_celltype_col
  condition_col <- cfg$data$sc_condition_col
  if (!(celltype_col %in% colnames(seu@meta.data))) {
    stop("step09: celltype column '", celltype_col, "' not in seu meta.data")
  }
  if (!(condition_col %in% colnames(seu@meta.data))) {
    stop("step09: condition column '", condition_col, "' not in seu meta.data")
  }

  # --------------------------------------------------------------------------
  # 9.1 神经元亚群拟时序分析
  # --------------------------------------------------------------------------
  neuron_types <- c("Neuron", "NeuronsGABA", "NeuronsGLUT", "Neurons-GABA",
                    "Neurons-GLUT", "ExcitatoryNeurons", "InhibitoryNeurons")
  available_cts <- unique(seu@meta.data[[celltype_col]])
  neuron_in_data <- available_cts[available_cts %in% neuron_types]

  cds_neuron <- NULL
  if (length(neuron_in_data) > 0) {
    log_info("[Step09] Subsetting neurons for pseudotime: ",
             paste(neuron_in_data, collapse = ", "))
    neuron_sub <- subset(seu, subset = .data[[celltype_col]] %in% neuron_in_data)
    if (ncol(neuron_sub) < 50) {
      log_warn("[Step09] Too few neurons (n=", ncol(neuron_sub),
               "); skipping monocle3.")
    } else {
      cds_neuron <- .run_monocle3(neuron_sub, cfg,
                                    celltype_col = celltype_col,
                                    condition_col = condition_col)
    }
  } else {
    log_warn("[Step09] No neuron cell types found. Skipping neuron pseudotime.")
  }

  # 9.2 Ferrosenescence_High 亚群拟时序 (若 step08 已标记)
  cds_fs <- NULL
  if ("ferrosenescence_status" %in% colnames(seu@meta.data)) {
    fs_sub <- subset(seu, subset = ferrosenescence_status == "Ferrosenescence_High")
    if (ncol(fs_sub) >= 50) {
      log_info("[Step09] Pseudotime on Ferrosenescence_High cells (n=",
               ncol(fs_sub), ")")
      cds_fs <- .run_monocle3(fs_sub, cfg,
                                celltype_col = celltype_col,
                                condition_col = condition_col,
                                tag = "ferrosenescence_high")
    } else {
      log_warn("[Step09] Too few Ferrosenescence_High cells (n=", ncol(fs_sub),
               "); skipping.")
    }
  }

  # --------------------------------------------------------------------------
  # 9.3 Augur 细胞类型优先级
  # --------------------------------------------------------------------------
  augur_res <- .run_augur(seu, cfg, celltype_col = celltype_col,
                            condition_col = condition_col)

  # 保存结果
  if (!is.null(cds_neuron)) save_rds(cds_neuron, "09_cds_neuron", cfg)
  if (!is.null(cds_fs))     save_rds(cds_fs, "09_cds_ferrosenescence", cfg)
  if (!is.null(augur_res))  save_rds(augur_res, "09_augur_result", cfg)

  log_info("[Step09] Pseudotime + Augur done.")
  invisible(list(cds_neuron = cds_neuron,
                  cds_fs = cds_fs,
                  augur = augur_res))
}

# ----------------------------------------------------------------------------
# monocle3 拟时序子流程
# ----------------------------------------------------------------------------
.run_monocle3 <- function(seu_sub, cfg, celltype_col, condition_col, tag = "neuron") {
  log_info("[Step09] Converting Seurat -> cell_data_set (tag=", tag, ")...")

  # Seurat v5: layers 可能 split, 需要先 JoinLayers
  if (inherits(seu_sub[["RNA"]], "Assay5")) {
    seu_sub[["RNA"]] <- JoinLayers(seu_sub[["RNA"]])
  }

  cds <- as.cell_data_set(seu_sub)
  cds@colData@metadata$n_cells <- NULL  # 修正 Seurat→CDS 兼容性

  # 聚类 + 主图
  cds <- cluster_cells(cds, reduction_method = "UMAP")
  cds <- learn_graph(cds, use_partition = TRUE)

  # 选根细胞 (Control / Ctrl 条件)
  ctrl_cells <- colnames(cds)[colData(cds)[[condition_col]] %in% c("Control", "Ctrl")]
  if (length(ctrl_cells) == 0) {
    log_warn("[Step09] No Control cells for root. Using graph principal node.")
    root_node <- NULL
  } else {
    # 选择 control 细胞最多的聚类节点
    root_node <- monocle3:::get_principal_node(cds, ctrl_cells)
    log_info("[Step09] Root node selected from ", length(ctrl_cells), " control cells")
  }

  if (!is.null(root_node)) {
    cds <- order_cells(cds, root_pr_nodes = root_node)
  } else {
    cds <- order_cells(cds)
  }

  # 拟时序可视化
  p_pseudo <- plot_cells(cds, color_cells_by = "pseudotime",
                          label_branch_points = TRUE,
                          label_leaves = TRUE,
                          cell_size = 0.6) +
    ggtitle(paste("Pseudotime (", tag, ")", sep = "")) +
    theme_pub(base_size = 10)
  save_figure(p_pseudo, paste0("09_pseudotime_", tag), cfg, width = 9, height = 7)

  # 按 condition 着色
  p_cond <- plot_cells(cds, color_cells_by = condition_col,
                        cell_size = 0.6) +
    ggtitle(paste("Cells by", condition_col, "(", tag, ")")) +
    theme_pub(base_size = 10)
  save_figure(p_cond, paste0("09_pseudotime_", tag, "_by_condition"), cfg,
              width = 9, height = 7)

  # 沿拟时序展示铁死亡/衰老关键基因动态
  key_genes <- c("Gpx4", "Acsl4", "Sat1", "Slc7a11",
                  "Cdkn1a", "Tp53", "Il6", "Lmnb1")
  key_genes <- key_genes[key_genes %in% rownames(cds)]
  if (length(key_genes) > 0) {
    p_genes <- plot_genes_in_pseudotime(cds[key_genes, ],
                                         color_cells_by = condition_col,
                                         min_expr = 0.1) +
      ggtitle(paste("Key genes along pseudotime (", tag, ")", sep = "")) +
      theme_pub(base_size = 9)
    save_figure(p_genes, paste0("09_pseudotime_", tag, "_key_genes"), cfg,
                width = 12, height = 8)
  }

  # 拟时序 vs UCell 得分相关性
  pseudo_scores <- c("Ferroptosis_UCell", "Senescence_UCell",
                      "Ferrosenescence_UCell", "Ferroaging_UCell")
  pseudo_scores <- pseudo_scores[pseudo_scores %in% colnames(colData(cds))]
  if (length(pseudo_scores) > 0) {
    pseudo_df <- data.frame(
      pseudotime = pseudotime(cds),
      colData(cds)[, c(condition_col, celltype_col, pseudo_scores), drop = FALSE]
    )
    save_table(pseudo_df, paste0("09_pseudotime_", tag, "_scores"), cfg)

    pseudo_long <- reshape2::melt(pseudo_df,
                                   id.vars = c("pseudotime", condition_col, celltype_col),
                                   variable.name = "Signature",
                                   value.name = "UCell_Score")

    p_pseudo_score <- ggplot(pseudo_long,
                              aes(x = pseudotime, y = UCell_Score,
                                  color = .data[[condition_col]])) +
      geom_point(alpha = 0.4, size = 0.6) +
      geom_smooth(method = "loess", se = TRUE, alpha = 0.2) +
      facet_wrap(~ Signature, scales = "free_y", ncol = 2) +
      scale_color_manual(values = get_condition_colors(unique(pseudo_long[[condition_col]]))) +
      labs(title = paste("UCell scores along pseudotime (", tag, ")", sep = ""),
           x = "Pseudotime", y = "UCell score",
           color = "Condition") +
      theme_pub(base_size = 9)
    save_figure(p_pseudo_score, paste0("09_pseudotime_", tag, "_score_dynamics"),
                cfg, width = 11, height = 9)
  }

  invisible(cds)
}

# ----------------------------------------------------------------------------
# Augur 子流程: 评估各细胞类型在缺血条件下的扰动强度
# ----------------------------------------------------------------------------
.run_augur <- function(seu, cfg, celltype_col, condition_col) {
  log_info("[Step09] Running Augur cell-type prioritization...")

  if (!requireNamespace("Augur", quietly = TRUE)) {
    log_warn("[Step09] Augur package not installed. Skipping.")
    return(NULL)
  }

  # Augur 要求 condition 为二分类
  conds <- unique(seu@meta.data[[condition_col]])
  if (length(conds) < 2) {
    log_warn("[Step09] Need >=2 conditions for Augur. Skipping.")
    return(NULL)
  }
  control_labels <- c("Ctrl", "Control")
  if (!any(conds %in% control_labels)) {
    log_warn("[Step09] No control condition found (Ctrl/Control). Using first as ref.")
    control_labels <- conds[1]
  }

  seu$augur_condition <- ifelse(seu@meta.data[[condition_col]] %in% control_labels,
                                 "ctrl", "stim")

  if (inherits(seu[["RNA"]], "Assay5")) {
    seu[["RNA"]] <- JoinLayers(seu[["RNA"]])
  }

  # 抽样加速 (Augur 在大细胞数下耗时极长, 抽样每细胞类型 ≤500 细胞)
  set.seed(cfg$reproducibility$r_seed)
  meta_df <- seu@meta.data
  meta_df$cell_id <- rownames(meta_df)
  sampled_cells <- unlist(lapply(unique(meta_df[[celltype_col]]), function(ct) {
    cells_ct <- meta_df$cell_id[meta_df[[celltype_col]] == ct]
    if (length(cells_ct) <= 500) return(cells_ct)
    sample(cells_ct, 500)
  }))
  seu_sub <- subset(seu, cells = sampled_cells)
  log_info("[Step09] Augur subsample: ", ncol(seu_sub), " cells (≤500/type)")

  expr_mat <- as.matrix(GetAssayData(seu_sub, assay = "RNA", layer = "data"))
  cell_types <- seu_sub@meta.data[[celltype_col]]
  condition <- seu_sub$augur_condition

  augur_res <- tryCatch({
    Augur::calculate_auc(
      input = expr_mat,
      cell_meta = data.frame(cell_type = cell_types,
                              label = condition,
                              row.names = colnames(expr_mat)),
      type = "binary",
      n_threads = cfg$sc$augur_n_threads,
      n_subsamples = cfg$sc$augur_subsample_size,
      folds = cfg$sc$augur_folds,
      features_percent = cfg$sc$augur_features_percent
    )
  }, error = function(e) {
    log_error("[Step09] Augur failed: ", conditionMessage(e))
    return(NULL)
  })

  if (is.null(augur_res)) return(NULL)

  # AUC 排名
  if (!is.null(augur_res$AUC)) {
    auc_df <- augur_res$AUC
    if (!is.data.frame(auc_df)) {
      auc_df <- data.frame(cell_type = names(augur_res$AUC),
                            AUC = unname(augur_res$AUC))
    }
    auc_df <- auc_df[order(-auc_df$AUC), ]
    save_table(auc_df, "09_augur_auc_ranking", cfg)

    p_auc <- ggplot(auc_df, aes(x = reorder(cell_type, AUC), y = AUC,
                                  fill = AUC)) +
      geom_col() +
      geom_hline(yintercept = 0.5, linetype = "dashed", color = "grey50") +
      scale_fill_gradient(low = "#2166AC", high = "#B2182B") +
      coord_flip() +
      labs(title = "Augur: cell-type perturbation priority",
           x = "Cell type", y = "AUC (Control vs Ischemia)") +
      theme_pub(base_size = 10) +
      theme(legend.position = "right")
    save_figure(p_auc, "09_augur_auc_barplot", cfg, width = 9, height = 7)
  }

  invisible(augur_res)
}
