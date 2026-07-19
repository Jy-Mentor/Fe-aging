# ============================================================================
# STEP 05: L2 Spatial 铁衰老评分与组织空间定位
# - 在每个切片上计算铁死亡/衰老/铁衰老基因集得分 (UCell 优先)
# - SpatialFeaturePlot 可视化得分在组织切片上的分布
# - 关键铁衰老基因 (Gpx4/Sat1/Cdkn1a/Il6 等) 空间表达模式
# 参考:
#   - Andreatta M & Carmona SJ 2021 bioRxiv (UCell)
#   - Han B et al. 2024 Sci Transl Med (空间+sc 整合标杆)
# ============================================================================

step05_spatial_module_score <- function(spatial_list, cfg) {
  log_info("[Step05-L2] Spatial module scoring...")

  if (is.null(spatial_list)) stop("Spatial list is NULL. Run step 04 first.")
  require_packages(c("Seurat"))
  suppressPackageStartupMessages(library(Seurat))

  gene_sets <- build_gene_sets(cfg, organism = "mouse")

  # 优先使用 UCell (对 spot 组成不敏感), 否则用 AddModuleScore
  use_ucell <- requireNamespace("UCell", quietly = TRUE)
  if (use_ucell) {
    log_info("[Step05] Using UCell for gene set scoring (robust across samples)")
    suppressPackageStartupMessages(library(UCell))
  } else {
    log_warn("[Step05] UCell not installed; falling back to AddModuleScore ",
             "(less robust for cross-sample comparison)")
  }

  # 基因集列表
  signatures <- list(
    Ferroptosis     = gene_sets$ferroptosis,
    Senescence      = gene_sets$senescence,
    Ferroaging      = gene_sets$ferroaging,
    Ferrosenescence = gene_sets$ferrosenescence,
    BCP_Up          = gene_sets$bcp_up,
    BCP_Down        = gene_sets$bcp_down
  )

  # --------------------------------------------------------------------------
  # 5.1 对每个切片计算得分
  # --------------------------------------------------------------------------
  spatial_merged <- NULL
  sample_names <- names(spatial_list)

  for (sn in sample_names) {
    seu <- spatial_list[[sn]]
    log_info("[Step05] Scoring sample: ", sn, " (", ncol(seu), " spots)")

    # 验证基因集与表达矩阵的重叠
    expr_genes <- rownames(GetAssayData(seu, assay = "SCT", layer = "data"))
    for (sig_name in names(signatures)) {
      validate_gene_set_overlap(signatures[[sig_name]], expr_genes,
                                set_name = paste0(sn, "_", sig_name))
    }

    if (use_ucell) {
      seu <- AddModuleScore_UCell(
        seu,
        features = signatures,
        name = NULL,
        w_neg = 1.0,
        maxRank = cfg$sc$ucell_max_rank,
        chunk.size = 100,
        ncores = 1
      )
      # UCell 添加的列名即基因集名本身
    } else {
      # AddModuleScore 添加后缀 _1, _2, ...
      seu <- AddModuleScore(seu, features = signatures, name = "ModuleScore_")
      # 重命名为标准名
      new_names <- paste0("ModuleScore_", seq_along(signatures))
      old_names <- paste0(names(signatures), "_1")
      for (i in seq_along(signatures)) {
        if (old_names[i] %in% colnames(seu@meta.data)) {
          colnames(seu@meta.data)[colnames(seu@meta.data) == old_names[i]] <-
            names(signatures)[i]
        }
      }
    }

    spatial_list[[sn]] <- seu
  }

  # --------------------------------------------------------------------------
  # 5.2 合并切片 (用于跨样本比较)
  # --------------------------------------------------------------------------
  if (length(spatial_list) > 1) {
    spatial_merged <- merge(spatial_list[[1]], y = spatial_list[-1],
                             add.cell.ids = sample_names)
  } else {
    spatial_merged <- spatial_list[[1]]
  }
  save_rds(spatial_merged, "05_spatial_merged_scored", cfg)

  # --------------------------------------------------------------------------
  # 5.3 SpatialFeaturePlot: 铁死亡/衰老得分空间分布
  # --------------------------------------------------------------------------
  score_cols <- names(signatures)

  for (sn in sample_names) {
    seu <- spatial_list[[sn]]
    for (score_col in score_cols) {
      if (!(score_col %in% colnames(seu@meta.data))) {
        log_warn("[Step05] ", sn, " missing score column: ", score_col)
        next
      }
      tryCatch({
        p <- SpatialFeaturePlot(seu, features = score_col,
                                 crop = TRUE, alpha = c(0.1, 1)) +
          scale_fill_gradientn(colors = DIVERGE_PALETTE) +
          labs(title = paste0(sn, " - ", score_col, " score")) +
          theme(legend.position = "right")
        save_figure(p, sprintf("05_spatial_%s_%s_score", sn, score_col), cfg,
                    width = 8, height = 7)
      }, error = function(e) {
        log_warn("[Step05] SpatialFeaturePlot failed for ", sn, " - ",
                 score_col, ": ", conditionMessage(e))
      })
    }
  }

  # --------------------------------------------------------------------------
  # 5.4 关键铁衰老基因的空间表达 (Gpx4, Sat1, Acsl4, Cdkn1a, Il6 等)
  # --------------------------------------------------------------------------
  key_genes <- c("Gpx4", "Sat1", "Acsl4", "Slc7a11", "Tfrc",
                 "Cdkn1a", "Cdkn2a", "Il6", "Tnf", "Hmox1", "Nfe2l2")

  for (sn in sample_names) {
    seu <- spatial_list[[sn]]
    expr_genes <- rownames(GetAssayData(seu, assay = "SCT", layer = "data"))
    genes_avail <- intersect(key_genes, expr_genes)

    for (gene in genes_avail) {
      tryCatch({
        p <- SpatialFeaturePlot(seu, features = gene,
                                 crop = TRUE, alpha = c(0.1, 1)) +
          labs(title = paste0(sn, " - ", gene, " expression")) +
          theme(legend.position = "right")
        save_figure(p, sprintf("05_spatial_%s_gene_%s", sn, gene), cfg,
                    width = 8, height = 7)
      }, error = function(e) {
        log_warn("[Step05] SpatialFeaturePlot gene ", gene, " failed for ",
                 sn, ": ", conditionMessage(e))
      })
    }
  }

  # --------------------------------------------------------------------------
  # 5.5 跨样本得分统计 (条件 × 得分)
  # --------------------------------------------------------------------------
  score_df <- do.call(rbind, lapply(sample_names, function(sn) {
    seu <- spatial_list[[sn]]
    meta <- seu@meta.data
    df <- data.frame(
      spot_id = colnames(seu),
      sample = sn,
      stringsAsFactors = FALSE
    )
    for (col in score_cols) {
      if (col %in% colnames(meta)) {
        df[[col]] <- meta[[col]]
      }
    }
    df
  }))

  save_table(score_df, "05_spatial_scores_per_spot", cfg)

  # 跨样本得分箱线图
  for (col in score_cols) {
    if (!(col %in% colnames(score_df))) next
    p <- ggplot(score_df, aes(x = sample, y = .data[[col]], fill = sample)) +
      geom_violin(trim = FALSE, alpha = 0.6) +
      geom_boxplot(width = 0.15, outlier.size = 0.3) +
      scale_fill_manual(values = get_condition_colors(sample_names)) +
      labs(title = paste0("Spatial ", col, " score by sample"),
           x = "Sample", y = "UCell score") +
      theme_pub(base_size = 10) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
      guides(fill = "none")
    save_figure(p, sprintf("05_spatial_score_violin_%s", col), cfg,
                width = 8, height = 6)
  }

  log_info("[Step05] Spatial module scoring done.")
  invisible(spatial_merged)
}
