# ============================================================================
# STEP 4: 差异表达分析
# - 每个细胞类型 Ctrl vs 1DPI/3DPI/7DPI
# - 使用 Seurat FindMarkers (wilcox), 阈值 adj.p<0.05, |log2FC|>0.58
# - 输出所有 DEG 表与 Top 基因热图
# ============================================================================

step04_de_analysis <- function(seu, cfg) {
  log_info("[Step4] Differential expression analysis...")

  Seurat::Idents(seu) <- "CellType"
  cell_types <- unique(seu$CellType)
  conditions <- cfg$analysis$condition_levels
  dpi_conditions <- setdiff(conditions, "Ctrl")

  all_degs <- list()
  default_assay <- "RNA"

  for (ct in cell_types) {
    ct_cells <- Cells(seu)[seu$CellType == ct]
    n_ct <- length(ct_cells)
    log_info("[Step4] Cell type: ", ct, " (n=", n_ct, ")")
    if (n_ct < 30) {
      log_warn("[Step4]   Skipping ", ct, ": too few cells (", n_ct, ")")
      next
    }

    seu_sub <- subset(seu, cells = ct_cells)
    seu_sub$Condition <- factor(seu_sub$Condition, levels = conditions)
    n_ctrl <- sum(seu_sub$Condition == "Ctrl")

    if (n_ctrl < 5) {
      log_warn("[Step4]   Skipping ", ct, ": too few Ctrl cells (", n_ctrl, ")")
      next
    }

    Seurat::Idents(seu_sub) <- "Condition"

    for (cond in dpi_conditions) {
      n_cond <- sum(seu_sub$Condition == cond)
      if (n_cond < 5) {
        log_warn("[Step4]   Skip ", ct, " ", cond, " vs Ctrl (n_cond=", n_cond, ")")
        next
      }
      log_info("[Step4]   ", ct, ": ", cond, " (n=", n_cond,
               ") vs Ctrl (n=", n_ctrl, ")")

      de_res <- tryCatch({
        Seurat::FindMarkers(
          seu_sub, ident.1 = cond, ident.2 = "Ctrl",
          test.use = cfg$de$test_use,
          logfc.threshold = cfg$de$logfc_threshold * 0.5,
          min.pct = cfg$de$min_pct,
          assay = default_assay,
          verbose = FALSE
        )
      }, error = function(e) {
        log_error("[Step4]   FindMarkers failed for ", ct, " ", cond,
                  ": ", conditionMessage(e))
        stop("FindMarkers failed for ", ct, " ", cond,
             ": ", conditionMessage(e))
      })

      if (is.null(de_res) || nrow(de_res) == 0) next

      de_res$gene <- rownames(de_res)
      de_res$cell_type <- ct
      de_res$comparison <- paste0(cond, "_vs_Ctrl")
      de_res$signif <- ifelse(de_res$p_val_adj < cfg$de$adj_pval_threshold &
                                abs(de_res$avg_log2FC) >= cfg$de$logfc_threshold,
                              "signif", "ns")
      all_degs[[length(all_degs) + 1]] <- de_res
    }
    rm(seu_sub); gc(verbose = FALSE)
  }

  if (length(all_degs) == 0) {
    log_warn("[Step4] No DEGs identified.")
    return(invisible(NULL))
  }

  deg_all <- do.call(rbind, all_degs)
  deg_all$direction <- ifelse(deg_all$avg_log2FC > 0, "up", "down")
  save_table(deg_all, "04_all_degs", cfg)

  sig_degs <- deg_all[deg_all$signif == "signif", ]
  save_table(sig_degs, "04_signif_degs", cfg)
  log_info("[Step4] Total DEGs: ", nrow(deg_all),
           " | Significant: ", nrow(sig_degs))

  # 4.1 DEG 数量条形图
  deg_summary <- aggregate(gene ~ cell_type + comparison + direction,
                           data = sig_degs, FUN = length)
  colnames(deg_summary)[4] <- "n_degs"
  deg_summary$comparison <- factor(deg_summary$comparison,
                                   levels = paste0(dpi_conditions, "_vs_Ctrl"))
  p_deg_bar <- ggplot(deg_summary, aes(x = cell_type, y = n_degs,
                                       fill = direction)) +
    geom_col(position = "dodge") +
    facet_wrap(~comparison, ncol = 3) +
    scale_fill_manual(values = c(up = "#E64B35", down = "#4DBBD5")) +
    labs(title = "Significant DEGs per cell type",
         x = "Cell type", y = "# DEGs (adj.p<0.05, |log2FC|>=0.58)") +
    theme_pub(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_figure(p_deg_bar, "04_deg_count_barplot", cfg, width = 12, height = 6)

  # 4.2 Top DEG 热图 (每细胞类型 top 20)
  top_genes_per_ct <- lapply(unique(sig_degs$cell_type), function(ct) {
    sub <- sig_degs[sig_degs$cell_type == ct, ]
    sub <- sub[order(-abs(sub$avg_log2FC)), ]
    head(sub$gene, 20)
  })
  top_genes <- unique(unlist(top_genes_per_ct))
  if (length(top_genes) > 0) {
    expr_data <- Seurat::GetAssayData(seu, assay = "RNA", layer = "data")
    top_genes_avail <- intersect(top_genes, rownames(expr_data))
    if (length(top_genes_avail) >= 5) {
      mat <- as.matrix(expr_data[top_genes_avail, ])
      meta_heat <- seu@meta.data[, c("CellType", "Condition")]
      anno_colors <- list(
        Condition = CONDITION_COLORS,
        CellType = safe_color(unique(meta_heat$CellType))
      )
      if (requireNamespace("ComplexHeatmap", quietly = TRUE)) {
        library(ComplexHeatmap)
        library(circlize)
        col_fun <- colorRamp2(c(-1, 0, 1), c("#2166AC", "white", "#B2182B"))
        ha <- HeatmapAnnotation(
          Condition = meta_heat$Condition,
          CellType = meta_heat$CellType,
          col = anno_colors,
          show_annotation_name = TRUE
        )
        ht <- Heatmap(mat, name = "Expr", col = col_fun,
                      top_annotation = ha,
                      show_column_names = FALSE,
                      show_row_names = TRUE,
                      row_names_gp = gpar(fontsize = 6),
                      cluster_columns = FALSE,
                      column_split = factor(meta_heat$Condition,
                                            levels = cfg$analysis$condition_levels),
                      use_raster = TRUE, raster_quality = 3)
        png(file.path(cfg$project$figures_dir, "04_top_deg_heatmap.png"),
            width = 12, height = 10, units = "in", res = 300)
        draw(ht, column_title = "Top DEGs across conditions",
             column_title_gp = gpar(fontsize = 14, fontface = "bold"))
        dev.off()
        log_info("[Step4] Saved heatmap: 04_top_deg_heatmap.png")
      }
    }
  }

  # 4.3 铁衰老基因在 DEG 中的富集
  fa_genes <- load_ferroaging_genes(cfg)
  fa_mouse <- map_human_to_mouse(fa_genes)
  fa_in_deg <- sig_degs[sig_degs$gene %in% fa_mouse, ]
  if (nrow(fa_in_deg) > 0) {
    save_table(fa_in_deg, "04_ferroaging_degs", cfg)
    log_info("[Step4] Ferroaging genes in DEGs: ", nrow(fa_in_deg))

    p_fa_deg <- ggplot(fa_in_deg, aes(x = cell_type, y = avg_log2FC,
                                      fill = comparison)) +
      geom_boxplot(outlier.size = 0.3) +
      scale_fill_manual(values = c("1DPI_vs_Ctrl" = "#E64B35",
                                   "3DPI_vs_Ctrl" = "#F39B7F",
                                   "7DPI_vs_Ctrl" = "#8491B4")) +
      labs(title = "Ferroaging DEGs log2FC distribution",
           x = "Cell type", y = "avg log2FC") +
      theme_pub(base_size = 10) +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    save_figure(p_fa_deg, "04_ferroaging_deg_boxplot", cfg,
                width = 11, height = 6)
  }

  saveRDS(deg_all, file.path(cfg$project$rds_dir, "deg_all.rds"))
  log_info("[Step4] DE analysis done.")
  invisible(deg_all)
}
