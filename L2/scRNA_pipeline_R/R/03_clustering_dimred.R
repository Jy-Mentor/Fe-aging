# ============================================================================
# STEP 3: 细胞分群与降维可视化
# - 使用已有 PCA/UMAP/Celltypes 注释
# - 添加铁衰老基因集评分 (UCell)
# - 可视化条件与细胞类型的 UMAP
# ============================================================================

step03_clustering_dimred <- function(seu, cfg) {
  log_info("[Step3] Clustering & dimensionality reduction...")

  # 3.1 检查现有 reductions
  reductions <- SeuratObject::Reductions(seu)
  if (!"umap" %in% reductions) {
    stop("UMAP not found in Seurat object. Run preprocessing first.")
  }
  log_info("[Step3] Existing reductions: ", paste(reductions, collapse = ", "))

  seu$Condition <- factor(seu[[cfg$analysis$condition_col, drop = TRUE]],
                          levels = cfg$analysis$condition_levels)
  seu$CellType <- seu[[cfg$analysis$celltype_col, drop = TRUE]]

  # 3.2 UMAP - cell type
  p_ct <- Seurat::DimPlot(seu, group.by = "CellType", reduction = "umap",
                          label = TRUE, label.size = 3) +
    scale_color_manual(values = safe_color(unique(seu$CellType))) +
    labs(title = "UMAP - Cell Types") +
    theme_pub() + theme(legend.position = "right")
  save_figure(p_ct, "03_umap_celltype", cfg, width = 11, height = 8)

  # 3.3 UMAP - condition split
  p_cond <- Seurat::DimPlot(seu, group.by = "Condition", reduction = "umap") +
    scale_color_manual(values = CONDITION_COLORS) +
    labs(title = "UMAP - Condition") +
    theme_pub()
  save_figure(p_cond, "03_umap_condition", cfg, width = 9, height = 7)

  p_cond_split <- Seurat::DimPlot(seu, group.by = "Condition",
                                  reduction = "umap", split.by = "Condition",
                                  ncol = 4) +
    scale_color_manual(values = CONDITION_COLORS) +
    labs(title = "UMAP - Condition (split)") +
    theme_pub() + theme(legend.position = "none")
  save_figure(p_cond_split, "03_umap_condition_split", cfg,
              width = 16, height = 5)

  # 3.4 铁衰老基因集评分 (UCell) - 优先复用既有 FA_96_UCell
  fa_col <- cfg$data$ferroaging_col_ucell
  if (fa_col %in% colnames(seu@meta.data)) {
    log_info("[Step3] Reusing existing ferroaging score column: ", fa_col)
    seu$Ferroaging <- seu@meta.data[[fa_col]]
  } else {
    fa_genes <- load_ferroaging_genes(cfg)
    fa_mouse <- map_human_to_mouse(fa_genes)
    log_info("[Step3] Ferroaging genes (human): ", length(fa_genes),
             "; mapped to mouse: ", length(fa_mouse))
    gene_check <- intersect_with_seurat(fa_mouse, seu)
    fa_available <- gene_check$common

    if (requireNamespace("UCell", quietly = TRUE)) {
      library(UCell)
      log_info("[Step3] Computing UCell ferroaging score with ",
               length(fa_available), " genes...")
      seu <- UCell::AddModuleScore_UCell(
        seu, features = list(Ferroaging = fa_available),
        ncores = 2, BPPARAM = BiocParallel::SerialParam()
      )
      log_info("[Step3] UCell score column: Ferroaging")
    } else {
      log_warn("[Step3] UCell not installed, falling back to Seurat AddModuleScore")
      seu <- Seurat::AddModuleScore(seu, features = list(Ferroaging = fa_available),
                                    name = "Ferroaging")
      colnames(seu@meta.data)[grepl("^Ferroaging", colnames(seu@meta.data))] <- "Ferroaging"
    }
  }

  # 3.5 UMAP ferroaging score
  p_fa <- Seurat::FeaturePlot(seu, features = "Ferroaging",
                              reduction = "umap") +
    scale_colour_gradientn(colors = DIVERGE_PALETTE) +
    labs(title = "Ferroaging score (UCell)") +
    theme_pub()
  save_figure(p_fa, "03_umap_ferroaging_score", cfg, width = 9, height = 7)

  # 3.6 铁衰老评分按条件
  meta_df <- seu@meta.data
  p_fa_cond <- ggplot(meta_df, aes(x = Condition, y = Ferroaging,
                                   fill = Condition)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.2, outlier.size = 0.3, fill = "white") +
    scale_fill_manual(values = CONDITION_COLORS) +
    labs(title = "Ferroaging score by Condition",
         y = "UCell Ferroaging score") +
    theme_pub() + theme(legend.position = "none")
  save_figure(p_fa_cond, "03_ferroaging_score_by_condition", cfg,
              width = 8, height = 6)

  # 3.7 铁衰老评分按条件 × 细胞类型
  meta_df$CellType <- factor(meta_df[[cfg$analysis$celltype_col]])
  p_fa_ct <- ggplot(meta_df, aes(x = Condition, y = Ferroaging,
                                 fill = Condition)) +
    geom_boxplot(outlier.size = 0.2, alpha = 0.7) +
    scale_fill_manual(values = CONDITION_COLORS) +
    facet_wrap(~CellType, scales = "free_y", ncol = 4) +
    labs(title = "Ferroaging score by CellType x Condition",
         y = "UCell score") +
    theme_pub(base_size = 9) + theme(legend.position = "none",
                                     axis.text.x = element_text(angle = 45, hjust = 1))
  save_figure(p_fa_ct, "03_ferroaging_score_celltype_condition", cfg,
              width = 14, height = 9)

  # 3.8 统计：Wilcoxon Ctrl vs DPI 各组
  stat_rows <- list()
  for (ct in unique(meta_df$CellType)) {
    for (cond in c("1DPI", "3DPI", "7DPI")) {
      x <- meta_df$Ferroaging[meta_df$CellType == ct & meta_df$Condition == cond]
      y <- meta_df$Ferroaging[meta_df$CellType == ct & meta_df$Condition == "Ctrl"]
      if (length(x) >= 5 && length(y) >= 5) {
        wt <- suppressWarnings(wilcox.test(x, y))
        stat_rows[[length(stat_rows) + 1]] <- data.frame(
          cell_type = ct, comparison = paste0(cond, "_vs_Ctrl"),
          n_target = length(x), n_ctrl = length(y),
          mean_target = mean(x), mean_ctrl = mean(y),
          log2FC = log2(mean(x) + 1e-6) - log2(mean(y) + 1e-6),
          p_value = wt$p.value
        )
      }
    }
  }
  stat_df <- do.call(rbind, stat_rows)
  stat_df$padj <- p.adjust(stat_df$p_value, method = "BH")
  save_table(stat_df, "03_ferroaging_score_wilcoxon", cfg)

  # 3.9 细胞类型组成
  comp_df <- as.data.frame(table(seu$Condition, seu$CellType))
  colnames(comp_df) <- c("Condition", "CellType", "n")
  comp_df$prop <- ave(comp_df$n, comp_df$Condition, FUN = function(x) x / sum(x))
  p_comp <- ggplot(comp_df, aes(x = Condition, y = prop, fill = CellType)) +
    geom_col(width = 0.9) +
    scale_fill_manual(values = safe_color(unique(comp_df$CellType))) +
    labs(title = "Cell type composition by Condition",
         y = "Proportion") +
    theme_pub()
  save_figure(p_comp, "03_celltype_composition", cfg, width = 9, height = 6)

  saveRDS(seu, file.path(cfg$project$rds_dir, "seurat_with_ferroaging.rds"))
  log_info("[Step3] Saved Seurat with ferroaging score.")
  invisible(seu)
}
