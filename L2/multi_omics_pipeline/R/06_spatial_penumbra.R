# ============================================================================
# STEP 06: L2 Spatial 半暗带识别 + 空间变量特征
# - 基于 Neuron score (高) + Stress score (中) 定义半暗带
# - 与梗死核心 (Neuron 低 + 应激极高) + 对侧 (Neuron 高 + 应激低) 对比
# - FindSpatiallyVariableFeatures (Moran's I) 提取空间变量基因
# - 验证半暗带 vs 梗死核心 vs 对侧的铁衰老得分差异
# 参考:
#   - Han B et al. 2024 Sci Transl Med PMID:38324639 (空间+sc 整合标杆)
#   - Zucha D et al. 2024 PNAS PMID:39499634 (时空转录组)
# ============================================================================

step06_spatial_penumbra <- function(spatial_merged, cfg) {
  log_info("[Step06-L2] Penumbra identification + spatial variable features...")

  if (is.null(spatial_merged)) stop("spatial_merged is NULL. Run step 05 first.")
  require_packages(c("Seurat"))
  suppressPackageStartupMessages(library(Seurat))

  gene_sets <- build_gene_sets(cfg, organism = "mouse")

  # --------------------------------------------------------------------------
  # 6.1 计算 Neuron / Stress / Infarct_core 得分
  # --------------------------------------------------------------------------
  use_ucell <- requireNamespace("UCell", quietly = TRUE)
  if (use_ucell) suppressPackageStartupMessages(library(UCell))

  region_signatures <- list(
    Neuron      = gene_sets$celltype_markers$Neuron,
    Stress      = gene_sets$stress,
    InfarctCore = gene_sets$infarct_core
  )

  log_info("[Step06] Computing Neuron/Stress/InfarctCore scores...")
  if (use_ucell) {
    spatial_merged <- AddModuleScore_UCell(
      spatial_merged, features = region_signatures,
      name = NULL, w_neg = 1.0,
      maxRank = cfg$sc$ucell_max_rank
    )
  } else {
    spatial_merged <- AddModuleScore(
      spatial_merged, features = region_signatures, name = "RegionScore_"
    )
    # 重命名 (假设顺序与 region_signatures 一致)
    new_cols <- paste0("RegionScore_", seq_along(region_signatures), "1")
    for (i in seq_along(region_signatures)) {
      old_col <- paste0("RegionScore_", i, "1")
      new_col <- names(region_signatures)[i]
      if (old_col %in% colnames(spatial_merged@meta.data)) {
        colnames(spatial_merged@meta.data)[
          colnames(spatial_merged@meta.data) == old_col] <- new_col
      }
    }
  }

  # --------------------------------------------------------------------------
  # 6.2 基于评分定义组织区域 (Penumbra / InfarctCore / Healthy / Other)
  # --------------------------------------------------------------------------
  neuron_thr <- cfg$spatial$penumbra_neuron_score_min
  stress_thr <- cfg$spatial$penumbra_stress_score_min

  meta <- spatial_merged@meta.data
  if (!all(c("Neuron", "Stress") %in% colnames(meta))) {
    stop("Neuron/Stress scores not found in meta.data")
  }

  spatial_merged$region <- with(meta, ifelse(
    Neuron > neuron_thr & Stress > stress_thr,
    "Penumbra",
    ifelse(Neuron < neuron_thr & Stress > stress_thr,
           "InfarctCore",
           ifelse(Neuron > neuron_thr & Stress < stress_thr,
                  "Healthy", "Other"))
  ))

  # 统计每个样本的区域分布
  region_tab <- as.data.frame(table(spatial_merged$sample, spatial_merged$region))
  colnames(region_tab) <- c("sample", "region", "n_spots")
  save_table(region_tab, "06_spatial_region_distribution", cfg)

  log_info("[Step06] Region distribution per sample:")
  print(region_tab)

  # --------------------------------------------------------------------------
  # 6.3 区域可视化 (SpatialDimPlot)
  # --------------------------------------------------------------------------
  samples <- unique(spatial_merged$sample)
  for (sn in samples) {
    cells_sn <- colnames(spatial_merged)[spatial_merged$sample == sn]
    seu_sub <- spatial_merged[, cells_sn]
    tryCatch({
      p <- SpatialDimPlot(seu_sub, group.by = "region",
                           cols = c(Penumbra = "#F4A582",
                                    InfarctCore = "#B2182B",
                                    Healthy = "#2166AC",
                                    Other = "grey80")) +
        labs(title = paste0(sn, " - Defined regions"))
      save_figure(p, sprintf("06_spatial_region_%s", sn), cfg, width = 8, height = 7)
    }, error = function(e) {
      log_warn("[Step06] SpatialDimPlot failed for ", sn, ": ", conditionMessage(e))
    })
    rm(seu_sub); gc(verbose = FALSE)
  }

  # --------------------------------------------------------------------------
  # 6.4 各区域的铁衰老得分对比 (VlnPlot)
  # --------------------------------------------------------------------------
  fa_cols <- c("Ferroptosis", "Senescence", "Ferroaging", "Ferrosenescence")
  fa_avail <- intersect(fa_cols, colnames(spatial_merged@meta.data))

  if (length(fa_avail) > 0) {
    for (col in fa_avail) {
      tryCatch({
        p <- VlnPlot(spatial_merged, features = col, group.by = "region",
                      split.by = "sample", pt.size = 0.1,
                      cols = get_condition_colors(samples)) +
          labs(title = paste0(col, " score by region x sample")) +
          theme_pub(base_size = 10)
        save_figure(p, sprintf("06_spatial_region_violin_%s", col), cfg,
                    width = 12, height = 6)
      }, error = function(e) {
        log_warn("[Step06] VlnPlot failed for ", col, ": ", conditionMessage(e))
      })
    }

    # 区域 × 样本的铁衰老得分统计
    region_score_df <- do.call(rbind, lapply(samples, function(sn) {
      cells_sn <- colnames(spatial_merged)[spatial_merged$sample == sn]
      meta <- spatial_merged@meta.data[cells_sn, , drop = FALSE]
      meta$spot_id <- cells_sn
      meta
    }))
    save_table(region_score_df[, c("spot_id", "sample", "region", fa_avail)],
               "06_spatial_region_scores", cfg)
  }

  # --------------------------------------------------------------------------
  # 6.5 FindSpatiallyVariableFeatures (Moran's I)
  # --------------------------------------------------------------------------
  log_info("[Step06] FindSpatiallyVariableFeatures (Moran's I)...")
  for (sn in samples) {
    cells_sn <- colnames(spatial_merged)[spatial_merged$sample == sn]
    seu_sub <- spatial_merged[, cells_sn]

    tryCatch({
      seu_sub <- FindSpatiallyVariableFeatures(
        seu_sub,
        assay = "SCT",
        features = VariableFeatures(seu_sub)[seq_len(min(cfg$spatial$moransi_nfeatures,
                                                          length(VariableFeatures(seu_sub))))],
        selection.method = "moransi",
        r.metric = cfg$spatial$moransi_r_metric,
        x.cuts = cfg$spatial$moransi_x_cuts,
        y.cuts = cfg$spatial$moransi_y_cuts
      )

      spatial_features <- SpatiallyVariableFeatures(seu_sub,
                                                    selection.method = "moransi")
      log_info("[Step06] ", sn, " spatially variable features: ",
               length(spatial_features))

      # 与铁衰老基因集的交集
      fa_overlap <- intersect(spatial_features, gene_sets$ferrosenescence)
      log_info("[Step06] ", sn, " spatial-FA overlap: ", length(fa_overlap),
               " - ", paste(head(fa_overlap, 10), collapse = ", "))

      save_table(data.frame(
        sample = sn,
        feature = spatial_features,
        rank = seq_along(spatial_features),
        is_ferroaging = spatial_features %in% gene_sets$ferrosenescence
      ), sprintf("06_spatial_variable_features_%s", sn), cfg)
    }, error = function(e) {
      log_warn("[Step06] FindSpatiallyVariableFeatures failed for ", sn,
               ": ", conditionMessage(e))
    })

    rm(seu_sub); gc(verbose = FALSE)
  }

  save_rds(spatial_merged, "06_spatial_with_regions", cfg)
  log_info("[Step06] Spatial penumbra + SVF done.")
  invisible(spatial_merged)
}
