# ============================================================================
# STEP 04: L2 Spatial 空间转录组数据加载与 SCTransform
# - 读取 GSE233814 的 10x Visium 数据 (C1/D1/D3/D7)
# - SCTransform 标准化 (替代 LogNormalize, 适合 spot 差异大)
# - PCA + UMAP 降维
# 参考:
#   - Hao Y et al. 2024 Nat Biotechnol (Seurat v5)
#   - Stuart T et al. 2019 Cell (SCTransform)
# ============================================================================

step04_spatial_load_qc <- function(cfg) {
  log_info("[Step04-L2] Spatial transcriptomics loading & SCTransform...")

  require_packages(c("Seurat"),
                   install_hint = "install.packages('Seurat')")
  suppressPackageStartupMessages(library(Seurat))

  spatial_samples <- cfg$data$spatial_samples
  if (is.null(spatial_samples) || length(spatial_samples) == 0) {
    stop("No spatial samples configured in cfg$data$spatial_samples")
  }

  spatial_list <- list()
  for (sn in names(spatial_samples)) {
    sample_dir <- spatial_samples[[sn]]
    log_info("[Step04] Loading spatial sample: ", sn, " from ", sample_dir)

    if (!dir.exists(sample_dir)) {
      log_warn("[Step04] Sample dir not found: ", sample_dir, ". Skipping ", sn)
      next
    }

    # 检查 spaceranger 输出结构
    # 期望: filtered_feature_bc_matrix/ + spatial/ (含 tissue_positions_list.csv)
    seu <- tryCatch({
      Load10X_Spatial(data.dir = sample_dir)
    }, error = function(e) {
      log_error("[Step04] Load10X_Spatial failed for ", sn, ": ",
                conditionMessage(e))
      return(NULL)
    })
    if (is.null(seu)) next

    # 添加样本元数据
    seu$sample <- sn
    seu$condition <- sn  # C1/D1/D3/D7

    log_info("[Step04] ", sn, ": ", nrow(seu), " genes x ", ncol(seu), " spots")

    # QC: 检查 nCount_Spatial, nFeature_Spatial
    if (!"nCount_Spatial" %in% colnames(seu@meta.data)) {
      seu$nCount_Spatial <- Matrix::colSums(GetAssayData(seu, assay = "Spatial", layer = "counts"))
      seu$nFeature_Spatial <- Matrix::colSums(GetAssayData(seu, assay = "Spatial", layer = "counts") > 0)
    }

    log_info("[Step04] ", sn, " QC summary:")
    log_info("  nCount_Spatial: [", min(seu$nCount_Spatial), ", ",
             max(seu$nCount_Solar), "]")
    log_info("  nFeature_Spatial: [", min(seu$nFeature_Spatial), ", ",
             max(seu$nFeature_Spatial), "]")

    # SCTransform 标准化 (替代 NormalizeData + ScaleData)
    log_info("[Step04] SCTransform for ", sn)
    seu <- SCTransform(seu, assay = "Spatial",
                       verbose = FALSE,
                       variable.features.n = cfg$spatial$sct_nfeatures)

    # PCA + UMAP
    seu <- RunPCA(seu, npcs = cfg$spatial$pca_npcs, verbose = FALSE)
    seu <- RunUMAP(seu, dims = 1:cfg$spatial$pca_npcs, verbose = FALSE)

    spatial_list[[sn]] <- seu
    log_info("[Step04] ", sn, " processed: ", nrow(seu), " genes x ",
             ncol(seu), " spots")
  }

  if (length(spatial_list) == 0) {
    stop("No spatial samples loaded successfully. Check data paths.")
  }

  # --------------------------------------------------------------------------
  # 4.1 合并多个切片 (Seurat v5 推荐使用 merge + IntegrateLayers)
  # --------------------------------------------------------------------------
  if (length(spatial_list) > 1) {
    log_info("[Step04] Merging ", length(spatial_list), " spatial samples...")
    spatial_merged <- merge(
      spatial_list[[1]],
      y = spatial_list[-1],
      add.cell.ids = names(spatial_list),
      project = "GSE233814_spatial"
    )
    log_info("[Step04] Merged object: ", nrow(spatial_merged), " genes x ",
             ncol(spatial_merged), " spots")

    # v5 layer split by sample
    spatial_merged[["Spatial"]] <- split(spatial_merged[["Spatial"]],
                                          f = spatial_merged$sample)

    # IntegrateLayers with Harmony (推荐用于空间样本整合)
    if (requireNamespace("harmony", quietly = TRUE)) {
      log_info("[Step04] IntegrateLayers (Harmony) for multi-sample...")
      spatial_merged <- IntegrateLayers(
        spatial_merged,
        method = HarmonyIntegration,
        orig.reduction = "pca",
        new.reduction = "harmony",
        verbose = FALSE
      )
      spatial_merged <- RunUMAP(spatial_merged, reduction = "harmony",
                                 dims = 1:cfg$spatial$pca_npcs,
                                 reduction.name = "umap.harmony",
                                 verbose = FALSE)
    } else {
      log_warn("[Step04] harmony not installed; using merge without integration")
    }
  } else {
    spatial_merged <- spatial_list[[1]]
  }

  save_rds(spatial_merged, "04_spatial_merged", cfg)
  save_rds(spatial_list, "04_spatial_list", cfg)

  # --------------------------------------------------------------------------
  # 4.2 QC 可视化
  # --------------------------------------------------------------------------
  qc_df <- do.call(rbind, lapply(names(spatial_list), function(sn) {
    seu <- spatial_list[[sn]]
    data.frame(
      sample = sn,
      spot_id = colnames(seu),
      nCount = seu$nCount_Spatial,
      nFeature = seu$nFeature_Spatial,
      stringsAsFactors = FALSE
    )
  }))

  p1 <- ggplot(qc_df, aes(x = sample, y = log10(nCount), fill = sample)) +
    geom_violin(trim = FALSE) +
    geom_boxplot(width = 0.1, outlier.size = 0.3) +
    scale_fill_manual(values = get_condition_colors(names(spatial_list))) +
    labs(title = "Spatial QC: nCount (log10) by sample",
         x = "Sample", y = "log10(nCount_Spatial)") +
    theme_pub(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    guides(fill = "none")
  save_figure(p1, "04_spatial_qc_violin_ncount", cfg, width = 8, height = 6)

  p2 <- ggplot(qc_df, aes(x = sample, y = nFeature, fill = sample)) +
    geom_violin(trim = FALSE) +
    geom_boxplot(width = 0.1, outlier.size = 0.3) +
    scale_fill_manual(values = get_condition_colors(names(spatial_list))) +
    labs(title = "Spatial QC: nFeature by sample",
         x = "Sample", y = "nFeature_Spatial") +
    theme_pub(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1)) +
    guides(fill = "none")
  save_figure(p2, "04_spatial_qc_violin_nfeature", cfg, width = 8, height = 6)

  log_info("[Step04] Spatial loading & SCTransform done.")
  invisible(spatial_list)
}
