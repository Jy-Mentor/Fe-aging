# ============================================================================
# STEP 07: L3 scRNA-seq 数据加载 + Harmony 多样本整合
# - 优先加载已处理的 Seurat RDS (GSE233815, Zucha 2024, PMID:39499634)
# - 否则从 10x 输出读取原始 GSE233815 scRNA/snRNA 文件
# - QC: nFeature/percent.mt 过滤
# - Normalize + HVG + PCA + Harmony 整合
# 参考:
#   - Hao Y et al. 2024 Nat Biotechnol 42:293-304 (Seurat v5, PMID:37231261)
#   - Korsunsky I et al. 2019 Nat Methods (Harmony)
# ============================================================================

step07_sc_load_integrate <- function(cfg) {
  log_info("[Step07-L3] scRNA loading + Harmony integration...")

  require_packages(c("Seurat", "harmony"),
                   install_hint = "install.packages(c('Seurat','harmony'))")
  suppressPackageStartupMessages({
    library(Seurat)
    library(harmony)
  })

  # --------------------------------------------------------------------------
  # 7.1 数据加载 (优先 Seurat RDS, 否则 Read10X)
  # --------------------------------------------------------------------------
  seu_rds <- cfg$data$sc_seurat_rds
  sc_samples <- cfg$data$sc_samples

  seu <- NULL

  if (!is.null(seu_rds) && file.exists(seu_rds)) {
    log_info("[Step07] Loading pre-built Seurat object: ", seu_rds)
    seu <- readRDS(seu_rds)
    log_info("[Step07] Loaded: ", nrow(seu), " genes x ", ncol(seu), " cells")
  } else {
    log_info("[Step07] Pre-built RDS not found, reading 10x outputs from GSE233815...")
    seu_list <- list()
    for (sn in names(sc_samples)) {
      sample_dir <- sc_samples[[sn]]
      if (!dir.exists(sample_dir)) {
        log_warn("[Step07] Sample dir not found: ", sample_dir, ". Skip ", sn)
        next
      }
      log_info("[Step07] Reading 10x: ", sn)
      mat <- Read10X(sample_dir)
      seu_s <- CreateSeuratObject(mat, project = sn, min.cells = 3, min.features = 200)
      seu_s$orig.ident <- sn
      seu_list[[sn]] <- seu_s
    }
    if (length(seu_list) == 0) {
      stop("No scRNA samples loaded. Check data paths in config.")
    }
    seu <- merge(seu_list[[1]], y = seu_list[-1],
                  add.cell.ids = names(seu_list))
    log_info("[Step07] Merged: ", nrow(seu), " genes x ", ncol(seu), " cells")
  }

  # --------------------------------------------------------------------------
  # 7.2 QC: 计算 percent.mt + 过滤
  # --------------------------------------------------------------------------
  if (!"percent.mt" %in% colnames(seu@meta.data)) {
    log_info("[Step07] Computing percent.mt...")
    seu[["percent.mt"]] <- PercentageFeatureSet(seu, pattern = "^mt-")
  }
  if (!"nFeature_RNA" %in% colnames(seu@meta.data)) {
    seu$nFeature_RNA <- Matrix::colSums(GetAssayData(seu, assay = "RNA",
                                                      layer = "counts") > 0)
  }
  if (!"nCount_RNA" %in% colnames(seu@meta.data)) {
    seu$nCount_RNA <- Matrix::colSums(GetAssayData(seu, assay = "RNA",
                                                    layer = "counts"))
  }

  n_before <- ncol(seu)
  seu <- seu[, seu$nFeature_RNA >= cfg$sc$min_nfeature &
              seu$nFeature_RNA <= cfg$sc$max_nfeature &
              seu$percent.mt < cfg$sc$max_percent_mt]
  n_after <- ncol(seu)
  log_info("[Step07] QC: ", n_before, " -> ", n_after,
           " cells (nFeature ", cfg$sc$min_nfeature, "-",
           cfg$sc$max_nfeature, ", %mt < ", cfg$sc$max_percent_mt, ")")

  # QC 可视化
  qc_df <- seu@meta.data[, c("nCount_RNA", "nFeature_RNA", "percent.mt")]
  if ("Condition" %in% colnames(seu@meta.data)) {
    qc_df$Condition <- seu$Condition
  } else if ("orig.ident" %in% colnames(seu@meta.data)) {
    qc_df$Condition <- seu$orig.ident
  }

  p_qc <- ggplot(qc_df, aes(x = nCount_RNA, y = nFeature_RNA, color = percent.mt)) +
    geom_point(alpha = 0.4, size = 0.6) +
    scale_color_viridis_c() +
    geom_hline(yintercept = cfg$sc$min_nfeature, linetype = "dashed", color = "red") +
    geom_hline(yintercept = cfg$sc$max_nfeature, linetype = "dashed", color = "red") +
    labs(title = "QC: nFeature vs nCount (color = %mt)",
         x = "nCount_RNA", y = "nFeature_RNA", color = "%mt") +
    theme_pub(base_size = 10)
  save_figure(p_qc, "07_sc_qc_scatter", cfg, width = 8, height = 6)

  # --------------------------------------------------------------------------
  # 7.3 Normalize + HVG + Scale + PCA
  # --------------------------------------------------------------------------
  log_info("[Step07] NormalizeData + FindVariableFeatures + ScaleData + RunPCA...")
  seu <- NormalizeData(seu, verbose = FALSE)
  seu <- FindVariableFeatures(seu, nfeatures = cfg$sc$nhvg, verbose = FALSE)
  seu <- ScaleData(seu, verbose = FALSE)
  seu <- RunPCA(seu, npcs = cfg$sc$npcs, verbose = FALSE)

  # --------------------------------------------------------------------------
  # 7.4 Harmony 整合 (按 batch/condition)
  # --------------------------------------------------------------------------
  batch_col <- if ("Condition" %in% colnames(seu@meta.data)) {
    "Condition"
  } else if ("orig.ident" %in% colnames(seu@meta.data)) {
    "orig.ident"
  } else {
    stop("Neither 'Condition' nor 'orig.ident' in meta.data")
  }

  log_info("[Step07] RunHarmony by '", batch_col, "'...")
  seu <- RunHarmony(seu,
                    group.by.vars = batch_col,
                    reduction.use = "pca",
                    dims.use = 1:cfg$sc$npcs,
                    reduction.save = "harmony",
                    theta = cfg$sc$harmony_theta,
                    lambda = cfg$sc$harmony_lambda,
                    max_iter = cfg$sc$harmony_max_iter,
                    verbose = FALSE)

  # --------------------------------------------------------------------------
  # 7.5 UMAP (基于 harmony 嵌入)
  # --------------------------------------------------------------------------
  seu <- RunUMAP(seu, reduction = "harmony",
                  dims = 1:cfg$sc$npcs,
                  n.neighbors = cfg$sc$umap_n_neighbors,
                  min.dist = cfg$sc$umap_min_dist,
                  verbose = FALSE)

  # 聚类 (基于 harmony)
  seu <- FindNeighbors(seu, reduction = "harmony",
                        dims = 1:cfg$sc$npcs, verbose = FALSE)
  seu <- FindClusters(seu, resolution = cfg$sc$cluster_resolution, verbose = FALSE)

  # --------------------------------------------------------------------------
  # 7.6 整合效果可视化
  # --------------------------------------------------------------------------
  p_umap_batch_before <- DimPlot(seu, reduction = "umap", group.by = batch_col,
                                  cols = get_condition_colors(unique(seu@meta.data[[batch_col]]))) +
    labs(title = "UMAP after Harmony (colored by batch)") +
    theme_pub(base_size = 10)
  save_figure(p_umap_batch_before, "07_sc_umap_by_batch_after_harmony", cfg,
              width = 9, height = 7)

  p_umap_cluster <- DimPlot(seu, reduction = "umap", group.by = "seurat_clusters") +
    labs(title = "UMAP clusters (after Harmony)") +
    theme_pub(base_size = 10)
  save_figure(p_umap_cluster, "07_sc_umap_clusters", cfg, width = 9, height = 7)

  save_rds(seu, "07_sc_seurat_integrated", cfg)
  log_info("[Step07] scRNA loading + Harmony done. ", ncol(seu), " cells.")
  invisible(seu)
}
