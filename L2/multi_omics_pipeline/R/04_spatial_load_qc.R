# ============================================================================
# STEP 04: L2 Spatial 空间转录组数据加载与 SCTransform
# - 优先使用作者已处理的 Seurat RDS (mendeley 数据集, 含空间坐标)
# - 若 RDS 不可用, 回退到原始 10x 文件 (需手动构建, 因为 RAW.tar 未提供
#   tissue_positions_list.csv, 此情况下仅能构建非空间 Seurat 对象)
# - 数据来源: GSE233815 (Zucha et al. 2024 PNAS, PMID:39499634)
#   - 5 个 10x Visium 样本: C1-control / B1-D1 / D1-D3 / C1-D7 / D1-D7
#   - 作者 RDS: seurat_1stSpatial.rds (含 C1-control+B1-D1 整合),
#               seurat_2ndSpatial.rds (含 D1-D3+C1-D7 整合),
#               spatial_seurat_1DP_13_nygen.rds (D1-D7)
# 参考:
#   - Hao Y et al. 2024 Nat Biotechnol 42:293-304 (Seurat v5, PMID:37231261)
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

  # --------------------------------------------------------------------------
  # 4.1 优先加载作者已处理的 Seurat RDS (含空间坐标 + QC)
  # --------------------------------------------------------------------------
  rds_1st <- cfg$data$spatial_seurat_1st_rds
  rds_2nd <- cfg$data$spatial_seurat_2nd_rds
  rds_1dp <- cfg$data$spatial_seurat_1DP_rds

  spatial_list <- list()
  used_rds <- FALSE

  if (!is.null(rds_1st) && file.exists(rds_1st) &&
      !is.null(rds_2nd) && file.exists(rds_2nd)) {
    log_info("[Step04] Loading author-provided spatial Seurat RDS files...")
    log_info("[Step04]   1st: ", rds_1st)
    log_info("[Step04]   2nd: ", rds_2nd)

    # RDS 可能是单个 Seurat 或 list of Seurat (作者按切片组织)
    .load_rds_seurat_list <- function(rds_path, tag) {
      obj <- readRDS(rds_path)
      # 修复旧版 Seurat 对象 VisiumV1/VisiumV2 misc slot 缺失问题
      # SeuratObject 5.2+ 在 SpatialImage 类新增 misc slot (OptionalList)
      # 旧版 RDS 中 VisiumV1 对象没有 misc slot, merge 时 validObject 失败
      # 直接给缺失 misc slot 的 image 添加 list() 即可, 无需 UpdateSeuratObject
      .fix_visium_misc <- function(seu) {
        if (!inherits(seu, "Seurat")) return(seu)
        for (im_name in Images(seu)) {
          im_obj <- seu@images[[im_name]]
          need_fix <- tryCatch({
            validObject(im_obj); FALSE
          }, error = function(e) {
            msg <- conditionMessage(e)
            grepl("slots in class definition but not in object", msg)
          })
          if (need_fix) {
            cls_slots <- slotNames(class(im_obj))
            if ("misc" %in% cls_slots) {
              slot(im_obj, "misc") <- list()
              validObject(im_obj)
              seu@images[[im_name]] <- im_obj
              log_info("[Step04]   ", tag, "/", im_name,
                       ": misc slot added (Visium 5.2+ compat)")
            }
          }
        }
        seu
      }
      if (inherits(obj, "Seurat")) {
        obj <- .fix_visium_misc(obj)
        log_info("[Step04] ", tag, ": ", nrow(obj), " genes x ",
                 ncol(obj), " spots (single Seurat)")
        return(setNames(list(obj), tag))
      }
      if (is.list(obj)) {
        out <- list()
        for (i in seq_along(obj)) {
          el <- obj[[i]]
          if (inherits(el, "Seurat")) {
            el <- .fix_visium_misc(el)
            nm <- paste0(tag, "_", i)
            # 优先用 Sample 列作为切片标识
            if ("Sample" %in% colnames(el@meta.data)) {
              smp <- unique(el$Sample)
              if (length(smp) == 1) nm <- paste0(tag, "_", smp)
            }
            log_info("[Step04] ", nm, ": ", nrow(el), " genes x ",
                     ncol(el), " spots")
            out[[nm]] <- el
          }
        }
        return(out)
      }
      log_warn("[Step04] ", tag, " is neither Seurat nor list of Seurat (class: ",
               paste(class(obj), collapse = ","), "); skipping")
      list()
    }

    spatial_list <- c(spatial_list, .load_rds_seurat_list(rds_1st, "1stSpatial"))
    spatial_list <- c(spatial_list, .load_rds_seurat_list(rds_2nd, "2ndSpatial"))

    if (!is.null(rds_1dp) && file.exists(rds_1dp)) {
      spatial_list <- c(spatial_list, .load_rds_seurat_list(rds_1dp, "1DP"))
    }

    used_rds <- length(spatial_list) > 0
  } else {
    log_warn("[Step04] Author RDS files not all available; attempting raw 10x load")
    log_warn("[Step04] Note: RAW.tar does not contain tissue_positions_list.csv;")
    log_warn("[Step04] spatial coordinates will be unavailable if loading from raw.")
  }

  # --------------------------------------------------------------------------
  # 4.2 回退: 从原始 10x 文件加载 (无空间坐标, 仅表达矩阵)
  # --------------------------------------------------------------------------
  if (!used_rds) {
    for (sn in names(spatial_samples)) {
      sample_dir <- spatial_samples[[sn]]
      log_info("[Step04] Loading spatial sample from raw 10x: ", sn,
               " from ", sample_dir)

      if (!dir.exists(sample_dir)) {
        log_warn("[Step04] Sample dir not found: ", sample_dir, ". Skipping ", sn)
        next
      }

      # 查找 barcodes/features/matrix 文件 (文件名前缀含 GSM ID)
      barcode_file <- list.files(sample_dir, pattern = "_barcodes\\.tsv\\.gz$",
                                  full.names = TRUE)
      feature_file <- list.files(sample_dir, pattern = "_features\\.tsv\\.gz$",
                                  full.names = TRUE)
      matrix_file <- list.files(sample_dir, pattern = "_matrix\\.mtx\\.gz$",
                                 full.names = TRUE)

      if (length(barcode_file) == 0 || length(feature_file) == 0 ||
          length(matrix_file) == 0) {
        log_error("[Step04] Missing 10x files for ", sn,
                  ": barcodes=", length(barcode_file),
                  ", features=", length(feature_file),
                  ", matrix=", length(matrix_file))
        next
      }

      seu <- tryCatch({
        mat <- ReadMtx(mtx = matrix_file[1],
                       cells = barcode_file[1],
                       features = feature_file[1],
                       feature.column = 2)
        CreateSeuratObject(counts = mat, project = sn, assay = "Spatial")
      }, error = function(e) {
        log_error("[Step04] Raw 10x load failed for ", sn, ": ",
                  conditionMessage(e))
        NULL
      })
      if (is.null(seu)) next

      seu$sample <- sn
      seu$condition <- sn

      # 计算 QC 指标
      seu$nCount_Spatial <- Matrix::colSums(GetAssayData(seu, assay = "Spatial",
                                                          layer = "counts"))
      seu$nFeature_Spatial <- Matrix::colSums(GetAssayData(seu, assay = "Spatial",
                                                            layer = "counts") > 0)

      log_info("[Step04] ", sn, ": ", nrow(seu), " genes x ", ncol(seu), " spots")
      spatial_list[[sn]] <- seu
    }
  }

  if (length(spatial_list) == 0) {
    stop("No spatial samples loaded successfully. Check data paths in config.")
  }

  # 统一添加 sample 列 (小写) 用于后续 merge 后 split
  # 作者 RDS 中可能是 Sample (大写), 此处统一为 sample (小写)
  for (sn in names(spatial_list)) {
    seu <- spatial_list[[sn]]
    if (!"sample" %in% colnames(seu@meta.data)) {
      if ("Sample" %in% colnames(seu@meta.data)) {
        seu$sample <- seu$Sample
      } else {
        seu$sample <- sn
      }
    }
    # 统一添加 condition 列 (小写) 用于 Step 10/11/13 跨样本分组
    # 作者 RDS 中是 Condition (大写), 此处统一为 condition (小写)
    if (!"condition" %in% colnames(seu@meta.data)) {
      if ("Condition" %in% colnames(seu@meta.data)) {
        seu$condition <- seu$Condition
      } else {
        seu$condition <- sn
      }
    }
    spatial_list[[sn]] <- seu
  }

  # --------------------------------------------------------------------------
  # 4.3 SCTransform 标准化 (若作者 RDS 未做)
  # --------------------------------------------------------------------------
  for (sn in names(spatial_list)) {
    seu <- spatial_list[[sn]]
    if (!"SCT" %in% Assays(seu)) {
      log_info("[Step04] SCTransform for ", sn)
      seu <- SCTransform(seu, assay = "Spatial",
                         verbose = FALSE,
                         variable.features.n = cfg$spatial$sct_nfeatures)
      spatial_list[[sn]] <- seu
    } else {
      log_info("[Step04] ", sn, " already has SCT assay; skipping SCTransform")
    }

    # PCA + UMAP (若缺失)
    if (!"pca" %in% Reductions(seu)) {
      seu <- RunPCA(seu, npcs = cfg$spatial$pca_npcs, verbose = FALSE)
      spatial_list[[sn]] <- seu
    }
    if (!"umap" %in% Reductions(seu)) {
      seu <- RunUMAP(seu, dims = 1:cfg$spatial$pca_npcs, verbose = FALSE)
      spatial_list[[sn]] <- seu
    }

    # QC 报告
    if ("nCount_Spatial" %in% colnames(seu@meta.data)) {
      log_info("[Step04] ", sn, " QC summary:")
      log_info("  nCount_Spatial: [", min(seu$nCount_Spatial), ", ",
               max(seu$nCount_Spatial), "]")
      log_info("  nFeature_Spatial: [", min(seu$nFeature_Spatial), ", ",
               max(seu$nFeature_Spatial), "]")
    }
  }

  # --------------------------------------------------------------------------
  # 4.4 合并多个切片 (Seurat v5 merge + IntegrateLayers)
  # --------------------------------------------------------------------------
  if (length(spatial_list) > 1) {
    log_info("[Step04] Merging ", length(spatial_list), " spatial samples...")
    spatial_merged <- merge(
      spatial_list[[1]],
      y = spatial_list[-1],
      add.cell.ids = names(spatial_list),
      project = "GSE233815_spatial"
    )
    log_info("[Step04] Merged object: ", nrow(spatial_merged), " genes x ",
             ncol(spatial_merged), " spots")

    # v5 layer split by sample
    if ("Spatial" %in% Assays(spatial_merged)) {
      spatial_merged[["Spatial"]] <- split(spatial_merged[["Spatial"]],
                                            f = spatial_merged$sample)
    }

    # IntegrateLayers with Harmony (推荐用于空间样本整合)
    if (requireNamespace("harmony", quietly = TRUE) &&
        "pca" %in% Reductions(spatial_merged)) {
      log_info("[Step04] IntegrateLayers (Harmony) for multi-sample...")
      spatial_merged <- tryCatch({
        IntegrateLayers(
          spatial_merged,
          method = HarmonyIntegration,
          orig.reduction = "pca",
          new.reduction = "harmony",
          verbose = FALSE
        )
      }, error = function(e) {
        log_warn("[Step04] Harmony integration failed: ", conditionMessage(e))
        spatial_merged
      })
      if ("harmony" %in% Reductions(spatial_merged)) {
        spatial_merged <- RunUMAP(spatial_merged, reduction = "harmony",
                                   dims = 1:cfg$spatial$pca_npcs,
                                   reduction.name = "umap.harmony",
                                   verbose = FALSE)
      }
    } else {
      log_warn("[Step04] harmony not installed or PCA missing; using merge without integration")
    }
  } else {
    spatial_merged <- spatial_list[[1]]
  }

  save_rds(spatial_merged, "04_spatial_merged", cfg)
  save_rds(spatial_list, "04_spatial_list", cfg)

  # --------------------------------------------------------------------------
  # 4.5 QC 可视化
  # --------------------------------------------------------------------------
  qc_df <- do.call(rbind, lapply(names(spatial_list), function(sn) {
    seu <- spatial_list[[sn]]
    if (!"nCount_Spatial" %in% colnames(seu@meta.data)) return(NULL)
    data.frame(
      sample = sn,
      spot_id = colnames(seu),
      nCount = seu$nCount_Spatial,
      nFeature = seu$nFeature_Spatial,
      stringsAsFactors = FALSE
    )
  }))

  if (!is.null(qc_df) && nrow(qc_df) > 0) {
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
  } else {
    log_warn("[Step04] QC data frame empty; skipping violin plots")
  }

  log_info("[Step04] Spatial loading & SCTransform done.")
  invisible(spatial_list)
}
