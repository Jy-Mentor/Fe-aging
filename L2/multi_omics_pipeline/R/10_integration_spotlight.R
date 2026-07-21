# ============================================================================
# STEP 10: L4 SPOTlight 空间去卷积
# - 用 scRNA (GSE233815) 训练细胞类型特异 marker
# - 将细胞类型比例投影到空间切片 (GSE233815 spatial)
# - 比较梗死核心 / 半暗带 / 健康区的细胞类型比例
# - 评估神经元比例与铁衰老得分的相关性
# 参考:
#   - Macosko EZ et al. 2015 (SPOTlight based on NMFreg, PMID: 26000488)
#   - Moncada R et al. 2020 Nat Commun (SPOTlight, PMID: 31844000)
#   - Diaz-Mejia JJ et al. 2019 Mol Syst Biol (BuildSimilarity)
#   - Lun A 2016 scran (DOI: 10.18129/B9.bioc.scran, Bioconductor 3.22 v1.38.1)
#     - scoreMarkers: AUC + Cohen's d effect sizes, no p-value dependency
#     - 文档: https://bioconductor.org/packages/3.22/bioc/html/scran.html
#     - OSCA.book 推荐 mean.AUC > 0.8 作为 marker 筛选阈值
# ----------------------------------------------------------------------------
# 方法学说明 (SPOTlight vs RCTD, PubMed 文献验证 2026-07-20):
#   GSE233815 原论文 (Zucha et al. 2024 PNAS, PMID: 39499634) 使用 RCTD
#   (Cable et al. 2022 Nat Biotechnol, PMID: 33603203) 进行空间去卷积.
#   多项独立 benchmark 研究显示 RCTD 在 Visium 数据上表现为 top-performing:
#     - Sang-Aram 2024 eLife (PMID: 38787371): 11 方法中 cell2location/RCTD 最佳
#     - Li 2023 Nat Commun (PMID: 36941264): 18 方法中 CARD/Cell2location/Tangram 最佳
#     - Slabowska 2024 (PMID: 38601476): CVD 样本中 RCTD 准确性最高
#       (注: 仅比较 RCTD/Cell2location/spatialDWLS, 未含 SPOTlight)
#   本项目选 SPOTlight (PMID: 33544846) 仅为工程性理由:
#     1) Bioconductor 正式包, 与 SpatialExperiment/SingleCellExperiment 原生兼容
#     2) seeded-NMF 提供可解释 topic profile
#     3) 对 shallowly sequenced scRNA-seq 参考稳健 (PMID: 33544846)
#   承认此选择非方法学最优, 后续将补充 RCTD 复跑做敏感性分析.
#
# Marker 检测方法学说明 (2026-07-21 升级):
#   原: Seurat::FindAllMarkers (Wilcoxon rank-sum test, p_val_adj < 0.05)
#   新: scran::scoreMarkers (AUC + Cohen's d, mean.AUC > 0.8)
#   理由 (scran GitHub MarioniLab/scran R/scoreMarkers.R 源码 + OSCA.book):
#     1) p-value 在单细胞 marker 检测中 "largely meaningless" (Aaron Lun 原文),
#        因单个细胞不是实验重复单元, 且 cluster 本身由数据定义
#     2) AUC (area under ROC curve) 对分布形状鲁棒, 不假设正态, 不受异常值子群影响
#     3) Cohen's d 考虑表达变化幅度, AUC 完美分离 (1.0) 后无法区分好/极好 marker
#     4) OSCA.book 推荐使用 mean.AUC > 0.8 (非 min.AUC, 因过于严格)
#     5) scoreMarkers 同时返回 mean/min/median/max/rank, 供多角度筛选
#   保留 FindAllMarkers 作为 fallback (scran 不可用时)
# ============================================================================

step10_integration_spotlight <- function(sc_seu, spatial_merged, cfg) {
  log_info("[Step10-L4] SPOTlight spatial deconvolution...")

  require_packages(c("SPOTlight", "SingleCellExperiment"),
                   install_hint = paste("BiocManager::install(c('SPOTlight',",
                                        "'SingleCellExperiment', 'scran'))"))
  suppressPackageStartupMessages({
    library(Seurat)
    library(SPOTlight)
    library(SingleCellExperiment)
    library(ggplot2)
  })
  scran_available <- requireNamespace("scran", quietly = TRUE)
  if (scran_available) {
    suppressPackageStartupMessages(library(scran))
    log_info("[Step10] scran ", as.character(packageVersion("scran")),
             " available; using scoreMarkers (AUC>0.8) for marker detection")
  } else {
    log_warn("[Step10] scran not available; falling back to Seurat::FindAllMarkers",
             " (Wilcoxon). Install scran for improved marker detection: ",
             "BiocManager::install('scran')")
  }

  if (is.null(sc_seu) || is.null(spatial_merged)) {
    stop("step10: sc_seu or spatial_merged is NULL. Run step07/08 and step04-06 first.")
  }

  # 兼容性: 确保 spatial_merged 有 condition 列 (小写)
  # 旧版 06_spatial_with_regions.rds 可能只有 Condition (大写)
  if (!"condition" %in% colnames(spatial_merged@meta.data)) {
    if ("Condition" %in% colnames(spatial_merged@meta.data)) {
      spatial_merged$condition <- spatial_merged$Condition
      log_info("[Step10] Created 'condition' column from 'Condition' (case compatibility)")
    } else {
      stop("step10: spatial_merged has neither 'condition' nor 'Condition' column. ",
           "Run step04-06 first.")
    }
  }

  celltype_col <- cfg$data$sc_celltype_col
  if (!(celltype_col %in% colnames(sc_seu@meta.data))) {
    stop("step10: celltype column '", celltype_col, "' not in sc_seu meta.data")
  }

  # --------------------------------------------------------------------------
  # 10.1 准备单细胞 SCE 对象 (SPOTlight 要求 SCE)
  # --------------------------------------------------------------------------
  # 必须显式将 Idents 设为细胞类型列, 否则:
  #   - FindAllMarkers 会用默认 Idents (如 seurat_clusters, 18+ 聚类)
  #   - as.SingleCellExperiment 会把 Idents 写入 colData$label (非 Celltypes)
  # 导致 SPOTlight 用错误的分组。必须在 SCE 转换之前设置。
  orig_idents <- Idents(sc_seu)
  Idents(sc_seu) <- celltype_col
  n_idents <- length(unique(Idents(sc_seu)))
  log_info("[Step10] Idents set to '", celltype_col, "' (", n_idents, " levels)")
  if (n_idents < 2) {
    log_error("[Step10] Idents has < 2 levels; cannot FindAllMarkers. ",
              "Check celltype column: ", celltype_col)
    return(NULL)
  }

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
  # 优先使用 scran::scoreMarkers (AUC-based, OSCA.book 推荐)
  # 回退到 Seurat::FindAllMarkers (Wilcoxon, 仅 scran 不可用时)
  #
  # block= 参数决策 (2026-07-21 第二轮审查):
  #   scran::scoreMarkers 支持 block= 参数处理批次效应 (intra-batch 比较).
  #   本项目不使用 block=, 原因:
  #     1) sc_seu 的 "Sample" 列 (sn_D1/sn_D3/sn_D7/sn_sham) 是生物学条件
  #        (MCAO 后 1/3/7 天 vs 假手术), 不是技术性 batch.
  #     2) block= 会消除条件间真实生物学差异, 与研究目标 (铁衰老时序) 冲突.
  #     3) 稀有细胞类型 (Neuroblasts 51 cells, VLMCs 87 cells) 在某些条件下
  #        <10 cells, block= 会导致这些 cluster 对返回 NA (Aaron Lun:
  #        "cells from two clusters never co-occur in the same batch,
  #        the comparison will be impossible").
  #     4) Step 07 已用 Harmony 整合 (基于 Condition), 但 Harmony 校正的是
  #        PCA embeddings 用于聚类, logcounts 仍保留原始值供 DE 分析.
  #        这是 OSCA 推荐做法 (batch correction for clustering, not for DE).
  #   参考:
  #     - simpleSingleCell vignette Section 2.1 (block= 用途与限制)
  #     - OSCA.basic Chapter 6 (scoreMarkers 默认无 block)
  #     - libscran scran_markers::score_markers_summary_blocked (C++ API)
  top_n <- cfg$integration$spotlight_n_top_mgs
  mgs_top <- if (scran_available) {
    log_info("[Step10] Scoring cell-type markers (scran::scoreMarkers, AUC>0.8)...")
    # scoreMarkers 需要 logcounts + colLabels
    # sc_sce 已含 logcounts (上面已 as.matrix); colLabels 用 celltype_col
    colLabels(sc_sce) <- factor(colData(sc_sce)[[celltype_col]])

    marker_scores <- tryCatch(
      scoreMarkers(sc_sce, colLabels(sc_sce)),
      error = function(e) {
        log_error("[Step10] scoreMarkers failed: ", conditionMessage(e),
                  "; falling back to FindAllMarkers")
        NULL
      }
    )

    if (is.null(marker_scores)) {
      # scoreMarkers 失败, fallback 到 FindAllMarkers
      log_warn("[Step10] Falling back to FindAllMarkers (Wilcoxon)")
      all_markers <- FindAllMarkers(sc_seu, only.pos = TRUE,
                                     min.pct = 0.25, logfc.threshold = 0.25)
      all_markers <- all_markers[all_markers$p_val_adj < 0.05, ]
      do.call(rbind, lapply(split(all_markers, all_markers$cluster),
                              function(x) head(x[order(-x$avg_log2FC), ],
                                                n = top_n)))
    } else {
      # scoreMarkers 成功: 用 mean.AUC > 0.8 筛选 (OSCA.book 推荐阈值)
      # 输出格式: List of DataFrames, 每个 cluster 一个 DataFrame
      # 列: self.average, other.average, self.detected, other.detected,
      #     mean.AUC, min.AUC, median.AUC, max.AUC, rank.AUC,
      #     mean.logFC.cohen, ..., mean.logFC.detected, ...
      auc_threshold <- 0.8
      log_info("[Step10] scoreMarkers returned ", length(marker_scores),
               " clusters; filtering by mean.AUC > ", auc_threshold)

      mgs_list <- lapply(names(marker_scores), function(cl) {
        df <- as.data.frame(marker_scores[[cl]])
        # 筛选: mean.AUC > 0.8 表示基因在该 cluster 上调 (OSCA.book)
        df <- df[!is.na(df$mean.AUC) & df$mean.AUC > auc_threshold, ]
        if (nrow(df) == 0) {
          log_warn("[Step10] Cluster ", cl, ": 0 markers pass mean.AUC > ",
                   auc_threshold, "; taking top ", top_n, " by mean.AUC")
          df <- as.data.frame(marker_scores[[cl]])
          df <- df[order(df$mean.AUC, decreasing = TRUE), ]
        } else {
          # 按 mean.AUC 降序排序
          df <- df[order(df$mean.AUC, decreasing = TRUE), ]
        }
        # 取 top N
        df <- head(df, n = top_n)
        # 转换为 SPOTlight 兼容的格式:
        #   cluster, gene, avg_log2FC, pct.1, pct.2, p_val_adj
        data.frame(
          cluster     = cl,
          gene        = rownames(df),
          avg_log2FC  = df$mean.logFC.cohen,  # Cohen's d 作为 effect size
          pct.1       = df$self.detected,     # 该 cluster 中检测比例
          pct.2       = df$other.detected,    # 其他 cluster 中检测比例
          p_val_adj   = NA_real_,             # scoreMarkers 不返回 p-value
          mean.AUC    = df$mean.AUC,
          min.AUC     = df$min.AUC,
          median.AUC  = df$median.AUC,
          max.AUC     = df$max.AUC,
          rank.AUC    = df$rank.AUC,
          stringsAsFactors = FALSE
        )
      })
      do.call(rbind, mgs_list)
    }
  } else {
    log_info("[Step10] Finding cell-type markers (FindAllMarkers, Wilcoxon)...")
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

    all_markers <- all_markers[all_markers$p_val_adj < 0.05, ]
    do.call(rbind, lapply(split(all_markers, all_markers$cluster),
                            function(x) head(x[order(-x$avg_log2FC), ],
                                              n = top_n)))
  }
  log_info("[Step10] Top markers per cell type: ", nrow(mgs_top),
           " (n=", top_n, " per type, ", length(unique(mgs_top$cluster)), " types)")

  # 每 cluster 实际 marker 数报告 (用于诊断 marker 质量与 cluster 大小关系)
  # scran 路径下: marker 数 = min(passed_AUC>0.8, top_n); 不足 top_n 表示
  # 该 cluster 通过 AUC>0.8 的基因 < top_n (可能因 cluster 小或与其他 cluster 相似)
  cluster_marker_counts <- table(mgs_top$cluster)
  log_info("[Step10] Markers per cluster (actual vs requested top_n=", top_n, "):")
  for (cl in names(cluster_marker_counts)) {
    n_actual <- unname(cluster_marker_counts[cl])
    status <- if (n_actual == top_n) "FULL" else
      if (n_actual >= top_n / 2) "PARTIAL" else "LOW"
    log_info(sprintf("  %-25s %4d/%d  [%s]", cl, n_actual, top_n, status))
  }

  save_table(mgs_top, "10_spotlight_top_markers", cfg)

  # --------------------------------------------------------------------------
  # 10.3 对每个空间切片运行 SPOTlight 去卷积
  # --------------------------------------------------------------------------
  spatial_conds <- unique(spatial_merged$condition)
  spotlight_results <- list()

  for (cond in spatial_conds) {
    log_info("[Step10] Deconvolving spatial sample: ", cond)
    # Seurat v5: 用 cells= 传递索引避免 .data[[]] 解析问题
    keep_cond <- spatial_merged@meta.data$condition == cond
    sp_sub <- subset(spatial_merged, cells = colnames(spatial_merged)[keep_cond])
    if (ncol(sp_sub) == 0) next

    # SPOTlight 输入: 空间数据 assay 矩阵
    # 官方文档 (PMID:33544846) 推荐 raw counts 作为输入, SPOTlight 内部做 unit variance scaling
    # 参考: https://marcelosua.github.io/SPOTlight/reference/SPOTlight.html
    sp_assay <- "SCT" %in% names(sp_sub@assays)
    sp_mat <- if (sp_assay) {
      as.matrix(GetAssayData(sp_sub, assay = "SCT", layer = "counts"))
    } else {
      as.matrix(GetAssayData(sp_sub, assay = "Spatial", layer = "counts"))
    }

    # 共享基因
    shared_genes <- intersect(rownames(logcounts(sc_sce)), rownames(sp_mat))
    log_info("[Step10] Shared genes: ", length(shared_genes))
    sc_sce_shared <- sc_sce[shared_genes, ]
    sp_mat_shared <- sp_mat[shared_genes, ]

    # 确定分组列: 优先 celltype_col, 否则用 Seurat 转换写入的 label
    if (celltype_col %in% colnames(colData(sc_sce_shared))) {
      group_vec <- colData(sc_sce_shared)[[celltype_col]]
    } else if ("label" %in% colnames(colData(sc_sce_shared))) {
      group_vec <- colData(sc_sce_shared)[["label"]]
      log_warn("[Step10] ", celltype_col, " not in colData; using 'label' column.")
    } else {
      log_error("[Step10] No celltype column in colData for ", cond, "; skipping.")
      next
    }

    set.seed(cfg$reproducibility$r_seed)
    t_nmf_start <- Sys.time()
    spot_res <- tryCatch({
      SPOTlight(
        x = sc_sce_shared,
        y = sp_mat_shared,
        groups = group_vec,
        mgs = mgs_top[, c("cluster", "gene", "avg_log2FC",
                           "pct.1", "pct.2", "p_val_adj")],
        weight_id = "avg_log2FC",
        group_id = "cluster",
        gene_id = "gene",
        assay = "counts",
        min_prop = cfg$integration$spotlight_min_prop,
        tol = 1e-5,
        maxit = 200
      )
    }, error = function(e) {
      log_error("[Step10] SPOTlight failed for ", cond, ": ",
                conditionMessage(e))
      return(NULL)
    })

    if (is.null(spot_res)) next

    # NMF 收敛诊断 (官方 tol=1e-5 为 "publication quality")
    # SPOTlight() 返回 list(mat, res_ss, NMF) - 核实自 R/SPOTlight.R 源码
    nmf_mod <- spot_res$NMF
    nmf_elapsed <- round(as.numeric(difftime(Sys.time(), t_nmf_start, units = "secs")), 1)
    if (!is.null(nmf_mod)) {
      log_info(sprintf("[Step10] %s: NMF trained in %.1fs; residual SS median=%.4f, max=%.4f",
                       cond, nmf_elapsed,
                       median(spot_res$res_ss, na.rm = TRUE),
                       max(spot_res$res_ss, na.rm = TRUE)))
    } else {
      log_warn("[Step10] ", cond, ": NMF model is NULL (field name mismatch?). ",
               "Check SPOTlight version compatibility.")
    }

    # 提取细胞类型比例矩阵
    prop_mat <- spot_res$mat
    colnames(prop_mat) <- paste0("prop_", colnames(prop_mat))

    # 写入空间对象 meta.data
    common_cells <- intersect(rownames(prop_mat), colnames(sp_sub))
    for (ct in colnames(prop_mat)) {
      sp_sub@meta.data[[ct]] <- NA_real_
      sp_sub@meta.data[common_cells, ct] <- prop_mat[common_cells, ct]
      # 同步写入原始 spatial_merged (保留 Spatial/SCT/integrated assays + images)
      # 这样 Step 11 通过 RDS restore 加载 spatial_merged 时可直接使用
      spatial_merged@meta.data[[ct]] <- if (ct %in% colnames(spatial_merged@meta.data))
        spatial_merged@meta.data[[ct]] else NA_real_
      spatial_merged@meta.data[common_cells, ct] <- prop_mat[common_cells, ct]
    }

    spotlight_results[[cond]] <- list(
      prop_mat = prop_mat,
      nlm = spot_res$NMF
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
  # 10.4 保存含 prop_ 列的原始 spatial_merged
  # --------------------------------------------------------------------------
  # 关键: 直接在原始 spatial_merged 上添加 prop_ 列, 保留所有原始 assay
  # (Spatial/SCT/integrated) 和 images (VisiumV1/V2), 这样:
  #   1) Step 11 通过 RDS restore 加载此文件, 可直接使用 Spatial/SCT assay
  #      和 GetTissueCoordinates()
  #   2) 避免使用 merge() 触发 Seurat v5 在混合 assay/image 结构上的
  #      "no available method for coercing this S4 class to vector" 错误
  #   3) prop_ 列已在上面的循环中同步写入 spatial_merged@meta.data
  prop_in_merged <- grep("^prop_", colnames(spatial_merged@meta.data), value = TRUE)
  log_info("[Step10] spatial_merged now has ", length(prop_in_merged),
           " prop_ columns; total spots: ", ncol(spatial_merged))

  # 清理 SCTModels: 旧版 Seurat 创建的 SCTModel 类缺少 'median_umi' 插槽,
  # saveRDS 调用 containsOutOfMemoryData 时会报错。SCTModels 仅用于反向 SCT,
  # 下游分析 (CellChat) 不需要, 可安全清除 @SCTModel.list
  # 注意: SCTModel 对象存储在 SCTAssay@SCTModel.list (不是 @misc)
  for (assay_name in names(spatial_merged@assays)) {
    assay_obj <- spatial_merged@assays[[assay_name]]
    if (inherits(assay_obj, "SCTAssay") &&
        !is.null(assay_obj@SCTModel.list) &&
        length(assay_obj@SCTModel.list) > 0) {
      n_models <- length(assay_obj@SCTModel.list)
      log_info("[Step10] Dropping ", n_models, " SCTModel(s) from '",
               assay_name, "'@SCTModel.list to avoid saveRDS serialization ",
               "issue (missing 'median_umi' slot in old SCTModel class)")
      assay_obj@SCTModel.list <- list()
      spatial_merged@assays[[assay_name]] <- assay_obj
    }
  }

  save_rds(spatial_merged, "10_spatial_with_proportions", cfg)

  # 神经元比例 vs Ferroptosis 得分 (若 step05 已计算)
  # 兼容 Ferroptosis_UCell (新版 Step05) 和 Ferroptosis (旧版 Step05)
  fp_col <- if ("Ferroptosis_UCell" %in% colnames(spatial_merged@meta.data)) {
    "Ferroptosis_UCell"
  } else if ("Ferroptosis" %in% colnames(spatial_merged@meta.data)) {
    "Ferroptosis"
  } else {
    NA_character_
  }
  # 神经元比例列: SPOTlight 生成 prop_<celltype>, Celltypes 含 NeuronsGABA/NeuronsGLUT
  neuron_prop_cols <- grep("^prop_Neuron", colnames(spatial_merged@meta.data), value = TRUE)
  if (length(neuron_prop_cols) > 0 && !is.na(fp_col)) {
    # 若有多个神经元亚型 (NeuronsGABA, NeuronsGLUT), 合并为总神经元比例
    cor_df <- data.frame(
      neuron_prop = rowSums(spatial_merged@meta.data[, neuron_prop_cols, drop = FALSE]),
      fp_score = spatial_merged@meta.data[[fp_col]],
      condition = spatial_merged$condition,
      region = if ("region" %in% colnames(spatial_merged@meta.data))
        spatial_merged$region else "NA"
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
  if ("region" %in% colnames(spatial_merged@meta.data)) {
    prop_cols <- grep("^prop_", colnames(spatial_merged@meta.data), value = TRUE)
    region_df <- reshape2::melt(spatial_merged@meta.data[, c("region", "condition", prop_cols)],
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
    spatial_merged = spatial_merged,
    per_sample = spotlight_results
  ))
}
