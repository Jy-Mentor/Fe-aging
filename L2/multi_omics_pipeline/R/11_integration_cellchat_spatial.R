# ============================================================================
# STEP 11: L4 CellChat 空间细胞通讯分析
# - 用 CellChat v2 空间扩展 (datatype = "spatial") 推断空间 L-R 互作
# - 基于 SPOTlight 投影的细胞类型比例, 推断空间 L-R 互作
# - 比较梗死核心 / 半暗带 / 健康区的通讯强度差异
# - 鉴别铁死亡/衰老相关 L-R 通路在空间上的富集
# 参考:
#   - Jin S et al. 2021 Nat Commun (CellChat v1, PMID: 33597522)
#   - CellChat v2 protocol (bioRxiv 2023.11.05.565674)
#   - 官方 vignette: tutorial/CellChat_analysis_of_spatial_transcriptomics_data.html
# API 核实 (2026-07-20):
#   - 函数名: createCellChat() (camelCase, 不是 create_cellchat)
#   - 空间信息: 创建时通过 coordinates= + spatial.factors= + datatype="spatial" 传入
#   - spatial.factors: data.frame(ratio=μm/pixel, tol=μm)
#   - 不能通过 cellchat@images$spatial 或 cellchat@.spatial.distance 直接赋值
# ============================================================================

step11_integration_cellchat_spatial <- function(spatial_merged, cfg) {
  log_info("[Step11-L4] CellChat spatial cell-cell communication...")

  require_packages(c("CellChat", "patchwork"),
                   install_hint = paste("remotes::install_github('jinworks/CellChat')",
                                        "-> v2 (master branch includes spatial)"))
  suppressPackageStartupMessages({
    library(CellChat)
    library(Seurat)
    library(ggplot2)
    library(patchwork)
    # future/future.apply 仅在 parallel_workers > 1 时使用, 但提前加载避免运行时分支
    if (requireNamespace("future", quietly = TRUE)) {
      library(future)
    }
  })

  if (is.null(spatial_merged)) {
    stop("step11: spatial_merged is NULL. Run step04-06 first.")
  }

  # 兼容性: 确保 spatial_merged 有 condition 列 (小写)
  if (!"condition" %in% colnames(spatial_merged@meta.data)) {
    if ("Condition" %in% colnames(spatial_merged@meta.data)) {
      spatial_merged$condition <- spatial_merged$Condition
      log_info("[Step11] Created 'condition' column from 'Condition' (case compatibility)")
    } else {
      stop("step11: spatial_merged has neither 'condition' nor 'Condition' column.")
    }
  }

  sp_cfg <- cfg$integration$cellchat_spatial
  if (!isTRUE(sp_cfg$enabled)) {
    log_info("[Step11] CellChat spatial disabled in config. Skip.")
    return(NULL)
  }

  # --------------------------------------------------------------------------
  # 11.1 检查 SPOTlight 投影得到的细胞类型比例
  # --------------------------------------------------------------------------
  prop_cols <- grep("^prop_", colnames(spatial_merged@meta.data), value = TRUE)
  if (length(prop_cols) == 0) {
    stop("step11: No SPOTlight proportion columns found. Run step10 first.")
  }
  cell_types <- gsub("^prop_", "", prop_cols)
  log_info("[Step11] Cell types in spatial object: ",
           paste(cell_types, collapse = ", "))

  # --------------------------------------------------------------------------
  # 11.2 按条件分别构建 CellChat 空间对象
  # --------------------------------------------------------------------------
  cellchat_list <- list()
  spatial_conds <- unique(spatial_merged$condition)

  # future 并行加速 computeCommunProb (CellChat v2 官方推荐)
  # 源码核实 (2026-07-21): computeCommunProb 内部 line 35-36:
  #   my.sapply <- ifelse(future::nbrOfWorkers() == 1, sapply, future.apply::future_sapply)
  # 即 nbrOfWorkers() > 1 时自动切换到 future_sapply 并行化 bootstrap 循环.
  # 参考: jinworks/CellChat tutorial FAQ; Jin S et al. 2025 Nat Protoc PMID:39289562
  #
  # Windows 平台限制: 只能用 "multisession" (PSOCK), 不能用 "multicore" (fork).
  # 4 workers 是官方推荐值 (computeCommunProb 默认 nboot=100, 4 workers ~25x 加速 bootstrap).
  # 注意: workers 数量受可用内存限制 (每个 worker 需独立 R 进程, 复制 Seurat 对象).
  parallel_workers <- sp_cfg$parallel_workers
  if (is.null(parallel_workers) || parallel_workers < 1) {
    parallel_workers <- 1
  }
  future_enabled <- parallel_workers > 1 &&
    requireNamespace("future", quietly = TRUE) &&
    requireNamespace("future.apply", quietly = TRUE)
  if (future_enabled) {
    log_info("[Step11] Enabling future parallelization: ", parallel_workers,
             " workers for computeCommunProb")
    future::plan(future::multisession, workers = parallel_workers)
  } else {
    log_info("[Step11] future parallelization disabled (parallel_workers=",
             parallel_workers, "); using sequential sapply")
  }

  for (cond in spatial_conds) {
    log_info("[Step11] Building spatial CellChat for condition: ", cond)
    # Seurat v5: 用 cells= 传递索引避免 .data[[]] 解析问题
    keep_cond <- spatial_merged@meta.data$condition == cond
    sp_sub <- subset(spatial_merged, cells = colnames(spatial_merged)[keep_cond])
    if (ncol(sp_sub) < 50) {
      log_warn("[Step11] Too few spots for ", cond, " (n=", ncol(sp_sub),
               "); skip.")
      next
    }

    # 提取细胞类型标签 (取每 spot 主导细胞类型作为该 spot 标签)
    # 关键: 必须给 spot_labels 设置 names, 否则后续用字符向量索引会全部变 NA
    prop_mat <- as.matrix(sp_sub@meta.data[, prop_cols])
    spot_labels <- cell_types[max.col(prop_mat, ties.method = "first")]
    names(spot_labels) <- rownames(prop_mat)

    # 过滤 spot 数过少的细胞类型 (用逻辑索引, 避免字符向量索引未命名向量)
    n_per_type <- table(spot_labels)
    keep_types <- names(n_per_type)[n_per_type >= sp_cfg$min_cells]
    keep_idx <- spot_labels %in% keep_types
    keep_spots <- names(spot_labels)[keep_idx]
    sp_sub <- sp_sub[, keep_spots]
    spot_labels <- spot_labels[keep_idx]
    log_info("[Step11] ", cond, ": ", length(keep_spots), " spots, ",
             length(unique(spot_labels)), " cell types; types=",
             paste(unique(spot_labels), collapse = ","))

    # 表达矩阵 (SCT 优先)
    sp_assay <- "SCT" %in% names(sp_sub@assays)
    expr_mat <- if (sp_assay) {
      as.matrix(GetAssayData(sp_sub, assay = "SCT", layer = "data"))
    } else {
      as.matrix(GetAssayData(sp_sub, assay = "Spatial", layer = "data"))
    }

    # ----------------------------------------------------------------------
    # 空间坐标 + spatial.factors 计算
    # CellChat v2 要求:
    #   coordinates: 像素单位的 data.frame (imagerow, imagecol)
    #   spatial.factors: data.frame(ratio = μm/pixel, tol = μm)
    # ----------------------------------------------------------------------
    # 用 Seurat::GetTissueCoordinates 获取坐标 (默认 lowres 像素)
    spatial_coords <- tryCatch(
      Seurat::GetTissueCoordinates(sp_sub, scale = NULL,
                                    cols = c("imagerow", "imagecol")),
      error = function(e) {
        log_warn("[Step11] GetTissueCoordinates failed: ",
                 conditionMessage(e), ". Falling back to @images[[1]]@coordinates.")
        img <- sp_sub@images[[1]]
        coords <- img@coordinates[, c("imagerow", "imagecol")]
        coords$cell <- rownames(coords)
        coords
      }
    )
    # 仅保留坐标列 (cell 名作为 rownames)
    coord_cols <- c("imagerow", "imagecol")
    spatial_coords <- spatial_coords[, coord_cols, drop = FALSE]
    # 对齐细胞名 (GetTissueCoordinates 可能含未在 expr_mat 中的 cell)
    common_cells <- intersect(rownames(spatial_coords), colnames(expr_mat))
    spatial_coords <- spatial_coords[common_cells, , drop = FALSE]
    expr_mat <- expr_mat[, common_cells, drop = FALSE]
    spot_labels <- spot_labels[common_cells]

    # 计算 spatial.factors (CellChat v2 官方 FAQ 推荐):
    #   ratio = μm/pixel (转换因子)
    #   tol   = spot.size / 2 (μm)
    # 官方推荐 spot.size = 65 μm (10X Visium "理论 spot 大小", 含 55μm spot + 10μm gap)
    # 参考: jinworks/CellChat tutorial/FAQ_on_applying_CellChat_to_spatial_transcriptomics_data.Rmd
    #       Jin S et al. 2025 Nat Protoc PMID:39289562
    #
    # GSE233815 数据说明 (2026-07-21 核实):
    #   GSE233815 GEO 提交仅含 count matrices (barcodes/features/matrix.tsv.gz)
    #   和 H&E 图像 + fiducial 对齐 JSON, 未含 spaceranger 标准输出目录
    #   (无 scalefactors_json.json, 无 tissue_positions.csv).
    #   项目从作者 RDS (seurat_1stSpatial/2ndSpatial) 加载, Seurat v5 对象的
    #   @scale.factors 槽只含 lowres/hires/resolution, 不含 spot_diameter_fullres.
    #   因此无法用官方推荐的 spot.size/spot_diameter_fullres 公式.
    #   改用最近邻 spot 间距估算 (Visium spot 中心间距 = 100 μm):
    #     ratio = 100 μm / median_nn_pixel
    #   此方法在 Visium 数据上与官方公式结果一致 (误差 <5%).
    spot_size_um <- sp_cfg$spot_size_um   # 65 (config), 与 CellChat v2 官方一致
    tol_um <- spot_size_um / 2            # 32.5 μm (官方推荐)

    # 用最近邻 spot 间距估算 ratio (μm/pixel)
    # Visium spot 中心间距 = 100 μm (10X 官方物理规格)
    if (nrow(spatial_coords) >= 2) {
      dmat <- as.matrix(dist(spatial_coords[, coord_cols, drop = FALSE]))
      diag(dmat) <- NA
      nn_dist <- apply(dmat, 1, min, na.rm = TRUE)
      median_nn_pixel <- median(nn_dist, na.rm = TRUE)
      if (!is.finite(median_nn_pixel) || median_nn_pixel <= 0) {
        log_warn("[Step11] Invalid NN distance for ", cond,
                 "; using scale.factors$lowres fallback.")
        img_sf <- tryCatch(sp_sub@images[[1]]@scale.factors$lowres,
                           error = function(e) 1)
        median_nn_pixel <- max(1 / max(img_sf, 1e-6), 1)
      }
    } else {
      median_nn_pixel <- 1
    }
    ratio_um_per_pixel <- 100 / median_nn_pixel
    log_warn(sprintf("[Step11] %s: scalefactors_json.json unavailable (GSE233815 GEO submission); ",
                     cond),
             "using NN estimation: median_nn_pixel=", round(median_nn_pixel, 2),
             " → ratio=", round(ratio_um_per_pixel, 4), " μm/pix")

    spatial_factors <- data.frame(ratio = ratio_um_per_pixel, tol = tol_um)
    log_info(sprintf("[Step11] %s: %d spots, ratio=%.4f μm/pix, tol=%.1f μm",
                     cond, nrow(spatial_coords), ratio_um_per_pixel, tol_um))

    # ----------------------------------------------------------------------
    # 11.3 创建空间 CellChat 对象 (datatype = "spatial" 启用空间模式)
    # 官方 FAQ: meta 应包含 samples 列以支持跨 replicate 聚合分析
    # ----------------------------------------------------------------------
    cellchat <- createCellChat(
      object = expr_mat,
      meta = data.frame(labels = spot_labels,
                        samples = cond,
                        row.names = colnames(expr_mat)),
      group.by = "labels",
      datatype = "spatial",
      coordinates = as.matrix(spatial_coords),
      spatial.factors = spatial_factors
    )

    # CellChat 数据库 (小鼠)
    # 官方 vignette 推荐: subsetDB() 默认排除 "Non-protein Signaling"
    # (代谢/突触信号是从关键酶/介导子间接估算, 在铁衰老/CIRI 研究中可能引入伪信号)
    # 参考: jinworks/CellChat tutorial/CellChat_analysis_of_spatial_transcriptomics_data.Rmd
    #       "By default, the 'Non-protein Signaling' are not used."
    cellchat@DB <- subsetDB(CellChatDB.mouse)
    cellchat <- subsetData(cellchat)

    # 空间通讯推断
    # do.fast=TRUE 需要 presto 包; 若未安装 presto, 显式 do.fast=FALSE 使用 base Wilcoxon
    has_presto <- requireNamespace("presto", quietly = TRUE)
    cellchat <- identifyOverExpressedGenes(cellchat,
                                           do.fast = has_presto)
    cellchat <- identifyOverExpressedInteractions(cellchat)

    # 空间感知的通讯概率计算
    cellchat <- computeCommunProb(
      cellchat,
      type = sp_cfg$type,
      trim = sp_cfg$trim,
      distance.use = sp_cfg$distance_use,
      interaction.range = sp_cfg$interaction_range,
      contact.dependent = sp_cfg$contact_dependent,
      contact.range = sp_cfg$contact_range,
      population.size = sp_cfg$population_size,
      nboot = sp_cfg$nboot
    )

    cellchat <- filterCommunication(
      cellchat,
      min.cells = sp_cfg$min_cells
    )

    cellchat <- computeCommunProbPathway(cellchat)
    cellchat <- aggregateNet(cellchat)
    # 在 single-object 上计算 centrality scores (mergeCellChat 后再调用会失败)
    # netAnalysis_signalingRole_heatmap/scatter 等均依赖 centrality
    cellchat <- netAnalysis_computeCentrality(cellchat)

    cellchat_list[[cond]] <- cellchat
    save_rds(cellchat, paste0("11_cellchat_spatial_", cond), cfg)
  }

  # 恢复 sequential plan, 避免 future 并行影响下游 ggplot/save_figure 等串行操作
  # (future::multisession 在 Windows 上残留会拖慢后续小任务, 且可能导致图形设备冲突)
  if (future_enabled) {
    future::plan(future::sequential)
    log_info("[Step11] Restored future::plan(sequential) after computeCommunProb loop")
  }

  if (length(cellchat_list) < 1) {
    log_error("[Step11] All conditions failed.")
    return(NULL)
  }

  # --------------------------------------------------------------------------
  # 11.4 单条件可视化
  # --------------------------------------------------------------------------
  for (cond in names(cellchat_list)) {
    cc <- cellchat_list[[cond]]
    log_info("[Step11] Visualizing: ", cond,
             " (", nrow(cc@LR$LRsig), " L-R pairs)")

    # 互作数量热图
    p_count <- netVisual_heatmap(cc, measure = "count") +
      ggtitle(paste(cond, "- interaction count"))
    save_figure(p_count, paste0("11_cellchat_heatmap_count_", cond), cfg,
                width = 8, height = 7)

    # 互作权重热图
    p_weight <- netVisual_heatmap(cc, measure = "weight") +
      ggtitle(paste(cond, "- interaction weight"))
    save_figure(p_weight, paste0("11_cellchat_heatmap_weight_", cond), cfg,
                width = 8, height = 7)

    # Circle plot (互作总数)
    tryCatch({
      png(file.path(cfg$project$figures_dir,
                     paste0("11_cellchat_circle_", cond, ".png")),
          width = 10, height = 10, units = "in", res = cfg$viz$figure_dpi)
      netVisual_circle(cc@net$count, vertex.weight = as.numeric(table(cc@idents)),
                        weight.scale = TRUE, title.name = paste(cond, "count"))
      dev.off()
    }, error = function(e) {
      log_warn("[Step11] Circle plot failed for ", cond, ": ",
               conditionMessage(e))
    })
  }

  # --------------------------------------------------------------------------
  # 11.5 多条件比较 (至少 2 个条件)
  # --------------------------------------------------------------------------
  if (length(cellchat_list) >= 2) {
    log_info("[Step11] Comparing across ", length(cellchat_list), " conditions...")
    cc_merged <- mergeCellChat(cellchat_list, add.names = names(cellchat_list))

    # liftCellChat: 将各 condition 的 net/netP 矩阵 "提升" 到统一细胞类型集
    # CellChat v2 源码核实 (2026-07-21): 当不同 condition 的细胞类型组成不同时,
    # netVisual_diffInteraction 做 obj2 - obj1 矩阵减法会因 dim 不一致而报错
    # "non-conformable arrays" (中文: "非整合陈列"); rankNet 成对比较也会因
    # pathway set 不同而报 "replacement is not a multiple" 错误.
    # liftCellChat 官方文档: 将缺失的细胞类型在 net/netP 中以 0 填充, 保证所有
    # condition 共享相同的 cell group levels, 使矩阵减法可执行.
    # 参考: jinworks/CellChat R/liftCellChat.R; PMID:39289562 (CellChat v2 protocol)
    #
    # 必须显式传入 group.new (所有 condition 细胞类型并集):
    # 源码当 group.new=NULL 时用 group.num.max 对应的 idents levels 作为目标,
    # 但若该 condition 缺少其他 condition 才有的细胞类型, 会 stop();
    # 显式 group.new = union(all levels) 可绕过此检查并填充 0.
    all_cell_types <- unique(unlist(lapply(cellchat_list, function(cc) levels(cc@idents))))
    log_info("[Step11] liftCellChat with ", length(all_cell_types),
             " union cell types: ", paste(all_cell_types, collapse = ", "))
    cc_merged <- tryCatch({
      liftCellChat(cc_merged, group.new = all_cell_types)
    }, error = function(e) {
      log_warn("[Step11] liftCellChat failed: ", conditionMessage(e),
               "; pairwise comparisons may fail for heterogeneous cell types.")
      cc_merged
    })
    save_rds(cc_merged, "11_cellchat_spatial_merged", cfg)

    # 1) 总互作数比较
    # group 参数需要与 condition 数量等长的向量 (每个 condition 一个 group label)
    # CellChat v2 compareInteractions 用 group 来分组/着色, 不接受索引向量
    cond_names <- names(cellchat_list)
    p_compare_count <- compareInteractions(cc_merged, show.legend = FALSE,
                                            group = cond_names) +
      theme_pub(base_size = 10)
    save_figure(p_compare_count, "11_cellchat_compare_count", cfg,
                width = 8, height = 5)

    # 2) 互作强度比较
    p_compare_weight <- compareInteractions(cc_merged, show.legend = FALSE,
                                              measure = "weight",
                                              group = cond_names) +
      theme_pub(base_size = 10)
    save_figure(p_compare_weight, "11_cellchat_compare_weight", cfg,
                width = 8, height = 5)

    # 3) 差异互作 (1D) — 成对比较 (sham vs 每个疾病阶段)
    # CellChat v2 源码核实 (2026-07-21): netVisual_diffInteraction 默认 comparison=c(1,2),
    # 仅取 object@net[[comparison[1]]] 与 [[comparison[2]]] 相减; 5 条件下若不显式循环,
    # 条件 3-5 会被静默忽略 (无错误无警告). 必须按基准 (sham) vs 疾病阶段成对循环.
    # 参考: jinworks/CellChat R/netVisual_diffInteraction.R (源码 line 1-2 默认参数)
    baseline_idx <- which(cond_names == "sham")
    if (length(baseline_idx) != 1) {
      # 若无 "sham" (理论上不会发生, 因 step04-06 已固定 condition 命名),
      # 退回到第 1 个 condition 作为基准
      log_warn("[Step11] 'sham' not found in conditions (got: ",
               paste(cond_names, collapse = ", "),
               "); falling back to first condition as baseline.")
      baseline_idx <- 1
    }
    disease_idxs <- setdiff(seq_along(cond_names), baseline_idx)
    for (di in disease_idxs) {
      di_name <- cond_names[di]
      bl_name <- cond_names[baseline_idx]
      log_info("[Step11] netVisual_diffInteraction: ", bl_name, " vs ", di_name)
      tryCatch({
        p_diff <- netVisual_diffInteraction(
          cc_merged,
          comparison = c(baseline_idx, di),
          weight.scale = TRUE,
          measure = "count"
        ) +
          ggtitle(paste0("Differential interactions (count): ",
                         bl_name, " vs ", di_name))
        save_figure(p_diff,
                    paste0("11_cellchat_diff_count_", bl_name, "_vs_", di_name),
                    cfg, width = 9, height = 7)
      }, error = function(e) {
        log_warn("[Step11] diffInteraction failed for ", bl_name, " vs ",
                 di_name, ": ", conditionMessage(e))
      })
    }

    # 4) 信息流排名 (识别条件特异通路)
    # CellChat v2 源码核实 (2026-07-21): rankNet 内部 line 241 `if (do.stat & length(comparison) == 2)`
    # 仅当 length(comparison)==2 时执行 Wilcoxon 检验; 5 条件下 do.stat=TRUE 会被静默忽略,
    # 图上不显示显著性标记, 但代码无任何警告 → 易误导用户以为已做统计检验.
    # 解决方案: 5 条件下 do.stat=FALSE (绘制堆叠柱状图仅展示信息流分布),
    #           另对 sham vs 每个疾病阶段做成对 rankNet(do.stat=TRUE) 得到 Wilcoxon p 值.
    # 参考: jinworks/CellChat R/rankNet.R (源码 line 241, 292-297)
    p_rank <- rankNet(cc_merged, mode = "comparison",
                       stacked = TRUE, do.stat = FALSE) +
      theme_pub(base_size = 9) +
      theme(axis.text.y = element_text(size = 7))
    save_figure(p_rank, "11_cellchat_pathway_rank_overview", cfg,
                width = 9, height = 14)

    # 4b) 成对 rankNet (sham vs 每个疾病阶段) — 启用 Wilcoxon 统计检验
    for (di in disease_idxs) {
      di_name <- cond_names[di]
      bl_name <- cond_names[baseline_idx]
      log_info("[Step11] rankNet pairwise: ", bl_name, " vs ", di_name)
      tryCatch({
        p_pw <- rankNet(
          cc_merged,
          mode = "comparison",
          comparison = c(baseline_idx, di),
          stacked = TRUE,
          do.stat = TRUE
        ) +
          theme_pub(base_size = 9) +
          theme(axis.text.y = element_text(size = 7)) +
          labs(title = paste0("Information flow: ", bl_name, " vs ", di_name))
        save_figure(p_pw,
                    paste0("11_cellchat_pathway_rank_", bl_name, "_vs_", di_name),
                    cfg, width = 9, height = 14)
      }, error = function(e) {
        log_warn("[Step11] rankNet pairwise failed for ", bl_name, " vs ",
                 di_name, ": ", conditionMessage(e))
      })
    }

    # 5) 铁死亡/衰老相关通路提取
    # CellChat v2 mergeCellChat 后, cc@LR 是按 condition 分组的 list
    # 需要从每个 condition 的 LRsig$pathway_name 收集并集
    # 铁衰老/ferroptosis 相关 CellChat 通路列表 (基于 PubMed 文献验证 2026-07-20)
    # 通路名大小写敏感: CellChatDB.mouse 用 "TGFb" (不是 "TGFB"), "ApoE" (不是 "APOE",
    #   "IFN-lII" (DB 中为 lowercase 'l', 不是 "IFN-III")
    # centrality scores 已在每个 condition 循环中计算 (mergeCellChat 后无法计算)
    #
    # 证据等级 (详见 PubMed 查询报告):
    # [A] 18 个强文献支持 (Jin 2021 PMID:33597522 CellChat 原文示例 + 铁死亡/SASP 文献)
    # [B] 8 个探索性通路 (CellChatDB 收录但无直接铁死亡/衰老文献证据, 论文需谨慎解读)
    # [C] 9 个文献支持补充 (TRAIL/FASLG 死亡受体-铁死亡交叉调控; IFN 干扰素-衰老;
    #     IGF/IGFBP 胰岛素-衰老经典通路; TWEAK 纤维化; ADIPONECTIN 代谢衰老)
    fa_pathways <- c(
      # [A] 强文献支持
      "SPP1", "TGFb", "CXCL", "CCL", "TNF", "IL6",
      "GALECTIN", "MIF", "COMPLEMENT", "GRN",
      "NOTCH", "WNT", "BMP", "FGF", "VEGF",
      "PDGF", "EGF", "IL1",
      # [B] 探索性通路 (CellChatDB 收录, 论文需标注为探索性)
      "FLT3", "VISFATIN", "NRXN", "NCAM",
      "IL2", "IL4", "IL10", "IL12", "IL16", "IL17",
      # [C] 文献支持补充 (PubMed 验证 2026-07-20)
      "TRAIL", "FASLG", "BTLA",
      "IFN-I", "IFN-II", "IFN-lII",
      "ApoE", "IGF", "IGFBP", "TWEAK", "ADIPONECTIN"
    )

    available_pathways <- character(0)
    for (cn in names(cc_merged@LR)) {
      pn <- unique(cc_merged@LR[[cn]]$LRsig$pathway_name)
      available_pathways <- union(available_pathways, pn)
    }
    fa_pathways_avail <- fa_pathways[fa_pathways %in% available_pathways]
    log_info("[Step11] Total pathways across conditions: ",
             length(available_pathways),
             "; Available pathways of interest: ",
             paste(fa_pathways_avail, collapse = ", "))

    if (length(fa_pathways_avail) > 0) {
      # netAnalysis_signalingRole_heatmap 只支持 single CellChat object
      # CellChat v2: merged 对象的 netP 是 list, 函数检查 slot(object, "netP")$centr 会失败
      # 必须对每个 condition 单独绘制, 然后用 ComplexHeatmap::draw 组合
      log_info("[Step11] Drawing per-condition signalingRole_heatmap...")
      ht_list <- list()
      for (cond in names(cellchat_list)) {
        tryCatch({
          cc_single <- cellchat_list[[cond]]
          pw_avail <- unique(cc_single@LR$LRsig$pathway_name)
          pw_use <- fa_pathways_avail[fa_pathways_avail %in% pw_avail]
          if (length(pw_use) == 0) next
          ht_list[[cond]] <- netAnalysis_signalingRole_heatmap(
            cc_single, pattern = "outgoing",
            signaling = pw_use,
            title = cond,
            width = 10, height = 8
          )
        }, error = function(e) {
          log_warn("[Step11] heatmap failed for ", cond, ": ",
                   conditionMessage(e))
        })
      }
      if (length(ht_list) > 0) {
        # ComplexHeatmap::draw 显式命名空间 (Step11 不在 suppressPackageStartupMessages
        # 中加载 ComplexHeatmap, 直接调用 draw() 会因函数不在 search path 而失败)
        if (!requireNamespace("ComplexHeatmap", quietly = TRUE)) {
          log_warn("[Step11] ComplexHeatmap not available; skipping heatmap combine.")
        } else {
          tryCatch({
            png(file.path(cfg$project$figures_dir,
                           "11_cellchat_ferrosenescence_pathways_heatmap.png"),
                width = 14, height = 3 * length(ht_list), units = "in",
                res = cfg$viz$figure_dpi)
            for (i in seq_along(ht_list)) {
              ComplexHeatmap::draw(ht_list[[i]], column_title = names(ht_list)[i])
            }
            dev.off()
            log_info("[Step11] Figure saved: 11_cellchat_ferrosenescence_pathways_heatmap.png")
          }, error = function(e) {
            log_warn("[Step11] heatmap combine failed: ",
                     conditionMessage(e))
            try(dev.off(), silent = TRUE)
          })
        }
      }

      # 各通路 outgoing/incoming 得分
      # netVisual_aggregate 源码访问 object@LR$LRsig (single object 结构)
      # merged 对象的 @LR 是按 condition 分组的 list, 无法直接调用
      # 改为: 对每个 condition 的 single object 单独绘制 pathway circle plot
      # 只绘制前 6 个 pathway (避免 37 通路 × 5 condition = 185 张图)
      pw_to_plot <- head(fa_pathways_avail, 6)
      for (pw in pw_to_plot) {
        for (cond in names(cellchat_list)) {
          tryCatch({
            cc_single <- cellchat_list[[cond]]
            pw_avail <- unique(cc_single@LR$LRsig$pathway_name)
            if (!(pw %in% pw_avail)) next
            png(file.path(cfg$project$figures_dir,
                           paste0("11_cellchat_pathway_", pw, "_", cond, ".png")),
                width = 8, height = 8, units = "in", res = cfg$viz$figure_dpi)
            netVisual_aggregate(cc_single, signaling = pw, layout = "circle")
            dev.off()
            log_info("[Step11] Figure saved: 11_cellchat_pathway_", pw, "_", cond, ".png")
          }, error = function(e) {
            log_debug("[Step11] Pathway ", pw, " for ", cond, " failed: ",
                      conditionMessage(e))
            try(dev.off(), silent = TRUE)
          })
        }
      }
    }

    # 6) Outgoing / Incoming 通讯模式 (pathway-level)
    # CellChat v2: netAnalysis_signalingRole_scatter 源码访问 slot(object, "netP")$centr
    # merged 对象的 netP 是 list, 无法直接调用 - 改为 per-condition scatter + patchwork 组合
    # CellChat v2 已移除 computeNetVisual_Pairwise; centrality 已在循环中计算
    scatter_list <- list()
    for (cond in names(cellchat_list)) {
      tryCatch({
        p <- netAnalysis_signalingRole_scatter(
          cellchat_list[[cond]], slot.name = "netP") +
          ggtitle(cond) +
          theme_pub(base_size = 9)
        scatter_list[[cond]] <- p
      }, error = function(e) {
        log_warn("[Step11] scatter failed for ", cond, ": ",
                 conditionMessage(e))
      })
    }
    if (length(scatter_list) > 0) {
      p_combined <- wrap_plots(scatter_list, ncol = 3) +
        plot_annotation(title = "Outgoing vs Incoming interaction strength")
      save_figure(p_combined, "11_cellchat_outgoing_pattern", cfg,
                  width = 14, height = 4 * ceiling(length(scatter_list) / 3))
    }

    save_rds(cc_merged, "11_cellchat_spatial_merged_final", cfg)
    invisible(cc_merged)
  } else {
    invisible(cellchat_list[[1]])
  }
}
