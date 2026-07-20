#!/usr/bin/env Rscript
# ============================================================================
# 验证 Step 11 多条件比较修复 (加载已保存的 9 个 CellChat RDS, 只跑比较部分)
# ============================================================================
.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

suppressPackageStartupMessages({
  library(CellChat)
  library(Seurat)
  library(ggplot2)
  library(patchwork)
  library(ComplexHeatmap)
})

source("utils/io_helpers.R")
source("utils/plot_helpers.R")
cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("verify_step11_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")

# ----------------------------------------------------------------------------
# 加载 9 个 condition 的 CellChat 对象
# ----------------------------------------------------------------------------
rds_dir <- cfg$project$rds_dir
cond_list <- c("D1", "D3", "D7", "sham", "D7b", "1DPI", "3DPI", "7DPI", "Ctrl")
cellchat_list <- list()
for (cond in cond_list) {
  rds_path <- file.path(rds_dir, paste0("11_cellchat_spatial_", cond, ".rds"))
  if (!file.exists(rds_path)) {
    log_warn("Missing: ", rds_path); next
  }
  log_info("Loading ", basename(rds_path))
  cellchat_list[[cond]] <- readRDS(rds_path)
}
log_info("Loaded ", length(cellchat_list), " CellChat objects")

# 先对每个 condition 计算 centrality scores (mergeCellChat 后无法计算)
# 与 Step 11 修复后代码一致: netAnalysis_computeCentrality 在每个 condition 循环中调用
# 槽位名核实: netAnalysis_signalingRole_heatmap 源码用 slot(object, "netP")$centr (非 centrality)
for (cond in names(cellchat_list)) {
  if (length(cellchat_list[[cond]]@netP$centr) == 0) {
    log_info("Computing centrality for ", cond, "...")
    tryCatch({
      cellchat_list[[cond]] <- netAnalysis_computeCentrality(cellchat_list[[cond]])
    }, error = function(e) {
      log_warn("centrality failed for ", cond, ": ", conditionMessage(e))
    })
  } else {
    log_info("centrality already computed for ", cond)
  }
}

# ----------------------------------------------------------------------------
# 多条件比较 (与 Step 11 修复后代码一致)
# ----------------------------------------------------------------------------
log_info("Merging across ", length(cellchat_list), " conditions...")
cc_merged <- mergeCellChat(cellchat_list, add.names = names(cellchat_list))
save_rds(cc_merged, "11_cellchat_spatial_merged", cfg)

cond_names <- names(cellchat_list)

# 1) 总互作数比较
log_info("compareInteractions count...")
p_compare_count <- compareInteractions(cc_merged, show.legend = FALSE,
                                        group = cond_names) +
  theme_pub(base_size = 10)
save_figure(p_compare_count, "11_cellchat_compare_count", cfg,
            width = 8, height = 5)

# 2) 互作强度比较
log_info("compareInteractions weight...")
p_compare_weight <- compareInteractions(cc_merged, show.legend = FALSE,
                                          measure = "weight",
                                          group = cond_names) +
  theme_pub(base_size = 10)
save_figure(p_compare_weight, "11_cellchat_compare_weight", cfg,
            width = 8, height = 5)

# 3) 差异互作
log_info("netVisual_diffInteraction...")
p_diff <- netVisual_diffInteraction(cc_merged, weight.scale = TRUE,
                                      measure = "count") +
  ggtitle("Differential interactions (count)")
save_figure(p_diff, "11_cellchat_diff_count", cfg, width = 9, height = 7)

# 4) 信息流排名
log_info("rankNet...")
p_rank <- rankNet(cc_merged, mode = "comparison",
                   stacked = TRUE, do.stat = TRUE) +
  theme_pub(base_size = 9) +
  theme(axis.text.y = element_text(size = 7))
save_figure(p_rank, "11_cellchat_pathway_rank", cfg, width = 9, height = 14)

# 5) 铁死亡/衰老相关通路 (与 Step 11 修复后代码一致)
fa_pathways <- c("SPP1", "TGFb", "CXCL", "CCL", "TNF", "IL6",
                  "GALECTIN", "MIF", "COMPLEMENT", "FLT3",
                  "GRN", "VISFATIN", "NRXN", "NCAM",
                  "NOTCH", "WNT", "BMP", "FGF", "VEGF",
                  "PDGF", "EGF", "IL1", "IL2",
                  "IL4", "IL10", "IL12", "IL16", "IL17")
available_pathways <- character(0)
for (cn in names(cc_merged@LR)) {
  pn <- unique(cc_merged@LR[[cn]]$LRsig$pathway_name)
  available_pathways <- union(available_pathways, pn)
}
fa_pathways_avail <- fa_pathways[fa_pathways %in% available_pathways]
log_info("Total pathways: ", length(available_pathways),
         "; Available of interest: ",
         paste(fa_pathways_avail, collapse = ", "))

if (length(fa_pathways_avail) > 0) {
  # netAnalysis_signalingRole_heatmap 只支持 single CellChat object
  # 对每个 condition 单独绘制, 然后组合成 multi-panel
  log_info("netAnalysis_signalingRole_heatmap per condition...")
  ht_list <- list()
  for (cond in names(cellchat_list)) {
    tryCatch({
      cc_single <- cellchat_list[[cond]]
      # 只保留该 condition 实际有的 pathway
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
      log_warn("heatmap failed for ", cond, ": ", conditionMessage(e))
    })
  }
  # 保存为单独文件 (ComplexHeatmap object 不能用 ggplot2::ggsave)
  if (length(ht_list) > 0) {
    png(file.path(cfg$project$figures_dir,
                   "11_cellchat_ferrosenescence_pathways_heatmap.png"),
        width = 14, height = 3 * length(ht_list), units = "in",
        res = cfg$viz$figure_dpi)
    for (i in seq_along(ht_list)) {
      draw(ht_list[[i]], column_title = names(ht_list)[i])
    }
    dev.off()
    log_info("Figure saved: 11_cellchat_ferrosenescence_pathways_heatmap.png")
  }

  # netVisual_aggregate 源码访问 object@LR$LRsig (single object 结构)
  # merged 对象的 @LR 是按 condition 分组的 list, 无法直接调用
  # 改为: 对每个 condition 的 single object 单独绘制 pathway circle plot
  log_info("netVisual_aggregate per condition (top pathways)...")
  # 只绘制前 6 个 pathway (避免 28 个 pathway × 9 condition = 252 张图)
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
        log_info("Figure saved: 11_cellchat_pathway_", pw, "_", cond, ".png")
      }, error = function(e) {
        log_debug("Pathway ", pw, " for ", cond, " failed: ", conditionMessage(e))
        try(dev.off(), silent = TRUE)
      })
    }
  }
}

# 7) Outgoing / Incoming 通讯模式
# CellChat v2: netAnalysis_signalingRole_scatter 源码访问 slot(object, "netP")$centr
# merged 对象的 netP 是 list, 无法直接调用 - 改为 per-condition scatter + patchwork 组合
log_info("netAnalysis_signalingRole_scatter per condition...")
scatter_list <- list()
for (cond in names(cellchat_list)) {
  tryCatch({
    p <- netAnalysis_signalingRole_scatter(
      cellchat_list[[cond]], slot.name = "netP") +
      ggtitle(cond) +
      theme_pub(base_size = 9)
    scatter_list[[cond]] <- p
  }, error = function(e) {
    log_warn("scatter failed for ", cond, ": ", conditionMessage(e))
  })
}
if (length(scatter_list) > 0) {
  p_combined <- wrap_plots(scatter_list, ncol = 3) +
    plot_annotation(title = "Outgoing vs Incoming interaction strength")
  save_figure(p_combined, "11_cellchat_outgoing_pattern", cfg,
              width = 14, height = 4 * ceiling(length(scatter_list) / 3))
}

save_rds(cc_merged, "11_cellchat_spatial_merged_final", cfg)
log_info("Verify done.")
