#!/usr/bin/env Rscript
# ============================================================================
# 验证 Step 10 scran::scoreMarkers 升级
# 加载 sc_seu 真实数据, 仅运行 marker 检测部分, 不运行 SPOTlight 全流程
# ============================================================================
.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

suppressPackageStartupMessages({
  library(Seurat)
  library(SingleCellExperiment)
  library(scran)
  library(scuttle)
})

script_dir <- "d:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
setwd(script_dir)

for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("verify_step10_scran_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")
log_info("=== Verify Step 10 scran::scoreMarkers upgrade ===")

# 加载 sc_seu (来自 Step 07/08)
rds_path <- file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds")
if (!file.exists(rds_path)) {
  rds_path <- file.path(cfg$project$rds_dir, "07_sc_seurat_integrated.rds")
}
if (!file.exists(rds_path)) {
  log_error("Missing: ", rds_path, " (run Step 07/08 first)")
  quit(status = 1)
}
log_info("[Restore] Loading ", basename(rds_path))
sc_seu <- readRDS(rds_path)
log_info("[Restore] sc_seu: ", ncol(sc_seu), " cells, ",
         nrow(sc_seu), " genes")

celltype_col <- cfg$data$sc_celltype_col
log_info("Celltype column: ", celltype_col)
if (!(celltype_col %in% colnames(sc_seu@meta.data))) {
  log_error("Celltype column not found in sc_seu meta.data")
  quit(status = 1)
}

# 设置 Idents 为细胞类型列
Idents(sc_seu) <- celltype_col
log_info("Idents levels: ", paste(levels(Idents(sc_seu)), collapse = ", "))

# JoinLayers (Seurat v5)
if (inherits(sc_seu[["RNA"]], "Assay5")) {
  sc_seu[["RNA"]] <- JoinLayers(sc_seu[["RNA"]])
}

# 转 SingleCellExperiment
log_info("Converting sc Seurat -> SingleCellExperiment...")
sc_sce <- as.SingleCellExperiment(sc_seu, assay = "RNA")
logcounts(sc_sce) <- as.matrix(logcounts(sc_sce))
colLabels(sc_sce) <- factor(colData(sc_sce)[[celltype_col]])

# 运行 scoreMarkers
log_info("Running scran::scoreMarkers...")
t0 <- Sys.time()
marker_scores <- scoreMarkers(sc_sce, colLabels(sc_sce))
elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
log_info(sprintf("scoreMarkers done in %.1fs; returned %d clusters",
                 elapsed, length(marker_scores)))

# 输出每个 cluster 的 marker 统计
auc_threshold <- 0.8
log_info(sprintf("\nMarker summary (mean.AUC > %.2f):", auc_threshold))
log_info(sprintf("%-25s %10s %10s %10s %10s %10s",
                 "Cluster", "n_total", "n_AUC>0.8", "top_AUC", "top_d",
                 "top_detected"))

top_n <- cfg$integration$spotlight_n_top_mgs
mgs_list <- list()

for (cl in names(marker_scores)) {
  df <- as.data.frame(marker_scores[[cl]])
  n_total <- nrow(df)
  df_pass <- df[!is.na(df$mean.AUC) & df$mean.AUC > auc_threshold, ]
  n_pass <- nrow(df_pass)

  if (n_pass == 0) {
    log_warn("Cluster ", cl, ": 0 markers pass mean.AUC > ", auc_threshold)
    df_use <- df[order(df$mean.AUC, decreasing = TRUE), ]
  } else {
    df_use <- df_pass[order(df_pass$mean.AUC, decreasing = TRUE), ]
  }

  df_top <- head(df_use, n = top_n)
  top_auc <- if (nrow(df_top) > 0) round(max(df_top$mean.AUC), 3) else NA
  top_d <- if (nrow(df_top) > 0) round(max(df_top$mean.logFC.cohen), 3) else NA
  top_det <- if (nrow(df_top) > 0) round(max(df_top$self.detected), 3) else NA

  log_info(sprintf("%-25s %10d %10d %10s %10s %10s",
                   cl, n_total, n_pass,
                   top_auc, top_d, top_det))

  mgs_list[[cl]] <- data.frame(
    cluster     = cl,
    gene        = rownames(df_top),
    avg_log2FC  = df_top$mean.logFC.cohen,
    pct.1       = df_top$self.detected,
    pct.2       = df_top$other.detected,
    p_val_adj   = NA_real_,
    mean.AUC    = df_top$mean.AUC,
    min.AUC     = df_top$min.AUC,
    median.AUC  = df_top$median.AUC,
    max.AUC     = df_top$max.AUC,
    rank.AUC    = df_top$rank.AUC,
    stringsAsFactors = FALSE
  )
}

mgs_top <- do.call(rbind, mgs_list)
log_info("\nTotal top markers: ", nrow(mgs_top),
         " (across ", length(unique(mgs_top$cluster)), " clusters)")

# 保存 marker 表
save_table(mgs_top, "10_spotlight_top_markers_scran_verify", cfg)

# 与旧版 FindAllMarkers 对比 (可选, 仅检查 marker 数量)
log_info("\n[Optional] Comparing with FindAllMarkers...")
t0 <- Sys.time()
fam_markers <- tryCatch({
  FindAllMarkers(sc_seu, only.pos = TRUE,
                  min.pct = 0.25, logfc.threshold = 0.25,
                  test.use = "wilcox")
}, error = function(e) {
  log_warn("FindAllMarkers failed: ", conditionMessage(e))
  NULL
})
if (!is.null(fam_markers)) {
  elapsed_fam <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  fam_sig <- fam_markers[fam_markers$p_val_adj < 0.05, ]
  log_info(sprintf("FindAllMarkers done in %.1fs; %d markers (FDR<0.05)",
                   elapsed_fam, nrow(fam_sig)))

  fam_top <- do.call(rbind, lapply(split(fam_sig, fam_sig$cluster),
                                     function(x) head(x[order(-x$avg_log2FC), ],
                                                       n = top_n)))
  log_info("FindAllMarkers top: ", nrow(fam_top),
           " (across ", length(unique(fam_top$cluster)), " clusters)")

  # 比较: scran marker 与 FindAllMarkers marker 的重叠
  overlap_summary <- sapply(unique(mgs_top$cluster), function(cl) {
    scran_genes <- mgs_top$gene[mgs_top$cluster == cl]
    fam_genes <- fam_top$gene[fam_top$cluster == cl]
    overlap <- length(intersect(scran_genes, fam_genes))
    data.frame(
      cluster = cl,
      scran_n = length(scran_genes),
      fam_n   = length(fam_genes),
      overlap = overlap,
      scran_only = length(setdiff(scran_genes, fam_genes)),
      fam_only   = length(setdiff(fam_genes, scran_genes))
    )
  }, simplify = FALSE)
  overlap_df <- do.call(rbind, overlap_summary)
  log_info("\nMarker overlap (scran vs FindAllMarkers):")
  print(overlap_df)
}

log_info("\n[OK] scran::scoreMarkers verification complete")
