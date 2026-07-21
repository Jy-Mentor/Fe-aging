#!/usr/bin/env Rscript
# ============================================================================
# 独立运行 Augur 分析 (仅 step09 的 Augur 部分, 跳过 monocle3 拟时序)
# 修正: n_subsamples=50 (官方默认), 按时间点分层分析
# 运行: Rscript run_augur_only.R
# ============================================================================

# 设置 Rtools (源码包编译需要)
Sys.setenv(RTOOLS40_HOME = "C:/rtools45")

# libPaths: D:/R-library 优先 (Augur 安装在此)
.libPaths(c("D:/R-library", .libPaths()))
cat("[INIT] .libPaths():\n"); print(.libPaths())

# 加载 Seurat (GetAssayData/JoinLayers 等) + ggplot2 (theme_pub 依赖)
suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
})
cat("[INIT] Seurat version:", as.character(packageVersion("Seurat")), "\n")

# ----------------------------------------------------------------------------
# 加载 utils 和 step09 模块
# ----------------------------------------------------------------------------
script_dir <- getwd()
if (!dir.exists(file.path(script_dir, "utils"))) {
  script_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
}
cat("[INIT] script_dir:", script_dir, "\n")

# source utils
for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  cat("[INIT] sourcing", basename(f), "\n")
  source(f)
}

# source step09 (含 .run_augur)
cat("[INIT] sourcing 09_sc_pseudotime_augur.R\n")
source(file.path(script_dir, "R", "09_sc_pseudotime_augur.R"))

# ----------------------------------------------------------------------------
# 加载配置
# ----------------------------------------------------------------------------
cfg <- load_config(file.path(script_dir, "config.yaml"))
init_logger(file.path(cfg$project$log_dir,
                      sprintf("augur_run_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))),
            level = "INFO")
log_info("============================================================")
log_info("Augur 独立运行 (修正版: n_subsamples=50, 按时间点分层)")
log_info("R 版本: ", R.version.string)
log_info("Augur 版本: ", as.character(packageVersion("Augur")))
log_info("随机种子: ", cfg$reproducibility$r_seed)
log_info("============================================================")

set.seed(cfg$reproducibility$r_seed)

# ----------------------------------------------------------------------------
# 加载 step08 输出的 Seurat 对象 (含 Celltypes 注释 + UCell 评分)
# ----------------------------------------------------------------------------
seu_path <- file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds")
if (!file.exists(seu_path)) {
  stop("Seurat 对象不存在: ", seu_path)
}
log_info("[LOAD] 加载 Seurat 对象: ", seu_path,
         " (", round(file.size(seu_path) / 1e6, 1), " MB)")
t0 <- Sys.time()
seu <- readRDS(seu_path)
log_info("[LOAD] 加载完成. 耗时: ",
         round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), " s")
log_info("[LOAD] 细胞数: ", ncol(seu), "; 基因数: ", nrow(seu))

celltype_col <- cfg$data$sc_celltype_col
condition_col <- cfg$data$sc_condition_col
log_info("[LOAD] celltype_col=", celltype_col, "; condition_col=", condition_col)
log_info("[LOAD] 细胞类型: ", paste(unique(seu@meta.data[[celltype_col]]), collapse=","))
log_info("[LOAD] 条件: ", paste(unique(seu@meta.data[[condition_col]]), collapse=","))

# ----------------------------------------------------------------------------
# 仅运行 Augur (跳过 monocle3 拟时序)
# ----------------------------------------------------------------------------
log_info("============================================================")
log_info("[AUGUR] 开始运行 Augur (按时间点分层)")
log_info("============================================================")
t_augur <- Sys.time()
augur_res <- .run_augur(seu, cfg,
                         celltype_col = celltype_col,
                         condition_col = condition_col)
log_info("[AUGUR] 完成. 耗时: ",
         round(as.numeric(difftime(Sys.time(), t_augur, units = "mins")), 2), " mins")

if (!is.null(augur_res)) {
  log_info("[AUGUR] AUC 结果预览:")
  print(augur_res$AUC)
  save_rds(augur_res, "09_augur_result", cfg)
  log_info("[AUGUR] 已保存 09_augur_result.rds")
} else {
  log_error("[AUGUR] Augur 返回 NULL, 请检查日志")
}

log_info("============================================================")
log_info("Augur 独立运行完成")
log_info("============================================================")
