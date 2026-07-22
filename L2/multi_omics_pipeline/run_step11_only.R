#!/usr/bin/env Rscript
# ============================================================================
# 临时脚本: 仅运行 Step 11 (绕过 14_metabolomics_ferroptosis.R 的 main() 自动执行问题)
# 用途: 验证 Step 11 全部修复 (subsetDB + liftCellChat + Bug #5/#6 + future 并行)
# ============================================================================
.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

script_dir <- getwd()
if (!file.exists(file.path(script_dir, "config.yaml"))) {
  script_dir <- "d:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
}
setwd(script_dir)

# 加载 utils
for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

# 仅加载 Step 11 所需模块 (跳过 14_metabolomics_ferroptosis.R 的自动 main() 调用)
modules_needed <- c("11_integration_cellchat_spatial.R")
for (f in file.path(script_dir, "R", modules_needed)) {
  if (file.exists(f)) {
    log_info("Loading module: ", basename(f))
    source(f)
  }
}

cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("run_step11_only_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")
log_info("=== Run Step 11 only (with all fixes) ===")

# 从 RDS 恢复 spatial_merged (含 prop_* 列, 来自 Step 10)
rds_path <- file.path(cfg$project$rds_dir, "10_spatial_with_proportions.rds")
if (!file.exists(rds_path)) {
  log_error("Missing: ", rds_path, " (run Step 10 first)")
  quit(status = 1)
}
log_info("[Restore] Loading 10_spatial_with_proportions.rds")
spatial_merged <- readRDS(rds_path)
log_info("[Restore] spatial_merged: ", ncol(spatial_merged), " spots, ",
         length(unique(spatial_merged$condition)), " conditions")

t0 <- Sys.time()
cellchat_spatial <- step11_integration_cellchat_spatial(spatial_merged, cfg)
elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2)
log_info(sprintf("=== Step 11 done in %.2f min ===", elapsed))
if (!is.null(cellchat_spatial)) {
  log_info("Step 11 returned CellChat object successfully")
} else {
  log_error("Step 11 returned NULL")
  quit(status = 1)
}
