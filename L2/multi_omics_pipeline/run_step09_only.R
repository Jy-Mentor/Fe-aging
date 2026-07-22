#!/usr/bin/env Rscript
# ============================================================================
# 独立运行完整 Step 09 (monocle3 拟时序 + Augur)
# 绕过 run_pipeline.R 中 19/20 模块 source 时的副作用
# ============================================================================

Sys.setenv(RTOOLS40_HOME = "C:/rtools45")
# 优先 D:/R-library/4.5 (含 monocle3/SeuratWrappers), 然后 D:/R-library (含 Augur)
.libPaths(c("D:/R-library/4.5", "D:/R-library", .libPaths()))

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratWrappers)
  library(monocle3)
  library(ggplot2)
})

script_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) source(f)
source(file.path(script_dir, "R", "09_sc_pseudotime_augur.R"))

cfg <- load_config(file.path(script_dir, "config.yaml"))
init_logger(file.path(cfg$project$log_dir,
                      sprintf("step09_full_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))),
            level = "INFO")

log_info("============================================================")
log_info("Step 09 独立运行 (monocle3 + Augur)")
log_info("R 版本: ", R.version.string)
log_info("Seurat: ", as.character(packageVersion("Seurat")))
log_info("monocle3: ", as.character(packageVersion("monocle3")))
log_info("Augur: ", as.character(packageVersion("Augur")))
log_info("============================================================")

set.seed(cfg$reproducibility$r_seed)

seu <- readRDS(file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds"))
log_info("[LOAD] sc_seu: ", ncol(seu), " cells x ", nrow(seu), " genes")

t0 <- Sys.time()
res <- step09_sc_pseudotime_augur(seu, cfg)
log_info("[DONE] Step 09 completed in ",
         round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), " mins")

if (!is.null(res$cds_neuron)) {
  log_info("[DONE] cds_neuron saved: ",
           file.path(cfg$project$rds_dir, "09_cds_neuron.rds"))
}
if (!is.null(res$cds_fs)) {
  log_info("[DONE] cds_ferrosenescence saved: ",
           file.path(cfg$project$rds_dir, "09_cds_ferrosenescence.rds"))
}
if (!is.null(res$augur)) {
  log_info("[DONE] Augur AUC result:")
  print(res$augur$AUC)
}
