#!/usr/bin/env Rscript
# debug monocle3 .run_monocle3 错误

Sys.setenv(RTOOLS40_HOME = "C:/rtools45")
.libPaths(c("D:/R-library/4.5", "D:/R-library", .libPaths()))

suppressPackageStartupMessages({
  library(Seurat)
  library(SeuratWrappers)
  library(monocle3)
  library(ggplot2)
})

script_dir <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$", full.names = TRUE)) source(f)
source(file.path(script_dir, "R", "09_sc_pseudotime_augur.R"))

cfg <- load_config(file.path(script_dir, "config.yaml"))
init_logger(file.path(cfg$project$log_dir, "debug_monocle3.log"), level = "INFO")

seu <- readRDS(file.path(cfg$project$rds_dir, "08_sc_seurat_annotated_scored.rds"))
log_info("cells:", ncol(seu), " features:", nrow(seu))

keep <- seu@meta.data[[cfg$data$sc_celltype_col]] %in% c("NeuronsGABA", "NeuronsGLUT")
neuron_sub <- subset(seu, cells = colnames(seu)[keep])
log_info("neuron cells:", ncol(neuron_sub))

options(error = recover)
res <- tryCatch({
  .run_monocle3(neuron_sub, cfg,
                celltype_col = cfg$data$sc_celltype_col,
                condition_col = cfg$data$sc_condition_col,
                tag = "neuron")
}, error = function(e) {
  log_error("ERROR: ", conditionMessage(e))
  print(sys.calls())
  return(NULL)
})