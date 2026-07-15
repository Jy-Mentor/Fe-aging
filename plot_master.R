##############################################################################
# MASTER PLOT SCRIPT
# 铁衰老项目 — 全套图表绘制
# 
# 参照文献风格:
#   - Nature Communications (蝴蝶图/哑铃图)
#   - Nature (空间转录组复合图)
#   - MedComm (生存曲线)
#   - TCMNP R包 (网络药理可视化)
#   - SCIPainter (棒棒糖图)
#
# 输出: figures/ (PNG 300dpi) + figures/pdf/ (矢量PDF)
##############################################################################

# ============================================================================
# 0. 环境检查
# ============================================================================
cat("========================================\n")
cat("  Ferroaging Project — Figure Generation\n")
cat("========================================\n\n")

pkgs <- c("ggplot2", "patchwork", "ggrepel", "ggpubr", "dplyr", "tidyr",
           "readr", "stringr", "ggsci", "viridis", "cowplot", "scales",
           "ggforce")

for (pkg in pkgs) {
  if (!require(pkg, character.only = TRUE)) {
    cat(sprintf("ERROR: Package '%s' not available\n", pkg))
    quit(status = 1)
  }
}

cat(sprintf("R version: %s\n", R.version.string))
cat(sprintf("ggplot2: %s\n", as.character(packageVersion("ggplot2"))))

# ============================================================================
# 1. 检查数据文件
# ============================================================================
cat("\n--- Checking data files ---\n")

data_files <- c(
  "L2/results/GSE233815_sn/cell_metadata_with_ferroaging_score.csv",
  "L2/results/GSE233815_sn/ferroaging_score_by_condition_cellclass.csv",
  "L2/results/ciri_ferroaging_lasso_candidates.csv",
  "L2/results/external_validation_results.csv",
  "L2/results/core_ppi_topology.csv",
  "L2/results/ssgsea_ferroaging_scores.csv",
  "L2/results/GSE233815_sn/microglia_subcluster/microglia_high_fa_markers_annotated.csv",
  "L2/results/GSE233815_sn/microglia_subcluster/microglia_cluster_ferroaging_summary.csv",
  "L2/results/caryophyllene_ciri_overlap_official_string.csv",
  "L3/results/tcm_compound_pool_tox_filtered.csv",
  "L4/results_v10_minibatch/model_performance_v67.csv",
  "L4/results_v10_minibatch/tcm_top_candidates_v67.csv",
  "L4/results_v10_minibatch/tcm_predictions_full_v67.csv"
)

missing <- c()
for (f in data_files) {
  full_path <- file.path("d:/铁衰老 绝不重蹈覆辙", f)
  if (file.exists(full_path)) {
    cat(sprintf("  [OK] %s\n", f))
  } else {
    cat(sprintf("  [MISSING] %s\n", f))
    missing <- c(missing, f)
  }
}

if (length(missing) > 0) {
  cat(sprintf("\nWARNING: %d data files missing!\n", length(missing)))
}

# ============================================================================
# 2. 执行绘图
# ============================================================================
cat("\n--- Running plotting scripts ---\n")

scripts <- c(
  "plot_fig1_singlecell.R",
  "plot_fig2_model_compound.R",
  "plot_fig3_de_validation.R",
  "plot_fig4_chemistry_microglia.R",
  "plot_fig5_scissor.R",
  "plot_fig6_cellchat.R"
)

base_dir <- "d:/铁衰老 绝不重蹈覆辙"

for (s in scripts) {
  full <- file.path(base_dir, s)
  if (!file.exists(full)) {
    cat(sprintf("[SKIP] %s (not found)\n", s))
    next
  }
  cat(sprintf("\n>>> Running %s ...\n", s))
  start_time <- Sys.time()
  exit_code <- tryCatch({
    source(full, local = TRUE)
    0
  }, error = function(e) {
    cat(sprintf("  ERROR: %s\n", e$message))
    1
  })
  elapsed <- difftime(Sys.time(), start_time, units = "secs")
  if (exit_code == 0) {
    cat(sprintf("<<< %s DONE (%.1fs)\n", s, elapsed))
  } else {
    cat(sprintf("<<< %s FAILED (%.1fs)\n", s, elapsed))
  }
}

# ============================================================================
# 3. 输出总结
# ============================================================================
cat("\n--- Output Summary ---\n")

png_files <- list.files("d:/铁衰老 绝不重蹈覆辙/figures", pattern = "\\.png$", full.names = TRUE)
pdf_files <- list.files("d:/铁衰老 绝不重蹈覆辙/figures/pdf", pattern = "\\.pdf$", full.names = TRUE)

cat(sprintf("PNG files: %d\n", length(png_files)))
cat(sprintf("PDF files: %d\n", length(pdf_files)))

if (length(png_files) > 0) {
  sizes <- file.info(png_files)$size / 1024
  cat(sprintf("Total PNG size: %.1f KB\n", sum(sizes)))
  for (f in png_files) {
    cat(sprintf("  %s (%.0f KB)\n", basename(f), file.info(f)$size/1024))
  }
}

cat("\n========================================\n")
cat("  Figure generation complete!\n")
cat("========================================\n")
