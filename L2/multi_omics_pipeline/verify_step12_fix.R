#!/usr/bin/env Rscript
# ============================================================================
# 验证 Step 12 (CMap) 修复: 12.5 节单样本 Wilcoxon 替代无效的 cor.test
# 加载真实 bulk_dea_list, 重跑 Step 12, 检查输出 CSV 不再全 NA
# ============================================================================
.libPaths(c("d:/铁衰老 绝不重蹈覆辙/R-library/4.5",
            "D:/R-library/4.5", .libPaths()))

script_dir <- "d:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
setwd(script_dir)

for (f in list.files(file.path(script_dir, "utils"), pattern = "\\.R$",
                     full.names = TRUE)) {
  source(f)
}

modules_needed <- c("12_integration_cmap.R")
for (f in file.path(script_dir, "R", modules_needed)) {
  if (file.exists(f)) {
    log_info("Loading module: ", basename(f))
    source(f)
  }
}

cfg <- load_config("config.yaml")
init_logger(file.path(cfg$project$log_dir,
                      sprintf("verify_step12_fix_%s.log",
                              format(Sys.time(), "%Y%m%d_%H%M%S"))), "INFO")
log_info("=== Verify Step 12 fix: wilcox.test replaces invalid cor.test ===")

# 加载 bulk_dea_list (Step 02 输出)
bulk_rds <- file.path(cfg$project$rds_dir, "02_bulk_dea_list.rds")
if (!file.exists(bulk_rds)) {
  log_error("Missing: ", bulk_rds, " (run Step 02 first)")
  quit(status = 1)
}
log_info("[Restore] Loading ", basename(bulk_rds))
bulk_dea_list <- readRDS(bulk_rds)
log_info("[Restore] bulk_dea_list: ", length(bulk_dea_list), " time points")
log_info("  names: ", paste(names(bulk_dea_list), collapse = ", "))

# 运行 Step 12
t0 <- Sys.time()
result <- step12_integration_cmap(bulk_dea_list, cfg)
elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
log_info(sprintf("=== Step 12 done in %.1fs ===", elapsed))

if (is.null(result)) {
  log_error("Step 12 returned NULL")
  quit(status = 1)
}

# 验证 12_bcp_ischemia_correlation.csv 不再全 NA
cor_csv <- file.path(cfg$project$tables_dir,
                      "12_bcp_ischemia_correlation.csv")
if (!file.exists(cor_csv)) {
  log_error("Missing: ", cor_csv)
  quit(status = 1)
}

cor_df <- read.csv(cor_csv, stringsAsFactors = FALSE)
log_info("[Verify] 12_bcp_ischemia_correlation.csv: ", nrow(cor_df), " rows")
log_info("  columns: ", paste(colnames(cor_df), collapse = ", "))

# 关键验证: p_value 不再全 NA (旧版全 NA 因 cor.test 对常数向量无效)
n_na_pvalue <- sum(is.na(cor_df$p_value))
n_non_na_pvalue <- sum(!is.na(cor_df$p_value))
log_info(sprintf("[Verify] p_value: %d NA, %d non-NA (旧版全 NA = bug)",
                 n_na_pvalue, n_non_na_pvalue))

if (n_non_na_pvalue > 0) {
  log_info("[OK] Fix verified: p_value has non-NA values")
  # 显示前几行
  log_info("[Verify] First 5 rows:")
  print(head(cor_df, 5))
} else {
  log_error("[FAIL] p_value still all NA — fix did not work")
  quit(status = 1)
}

# 验证 median_lfc 列存在 (新增列)
if ("median_lfc" %in% colnames(cor_df)) {
  log_info("[OK] median_lfc column present (new in fix)")
  log_info("  range: [", round(min(cor_df$median_lfc, na.rm = TRUE), 3),
           ", ", round(max(cor_df$median_lfc, na.rm = TRUE), 3), "]")
} else {
  log_error("[FAIL] median_lfc column missing")
  quit(status = 1)
}

# 验证 direction 列存在 (新增列)
if ("direction" %in% colnames(cor_df)) {
  log_info("[OK] direction column present (new in fix)")
  dir_table <- table(cor_df$direction)
  log_info("  distribution: ", paste(names(dir_table), "=",
                                      unname(dir_table), collapse = ", "))
} else {
  log_error("[FAIL] direction column missing")
  quit(status = 1)
}

log_info("\n[OK] Step 12 fix verification complete")
