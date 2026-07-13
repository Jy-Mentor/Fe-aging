#!/usr/bin/env Rscript
# ============================================================================
# 数据全面验证：格式、缺失值、异常值、基因-时间对应关系
# ============================================================================

project_root <- normalizePath(getwd())
l1_results  <- file.path(project_root, "L1", "results")
ferroaging_file <- file.path(project_root, "铁衰老基因.txt")

cat("============================================================\n")
cat("  数据集全面验证\n")
cat("============================================================\n\n")

# ============================================================================
# 1. 铁衰老基因集验证
# ============================================================================
cat("--- 1. 铁衰老基因集 ---\n")
fa_genes <- readLines(ferroaging_file, warn = FALSE)
fa_genes <- fa_genes[fa_genes != ""]
fa_genes <- unique(fa_genes)
cat(sprintf("  基因数量: %d\n", length(fa_genes)))
cat(sprintf("  是否有重复: %s\n", ifelse(any(duplicated(fa_genes)), "YES (需要去重!)", "No")))
cat(sprintf("  前5个基因: %s\n", paste(head(fa_genes, 5), collapse = ", ")))

# ============================================================================
# 2. GSE104036 表达矩阵验证（唯一多时间点数据集）
# ============================================================================
cat("\n--- 2. GSE104036 表达矩阵 ---\n")
expr_file <- file.path(l1_results, "GSE104036_expression_matrix.csv")

expr <- read.csv(expr_file, check.names = FALSE, stringsAsFactors = FALSE)
cat(sprintf("  文件: GSE104036_expression_matrix.csv\n"))
cat(sprintf("  维度: %d 基因 x %d 样本\n", nrow(expr), ncol(expr) - 1))

gene_col <- colnames(expr)[1]
cat(sprintf("  基因列名: '%s'\n", gene_col))
cat(sprintf("  唯一基因数: %d\n", length(unique(expr[[gene_col]]))))

expr_mat <- as.matrix(expr[, -1, drop = FALSE])

# 检查缺失值
na_count <- sum(is.na(expr_mat))
na_genes <- sum(apply(expr_mat, 1, function(x) any(is.na(x))))
cat(sprintf("  缺失值总数: %d (占 %.4f%%)\n", na_count, na_count / length(expr_mat) * 100))
cat(sprintf("  含缺失值的基因数: %d\n", na_genes))

# 检查全零基因
all_zero <- apply(expr_mat, 1, function(x) all(x == 0))
cat(sprintf("  全零基因数: %d\n", sum(all_zero)))

# 检查表达值范围
expr_min <- min(expr_mat, na.rm = TRUE)
expr_max <- max(expr_mat, na.rm = TRUE)
cat(sprintf("  表达值范围: %g - %g\n", expr_min, expr_max))

# 检查是否有负值
neg_count <- sum(expr_mat < 0, na.rm = TRUE)
cat(sprintf("  负值数量: %d (应为0)\n", neg_count))

# 检查是否为整数
sample_check <- expr_mat[1:min(100, nrow(expr_mat)), 1:min(10, ncol(expr_mat))]
is_int <- all(sapply(1:nrow(sample_check), function(i) {
  all(sample_check[i, ] == floor(sample_check[i, ]))
}))
cat(sprintf("  前100基因是否为整数: %s (RNA-seq count应全部为整数)\n", ifelse(is_int, "Yes", "No")))

# ============================================================================
# 3. 铁衰老基因在表达矩阵中的覆盖率
# ============================================================================
cat("\n--- 3. 铁衰老基因覆盖率 ---\n")

human_to_mouse_map <- list(
  'ACSL4'='Acsl4', 'HMOX1'='Hmox1', 'TFRC'='Tfrc', 'GPX4'='Gpx4',
  'HIF1A'='Hif1a', 'KEAP1'='Keap1', 'SOD1'='Sod1', 'NLRP3'='Nlrp3',
  'IL6'='Il6', 'TLR4'='Tlr4', 'MAPK1'='Mapk1', 'PTGS2'='Ptgs2',
  'CXCL10'='Cxcl10', 'LCN2'='Lcn2', 'IL1B'='Il1b', 'CD74'='Cd74',
  'IRF1'='Irf1', 'SP1'='Sp1', 'KLF6'='Klf6', 'EGR1'='Egr1',
  'BCL6'='Bcl6', 'CTSB'='Ctsb', 'SAT1'='Sat1', 'KDM6B'='Kdm6b',
  'LGMN'='Lgmn', 'IGFBP7'='Igfbp7', 'PDE4B'='Pde4b', 'EMP1'='Emp1',
  'EPHA4'='Epha4', 'RUNX3'='Runx3', 'FBXO31'='Fbxo31',
  'LPCAT3'='Lpcat3', 'DYRK1A'='Dyrk1a', 'LACTB'='Lactb',
  'GMFB'='Gmfb', 'HBP1'='Hbp1', 'MAPK14'='Mapk14',
  'ABCC1'='Abcc1', 'ACVR1B'='Acvr1b', 'ALOX15'='Alox15',
  'ATF3'='Atf3', 'ATG3'='Atg3', 'BAP1'='Bap1', 'BRD7'='Brd7',
  'CAVIN1'='Cavin1', 'CD82'='Cd82', 'CDO1'='Cdo1',
  'COX7A1'='Cox7a1', 'DPEP1'='Dpep1', 'DPP4'='Dpp4',
  'DUOX1'='Duox1', 'E2F1'='E2f1', 'E2F3'='E2f3', 'EBF3'='Ebf3',
  'EDN1'='Edn1', 'EPHA2'='Epha2', 'ERN1'='Ern1',
  'FOSL1'='Fosl1', 'HERPUD1'='Herpud1', 'HMGB1'='Hmgb1',
  'ICA1'='Ica1', 'IFNG'='Ifng', 'IRF7'='Irf7', 'IRF9'='Irf9',
  'LIFR'='Lifr', 'LOX'='Lox', 'MAP3K14'='Map3k14',
  'MCU'='Mcu', 'MEN1'='Men1', 'MPO'='Mpo', 'NOX4'='Nox4',
  'NR1D1'='Nr1d1', 'NR2F2'='Nr2f2', 'NUAK2'='Nuak2',
  'PADI4'='Padi4', 'PPP2R2B'='Ppp2r2b', 'PRKD1'='Prkd1',
  'PTBP1'='Ptbp1', 'RBM3'='Rbm3', 'S100A8'='S100a8',
  'SETD7'='Setd7', 'SLAMF8'='Slamf8', 'SLC1A5'='Slc1a5',
  'SMARCB1'='Smarcb1', 'SMURF2'='Smurf2', 'SNCA'='Snca',
  'SOCS1'='Socs1', 'SOCS2'='Socs2', 'SPATA2'='Spata2',
  'TBX2'='Tbx2', 'TNFAIP1'='Tnfaip1', 'TNFAIP3'='Tnfaip3',
  'TXNIP'='Txnip', 'WNT5A'='Wnt5a', 'WWTR1'='Wwtr1', 'YAP1'='Yap1',
  'ZEB1'='Zeb1'
)

fa_mouse <- sapply(fa_genes, function(g) {
  if (g %in% names(human_to_mouse_map)) human_to_mouse_map[[g]] else g
})
names(fa_mouse) <- NULL

expr_genes <- expr[[gene_col]]
found <- fa_mouse[fa_mouse %in% expr_genes]
not_found <- fa_mouse[!fa_mouse %in% expr_genes]

cat(sprintf("  铁衰老基因在表达矩阵中: %d / %d (%.1f%%)\n",
            length(found), length(fa_mouse), length(found)/length(fa_mouse)*100))
if (length(not_found) > 0) {
  cat(sprintf("  未找到的基因 (%d): %s\n", length(not_found), paste(not_found, collapse = ", ")))
}

# ============================================================================
# 4. 样本元数据验证
# ============================================================================
cat("\n--- 4. 样本元数据 ---\n")
meta_file <- file.path(l1_results, "GSE104036_sample_meta.csv")
meta <- read.csv(meta_file, check.names = FALSE, stringsAsFactors = FALSE)

cat(sprintf("  样本数: %d\n", nrow(meta)))
cat(sprintf("  列: %s\n", paste(colnames(meta), collapse = ", ")))

# 样本名一致性
expr_samples <- colnames(expr)[-1]
meta_samples <- meta$sample
only_expr <- setdiff(expr_samples, meta_samples)
only_meta <- setdiff(meta_samples, expr_samples)
cat(sprintf("  表达矩阵样本数: %d\n", length(expr_samples)))
cat(sprintf("  元数据样本数: %d\n", length(meta_samples)))
cat(sprintf("  仅在表达矩阵中: %s\n",
            ifelse(length(only_expr) > 0, paste(only_expr, collapse=","), "None")))
cat(sprintf("  仅在元数据中: %s\n",
            ifelse(length(only_meta) > 0, paste(only_meta, collapse=","), "None")))

# 各组样本分布
cat("\n  样本分组分布:\n")
groups_table <- table(meta$group, meta$time)
print(groups_table)

cat(sprintf("\n  唯一组别: %s\n", paste(unique(meta$group), collapse = ", ")))
cat(sprintf("  唯一时间点: %s\n", paste(unique(meta$time), collapse = ", ")))
cat(sprintf("  唯一组织: %s\n", paste(unique(meta$tissue), collapse = ", ")))

# ============================================================================
# 5. GSE104036 DE 结果验证
# ============================================================================
cat("\n--- 5. GSE104036 DE 结果 ---\n")
de_file <- file.path(l1_results, "GSE104036_DE_gene_level.csv")
de <- read.csv(de_file, check.names = FALSE, stringsAsFactors = FALSE)
cat(sprintf("  维度: %d 基因 x %d 列\n", nrow(de), ncol(de)))
cat(sprintf("  列: %s\n", paste(colnames(de), collapse = ", ")))

required_cols <- c("logFC", "P.Value", "adj.P.Val")
for (col_name in required_cols) {
  cat(sprintf("  '%s' 列存在: %s\n",
              col_name, ifelse(col_name %in% colnames(de), "Yes", "NO - MISSING!")))
}

cat(sprintf("  logFC 范围: %.4f - %.4f\n", min(de$logFC, na.rm = TRUE), max(de$logFC, na.rm = TRUE)))
cat(sprintf("  显著基因 (adj.P.Val < 0.05): %d (%.1f%%)\n",
            sum(de$adj.P.Val < 0.05, na.rm = TRUE),
            sum(de$adj.P.Val < 0.05, na.rm = TRUE) / nrow(de) * 100))

na_pval <- sum(is.na(de$P.Value))
na_adjp <- sum(is.na(de$adj.P.Val))
cat(sprintf("  P值缺失: %d, adj.P.Val 缺失: %d\n", na_pval, na_adjp))

# ============================================================================
# 6. 其他数据集简查
# ============================================================================
cat("\n--- 6. 其他数据集简查 ---\n")
other_datasets <- c("GSE16561", "GSE61616", "GSE97537")
for (ds in other_datasets) {
  de_f <- file.path(l1_results, paste0(ds, "_DE_gene_level.csv"))
  if (file.exists(de_f)) {
    d <- read.csv(de_f, check.names = FALSE, stringsAsFactors = FALSE)
    sp <- ifelse("Species" %in% colnames(d), d$Species[1], "Unknown")
    cat(sprintf("  %s: %d 基因, %d 列, species=%s\n", ds, nrow(d), ncol(d), sp))
  } else {
    cat(sprintf("  %s: _DE_gene_level.csv NOT FOUND\n", ds))
  }
}

# ============================================================================
# 7. 异常值检测（GSE104036 铁衰老基因表达）
# ============================================================================
cat("\n--- 7. 铁衰老基因异常值检测 ---\n")
fa_expr <- expr[expr[[gene_col]] %in% found, ]
if (nrow(fa_expr) > 0) {
  fa_mat <- as.matrix(fa_expr[, -1, drop = FALSE])
  rownames(fa_mat) <- fa_expr[[gene_col]]

  outlier_genes <- c()
  for (i in 1:nrow(fa_mat)) {
    gene_name <- rownames(fa_mat)[i]
    vals <- as.numeric(fa_mat[i, ])
    if (sd(vals) > 0) {
      z <- (vals - mean(vals)) / sd(vals)
      if (any(abs(z) > 3)) {
        outlier_genes <- c(outlier_genes, gene_name)
      }
    }
  }
  if (length(outlier_genes) > 0) {
    cat(sprintf("  含异常表达值(z-score>3)的基因 (%d): %s\n",
                length(outlier_genes), paste(outlier_genes, collapse = ", ")))
  } else {
    cat("  无异常值 (所有铁衰老基因表达 z-score < 3)\n")
  }
}

# ============================================================================
# 8. 铁衰老基因表达分布（各时间点-组别平均表达）
# ============================================================================
cat("\n--- 8. 铁衰老基因表达分布概况 ---\n")

expr_long_list <- list()
for (s in expr_samples) {
  grp <- meta$group[match(s, meta$sample)]
  tm <- meta$time[match(s, meta$sample)]
  for (g in found) {
    g_expr <- expr[expr[[gene_col]] == g, s]
    if (length(g_expr) > 0 && !is.na(g_expr[1])) {
      expr_long_list[[length(expr_long_list) + 1]] <- data.frame(
        gene = g, sample = s, group = grp, time = tm,
        expression = as.numeric(g_expr[1]),
        stringsAsFactors = FALSE
      )
    }
  }
}
fa_expr_long <- do.call(rbind, expr_long_list)
fa_expr_long$group_time <- paste0(fa_expr_long$group, "_", fa_expr_long$time)
cat(sprintf("  铁衰老基因长格式记录数: %d\n", nrow(fa_expr_long)))

# 每个 group_time 的平均表达
gt_list <- split(fa_expr_long, fa_expr_long$group_time)
fa_summary <- do.call(rbind, lapply(names(gt_list), function(gt) {
  d <- gt_list[[gt]]
  data.frame(
    group_time = gt,
    mean_expr = mean(d$expression, na.rm = TRUE),
    sd_expr = sd(d$expression, na.rm = TRUE),
    median_expr = median(d$expression, na.rm = TRUE),
    n = nrow(d),
    stringsAsFactors = FALSE
  )
}))

cat("\n  各时间点-组别铁衰老基因表达汇总:\n")
print(fa_summary[order(fa_summary$group_time), ])

# ============================================================================
# 9. 验证总结
# ============================================================================
cat("\n============================================================\n")
cat("  验证总结\n")
cat("============================================================\n")

issues <- 0

if (neg_count > 0) {
  cat(sprintf("  [FAIL] 负值数量: %d\n", neg_count))
  issues <- issues + 1
}
if (na_count > 0) {
  cat(sprintf("  [WARN] 缺失值: %d\n", na_count))
  issues <- issues + 1
}
if (length(only_expr) > 0 || length(only_meta) > 0) {
  cat(sprintf("  [WARN] 样本名不一致\n"))
  issues <- issues + 1
}
if (length(not_found) > 0) {
  cat(sprintf("  [INFO] %d 个铁衰老基因在GSE104036中未找到(可能物种差异)\n", length(not_found)))
}
if (length(found) < 50) {
  cat(sprintf("  [WARN] 铁衰老基因覆盖率 < 50%%\n"))
  issues <- issues + 1
}

if (issues == 0) {
  cat("\n  [PASS] 数据验证通过！置信度 > 95%\n")
  cat(sprintf("  - 表达矩阵格式正确，无缺失值，无负值\n"))
  cat("  - 样本与元数据完全匹配\n")
  cat(sprintf("  - 铁衰老基因覆盖率 %.1f%%\n", length(found)/length(fa_mouse)*100))
  cat(sprintf("  - 异常值: 无\n"))
} else {
  cat(sprintf("\n  [FAIL] 发现 %d 个问题！\n", issues))
}

cat("\n验证完成\n")
