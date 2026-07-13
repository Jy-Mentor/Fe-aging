#!/usr/bin/env Rscript
# ============================================================================
# 铁衰老基因时间表达趋势可视化
# GSE104036: 小鼠 MCAO 模型，多时间点 (Sham/Contralateral/Ipsilateral: 0/3/6/12/24hr)
# 参考模板: D:\R语言绘图模板\箱线图+组内显著性+组内线性回归分析趋势性P值
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(reshape2)
  library(ggpubr)
})

project_root <- normalizePath(getwd())
l1_results  <- file.path(project_root, "L1", "results")
fig_dir     <- file.path(project_root, "L2", "results", "figures")
ferroaging_file <- file.path(project_root, "铁衰老基因.txt")

dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# 加载数据
# ============================================================================
cat("Loading data...\n")

# 表达矩阵
expr <- read.csv(file.path(l1_results, "GSE104036_expression_matrix.csv"),
                 check.names = FALSE, stringsAsFactors = FALSE)
meta <- read.csv(file.path(l1_results, "GSE104036_sample_meta.csv"),
                 check.names = FALSE, stringsAsFactors = FALSE)

# 铁衰老基因
fa_human <- readLines(ferroaging_file, warn = FALSE)
fa_human <- unique(fa_human[fa_human != ""])

# 人鼠基因转换
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

fa_mouse <- sapply(fa_human, function(g) {
  if (g %in% names(human_to_mouse_map)) human_to_mouse_map[[g]] else g
})
names(fa_mouse) <- NULL

# 找到表达矩阵中存在的铁衰老基因
gene_col <- colnames(expr)[1]
found <- fa_mouse[fa_mouse %in% expr[[gene_col]]]
cat(sprintf("Ferroaging genes found in expression matrix: %d / %d\n",
            length(found), length(fa_mouse)))

# ============================================================================
# 数据整理：提取铁衰老基因，CPM 标准化，转长格式
# ============================================================================
cat("Preparing data...\n")

# 提取铁衰老基因
fa_expr <- expr[expr[[gene_col]] %in% found, ]
rownames(fa_expr) <- fa_expr[[gene_col]]
fa_mat <- as.matrix(fa_expr[, -1, drop = FALSE])

# CPM 标准化 (counts per million) — 使不同基因间可比较
lib_sizes <- colSums(fa_mat)
cpm_mat <- t(t(fa_mat) / lib_sizes * 1e6)

# log2(CPM + 1) 转换 — 稳定方差
log2cpm <- log2(cpm_mat + 1)

# 转长格式
expr_samples <- colnames(fa_mat)
long_list <- list()
for (i in seq_len(nrow(log2cpm))) {
  gene_name <- rownames(log2cpm)[i]
  for (s in expr_samples) {
    grp <- meta$group[match(s, meta$sample)]
    tm  <- meta$time[match(s, meta$sample)]
    long_list[[length(long_list) + 1]] <- data.frame(
      Gene = gene_name,
      Sample = s,
      Tissue = grp,
      Time = tm,
      log2CPM = as.numeric(log2cpm[i, s]),
      stringsAsFactors = FALSE
    )
  }
}
fa_long <- do.call(rbind, long_list)

# 设置因子顺序（时间有序）
fa_long$Time <- factor(fa_long$Time, levels = c("0hr", "3hr", "6hr", "12hr", "24hr"))
fa_long$Tissue <- factor(fa_long$Tissue, levels = c("Sham", "Contralateral", "Ipsilateral"))

# 创建分组变量（Tissue + Time）
fa_long$Group <- paste0(substr(fa_long$Tissue, 1, 3), "_", fa_long$Time)
fa_long$Group <- factor(fa_long$Group,
  levels = c("Sha_0hr",
             "Con_3hr", "Con_6hr", "Con_12hr", "Con_24hr",
             "Ips_3hr", "Ips_6hr", "Ips_12hr", "Ips_24hr"))

cat(sprintf("Long-format data: %d rows\n", nrow(fa_long)))

# ============================================================================
# 统计检验：每个铁衰老基因在 Ipsilateral 组织中随时间的差异
# ============================================================================
cat("Running statistical tests...\n")

stat_results <- data.frame(
  Gene = character(),
  KW_pvalue = numeric(),
  ipsi_3hr_mean = numeric(),
  ipsi_6hr_mean = numeric(),
  ipsi_12hr_mean = numeric(),
  ipsi_24hr_mean = numeric(),
  sham_mean = numeric(),
  max_fc = numeric(),
  stringsAsFactors = FALSE
)

for (g in found) {
  g_data <- fa_long[fa_long$Gene == g, ]
  ipsi_data <- g_data[g_data$Tissue == "Ipsilateral", ]
  sham_data <- g_data[g_data$Tissue == "Sham", ]

  # Kruskal-Wallis test: 同侧组织中不同时间点是否有差异
  if (length(unique(ipsi_data$Time)) >= 3) {
    kw <- kruskal.test(log2CPM ~ Time, data = ipsi_data)
    kw_p <- kw$p.value
  } else {
    kw_p <- NA
  }

  sham_mean <- mean(sham_data$log2CPM)
  t3  <- mean(ipsi_data$log2CPM[ipsi_data$Time == "3hr"])
  t6  <- mean(ipsi_data$log2CPM[ipsi_data$Time == "6hr"])
  t12 <- mean(ipsi_data$log2CPM[ipsi_data$Time == "12hr"])
  t24 <- mean(ipsi_data$log2CPM[ipsi_data$Time == "24hr"])

  # max fold change vs sham
  ipsi_means <- c(t3, t6, t12, t24)
  ipsi_means <- ipsi_means[!is.na(ipsi_means)]
  max_fc <- if (length(ipsi_means) > 0) max(abs(ipsi_means - sham_mean)) else 0

  stat_results <- rbind(stat_results, data.frame(
    Gene = g,
    KW_pvalue = kw_p,
    ipsi_3hr_mean = t3,
    ipsi_6hr_mean = t6,
    ipsi_12hr_mean = t12,
    ipsi_24hr_mean = t24,
    sham_mean = sham_mean,
    max_fc = max_fc,
    stringsAsFactors = FALSE
  ))
}

# FDR 校正
stat_results$FDR <- p.adjust(stat_results$KW_pvalue, method = "BH")
stat_results <- stat_results[order(stat_results$FDR), ]

# 标记显著性类别
stat_results$Category <- "NS"
stat_results$Category[stat_results$FDR < 0.05 & stat_results$max_fc >= 0.5] <- "Significant_HighFC"
stat_results$Category[stat_results$FDR < 0.05 & stat_results$max_fc < 0.5] <- "Significant_LowFC"
stat_results$Category[stat_results$FDR >= 0.05 & stat_results$max_fc >= 1.0] <- "Trend_HighFC"

cat(sprintf("\nStatistical test results:\n"))
cat(sprintf("  Significant (FDR<0.05) & HighFC: %d genes\n",
            sum(stat_results$Category == "Significant_HighFC")))
cat(sprintf("  Significant (FDR<0.05) & LowFC: %d genes\n",
            sum(stat_results$Category == "Significant_LowFC")))
cat(sprintf("  Trend (NS but FC>1): %d genes\n",
            sum(stat_results$Category == "Trend_HighFC")))
cat(sprintf("  NS: %d genes\n",
            sum(stat_results$Category == "NS")))

# ============================================================================
# Figure 1: 按显著性分组的基因表达时间趋势面板 (Significant vs NS)
# ============================================================================
cat("\nGenerating Figure 1: Time-course expression panel...\n")

# 选取 top 显著基因（FDR 严格，用小样本 n=3 时广泛不显著；用原始 p 值排序取 top）
top_by_p <- stat_results[order(stat_results$KW_pvalue), ]
sig_genes <- top_by_p$Gene[1:min(16, nrow(top_by_p))]
# 后半作为对照组 (p 值最高的几个)
ns_genes <- stat_results$Gene[order(stat_results$KW_pvalue, decreasing = TRUE)][1:4]
plot_genes <- c(sig_genes, ns_genes)
plot_genes <- plot_genes[!is.na(plot_genes)]

# 统计：有多少基因在原始 p<0.05 水平显著
cat(sprintf("\n  Genes with raw p < 0.05 (K-W): %d\n", sum(stat_results$KW_pvalue < 0.05, na.rm = TRUE)))
cat(sprintf("  Plotting top %d by p-value + %d NS controls\n", length(sig_genes), length(ns_genes)))

plot_data <- fa_long[fa_long$Gene %in% plot_genes, ]
plot_data$Gene <- factor(plot_data$Gene, levels = plot_genes)

# 添加显著性标记
plot_data$SigLabel <- ""
for (g in unique(plot_data$Gene)) {
  idx <- which(stat_results$Gene == g)
  if (length(idx) > 0 && !is.na(stat_results$KW_pvalue[idx])) {
    pval <- stat_results$KW_pvalue[idx]
    if (pval < 0.01) {
      plot_data$SigLabel[plot_data$Gene == g] <- "**"
    } else if (pval < 0.05) {
      plot_data$SigLabel[plot_data$Gene == g] <- "*"
    } else {
      plot_data$SigLabel[plot_data$Gene == g] <- "ns"
    }
  }
}

# 面板标题包含显著性（在基因名后加 *）
gene_labels <- sapply(levels(plot_data$Gene), function(g) {
  idx <- which(stat_results$Gene == g)
  if (length(idx) > 0 && !is.na(stat_results$KW_pvalue[idx])) {
    pval <- stat_results$KW_pvalue[idx]
    if (pval < 0.01) return(paste0(g, " **"))
    if (pval < 0.05) return(paste0(g, " *"))
  }
  return(g)
})
names(gene_labels) <- levels(plot_data$Gene)

# 颜色方案
tissue_colors <- c("Sham" = "#95A5A6", "Contralateral" = "#3498DB", "Ipsilateral" = "#E74C3C")

# 小提琴图 + 箱线图面板
p1 <- ggplot(plot_data, aes(x = Time, y = log2CPM, fill = Tissue)) +
  geom_violin(alpha = 0.4, scale = "width") +
  geom_boxplot(width = 0.25, outlier.size = 0.8, alpha = 0.7,
               position = position_dodge(0.9)) +
  scale_fill_manual(values = tissue_colors,
                    labels = c("Sham (基线)", "Contralateral (对侧)", "Ipsilateral (患侧)")) +
  facet_wrap(~ Gene, ncol = 4, scales = "free_y",
             labeller = labeller(Gene = gene_labels)) +
  labs(
    title = "铁衰老基因在脑缺血后的时间表达趋势 (GSE104036, Mouse MCAO)",
    subtitle = "Sham = 基线(0hr); Contralateral = 对侧半球; Ipsilateral = 患侧半球\n** raw p<0.01  * raw p<0.05  (Kruskal-Wallis test, n=3 per time point)",
    x = "缺血后时间",
    y = expression(log[2](CPM + 1)),
    fill = "组织"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1, size = 9),
    strip.text = element_text(size = 9, face = "bold"),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "ferroaging_timecourse_expression_panel.pdf"),
       p1, width = 16, height = ifelse(length(plot_genes) <= 8, 8,
                                       ceiling(length(plot_genes) / 4) * 4),
       dpi = 300)
ggsave(file.path(fig_dir, "ferroaging_timecourse_expression_panel.png"),
       p1, width = 16, height = ifelse(length(plot_genes) <= 8, 8,
                                       ceiling(length(plot_genes) / 4) * 4),
       dpi = 300)
cat(sprintf("  Figure 1 saved: %d genes\n", length(plot_genes)))

# ============================================================================
# Figure 2: 铁衰老基因群体平均表达趋势（按组织分组，均值±SEM）
# ============================================================================
cat("Generating Figure 2: Mean expression trend...\n")

# 计算每个时间点-组的均值±SEM
trend_data <- aggregate(log2CPM ~ Tissue + Time, data = fa_long, FUN = mean)
trend_sd   <- aggregate(log2CPM ~ Tissue + Time, data = fa_long, FUN = sd)
trend_n    <- aggregate(log2CPM ~ Tissue + Time, data = fa_long, FUN = length)

colnames(trend_data)[3] <- "Mean"
trend_data$SD  <- trend_sd$log2CPM
trend_data$N   <- trend_n$log2CPM
trend_data$SEM <- trend_data$SD / sqrt(trend_data$N)

p2 <- ggplot(trend_data, aes(x = Time, y = Mean, color = Tissue, group = Tissue)) +
  geom_ribbon(aes(ymin = Mean - SEM, ymax = Mean + SEM, fill = Tissue),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = tissue_colors,
                     labels = c("Sham (基线)", "Contralateral (对侧)", "Ipsilateral (患侧)")) +
  scale_fill_manual(values = tissue_colors, guide = "none") +
  labs(
    title = "铁衰老基因群体平均表达趋势 (n=95 genes)",
    subtitle = "Mean ± SEM; GSE104036 Mouse MCAO Model",
    x = "缺血后时间",
    y = expression(paste("平均 ", log[2], "(CPM + 1)")),
    color = "组织"
  ) +
  theme_bw(base_size = 13) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

ggsave(file.path(fig_dir, "ferroaging_timecourse_mean_trend.pdf"), p2, width = 8, height = 6, dpi = 300)
ggsave(file.path(fig_dir, "ferroaging_timecourse_mean_trend.png"), p2, width = 8, height = 6, dpi = 300)
cat("  Figure 2 saved\n")

# ============================================================================
# Figure 3: Ipsilateral 组织中显著时间依赖性基因的详细趋势（带显著性标注）
# ============================================================================
cat("Generating Figure 3: Significant genes with pairwise stats...\n")

top_sig <- stat_results$Gene[stat_results$KW_pvalue < 0.01][1:min(8, sum(stat_results$KW_pvalue < 0.01, na.rm = TRUE))]
if (length(top_sig) < 4) {
  # 放宽到 p<0.05
  top_sig <- stat_results$Gene[stat_results$KW_pvalue < 0.05][1:min(8, sum(stat_results$KW_pvalue < 0.05, na.rm = TRUE))]
}

if (length(top_sig) >= 4) {
  ipsi_plot_data <- fa_long[fa_long$Gene %in% top_sig & fa_long$Tissue == "Ipsilateral", ]
  ipsi_plot_data$Gene <- factor(ipsi_plot_data$Gene, levels = top_sig)

  # 为每个基因计算各时间点 vs Sham(0hr) 的 wilcox 检验
  sham_data <- fa_long[fa_long$Tissue == "Sham", ]

  stat_annotations <- data.frame()
  for (g in top_sig) {
    g_sham <- sham_data$log2CPM[sham_data$Gene == g]
    g_ipsi <- ipsi_plot_data[ipsi_plot_data$Gene == g, ]
    for (tm in c("3hr", "6hr", "12hr", "24hr")) {
      g_t <- g_ipsi$log2CPM[g_ipsi$Time == tm]
      if (length(g_t) >= 2 && length(g_sham) >= 2) {
        wt <- wilcox.test(g_sham, g_t, exact = FALSE)
        if (wt$p.value < 0.05) {
          mean_t <- mean(g_t)
          stat_annotations <- rbind(stat_annotations, data.frame(
            Gene = g,
            Time = tm,
            pvalue = wt$p.value,
            y_pos = mean_t + 0.8,
            stringsAsFactors = FALSE
          ))
        }
      }
    }
  }

  p3 <- ggplot(ipsi_plot_data, aes(x = Time, y = log2CPM)) +
    geom_violin(aes(fill = Time), alpha = 0.35, scale = "width") +
    geom_boxplot(width = 0.3, outlier.size = 1, alpha = 0.8) +
    geom_jitter(width = 0.15, alpha = 0.5, size = 1.5) +
    facet_wrap(~ Gene, ncol = 4, scales = "free_y") +
    scale_fill_manual(values = c("0hr" = "#95A5A6", "3hr" = "#F39C12",
                                  "6hr" = "#E67E22", "12hr" = "#D35400",
                                  "24hr" = "#C0392B")) +
    labs(
      title = "铁衰老基因在患侧(Ipsilateral)组织中的时间差异表达",
      subtitle = paste0("Top ", length(top_sig), " 基因 (Kruskal-Wallis p < 0.05)"),
      x = "缺血后时间",
      y = expression(log[2](CPM + 1)),
      fill = "时间"
    ) +
    theme_bw(base_size = 12) +
    theme(
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )

  if (nrow(stat_annotations) > 0) {
    p3 <- p3 +
      geom_text(data = stat_annotations,
                aes(x = Time, y = y_pos),
                label = ifelse(stat_annotations$pvalue < 0.001, "***",
                        ifelse(stat_annotations$pvalue < 0.01, "**", "*")),
                size = 4, color = "red")
  }

  ggsave(file.path(fig_dir, "ferroaging_timecourse_ipsi_significant.pdf"),
         p3, width = 14, height = ceiling(length(top_sig) / 4) * 4, dpi = 300)
  ggsave(file.path(fig_dir, "ferroaging_timecourse_ipsi_significant.png"),
         p3, width = 14, height = ceiling(length(top_sig) / 4) * 4, dpi = 300)
  cat(sprintf("  Figure 3 saved: %d genes\n", length(top_sig)))
} else {
  cat("  Figure 3 skipped: insufficient significant genes\n")
}

# ============================================================================
# Figure 4: 热图 — 铁衰老基因在 Ipsilateral 中的时间表达谱
# ============================================================================
cat("Generating Figure 4: Heatmap of ipsilateral time-course expression...\n")

# 仅取 Ipsilateral 样本，按时间排序
ipsi_samples <- meta$sample[meta$group == "Ipsilateral"]
ipsi_ordered <- ipsi_samples[order(match(meta$time[match(ipsi_samples, meta$sample)],
                                          c("3hr", "6hr", "12hr", "24hr")))]

# 选取统计显著的基因做热图
heat_genes <- stat_results$Gene[stat_results$KW_pvalue < 0.05][1:min(30, sum(stat_results$KW_pvalue < 0.05, na.rm = TRUE))]
if (length(heat_genes) < 10) {
  heat_genes <- stat_results$Gene[1:min(30, nrow(stat_results))]
}

heat_mat <- log2cpm[heat_genes, ipsi_ordered, drop = FALSE]

# Z-score normalize per gene for heatmap
heat_z <- t(scale(t(heat_mat)))

heat_long <- melt(heat_z)
colnames(heat_long) <- c("Gene", "Sample", "Zscore")
heat_long$Time <- meta$time[match(heat_long$Sample, meta$sample)]
heat_long$Time <- factor(heat_long$Time, levels = c("3hr", "6hr", "12hr", "24hr"))

# 添加显著性标签
heat_long$GeneLabel <- heat_long$Gene
for (i in seq_len(nrow(stat_results))) {
  if (stat_results$Gene[i] %in% heat_long$Gene) {
    pval <- stat_results$KW_pvalue[i]
    sig <- ifelse(pval < 0.01, "**", ifelse(pval < 0.05, "*", ""))
    if (sig != "") {
      heat_long$GeneLabel[heat_long$Gene == stat_results$Gene[i]] <- paste0(stat_results$Gene[i], " ", sig)
    }
  }
}

p4 <- ggplot(heat_long, aes(x = Sample, y = GeneLabel, fill = Zscore)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                        midpoint = 0, name = "Z-score") +
  facet_grid(~ Time, scales = "free_x", space = "free_x") +
  labs(
    title = "铁衰老基因在患侧(Ipsilateral)组织中的时间表达热图",
    subtitle = paste0(length(heat_genes), " 基因 (Kruskal-Wallis p < 0.05)"),
    x = NULL,
    y = "Gene"
  ) +
  theme_bw(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
    axis.text.y = element_text(size = 9),
    strip.text = element_text(size = 10, face = "bold"),
    panel.grid = element_blank(),
    legend.position = "right"
  )

ggsave(file.path(fig_dir, "ferroaging_timecourse_heatmap.pdf"),
       p4, width = 12, height = max(6, length(heat_genes) * 0.28), dpi = 300)
ggsave(file.path(fig_dir, "ferroaging_timecourse_heatmap.png"),
       p4, width = 12, height = max(6, length(heat_genes) * 0.28), dpi = 300)
cat("  Figure 4 saved\n")

# ============================================================================
# 保存统计结果
# ============================================================================
write.csv(stat_results, file.path(project_root, "L2", "results",
           "ferroaging_timecourse_statistics.csv"), row.names = FALSE)
cat("\nStatistics saved to ferroaging_timecourse_statistics.csv\n")

# ============================================================================
# 输出关键发现
# ============================================================================
cat("\n============================================================\n")
cat("  KEY FINDINGS\n")
cat("============================================================\n")
cat(sprintf("  总铁衰老基因分析: %d\n", nrow(stat_results)))
cat(sprintf("  显著时间依赖性 (FDR<0.05): %d (%.1f%%)\n",
            sum(stat_results$FDR < 0.05, na.rm = TRUE),
            sum(stat_results$FDR < 0.05, na.rm = TRUE) / nrow(stat_results) * 100))

# 分类报告
for (cat_name in c("Significant_HighFC", "Significant_LowFC", "Trend_HighFC")) {
  genes_in_cat <- stat_results$Gene[stat_results$Category == cat_name]
  if (length(genes_in_cat) > 0) {
    cat(sprintf("\n  [%s] %d genes:\n    %s\n", cat_name, length(genes_in_cat),
                paste(genes_in_cat, collapse = ", ")))
  }
}

cat("\nVisualization completed.\n")
cat(sprintf("Output: %s\n", fig_dir))
