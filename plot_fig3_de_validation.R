##############################################################################
# Fig3: 时间序列 + 火山图 + 外部验证 + PPI Hub (参照 Nature 复现风格)
# - (A) 铁衰老评分时间序列折线图
# - (B) 微胶高铁衰老标志物火山图
# - (C) 外部验证森林图/棒棒糖
# - (D) PPI Hub 基因度中心性排名
##############################################################################

library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(ggsci)
library(viridis)

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")

theme_pub <- theme_bw(base_size = 11) +
  theme(
    panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.8),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10, color = "black"),
    plot.title = element_text(face = "bold", size = 13, hjust = 0),
    plot.tag = element_text(face = "bold", size = 16)
  )

# ============================================================================
# (A) 时间序列折线图 — 铁衰老评分动态
# ============================================================================
cat("[Fig3-A] Time series ferroaging score...\n")

ts_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/ssgsea_ferroaging_scores.csv"
if (!file.exists(ts_path)) {
  ts_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/gsva_ferroaging_scores.csv"
}
stopifnot(file.exists(ts_path))

ts <- read_csv(ts_path, show_col_types = FALSE)
cat(sprintf("  Time series: %d rows, cols: %s\n", nrow(ts), paste(colnames(ts), collapse=", ")))

# Identify time and group columns
ts_clean <- ts %>%
  filter(!is.na(group)) %>%
  mutate(group = as.character(group))

# If time info exists
if ("time" %in% colnames(ts_clean)) {
  ts_clean <- ts_clean %>%
    mutate(time_num = as.numeric(str_extract(as.character(time), "\\d+"))) %>%
    filter(!is.na(time_num))
}

# For ssGSEA data (dataset-based)
if ("dataset" %in% colnames(ts_clean) && length(unique(ts_clean$dataset)) > 1) {
  ds <- unique(ts_clean$dataset)[1]
  ts_sub <- ts_clean %>% filter(dataset == ds)

  p_ts <- ggplot(ts_sub, aes(x = group, y = Ferroaging_Score, fill = group)) +
    geom_boxplot(alpha = 0.6, outlier.size = 0.8, linewidth = 0.4) +
    geom_jitter(width = 0.15, alpha = 0.4, size = 0.8) +
    scale_fill_npg() +
    labs(x = NULL, y = "Ferroaging Score", tag = "A",
         title = paste("Ferroaging Score by Group -", ds)) +
    theme_pub
} else {
  # Simple time/group comparison
  p_ts <- ggplot(ts_clean, aes(x = group, y = Ferroaging_Score, fill = group)) +
    geom_boxplot(alpha = 0.6, outlier.size = 0.8, linewidth = 0.4) +
    scale_fill_npg() +
    labs(x = NULL, y = "Ferroaging Score", tag = "A",
         title = "Ferroaging Score by Condition") +
    theme_pub
}

ggsave(file.path(OUTDIR, "Fig3A_timeseries_ferroaging.png"), p_ts, width = 7, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig3A_timeseries_ferroaging.pdf"), p_ts, width = 7, height = 5)
cat("  -> Fig3A saved\n")

# ============================================================================
# (B) 火山图 — GSE61616 大鼠 MCAO vs Sham 差异基因 (Nature style)
# 关键修复：通过 GPL1355 探针→基因映射 + 大鼠→人类直系同源映射，
# 将探针ID正确转换为人类基因符号后再进行铁衰老基因高亮。
# ============================================================================
cat("[Fig3-B] Volcano plot from GSE61616 DE...\n")

de_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv"
stopifnot(file.exists(de_path))

de <- read_csv(de_path, show_col_types = FALSE)
cat(sprintf("  DEG: %d probes\n", nrow(de)))

# ---- 加载探针→基因映射 (GPL1355: Rat 230 2.0 Array) ----
gpl_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GPL1355_probe_to_gene.csv"
stopifnot(file.exists(gpl_path))
gpl1355 <- read_csv(gpl_path, show_col_types = FALSE)
cat(sprintf("  GPL1355 mapping: %d probes\n", nrow(gpl1355)))

# ---- 加载大鼠→人类直系同源映射 ----
ortho_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/rat_to_human_ortholog_mygene.csv"
if (file.exists(ortho_path)) {
  rat2human <- read_csv(ortho_path, show_col_types = FALSE) %>%
    select(rat_symbol, human_symbol) %>%
    distinct()
  cat(sprintf("  Ortholog mapping: %d rat->human pairs\n", nrow(rat2human)))
} else {
  rat2human <- data.frame(rat_symbol = character(), human_symbol = character())
  cat("  WARNING: rat_to_human_ortholog_mygene.csv not found\n")
}

# ---- 加载铁衰老基因列表 (人类符号) ----
fa_genes_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/ferroaging_genes_96.csv"
fa_genes <- if (file.exists(fa_genes_path)) {
  read_csv(fa_genes_path, show_col_types = FALSE)$gene_symbol
} else character(0)
cat(sprintf("  Ferroaging genes: %d\n", length(fa_genes)))

# ---- 探针→大鼠基因→人类直系同源 映射管道 ----
de <- de %>%
  left_join(gpl1355 %>% select(Probe, GeneSymbol), by = "Probe") %>%
  rename(rat_gene = GeneSymbol)

# 大鼠基因→人类直系同源：先查 mygene 缓存，失败则首字母大写回退
de$human_gene <- NA_character_
if (nrow(rat2human) > 0) {
  ortho_dict <- setNames(rat2human$human_symbol, rat2human$rat_symbol)
  de$human_gene <- ortho_dict[de$rat_gene]
}
na_mask <- is.na(de$human_gene) & !is.na(de$rat_gene)
de$human_gene[na_mask] <- toupper(de$rat_gene[na_mask])

n_mapped <- sum(!is.na(de$rat_gene))
n_ortho  <- sum(!is.na(de$human_gene))
cat(sprintf("  Probe->Gene: %d mapped, %d to human ortholog\n", n_mapped, n_ortho))

# ---- 构建火山图数据 ----
de_plot <- de %>%
  filter(!is.na(adj.P.Val), !is.na(logFC)) %>%
  mutate(
    neg_log10_p = -log10(adj.P.Val),
    significance = case_when(
      adj.P.Val < 0.01 & abs(logFC) > 1.5 ~ "padj<0.01 & |FC|>1.5",
      adj.P.Val < 0.05 & abs(logFC) > 0.8 ~ "padj<0.05 & |FC|>0.8",
      TRUE ~ "NS"
    ),
    # 使用人类直系同源符号匹配铁衰老基因
    is_fa = human_gene %in% fa_genes,
    # 标签：显著基因用基因符号(大写)，铁衰老基因用人类符号
    label = ifelse(
      (adj.P.Val < 0.001 & abs(logFC) > 1.5) | is_fa,
      coalesce(human_gene, rat_gene, Probe), ""
    )
  )

n_fa_highlighted <- sum(de_plot$is_fa, na.rm = TRUE)
cat(sprintf("  Ferroaging genes highlighted in volcano: %d\n", n_fa_highlighted))

p_volcano <- ggplot(de_plot, aes(x = logFC, y = neg_log10_p)) +
  geom_point(aes(color = significance), alpha = 0.5, size = 0.8) +
  geom_point(data = filter(de_plot, is_fa), aes(x = logFC, y = neg_log10_p),
             color = "#d72422", size = 1.2, alpha = 0.9) +
  geom_text_repel(
    aes(label = label),
    size = 2.8, max.overlaps = 30, box.padding = 0.3,
    segment.size = 0.2, color = "grey20"
  ) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey50", linewidth = 0.4) +
  scale_color_manual(values = c(
    "padj<0.01 & |FC|>1.5" = "#E41A1C",
    "padj<0.05 & |FC|>0.8" = "#377EB8",
    "NS" = "grey80"
  )) +
  labs(
    x = "log2 Fold Change (MCAO vs Sham)", y = "-log10(adjusted P-value)",
    tag = "B", title = "GSE61616: MCAO vs Sham — Ferroaging Genes Highlighted"
  ) +
  theme_pub +
  theme(legend.position.inside = c(0.14, 0.88), legend.title = element_blank())

ggsave(file.path(OUTDIR, "Fig3B_volcano_GSE61616.png"), p_volcano, width = 9, height = 7, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig3B_volcano_GSE61616.pdf"), p_volcano, width = 9, height = 7)
cat("  -> Fig3B saved\n")

# ============================================================================
# (B2) 微胶高铁衰老标志物补充图
# ============================================================================
cat("[Fig3-B2] Microglia high FA markers...\n")

mg_de_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/microglia_subcluster/microglia_high_fa_markers_annotated.csv"
stopifnot(file.exists(mg_de_path))

mg_de <- read_csv(mg_de_path, show_col_types = FALSE)
mg_de_plot <- mg_de %>%
  filter(!is.na(avg_log2FC), !is.na(p_val_adj)) %>%
  mutate(
    neg_log10_p = -log10(p_val_adj),
    significance = case_when(
      p_val_adj < 0.01 ~ "padj < 0.01",
      p_val_adj < 0.05 ~ "padj < 0.05",
      TRUE ~ "NS"
    ),
    in_fa96_label = ifelse(in_ferroaging_96, "In FA96", "Not in FA96")
  )

p_mg_volcano <- ggplot(mg_de_plot, aes(x = avg_log2FC, y = neg_log10_p)) +
  geom_point(aes(color = in_fa96_label), size = 4, alpha = 0.9) +
  geom_text_repel(
    aes(label = mouse_gene), size = 4, nudge_x = 0.1,
    fontface = "italic", segment.size = 0.4
  ) +
  scale_color_manual(values = c("In FA96" = "#d72422", "Not in FA96" = "#3A9AB2")) +
  labs(
    x = "avg log2FC (High vs Low FA Microglia)", y = "-log10(adjusted P-value)",
    tag = "B2", title = "Microglia: High vs Low Ferroaging Markers",
    color = ""
  ) +
  theme_pub +
  theme(legend.position.inside = c(0.2, 0.9))

ggsave(file.path(OUTDIR, "Fig3B2_microglia_markers.png"), p_mg_volcano, width = 7, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig3B2_microglia_markers.pdf"), p_mg_volcano, width = 7, height = 5)
cat("  -> Fig3B2 saved\n")

# ============================================================================
# (C) 外部验证森林图/棒棒糖 (参照 Nat Commun 哑铃图)
# ============================================================================
cat("[Fig3-C] External validation...\n")

ext_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/external_validation_results.csv"
stopifnot(file.exists(ext_path))

ext <- read_csv(ext_path, show_col_types = FALSE)
cat(sprintf("  External datasets: %s\n", paste(ext$Dataset, collapse=", ")))

# Fisher z-transform to compute 95% CI for Spearman rho
fisher_ci <- function(rho, n, alpha = 0.05) {
  z <- atanh(rho)
  se <- 1 / sqrt(n - 3)
  z_crit <- qnorm(1 - alpha / 2)
  list(lower = tanh(z - z_crit * se), upper = tanh(z + z_crit * se))
}

ext_plot <- ext %>%
  rowwise() %>%
  mutate(
    Dataset = factor(Dataset, levels = rev(Dataset)),
    metric = "Spearman rho",
    value = Spearman_rho,
    ci = list(fisher_ci(Spearman_rho, N_Valid)),
    lower = ci$lower,
    upper = ci$upper,
    sig = case_when(
      Spearman_p < 0.001 ~ "***",
      Spearman_p < 0.01  ~ "**",
      Spearman_p < 0.05  ~ "*",
      TRUE ~ "NS"
    )
  ) %>%
  ungroup()

cat(sprintf("  Fisher CI computed for %d datasets\n", nrow(ext_plot)))

p_ext <- ggplot(ext_plot, aes(x = value, y = Dataset)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey50", linewidth = 0.4) +
  geom_segment(aes(x = lower, xend = upper, y = Dataset, yend = Dataset), 
               color = "grey50", linewidth = 1.2, alpha = 0.6) +
  geom_point(aes(color = value, size = FA_AUC), alpha = 0.9) +
  geom_text(aes(label = sprintf("%.3f [%.2f,%.2f]%s", value, lower, upper, sig)), 
            hjust = -0.15, size = 3.2, fontface = "bold") +
  scale_color_viridis_c(option = "A", direction = -1, name = "Spearman rho") +
  scale_size_continuous(range = c(3, 8), name = "FA AUC") +
  scale_x_continuous(limits = c(min(ext_plot$lower) * 0.9, max(ext_plot$upper) * 1.25)) +
  labs(
    x = "Spearman Correlation (Predicted vs Observed FA Score)\n[95% CI via Fisher z-transform]", 
    y = NULL,
    tag = "C", title = "External Validation of Ferroaging Signature"
  ) +
  theme_pub

ggsave(file.path(OUTDIR, "Fig3C_external_validation.png"), p_ext, width = 8, height = 5, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig3C_external_validation.pdf"), p_ext, width = 8, height = 5)
cat("  -> Fig3C saved\n")

# ============================================================================
# (D) PPI Hub 基因度中心性排名 (参照 TCMNP degree_plot)
# ============================================================================
cat("[Fig3-D] PPI Hub gene ranking...\n")

ppi_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_topology.csv"
stopifnot(file.exists(ppi_path))

ppi <- read_csv(ppi_path, show_col_types = FALSE)
ppi_top <- ppi %>%
  arrange(desc(Degree)) %>%
  head(25) %>%
  mutate(Gene = factor(Gene, levels = rev(Gene)))

p_ppi <- ggplot(ppi_top, aes(x = Degree, y = Gene)) +
  geom_col(aes(fill = Betweenness), alpha = 0.85, width = 0.7) +
  scale_fill_viridis_c(option = "B", name = "Betweenness") +
  labs(x = "Degree Centrality", y = NULL,
       tag = "D", title = "PPI Network Hub Genes (Top 25)") +
  theme_pub +
  theme(axis.text.y = element_text(face = "italic", size = 9))

ggsave(file.path(OUTDIR, "Fig3D_PPI_hub_ranking.png"), p_ppi, width = 7, height = 7, dpi = 300)
ggsave(file.path(OUTDIR_PDF, "Fig3D_PPI_hub_ranking.pdf"), p_ppi, width = 7, height = 7)
cat("  -> Fig3D saved\n")

cat("[Fig3] All done!\n")
