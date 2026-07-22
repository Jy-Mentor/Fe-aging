# ===========================================================================
# 16_cross_omics_integration.R
# 铁衰老项目 - 跨组学整合: 代谢组(ST001637老化脑) × 转录组(GSE233815 MCAO FA-96)
#
# 目标:
#   1. 基因-代谢物通路轴交叉验证 (SAT1-多胺/ACSL4-脂质/HMOX1-铁/SLC1A5-谷氨酸等)
#   2. Signature 层面跨组学关联 (代谢 ferroptosis_score vs 转录 FA_96_UCell)
#   3. 跨组学证据汇总表 (基金/论文用)
#
# 数据源:
#   - ST001637: 老化小鼠脑代谢组 (3w/16w/59w, 10脑区, 1709代谢物, 2605样本)
#   - GSE233815: MCAO snRNA-seq + FA-96 UCell 评分 (Ctrl/1DPI/3DPI/7DPI)
#   - FA-96 基因集: L1/results/ferroaging_genes_96.csv (Liu 2026 + ACSL4)
# ===========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(pheatmap)
  library(RColorBrewer)
  library(tibble)
  library(stringr)
})

# ===========================================================================
# 0. 路径与输出配置
# ===========================================================================
BASE_DIR     <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
META_OUT     <- file.path(BASE_DIR, "output", "metabolomics")
META_TABLES  <- file.path(META_OUT, "tables")
SNRNA_DIR    <- "D:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn"
FA_GENE_CSV  <- "D:/铁衰老 绝不重蹈覆辙/L1/results/ferroaging_genes_96.csv"

OUT_DIR      <- file.path(BASE_DIR, "output", "cross_omics_integration")
FIG_DIR      <- file.path(OUT_DIR, "figures")
TAB_DIR      <- file.path(OUT_DIR, "tables")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR, showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# 1. 跨组学通路轴定义
#    FA-96 基因 → 代谢物底物/产物 → ST001637 老化方向 → GSE233815 MCAO 方向
# ===========================================================================
# 预期方向 (基于文献 + 铁衰老假说):
#   aging_axis (ST001637 59w vs 3w):
#     "UP"   = 老化时升高 (符合铁衰老激活)
#     "DOWN" = 老化时下降
#   mcao_axis (GSE233815 7DPI vs Ctrl):
#     "UP"   = MCAO 后升高 (符合铁死亡激活)
#     "DOWN" = MCAO 后下降

CROSS_OMICS_AXES <- list(

  list(
    axis_name        = "SAT1-polyamine",
    gene_symbol      = "SAT1",
    gene_function    = "spermidine/spermine N1-acetyltransferase",
    metabolites      = c("Spermidine", "Spermine", "Putrescine", "Ornithine"),
    met_category     = "polyamine_metabolism",
    expected_aging   = "DOWN",
    expected_mcao    = "UP",
    evidence_note    = "SAT1激活→精胺/亚精胺乙酰化→耗竭；MCAO后SAT1预期上调"
  ),

  list(
    axis_name        = "ACSL4-lipid_signaling",
    gene_symbol      = "ACSL4",
    gene_function    = "acyl-CoA synthetase long-chain 4 (PUFA-biased)",
    metabolites      = c("AA", "PA (16:0)", "SA (18:0)", "EPA (20:5)",
                         "LPC", "PE", "Ceramide"),
    met_category     = "lipid_signaling",
    expected_aging   = "UP",
    expected_mcao    = "UP",
    evidence_note    = "ACSL4激活→PUFA-磷脂重塑→脂质过氧化底物蓄积"
  ),

  list(
    axis_name        = "HMOX1-iron_metabolism",
    gene_symbol      = "HMOX1",
    gene_function    = "heme oxygenase 1 (Fe2+ release)",
    metabolites      = c("Heme", "Bilirubin", "Biliverdin", "PPIX"),
    met_category     = "iron_metabolism",
    expected_aging   = "UP",
    expected_mcao    = "UP",
    evidence_note    = "HMOX1→heme降解→Fe2+释放→铁过载"
  ),

  list(
    axis_name        = "SLC1A5-glutamate_cysteine",
    gene_symbol      = "SLC1A5",
    gene_function    = "glutamine transporter (xCT partner)",
    metabolites      = c("Glutamate", "Cysteine", "GSH (reduced)",
                         "Glutamine", "Methionine"),
    met_category     = "antioxidant_defense",
    expected_aging   = "DOWN",
    expected_mcao    = "DOWN",
    evidence_note    = "SLC1A5/SLC7A11抑制→胱氨酸摄取↓→GSH合成↓"
  ),

  list(
    axis_name        = "PTGS2-arachidonic_cascade",
    gene_symbol      = "PTGS2",
    gene_function    = "COX-2 (prostaglandin endoperoxide synthase)",
    metabolites      = c("AA", "EPA", "Adrenic acid"),
    met_category     = "lipid_signaling",
    expected_aging   = "UP",
    expected_mcao    = "UP",
    evidence_note    = "PTGS2=COX2激活→AA代谢→PGE2促炎"
  ),

  list(
    axis_name        = "NAMPT-NAD_metabolism",
    gene_symbol      = "NAMPT",
    gene_function    = "nicotinamide phosphoribosyltransferase (NAD+ salvage)",
    metabolites      = c("NAD+", "NAM", "NMN", "NR"),
    met_category     = "aging_related",
    expected_aging   = "DOWN",
    expected_mcao    = "DOWN",
    evidence_note    = "衰老→NAMPT↓→NAD+耗竭→线粒体功能障碍 (需NAMPT存在于FA-96或旁基因)"
  ),

  list(
    axis_name        = "HIF1A-hypoxia",
    gene_symbol      = "HIF1A",
    gene_function    = "hypoxia inducible factor 1 subunit alpha",
    metabolites      = c("Lactate", "Pyruvate", "Succinate", "Fumarate"),
    met_category     = "iron_metabolism",
    expected_aging   = "UP",
    expected_mcao    = "UP",
    evidence_note    = "HIF1A激活→糖酵解↑→乳酸蓄积；TCA中间产物累积"
  ),

  list(
    axis_name        = "IL6-inflammation",
    gene_symbol      = "IL6",
    gene_function    = "interleukin 6 (pro-inflammatory)",
    metabolites      = c("Taurine", "Hypotaurine", "Cysteine"),
    met_category     = "antioxidant_defense",
    expected_aging   = "DOWN",
    expected_mcao    = "UP",
    evidence_note    = "IL6激活→炎症→抗氧化储备耗竭"
  ),

  list(
    axis_name        = "NOX4-oxidative_stress",
    gene_symbol      = "NOX4",
    gene_function    = "NADPH oxidase 4 (ROS source)",
    metabolites      = c("4-HNE", "GSSG", "GSH (reduced)", "8-iso-PGF2alpha"),
    met_category     = "lipid_peroxidation",
    expected_aging   = "UP",
    expected_mcao    = "UP",
    evidence_note    = "NOX4→ROS↑→脂质过氧化↑→4-HNE/MDA蓄积"
  ),

  list(
    axis_name        = "KEAP1-NRF2_antioxidant",
    gene_symbol      = "KEAP1",
    gene_function    = "kelch-like ECH-associated protein 1 (NRF2 repressor)",
    metabolites      = c("GSH (reduced)", "Cysteine", "Taurine", "Ascorbate"),
    met_category     = "antioxidant_defense",
    expected_aging   = "DOWN",
    expected_mcao    = "UP",
    evidence_note    = "KEAP1↑→NRF2降解→抗氧化应答↓"
  )
)

# ===========================================================================
# 2. 加载 ST001637 代谢组数据 (老化)
# ===========================================================================
load_metabolomics_stats <- function() {
  pairwise_path <- file.path(META_TABLES, "metabolite_pairwise_comparison.csv")
  signature_path <- file.path(META_TABLES, "ferroptosis_signature_scores.csv")

  if (!file.exists(pairwise_path)) {
    stop("ST001637 代谢组统计文件不存在: ", pairwise_path,
         "\n请先运行 14_metabolomics_ferroptosis.R")
  }

  pairwise <- read.csv(pairwise_path, stringsAsFactors = FALSE)
  signature <- read.csv(signature_path, stringsAsFactors = FALSE)

  message(sprintf("[ST001637] 成对比较: %d 条, signature: %d 样本",
                  nrow(pairwise), nrow(signature)))

  # 老化比较: 3 weeks vs 59 weeks (最大跨度比较)
  aging_pairwise <- pairwise %>%
    filter((group1 == "3 weeks" & group2 == "59 weeks") |
           (group1 == "59 weeks" & group2 == "3 weeks")) %>%
    mutate(
      log2FC_aging = ifelse(group1 == "3 weeks", log2FC, -log2FC),
      direction_aging = ifelse(log2FC_aging > 0, "UP", "DOWN")
    )

  message(sprintf("[ST001637] 老化比较 (3w vs 59w): %d 条",
                  nrow(aging_pairwise)))

  list(
    pairwise_all    = pairwise,
    aging_pairwise  = aging_pairwise,
    signature       = signature
  )
}

# ===========================================================================
# 3. 加载 GSE233815 FA-96 转录组数据
# ===========================================================================
load_transcriptomic_fa96 <- function() {
  cell_meta_path <- file.path(SNRNA_DIR, "cell_metadata_with_ferroaging_score.csv")
  by_cond_path <- file.path(SNRNA_DIR, "ferroaging_score_by_condition.csv")
  by_cond_cc_path <- file.path(SNRNA_DIR, "ferroaging_score_by_condition_cellclass.csv")

  if (!file.exists(cell_meta_path)) {
    stop("GSE233815 FA-96 评分文件不存在: ", cell_meta_path,
         "\n请先运行 compute_ferroaging_score_gse233815.R")
  }

  cell_meta <- read.csv(cell_meta_path, stringsAsFactors = FALSE)
  by_cond <- read.csv(by_cond_path, stringsAsFactors = FALSE)
  by_cond_cc <- read.csv(by_cond_cc_path, stringsAsFactors = FALSE)

  message(sprintf("[GSE233815] 细胞数: %d, 条件: %d",
                  nrow(cell_meta), nrow(by_cond)))

  list(
    cell_meta      = cell_meta,
    by_condition   = by_cond,
    by_cond_cc     = by_cond_cc
  )
}

# ===========================================================================
# 4. 加载 FA-96 基因集
# ===========================================================================
load_fa96_genes <- function() {
  if (!file.exists(FA_GENE_CSV)) {
    stop("FA-96 基因集文件不存在: ", FA_GENE_CSV)
  }
  fa_genes <- read.csv(FA_GENE_CSV, stringsAsFactors = FALSE)
  message(sprintf("[FA-96] 基因数: %d", nrow(fa_genes)))
  fa_genes
}

# ===========================================================================
# 5. 通路轴交叉验证
#    对每个 axis: 提取 ST001637 老化方向 vs 文献预期方向 vs MCAO 预期方向
# ===========================================================================
build_axis_validation <- function(axes, met_stats) {
  aging_stats <- met_stats$aging_pairwise

  results <- list()

  for (axis in axes) {
    gene <- axis$gene_symbol
    mets <- axis$metabolites
    expected_aging <- axis$expected_aging
    expected_mcao <- axis$expected_mcao

    axis_met_stats <- aging_stats %>%
      filter(display_name %in% mets)

    if (nrow(axis_met_stats) == 0) {
      message(sprintf("  [跳过] %s (%s): ST001637 未匹配到代谢物",
                      axis$axis_name, gene))
      next
    }

    for (i in seq_len(nrow(axis_met_stats))) {
      row <- axis_met_stats[i, ]
      actual_aging <- row$direction_aging
      match_aging <- actual_aging == expected_aging

      results[[length(results) + 1]] <- data.frame(
        axis_name           = axis$axis_name,
        gene_symbol         = gene,
        gene_function       = axis$gene_function,
        metabolite          = row$metabolite,
        display_name        = row$display_name,
        met_category        = row$category,
        log2FC_aging        = round(row$log2FC_aging, 4),
        p_adj_aging         = row$p_adj,
        direction_aging     = actual_aging,
        expected_aging      = expected_aging,
        match_aging         = match_aging,
        expected_mcao       = expected_mcao,
        evidence_note       = axis$evidence_note,
        stringsAsFactors    = FALSE
      )
    }
  }

  if (length(results) == 0) {
    stop("所有通路轴均未匹配到代谢物, 检查 display_name 命名")
  }

  axis_table <- do.call(rbind, results)
  axis_table
}

# ===========================================================================
# 6. 跨组学 signature 层面整合
#    代谢 ferroptosis_score (ST001637 老化) vs FA_96_UCell (GSE233815 MCAO)
# ===========================================================================
build_signature_integration <- function(met_stats, snrna_data) {
  # 6.1 代谢 signature - 按 Age 聚合
  met_sig <- met_stats$signature %>%
    filter(!is.na(Age)) %>%
    group_by(Age) %>%
    summarise(
      n_samples_metab    = n(),
      mean_metab_score   = mean(ferroptosis_score, na.rm = TRUE),
      sd_metab_score     = sd(ferroptosis_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      zscore_metab = (mean_metab_score - mean(mean_metab_score)) /
                      sd(mean_metab_score)
    ) %>%
    arrange(Age)

  message("[Signature] 代谢 signature 按 Age:")
  print(met_sig)

  # 6.2 转录 signature - 按 Condition 聚合
  rna_sig <- snrna_data$by_condition %>%
    mutate(
      zscore_rna = (mean_score - mean(mean_score)) / sd(mean_score)
    ) %>%
    arrange(match(Condition, c("Ctrl", "1DPI", "3DPI", "7DPI")))

  message("[Signature] 转录 signature 按 Condition:")
  print(rna_sig)

  list(
    metabolic = met_sig,
    rna       = rna_sig
  )
}

# ===========================================================================
# 7. 跨组学证据汇总表 (基金/论文用)
# ===========================================================================
build_cross_omics_evidence <- function(axis_table, sig_int) {
  met_sig <- sig_int$metabolic
  rna_sig <- sig_int$rna

  # 老化 signature 趋势
  met_3w <- met_sig$mean_metab_score[met_sig$Age == "3 weeks"]
  met_59w <- met_sig$mean_metab_score[met_sig$Age == "59 weeks"]
  met_trend <- ifelse(met_59w > met_3w, "UP (aging)", "DOWN (aging)")

  # MCAO signature 趋势 (7DPI vs Ctrl)
  rna_ctrl <- rna_sig$mean_score[rna_sig$Condition == "Ctrl"]
  rna_7dpi <- rna_sig$mean_score[rna_sig$Condition == "7DPI"]
  rna_trend <- ifelse(rna_7dpi > rna_ctrl, "UP (MCAO)", "DOWN (MCAO)")

  # 通路轴汇总
  axis_summary <- axis_table %>%
    group_by(axis_name, gene_symbol, expected_aging, expected_mcao) %>%
    summarise(
      n_metabolites_tested   = n(),
      n_significant_aging    = sum(p_adj_aging < 0.05),
      n_match_expected       = sum(match_aging),
      mean_log2FC_aging      = round(mean(log2FC_aging, na.rm = TRUE), 4),
      direction_consistent   = all(match_aging),
      .groups = "drop"
    ) %>%
    mutate(
      signature_metab_trend  = met_trend,
      signature_rna_trend    = rna_trend,
      overall_evidence = case_when(
        direction_consistent & n_significant_aging >= 1 ~ "Strong",
        n_match_expected >= ceiling(n_metabolites_tested / 2) ~ "Moderate",
        TRUE ~ "Weak"
      )
    )

  axis_summary
}

# ===========================================================================
# 8. 可视化
# ===========================================================================
plot_axis_heatmap <- function(axis_table) {
  plot_data <- axis_table %>%
    mutate(
      significance = case_when(
        p_adj_aging < 0.001 ~ "***",
        p_adj_aging < 0.01  ~ "**",
        p_adj_aging < 0.05  ~ "*",
        TRUE ~ "ns"
      ),
      label = sprintf("%+.2f%s", log2FC_aging, significance)
    )

  p <- ggplot(plot_data, aes(
    x = axis_name,
    y = display_name,
    fill = log2FC_aging
  )) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = label), size = 3.2, color = "black") +
    scale_fill_gradient2(
      low  = "#2166AC",
      mid  = "white",
      high = "#B2182B",
      midpoint = 0,
      name = "log2FC\n(59w vs 3w)"
    ) +
    facet_grid(met_category ~ ., scales = "free_y", space = "free_y") +
    labs(
      title = "跨组学通路轴: 代谢物在老化(ST001637 59w vs 3w)中的变化",
      x = "通路轴 (FA-96 基因 → 代谢物)",
      y = "代谢物"
    ) +
    theme_bw(base_size = 11) +
    theme(
      axis.text.x = element_text(angle = 35, hjust = 1, size = 9),
      strip.background = element_rect(fill = "grey90"),
      panel.grid = element_blank()
    )

  ggsave(file.path(FIG_DIR, "cross_omics_axis_heatmap.pdf"),
         p, width = 11, height = 9, dpi = 300)
  message("通路轴热图已保存")
}

plot_signature_trajectory <- function(sig_int) {
  met_sig <- sig_int$metabolic %>%
    mutate(
      Age = factor(Age, levels = c("3 weeks", "16 weeks", "59 weeks")),
      modality = "Metabolomics (ST001637)"
    ) %>%
    rename(score = mean_metab_score, zscore = zscore_metab)

  rna_sig <- sig_int$rna %>%
    mutate(
      Condition = factor(Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI")),
      modality = "snRNA-seq FA-96 (GSE233815)"
    ) %>%
    rename(score = mean_score, zscore = zscore_rna) %>%
    mutate(Age = as.character(Condition)) %>%
    select(Age, score, zscore, modality)

  combined <- bind_rows(
    met_sig %>% select(Age, score, zscore, modality),
    rna_sig %>% select(Age, score, zscore, modality)
  )

  p1 <- ggplot(combined, aes(
    x = factor(Age, levels = c("3 weeks", "16 weeks", "59 weeks",
                                "Ctrl", "1DPI", "3DPI", "7DPI")),
    y = zscore,
    color = modality,
    group = modality
  )) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3) +
    scale_color_manual(values = c("#1F78B4", "#E31A1C")) +
    labs(
      title = "跨组学 signature 轨迹 (Z-score)",
      subtitle = "左: 老化时序 (代谢组) | 右: MCAO 时序 (转录组)",
      x = "时间点",
      y = "Z-score (跨数据集)"
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom")

  ggsave(file.path(FIG_DIR, "cross_omics_signature_trajectory.pdf"),
         p1, width = 9, height = 6, dpi = 300)
  message("Signature 轨迹图已保存")
}

plot_evidence_summary <- function(evidence_table) {
  plot_data <- evidence_table %>%
    mutate(
      axis_name = factor(axis_name,
                         levels = axis_name[order(-n_match_expected)])
    )

  p <- ggplot(plot_data, aes(
    x = axis_name,
    y = n_match_expected / n_metabolites_tested,
    fill = overall_evidence
  )) +
    geom_col(color = "black", linewidth = 0.4) +
    scale_fill_manual(
      values = c("Strong" = "#1A9850", "Moderate" = "#FDAE61", "Weak" = "#D73027"),
      name = "证据强度"
    ) +
    geom_text(aes(label = sprintf("%d/%d", n_match_expected, n_metabolites_tested)),
              vjust = -0.4, size = 3.4) +
    labs(
      title = "跨组学通路轴证据强度",
      subtitle = "绿色=一致 Strong | 橙色=中等 Moderate | 红色=弱 Weak",
      x = "通路轴",
      y = "老化方向匹配比例 (匹配数/测试数)"
    ) +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  ggsave(file.path(FIG_DIR, "cross_omics_evidence_summary.pdf"),
         p, width = 10, height = 6, dpi = 300)
  message("证据汇总图已保存")
}

plot_dual_modality_panel <- function(sig_int) {
  met_sig <- sig_int$metabolic %>%
    mutate(
      Age = factor(Age, levels = c("3 weeks", "16 weeks", "59 weeks")),
      modality = "Metabolomics\n(ST001637 老化)"
    )

  rna_sig <- sig_int$rna %>%
    mutate(
      Condition = factor(Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI")),
      modality = "snRNA-seq FA-96\n(GSE233815 MCAO)"
    )

  p1 <- ggplot(met_sig, aes(x = Age, y = mean_metab_score, fill = Age)) +
    geom_col(color = "black", linewidth = 0.4, alpha = 0.85) +
    geom_errorbar(aes(ymin = mean_metab_score - sd_metab_score,
                      ymax = mean_metab_score + sd_metab_score),
                  width = 0.2) +
    scale_fill_brewer(palette = "YlOrRd") +
    labs(
      title = "A. 代谢 ferroptosis signature (老化)",
      x = "年龄",
      y = "Mean metabolite signature score"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1))

  p2 <- ggplot(rna_sig, aes(x = Condition, y = mean_score, fill = Condition)) +
    geom_col(color = "black", linewidth = 0.4, alpha = 0.85) +
    geom_errorbar(aes(ymin = mean_score - sd_score,
                      ymax = mean_score + sd_score),
                  width = 0.2) +
    scale_fill_brewer(palette = "Blues") +
    labs(
      title = "B. FA-96 UCell signature (MCAO)",
      x = "Condition",
      y = "Mean FA-96 UCell score"
    ) +
    theme_bw(base_size = 11) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 30, hjust = 1))

  combined <- ggpubr::ggarrange(p1, p2, ncol = 2, align = "h")

  ggsave(file.path(FIG_DIR, "cross_omics_dual_modality.pdf"),
         combined, width = 12, height = 5.5, dpi = 300)
  message("双模态对照图已保存")
}

# ===========================================================================
# 9. 主流程
# ===========================================================================
main <- function() {
  message("========================================")
  message("跨组学整合: 代谢组(老化) × 转录组(MCAO FA-96)")
  message("========================================")

  message("\n[1/6] 加载 ST001637 代谢组数据...")
  met_stats <- load_metabolomics_stats()

  message("\n[2/6] 加载 GSE233815 FA-96 转录组数据...")
  snrna_data <- load_transcriptomic_fa96()

  message("\n[3/6] 加载 FA-96 基因集...")
  fa_genes <- load_fa96_genes()
  message(sprintf("  FA-96 基因数: %d", nrow(fa_genes)))

  message("\n[4/6] 构建通路轴交叉验证...")
  axis_table <- build_axis_validation(CROSS_OMICS_AXES, met_stats)

  write.csv(axis_table,
            file.path(TAB_DIR, "cross_omics_axis_table.csv"),
            row.names = FALSE)

  n_axes <- length(unique(axis_table$axis_name))
  n_match <- sum(axis_table$match_aging)
  message(sprintf("  通路轴数: %d", n_axes))
  message(sprintf("  代谢物测试数: %d", nrow(axis_table)))
  message(sprintf("  方向匹配数: %d (%.1f%%)",
                  n_match, 100 * n_match / nrow(axis_table)))

  message("\n[5/6] 跨组学 signature 整合...")
  sig_int <- build_signature_integration(met_stats, snrna_data)

  message("\n构建证据汇总表...")
  evidence_table <- build_cross_omics_evidence(axis_table, sig_int)

  write.csv(evidence_table,
            file.path(TAB_DIR, "cross_omics_evidence_summary.csv"),
            row.names = FALSE)

  write.csv(sig_int$metabolic,
            file.path(TAB_DIR, "metabolic_signature_by_age.csv"),
            row.names = FALSE)

  write.csv(sig_int$rna,
            file.path(TAB_DIR, "fa96_signature_by_condition.csv"),
            row.names = FALSE)

  message("\n[6/6] 生成可视化...")
  plot_axis_heatmap(axis_table)
  plot_signature_trajectory(sig_int)
  plot_evidence_summary(evidence_table)
  plot_dual_modality_panel(sig_int)

  message("\n========================================")
  message("跨组学整合完成!")
  message(sprintf("  输出目录: %s", OUT_DIR))
  message(sprintf("  图表: %s", FIG_DIR))
  message(sprintf("  表格: %s", TAB_DIR))
  message("========================================")

  results <- list(
    axis_table       = axis_table,
    evidence_table   = evidence_table,
    signature_integration = sig_int,
    fa_genes         = fa_genes
  )

  saveRDS(results, file.path(OUT_DIR, "cross_omics_integration_results.rds"))

  message("\n关键发现:")
  strong_axes <- evidence_table %>%
    filter(overall_evidence == "Strong") %>%
    pull(axis_name)
  if (length(strong_axes) > 0) {
    message("  Strong 证据轴: ", paste(strong_axes, collapse = ", "))
  }
  moderate_axes <- evidence_table %>%
    filter(overall_evidence == "Moderate") %>%
    pull(axis_name)
  if (length(moderate_axes) > 0) {
    message("  Moderate 证据轴: ", paste(moderate_axes, collapse = ", "))
  }

  invisible(results)
}

# ===========================================================================
# 10. 执行
# ===========================================================================
results <- main()
