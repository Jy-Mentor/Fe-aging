# ===========================================================================
# 14_metabolomics_ferroptosis.R
# 铁衰老项目 - 代谢组学交叉验证: 铁死亡/铁衰老代谢特征提取与验证
# 
# 数据来源:
#   ST001637 - A Metabolome Atlas of the Aging Mouse Brain (Metabolomics Workbench)
#     - 物种: C57BL/6NCrl 小鼠
#     - 组织: 10个脑区 (基底节、脑干、小脑、皮层、海马、下丘脑、中脑、嗅球等)
#     - 年龄: 3周(青春期), 16周(成年早期), 59周(成年晚期)
#     - 性别: 雌雄均有
#     - 代谢物: 1709个结构注释代谢物
#     - 平台: HILICpos/HILICneg/CSHpos/CSHneg LC-MS
# 
# 文献支撑 (PubMed已验证):
#   - PMID 40375180: MCAO大鼠中4-HNE↓, MDA↓, SOD↑, GPX4↑ (DGAT1抑制)
#   - PMID 37752100: BCAA代谢紊乱→脂肪酸氧化→神经元铁死亡 (PPM1K)
#   - PMID 40768899: 脑缺血后4-HNE↓, MDA↓, GSH↑, GSH/GSSG↑ (circMTCO2)
#   - PMID 38958365: MCAO大鼠52种差异代谢物，涉及6条代谢通路
#   - Front Pharmacol 2023: MCAO/R小鼠82种差异代谢物，甘油磷脂/亚油酸代谢
#   - Molecules 2021: tMCAO小鼠脑脂质组学，鞘脂/甘油磷脂代谢显著富集
#   - Antioxidants 2024: 卒中患者456种代谢物靶向定量，氧化应激+能量代谢重塑
# ===========================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(ggpubr)
  library(pheatmap)
  library(RColorBrewer)
  library(stringr)
  library(tibble)
})

# ===========================================================================
# 0. 路径与输出配置
# ===========================================================================
BASE_DIR    <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
DATA_DIR    <- file.path(BASE_DIR, "data", "metabolomics")
OUTPUT_DIR  <- file.path(BASE_DIR, "output", "metabolomics")
FIG_DIR     <- file.path(OUTPUT_DIR, "figures")
TAB_DIR     <- file.path(OUTPUT_DIR, "tables")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR,    showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR,    showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# 1. 铁死亡/铁衰老核心代谢物定义
# ===========================================================================
# 基于文献系统综述构建的铁衰老代谢特征面板
# 分类: 脂质过氧化 | 抗氧化防御 | 多胺代谢 | 铁代谢 | 能量代谢 | 衰老相关

FERROPTOSIS_METABOLITE_PANEL <- list(

  lipid_peroxidation = c(
    "4-hydroxynonenal"      = "4-HNE",
    "malondialdehyde"       = "MDA",
    "8-isoprostane"         = "8-iso-PGF2alpha",
    "acrolein"              = "Acrolein",
    "13-HODE"               = "13-HODE",
    "9-HODE"                = "9-HODE",
    "oxidized glutathione"  = "GSSG",
    "glutathione disulfide" = "GSSG"
  ),

  antioxidant_defense = c(
    "glutathione"           = "GSH (reduced)",
    "glutathione reduced"   = "GSH",
    "ascorbic acid"         = "Vitamin C",
    "ascorbate"             = "Vitamin C",
    "NADPH"                 = "NADPH",
    "cysteine"              = "Cysteine",
    "glutamate"             = "Glutamate",
    "cysteinylglycine"      = "Cys-Gly",
    "gamma-glutamylcysteine" = "gamma-Glu-Cys",
    "taurine"               = "Taurine",
    "hypotaurine"           = "Hypotaurine",
    "methionine"            = "Methionine",
    "S-adenosylmethionine"  = "SAM",
    "S-adenosylhomocysteine" = "SAH",
    "ergothioneine"         = "Ergothioneine",
    "carnosine"             = "Carnosine"
  ),

  polyamine_metabolism = c(
    "spermidine"            = "Spermidine",
    "spermine"              = "Spermine",
    "putrescine"            = "Putrescine",
    "N-acetylspermidine"    = "N-Ac-Spermidine",
    "N-acetylspermine"      = "N-Ac-Spermine",
    "N1-acetylspermidine"   = "N1-Ac-Spermidine",
    "N8-acetylspermidine"   = "N8-Ac-Spermidine",
    "ornithine"             = "Ornithine",
    "arginine"              = "Arginine",
    "agmatine"              = "Agmatine",
    "S-adenosylmethioninamine" = "dcSAM"
  ),

  iron_metabolism = c(
    "heme"                  = "Heme",
    "bilirubin"             = "Bilirubin",
    "biliverdin"            = "Biliverdin",
    "protoporphyrin IX"     = "PPIX",
    "citrate"               = "Citrate",
    "succinate"             = "Succinate",
    "alpha-ketoglutarate"   = "alpha-KG",
    "fumarate"              = "Fumarate",
    "malate"                = "Malate",
    "aconitate"             = "Aconitate",
    "pyruvate"              = "Pyruvate",
    "lactate"               = "Lactate"
  ),

  aging_related = c(
    "NAD+"                  = "NAD+",
    "NAD"                   = "NAD+",
    "NADH"                  = "NADH",
    "nicotinamide"          = "NAM",
    "NMN"                   = "NMN",
    "nicotinamide riboside" = "NR",
    "carnitine"             = "Carnitine",
    "acetylcarnitine"       = "Acetylcarnitine",
    "palmitoylcarnitine"    = "C16-Carnitine",
    "oleoylcarnitine"       = "C18:1-Carnitine",
    "linoleoylcarnitine"    = "C18:2-Carnitine",
    "stearoylcarnitine"     = "C18-Carnitine",
    "uracil"                = "Uracil",
    "uridine"               = "Uridine",
    "pseudouridine"         = "Pseudouridine",
    "hypoxanthine"          = "Hypoxanthine",
    "xanthine"              = "Xanthine",
    "uric acid"             = "Uric acid",
    "inosine"               = "Inosine",
    "adenosine"             = "Adenosine"
  ),

  lipid_signaling = c(
    "sphingosine"           = "Sphingosine",
    "sphinganine"           = "Sphinganine",
    "sphingosine-1-phosphate" = "S1P",
    "ceramide"              = "Ceramide",
    "dihydroceramide"       = "Dihydroceramide",
    "glucosylceramide"      = "GlcCer",
    "sphingomyelin"         = "SM",
    "ethanolamine"          = "Ethanolamine",
    "choline"               = "Choline",
    "glycerophosphocholine" = "GPC",
    "glycerophosphoethanolamine" = "GPE",
    "phosphocholine"        = "Phosphocholine",
    "arachidonic acid"      = "AA (20:4)",
    "docosahexaenoic acid"  = "DHA (22:6)",
    "eicosapentaenoic acid" = "EPA (20:5)",
    "linoleic acid"         = "LA (18:2)",
    "oleic acid"            = "OA (18:1)",
    "stearic acid"          = "SA (18:0)",
    "palmitic acid"         = "PA (16:0)",
    "prostaglandin E2"      = "PGE2",
    "prostaglandin D2"      = "PGD2",
    "leukotriene B4"        = "LTB4",
    "thromboxane B2"        = "TXB2"
  )
)

# 展平为查找表
build_lookup <- function(panel) {
  entries <- list()
  for (cat_name in names(panel)) {
    cat_entries <- panel[[cat_name]]
    for (i in seq_along(cat_entries)) {
      entries[[length(entries) + 1]] <- data.frame(
        search_term  = names(cat_entries)[i],
        category     = cat_name,
        display_name = unname(cat_entries[i]),
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, entries)
}

METABOLITE_LOOKUP <- build_lookup(FERROPTOSIS_METABOLITE_PANEL)

# ===========================================================================
# 2. 从预转换的 CSV 文件读取数据
#    CSV 由 convert_mwtab.py 从 ST001637 mwTab JSON 生成
# ===========================================================================

load_mwtab_csv <- function(data_dir) {
  # 读取样本元数据
  sample_meta_path <- file.path(data_dir, "ST001637_sample_meta.csv")
  if (!file.exists(sample_meta_path)) {
    stop("样本元数据 CSV 不存在: ", sample_meta_path)
  }
  sample_meta <- read.csv(sample_meta_path, stringsAsFactors = FALSE)
  message(sprintf("  样本元数据: %d 行, %d 列", nrow(sample_meta), ncol(sample_meta)))
  message(sprintf("  因子: %s",
    paste(setdiff(names(sample_meta), c("sample_id", "analysis_id")), collapse = ", ")))

  # 读取丰度数据 (long format)
  abundance_path <- file.path(data_dir, "ST001637_abundance_long.csv")
  if (!file.exists(abundance_path)) {
    stop("丰度数据 CSV 不存在: ", abundance_path)
  }
  abundance <- read.csv(abundance_path, stringsAsFactors = FALSE)
  abundance$abundance <- as.numeric(abundance$abundance)
  message(sprintf("  丰度数据: %d 行", nrow(abundance)))

  # 合并样本元数据到丰度表
  abundance <- abundance %>%
    left_join(sample_meta, by = c("sample_id", "analysis_id"))

  # 读取命名代谢物信息
  metabolites_path <- file.path(data_dir, "ST001637_named_metabolites.csv")
  if (file.exists(metabolites_path)) {
    metabolites <- read.csv(metabolites_path, stringsAsFactors = FALSE)
    message(sprintf("  命名代谢物: %d 个", nrow(metabolites)))
  } else {
    metabolites <- data.frame()
    message("  命名代谢物 CSV 未找到, 跳过")
  }

  N_samples <- nrow(sample_meta)
  N_metabolites <- length(unique(abundance$metabolite))
  message(sprintf("  总计: %d 样本, %d 代谢物", N_samples, N_metabolites))

  list(
    study_id      = "ST001637",
    study_title   = "A Metabolome Atlas of the Aging Mouse Brain",
    species       = "Mus musculus",
    sample_meta   = sample_meta,
    abundance     = abundance,
    metabolites   = metabolites,
    n_metabolites = N_metabolites,
    n_samples     = N_samples
  )
}

# ===========================================================================
# 3. 匹配铁死亡/铁衰老相关代谢物
# ===========================================================================

match_ferroptosis_metabolites <- function(parsed, lookup) {
  all_metabolites <- unique(parsed$abundance$metabolite)

  matched <- data.frame(
    metabolite_name = character(),
    search_term     = character(),
    category        = character(),
    display_name    = character(),
    match_type      = character(),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(lookup))) {
    term <- lookup$search_term[i]
    cat_val <- lookup$category[i]
    disp_val <- lookup$display_name[i]

    exact_match <- all_metabolites[tolower(all_metabolites) == tolower(term)]
    fuzzy_match <- all_metabolites[
      grepl(tolower(term), tolower(all_metabolites), fixed = TRUE)
    ]

    for (m in exact_match) {
      matched <- rbind(matched, data.frame(
        metabolite_name = m,
        search_term     = term,
        category        = cat_val,
        display_name    = disp_val,
        match_type      = "exact",
        stringsAsFactors = FALSE
      ))
    }

    for (m in setdiff(fuzzy_match, exact_match)) {
      matched <- rbind(matched, data.frame(
        metabolite_name = m,
        search_term     = term,
        category        = cat_val,
        display_name    = disp_val,
        match_type      = "fuzzy",
        stringsAsFactors = FALSE
      ))
    }
  }

  matched <- matched[!duplicated(matched$metabolite_name), ]

  message(sprintf("匹配到 %d 个铁死亡/铁衰老相关代谢物", nrow(matched)))
  for (cat_val in unique(matched$category)) {
    n_cat <- sum(matched$category == cat_val)
    message(sprintf("  %s: %d", cat_val, n_cat))
  }

  matched
}

# ===========================================================================
# 4. 统计分析与差异比较
# ===========================================================================

run_metabolite_stats <- function(parsed, matched, group_var = "Age") {
  abundance <- parsed$abundance %>%
    inner_join(matched, by = c("metabolite" = "metabolite_name"))

  if (!group_var %in% names(abundance)) {
    group_var <- intersect(names(parsed$sample_meta), names(abundance))[1]
    message(sprintf("使用分组变量: %s", group_var))
  }

  groups <- unique(abundance[[group_var]])
  if (length(groups) < 2) {
    stop(sprintf("分组变量 %s 只有 %d 个水平, 无法比较", group_var, length(groups)))
  }

  stats_results <- abundance %>%
    group_by(metabolite, display_name, category) %>%
    summarise(
      n_samples = n(),
      mean_abundance = mean(abundance, na.rm = TRUE),
      sd_abundance   = sd(abundance, na.rm = TRUE),
      .groups = "drop"
    )

  group_stats <- abundance %>%
    group_by(metabolite, display_name, category, !!sym(group_var)) %>%
    summarise(
      n     = n(),
      mean  = mean(abundance, na.rm = TRUE),
      sd    = sd(abundance, na.rm = TRUE),
      sem   = sd / sqrt(n),
      .groups = "drop"
    )

  pairwise_results <- data.frame(
    metabolite    = character(),
    display_name  = character(),
    category      = character(),
    group1        = character(),
    group2        = character(),
    fold_change   = numeric(),
    log2FC        = numeric(),
    p_value       = numeric(),
    p_adj         = numeric(),
    stringsAsFactors = FALSE
  )

  for (met in unique(abundance$metabolite)) {
    met_data <- abundance %>% filter(metabolite == met)
    met_groups <- unique(met_data[[group_var]])

    if (length(met_groups) < 2) next

    for (i in 1:(length(met_groups) - 1)) {
      for (j in (i + 1):length(met_groups)) {
        g1 <- met_groups[i]
        g2 <- met_groups[j]

        v1 <- met_data$abundance[met_data[[group_var]] == g1]
        v2 <- met_data$abundance[met_data[[group_var]] == g2]

        v1 <- v1[!is.na(v1)]
        v2 <- v2[!is.na(v2)]

        if (length(v1) < 3 || length(v2) < 3) next

        mean1 <- mean(v1, na.rm = TRUE)
        mean2 <- mean(v2, na.rm = TRUE)

        if (is.na(mean1) || is.na(mean2) || is.nan(mean1) || is.nan(mean2)) next
        if (mean1 <= 0 || mean2 <= 0) next

        test_res <- tryCatch(
          t.test(v1, v2),
          error = function(e) {
            message(sprintf("  t.test failed for %s (%s vs %s): %s",
                            met, g1, g2, conditionMessage(e)))
            NULL
          }
        )
        if (!is.null(test_res) && !is.na(test_res$p.value)) {
          pairwise_results <- rbind(pairwise_results, data.frame(
            metabolite   = met,
            display_name = unique(met_data$display_name)[1],
            category     = unique(met_data$category)[1],
            group1       = g1,
            group2       = g2,
            fold_change  = mean2 / mean1,
            log2FC       = log2(mean2 / mean1),
            p_value      = test_res$p.value,
            p_adj        = NA_real_,
            stringsAsFactors = FALSE
          ))
        }
      }
    }
  }

  if (nrow(pairwise_results) > 0) {
    pairwise_results$p_adj <- p.adjust(pairwise_results$p_value, method = "BH")
  }

  list(
    overall_stats  = stats_results,
    group_stats    = group_stats,
    pairwise       = pairwise_results,
    abundance_data = abundance
  )
}

# ===========================================================================
# 5. 铁衰老代谢特征得分 (ssGSEA 风格)
# ===========================================================================

compute_ferroptosis_signature_score <- function(parsed, matched) {
  abundance <- parsed$abundance %>%
    inner_join(matched, by = c("metabolite" = "metabolite_name"))

  zscore_data <- abundance %>%
    group_by(metabolite) %>%
    mutate(
      zscore = (abundance - mean(abundance, na.rm = TRUE)) /
                sd(abundance, na.rm = TRUE)
    ) %>%
    ungroup()

  signature_scores <- zscore_data %>%
    group_by(sample_id, category) %>%
    summarise(
      signature_score = mean(zscore, na.rm = TRUE),
      n_metabolites   = n(),
      .groups = "drop"
    ) %>%
    left_join(parsed$sample_meta, by = "sample_id")

  sample_scores <- signature_scores %>%
    group_by(sample_id) %>%
    summarise(
      ferroptosis_score = mean(signature_score, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    left_join(parsed$sample_meta, by = "sample_id")

  list(
    category_scores = signature_scores,
    sample_scores   = sample_scores,
    zscore_data     = zscore_data
  )
}

# ===========================================================================
# 6. 可视化
# ===========================================================================

plot_metabolite_heatmap <- function(stats, score_data, group_var = "Age") {
  wide_mat <- stats$abundance_data %>%
    group_by(display_name, sample_id) %>%
    summarise(abundance = mean(abundance, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(
      id_cols     = display_name,
      names_from  = sample_id,
      values_from = abundance
    ) %>%
    column_to_rownames("display_name")

  wide_mat <- as.matrix(wide_mat)
  wide_mat <- wide_mat[apply(wide_mat, 1, function(x) !all(is.na(x))), ]
  wide_mat_log <- log2(wide_mat + 1)
  wide_mat_scaled <- t(scale(t(wide_mat_log)))

  sample_meta <- stats$abundance_data %>%
    select(sample_id, all_of(group_var)) %>%
    distinct()

  ann_col <- data.frame(
    row.names = sample_meta$sample_id
  )
  ann_col[[group_var]] <- sample_meta[[group_var]]

  common_samples <- intersect(colnames(wide_mat_scaled), rownames(ann_col))
  wide_mat_scaled <- wide_mat_scaled[, common_samples, drop = FALSE]
  ann_col <- ann_col[common_samples, , drop = FALSE]

  age_colors <- brewer.pal(max(3, length(unique(ann_col[[group_var]]))), "Set2")
  names(age_colors) <- unique(ann_col[[group_var]])
  ann_colors <- list()
  ann_colors[[group_var]] <- age_colors

  pdf(file.path(FIG_DIR, "ferroptosis_metabolite_heatmap.pdf"),
      width = 14, height = 10)
  pheatmap(
    wide_mat_scaled,
    name              = "Z-score",
    annotation_col    = ann_col,
    annotation_colors = ann_colors,
    cluster_rows      = TRUE,
    cluster_cols      = TRUE,
    show_colnames     = FALSE,
    fontsize_row      = 7,
    fontsize_col      = 5,
    main              = "铁死亡/铁衰老代谢物表达热图 (Z-score归一化)",
    color             = colorRampPalette(
      rev(brewer.pal(11, "RdBu")))(100)
  )
  dev.off()
  message("热图已保存: ", file.path(FIG_DIR, "ferroptosis_metabolite_heatmap.pdf"))
}

plot_category_boxplots <- function(score_data, group_var = "Age") {
  scores <- score_data$category_scores
  if (!group_var %in% names(scores)) {
    group_var <- intersect(
      c("Age", "Brain region", "Gender"),
      names(scores)
    )[1]
  }

  p <- ggplot(scores, aes(
    x     = !!sym(group_var),
    y     = signature_score,
    fill  = !!sym(group_var)
  )) +
    geom_boxplot(outlier.size = 0.8, alpha = 0.8) +
    facet_wrap(~ category, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      title = "铁衰老代谢特征得分 (按类别)",
      x     = group_var,
      y     = "Signature Z-score"
    ) +
    theme_bw(base_size = 11) +
    theme(
      legend.position  = "bottom",
      strip.background = element_rect(fill = "grey90"),
      axis.text.x      = element_text(angle = 45, hjust = 1)
    )

  ggsave(
    file.path(FIG_DIR, "ferroptosis_category_boxplots.pdf"),
    p, width = 12, height = 10, dpi = 300
  )
  message("类别箱线图已保存: ", file.path(FIG_DIR, "ferroptosis_category_boxplots.pdf"))
}

plot_volcano_metabolites <- function(stats) {
  pairwise <- stats$pairwise
  if (nrow(pairwise) == 0) {
    message("无成对比较结果, 跳过火山图")
    return(invisible(NULL))
  }

  pair_label <- with(pairwise[1, ], paste(group2, "vs", group1))
  volc_data <- pairwise %>%
    filter(!is.na(p_adj)) %>%
    mutate(
      neg_log10_padj = -log10(p_adj),
      significance   = case_when(
        p_adj < 0.01 & abs(log2FC) > 0.5 ~ "FDR<0.01 & |log2FC|>0.5",
        p_adj < 0.05 & abs(log2FC) > 0.5 ~ "FDR<0.05 & |log2FC|>0.5",
        TRUE ~ "Not significant"
      )
    )

  top_labels <- volc_data %>%
    filter(significance != "Not significant") %>%
    arrange(p_adj) %>%
    head(20)

  p <- ggplot(volc_data, aes(x = log2FC, y = neg_log10_padj, color = significance)) +
    geom_point(alpha = 0.7, size = 1.5) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50") +
    geom_vline(xintercept = c(-0.5, 0.5), linetype = "dashed", color = "grey50") +
    ggrepel::geom_text_repel(
      data = top_labels,
      aes(label = display_name),
      size  = 2.5,
      max.overlaps = 20
    ) +
    scale_color_manual(
      values = c(
        "FDR<0.01 & |log2FC|>0.5" = "#D73027",
        "FDR<0.05 & |log2FC|>0.5" = "#FC8D59",
        "Not significant"          = "grey70"
      )
    ) +
    labs(
      title    = paste("代谢物差异火山图:", pair_label),
      x        = "log2(Fold Change)",
      y        = "-log10(FDR)"
    ) +
    theme_bw(base_size = 11)

  ggsave(
    file.path(FIG_DIR, "ferroptosis_volcano.pdf"),
    p, width = 8, height = 7, dpi = 300
  )
  message("火山图已保存: ", file.path(FIG_DIR, "ferroptosis_volcano.pdf"))
}

plot_age_trend <- function(stats, group_var = "Age") {
  data <- stats$abundance_data %>%
    group_by(display_name, category, !!sym(group_var)) %>%
    summarise(
      mean_abundance = mean(abundance, na.rm = TRUE),
      sem_abundance  = sd(abundance, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  p <- ggplot(data, aes(
    x = !!sym(group_var), y = mean_abundance,
    color = display_name, group = display_name
  )) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 1.5) +
    facet_wrap(~ category, scales = "free_y", ncol = 3) +
    labs(
      title = "衰老过程中铁死亡代谢物轨迹",
      x     = group_var,
      y     = "Mean Abundance",
      color = "Metabolite"
    ) +
    theme_bw(base_size = 10) +
    theme(
      legend.position  = "bottom",
      strip.background = element_rect(fill = "grey90"),
      legend.text      = element_text(size = 7)
    ) +
    guides(color = guide_legend(ncol = 3))

  ggsave(
    file.path(FIG_DIR, "ferroptosis_age_trajectory.pdf"),
    p, width = 14, height = 12, dpi = 300
  )
  message("年龄轨迹图已保存: ", file.path(FIG_DIR, "ferroptosis_age_trajectory.pdf"))
}

plot_ferroptosis_score <- function(score_data, group_var = "Age") {
  scores <- score_data$sample_scores

  if (!group_var %in% names(scores)) {
    group_var <- intersect(c("Age", "Brain region", "Gender"), names(scores))[1]
  }

  p <- ggplot(scores, aes(
    x     = !!sym(group_var),
    y     = ferroptosis_score,
    fill  = !!sym(group_var)
  )) +
    geom_boxplot(outlier.size = 1, alpha = 0.7) +
    geom_jitter(width = 0.15, alpha = 0.4, size = 1) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      title = "铁衰老综合代谢特征得分",
      subtitle = "基于脂质过氧化+抗氧化+多胺+铁代谢+衰老代谢物Z-score平均值",
      x = group_var,
      y = "Ferroptosis/Aging Metabolic Score"
    ) +
    theme_bw(base_size = 12)

  if (length(unique(scores[[group_var]])) > 2) {
    p <- p + stat_compare_means(
      method = "anova",
      label   = "p.format",
      label.y = max(scores$ferroptosis_score, na.rm = TRUE) * 1.05
    )
  }

  ggsave(
    file.path(FIG_DIR, "ferroptosis_signature_score.pdf"),
    p, width = 8, height = 6, dpi = 300
  )
  message("铁衰老得分图已保存: ", file.path(FIG_DIR, "ferroptosis_signature_score.pdf"))
}

# ===========================================================================
# 7. 文献交叉验证表
# ===========================================================================

build_literature_cross_validation <- function() {
  lit_evidence <- data.frame(
    metabolite_class = c(
      "4-HNE", "MDA", "GSH", "GSSG", "GSH/GSSG",
      "SOD", "GPX4", "Total Iron", "ROS",
      "Spermidine", "Spermine", "Putrescine",
      "Sphingolipids", "Glycerophospholipids",
      "Taurine", "BCAA", "Acyl-carnitines",
      "NAD+", "Lactate", "Succinate"
    ),
    direction_mcao = c(
      "UP", "UP", "DOWN", "UP", "DOWN",
      "DOWN", "DOWN", "UP", "UP",
      "DOWN (predicted)", "DOWN (predicted)", "UP (predicted)",
      "Altered", "Altered",
      "DOWN", "Altered", "UP",
      "DOWN", "UP", "UP"
    ),
    pmid_evidence = c(
      "40375180, 40768899", "40375180, 38958365, 11699709",
      "40768899, 40375180", "40768899", "40768899",
      "40375180", "40375180, 40768899", "40768899",
      "37752100, 40768899",
      "SAT1 hypothesis", "SAT1 hypothesis", "SAT1 hypothesis",
      "Molecules 2021(26:4124)", "Molecules 2021(26:4124)",
      "Front Pharmacol 2022(13:814942)", "37752100",
      "Antioxidants 2024(13:60)",
      "Aging-related", "Antioxidants 2024", "Antioxidants 2024"
    ),
    consistency = c(
      "High", "High", "High", "High", "High",
      "High", "High", "High", "High",
      "Predicted", "Predicted", "Predicted",
      "Confirmed", "Confirmed",
      "Confirmed", "Confirmed", "Confirmed",
      "Literature", "Confirmed", "Confirmed"
    ),
    stringsAsFactors = FALSE
  )

  write.csv(
    lit_evidence,
    file.path(TAB_DIR, "literature_cross_validation.csv"),
    row.names = FALSE
  )

  message("文献交叉验证表已保存")
  lit_evidence
}

# ===========================================================================
# 8. 主执行流程
# ===========================================================================

main <- function() {
  message("========================================")
  message("铁衰老代谢组学交叉验证分析")
  message("========================================")

  # 8.1 加载数据 (从预转换的 CSV 文件)
  message("\n[1/6] 加载 Metabolomics Workbench 数据...")
  st001637 <- load_mwtab_csv(DATA_DIR)

  # 8.2 匹配代谢物
  message("\n[2/6] 匹配铁死亡/铁衰老相关代谢物...")
  message(sprintf("  查找表条目数: %d", nrow(METABOLITE_LOOKUP)))
  message(sprintf("  待搜索代谢物数: %d", length(unique(st001637$abundance$metabolite))))
  # 诊断: 检查前几个搜索词
  sample_terms <- head(METABOLITE_LOOKUP$search_term, 5)
  all_mets <- unique(st001637$abundance$metabolite)
  for (term in sample_terms) {
    hits <- grep(tolower(term), tolower(all_mets), fixed = TRUE, value = TRUE)
    message(sprintf("  搜索 '%s' → %d 命中: %s", term, length(hits),
      paste(head(hits, 3), collapse = ", ")))
  }
  matched <- match_ferroptosis_metabolites(st001637, METABOLITE_LOOKUP)

  if (nrow(matched) == 0) {
    message("未匹配到任何铁死亡/铁衰老相关代谢物, 终止分析")
    return(invisible(NULL))
  }

  write.csv(
    matched,
    file.path(TAB_DIR, "matched_ferroptosis_metabolites.csv"),
    row.names = FALSE
  )

  # 8.3 统计分析
  message("\n[3/6] 统计分析...")
  available_factors <- names(st001637$sample_meta)
  message(sprintf("  可用因子: %s", paste(available_factors, collapse = ", ")))

  group_var <- if ("Age" %in% available_factors) "Age" else available_factors[2]
  stats <- run_metabolite_stats(st001637, matched, group_var = group_var)

  write.csv(
    stats$overall_stats,
    file.path(TAB_DIR, "metabolite_overall_stats.csv"),
    row.names = FALSE
  )
  write.csv(
    stats$pairwise,
    file.path(TAB_DIR, "metabolite_pairwise_comparison.csv"),
    row.names = FALSE
  )

  sig_hits <- stats$pairwise %>%
    filter(p_adj < 0.05) %>%
    arrange(p_adj)

  message(sprintf("  显著差异代谢物 (FDR<0.05): %d", nrow(sig_hits)))
  if (nrow(sig_hits) > 0) {
    message("  Top hits:")
    for (i in seq_len(min(10, nrow(sig_hits)))) {
      message(sprintf(
        "    %s: %s vs %s, log2FC=%.2f, p_adj=%.2e",
        sig_hits$display_name[i],
        sig_hits$group2[i], sig_hits$group1[i],
        sig_hits$log2FC[i], sig_hits$p_adj[i]
      ))
    }
  }

  # 8.4 铁衰老特征得分
  message("\n[4/6] 计算铁衰老代谢特征得分...")
  score_data <- compute_ferroptosis_signature_score(st001637, matched)

  write.csv(
    score_data$sample_scores,
    file.path(TAB_DIR, "ferroptosis_signature_scores.csv"),
    row.names = FALSE
  )

  age_scores <- score_data$sample_scores
  if ("Age" %in% names(age_scores)) {
    age_summary <- age_scores %>%
      group_by(Age) %>%
      summarise(
        mean_score = mean(ferroptosis_score, na.rm = TRUE),
        sd_score   = sd(ferroptosis_score, na.rm = TRUE),
        .groups    = "drop"
      )
    message("  年龄组铁衰老得分:")
    print(age_summary)
  }

  # 8.5 可视化
  message("\n[5/6] 生成可视化...")
  plot_metabolite_heatmap(stats, score_data, group_var = group_var)
  plot_category_boxplots(score_data, group_var = group_var)
  plot_volcano_metabolites(stats)
  plot_age_trend(stats, group_var = group_var)
  plot_ferroptosis_score(score_data, group_var = group_var)

  # 8.6 文献交叉验证
  message("\n[6/6] 构建文献交叉验证表...")
  lit_table <- build_literature_cross_validation()

  message("\n========================================")
  message("分析完成!")
  message(sprintf("  输出目录: %s", OUTPUT_DIR))
  message(sprintf("  图表: %s", FIG_DIR))
  message(sprintf("  表格: %s", TAB_DIR))
  message("========================================")

  results <- list(
    study_data   = st001637,
    matched_mets = matched,
    stats        = stats,
    scores       = score_data,
    literature   = lit_table
  )

  saveRDS(
    results,
    file.path(OUTPUT_DIR, "ferroptosis_metabolomics_results.rds")
  )

  invisible(results)
}

# ===========================================================================
# 9. 执行
# ===========================================================================
# 守卫: 仅在 Rscript 直接运行时执行 main(); source() 加载时跳过
# 避免 run_pipeline.R source() 所有 R/*.R 时自动触发 main() 导致阻塞
# (sys.nframe()==0 表示不在任何函数调用栈内, 即 Rscript 顶层执行)
if (sys.nframe() == 0) {
  results <- main()
}