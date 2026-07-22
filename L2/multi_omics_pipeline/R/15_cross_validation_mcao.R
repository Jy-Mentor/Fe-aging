# ===========================================================================
# 15_cross_validation_mcao.R
# 铁衰老项目 - 脑缺血代谢组学外部数据交叉验证
#
# 数据来源:
#   ST002042 - tMCAO 小鼠模型长期代谢组学分析 (Metabolomics Workbench)
#     - 物种: C57BL/6 小鼠
#     - 样本: 57 个血浆样本
#     - 时间点: 1 天/1 周/1 个月/6 个月 (tMCAO vs sham vs Normal)
#     - 平台: LC-MS (负离子模式)
#     - 代谢物: 44 个脂质氧化产物
#
#   ST002080 - 铁死亡诱导剂 RSL3 对心肌细胞代谢组的影响
#     - 物种: 大鼠 (Rattus norvegicus)
#     - 样本: 44 个细胞样本
#     - 处理: Control / RSL3 / Ferrostatin-1 / RSL3+Fer1 / RSL3+TSM / RSL3+XJB
#     - 平台: GC-MS
#     - 代谢物: 52 个 (含 TCA 循环、氨基酸、GSH 前体)
#
# 文献支撑 (PubMed 已验证):
#   - PMID 34654818: 小鼠脑代谢图谱 (Nature Commun 2021)
#   - PMID 38442890: 铁死亡自噬共识指南 (Autophagy 2024)
#   - PMID 32080622: POR 参与铁死亡磷脂过氧化 (Nat Chem Biol 2020)
#   - PMID 30799221: IKE 诱导铁死亡 (Cell Chem Biol 2019)
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
  library(jsonlite)
})

# ===========================================================================
# 0. 路径与输出配置
# ===========================================================================
BASE_DIR    <- "D:/铁衰老 绝不重蹈覆辙/L2/multi_omics_pipeline"
DATA_DIR    <- file.path(BASE_DIR, "data", "metabolomics")
OUTPUT_DIR  <- file.path(BASE_DIR, "output", "metabolomics_cross_validation")
FIG_DIR     <- file.path(OUTPUT_DIR, "figures")
TAB_DIR     <- file.path(OUTPUT_DIR, "tables")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIG_DIR,    showWarnings = FALSE, recursive = TRUE)
dir.create(TAB_DIR,    showWarnings = FALSE, recursive = TRUE)

# ===========================================================================
# 1. 铁衰老核心代谢特征面板 (基于文献系统综述)
# ===========================================================================
# 依据:
#   - PMID 38442890 (Autophagy 2024) 铁死亡共识
#   - PMID 40375180 (DGAT1 抑制 MCAO) 4-HNE/MDA/SOD/GPX4
#   - PMID 40768899 (circMTCO2) GSH/GSSG
#   - PMID 37752100 (PPM1K) BCAA 代谢紊乱
#   - SAT1 多胺假说 (铁衰老核心机制)

FERROPTOSIS_AGING_PANEL <- list(

  lipid_peroxidation = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "12-HETE",             "12-HETE",          "UP",
    "15-HETE",             "15-HETE",          "UP",
    "12-HOTrE",            "12-HOTrE",         "UP",
    "13-HODE",             "13-HODE",          "UP",
    "9-HODE",              "9-HODE",           "UP",
    "9-KODE",              "9-KODE",           "UP",
    "EpOME",               "EpOME",            "UP",
    "DiHOME",              "DiHOME",           "UP",
    "HDoHE",               "HDoHE",            "UP",
    "HOTrE",               "HOTrE",            "UP",
    "isoprostane",         "8-iso-PGF2alpha",  "UP"
  ),

  antioxidant_defense = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "glutathione",         "GSH",              "DOWN",
    "cysteine",            "Cysteine",         "DOWN",
    "glutamate",           "Glutamate",        "UP",
    "glutamine",           "Glutamine",        "UP",
    "glycine",             "Glycine",          "DOWN",
    "5-oxoproline",        "5-Oxo-Pro",        "UP",
    "pyroglutamic",        "PyroGlu",           "UP",
    "taurine",             "Taurine",          "DOWN",
    "hypotaurine",         "Hypotaurine",      "DOWN",
    "methionine",          "Methionine",       "UP",
    "sarcosine",           "Sarcosine",        "UP",
    "nicotinamide",        "NAM-N-oxide",      "UP",
    "ascorbic",            "VitC",             "DOWN"
  ),

  polyamine_metabolism = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "spermidine",          "Spermidine",       "DOWN",
    "spermine",            "Spermine",         "DOWN",
    "putrescine",          "Putrescine",       "UP",
    "ornithine",           "Ornithine",        "UP",
    "arginine",            "Arginine",         "UP",
    "agmatine",            "Agmatine",         "UP"
  ),

  tca_energy = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "citric acid",         "Citrate",          "UP",
    "isocitric",           "Isocitrate",       "UP",
    "alpha-ketoglutaric",  "alpha-KG",         "UP",
    "oxoglutaric",         "alpha-KG",         "UP",
    "succinic",            "Succinate",        "UP",
    "fumaric",             "Fumarate",         "UP",
    "malic",               "Malate",           "UP",
    "lactic",              "Lactate",          "UP",
    "pyruvic",             "Pyruvate",         "UP",
    "pantothenic",         "Pantothenate",     "DOWN"
  ),

  lipid_signaling = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "arachidonic",         "AA",               "UP",
    "docosahexaenoic",     "DHA",              "UP",
    "eicosapentaenoic",    "EPA",              "UP",
    "adrenic",             "Adrenic acid",     "UP",
    "palmitic",            "PA (16:0)",        "UP",
    "stearic",             "SA (18:0)",        "UP",
    "oleic",               "OA (18:1)",        "UP",
    " LPC ",               "LPC",              "DOWN",
    "LPC",                 "LPC",              "DOWN",
    "PE ",                 "PE",               "DOWN",
    "Cer",                 "Ceramide",         "UP",
    "sphingosine",         "Sphingosine",      "UP",
    "sphinganine",         "Sphinganine",      "UP"
  ),

  nucleotide_aging = tribble(
    ~search_term,          ~display_name,      ~expected_direction,
    "hypoxanthine",        "Hypoxanthine",     "UP",
    "xanthine",            "Xanthine",         "UP",
    "uric",                "Uric acid",        "UP",
    "inosine",             "Inosine",          "UP",
    "uracil",              "Uracil",           "UP",
    "uridine",             "Uridine",          "UP",
    "pseudouridine",       "Pseudouridine",    "UP"
  )
)

# 合并为查找表
METABOLITE_PANEL <- bind_rows(
  lapply(names(FERROPTOSIS_AGING_PANEL), function(cat) {
    FERROPTOSIS_AGING_PANEL[[cat]] %>% mutate(category = cat)
  })
)

# ===========================================================================
# 2. 从 JSON 数据加载并展平为长格式
# ===========================================================================

load_mwtab_json <- function(json_path, study_id) {
  if (!file.exists(json_path)) {
    stop("JSON 文件不存在: ", json_path)
  }

  json_data <- fromJSON(json_path, flatten = TRUE)

  # 从本地 factors JSON 加载样本元数据
  factors_path <- file.path(DATA_DIR, paste0(study_id, "_factors.json"))
  if (!file.exists(factors_path)) {
    stop("样本因子 JSON 不存在: ", factors_path,
         "\n请先运行 PowerShell 下载该文件。")
  }
  factors <- fromJSON(factors_path, flatten = TRUE)

  if (is.data.frame(factors)) {
    sample_meta <- factors
  } else if (is.list(factors)) {
    sample_meta <- bind_rows(lapply(factors, as.data.frame,
                                    stringsAsFactors = FALSE))
  } else {
    stop("无法解析 factors JSON: ", factors_path)
  }

  # 移除样本名前后空格
  if ("local_sample_id" %in% names(sample_meta)) {
    sample_meta$local_sample_id <- str_trim(sample_meta$local_sample_id)
  }

  # 提取代谢物丰度数据 (长格式)
  metabolites_list <- lapply(seq_along(json_data), function(i) {
    m <- json_data[[i]]
    if (is.null(m$metabolite_name)) return(NULL)

    data <- m$DATA
    if (is.null(data) || length(data) == 0) return(NULL)

    data.frame(
      study_id       = study_id,
      metabolite     = m$metabolite_name,
      refmet_name    = if (is.null(m$refmet_name)) "" else m$refmet_name,
      metabolite_id  = if (is.null(m$metabolite_id)) "" else m$metabolite_id,
      sample_id      = names(data),
      abundance      = as.numeric(unlist(data)),
      stringsAsFactors = FALSE
    )
  })

  abundance <- bind_rows(metabolites_list)
  abundance$sample_id <- str_trim(abundance$sample_id)

  list(
    study_id    = study_id,
    sample_meta = sample_meta,
    abundance   = abundance,
    n_metabolites = length(unique(abundance$metabolite)),
    n_samples    = length(unique(abundance$sample_id))
  )
}

# ===========================================================================
# 3. ST002042 专用: 解析时间点和处理组
# ===========================================================================

parse_st002042_groups <- function(sample_meta) {
  # local_sample_id 格式: 1dM_1 (1天 MCAO), 1wM_1 (1周 MCAO),
  #                        1mM_1 (1月 MCAO), 6mM_1 (6月 MCAO),
  #                        6mS_1 (6月 sham), N_1 (Normal), sham_1
  sample_meta %>%
    mutate(
      timepoint = case_when(
        grepl("^1d", local_sample_id) ~ "1-day",
        grepl("^1w", local_sample_id) ~ "1-week",
        grepl("^1m", local_sample_id) ~ "1-month",
        grepl("^6m", local_sample_id) ~ "6-month",
        grepl("^N_", local_sample_id) ~ "Normal",
        grepl("^sham", local_sample_id) ~ "Sham",
        TRUE ~ "Unknown"
      ),
      treatment = case_when(
        grepl("^1d[A-Z]", local_sample_id) ~ "tMCAO",
        grepl("^1w[A-Z]", local_sample_id) ~ "tMCAO",
        grepl("^1m[A-Z]", local_sample_id) ~ "tMCAO",
        grepl("^6mM", local_sample_id) ~ "tMCAO",
        grepl("^6mS", local_sample_id) ~ "Sham",
        grepl("^N_", local_sample_id) ~ "Normal",
        grepl("^sham_", local_sample_id) ~ "Sham",
        TRUE ~ "Unknown"
      ),
      timepoint = factor(timepoint,
        levels = c("Normal", "Sham", "1-day", "1-week", "1-month", "6-month"))
    )
}

# ===========================================================================
# 4. ST002080 专用: 解析处理组
# ===========================================================================

parse_st002080_groups <- function(sample_meta) {
  sample_meta %>%
    mutate(
      treatment = case_when(
        grepl("^Control", local_sample_id) ~ "Control",
        grepl("^Fer1", local_sample_id) ~ "Ferrostatin-1",
        grepl("^RSL3_", local_sample_id) ~ "RSL3",
        grepl("^RF_", local_sample_id) ~ "RSL3+Fer1",
        grepl("^RT_", local_sample_id) ~ "RSL3+TSM",
        grepl("^RX_", local_sample_id) ~ "RSL3+XJB",
        grepl("^TSM_", local_sample_id) ~ "TSM-1005-44",
        grepl("^XJB_", local_sample_id) ~ "XJB-5-131",
        TRUE ~ "Unknown"
      ),
      is_ferroptosis = treatment == "RSL3",
      is_rescue = grepl("^RSL3\\+", treatment)
    )
}

# ===========================================================================
# 5. 匹配铁死亡代谢物
# ===========================================================================

match_panel_metabolites <- function(parsed, panel) {
  all_metabolites <- unique(parsed$abundance$metabolite)

  matched <- data.frame(
    metabolite_name = character(),
    refmet_name     = character(),
    search_term     = character(),
    category        = character(),
    display_name    = character(),
    expected_direction = character(),
    match_type      = character(),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(panel))) {
    term <- panel$search_term[i]
    disp <- panel$display_name[i]
    dir  <- panel$expected_direction[i]
    cat_val <- panel$category[i]

    # 模糊匹配 (忽略大小写)
    matches <- all_metabolites[
      grepl(tolower(term), tolower(all_metabolites), fixed = TRUE)
    ]

    for (m in matches) {
      matched <- rbind(matched, data.frame(
        metabolite_name     = m,
        refmet_name         = parsed$abundance$refmet_name[
          parsed$abundance$metabolite == m][1],
        search_term         = term,
        category            = cat_val,
        display_name        = disp,
        expected_direction  = dir,
        match_type          = "fuzzy",
        stringsAsFactors    = FALSE
      ))
    }
  }

  # 去重: 同一代谢物保留第一个匹配
  matched <- matched[!duplicated(matched$metabolite_name), ]

  message(sprintf("  匹配到 %d 个铁死亡/铁衰老相关代谢物", nrow(matched)))
  for (cat_val in unique(matched$category)) {
    n <- sum(matched$category == cat_val)
    message(sprintf("    %s: %d", cat_val, n))
  }

  matched
}

# ===========================================================================
# 6. 统计分析 (NA-safe 版本)
# ===========================================================================

run_pairwise_stats <- function(abundance, group_var) {
  abundance <- abundance[!is.na(abundance$abundance), ]

  pairwise_results <- data.frame(
    metabolite    = character(),
    display_name  = character(),
    category      = character(),
    expected_direction = character(),
    group1        = character(),
    group2        = character(),
    mean_group1   = numeric(),
    mean_group2   = numeric(),
    fold_change   = numeric(),
    log2FC        = numeric(),
    p_value       = numeric(),
    p_adj         = numeric(),
    direction_match = logical(),
    stringsAsFactors = FALSE
  )

  for (met in unique(abundance$metabolite)) {
    met_data <- abundance %>% filter(metabolite == met)
    met_groups <- unique(met_data[[group_var]])
    met_groups <- met_groups[!is.na(met_groups)]

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

        mean1 <- mean(v1)
        mean2 <- mean(v2)

        if (mean1 > 0 && mean2 > 0) {
          test_res <- tryCatch(
            t.test(v1, v2),
            error = function(e) {
              message(sprintf("  t.test failed for %s (%s vs %s): %s",
                              met, g1, g2, conditionMessage(e)))
              NULL
            }
          )

          if (!is.null(test_res)) {
            expected <- unique(met_data$expected_direction)[1]
            actual_up <- mean2 > mean1
            dir_match <- (expected == "UP" && actual_up) ||
                         (expected == "DOWN" && !actual_up)

            pairwise_results <- rbind(pairwise_results, data.frame(
              metabolite    = met,
              display_name = unique(met_data$display_name)[1],
              category     = unique(met_data$category)[1],
              expected_direction = expected,
              group1       = g1,
              group2       = g2,
              mean_group1  = mean1,
              mean_group2  = mean2,
              fold_change  = mean2 / mean1,
              log2FC       = log2(mean2 / mean1),
              p_value      = test_res$p.value,
              p_adj        = NA_real_,
              direction_match = dir_match,
              stringsAsFactors = FALSE
            ))
          }
        }
      }
    }
  }

  if (nrow(pairwise_results) > 0) {
    pairwise_results$p_adj <- p.adjust(
      pairwise_results$p_value, method = "BH"
    )
  }

  pairwise_results
}

# ===========================================================================
# 7. 铁衰老代谢特征评分 (ssGSEA-like)
# ===========================================================================

compute_signature_score <- function(abundance, matched, group_var) {
  abundance_matched <- abundance %>%
    inner_join(matched, by = c("metabolite" = "metabolite_name"))

  # Z-score 归一化 (按代谢物)
  zscore_data <- abundance_matched %>%
    group_by(metabolite) %>%
    mutate(
      zscore = (abundance - mean(abundance, na.rm = TRUE)) /
                sd(abundance, na.rm = TRUE)
    ) %>%
    ungroup() %>%
    filter(!is.na(zscore) & is.finite(zscore))

  # 按类别计算 signature score
  category_scores <- zscore_data %>%
    group_by(sample_id, category) %>%
    summarise(
      signature_score = mean(zscore, na.rm = TRUE),
      n_metabolites    = n(),
      .groups = "drop"
    )

  # 样本总体铁衰老得分 (所有类别平均)
  sample_scores <- category_scores %>%
    group_by(sample_id) %>%
    summarise(
      ferroptosis_score = mean(signature_score, na.rm = TRUE),
      .groups = "drop"
    )

  # 加入组别信息
  group_info <- abundance_matched %>%
    select(sample_id, all_of(group_var)) %>%
    distinct()

  sample_scores <- sample_scores %>%
    left_join(group_info, by = "sample_id")

  category_scores <- category_scores %>%
    left_join(group_info, by = "sample_id")

  list(
    zscore_data      = zscore_data,
    category_scores  = category_scores,
    sample_scores    = sample_scores
  )
}

# ===========================================================================
# 8. 可视化函数
# ===========================================================================

plot_timecourse_heatmap <- function(abundance, matched, sample_meta,
                                    group_var = "timepoint") {
  abundance_matched <- abundance %>%
    inner_join(matched, by = c("metabolite" = "metabolite_name"))

  wide_mat <- abundance_matched %>%
    group_by(display_name, sample_id) %>%
    summarise(abundance = mean(abundance, na.rm = TRUE), .groups = "drop") %>%
    pivot_wider(
      id_cols     = display_name,
      names_from  = sample_id,
      values_from = abundance
    ) %>%
    column_to_rownames("display_name") %>%
    as.matrix()

  wide_mat <- wide_mat[apply(wide_mat, 1, function(x) !all(is.na(x))), ]
  wide_mat_log <- log2(wide_mat + 1)
  wide_mat_scaled <- t(scale(t(wide_mat_log)))

  # sample_meta 的样本 ID 列名为 local_sample_id
  id_col <- if ("local_sample_id" %in% names(sample_meta)) {
    "local_sample_id"
  } else {
    "sample_id"
  }

  ann_col <- sample_meta %>%
    select(all_of(c(id_col, group_var))) %>%
    distinct()
  ann_col <- ann_col[!duplicated(ann_col[[id_col]]), ]
  ann_col <- as.data.frame(ann_col)
  rownames(ann_col) <- ann_col[[id_col]]
  ann_col[[id_col]] <- NULL

  common_samples <- intersect(colnames(wide_mat_scaled), rownames(ann_col))
  if (length(common_samples) == 0) {
    message("  警告: 样本 ID 无交集, 跳过热图")
    return(invisible(NULL))
  }
  wide_mat_scaled <- wide_mat_scaled[, common_samples, drop = FALSE]
  ann_col <- ann_col[common_samples, , drop = FALSE]

  ann_colors <- list()
  levels_vec <- unique(ann_col[[group_var]])
  levels_vec <- levels_vec[!is.na(levels_vec)]
  colors_vec <- brewer.pal(max(3, length(levels_vec)), "Set2")[seq_along(levels_vec)]
  names(colors_vec) <- levels_vec
  ann_colors[[group_var]] <- colors_vec

  pdf(file.path(FIG_DIR, "ST002042_timecourse_heatmap.pdf"),
      width = 14, height = 10)
  pheatmap(
    wide_mat_scaled,
    name              = "Z-score",
    annotation_col    = ann_col,
    annotation_colors = ann_colors,
    cluster_rows      = TRUE,
    cluster_cols      = TRUE,
    show_colnames     = FALSE,
    fontsize_row      = 8,
    main              = "ST002042: tMCAO 时间进程铁衰老代谢物热图 (Z-score)",
    color             = colorRampPalette(rev(brewer.pal(11, "RdBu")))(100)
  )
  dev.off()
  message("时序热图已保存")
}

plot_timecourse_boxplot <- function(score_data, group_var, study_label) {
  scores <- score_data$category_scores

  p <- ggplot(scores, aes(
    x     = !!sym(group_var),
    y     = signature_score,
    fill  = !!sym(group_var)
  )) +
    geom_boxplot(outlier.size = 0.8, alpha = 0.8) +
    geom_jitter(width = 0.15, alpha = 0.4, size = 1) +
    facet_wrap(~ category, scales = "free_y", ncol = 3) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      title = paste0(study_label, ": 铁衰老代谢特征类别得分"),
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
    file.path(FIG_DIR, paste0(study_label, "_category_boxplots.pdf")),
    p, width = 12, height = 10, dpi = 300
  )
  message(study_label, " 类别箱线图已保存")
}

plot_total_score <- function(score_data, group_var, study_label) {
  scores <- score_data$sample_scores

  p <- ggplot(scores, aes(
    x     = !!sym(group_var),
    y     = ferroptosis_score,
    fill  = !!sym(group_var)
  )) +
    geom_boxplot(outlier.size = 1, alpha = 0.7) +
    geom_jitter(width = 0.15, alpha = 0.4, size = 1.5) +
    scale_fill_brewer(palette = "Set2") +
    labs(
      title = paste0(study_label, ": 铁衰老综合代谢特征得分"),
      subtitle = "基于脂质过氧化+抗氧化+多胺+TCA+脂质信号+核苷酸代谢物",
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
    file.path(FIG_DIR, paste0(study_label, "_total_score.pdf")),
    p, width = 8, height = 6, dpi = 300
  )
  message(study_label, " 综合得分图已保存")
}

plot_direction_concordance <- function(pairwise, study_label) {
  if (nrow(pairwise) == 0) {
    message(study_label, " 无成对比较结果, 跳过方向一致性图")
    return(invisible(NULL))
  }

  summary_df <- pairwise %>%
    filter(!is.na(p_adj)) %>%
    mutate(
      significance = case_when(
        p_adj < 0.01  ~ "FDR<0.01",
        p_adj < 0.05  ~ "FDR<0.05",
        TRUE ~ "NS"
      )
    ) %>%
    group_by(category, expected_direction, significance, direction_match) %>%
    summarise(count = n(), .groups = "drop")

  p <- ggplot(summary_df, aes(
    x = category,
    y = count,
    fill = significance
  )) +
    geom_col(position = "dodge") +
    facet_wrap(~ direction_match,
      labeller = as_labeller(c(`TRUE` = "方向一致", `FALSE` = "方向相反"))
    ) +
    scale_fill_manual(
      values = c("FDR<0.01" = "#D73027", "FDR<0.05" = "#FC8D59", "NS" = "grey70")
    ) +
    labs(
      title = paste0(study_label, ": 代谢物变化方向与预期一致性"),
      x = "代谢物类别",
      y = "成对比较数"
    ) +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(
    file.path(FIG_DIR, paste0(study_label, "_direction_concordance.pdf")),
    p, width = 10, height = 6, dpi = 300
  )
  message(study_label, " 方向一致性图已保存")
}

# ===========================================================================
# 9. 主分析流程
# ===========================================================================

analyze_study <- function(json_path, study_id, group_parser, group_var,
                          study_label) {
  message("\n========================================")
  message(sprintf("分析 %s - %s", study_id, study_label))
  message("========================================")

  # 加载数据
  message("\n[1/5] 加载数据...")
  parsed <- load_mwtab_json(json_path, study_id)
  message(sprintf("  代谢物: %d, 样本: %d",
    parsed$n_metabolites, parsed$n_samples))

  # 解析分组
  parsed$sample_meta <- group_parser(parsed$sample_meta)
  group_levels <- unique(parsed$sample_meta[[group_var]])
  message(sprintf("  分组 (%s): %s",
    group_var, paste(group_levels, collapse = ", ")))

  # 合并元数据到丰度表
  abundance <- parsed$abundance %>%
    left_join(parsed$sample_meta, by = c("sample_id" = "local_sample_id"))

  # 匹配铁死亡代谢物
  message("\n[2/5] 匹配铁衰老代谢物...")
  matched <- match_panel_metabolites(parsed, METABOLITE_PANEL)
  if (nrow(matched) == 0) {
    message("  未匹配到代谢物, 跳过该数据集")
    return(invisible(NULL))
  }

  write.csv(matched,
    file.path(TAB_DIR, paste0(study_id, "_matched_metabolites.csv")),
    row.names = FALSE)

  # 统计分析
  message("\n[3/5] 成对统计比较...")
  abundance_matched <- abundance %>%
    inner_join(matched, by = c("metabolite" = "metabolite_name"))

  pairwise <- run_pairwise_stats(abundance_matched, group_var)
  write.csv(pairwise,
    file.path(TAB_DIR, paste0(study_id, "_pairwise_stats.csv")),
    row.names = FALSE)

  sig_hits <- pairwise %>% filter(p_adj < 0.05)
  message(sprintf("  显著差异 (FDR<0.05): %d / %d",
    nrow(sig_hits), nrow(pairwise)))

  if (nrow(sig_hits) > 0) {
    direction_match_rate <- mean(sig_hits$direction_match, na.rm = TRUE)
    message(sprintf("  方向与预期一致率: %.1f%%",
      direction_match_rate * 100))
    message("  Top 10 显著代谢物:")
    sig_hits %>%
      arrange(p_adj) %>%
      head(10) %>%
      select(display_name, group1, group2, log2FC, p_adj,
             expected_direction, direction_match) %>%
      print()
  }

  # 特征评分
  message("\n[4/5] 铁衰老特征评分...")
  score_data <- compute_signature_score(abundance, matched, group_var)
  write.csv(score_data$sample_scores,
    file.path(TAB_DIR, paste0(study_id, "_signature_scores.csv")),
    row.names = FALSE)

  # 可视化
  message("\n[5/5] 生成图表...")
  if (study_id == "ST002042") {
    plot_timecourse_heatmap(abundance, matched, parsed$sample_meta, group_var)
  }
  plot_timecourse_boxplot(score_data, group_var, study_id)
  plot_total_score(score_data, group_var, study_id)
  plot_direction_concordance(pairwise, study_id)

  invisible(list(
    parsed    = parsed,
    matched   = matched,
    pairwise  = pairwise,
    scores    = score_data
  ))
}

# ===========================================================================
# 10. 跨数据集汇总
# ===========================================================================

build_cross_validation_summary <- function(st002042_res, st002080_res) {
  summary_list <- list()

  for (study_name in c("ST002042", "ST002080")) {
    res <- if (study_name == "ST002042") st002042_res else st002080_res
    if (is.null(res) || is.null(res$pairwise) || nrow(res$pairwise) == 0) next

    sig <- res$pairwise %>% filter(p_adj < 0.05)
    if (nrow(sig) == 0) next

    summary_list[[study_name]] <- sig %>%
      mutate(study_id = study_name) %>%
      select(study_id, display_name, category,
             group1, group2, log2FC, p_adj,
             expected_direction, direction_match)
  }

  if (length(summary_list) > 0) {
    summary_df <- bind_rows(summary_list)
    write.csv(summary_df,
      file.path(TAB_DIR, "cross_validation_summary.csv"),
      row.names = FALSE)
    message("\n跨数据集验证汇总表已保存")
    return(summary_df)
  }
  invisible(NULL)
}

# ===========================================================================
# 11. 执行
# ===========================================================================

message("################################################")
message("# 铁衰老代谢组学外部数据交叉验证")
message("# 数据来源: Metabolomics Workbench")
message("#   ST002042 - tMCAO 小鼠模型长期时序 (1d-6m)")
message("#   ST002080 - RSL3 诱导铁死亡 (Fer1 rescue)")
message("################################################")

st002042_res <- analyze_study(
  json_path    = file.path(DATA_DIR, "ST002042_data.json"),
  study_id     = "ST002042",
  group_parser = parse_st002042_groups,
  group_var    = "timepoint",
  study_label  = "tMCAO time-course (ST002042)"
)

st002080_res <- analyze_study(
  json_path    = file.path(DATA_DIR, "ST002080_data.json"),
  study_id     = "ST002080",
  group_parser = parse_st002080_groups,
  group_var    = "treatment",
  study_label  = "RSL3 ferroptosis (ST002080)"
)

message("\n################################################")
message("# 跨数据集汇总验证")
message("################################################")
cv_summary <- build_cross_validation_summary(st002042_res, st002080_res)

if (!is.null(cv_summary)) {
  message("\n跨数据集显著一致变化:")
  consistent <- cv_summary %>%
    filter(direction_match) %>%
    group_by(display_name, category, expected_direction) %>%
    summarise(
      n_studies = n_distinct(study_id),
      mean_log2FC = mean(log2FC, na.rm = TRUE),
      min_padj = min(p_adj, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(n_studies), min_padj)

  print(consistent)
  write.csv(consistent,
    file.path(TAB_DIR, "cross_dataset_consistent_metabolites.csv"),
    row.names = FALSE)
}

# 保存全部结果
saveRDS(
  list(
    st002042 = st002042_res,
    st002080 = st002080_res,
    cv_summary = cv_summary
  ),
  file.path(OUTPUT_DIR, "cross_validation_results.rds")
)

message("\n========================================")
message("交叉验证分析完成!")
message(sprintf("  输出目录: %s", OUTPUT_DIR))
message(sprintf("  图表: %s", FIG_DIR))
message(sprintf("  表格: %s", TAB_DIR))
message("========================================")
