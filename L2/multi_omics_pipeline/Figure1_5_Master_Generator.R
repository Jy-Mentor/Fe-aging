# ============================================================================
# Figure1_5_Master_Generator.R
# 基于 GSE233815 与 ST001637 真实公共数据生成多组学预验证 Figure 1-5
# 运行环境: R >= 4.2
# ============================================================================

rm(list = ls())

# ----------------------------------------------------------------------------
# 0. 基础设置
# ----------------------------------------------------------------------------
base_dir <- "D:/铁衰老 绝不重蹈覆辙/L2"
setwd(base_dir)

stopifnot(dir.exists(base_dir))

# 输出目录
out_fig <- file.path(base_dir, "multi_omics_pipeline/outputs/figures")
dir.create(out_fig, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------------------------
# 1. 加载依赖包
# ----------------------------------------------------------------------------
pkgs <- c("tidyverse", "Seurat", "patchwork", "ComplexHeatmap", "circlize",
          "ggExtra", "ggpubr", "RColorBrewer", "grid", "cowplot", "viridis",
          "ggrepel")

missing_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing_pkgs) > 0) {
  stop("缺少以下 R 包，请先安装: ", paste(missing_pkgs, collapse = ", "))
}

for (p in pkgs) library(p, character.only = TRUE)

select <- dplyr::select

# ----------------------------------------------------------------------------
# 2. 全局主题与配色
# ----------------------------------------------------------------------------
theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) %+replace%
    theme(
      axis.text = element_text(color = "black", size = rel(0.9)),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      legend.position = "right",
      legend.key.size = unit(0.3, "cm"),
      legend.title = element_text(size = rel(0.8), face = "bold"),
      plot.tag = element_text(size = rel(1.3), face = "bold"),
      strip.background = element_blank(),
      strip.text = element_text(size = rel(1), face = "bold")
    )
}
theme_set(theme_pub())

# CVD-safe 铁衰老主题色
pal_pos_neg <- c("#2166ac", "#b2182b")
pal_ferro <- c("#313695", "#4575b4", "#ffffbf", "#fee090", "#d73027")

# 同时导出 PDF (cairo) 和 SVG 的辅助函数
save_plot <- function(plot, filename, width, height) {
  pdf_path <- file.path(out_fig, paste0(filename, ".pdf"))
  svg_path <- file.path(out_fig, paste0(filename, ".svg"))
  ggsave(pdf_path, plot, width = width, height = height, device = cairo_pdf)
  ggsave(svg_path, plot, width = width, height = height, device = svg)
  cat("  已保存:", pdf_path, "\n")
  cat("  已保存:", svg_path, "\n")
}

# ----------------------------------------------------------------------------
# 3. 文件路径定义
# ----------------------------------------------------------------------------
paths <- list(
  bulk_degs = "multi_omics_pipeline/outputs/tables/02_bulk_all_degs.csv",
  bulk_lfc = "multi_omics_pipeline/outputs/tables/02_bulk_ferroaging_lfc_matrix.csv",
  spatial_rds = "multi_omics_pipeline/outputs/rds/10_spatial_with_proportions.rds",
  spatial_scores = "multi_omics_pipeline/outputs/tables/06_spatial_region_scores.csv",
  spatial_neuron = "multi_omics_pipeline/outputs/tables/10_neuron_prop_vs_ferroptosis.csv",
  scrna_rds = "multi_omics_pipeline/outputs/rds/08_sc_seurat_annotated_scored.rds",
  sat1_vs_fp = "multi_omics_pipeline/outputs/tables/08_sat1_vs_ferroptosis_score.csv",
  augur_csv = "multi_omics_pipeline/outputs/tables/09_augur_auc_ranking.csv",
  metab_long = "multi_omics_pipeline/data/metabolomics/ST001637_abundance_long.csv",
  metab_meta = "multi_omics_pipeline/data/metabolomics/ST001637_sample_meta.csv",
  fgsea_bcp = "multi_omics_pipeline/outputs/tables/12_fgsea_bcp_all_timepoints.csv",
  kegg_summary = "multi_omics_pipeline/output/kegg_pathway_integration/tables/cross_omics_shared_pathways.csv",
  pathway_axis = "multi_omics_pipeline/outputs/tables/13_pathway_axis_match_rate.csv",
  cross_omics_axis = "multi_omics_pipeline/output/cross_omics_integration/tables/cross_omics_axis_table.csv",
  out_fig = out_fig
)

# 检查关键文件存在性
for (p in paths) {
  if (!dir.exists(p) && !file.exists(p)) {
    stop("文件不存在: ", p)
  }
}

# ----------------------------------------------------------------------------
# 4. 自包含 GSEA 计算（无需 fgsea/clusterProfiler 包）
# ----------------------------------------------------------------------------
cat("[1/6] 自包含 GSEA 计算...\n")

# 基因集定义（与 config.yaml 保持一致，鼠源符号）
gene_sets_list <- list(
  Ferroptosis = c("Gpx4", "Acsl4", "Slc7a11", "Tfrc", "Fth1", "Ftl1", "Hmox1",
                  "Nfe2l2", "Keap1", "Sat1", "Alox15", "Ncoa4", "Slc3a2",
                  "Steap3", "Bach1", "Ptgs2", "Chac1", "Nqo1"),
  Senescence = c("Cdkn1a", "Cdkn2a", "Tp53", "Il6", "Il1b", "Tnf", "Mmp3",
                 "H2ax", "Lmnb1", "Chek1", "Glb1", "Serpine1"),
  Ferroaging = {
    fa_human <- readLines(file.path(base_dir, "../铁衰老基因.txt"), encoding = "UTF-8")
    fa_human <- trimws(fa_human)
    fa_human <- fa_human[nzchar(fa_human) & !startsWith(fa_human, "#")]
    # 人类全大写符号 -> 小鼠首字母大写符号
    vapply(fa_human, function(g) {
      if (g == toupper(g)) {
        paste0(toupper(substr(g, 1, 1)), tolower(substr(g, 2, nchar(g))))
      } else {
        g
      }
    }, character(1), USE.NAMES = FALSE)
  }
)

# 计算 ES / NES 的 GSEA 核心实现（基于经典 Subramanian 算法，p=1 加权）
compute_gsea_nes <- function(stats, gene_set, nperm = 1000, seed = 42) {
  set.seed(seed)
  stats <- sort(stats, decreasing = TRUE)
  in_set <- names(stats) %in% gene_set
  n <- length(stats)
  m <- sum(in_set)

  if (m == 0L || m == n) {
    return(list(ES = 0, NES = 0, pval = 1))
  }

  hits <- which(in_set)
  weights <- abs(stats[hits])
  weights <- weights / sum(weights)
  miss_step <- -1 / (n - m)

  step_vals <- rep(miss_step, n)
  step_vals[hits] <- weights
  running_sum <- cumsum(step_vals)
  es <- running_sum[which.max(abs(running_sum))]

  # 置换零分布（保留符号方向分别归一化）
  null_es <- vapply(seq_len(nperm), function(k) {
    perm_in_set <- sample(in_set)
    perm_hits <- which(perm_in_set)
    perm_weights <- abs(stats[perm_hits])
    perm_weights <- perm_weights / sum(perm_weights)
    perm_steps <- rep(miss_step, n)
    perm_steps[perm_hits] <- perm_weights
    perm_rs <- cumsum(perm_steps)
    perm_rs[which.max(abs(perm_rs))]
  }, numeric(1))

  if (es >= 0) {
    denom <- mean(null_es[null_es > 0])
    nes <- if (denom > 0) es / denom else es
    pval <- (sum(null_es >= es) + 1) / (length(null_es) + 1)
  } else {
    denom <- mean(abs(null_es[null_es < 0]))
    nes <- if (denom > 0) es / denom else es
    pval <- (sum(null_es <= es) + 1) / (length(null_es) + 1)
  }

  list(ES = es, NES = nes, pval = pval)
}

# 基于 02_bulk_all_degs.csv 为三个关键基因集计算全时间窗 NES
degs <- read.csv(paths$bulk_degs, stringsAsFactors = FALSE)
if (!"log2FoldChange" %in% names(degs)) {
  stop("02_bulk_all_degs.csv 缺少 log2FoldChange 列")
}
if (any(!is.na(degs$log2FoldChange) & is.na(as.numeric(degs$log2FoldChange)))) {
  stop("02_bulk_all_degs.csv 的 log2FoldChange 列包含无法转换为数值的值")
}
degs$log2FoldChange <- as.numeric(degs$log2FoldChange)

cmp_map <- c("3h_vs_Ctrl" = "3h", "12h_vs_Ctrl" = "12h",
             "24h_vs_Ctrl" = "1DPI", "3D_vs_Ctrl" = "3DPI",
             "7D_vs_Ctrl" = "7DPI")

gsea_results <- lapply(names(cmp_map), function(cmp) {
  df_cmp <- degs[degs$comparison == cmp, ]
  df_cmp <- df_cmp[!is.na(df_cmp$log2FoldChange) & !is.na(df_cmp$gene), ]
  if (nrow(df_cmp) == 0) {
    stop("comparison ", cmp, " 在 02_bulk_all_degs.csv 中无有效数据")
  }
  gene_list <- setNames(df_cmp$log2FoldChange, df_cmp$gene)
  gene_list <- sort(gene_list, decreasing = TRUE)

  res <- lapply(names(gene_sets_list), function(term) {
    gsea <- compute_gsea_nes(gene_list, gene_sets_list[[term]], nperm = 1000)
    data.frame(
      comparison = cmp,
      timepoint = cmp_map[[cmp]],
      term = term,
      NES = gsea$NES,
      pvalue = gsea$pval,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, res)
})
gsea_df <- do.call(rbind, gsea_results)
gsea_df$timepoint <- factor(gsea_df$timepoint,
                            levels = c("3h", "12h", "1DPI", "3DPI", "7DPI"))
gsea_df$term <- factor(gsea_df$term,
                       levels = c("Ferroptosis", "Senescence", "Ferroaging"))

# 4.2 空间: 合并 region/scores 到 Seurat
spatial_obj <- readRDS(paths$spatial_rds)
spatial_scores <- read.csv(paths$spatial_scores, stringsAsFactors = FALSE)
rownames(spatial_scores) <- spatial_scores$spot_id

common_spots <- intersect(Cells(spatial_obj), spatial_scores$spot_id)
if (length(common_spots) == 0) {
  stop("空间 Seurat 与 scores CSV 的 spot_id 无交集")
}

spatial_obj <- subset(spatial_obj, cells = common_spots)
spatial_scores <- spatial_scores[common_spots, ]
spatial_obj$region <- spatial_scores$region
spatial_obj$Ferroptosis <- spatial_scores$Ferroptosis
spatial_obj$Senescence <- spatial_scores$Senescence
spatial_obj$Ferroaging <- spatial_scores$Ferroaging
spatial_obj$Ferrosenescence <- spatial_scores$Ferrosenescence
spatial_obj$neuron_prop <- spatial_obj$prop_NeuronsGABA + spatial_obj$prop_NeuronsGLUT

# 5.2 空间已处理

# 5.3 单细胞: 检查 Sat1 存在并提取表达（不写入 metadata 避免与 assay 冲突）
sc_obj <- readRDS(paths$scrna_rds)
if (!"Sat1" %in% rownames(sc_obj)) {
  stop("Sat1 不在 scRNA 表达矩阵中")
}

# 5.4 代谢组: 计算 3w vs 59w 的 log2FC
metab_long <- read.csv(paths$metab_long, stringsAsFactors = FALSE)
metab_meta <- read.csv(paths$metab_meta, stringsAsFactors = FALSE)
# sample_meta 中同一 sample_id 重复 5 次（不同检测批次），去重后保留 Age
metab_meta <- metab_meta %>%
  filter(!is.na(Age), Age %in% c("3 weeks", "59 weeks")) %>%
  select(sample_id, Age) %>%
  distinct() %>%
  mutate(age_group = ifelse(Age == "3 weeks", "Young", "Old"))

metab_long <- metab_long %>%
  inner_join(metab_meta %>% select(sample_id, age_group), by = "sample_id") %>%
  filter(abundance > 0)

metab_res <- metab_long %>%
  group_by(metabolite) %>%
  filter(n_distinct(age_group) == 2, n() >= 6) %>%
  summarise(
    log2FC = log2(mean(abundance[age_group == "Old"], na.rm = TRUE) /
                    mean(abundance[age_group == "Young"], na.rm = TRUE)),
    pvalue = t.test(abundance ~ age_group)$p.value,
    .groups = "drop"
  ) %>%
  mutate(padj = p.adjust(pvalue, method = "BH"))

# 多胺/铁死亡相关代谢物标签
polyamine_terms <- c("ornithine", "putrescine", "spermidine", "spermine",
                     "acetylspermidine", "N8-acetylspermidine", "N1-acetylspermidine")
gsh_terms <- c("glutathione", "GSH", "GSSG", "cys-gly", "cysteine")
lipid_terms <- c("4-HNE", "HNE", "malondialdehyde", "MDA", "arachidonic acid", "DHA")

metab_res <- metab_res %>%
  mutate(category = case_when(
    str_detect(tolower(metabolite), paste(tolower(polyamine_terms), collapse = "|")) ~ "Polyamine",
    str_detect(tolower(metabolite), paste(tolower(gsh_terms), collapse = "|")) ~ "Antioxidant",
    str_detect(tolower(metabolite), paste(tolower(lipid_terms), collapse = "|")) ~ "Lipid peroxidation",
    TRUE ~ "Other"
  ))

# 显著差异代谢物（用于图 1C）
metab_sig <- metab_res %>%
  filter(padj < 0.05) %>%
  arrange(log2FC) %>%
  mutate(metabolite = fct_inorder(metabolite))

# 4.5 CMap fgsea BCP
fgsea_bcp <- read.csv(paths$fgsea_bcp, stringsAsFactors = FALSE)
fgsea_bcp$timepoint <- dplyr::recode(fgsea_bcp$comparison,
                                     "24h" = "1DPI",
                                     "3D" = "3DPI",
                                     "7D" = "7DPI",
                                     .default = fgsea_bcp$comparison)
fgsea_bcp$timepoint <- factor(fgsea_bcp$timepoint,
                              levels = c("3h", "12h", "1DPI", "3DPI", "7DPI"))

# 4.6 通路轴匹配率
pathway_axis <- read.csv(paths$pathway_axis, stringsAsFactors = FALSE)

# 4.7 SAT1-多胺轴代谢物（用于图 5A）
cross_axis <- read.csv(paths$cross_omics_axis, stringsAsFactors = FALSE)
if (!"axis_name" %in% colnames(cross_axis)) {
  stop("cross_omics_axis_table.csv 缺少 axis_name 列")
}

# 4.8 KEGG 跨组学共享通路（用于图 5C）
kegg_summary <- read.csv(paths$kegg_summary, stringsAsFactors = FALSE)
if (!all(c("pathway_name", "cross_omics_score") %in% colnames(kegg_summary))) {
  stop("cross_omics_shared_pathways.csv 缺少 pathway_name 或 cross_omics_score 列")
}
kegg_summary <- kegg_summary %>%
  arrange(desc(cross_omics_score)) %>%
  head(10)

# ----------------------------------------------------------------------------
# 5. Figure 2: Bulk RNA-seq 时序 GSEA + FA-96 热图
# ----------------------------------------------------------------------------
cat("[2/6] 生成 Figure 2...\n")

target_terms <- c("Ferroptosis", "Senescence", "Ferroaging")
plot_data <- gsea_df %>%
  filter(term %in% target_terms) %>%
  mutate(term = factor(term, levels = target_terms))

p2_A <- ggplot(plot_data, aes(x = timepoint, y = NES, group = term, color = term, fill = term)) +
  geom_ribbon(aes(ymin = NES - 0.15, ymax = NES + 0.15), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3, shape = 21, color = "white", stroke = 1.2) +
  scale_color_manual(values = c("Ferroptosis" = "#d73027",
                                "Senescence" = "#4575b4",
                                "Ferroaging" = "#fee090")) +
  scale_fill_manual(values = c("Ferroptosis" = "#d73027",
                               "Senescence" = "#4575b4",
                               "Ferroaging" = "#fee090")) +
  labs(x = "Timepoint", y = "Normalized Enrichment Score (NES)", tag = "A") +
  facet_wrap(~ term, nrow = 1) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1))

lfc_mat <- read.csv(paths$bulk_lfc, row.names = 1, stringsAsFactors = FALSE)
lfc_mat <- lfc_mat[, c("X3h", "X12h", "X24h", "X3D", "X7D")]
colnames(lfc_mat) <- c("3h", "12h", "1DPI", "3DPI", "7DPI")

set.seed(42)
clusters <- kmeans(lfc_mat, centers = 3)$cluster
row_ha <- rowAnnotation(Kmeans = as.character(clusters),
                        col = list(Kmeans = c("1" = "#e41a1c",
                                              "2" = "#377eb8",
                                              "3" = "#4daf4a")))
col_fun <- colorRamp2(c(-2, 0, 2), c("#2166ac", "white", "#b2182b"))

ht <- Heatmap(as.matrix(lfc_mat), name = "Z-score",
              col = col_fun,
              left_annotation = row_ha,
              cluster_rows = TRUE,
              cluster_columns = FALSE,
              show_row_names = FALSE,
              show_column_names = TRUE,
              column_title = "Timepoints",
              heatmap_legend_param = list(title = "Z-score"))

p2_B <- grid.grabExpr(draw(ht))

p2_combined <- wrap_elements(p2_A) + wrap_elements(p2_B) +
  plot_layout(widths = c(0.45, 0.55))

save_plot(p2_combined, "Figure2_bulk_gsea", width = 12, height = 5)

# ----------------------------------------------------------------------------
# 6. Figure 3: 空间转录组定位
# ----------------------------------------------------------------------------
cat("[3/6] 生成 Figure 3...\n")

p3_A <- SpatialFeaturePlot(spatial_obj, features = "Ferroptosis",
                           pt.size.factor = 1.5, stroke = 0) +
  scale_fill_gradientn(colours = pal_ferro, name = "Ferroptosis\nscore") +
  ggtitle("A: Ferroptosis score") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right")

p3_B <- SpatialDimPlot(spatial_obj, group.by = "region",
                       pt.size.factor = 1.5, stroke = 0) +
  scale_fill_manual(values = c("Healthy" = "#313695",
                               "Other" = "#ffffbf",
                               "Penumbra" = "#d73027")) +
  ggtitle("B: Tissue region") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right")

violin_data <- spatial_obj@meta.data
p3_C <- ggplot(violin_data, aes(x = region, y = Ferroaging, fill = region)) +
  geom_violin(alpha = 0.5, trim = FALSE) +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  geom_jitter(size = 0.4, alpha = 0.1, width = 0.2) +
  scale_fill_manual(values = c("Healthy" = "#313695",
                               "Other" = "#ffffbf",
                               "Penumbra" = "#d73027")) +
  labs(x = "Region", y = "Ferroaging score", tag = "C") +
  theme(legend.position = "none")

cor_data <- read.csv(paths$spatial_neuron, stringsAsFactors = FALSE)
p3_D <- ggplot(cor_data, aes(x = neuron_prop, y = fp_score)) +
  geom_point(alpha = 0.4, size = 0.8) +
  stat_cor(method = "spearman", size = 3.5) +
  labs(x = "Neuron proportion", y = "Ferroptosis score", tag = "D")
p3_D <- ggMarginal(p3_D, type = "density", margins = "both",
                   size = 5, fill = "grey80")

p3_top <- wrap_elements(p3_A) + wrap_elements(p3_B) + plot_layout(widths = c(1, 1))
p3_bot <- wrap_elements(p3_C) + wrap_elements(p3_D) + plot_layout(widths = c(0.8, 1.2))
p3_combined <- p3_top / p3_bot

save_plot(p3_combined, "Figure3_spatial", width = 12, height = 10)

# ----------------------------------------------------------------------------
# 7. Figure 4: 单细胞核转录组
# ----------------------------------------------------------------------------
cat("[4/6] 生成 Figure 4...\n")

p4_A <- DimPlot(sc_obj, reduction = "umap", group.by = "Celltypes") +
  ggtitle("A: Cell types") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right")

p4_B <- FeaturePlot(sc_obj, features = "Sat1", order = TRUE, pt.size = 0.5) +
  NoAxes() +
  scale_color_viridis_c(option = "A") +
  ggtitle("B: Sat1 expression")

p4_C <- FeaturePlot(sc_obj, features = "Ferroaging_UCell", order = TRUE, pt.size = 0.5) +
  NoAxes() +
  scale_color_viridis_c(option = "A") +
  ggtitle("C: Ferroaging score")

sc_meta <- sc_obj@meta.data %>%
  select(Ferroptosis = Ferroptosis_UCell,
         Senescence = Senescence_UCell,
         Ferroaging = Ferroaging_UCell,
         Celltypes = Celltypes,
         Condition = Condition)
sc_meta$Sat1 <- FetchData(sc_obj, vars = "Sat1")[, 1]

p4_D <- ggplot(sc_meta, aes(x = Ferroptosis, y = Senescence)) +
  geom_point(size = 0.3, alpha = 0.3) +
  stat_cor(method = "spearman", size = 3) +
  labs(x = "Ferroptosis", y = "Senescence", tag = "D")

p4_E <- ggplot(sc_meta, aes(x = Ferroptosis, y = Sat1)) +
  geom_point(size = 0.3, alpha = 0.3) +
  stat_cor(method = "spearman", size = 3) +
  labs(x = "Ferroptosis", y = "Sat1", tag = "E")

p4_F <- ggplot(sc_meta, aes(x = Senescence, y = Sat1)) +
  geom_point(size = 0.3, alpha = 0.3) +
  stat_cor(method = "spearman", size = 3) +
  labs(x = "Senescence", y = "Sat1", tag = "F")

augur <- read.csv(paths$augur_csv, stringsAsFactors = FALSE) %>%
  arrange(desc(AUC))
p4_G <- ggplot(augur, aes(x = reorder(cell_type, AUC), y = AUC)) +
  geom_segment(aes(xend = cell_type, yend = 0), color = "grey50") +
  geom_point(size = 4, color = "#d73027") +
  coord_flip() +
  labs(x = "", y = "AUC (Augur priority)", tag = "G")

p4_top <- p4_A + p4_B + p4_C + plot_layout(design = "AAB\nAAC", widths = c(2, 1, 1))
p4_bot <- (p4_D | p4_E | p4_F) / p4_G + plot_layout(heights = c(2, 1))
p4_combined <- p4_top / p4_bot + plot_annotation(tag_levels = "A")

save_plot(p4_combined, "Figure4_singlecell", width = 14, height = 12)

# ----------------------------------------------------------------------------
# 8. Figure 5: 代谢组 + KEGG
# ----------------------------------------------------------------------------
cat("[5/6] 生成 Figure 5...\n")

# 图 5A: SAT1-多胺轴代谢物 log2FC 瀑布图（基于 cross_omics_axis_table.csv）
sat1_axis <- cross_axis %>%
  filter(axis_name == "SAT1-polyamine") %>%
  mutate(
    display_name = factor(display_name, levels = display_name[order(log2FC_aging)]),
    direction = ifelse(log2FC_aging > 0, "Up", "Down")
  )

if (nrow(sat1_axis) == 0) {
  stop("cross_omics_axis_table.csv 中无 SAT1-polyamine 轴数据")
}

p5_A <- ggplot(sat1_axis, aes(x = display_name, y = log2FC_aging, fill = direction)) +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Down" = "#2166ac", "Up" = "#b2182b"), guide = "none") +
  labs(x = "Metabolite", y = "log2 fold change (59w / 3w)", tag = "A")

p5_B <- ggplot(pathway_axis, aes(x = reorder(Pathway_Axis, Match_Rate),
                                 y = Match_Rate, fill = Evidence_Level)) +
  geom_bar(stat = "identity", width = 0.6) +
  scale_fill_manual(values = c("Moderate" = "#d73027",
                               "Weak" = "#4575b4")) +
  coord_flip() +
  labs(x = "", y = "Match rate", tag = "B") +
  theme(legend.position = "bottom")

p5_C <- ggplot(kegg_summary, aes(x = reorder(pathway_name, cross_omics_score),
                                 y = cross_omics_score)) +
  geom_segment(aes(xend = pathway_name, yend = 0), color = "grey50") +
  geom_point(size = 4, color = "#4575b4") +
  coord_flip() +
  labs(x = "KEGG pathway", y = "Cross-omics coverage score", tag = "C")

p5_combined <- (wrap_elements(p5_A) + (wrap_elements(p5_B) / wrap_elements(p5_C))) +
  plot_layout(widths = c(2, 1))

save_plot(p5_combined, "Figure5_metabolomics", width = 12, height = 8)

# ----------------------------------------------------------------------------
# 9. Figure 1: 多组学整合示意图（导出 4 个子组件 + 尝试拼合）
# ----------------------------------------------------------------------------
cat("[6/6] 生成 Figure 1 子组件...\n")

p1_A <- SpatialFeaturePlot(spatial_obj, features = "Ferroaging",
                           pt.size.factor = 1.6, stroke = 0) +
  scale_fill_gradientn(colours = pal_ferro, name = "Ferroaging\nscore") +
  ggtitle("A: Spatial Ferroaging") +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))

p1_B <- FeaturePlot(sc_obj, features = "Ferroaging_UCell", order = TRUE, pt.size = 0.8) +
  NoAxes() +
  scale_color_viridis_c(option = "A", name = "Ferroaging\nscore") +
  ggtitle("B: Single-cell Ferroaging")

# 图 1C: 取 Top 10 显著差异代谢物
metab_top10 <- metab_sig %>%
  slice_max(order_by = abs(log2FC), n = 10) %>%
  arrange(log2FC) %>%
  mutate(metabolite = fct_inorder(metabolite))

p1_C <- ggplot(metab_top10, aes(x = metabolite, y = log2FC, fill = log2FC > 0)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = pal_pos_neg, guide = "none") +
  coord_flip() +
  labs(x = "", y = "log2 FC (59w / 3w)", title = "C: Metabolomics")

# 图 1D: CMap BCP NES
p1_D <- ggplot(fgsea_bcp, aes(x = timepoint, y = NES, fill = NES > 0)) +
  geom_col(width = 0.6) +
  scale_fill_manual(values = pal_pos_neg, guide = "none") +
  facet_wrap(~ pathway, nrow = 1) +
  labs(x = "Timepoint", y = "NES (BCP signature)", title = "D: CMap reversal") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 导出独立子图 PDF
save_plot(p1_A, "Figure1_Sub_A_spatial", width = 4, height = 4)
save_plot(p1_B, "Figure1_Sub_B_umap", width = 4, height = 4)
save_plot(p1_C, "Figure1_Sub_C_metab", width = 4, height = 4)
save_plot(p1_D, "Figure1_Sub_D_cmap", width = 6, height = 4)

# patchwork 叙事流拼合：使用 wrap_elements 将含 facet 的子图视为单一块
p1_combined <- wrap_elements(p1_A) + wrap_elements(p1_B) +
  wrap_elements(p1_C) + wrap_elements(p1_D) +
  plot_layout(nrow = 1, widths = c(1, 1, 1, 1.2)) +
  plot_annotation(tag_levels = "A")

save_plot(p1_combined, "Figure1_multimics_integration", width = 16, height = 5)

cat("[完成] 所有图片已输出到: ", out_fig, "\n")
cat("生成文件列表:\n")
print(list.files(out_fig, pattern = "Figure[1-5].*\\.(pdf|svg)$", full.names = TRUE))
