##############################################################################
# anatomy-skill 功能验证脚本
# 用项目真实 immune_cell_scores_GSE104036.csv 数据测试
# gganatogram 未装 → 回退到器官-细胞型热图
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(viridis); library(Cairo)
})
stopifnot(requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Anatomy Skill Test\n")
cat("========================================\n\n")

# ---- 1. 加载真实免疫数据 ----
cat("--- Loading real immune cell scores ---\n")
immune_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/immune_cell_scores_GSE104036.csv"
stopifnot(file.exists(immune_path))
immune <- read_csv(immune_path, show_col_types=FALSE)
stopifnot(nrow(immune) > 0)
cat(sprintf("  Immune scores: %d rows | cols: %s\n",
            nrow(immune), paste(names(immune), collapse=", ")))
cat(sprintf("  Head:\n"))
print(head(immune))

# ---- 2. 检查 gganatogram 可用性 ----
has_gganato <- requireNamespace("gganatogram", quietly=TRUE)
cat(sprintf("\n  gganatogram: %s\n", ifelse(has_gganato, "AVAILABLE", "NOT INSTALLED (fallback to heatmap)")))

# ---- 3. 数据处理:样本×细胞型矩阵 ----
sample_col <- intersect(c("sample_id","sample","Sample","geo_accession"), names(immune))[1]
if (is.na(sample_col)) sample_col <- names(immune)[1]

# 长格式 → 宽格式
immune_long <- immune %>%
  pivot_longer(-all_of(sample_col), names_to="cell_type", values_to="score") %>%
  filter(!is.na(score))

cell_summary <- immune_long %>%
  group_by(cell_type) %>%
  summarise(mean_score=mean(score, na.rm=TRUE),
            sd_score=sd(score, na.rm=TRUE),
            n=n(), .groups="drop") %>%
  arrange(desc(mean_score))
cat(sprintf("\n  Cell type summary:\n"))
print(cell_summary)

# 项目数据来自 GSE104036 (大鼠 MCAO 脑组织),主要器官 = brain
# 创建器官-细胞型热图(回退方案)
organ_map <- data.frame(
  cell_type = unique(immune_long$cell_type),
  organ = "brain",
  stringsAsFactors = FALSE
)
organ_scores <- immune_long %>%
  left_join(organ_map, by="cell_type") %>%
  group_by(organ, cell_type) %>%
  summarise(mean_score=mean(score, na.rm=TRUE), .groups="drop")

# ---- 4A. gganatogram 可用 → 用真实数据 ----
if (has_gganato) {
  cat("\n--- [Type 1] gganatogram (brain) ---\n")
  suppressPackageStartupMessages(library(gganatogram))

  gganato_df <- organ_scores %>%
    mutate(organ=tolower(organ),
           value=mean_score) %>%
    select(organ, value) %>%
    distinct()

  png_path <- file.path(OUTDIR, "anatomy_gganatogram_test.png")
  pdf_path <- file.path(OUTDIR, "anatomy_gganatogram_test.pdf")

  png(png_path, width=10, height=10, units="in", res=300, bg="white")
  p_ganato <- gganatogram(data=gganato_df, fill="value",
                          organism="human", sex="male",
                          fill_palette=viridis(100)) +
    labs(title="Immune Cell Score — Brain (GSE104036 real data)") +
    theme_bw(base_size=10)
  print(p_ganato)
  dev.off()

  Cairo::CairoPDF(pdf_path, width=10, height=10)
  print(p_ganato)
  dev.off()

  cat(sprintf("  -> %s (%.0f KB)\n", png_path, file.info(png_path)$size/1024))
} else {
  cat("\n--- [Fallback] Organ x Cell-type Bar Chart (single organ → use ggplot2 geom_tile) ---\n")
  cat("  gganatogram not installed AND organ_scores has single row (brain only).\n")
  cat("  Using ggplot2 geom_tile to avoid pheatmap seq.default error on 1-row matrix.\n")

  # 单 organ (brain) — pheatmap 单行矩阵会失败,改用 ggplot2 geom_tile + bar
  png_path <- file.path(OUTDIR, "anatomy_organ_celltile_test.png")
  pdf_path <- file.path(OUTDIR, "anatomy_organ_celltile_test.pdf")

  # 按 mean_score 排序 cell_type
  organ_scores_sorted <- organ_scores %>%
    mutate(cell_type=factor(cell_type, levels=unique(cell_type[order(desc(mean_score))])))

  p_heat <- ggplot(organ_scores_sorted, aes(x=cell_type, y=organ, fill=mean_score)) +
    geom_tile(color="white", linewidth=0.6) +
    geom_text(aes(label=sprintf("%.2f", mean_score)), size=3.2, color="white", fontface="bold") +
    scale_fill_viridis_c(option="D", name="Mean\nScore") +
    labs(x="Immune Cell Type", y="Organ (brain)",
         title="Immune Cell Score by Cell Type — GSE104036 Brain (real data)") +
    theme_bw(base_size=10) +
    theme(axis.text.x=element_text(angle=30, hjust=1, face="bold"),
          plot.title=element_text(face="bold"))

  ggsave(png_path, p_heat, width=10, height=4, dpi=300, bg="white")
  ggsave(pdf_path, p_heat, width=10, height=4, bg="white", device=Cairo::CairoPDF)

  cat(sprintf("  -> %s (%.0f KB)\n", png_path, file.info(png_path)$size/1024))
  cat(sprintf("  -> %s (%.0f KB)\n", pdf_path, file.info(pdf_path)$size/1024))
}

# ---- 5. 按样本×细胞型热图(更详细) ----
cat("\n--- [Type 2] Sample x Cell-type Heatmap ---\n")
sample_cell <- immune_long %>%
  pivot_wider(names_from=cell_type, values_from=score) %>%
  as.data.frame()
rownames(sample_cell) <- sample_cell[[sample_col]]
sample_cell[[sample_col]] <- NULL
sample_mat <- as.matrix(sample_cell)
storage.mode(sample_mat) <- "double"
sample_mat[!is.finite(sample_mat)] <- NA
sample_mat <- sample_mat[, colSums(!is.na(sample_mat)) > 0]

png2 <- file.path(OUTDIR, "anatomy_sample_heatmap_test.png")
pdf2 <- file.path(OUTDIR, "anatomy_sample_heatmap_test.pdf")

has_pheat <- requireNamespace("pheatmap", quietly=TRUE)
if (has_pheat) {
  suppressPackageStartupMessages(library(pheatmap))
  pheatmap(t(sample_mat),
           scale="column",
           clustering_method="ward.D2",
           color=colorRampPalette(c("blue","white","red"))(100),
           show_colnames=FALSE, fontsize_row=8,
           main="Immune Score: Cell Type x Sample (GSE104036)",
           filename=png2, width=10, height=5, dpi=300)
  pheatmap(t(sample_mat),
           scale="column",
           clustering_method="ward.D2",
           color=colorRampPalette(c("blue","white","red"))(100),
           show_colnames=FALSE, fontsize_row=8,
           main="Immune Score: Cell Type x Sample (GSE104036)",
           filename=pdf2, width=10, height=5)
} else {
  sample_long <- as.data.frame(sample_mat) %>%
    tibble::rownames_to_column("sample") %>%
    pivot_longer(-sample, names_to="cell_type", values_to="score")
  p_s <- ggplot(sample_long, aes(x=sample, y=cell_type, fill=score)) +
    geom_tile() +
    scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0) +
    labs(x="Sample", y="Cell Type", title="Immune Score: Cell Type x Sample") +
    theme_bw(base_size=8) +
    theme(axis.text.x=element_blank(), axis.ticks.x=element_blank())
  ggsave(png2, p_s, width=10, height=5, dpi=300, bg="white")
  ggsave(pdf2, p_s, width=10, height=5, bg="white", device=Cairo::CairoPDF)
}
cat(sprintf("  -> %s (%.0f KB)\n", png2, file.info(png2)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  Immune scores: %d samples x %d cell types\n",
            nrow(immune), ncol(immune)-1))
cat(sprintf("  gganatogram: %s\n", ifelse(has_gganato, "USED", "FALLBACK heatmap")))
cat("  Real data: immune_cell_scores_GSE104036.csv\n")
cat("  Anatomy skill test PASSED.\n")
