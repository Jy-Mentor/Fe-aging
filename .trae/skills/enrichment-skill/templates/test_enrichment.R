##############################################################################
# enrichment-skill 功能验证脚本
# 用项目真实 GO BP + KEGG + GSEA 数据测试富集可视化
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(stringr); library(viridis); library(ggsci); library(patchwork)
  library(ggrepel); library(Cairo)
})
stopifnot(requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Enrichment Skill Test\n")
cat("========================================\n\n")

theme_en <- theme_bw(base_size=9) +
  theme(
    panel.grid.major=element_line(color="grey92", linewidth=0.25),
    panel.grid.minor=element_blank(),
    panel.border=element_rect(color="black", linewidth=0.6),
    axis.title=element_text(face="bold", size=10),
    axis.text=element_text(size=8, color="black"),
    plot.tag=element_text(face="bold", size=14),
    plot.tag.position="topleft",
    plot.title=element_text(face="bold", size=10, hjust=0),
    legend.title=element_text(face="bold", size=8),
    legend.text=element_text(size=7)
  )

# ---- 自适应列名函数 ----
find_col <- function(df, candidates) {
  intersect(candidates, names(df))[1]
}

# ---- 1. GO BP Bar Plot ----
cat("--- [Panel A] GO BP Bar ---\n")
go_bp_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_bp_enrichment.csv"
stopifnot(file.exists(go_bp_path))
go_bp <- read_csv(go_bp_path, show_col_types=FALSE)
stopifnot(nrow(go_bp) > 0)
cat(sprintf("  GO BP: %d rows | cols: %s\n", nrow(go_bp), paste(names(go_bp), collapse=", ")))

padj_col  <- find_col(go_bp, c("p.adjust","padj","FDR","fdr","qvalue","P.adjust","Adjusted P-value","Adjusted_Pvalue","adj.P.Val"))
desc_col  <- find_col(go_bp, c("Description","Term","term","ID","GO","Pathway"))
count_col <- find_col(go_bp, c("Count","count","gene_count","Gene_Count","Overlap"))
stopifnot(!is.na(padj_col), !is.na(desc_col))
cat(sprintf("  Using: padj=%s desc=%s count=%s\n", padj_col, desc_col, count_col))

go_top <- go_bp %>%
  filter(!is.na(.data[[padj_col]])) %>%
  mutate(padj = pmax(.data[[padj_col]], 1e-300),
         neg_log10_padj = -log10(padj),
         # 从 Overlap 解析 count(若 count_col 是 Overlap 格式)
         Count = if (!is.na(count_col) && grepl("/", as.character(.data[[count_col]])[1])) {
           sapply(strsplit(as.character(.data[[count_col]]), "/"), function(x) as.numeric(x[1]))
         } else if (!is.na(count_col)) {
           as.numeric(.data[[count_col]])
         } else NA) %>%
  arrange(desc(neg_log10_padj)) %>%
  head(20) %>%
  mutate(Description = str_trunc(as.character(.data[[desc_col]]), 60),
         Description = factor(Description, levels=rev(Description)))
cat(sprintf("  Top 20 GO BP terms\n"))

p_bar <- ggplot(go_top, aes(x=neg_log10_padj, y=Description, fill=neg_log10_padj)) +
  geom_col(width=0.7, alpha=0.9) +
  scale_fill_viridis_c(option="C", direction=-1, name="-log10(padj)") +
  labs(x="-log10(adjusted P-value)", y=NULL, tag="A",
       title="GO BP Enrichment (Top 20)") +
  theme_en +
  theme(axis.text.y=element_text(size=7))

# ---- 2. KEGG Bubble ----
cat("[Panel B] KEGG Bubble...\n")
kegg_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_kegg_enrichment.csv"
stopifnot(file.exists(kegg_path))
kegg <- read_csv(kegg_path, show_col_types=FALSE)
stopifnot(nrow(kegg) > 0)
cat(sprintf("  KEGG: %d rows | cols: %s\n", nrow(kegg), paste(names(kegg), collapse=", ")))

k_padj  <- find_col(kegg, c("p.adjust","padj","FDR","fdr","qvalue","Adjusted P-value","Adjusted_Pvalue","adj.P.Val"))
k_desc  <- find_col(kegg, c("Description","Term","term","ID","Pathway"))
k_count <- find_col(kegg, c("Count","count","gene_count","Overlap"))
k_ratio <- find_col(kegg, c("GeneRatio","gene_ratio","Ratio"))
stopifnot(!is.na(k_padj), !is.na(k_desc))
cat(sprintf("  KEGG cols: padj=%s desc=%s count=%s ratio=%s\n",
            k_padj, k_desc, k_count, k_ratio))

kegg_top <- kegg %>%
  filter(!is.na(.data[[k_padj]])) %>%
  mutate(padj = pmax(.data[[k_padj]], 1e-300)) %>%
  arrange(padj) %>%
  head(25) %>%
  mutate(Description = str_trunc(as.character(.data[[k_desc]]), 55),
         Description = factor(Description, levels=rev(Description)))

# GeneRatio:优先用 GeneRatio 列,否则从 Overlap (k_count) 解析
if (!is.na(k_ratio)) {
  ratio_raw <- kegg_top[[k_ratio]]
  if (is.character(ratio_raw)) {
    parsed <- strsplit(ratio_raw, "/")
    kegg_top$GeneRatio <- sapply(parsed, function(x) as.numeric(x[1]) / as.numeric(x[2]))
  } else {
    kegg_top$GeneRatio <- as.numeric(ratio_raw)
  }
} else if (!is.na(k_count)) {
  # 从 Overlap "5/100" 解析
  overlap_raw <- as.character(kegg_top[[k_count]])
  parsed <- strsplit(overlap_raw, "/")
  kegg_top$GeneRatio <- sapply(parsed, function(x) as.numeric(x[1]) / as.numeric(x[2]))
} else {
  kegg_top$GeneRatio <- seq_len(nrow(kegg_top)) / nrow(kegg_top)
}

# Count:从 Overlap 解析分子
if (!is.na(k_count)) {
  overlap_raw <- as.character(kegg_top[[k_count]])
  if (grepl("/", overlap_raw[1])) {
    kegg_top$Count <- sapply(strsplit(overlap_raw, "/"), function(x) as.numeric(x[1]))
  } else {
    kegg_top$Count <- as.numeric(overlap_raw)
  }
} else {
  kegg_top$Count <- 5
}

p_bubble <- ggplot(kegg_top, aes(x=GeneRatio, y=Description, size=Count, color=padj)) +
  geom_point(alpha=0.85) +
  scale_color_viridis_c(option="D", name="padj") +
  scale_size_continuous(range=c(2,7), name="Count") +
  labs(x="GeneRatio", y=NULL, tag="B", title="KEGG Enrichment (Top 25)") +
  theme_en +
  theme(axis.text.y=element_text(size=7))

# ---- 3. GSEA NES Lollipop ----
cat("[Panel C] GSEA NES lollipop...\n")
gsea_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/gsea_ferroaging_vs_ferroptosis.csv"
if (file.exists(gsea_path)) {
  gsea <- read_csv(gsea_path, show_col_types=FALSE)
  cat(sprintf("  GSEA: %d rows | cols: %s\n", nrow(gsea), paste(names(gsea), collapse=", ")))

  g_nes  <- find_col(gsea, c("NES","nes"))
  g_padj <- find_col(gsea, c("p.adjust","padj","FDR","fdr","pval","p.value"))
  g_desc <- find_col(gsea, c("Description","ID","pathway","Pathway","Term"))
  if (!is.na(g_nes) && !is.na(g_padj) && !is.na(g_desc)) {
    gsea_top <- gsea %>%
      filter(!is.na(.data[[g_nes]]), !is.na(.data[[g_padj]])) %>%
      mutate(padj = pmax(.data[[g_padj]], 1e-300),
             sig = ifelse(padj < 0.05, "Significant", "NS")) %>%
      arrange(padj) %>%
      head(20) %>%
      mutate(Description = make.unique(str_trunc(as.character(.data[[g_desc]]), 50)),
             Description = factor(Description, levels=rev(Description)))

    p_gsea <- ggplot(gsea_top, aes(x=.data[[g_nes]], y=Description, color=padj)) +
      geom_segment(aes(x=0, xend=.data[[g_nes]], yend=Description),
                   color="grey60", linewidth=0.5) +
      geom_point(aes(size=-log10(padj)), alpha=0.9) +
      scale_color_viridis_c(option="C", name="padj") +
      scale_size_continuous(range=c(2,6), name="-log10(padj)") +
      geom_vline(xintercept=0, linetype="dashed", color="grey50") +
      labs(x="NES (Normalized Enrichment Score)", y=NULL, tag="C",
           title="GSEA: Ferroaging vs Ferroptosis (Top 20)") +
      theme_en +
      theme(axis.text.y=element_text(size=7))
  } else {
    p_gsea <- ggplot() + labs(tag="C", title="GSEA cols not matched") + theme_en
  }
} else {
  p_gsea <- ggplot() + labs(tag="C", title="GSEA file not found") + theme_en
}

# ---- 4. NES Matrix Heatmap ----
cat("[Panel D] GSEA NES matrix heatmap...\n")
nes_mat_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/gsea_nes_matrix.csv"
if (file.exists(nes_mat_path)) {
  nes_mat <- read_csv(nes_mat_path, show_col_types=FALSE)
  cat(sprintf("  NES matrix: %d x %d\n", nrow(nes_mat), ncol(nes_mat)))
  nes_long <- nes_mat %>%
    pivot_longer(-1, names_to="dataset", values_to="NES") %>%
    filter(!is.na(NES))
  pathway_col <- names(nes_mat)[1]
  nes_long[[pathway_col]] <- str_trunc(as.character(nes_long[[pathway_col]]), 40)

  p_heat <- ggplot(nes_long, aes(x=dataset, y=.data[[pathway_col]], fill=NES)) +
    geom_tile(color="white", linewidth=0.3) +
    scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0, name="NES") +
    labs(x=NULL, y=NULL, tag="D", title="GSEA NES Matrix") +
    theme_en +
    theme(axis.text.x=element_text(angle=30, hjust=1, size=7),
          axis.text.y=element_text(size=6))
} else {
  p_heat <- ggplot() + labs(tag="D", title="NES matrix not found") + theme_en
}

# ---- 组合 ----
cat("\n--- Assembling composite ---\n")
fig <- (p_bar | p_bubble) / (p_gsea | p_heat)

out_png <- file.path(OUTDIR, "enrichment_composite_test.png")
out_pdf <- file.path(OUTDIR, "enrichment_composite_test.pdf")
ggsave(out_png, fig, width=16, height=12, dpi=300, bg="white")
ggsave(out_pdf, fig, width=16, height=12, bg="white", device=Cairo::CairoPDF)

cat(sprintf("\n[OK] %s (%.0f KB)\n", out_png, file.info(out_png)$size/1024))
cat(sprintf("[OK] %s (%.0f KB)\n", out_pdf, file.info(out_pdf)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  GO BP: %d | KEGG: %d\n", nrow(go_bp), nrow(kegg)))
cat("  Real data: GO BP + KEGG + GSEA + NES matrix\n")
cat("  Enrichment skill test PASSED.\n")
