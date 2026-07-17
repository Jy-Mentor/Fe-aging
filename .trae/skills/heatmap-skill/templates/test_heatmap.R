##############################################################################
# heatmap-skill 功能验证脚本
# 用项目真实 GSE61616 DE 数据测试 pheatmap 聚类热图
##############################################################################

suppressPackageStartupMessages({
  library(pheatmap); library(ggplot2); library(dplyr); library(readr)
  library(viridis); library(RColorBrewer); library(Cairo)
})
stopifnot(requireNamespace("pheatmap", quietly=TRUE))
stopifnot(requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Heatmap Skill Test\n")
cat("========================================\n\n")

# ---- 1. 加载真实数据 ----
cat("--- Loading real GSE61616 DE data ---\n")
de_path   <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv"
expr_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_expression_matrix.csv"
meta_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_sample_meta.csv"
gpl_path  <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GPL1355_probe_to_gene.csv"
stopifnot(file.exists(de_path), file.exists(expr_path),
          file.exists(meta_path), file.exists(gpl_path))

de   <- read_csv(de_path, show_col_types=FALSE)
expr <- read_csv(expr_path, show_col_types=FALSE)
meta <- read_csv(meta_path, show_col_types=FALSE)
gpl  <- read_csv(gpl_path, show_col_types=FALSE)
stopifnot(nrow(de) > 0, nrow(expr) > 0, nrow(meta) > 0)
cat(sprintf("  DE: %d | Expr: %d x %d | Meta: %d | GPL: %d\n",
            nrow(de), nrow(expr), ncol(expr), nrow(meta), nrow(gpl)))
cat(sprintf("  DE cols: %s\n", paste(names(de), collapse=", ")))
cat(sprintf("  Meta cols: %s\n", paste(names(meta), collapse=", ")))

# ---- 2. 选 Top 50 DE 基因(用 gene-level 结果) ----
cat("\n--- Selecting Top 50 DE genes ---\n")
# 优先用 gene-level DE 文件(已映射基因符号)
de_gene_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_gene_level.csv"
if (file.exists(de_gene_path)) {
  de_gene <- read_csv(de_gene_path, show_col_types=FALSE)
  cat(sprintf("  Using gene-level DE: %d rows | cols: %s\n",
              nrow(de_gene), paste(names(de_gene), collapse=", ")))
  gene_col_de <- intersect(c("GeneSymbol","Gene","Symbol","gene","gene_symbol"), names(de_gene))[1]
  lfc_col   <- intersect(c("logFC","log2FoldChange"), names(de_gene))[1]
  padj_col  <- intersect(c("adj.P.Val","padj","FDR"), names(de_gene))[1]
  stopifnot(!is.na(gene_col_de), !is.na(lfc_col), !is.na(padj_col))

  top_genes <- de_gene %>%
    filter(!is.na(.data[[padj_col]]), !is.na(.data[[lfc_col]])) %>%
    mutate(sig = .data[[padj_col]] < 0.01 & abs(.data[[lfc_col]]) > 1) %>%
    filter(sig) %>%
    arrange(.data[[padj_col]]) %>%
    head(50) %>%
    pull(.data[[gene_col_de]])
  cat(sprintf("  Top DE genes: %d\n", length(top_genes)))
} else {
  # 回退:用探针级 DE + GPL 映射
  probe_col <- intersect(c("Probe","probe_id","ID"), names(de))[1]
  lfc_col   <- intersect(c("logFC","log2FoldChange"), names(de))[1]
  padj_col  <- intersect(c("adj.P.Val","padj","FDR"), names(de))[1]
  stopifnot(!is.na(probe_col), !is.na(lfc_col), !is.na(padj_col))
  probe_col_gpl <- intersect(c("Probe","probe_id","ID"), names(gpl))[1]
  gene_col_gpl  <- intersect(c("GeneSymbol","Gene","Symbol"), names(gpl))[1]
  probe2gene <- setNames(gpl[[gene_col_gpl]], gpl[[probe_col_gpl]])
  top_probes <- de %>%
    filter(!is.na(.data[[padj_col]]), !is.na(.data[[lfc_col]])) %>%
    mutate(sig = .data[[padj_col]] < 0.01 & abs(.data[[lfc_col]]) > 1) %>%
    filter(sig) %>% arrange(.data[[padj_col]]) %>% head(50) %>% pull(.data[[probe_col]])
  top_genes <- na.omit(probe2gene[top_probes])
  cat(sprintf("  Top DE genes (via probe mapping): %d\n", length(top_genes)))
}

# ---- 3. 表达矩阵处理(按基因符号过滤) ----
# expression_matrix.csv 第一列可能是 GeneSymbol 或 Probe
expr_id_col <- names(expr)[1]
cat(sprintf("  Expr ID column: %s\n", expr_id_col))
expr_sub <- expr %>%
  filter(.data[[expr_id_col]] %in% top_genes) %>%
  as.data.frame()
rownames(expr_sub) <- expr_sub[[expr_id_col]]
expr_sub[[expr_id_col]] <- NULL

# 数值化 + 过滤 NA/Inf
expr_mat <- as.matrix(expr_sub)
storage.mode(expr_mat) <- "double"
expr_mat[!is.finite(expr_mat)] <- NA
expr_mat <- expr_mat[complete.cases(expr_mat), ]
cat(sprintf("  Expression matrix after QC: %d genes x %d samples\n",
            nrow(expr_mat), ncol(expr_mat)))

# 行名已是基因符号(因 expression_matrix 第一列是 GeneSymbol)
# 去重(同名基因保留高方差行)
gene_syms <- rownames(expr_mat)
vars <- apply(expr_mat, 1, var, na.rm=TRUE)
ord <- order(vars, decreasing=TRUE)
expr_mat <- expr_mat[ord, ]
gene_syms <- gene_syms[ord]
dup_genes <- duplicated(gene_syms)
expr_mat <- expr_mat[!dup_genes, ]
rownames(expr_mat) <- gene_syms[!dup_genes]
cat(sprintf("  Final genes (dedup): %d\n", nrow(expr_mat)))

# ---- 4. 样本注释 ----
sample_col_meta <- intersect(c("sample_id","sample","Sample","geo_accession"), names(meta))[1]
cond_col_meta <- intersect(c("condition","Condition","group","Group","title"), names(meta))[1]
stopifnot(!is.na(sample_col_meta), !is.na(cond_col_meta))

# 匹配样本
meta_match <- meta[match(colnames(expr_mat), meta[[sample_col_meta]]), ]
cond <- meta_match[[cond_col_meta]]
# 简化 condition 标签 (Sham/MCAO 保留,其他如 XST 保留原标签)
cond <- ifelse(grepl("sham|Sham|SHAM|control|Control", cond), "Sham",
               ifelse(grepl("MCAO|mcao|ischem|Ischem", cond), "MCAO",
                      ifelse(is.na(cond), "Unknown", as.character(cond))))
annot_col <- data.frame(Condition = factor(cond), row.names = colnames(expr_mat))
# 动态生成 annotation_colors (覆盖所有出现的 condition,避免 XST 等未匹配)
cond_levels <- levels(annot_col$Condition)
cond_levels <- cond_levels[cond_levels != "Unknown"]
n_lev <- length(cond_levels)
if (n_lev <= 2) {
  cond_palette <- c(Sham="#6FB2C1", MCAO="#E07524")
  miss <- setdiff(cond_levels, names(cond_palette))
  if (length(miss) > 0) {
    cond_palette[miss] <- viridis(length(miss), option="D")
  }
  cond_palette <- cond_palette[cond_levels]
} else {
  cond_palette <- viridis(n_lev, option="D")
  names(cond_palette) <- cond_levels
}
annot_colors <- list(Condition = cond_palette)
cat(sprintf("  Conditions: %s\n", paste(unique(cond), collapse=", ")))

# ---- 5. pheatmap 绘图 ----
cat("\n--- Drawing pheatmap ---\n")
png_path <- file.path(OUTDIR, "heatmap_pheatmap_test.png")
pdf_path <- file.path(OUTDIR, "heatmap_pheatmap_test.pdf")

pheatmap(expr_mat,
         scale="row",
         cluster_rows=TRUE, cluster_cols=TRUE,
         clustering_distance_rows="euclidean",
         clustering_distance_cols="euclidean",
         clustering_method="ward.D2",
         color=colorRampPalette(c("blue","white","red"))(100),
         annotation_col=annot_col,
         annotation_colors=annot_colors,
         show_colnames=FALSE,
         fontsize_row=7,
         main="GSE61616 Top DE Genes Heatmap (ward.D2 + Z-score)",
         filename=png_path, width=10, height=8)

# PDF via Cairo
pheatmap(expr_mat,
         scale="row",
         cluster_rows=TRUE, cluster_cols=TRUE,
         clustering_distance_rows="euclidean",
         clustering_distance_cols="euclidean",
         clustering_method="ward.D2",
         color=colorRampPalette(c("blue","white","red"))(100),
         annotation_col=annot_col,
         annotation_colors=annot_colors,
         show_colnames=FALSE,
         fontsize_row=7,
         main="GSE61616 Top DE Genes Heatmap (ward.D2 + Z-score)",
         filename=pdf_path, width=10, height=8)

cat(sprintf("  -> %s (%.0f KB)\n", png_path, file.info(png_path)$size/1024))
cat(sprintf("  -> %s (%.0f KB)\n", pdf_path, file.info(pdf_path)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  Matrix: %d genes x %d samples\n", nrow(expr_mat), ncol(expr_mat)))
cat(sprintf("  Clustering: ward.D2 + euclidean\n"))
cat(sprintf("  Scale: row Z-score\n"))
cat("  Real data: DE=", nrow(de), " Expr=", nrow(expr), " Meta=", nrow(meta), "\n", sep="")
cat("  Heatmap skill test PASSED.\n")
