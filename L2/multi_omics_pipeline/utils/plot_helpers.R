# ============================================================================
# 绘图辅助函数 (plot_helpers.R)
# - ggplot2 出版级主题
# - 配色方案 (条件/细胞类型/发散/连续)
# - 通用可视化模板
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
})

# ----------------------------------------------------------------------------
# 出版级 ggplot2 主题
# ----------------------------------------------------------------------------
theme_pub <- function(base_size = 11, base_family = "sans") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      plot.title = element_text(size = base_size + 2, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = base_size, hjust = 0.5, color = "grey30"),
      axis.title = element_text(size = base_size, face = "bold"),
      axis.text = element_text(size = base_size - 1, color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.6),
      axis.ticks = element_line(color = "black", linewidth = 0.4),
      legend.title = element_text(size = base_size, face = "bold"),
      legend.text = element_text(size = base_size - 1),
      legend.key.size = unit(0.4, "cm"),
      strip.background = element_rect(fill = "grey92", color = "black"),
      strip.text = element_text(size = base_size, face = "bold"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(8, 8, 8, 8)
    )
}

# ----------------------------------------------------------------------------
# 配色方案
# ----------------------------------------------------------------------------
CONDITION_COLORS <- c(
  Ctrl     = "#2166AC",
  Control  = "#2166AC",
  "12h"    = "#92C5DE",
  D1       = "#F4A582",
  "1DPI"   = "#F4A582",
  D3       = "#D6604D",
  "3DPI"   = "#D6604D",
  D7       = "#B2182B",
  "7DPI"   = "#B2182B"
)

CELLTYPE_PALETTE <- c(
  "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
  "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
  "#7E6148", "#B09C85"
)

DIVERGE_PALETTE <- c("#2166AC", "#F7F7F7", "#B2182B")

# 安全配色: 命名向量, 自动补齐缺失值
get_celltype_colors <- function(celltypes) {
  n <- length(celltypes)
  if (n <= length(CELLTYPE_PALETTE)) {
    cols <- CELLTYPE_PALETTE[seq_len(n)]
  } else {
    cols <- colorRampPalette(CELLTYPE_PALETTE)(n)
  }
  names(cols) <- celltypes
  return(cols)
}

get_condition_colors <- function(conds) {
  cols <- CONDITION_COLORS[conds]
  missing <- is.na(cols)
  if (any(missing)) {
    cols[missing] <- colorRampPalette(c("#2166AC", "#B2182B"))(sum(missing))
  }
  return(cols)
}

# ----------------------------------------------------------------------------
# 通用可视化模板
# ----------------------------------------------------------------------------
# 火山图
plot_volcano <- function(deg_df, logfc_col = "log2FoldChange", padj_col = "padj",
                         gene_col = "gene", highlight_genes = NULL,
                         title = "Volcano plot") {
  df <- deg_df
  df$neg_log10_padj <- -log10(df[[padj_col]])
  df$significant <- ifelse(!is.na(df[[padj_col]]) & df[[padj_col]] < 0.05 &
                             abs(df[[logfc_col]]) >= 1, "Significant", "NS")
  df$highlight <- ifelse(df[[gene_col]] %in% highlight_genes, "Highlight", "Other")

  p <- ggplot(df, aes(x = .data[[logfc_col]], y = neg_log10_padj,
                       color = significant, alpha = highlight)) +
    geom_point(size = 1.2) +
    scale_color_manual(values = c("Significant" = "#B2182B", "NS" = "grey70")) +
    scale_alpha_manual(values = c("Highlight" = 1, "Other" = 0.5)) +
    geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "grey50") +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "grey50") +
    labs(title = title, x = "log2 Fold Change", y = "-log10(adj.p)") +
    theme_pub() +
    theme(legend.position = "right")

  if (!is.null(highlight_genes)) {
    df_lab <- df[df[[gene_col]] %in% highlight_genes, ]
    p <- p + ggrepel::geom_text_repel(data = df_lab,
                                       aes(label = .data[[gene_col]]),
                                       size = 3, max.overlaps = 20)
  }
  return(p)
}

# 箱线图 (基因/得分 × 分组)
plot_boxplot <- function(df, value_col, group_col, fill_col = NULL,
                         title = "", ylab = value_col, xlab = group_col) {
  p <- ggplot(df, aes(x = .data[[group_col]], y = .data[[value_col]],
                       fill = if (!is.null(fill_col)) .data[[fill_col]] else .data[[group_col]])) +
    geom_boxplot(outlier.size = 0.3, outlier.alpha = 0.3) +
    labs(title = title, x = xlab, y = ylab) +
    theme_pub(base_size = 10) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  if (!is.null(fill_col)) {
    p <- p + labs(fill = fill_col)
  } else {
    p <- p + guides(fill = "none")
  }
  return(p)
}

# 时间序列折线图 (NES/GSVA score 随时间变化)
plot_timeseries <- function(df, time_col, value_col, group_col = NULL,
                            title = "", ylab = value_col) {
  p <- ggplot(df, aes(x = .data[[time_col]], y = .data[[value_col]],
                       group = if (!is.null(group_col)) .data[[group_col]] else 1,
                       color = if (!is.null(group_col)) .data[[group_col]] else NULL)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2.5) +
    labs(title = title, x = "Time point", y = ylab) +
    theme_pub()
  if (!is.null(group_col)) {
    p <- p + labs(color = group_col)
  }
  return(p)
}

# 热图通用接口 (基于 pheatmap 或 ComplexHeatmap)
plot_heatmap <- function(mat, annotation_col = NULL, annotation_row = NULL,
                         cluster_rows = TRUE, cluster_cols = TRUE,
                         show_rownames = TRUE, show_colnames = TRUE,
                         color_palette = NULL, title = "") {
  if (!requireNamespace("pheatmap", quietly = TRUE)) {
    stop("pheatmap package required for plot_heatmap")
  }
  if (is.null(color_palette)) {
    color_palette <- colorRampPalette(DIVERGE_PALETTE)(100)
  }
  p <- pheatmap::pheatmap(
    mat,
    annotation_col = annotation_col,
    annotation_row = annotation_row,
    cluster_rows = cluster_rows,
    cluster_cols = cluster_cols,
    show_rownames = show_rownames,
    show_colnames = show_colnames,
    color = color_palette,
    main = title,
    fontsize = 9,
    border_color = NA
  )
  return(p)
}

# Venn 图 (最多 4 组)
plot_venn <- function(sets, names = NULL, title = "") {
  if (!requireNamespace("VennDiagram", quietly = TRUE)) {
    log_warn("VennDiagram not installed; using UpSetR fallback")
    return(NULL)
  }
  if (is.null(names)) names <- names(sets)
  f <- tempfile(fileext = ".png")
  VennDiagram::venn.diagram(
    x = sets, category.names = names,
    filename = f, imagetype = "png",
    output = TRUE, main = title,
    fill = CELLTYPE_PALETTE[seq_along(sets)],
    cat.cex = 1.2, cex = 1.2, fontfamily = "sans"
  )
  return(f)
}
