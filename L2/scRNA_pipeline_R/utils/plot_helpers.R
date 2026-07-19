# ============================================================================
# 可视化主题与颜色方案
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
})

theme_pub <- function(base_size = 12) {
  ggplot2::theme_bw(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      panel.grid.major = ggplot2::element_line(linewidth = 0.3, color = "grey90"),
      panel.border = ggplot2::element_rect(linewidth = 0.8, colour = "black"),
      axis.text = ggplot2::element_text(color = "black"),
      legend.key = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
      strip.background = ggplot2::element_rect(fill = "grey95", linewidth = 0.3)
    )
}

CONDITION_COLORS <- c(
  "Ctrl" = "#4DBBD5",
  "1DPI" = "#E64B35",
  "3DPI" = "#F39B7F",
  "7DPI" = "#8491B4"
)

CELLTYPE_PALETTE <- c(
  "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
  "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
  "#FDB462", "#BEBADA", "#FB8072", "#80B1D3", "#B3DE69"
)

DIVERGE_PALETTE <- c("#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
                     "#FDDBC7", "#F4A582", "#D6604D", "#B2182B")

safe_color <- function(labels, palette = CELLTYPE_PALETTE) {
  n <- length(labels)
  cols <- if (n <= length(palette)) palette[seq_len(n)] else
    colorRampPalette(palette)(n)
  names(cols) <- as.character(labels)
  return(cols)
}
