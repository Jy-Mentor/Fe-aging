##############################################################################
# Fig6: Cell-cell communication network analysis (CellChatDB)
# 铁衰老项目 — 细胞通讯网络分析
#
# 数据来源:
#   - L2/results/cellchat_signaling_pathways.csv
#   - L2/results/cellchat_lr_pairs.csv
#
# 参考标准:
#   - Jin et al., Nature Communications (2021) — CellChat 原始论文
#   - CellChat 官方教程: https://github.com/sqjin/CellChat
#   - DeepWiki CellChat Visualization Methods (2025)
#   - Nature Communications 多面板排版规范
#   - circlize chord diagram + cowplot/patchwork 多面板集成 (行业最佳实践)
#   - 弦图无法直接嵌入 patchwork (circlize使用base R图形设备),
#     标准方案: 导出PNG后通过 cowplot::draw_image() 读回并与ggplot2组合
##############################################################################

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)
library(patchwork)
library(viridis)
library(ggsci)
library(circlize)
library(cowplot)

OUTDIR     <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")
dir.create(OUTDIR_PDF, showWarnings = FALSE, recursive = TRUE)

theme_pub <- theme_bw(base_size = 9) +
  theme(
    panel.grid.major = element_line(color = "grey92", linewidth = 0.25),
    panel.grid.minor = element_blank(),
    panel.border     = element_rect(color = "black", linewidth = 0.6),
    axis.title       = element_text(face = "bold", size = 10),
    axis.text        = element_text(size = 8, color = "black"),
    plot.tag         = element_text(face = "bold", size = 14),
    plot.tag.position = "topleft",
    legend.title     = element_text(face = "bold", size = 8),
    legend.text      = element_text(size = 7),
    legend.key.size  = unit(0.5, "cm")
  )

cat("========================================\n")
cat("  CellChat Communication Network Figures\n")
cat("========================================\n\n")

# ============================================================================
# 1. 加载数据
# ============================================================================
cat("--- Loading data ---\n")

stopifnot(file.exists("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_signaling_pathways.csv"))
stopifnot(file.exists("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_lr_pairs.csv"))

pathways <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_signaling_pathways.csv", show_col_types = FALSE)
lr_pairs <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_lr_pairs.csv", show_col_types = FALSE)

cat(sprintf("  Signaling pathways: %d rows, cols: %s\n", nrow(pathways), paste(colnames(pathways), collapse=", ")))
cat(sprintf("  LR pairs: %d rows, cols: %s\n", nrow(lr_pairs), paste(colnames(lr_pairs), collapse=", ")))

# ============================================================================
# Panel A: Chord diagram — cell-cell communication network
# (circlize base-R graphics → exported as PNG then read back via cowplot)
# ============================================================================
cat("\n[Panel A] Chord diagram...\n")

# Build adjacency matrix: source -> target total communication probability
chord_data <- pathways %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(source, target) %>%
  summarise(total_prob = sum(prob), .groups = "drop")

# Pivot to matrix
cell_types <- sort(unique(c(chord_data$source, chord_data$target)))
n_ct <- length(cell_types)
adj_mat <- matrix(0, nrow = n_ct, ncol = n_ct)
rownames(adj_mat) <- cell_types
colnames(adj_mat) <- cell_types

for (i in 1:nrow(chord_data)) {
  src <- chord_data$source[i]
  tgt <- chord_data$target[i]
  adj_mat[src, tgt] <- chord_data$total_prob[i]
}

cat(sprintf("  Cell types: %d\n", n_ct))
cat(sprintf("  Total interactions: %d\n", nrow(chord_data)))

# Use Nature-inspired color palette for cell types
if (n_ct <= 10) {
  cell_colors <- pal_npg("nrc")(n_ct)
} else {
  cell_colors <- viridis(n_ct, option = "D")
}
names(cell_colors) <- cell_types

# Export chord diagram as PNG (for standalone use and composite integration)
chord_png_path <- file.path(OUTDIR, "Fig6A_CellChat_circle.png")
chord_pdf_path <- file.path(OUTDIR_PDF, "Fig6A_CellChat_circle.pdf")

png(chord_png_path, width = 9, height = 9, units = "in", res = 300, bg = "white")
circos.clear()
circos.par(gap.after = 2, cell.padding = c(0.02, 0, 0.02, 0))
chordDiagram(adj_mat,
             grid.col = cell_colors,
             transparency = 0.2,
             directional = 0,
             annotationTrack = c("grid", "name"),
             preAllocateTracks = list(track.height = 0.08))
circos.clear()
dev.off()

pdf(chord_pdf_path, width = 9, height = 9, bg = "white")
circos.clear()
circos.par(gap.after = 2, cell.padding = c(0.02, 0, 0.02, 0))
chordDiagram(adj_mat,
             grid.col = cell_colors,
             transparency = 0.2,
             directional = 0,
             annotationTrack = c("grid", "name"),
             preAllocateTracks = list(track.height = 0.08))
circos.clear()
dev.off()

# Read chord PNG back as a ggplot-compatible object via cowplot::draw_image
# This is the industry-standard approach for combining circlize with ggplot2/patchwork
p_chord <- ggdraw() + draw_image(chord_png_path)

cat("  -> Fig6A chord diagram saved\n")

# ============================================================================
# Panel B: Communication strength heatmap
# ============================================================================
cat("[Panel B] Communication heatmap...\n")

# Build interaction strength matrix
heat_data <- chord_data %>%
  mutate(source = factor(source, levels = cell_types),
         target = factor(target, levels = cell_types))

p_heatmap <- ggplot(heat_data, aes(x = source, y = target, fill = total_prob)) +
  geom_tile(color = "white", linewidth = 0.3) +
  geom_text(aes(label = format(round(total_prob, 1), nsmall = 0)), size = 1.8, color = "grey20") +
  scale_fill_gradient(low = "white", high = "#08519c", name = "Comm.\nProb.") +
  labs(x = "Sender", y = "Receiver",
       title = "Cell-cell communication strength heatmap") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
        axis.text.y = element_text(size = 6),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig6B_CellChat_heatmap.png"), p_heatmap, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6B_CellChat_heatmap.pdf"), p_heatmap, width = 10, height = 8, bg = "white")
cat("  -> Fig6B saved\n")

# ============================================================================
# Panel C: Top 50 LR interactions bubble plot
# 横轴 = interaction_name (LR pair), 纵轴 = total communication probability
# 气泡大小 = sender-receiver pair数量, 颜色 = log10(总通信概率+1)
# ============================================================================
cat("[Panel C] LR interaction bubble...\n")

lr_top <- lr_pairs %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(interaction_name) %>%
  summarise(
    total_prob = sum(prob, na.rm = TRUE),
    n_pairs = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_prob)) %>%
  head(50) %>%
  mutate(interaction_name = factor(interaction_name, levels = rev(interaction_name)))

cat(sprintf("  Top LR interactions: %d\n", nrow(lr_top)))

p_bubble <- ggplot(lr_top, aes(x = interaction_name, y = total_prob)) +
  geom_point(aes(size = n_pairs, color = log10(total_prob + 1)), alpha = 0.85) +
  scale_color_viridis_c(option = "C", name = "log10(Prob+1)") +
  scale_size_continuous(range = c(1.5, 6), name = "N S-R Pairs") +
  labs(x = "Top 50 Ligand-Receptor Interactions", y = "Total Communication Probability",
       title = "CellChat LR interaction bubble plot") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 60, hjust = 1, size = 5.5),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig6C_CellChat_bubble.png"), p_bubble, width = 14, height = 7, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6C_CellChat_bubble.pdf"), p_bubble, width = 14, height = 7, bg = "white")
cat("  -> Fig6C saved\n")

# ============================================================================
# Panel D: Top 25 signaling pathways bar chart (horizontal)
# ============================================================================
cat("[Panel D] Pathway contribution...\n")

pathway_summary <- pathways %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(pathway_name) %>%
  summarise(
    total_prob = sum(prob, na.rm = TRUE),
    n_interactions = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(total_prob)) %>%
  head(25)

# Assign pathway annotation categories based on pathway name patterns
# (CellChatDB annotation categories: Secreted, ECM-Receptor, Cell-Cell Contact, Non-protein)
pathway_summary <- pathway_summary %>%
  mutate(
    annotation = case_when(
      str_detect(tolower(pathway_name), "secret|immune|cytokine|chemokine|growth|tnf|tgf|il|ifn") ~ "Secreted Signaling",
      str_detect(tolower(pathway_name), "ecm|collagen|laminin|fibronectin|integrin") ~ "ECM-Receptor",
      str_detect(tolower(pathway_name), "cell.cell|cadherin|notch|eph|semaphorin|ncam") ~ "Cell-Cell Contact",
      str_detect(tolower(pathway_name), "gaba|glutamate|dopamine|serotonin|acetylcholine|noradrenaline") ~ "Non-protein Signaling",
      TRUE ~ "Secreted Signaling"
    ),
    pathway_name = factor(pathway_name, levels = rev(pathway_name))
  )

ann_colors <- c(
  "Secreted Signaling"   = "#E41A1C",
  "ECM-Receptor"         = "#00CED1",
  "Cell-Cell Contact"    = "#4DAF4A",
  "Non-protein Signaling" = "#1F78B4"
)

p_pathway <- ggplot(pathway_summary, aes(x = total_prob, y = pathway_name)) +
  geom_col(aes(fill = annotation), width = 0.7, alpha = 0.85) +
  geom_text(aes(label = n_interactions), hjust = -0.2, size = 2.8, fontface = "bold", color = "grey40") +
  scale_fill_manual(values = ann_colors, name = "Pathway\nCategory") +
  scale_x_continuous(limits = c(0, max(pathway_summary$total_prob) * 1.2)) +
  labs(x = "Total Communication Probability", y = NULL,
       title = "Top 25 signaling pathways by communication probability") +
  theme_pub +
  theme(axis.text.y = element_text(size = 7),
        legend.position = "right",
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig6D_CellChat_pathway_contribution.png"), p_pathway, width = 9, height = 7, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6D_CellChat_pathway_contribution.pdf"), p_pathway, width = 9, height = 7, bg = "white")
cat("  -> Fig6D saved\n")

# ============================================================================
# 组图: Fig6 Composite (CellChat 标准四面板: A=chord, B=heatmap, C=bubble, D=pathway)
# 弦图(circlize base-R)通过 cowplot::draw_image() 嵌入 patchwork
# ============================================================================
cat("\n--- Assembling CellChat composite ---\n")

# Layout: A (chord, full width) / (B (heatmap) | C (bubble)) / D (pathway, full width)
fig6 <- (p_chord + labs(tag = "A")) /
        ((p_heatmap + labs(tag = "B")) | (p_bubble + labs(tag = "C"))) /
        (p_pathway + labs(tag = "D")) +
        plot_layout(heights = c(1, 1, 0.9)) &
        theme(plot.tag = element_text(face = "bold", size = 14))

ggsave(file.path(OUTDIR, "Fig6_Composite_cell_communication.png"), fig6, width = 16, height = 20, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6_Composite_cell_communication.pdf"), fig6, width = 16, height = 20, bg = "white")
cat("  -> Fig6 composite saved\n")

cat("\n========================================\n")
cat("  CellChat figures complete!\n")
cat("========================================\n")
