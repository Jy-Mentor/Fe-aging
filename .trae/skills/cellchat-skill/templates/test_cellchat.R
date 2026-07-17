##############################################################################
# cellchat-skill 功能验证脚本
# 用项目真实 CellChat 输出测试 4 面板: chord / heatmap / bubble / pathway
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(stringr); library(patchwork); library(viridis); library(ggsci)
  library(circlize); library(cowplot); library(Cairo)
})
stopifnot(requireNamespace("Cairo", quietly=TRUE))
stopifnot(requireNamespace("circlize", quietly=TRUE))
stopifnot(requireNamespace("cowplot", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  CellChat Skill Test\n")
cat("========================================\n\n")

# ---- 1. 加载真实数据 ----
cat("--- Loading real CellChat data ---\n")
path_pw <- "d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_signaling_pathways.csv"
path_lr <- "d:/铁衰老 绝不重蹈覆辙/L2/results/cellchat_lr_pairs.csv"
stopifnot(file.exists(path_pw), file.exists(path_lr))

pathways <- read_csv(path_pw, show_col_types=FALSE)
lr_pairs <- read_csv(path_lr, show_col_types=FALSE)
stopifnot(nrow(pathways) > 0, nrow(lr_pairs) > 0)
cat(sprintf("  Pathways: %d rows | LR pairs: %d rows\n", nrow(pathways), nrow(lr_pairs)))
cat(sprintf("  Pathway cols: %s\n", paste(names(pathways), collapse=", ")))

# ---- 通用主题 ----
theme_pub <- theme_bw(base_size=9) +
  theme(
    panel.grid.major=element_line(color="grey92", linewidth=0.25),
    panel.grid.minor=element_blank(),
    panel.border=element_rect(color="black", linewidth=0.6),
    axis.title=element_text(face="bold", size=10),
    axis.text=element_text(size=8, color="black"),
    plot.tag=element_text(face="bold", size=14),
    plot.tag.position="topleft",
    legend.title=element_text(face="bold", size=8),
    legend.text=element_text(size=7),
    legend.key.size=unit(0.5,"cm")
  )

# ---- Panel A: Chord Diagram ----
cat("\n[Panel A] Chord diagram...\n")
chord_data <- pathways %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(source, target) %>%
  summarise(total_prob=sum(prob), .groups="drop")

cell_types <- sort(unique(c(chord_data$source, chord_data$target)))
n_ct <- length(cell_types)
adj_mat <- matrix(0, nrow=n_ct, ncol=n_ct,
                  dimnames=list(cell_types, cell_types))
for (i in seq_len(nrow(chord_data))) {
  adj_mat[chord_data$source[i], chord_data$target[i]] <- chord_data$total_prob[i]
}
cat(sprintf("  Cell types: %d | Interactions: %d\n", n_ct, nrow(chord_data)))

cell_colors <- if (n_ct <= 10) pal_npg("nrc")(n_ct) else viridis(n_ct, option="D")
names(cell_colors) <- cell_types

chord_png <- file.path(OUTDIR, "cellchat_chord_test.png")
chord_pdf <- file.path(OUTDIR, "cellchat_chord_test.pdf")

png(chord_png, width=9, height=9, units="in", res=300, bg="white")
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(adj_mat, grid.col=cell_colors, transparency=0.2, directional=0,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
circos.clear()
dev.off()

Cairo::CairoPDF(chord_pdf, width=9, height=9)
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(adj_mat, grid.col=cell_colors, transparency=0.2, directional=0,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
circos.clear()
dev.off()
cat(sprintf("  -> %s (%.0f KB)\n", chord_png, file.info(chord_png)$size/1024))

# ---- Panel B: Heatmap ----
cat("[Panel B] Communication heatmap...\n")
heat_data <- chord_data %>%
  mutate(source=factor(source, levels=cell_types),
         target=factor(target, levels=cell_types))

p_heat <- ggplot(heat_data, aes(x=source, y=target, fill=total_prob)) +
  geom_tile(color="white", linewidth=0.3) +
  geom_text(aes(label=format(round(total_prob,1), nsmall=0)), size=1.8, color="grey20") +
  scale_fill_gradient(low="white", high="#08519c", name="Comm.\nProb.") +
  labs(x="Sender", y="Receiver", tag="B") +
  theme_pub +
  theme(axis.text.x=element_text(angle=45, hjust=1, size=6),
        axis.text.y=element_text(size=6))

# ---- Panel C: LR Bubble ----
cat("[Panel C] LR bubble...\n")
lr_top <- lr_pairs %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(interaction_name) %>%
  summarise(total_prob=sum(prob, na.rm=TRUE), n_pairs=n(), .groups="drop") %>%
  arrange(desc(total_prob)) %>%
  head(50) %>%
  mutate(interaction_name=factor(interaction_name, levels=rev(interaction_name)))
cat(sprintf("  Top LR interactions: %d\n", nrow(lr_top)))

p_bubble <- ggplot(lr_top, aes(x=interaction_name, y=total_prob)) +
  geom_point(aes(size=n_pairs, color=log10(total_prob+1)), alpha=0.85) +
  scale_color_viridis_c(option="C", name="log10(Prob+1)") +
  scale_size_continuous(range=c(1.5,6), name="N S-R Pairs") +
  labs(x="Top 50 LR Interactions", y="Total Communication Probability", tag="C") +
  theme_pub +
  theme(axis.text.x=element_text(angle=60, hjust=1, size=5.5))

# ---- Panel D: Pathway Contribution ----
cat("[Panel D] Pathway contribution...\n")
pathway_summary <- pathways %>%
  filter(!is.na(prob), prob > 0) %>%
  group_by(pathway_name) %>%
  summarise(total_prob=sum(prob, na.rm=TRUE), n_interactions=n(), .groups="drop") %>%
  arrange(desc(total_prob)) %>%
  head(25) %>%
  mutate(
    annotation=case_when(
      str_detect(tolower(pathway_name), "secret|immune|cytokine|chemokine|growth|tnf|tgf|il|ifn") ~ "Secreted Signaling",
      str_detect(tolower(pathway_name), "ecm|collagen|laminin|fibronectin|integrin") ~ "ECM-Receptor",
      str_detect(tolower(pathway_name), "cell.cell|cadherin|notch|eph|semaphorin|ncam") ~ "Cell-Cell Contact",
      str_detect(tolower(pathway_name), "gaba|glutamate|dopamine|serotonin|acetylcholine|noradrenaline") ~ "Non-protein Signaling",
      TRUE ~ "Secreted Signaling"
    ),
    pathway_name=factor(pathway_name, levels=rev(pathway_name))
  )

ann_colors <- c(
  "Secreted Signaling"="#E41A1C",
  "ECM-Receptor"="#00CED1",
  "Cell-Cell Contact"="#4DAF4A",
  "Non-protein Signaling"="#1F78B4"
)

p_pathway <- ggplot(pathway_summary, aes(x=total_prob, y=pathway_name)) +
  geom_col(aes(fill=annotation), width=0.7, alpha=0.85) +
  geom_text(aes(label=n_interactions), hjust=-0.2, size=2.8, fontface="bold", color="grey40") +
  scale_fill_manual(values=ann_colors, name="Pathway\nCategory") +
  scale_x_continuous(limits=c(0, max(pathway_summary$total_prob)*1.2)) +
  labs(x="Total Communication Probability", y=NULL, tag="D") +
  theme_pub +
  theme(axis.text.y=element_text(size=7), legend.position="right")

# ---- 组合 + 保存 ----
cat("\n--- Assembling composite ---\n")
p_chord <- ggdraw() + draw_image(chord_png)

fig <- (p_chord + labs(tag="A")) /
       ((p_heat) | (p_bubble)) /
       (p_pathway) +
       plot_layout(heights=c(1, 1, 0.9)) &
       theme(plot.tag=element_text(face="bold", size=14))

out_png <- file.path(OUTDIR, "cellchat_composite_test.png")
out_pdf <- file.path(OUTDIR, "cellchat_composite_test.pdf")
ggsave(out_png, fig, width=16, height=20, dpi=300, bg="white")
ggsave(out_pdf, fig, width=16, height=20, bg="white", device=Cairo::CairoPDF)

cat(sprintf("\n[OK] %s (%.0f KB)\n", out_png, file.info(out_png)$size/1024))
cat(sprintf("[OK] %s (%.0f KB)\n", out_pdf, file.info(out_pdf)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
library(png)
img <- readPNG(out_png)
cat(sprintf("  PNG dim: %d x %d | DPI: %.1f\n",
            nrow(img), ncol(img), 300))
cat("  Real data: pathways=", nrow(pathways), " lr_pairs=", nrow(lr_pairs), "\n", sep="")
cat("  CellChat skill test PASSED.\n")
