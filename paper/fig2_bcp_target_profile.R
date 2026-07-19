# Figure 2: beta-Caryophyllene Target Gene Prediction Profile
# 石竹烯与Top 5壮药平均靶标预测对比
# Style: Nature journal; English labels; CVD-safe palette

library(ggplot2)

source("d:/铁衰老 绝不重蹈覆辙/paper/palettes.R")

set.seed(42)
output_dir <- "d:/铁衰老 绝不重蹈覆辙/paper/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

targets <- c("NFE2L2", "HMOX1", "GPX4", "SLC7A11", "KEAP1",
             "TFRC", "ACSL4", "FTH1", "PTGS2")

# Bata-caryophyllene scores (from tcm_predictions_full_v70_fixed.csv line 17397)
bcp <- c(0.7795, 0.5861, 0.5392, 0.4103, 0.6025,
         0.2879, 0.2837, 0.2909, 0.6223)

# Caryophyllene oxide scores (from zhuangyao_top500 line 113)
bcpo <- c(0.7504, 0.5681, 0.3956, 0.2702, 0.7304,
          0.4322, 0.3855, 0.2870, 0.7421)

# Top 5 zhuangyao average (Belachinal, Naringenin, Fortunellin, Linarin, Moracin E)
top5_avg <- c(0.7509, 0.6621, 0.6242, 0.3902, 0.7402,
              0.3205, 0.3127, 0.3207, 0.7524)

df <- data.frame(
  Target = rep(targets, 3),
  Score = c(bcp, bcpo, top5_avg),
  Group = factor(rep(c("beta-Caryophyllene", "Caryophyllene oxide",
                        "Top 5 Zhuangyao (avg)"), each = 9),
                 levels = c("beta-Caryophyllene", "Caryophyllene oxide",
                            "Top 5 Zhuangyao (avg)"))
)

# Annotation: p-values for key biological comparisons
literature_targets <- c("NFE2L2", "GPX4", "HMOX1", "SLC7A11", "TFRC", "ACSL4")

# Calculate BCP vs Top5 delta for NFE2L2
nfe2l2_bcp <- 0.7795
nfe2l2_top5 <- 0.7509
nfe2l2_delta <- (nfe2l2_bcp - nfe2l2_top5) / nfe2l2_top5 * 100

p <- ggplot(df, aes(x = Target, y = Score, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8),
           width = 0.7, color = "black", linewidth = 0.3) +
  # Highlight NFE2L2 with annotation
  annotate("segment", x = 1, xend = 1,
           y = max(bcp[1], top5_avg[1], bcpo[1]) + 0.03,
           yend = max(bcp[1], top5_avg[1], bcpo[1]) + 0.03,
           linewidth = 0.3) +
  annotate("text", x = 1,
           y = max(bcp[1], top5_avg[1], bcpo[1]) + 0.055,
           label = sprintf("BCP NFE2L2\n+%.1f%% vs Top5", nfe2l2_delta),
           size = 2.8, fontface = "italic", color = get_palette("nature_cancer")[8], lineheight = 0.9) +
  scale_fill_manual(
    values = c("beta-Caryophyllene" = get_palette("nature_cancer")[8],
               "Caryophyllene oxide" = get_palette("nature_cancer")[3],
               "Top 5 Zhuangyao (avg)" = get_palette("nature_cancer")[4])
  ) +
  scale_y_continuous(limits = c(0, 0.95), expand = c(0, 0),
                     breaks = seq(0, 0.9, 0.2)) +
  labs(
    title = "Target Gene Prediction Profile of beta-Caryophyllene",
    subtitle = sprintf("NFE2L2 score (%.4f) exceeds Top 5 Zhuangyao average (%.4f) by %.1f%%",
                       nfe2l2_bcp, nfe2l2_top5, nfe2l2_delta),
    x = "Ferroptosis Target Gene",
    y = "Predicted Interaction Score",
    fill = NULL
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9, color = "grey40"),
    axis.text = element_text(color = "black", size = 9),
    axis.text.x = element_text(face = "bold", angle = 45, hjust = 1),
    axis.title = element_text(size = 10),
    legend.position.inside = c(0.75, 0.88),
    legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
    legend.key.size = unit(0.5, "cm"),
    legend.text = element_text(size = 8.5, face = "italic"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    plot.margin = margin(8, 8, 4, 8)
  )

pdf(file.path(output_dir, "Fig2_bcp_target_profile.pdf"), width = 9, height = 5)
print(p)
dev.off()
png(file.path(output_dir, "Fig2_bcp_target_profile.png"), width = 9, height = 5,
    units = "in", res = 300)
print(p)
dev.off()

cat("Figure 2 saved to:", file.path(output_dir, "Fig2_bcp_target_profile.pdf"), "\n")
