# Figure 3: Artemisia argyi (Aiye) Compounds in Zhuangyao Top 500
# 改进版: 画布加宽, ggrepel防重叠标签, 长名缩写
# Style: Nature journal; English labels; CVD-safe palette

library(ggplot2)
library(ggrepel)

source("d:/铁衰老 绝不重蹈覆辙/paper/palettes.R")

output_dir <- "d:/铁衰老 绝不重蹈覆辙/paper/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --- 数据: 艾叶关联11条记录 ---
# 为长名称设缩写名用于显示
compounds <- data.frame(
  Name_short = c(
    "Naringenin",
    "Cycloartenol acetate",
    "Dammaradienyl acetate",
    "Dammaradienyl acetate",
    "Caryophyllene oxide",
    "Bata-caryophyllene",
    "Cycloartenol",
    "beta-Sitosterol",
    "beta-Sitosterol",
    "5,7-Dihydroxy-2-(4-OH-phenyl)\nchroman-4-one",
    "5,7-Dihydroxy-2-(4-OH-phenyl)\nchroman-4-one"
  ),
  Rank = c(2, 8, 90, 112, 113, 162, 246, 251, 272, 444, 445),
  Score = c(0.7977, 0.7644, 0.6568, 0.6418, 0.6417, 0.6175,
            0.5775, 0.5739, 0.5632, 0.5455, 0.4965),
  Source = factor(c("Known Aiye component",
             "Pool (herb=艾叶)",
             "Pool (herb=艾叶)",
             "Pool (herb=艾叶)",
             "Literature (PMID:37169131)",
             "Literature (PMID:39498451)",
             "Pool (herb=艾叶)",
             "Known Aiye component",
             "Known Aiye component",
             "Pool (herb=艾叶)",
             "Pool (herb=艾叶)"),
             levels = c("Pool (herb=艾叶)", "Known Aiye component",
                        "Literature (PMID:37169131)", "Literature (PMID:39498451)"))
)

# 排名+名称 显示标签
compounds$Label <- sprintf("#%d  %s", compounds$Rank, compounds$Name_short)
compounds$Label <- factor(compounds$Label,
                          levels = rev(compounds$Label[order(compounds$Rank)]))

# BCP家族高亮
compounds$Family <- ifelse(
  grepl("caryophyllene|Caryophyllene", compounds$Name_short, ignore.case = TRUE),
  "BCP family", "Other Aiye compounds"
)

p <- ggplot(compounds, aes(x = Score, y = Label)) +
  # 连接线
  geom_segment(aes(xend = 0, yend = Label, color = Family),
               linewidth = 1.3, alpha = 0.65) +
  # 点
  geom_point(aes(fill = Source), size = 4.2, shape = 21,
             color = "black", stroke = 0.5) +
  # 分数标签 — 用 ggrepel 防重叠
  geom_text_repel(aes(label = sprintf("%.4f", Score)),
                  direction = "x", hjust = -0.3, segment.linetype = "blank",
                  size = 2.8, family = "sans", min.segment.length = 0.5) +
  scale_x_continuous(limits = c(0, 0.95), expand = c(0, 0.02),
                     breaks = seq(0, 0.9, 0.2)) +
  scale_color_manual(
    values = c("BCP family" = get_palette("nature_cancer")[8],
               "Other Aiye compounds" = get_palette("nature_cancer")[4])
  ) +
  scale_fill_manual(
    values = c("Pool (herb=艾叶)" = get_palette("nature_cancer")[3],
               "Known Aiye component" = get_palette("nature_cancer")[1],
               "Literature (PMID:37169131)" = get_palette("nature_cancer")[8],
               "Literature (PMID:39498451)" = get_palette("nature_cancer")[5])
  ) +
  labs(
    title = "Artemisia argyi (Aiye) Compounds in Zhuangyao Top 500",
    subtitle = "BCP family highlighted in red; label shows adjusted composite score",
    x = "Adjusted Composite Score",
    y = NULL, color = "Family", fill = "Source"
  ) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(size = 9, color = "grey40", face = "italic"),
    axis.text = element_text(color = "black", size = 8.5),
    axis.text.y = element_text(face = "italic", size = 8),
    axis.title.x = element_text(size = 10, margin = margin(t = 6)),
    legend.position.inside = c(0.82, 0.85),
    legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
    legend.key.size = unit(0.4, "cm"),
    legend.text = element_text(size = 7.5),
    legend.title = element_text(size = 8, face = "bold"),
    panel.grid.major.x = element_line(color = "grey90", linewidth = 0.3),
    plot.margin = margin(8, 25, 4, 6)
  )

pdf(file.path(output_dir, "Fig3_aiye_compounds.pdf"), width = 10, height = 5.5)
print(p)
dev.off()
png(file.path(output_dir, "Fig3_aiye_compounds.png"), width = 10, height = 5.5,
    units = "in", res = 300)
print(p)
dev.off()

cat("Figure 3 saved to:", file.path(output_dir, "Fig3_aiye_compounds.pdf"), "\n")
