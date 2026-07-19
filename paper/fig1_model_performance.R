# Figure 1: Model Performance and External Validation
# 铁衰老 GNN 模型性能对比与外部验证
# Style: Nature journal conventions; English labels; CVD-safe palette

library(ggplot2)
library(patchwork)
library(scales)

source("d:/铁衰老 绝不重蹈覆辙/paper/palettes.R")

set.seed(42)
output_dir <- "d:/铁衰老 绝不重蹈覆辙/paper/figures"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Panel A: Model Performance Comparison ----
model_data <- data.frame(
  Model = factor(rep(c("SAGE", "HGT", "SimpleHGN"), each = 2),
                 levels = c("SAGE", "HGT", "SimpleHGN")),
  Metric = rep(c("AUC", "AUPR"), 3),
  Value = c(0.9971, 0.7779, 0.9978, 0.7806, 0.9992, 0.9564)
)

p1 <- ggplot(model_data, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.75),
           width = 0.65, color = "black", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.4f", Value)),
            position = position_dodge(width = 0.75),
            vjust = -0.5, size = 2.8, family = "sans") +
  scale_fill_project("nature_cancer") +
  scale_y_continuous(limits = c(0, 1.08), expand = c(0, 0),
                     breaks = seq(0, 1, 0.2)) +
  labs(title = "A  Model Performance",
       x = NULL, y = "Score", fill = "Metric") +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 10),
    axis.title.y = element_text(size = 10),
    legend.position.inside = c(0.15, 0.88),
    legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
    legend.key.size = unit(0.6, "cm"),
    legend.text = element_text(size = 9),
    legend.title = element_text(size = 9, face = "bold"),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    plot.margin = margin(8, 8, 4, 8)
  )

# ---- Panel B: External Validation ----
ext_data <- data.frame(
  Dataset = factor(c("GSE16561\n(Human, n=63)", "GSE61616\n(Rat, n=15)",
                      "GSE97537\n(Rat, n=12)"),
                   levels = c("GSE16561\n(Human, n=63)", "GSE61616\n(Rat, n=15)",
                              "GSE97537\n(Rat, n=12)")),
  Spearman_rho = c(0.5594, 0.7500, 0.8811),
  FA_AUC = c(0.7359, 1.0000, 1.0000),
  p_value = c(0.0040, 0.0020, 0.0040)
)

p2 <- ggplot(ext_data, aes(x = Dataset)) +
  geom_bar(aes(y = FA_AUC, fill = "AUC"), stat = "identity",
           width = 0.55, alpha = 0.85, color = "black", linewidth = 0.3) +
  geom_point(aes(y = Spearman_rho, shape = "Spearman \u03c1"),
             size = 3.5, color = get_palette("nature_cancer")[8]) +
  geom_text(aes(y = FA_AUC, label = sprintf("%.4f", FA_AUC)),
            vjust = -0.8, size = 2.8, family = "sans") +
  geom_text(aes(y = Spearman_rho, label = sprintf("\u03c1=%.3f", Spearman_rho)),
            vjust = 2.5, size = 2.8, family = "sans", color = get_palette("nature_cancer")[8]) +
  annotate("text", x = 1:3, y = 0.15,
           label = sprintf("p=%.4f", ext_data$p_value),
           size = 2.5, family = "sans", color = "grey40") +
  scale_fill_manual(values = c("AUC" = get_palette("nature_cancer")[4])) +
  scale_shape_manual(values = c("Spearman \u03c1" = 18)) +
  scale_y_continuous(limits = c(0, 1.15), expand = c(0, 0),
                     breaks = seq(0, 1, 0.2)) +
  labs(title = "B  External Validation",
       x = NULL, y = "Score", fill = NULL, shape = NULL) +
  theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    axis.text = element_text(color = "black", size = 9),
    axis.title.y = element_text(size = 10),
    legend.position.inside = c(0.22, 0.90),
    legend.background = element_rect(fill = "white", color = "grey80", linewidth = 0.3),
    legend.key.size = unit(0.5, "cm"),
    legend.text = element_text(size = 9),
    panel.grid.major.y = element_line(color = "grey90", linewidth = 0.3),
    plot.margin = margin(8, 8, 4, 8)
  )

# ---- Combine ----
fig1 <- p1 + p2 + plot_layout(widths = c(1, 1.2))

pdf(file.path(output_dir, "Fig1_model_performance.pdf"), width = 9, height = 4.2)
print(fig1)
dev.off()
png(file.path(output_dir, "Fig1_model_performance.png"), width = 9, height = 4.2,
    units = "in", res = 300)
print(fig1)
dev.off()

cat("Figure 1 saved to:", file.path(output_dir, "Fig1_model_performance.pdf"), "\n")
