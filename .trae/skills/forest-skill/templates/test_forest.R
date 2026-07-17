##############################################################################
# forest-skill 功能验证脚本
# 用项目真实 external_validation + model_performance 数据测试森林图
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(stringr)
  library(tidyr); library(patchwork); library(ggsci); library(viridis); library(Cairo)
})
stopifnot(requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Forest Skill Test\n")
cat("========================================\n\n")

theme_pub <- theme_bw(base_size=10) +
  theme(
    panel.grid.major=element_line(color="grey92", linewidth=0.25),
    panel.grid.minor=element_blank(),
    panel.border=element_rect(color="black", linewidth=0.6),
    axis.title=element_text(face="bold", size=11),
    axis.text=element_text(size=9, color="black"),
    plot.tag=element_text(face="bold", size=14),
    plot.tag.position="topleft",
    legend.title=element_text(face="bold", size=9),
    legend.text=element_text(size=8)
  )

# ---- 1. 外部验证森林图(Spearman rho + Fisher CI) ----
cat("--- [Type 1] External Validation Forest ---\n")
ext_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/external_validation_results.csv"
stopifnot(file.exists(ext_path))
ext <- read_csv(ext_path, show_col_types=FALSE)
stopifnot(nrow(ext) > 0)
cat(sprintf("  External validation: %d rows | cols: %s\n",
            nrow(ext), paste(names(ext), collapse=", ")))

# 找列名
rho_col  <- intersect(c("Spearman_rho","rho","cor"), names(ext))[1]
p_col    <- intersect(c("Spearman_p","p","pvalue"), names(ext))[1]
n_col    <- intersect(c("N_Valid","N","n","sample_size"), names(ext))[1]
ds_col   <- intersect(c("Dataset","dataset","study"), names(ext))[1]
auc_col  <- intersect(c("FA_AUC","AUC","auc"), names(ext))[1]
stopifnot(!is.na(rho_col), !is.na(p_col), !is.na(n_col), !is.na(ds_col))

fisher_ci <- function(rho, n, alpha=0.05) {
  z <- atanh(rho); se <- 1/sqrt(n-3); zc <- qnorm(1-alpha/2)
  list(lower=tanh(z-zc*se), upper=tanh(z+zc*se))
}

ext_plot <- ext %>%
  filter(!is.na(.data[[rho_col]]), !is.na(.data[[n_col]])) %>%
  rowwise() %>%
  mutate(
    ci = list(fisher_ci(.data[[rho_col]], .data[[n_col]])),
    lower = ci$lower, upper = ci$upper,
    sig = case_when(.data[[p_col]] < 0.001 ~ "***",
                    .data[[p_col]] < 0.01  ~ "**",
                    .data[[p_col]] < 0.05  ~ "*",
                    TRUE ~ "NS")
  ) %>%
  ungroup() %>%
  mutate(Dataset = factor(.data[[ds_col]], levels=rev(.data[[ds_col]])))
cat(sprintf("  Plotted: %d datasets\n", nrow(ext_plot)))

p_ext <- ggplot(ext_plot, aes(x=.data[[rho_col]], y=Dataset)) +
  geom_vline(xintercept=0, linetype="dashed", color="grey50", linewidth=0.4) +
  geom_segment(aes(x=lower, xend=upper, y=Dataset, yend=Dataset),
               color="grey50", linewidth=1.2, alpha=0.6) +
  geom_point(aes(color=.data[[rho_col]],
                 size=if (!is.na(auc_col)) .data[[auc_col]] else 5), alpha=0.9) +
  geom_text(aes(label=sprintf("%.3f [%.2f,%.2f]%s",
                              .data[[rho_col]], lower, upper, sig)),
            hjust=-0.15, size=3.2, fontface="bold") +
  scale_color_viridis_c(option="A", direction=-1, name="Spearman rho") +
  {if (!is.na(auc_col))
    scale_size_continuous(range=c(3,8), name="FA AUC")
   else
    scale_size_identity()} +
  scale_x_continuous(limits=c(min(ext_plot$lower)*0.9, max(ext_plot$upper)*1.25)) +
  labs(x="Spearman Correlation [95% CI via Fisher z]", y=NULL, tag="A",
       title="External Validation of Ferroaging Signature") +
  theme_pub

# ---- 2. 模型性能森林图(AUC/AUPR) ----
cat("\n--- [Type 2] Model Performance Forest ---\n")
mp_path <- "d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/model_performance_v67.csv"
stopifnot(file.exists(mp_path))
mp <- read_csv(mp_path, show_col_types=FALSE)
stopifnot(nrow(mp) > 0)
cat(sprintf("  Model performance: %d rows | cols: %s\n",
            nrow(mp), paste(names(mp), collapse=", ")))

model_col <- intersect(c("model","Model","model_name"), names(mp))[1]
auc_mp <- intersect(c("best_auc","AUC","auc"), names(mp))[1]
aupr_mp <- intersect(c("best_aupr","AUPR","aupr"), names(mp))[1]
stopifnot(!is.na(model_col), !is.na(auc_mp), !is.na(aupr_mp))

mp_long <- mp %>%
  select(.data[[model_col]], .data[[auc_mp]], .data[[aupr_mp]]) %>%
  pivot_longer(c(.data[[auc_mp]], .data[[aupr_mp]]),
               names_to="metric", values_to="value") %>%
  mutate(metric = recode(metric,
                         !!auc_mp := "AUC",
                         !!aupr_mp := "AUPR"),
         model = factor(.data[[model_col]]))

p_mp <- ggplot(mp_long, aes(x=value, y=model, color=metric)) +
  geom_point(size=4, alpha=0.9) +
  geom_segment(aes(x=value, xend=value, y=model, yend=model),
               linewidth=2, alpha=0.4) +
  geom_text(aes(label=sprintf("%.3f", value)), hjust=-0.3, size=3.2, fontface="bold") +
  scale_color_manual(values=c("AUC"="#1f87be", "AUPR"="#e19433"), name="Metric") +
  scale_x_continuous(limits=c(0, 1.15)) +
  geom_vline(xintercept=c(0.5, 0.8, 0.9), linetype="dashed", color="grey60", linewidth=0.3) +
  labs(x="Score", y=NULL, tag="B", title="GNN Model Performance") +
  theme_pub

# ---- 组合 ----
cat("\n--- Assembling composite ---\n")
fig <- (p_ext) / (p_mp) + plot_layout(heights=c(1.2, 1))

out_png <- file.path(OUTDIR, "forest_composite_test.png")
out_pdf <- file.path(OUTDIR, "forest_composite_test.pdf")
ggsave(out_png, fig, width=11, height=8, dpi=300, bg="white")
ggsave(out_pdf, fig, width=11, height=8, bg="white", device=Cairo::CairoPDF)

cat(sprintf("\n[OK] %s (%.0f KB)\n", out_png, file.info(out_png)$size/1024))
cat(sprintf("[OK] %s (%.0f KB)\n", out_pdf, file.info(out_pdf)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  External validation: %d datasets\n", nrow(ext_plot)))
cat(sprintf("  Models: %d\n", nrow(mp)))
cat("  Real data: ext=", nrow(ext), " model_perf=", nrow(mp), "\n", sep="")
cat("  Forest skill test PASSED.\n")
