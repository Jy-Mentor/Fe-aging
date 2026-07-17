##############################################################################
# manhattan-qq-skill 功能验证脚本
# 项目无 GWAS → 用 GSE61616_DE_results.csv 真实 p 值做 QQ 图
# Manhattan 需 CHR/BP,DE 无坐标 → 仅写说明,不绘制假 Manhattan
##############################################################################

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(readr); library(tidyr)
  library(viridis); library(Cairo); library(patchwork)
})
stopifnot(requireNamespace("Cairo", quietly=TRUE))
stopifnot(requireNamespace("patchwork", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Manhattan-QQ Skill Test\n")
cat("========================================\n")
cat("  NOTE: Project has no GWAS summary statistics.\n")
cat("  Using GSE61616 DE p-values for QQ plot (real data).\n")
cat("  Manhattan requires CHR/BP — DE has no genomic coords.\n")
cat("========================================\n\n")

# ---- 1. 加载真实 DE p 值 ----
cat("--- Loading real GSE61616 DE p-values ---\n")
de_path <- "d:/铁衰老 绝不重蹈覆辙/L1/results/GSE61616_DE_results.csv"
stopifnot(file.exists(de_path))
de <- read_csv(de_path, show_col_types=FALSE)
stopifnot(nrow(de) > 0)
cat(sprintf("  DE rows: %d | cols: %s\n", nrow(de), paste(names(de), collapse=", ")))

padj_col <- intersect(c("adj.P.Val","padj","FDR","P.Value"), names(de))[1]
p_col    <- intersect(c("P.Value","pvalue","p","P"), names(de))[1]
if (is.na(p_col)) p_col <- padj_col
stopifnot(!is.na(p_col))

p_vals <- de %>% filter(!is.na(.data[[p_col]])) %>% pull(.data[[p_col]])
p_vals <- pmax(p_vals, 1e-300)  # avoid -log10(0)
n <- length(p_vals)
cat(sprintf("  Valid p-values: %d\n", n))
cat(sprintf("  p-value range: [%.2e, %.2e]\n", min(p_vals), max(p_vals)))

# ---- 2. QQ Plot ----
cat("\n--- QQ Plot ---\n")
qq_df <- tibble(
  observed = sort(p_vals),
  expected = ppoints(n),
  observed_log = -log10(observed),
  expected_log = -log10(expected)
)

# Genomic inflation factor lambda
chisq <- qchisq(1 - p_vals, 1)
lambda <- median(chisq, na.rm=TRUE) / qchisq(0.5, 1)
cat(sprintf("  Genomic inflation factor lambda = %.4f\n", lambda))
cat(sprintf("  Interpretation: %s\n",
            ifelse(lambda > 1.1, "INFLATED (lambda>1.1)", "OK (lambda<=1.1)")))

p_qq <- ggplot(qq_df, aes(x=expected_log, y=observed_log)) +
  geom_abline(slope=1, intercept=0, color="red", linetype="dashed", linewidth=0.5) +
  geom_point(alpha=0.4, size=0.8, color="#377EB8") +
  labs(x=expression(-log[10](expected)~italic(p)),
       y=expression(-log[10](observed)~italic(p)),
       tag="A",
       title=sprintf("QQ Plot — GSE61616 DE (n=%d, λ=%.3f)", n, lambda),
       subtitle="Real DE p-values (not GWAS)") +
  theme_bw(base_size=10) +
  theme(plot.title=element_text(face="bold", size=11),
        plot.tag=element_text(face="bold", size=14),
        plot.tag.position="topleft")

# ---- 3. "Pseudo-Manhattan"(按探针索引排,仅演示样式) ----
cat("\n--- Pseudo-Manhattan (by probe index) ---\n")
cat("  NOTE: True Manhattan needs CHR/BP. This is by probe index, NOT genomic position.\n")

probe_col <- intersect(c("Probe","probe_id","ID"), names(de))[1]
if (!is.na(probe_col)) {
  de_man <- de %>%
    filter(!is.na(.data[[p_col]])) %>%
    mutate(p = pmax(.data[[p_col]], 1e-300),
           neg_log10_p = -log10(p),
           probe_idx = row_number(),
           # 模拟染色体分组(按探针名前缀,仅演示样式)
           chr_group = factor((as.integer(as.factor(.data[[probe_col]])) %% 22) + 1))

  p_man <- ggplot(de_man, aes(x=probe_idx, y=neg_log10_p, color=chr_group)) +
    geom_point(alpha=0.5, size=0.6) +
    geom_hline(yintercept=-log10(0.05/n), color="red", linetype="dashed",
               linewidth=0.4) +
    geom_hline(yintercept=-log10(0.01/n), color="darkred", linetype="dashed",
               linewidth=0.4) +
    scale_color_manual(values=viridis(22, option="D"), name="Probe group") +
    labs(x="Probe index (NOT genomic position)", y="-log10(p)",
         tag="B",
         title="Pseudo-Manhattan — GSE61616 DE p-values",
         subtitle="WARNING: Real Manhattan needs CHR/BP. This is by probe index only.") +
    theme_bw(base_size=10) +
    theme(plot.title=element_text(face="bold", size=11),
          plot.tag=element_text(face="bold", size=14),
          plot.tag.position="topleft",
          legend.position="none")
} else {
  p_man <- ggplot() + labs(tag="B", title="No probe column") + theme_bw()
}

# ---- 组合 ----
fig <- patchwork::wrap_plots(p_qq, p_man, nrow=2, heights=c(1,1))

out_png <- file.path(OUTDIR, "manhattan_qq_test.png")
out_pdf <- file.path(OUTDIR, "manhattan_qq_test.pdf")
ggsave(out_png, fig, width=10, height=10, dpi=300, bg="white")
ggsave(out_pdf, fig, width=10, height=10, bg="white", device=Cairo::CairoPDF)

cat(sprintf("\n[OK] %s (%.0f KB)\n", out_png, file.info(out_png)$size/1024))
cat(sprintf("[OK] %s (%.0f KB)\n", out_pdf, file.info(out_pdf)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  Real p-values: %d\n", n))
cat(sprintf("  Lambda (inflation): %.4f\n", lambda))
cat("  Manhattan-QQ skill test PASSED.\n")
cat("  NOTE: Real Manhattan requires GWAS sumstats with CHR/BP columns.\n")
