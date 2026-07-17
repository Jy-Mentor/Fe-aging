##############################################################################
# survival-skill 功能验证脚本
# 项目无患者生存数据 → 用 survival::lung 真实 NCCTG 肺癌数据测试
##############################################################################

suppressPackageStartupMessages({
  library(survival); library(survminer); library(ggplot2)
  library(dplyr); library(readr); library(Cairo)
})
stopifnot(requireNamespace("survival", quietly=TRUE),
          requireNamespace("survminer", quietly=TRUE),
          requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Survival Skill Test\n")
cat("========================================\n")
cat("  NOTE: Project has no patient survival data.\n")
cat("  Using survival::lung (real NCCTG lung cancer data, 228 patients)\n")
cat("========================================\n\n")

# ---- 1. 加载真实 NCCTG 数据 ----
cat("--- Loading real NCCTG lung data ---\n")
data(lung, package="survival")
lung <- lung %>%
  mutate(sex = factor(sex, levels=c(1,2), labels=c("Male","Female")),
         ECOG = factor(ph.ecog))
stopifnot(nrow(lung) > 0, all(c("time","status","sex") %in% names(lung)))
cat(sprintf("  Patients: %d | Events: %d | Censored: %d\n",
            nrow(lung), sum(lung$status==2), sum(lung$status==1)))
cat(sprintf("  Male: %d | Female: %d\n",
            sum(lung$sex=="Male"), sum(lung$sex=="Female")))
cat(sprintf("  Median follow-up: %.1f days\n", median(lung$time)))

# ---- 2. KM 曲线 ----
cat("\n--- Kaplan-Meier by Sex ---\n")
fit <- survfit(Surv(time, status) ~ sex, data=lung)
logrank <- survdiff(Surv(time, status) ~ sex, data=lung)
p_val <- 1 - pchisq(logrank$chisq, length(logrank$n)-1)
cat(sprintf("  Log-rank p = %.2e (chi²=%.2f)\n", p_val, logrank$chisq))
cat(sprintf("  Median survival: Male=%.0f days | Female=%.0f days\n",
            summary(fit)$table[1,"median"], summary(fit)$table[2,"median"]))

png_path <- file.path(OUTDIR, "survival_km_lung_test.png")
pdf_path <- file.path(OUTDIR, "survival_km_lung_test.pdf")

p_surv <- ggsurvplot(fit, data=lung,
                     pval=TRUE, pval.method=TRUE,
                     conf.int=TRUE,
                     risk.table=TRUE,
                     risk.table.col="strata",
                     risk.table.height=0.25,
                     legend.labs=c("Male","Female"),
                     legend.title="Sex",
                     xlab="Time (days)", ylab="Survival Probability",
                     title="NCCTG Lung Cancer Survival by Sex (real data)",
                     palette=c("#00599B","#D55E00"),
                     ggtheme=theme_bw(base_size=10),
                     surv.median.line="hv")

png(png_path, width=8, height=8, units="in", res=300, bg="white")
print(p_surv, newpage=TRUE)
dev.off()

Cairo::CairoPDF(pdf_path, width=8, height=8)
print(p_surv, newpage=TRUE)
dev.off()

cat(sprintf("  -> %s (%.0f KB)\n", png_path, file.info(png_path)$size/1024))
cat(sprintf("  -> %s (%.0f KB)\n", pdf_path, file.info(pdf_path)$size/1024))

# ---- 3. Cox 森林图 ----
cat("\n--- Cox Proportional Hazards ---\n")
cox_fit <- coxph(Surv(time, status) ~ sex + age + ph.ecog, data=lung)
cox_summary <- summary(cox_fit)
cat(sprintf("  Likelihood ratio test p = %.2e\n", cox_summary$sctest[3]))
cat(sprintf("  Concordance = %.3f\n", cox_summary$concordance[1]))

cox_forest_png <- file.path(OUTDIR, "survival_cox_forest_test.png")
cox_forest_pdf <- file.path(OUTDIR, "survival_cox_forest_test.pdf")

p_cox <- ggforest(cox_fit, data=lung,
                  main="Cox PH Model — NCCTG Lung (real data)",
                  cpositions=c(0.02, 0.22, 0.4),
                  fontsize=0.8)

ggsave(cox_forest_png, p_cox, width=8, height=5, dpi=300, bg="white")
ggsave(cox_forest_pdf, p_cox, width=8, height=5, bg="white", device=Cairo::CairoPDF)

cat(sprintf("  -> %s (%.0f KB)\n", cox_forest_png, file.info(cox_forest_png)$size/1024))
cat(sprintf("  -> %s (%.0f KB)\n", cox_forest_pdf, file.info(cox_forest_pdf)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  Data: NCCTG lung (228 real patients)\n"))
cat(sprintf("  KM log-rank p = %.2e\n", p_val))
cat(sprintf("  Cox concordance = %.3f\n", cox_summary$concordance[1]))
cat("  Survival skill test PASSED.\n")
cat("  NOTE: When project patient survival data available, replace lung with project CSV.\n")
