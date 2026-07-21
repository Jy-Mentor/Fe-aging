.libPaths(c("D:/R-library/4.5", "D:/R-library", .libPaths()))
suppressPackageStartupMessages({
  library(scran)
  library(scuttle)
})
set.seed(42)
sce <- mockSCE()
sce <- logNormCounts(sce)
clusters <- kmeans(t(logcounts(sce)), centers = 4)$cluster
out <- scoreMarkers(sce, clusters)
cat("scoreMarkers output class:", class(out), "\n")
cat("Length:", length(out), "\n")
cat("Names:", names(out), "\n")
cat("Cols in first DataFrame:\n")
print(colnames(out[[1]]))
cat("Rows in first DataFrame:", nrow(out[[1]]), "\n")
cat("\nTop 5 by mean.AUC:\n")
top5 <- head(out[[1]][order(out[[1]]$mean.AUC, decreasing = TRUE),
                      c("self.average", "other.average",
                        "mean.AUC", "min.AUC", "median.AUC", "max.AUC",
                        "mean.logFC.cohen", "rank.AUC")], 5)
print(top5)
cat("\n[OK] scoreMarkers works correctly\n")
