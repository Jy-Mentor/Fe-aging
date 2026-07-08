# Microglia subclustering and ferro-aging high-load subpopulation analysis.
# Input: Seurat object with ferro-aging scores.

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))

set.seed(42)

# ---- Paths ----
rds_path <- "L2/results/GSE233815_sn/Seurat_sn_MCAO_with_ferroaging_score.rds"
out_dir <- "L2/results/GSE233815_sn/microglia_subcluster"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load and subset ----
message("Loading Seurat object...")
seu <- readRDS(rds_path)
mg <- subset(seu, subset = cell_class == "Microglia")
message("Microglia cells: ", ncol(mg))

# ---- Re-process microglia subset ----
message("Re-normalizing and finding variable features...")
mg <- NormalizeData(mg)
mg <- FindVariableFeatures(mg, selection.method = "vst", nfeatures = 2000)
mg <- ScaleData(mg)
mg <- RunPCA(mg, features = VariableFeatures(object = mg))

message("Running UMAP and clustering...")
mg <- FindNeighbors(mg, dims = 1:15)
mg <- FindClusters(mg, resolution = 0.6)
mg <- RunUMAP(mg, dims = 1:15)

message("Microglia clusters: ")
print(table(mg$seurat_clusters))

# ---- Ferro-aging score per cluster ----
message("\nComputing ferro-aging scores per microglia cluster...")
score_cols <- c("AddModuleScore_FA95", "AddModuleScore_FA96", "FA_95_UCell", "FA_96_UCell")

cluster_summary <- mg@meta.data %>%
  pivot_longer(cols = all_of(score_cols), names_to = "score_method", values_to = "score") %>%
  group_by(seurat_clusters, Condition, score_method) %>%
  summarise(
    n_cells = n(),
    mean_score = mean(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    sd_score = sd(score, na.rm = TRUE),
    se_score = sd_score / sqrt(n_cells),
    .groups = "drop"
  )
write.csv(cluster_summary, file.path(out_dir, "microglia_cluster_ferroaging_summary.csv"), row.names = FALSE)

# Overall cluster summary
cluster_overall <- mg@meta.data %>%
  pivot_longer(cols = all_of(score_cols), names_to = "score_method", values_to = "score") %>%
  group_by(seurat_clusters, score_method) %>%
  summarise(
    n_cells = n(),
    mean_score = mean(score, na.rm = TRUE),
    median_score = median(score, na.rm = TRUE),
    sd_score = sd(score, na.rm = TRUE),
    se_score = sd_score / sqrt(n_cells),
    .groups = "drop"
  )
write.csv(cluster_overall, file.path(out_dir, "microglia_cluster_overall_summary.csv"), row.names = FALSE)

# Identify high ferro-aging clusters (top 25% by FA-96 UCell mean)
fa96_ucell <- cluster_overall %>%
  filter(score_method == "FA_96_UCell") %>%
  arrange(desc(mean_score))
message("\nTop microglia clusters by FA-96 UCell score:")
print(fa96_ucell)

threshold <- quantile(fa96_ucell$mean_score, 0.75)
high_fa_clusters <- fa96_ucell %>% filter(mean_score >= threshold) %>% pull(seurat_clusters)
message("\nHigh ferro-aging clusters (top 25%, threshold = ", round(threshold, 4), "): ",
        paste(high_fa_clusters, collapse = ", "))
mg$high_fa_cluster <- ifelse(mg$seurat_clusters %in% high_fa_clusters, "High_FA", "Low_FA")

# ---- Differential expression: high FA vs low FA clusters ----
message("\nRunning differential expression (High_FA vs Low_FA)...")
Idents(mg) <- "high_fa_cluster"
mg_deg <- FindAllMarkers(mg, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
write.csv(mg_deg, file.path(out_dir, "microglia_high_fa_vs_low_fa_deg.csv"), row.names = FALSE)

message("Top upregulated genes in High_FA microglia:")
top_high_fa <- mg_deg %>%
  filter(cluster == "High_FA") %>%
  arrange(p_val_adj, desc(avg_log2FC)) %>%
  head(30)
print(top_high_fa %>% select(gene, avg_log2FC, p_val_adj))

# ---- Save updated microglia object ----
saveRDS(mg, file.path(out_dir, "microglia_subset_with_subclusters.rds"))
write.csv(mg@meta.data, file.path(out_dir, "microglia_cell_metadata.csv"), row.names = FALSE)

# ---- Visualizations ----
message("\nGenerating visualizations...")
mg$Condition <- factor(mg$Condition, levels = c("Ctrl", "1DPI", "3DPI", "7DPI"))

# 1. UMAP by cluster
p1 <- DimPlot(mg, reduction = "umap", label = TRUE, group.by = "seurat_clusters") +
  labs(title = "Microglia subclusters") +
  theme_bw(base_size = 12)
ggsave(file.path(out_dir, "microglia_umap_clusters.pdf"), p1, width = 6, height = 5)
ggsave(file.path(out_dir, "microglia_umap_clusters.png"), p1, width = 6, height = 5, dpi = 300)

# 2. UMAP by condition
p2 <- DimPlot(mg, reduction = "umap", group.by = "Condition") +
  labs(title = "Microglia by condition") +
  theme_bw(base_size = 12)
ggsave(file.path(out_dir, "microglia_umap_condition.pdf"), p2, width = 6, height = 5)
ggsave(file.path(out_dir, "microglia_umap_condition.png"), p2, width = 6, height = 5, dpi = 300)

# 3. UMAP by FA-96 UCell score
p3 <- FeaturePlot(mg, features = "FA_96_UCell", reduction = "umap") +
  scale_color_gradientn(colors = c("blue", "white", "red")) +
  labs(title = "FA-96 UCell ferro-aging score") +
  theme_bw(base_size = 12)
ggsave(file.path(out_dir, "microglia_umap_fa96_ucell.pdf"), p3, width = 6, height = 5)
ggsave(file.path(out_dir, "microglia_umap_fa96_ucell.png"), p3, width = 6, height = 5, dpi = 300)

# 4. Cluster composition by condition
comp <- mg@meta.data %>%
  group_by(Condition, seurat_clusters) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Condition) %>%
  mutate(prop = n / sum(n))
p4 <- ggplot(comp, aes(x = Condition, y = prop, fill = seurat_clusters)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Microglia cluster composition by condition", y = "Proportion", x = NULL) +
  theme_bw(base_size = 12)
ggsave(file.path(out_dir, "microglia_cluster_composition.pdf"), p4, width = 7, height = 5)
ggsave(file.path(out_dir, "microglia_cluster_composition.png"), p4, width = 7, height = 5, dpi = 300)

# 5. FA-96 UCell score by cluster
p5_df <- cluster_overall %>% filter(score_method == "FA_96_UCell")
p5 <- ggplot(p5_df, aes(x = reorder(seurat_clusters, -mean_score), y = mean_score, fill = seurat_clusters)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score), width = 0.2) +
  labs(title = "Mean FA-96 UCell score by microglia cluster",
       x = "Cluster", y = "Mean FA-96 UCell score") +
  theme_bw(base_size = 12) +
  theme(legend.position = "none")
ggsave(file.path(out_dir, "microglia_fa96_by_cluster.pdf"), p5, width = 6, height = 4)
ggsave(file.path(out_dir, "microglia_fa96_by_cluster.png"), p5, width = 6, height = 4, dpi = 300)

# 6. Violin of FA-96 by cluster and condition
p6 <- ggplot(mg@meta.data, aes(x = seurat_clusters, y = FA_96_UCell, fill = Condition)) +
  geom_violin(scale = "width", trim = TRUE, position = position_dodge(width = 0.8)) +
  labs(title = "FA-96 UCell score by microglia cluster and condition",
       x = "Cluster", y = "FA-96 UCell score") +
  theme_bw(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(out_dir, "microglia_fa96_by_cluster_condition_violin.pdf"), p6, width = 8, height = 5)
ggsave(file.path(out_dir, "microglia_fa96_by_cluster_condition_violin.png"), p6, width = 8, height = 5, dpi = 300)

# 7. Top DEG heatmap for high FA clusters
top_genes <- top_high_fa$gene[1:min(20, nrow(top_high_fa))]
if (length(top_genes) > 0) {
  p7 <- DoHeatmap(mg, features = top_genes, group.by = "high_fa_cluster") +
    labs(title = "Top DEGs in High_FA microglia")
  ggsave(file.path(out_dir, "microglia_high_fa_top_deg_heatmap.pdf"), p7, width = 8, height = 6)
  ggsave(file.path(out_dir, "microglia_high_fa_top_deg_heatmap.png"), p7, width = 8, height = 6, dpi = 300)
}

# ---- Save high FA gene list for CPI cross-check ----
high_fa_genes <- top_high_fa$gene
write.table(high_fa_genes, file.path(out_dir, "microglia_high_fa_marker_genes.txt"),
            row.names = FALSE, col.names = FALSE, quote = FALSE)

# ---- Save summary report ----
report <- data.frame(
  item = c("Total microglia cells", "Number of clusters", "High FA clusters",
           "High FA cluster threshold (top 25%)", "High FA cells", "Low FA cells",
           "Top marker gene", "Top marker log2FC", "Top marker p_adj"),
  value = c(ncol(mg), length(unique(mg$seurat_clusters)), paste(high_fa_clusters, collapse = ", "),
            round(threshold, 5),
            sum(mg$high_fa_cluster == "High_FA"),
            sum(mg$high_fa_cluster == "Low_FA"),
            ifelse(nrow(top_high_fa) > 0, top_high_fa$gene[1], NA),
            ifelse(nrow(top_high_fa) > 0, round(top_high_fa$avg_log2FC[1], 3), NA),
            ifelse(nrow(top_high_fa) > 0, format(top_high_fa$p_val_adj[1], digits = 3), NA))
)
write.csv(report, file.path(out_dir, "microglia_subcluster_summary.csv"), row.names = FALSE)

message("\n=== Done ===")
message("Outputs in: ", normalizePath(out_dir))
