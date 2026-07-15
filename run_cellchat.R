##############################################################################
# CELL COMMUNICATION ANALYSIS (CellChatDB-based)
# 铁衰老项目 — GSE233815 snRNA-seq
#
# 方法 (参照 Jin et al., Nature Communications 2021, PMID:33597522):
#   - 使用 CellChatDB.mouse 配体-受体数据库 (3379 interactions)
#   - 配体/受体复合物表达 = 算术平均 (CellChat v2 'mean' aggregation mode)
#     注: CellChat 原论文用几何平均，但先做 STRINGdb 网络传播将0值投影为非0值。
#     本简化版未实现网络传播，geomean 在稀疏 scRNA-seq 中过于严格
#     (任一亚基为0 -> 复合物=0)，故采用算术平均以保留真实交互。
#   - 通讯概率 P = L * R (简化版，未实现 Hill function / agonist / antagonist)
#
# 已知简化 (在 Figure_Legends 中标注):
#   1. 用算术平均替代 trimean (EM = 0.5*Q2 + 0.25*(Q1+Q3))
#   2. 用算术平均替代 geomean (因未实现 STRINGdb 网络传播)
#   3. 未实现 Hill function 与 cofactor 调控
#   4. 未执行 permutation test 评估显著性
#
# 输出:
#   - L2/results/cellchat_lr_pairs.csv
#   - L2/results/cellchat_signaling_pathways.csv
#   - figures/Fig6A_CellChat_circle.png/pdf
#   - figures/Fig6B_CellChat_heatmap.png/pdf
#   - figures/Fig6C_CellChat_bubble.png/pdf
#   - figures/Fig6D_CellChat_pathway_contribution.png/pdf
##############################################################################

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(patchwork)
  library(reshape2)
  library(circlize)
  library(grid)
})

set.seed(42)

OUTDIR     <- "d:/铁衰老 绝不重蹈覆辙/figures"
OUTDIR_PDF <- file.path(OUTDIR, "pdf")
RESULTS    <- "d:/铁衰老 绝不重蹈覆辙/L2/results"
dir.create(OUTDIR_PDF, showWarnings = FALSE, recursive = TRUE)

theme_pub <- theme_bw(base_size = 9) +
  theme(
    panel.grid = element_blank(),
    panel.border = element_rect(color = "black", linewidth = 0.6),
    axis.title = element_text(face = "bold", size = 10),
    axis.text = element_text(size = 8, color = "black"),
    plot.tag = element_text(face = "bold", size = 14),
    plot.tag.position = "topleft",
    legend.title = element_text(face = "bold", size = 8),
    legend.text = element_text(size = 7)
  )

cat("========================================\n")
cat("  Cell Communication Analysis\n")
cat("  (CellChatDB-based)\n")
cat("========================================\n\n")

# ============================================================================
# 1. 加载数据
# ============================================================================
cat("--- Loading data ---\n")

# L-R database
load(file.path(RESULTS, "CellChatDB.mouse.rda"))
lr_db <- CellChatDB.mouse$interaction
cat(sprintf("  L-R database: %d interactions\n", nrow(lr_db)))

# Seurat object
seu <- readRDS("d:/铁衰老 绝不重蹈覆辙/L4/results/scissor_GSE61616_GSE233815/seurat_with_scissor.rds")
DefaultAssay(seu) <- "RNA"
seu <- NormalizeData(seu, verbose = FALSE)

expr <- as.matrix(GetAssayData(seu, layer = "data"))
meta <- seu@meta.data
cell_types <- as.character(meta$cell_type_1)
names(cell_types) <- rownames(meta)
unique_types <- sort(unique(cell_types))
cat(sprintf("  Expression: %d genes x %d cells\n", nrow(expr), ncol(expr)))
cat(sprintf("  Cell types: %s\n", paste(unique_types, collapse=", ")))

# ============================================================================
# 2. 计算每种细胞类型的平均表达
# ============================================================================
cat("\n--- Computing mean expression per cell type ---\n")

mean_expr <- sapply(unique_types, function(ct) {
  cells <- names(cell_types)[cell_types == ct]
  if (length(cells) < 10) return(rep(NA, nrow(expr)))
  rowMeans(expr[, cells, drop = FALSE], na.rm = TRUE)
})
rownames(mean_expr) <- rownames(expr)
cat(sprintf("  Mean expression matrix: %d genes x %d types\n", nrow(mean_expr), ncol(mean_expr)))

# ============================================================================
# 3. 解析 L-R 对，计算通讯概率
# ============================================================================
cat("\n--- Computing communication probability ---\n")

# Parse receptor complexes: "TGFbR1_R2" -> c("Tgfbr1", "Tgfbr2")
parse_genes <- function(symbol_str) {
  parts <- strsplit(symbol_str, "_")[[1]]
  genes <- c()
  for (p in parts) {
    # Try exact match first, then case-insensitive
    if (p %in% rownames(mean_expr)) {
      genes <- c(genes, p)
    } else {
      idx <- which(tolower(rownames(mean_expr)) == tolower(p))
      if (length(idx) > 0) genes <- c(genes, rownames(mean_expr)[idx[1]])
    }
  }
  genes
}

# Mean expression for ligand/receptor complexes
# CellChat reference (Jin et al., Nat Commun 2021, Eq.2) uses geometric mean,
# but applies network propagation on STRINGdb first to avoid zero values.
# Without network propagation (as in this simplified implementation), geometric
# mean is too strict for sparse scRNA-seq data (any zero subunit -> complex=0).
# We use arithmetic mean (CellChat v2 'mean' aggregation mode) as a pragmatic
# alternative that preserves non-zero interactions.
mean_complex <- function(expr_mat) {
  # expr_mat: genes x cell_types matrix for one complex
  # Preserve names attribute for single-gene case (critical for indexing by cell type)
  if (nrow(expr_mat) == 1) return(expr_mat[1, ])
  colMeans(expr_mat, na.rm = TRUE)
}

# For each L-R pair, compute score for all cell type pairs
lr_results <- list()
n_total <- nrow(lr_db)
n_valid <- 0

for (i in seq_len(n_total)) {
  ligand_str <- as.character(lr_db$ligand[i])
  receptor_str <- as.character(lr_db$receptor[i])

  # Parse ligand (may be single gene or complex)
  ligand_genes <- parse_genes(ligand_str)
  if (length(ligand_genes) == 0) next

  # Parse receptor (may be complex)
  receptor_genes <- parse_genes(receptor_str)
  if (length(receptor_genes) == 0) next

  # Ligand expression: mean across subunits (CellChat v2 'mean' aggregation)
  ligand_expr <- mean_complex(mean_expr[ligand_genes, , drop = FALSE])
  # Receptor expression: mean across subunits
  receptor_expr <- mean_complex(mean_expr[receptor_genes, , drop = FALSE])

  # Communication probability P = L * R (simplified, without Hill function)
  for (sender in unique_types) {
    for (receiver in unique_types) {
      score <- ligand_expr[sender] * receptor_expr[receiver]
      if (is.na(score) || score <= 0) next

      lr_results[[length(lr_results) + 1]] <- data.frame(
        pathway_name = lr_db$pathway_name[i],
        interaction_name = lr_db$interaction_name_2[i],
        ligand = ligand_str,
        receptor = receptor_str,
        annotation = lr_db$annotation[i],
        source = sender,
        target = receiver,
        prob = score,
        stringsAsFactors = FALSE
      )
      n_valid <- n_valid + 1
    }
  }

  if (i %% 500 == 0) cat(sprintf("  Processed %d / %d L-R pairs (%d interactions)\n", i, n_total, n_valid))
}

cat(sprintf("  Total valid interactions: %d\n", n_valid))

# Combine results
lr_df <- do.call(rbind, lr_results)
cat(sprintf("  Results: %d rows\n", nrow(lr_df)))

# Filter: keep top interactions (prob > 0)
lr_df <- lr_df %>% filter(prob > 0.01)

# Save L-R pairs
write.csv(lr_df, file.path(RESULTS, "cellchat_lr_pairs.csv"), row.names = FALSE)
cat(sprintf("  Saved cellchat_lr_pairs.csv (%d rows)\n", nrow(lr_df)))

# ============================================================================
# 4. 聚合到信号通路级别
# ============================================================================
cat("\n--- Aggregating to pathway level ---\n")

pathway_df <- lr_df %>%
  group_by(pathway_name, source, target) %>%
  summarise(prob = sum(prob), n_interactions = n(), .groups = "drop") %>%
  arrange(desc(prob))

write.csv(pathway_df, file.path(RESULTS, "cellchat_signaling_pathways.csv"), row.names = FALSE)
cat(sprintf("  Saved cellchat_signaling_pathways.csv (%d rows)\n", nrow(pathway_df)))

# ============================================================================
# 5. 可视化
# ============================================================================

# ------------------------------------------------------------------
# Panel A: Circle plot — interaction count
# ------------------------------------------------------------------
cat("\n[Panel A] Circle plot...\n")

count_matrix <- lr_df %>%
  count(source, target) %>%
  acast(source ~ target, value.var = "n", fill = 0)

# Cell type colors (consistent across panels)
cell_types_ordered <- sort(unique(c(rownames(count_matrix), colnames(count_matrix))))
cell_colors <- setNames(rainbow(length(cell_types_ordered)), cell_types_ordered)

# Draw circle plot function (reused for png/pdf/composite)
draw_circle <- function(mat, cols) {
  circos.clear()
  chordDiagram(mat, annotationTrack = "grid",
               preAllocateTracks = list(track.height = 0.05),
               grid.col = cols, transparency = 0.15)
  circos.trackPlotRegion(track.index = 1, panel.fun = function(x, y) {
    sector.name <- get.cell.meta.data("sector.index")
    xlim <- get.cell.meta.data("xlim")
    ylim <- get.cell.meta.data("ylim")
    circos.text(mean(xlim), ylim[1], sector.name, facing = "clockwise",
                niceFacing = TRUE, adj = c(0, 0.5), cex = 0.8)
  }, bg.border = NA)
  title("Cell-Cell Communication Network (Interaction Count)")
  circos.clear()
}

png(file.path(OUTDIR, "Fig6A_CellChat_circle.png"), width = 7, height = 7, units = "in", res = 300, bg = "white")
draw_circle(count_matrix, cell_colors)
dev.off()

pdf(file.path(OUTDIR_PDF, "Fig6A_CellChat_circle.pdf"), width = 7, height = 7)
draw_circle(count_matrix, cell_colors)
dev.off()
cat("  -> Fig6A saved\n")

# Capture circle plot as grid object for composite figure
p_circle <- wrap_elements(full = grid.grabExpr(draw_circle(count_matrix, cell_colors)))

# ------------------------------------------------------------------
# Panel B: Heatmap — communication weight (ggplot geom_tile)
# ------------------------------------------------------------------
cat("[Panel B] Heatmap...\n")

weight_matrix <- pathway_df %>%
  group_by(source, target) %>%
  summarise(total_prob = sum(prob), .groups = "drop") %>%
  acast(source ~ target, value.var = "total_prob", fill = 0)

# Hierarchical clustering for row/column ordering
hc_rows <- hclust(dist(weight_matrix))
hc_cols <- hclust(dist(t(weight_matrix)))
row_order <- rownames(weight_matrix)[hc_rows$order]
col_order <- colnames(weight_matrix)[hc_cols$order]

weight_long <- as.data.frame(as.table(weight_matrix))
colnames(weight_long) <- c("source", "target", "prob")
weight_long$prob <- as.numeric(weight_long$prob)
weight_long$source <- factor(weight_long$source, levels = row_order)
weight_long$target <- factor(weight_long$target, levels = col_order)

p_heatmap <- ggplot(weight_long, aes(x = target, y = source, fill = prob)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = ifelse(prob > 0.001, sprintf("%.2f", prob), "")),
            size = 2.5, color = "black", fontface = "bold") +
  scale_fill_gradientn(colors = c("white", "#FDB863", "#3C5488"),
                       name = "Comm. Prob.", trans = "sqrt") +
  labs(x = "Receiver", y = "Sender",
       title = "Cell-Cell Communication Strength") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8),
        plot.title = element_text(size = 10, face = "bold"),
        legend.position.inside = c(0.85, 0.15))

ggsave(file.path(OUTDIR, "Fig6B_CellChat_heatmap.png"), p_heatmap, width = 7, height = 6, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6B_CellChat_heatmap.pdf"), p_heatmap, width = 7, height = 6, bg = "white")
cat("  -> Fig6B saved\n")

# ------------------------------------------------------------------
# Panel C: Bubble plot — top L-R pairs
# ------------------------------------------------------------------
cat("[Panel C] Bubble plot...\n")

top_lr <- lr_df %>%
  group_by(interaction_name, pathway_name, source, target) %>%
  summarise(prob = max(prob), .groups = "drop") %>%
  arrange(desc(prob)) %>%
  head(50) %>%
  mutate(interaction_name = factor(interaction_name, levels = rev(unique(interaction_name))),
         pair = paste(source, "->", target))

p_bubble <- ggplot(top_lr, aes(x = pair, y = interaction_name, size = prob, color = prob)) +
  geom_point(alpha = 0.8) +
  scale_color_viridis_c(option = "D", name = "Comm. Prob.") +
  scale_size_continuous(range = c(1, 8), name = "Comm. Prob.") +
  labs(x = "Sender -> Receiver", y = "Ligand-Receptor Pair",
       title = "Top 50 Ligand-Receptor Interactions") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        axis.text.y = element_text(size = 6),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig6C_CellChat_bubble.png"), p_bubble, width = 10, height = 9, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6C_CellChat_bubble.pdf"), p_bubble, width = 10, height = 9, bg = "white")
cat("  -> Fig6C saved\n")

# ------------------------------------------------------------------
# Panel D: Pathway contribution bar chart
# ------------------------------------------------------------------
cat("[Panel D] Pathway contribution...\n")

pathway_summary <- lr_df %>%
  group_by(pathway_name, annotation) %>%
  summarise(total_prob = sum(prob), n_lr = n_distinct(interaction_name), .groups = "drop") %>%
  arrange(desc(total_prob)) %>%
  head(25) %>%
  mutate(pathway_name = make.unique(pathway_name),
         pathway_name = factor(pathway_name, levels = rev(pathway_name)))

p_pathway <- ggplot(pathway_summary, aes(x = total_prob, y = pathway_name, fill = annotation)) +
  geom_col(width = 0.7, alpha = 0.85) +
  geom_text(aes(label = n_lr), hjust = -0.2, size = 2.5, fontface = "bold") +
  scale_fill_manual(values = c("Secreted Signaling" = "#E64B35", "ECM-Receptor" = "#4DBBD5",
                                "Cell-Cell Contact" = "#00A087", "Non-protein Signaling" = "#3C5488"),
                    name = "Annotation") +
  labs(x = "Total Communication Probability", y = NULL,
       title = "Top 25 Signaling Pathways") +
  theme_pub +
  theme(axis.text.y = element_text(face = "bold", size = 7.5),
        legend.position.inside = c(0.7, 0.3),
        plot.title = element_text(size = 10, face = "bold"))

ggsave(file.path(OUTDIR, "Fig6D_CellChat_pathway_contribution.png"), p_pathway, width = 9, height = 7, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6D_CellChat_pathway_contribution.pdf"), p_pathway, width = 9, height = 7, bg = "white")
cat("  -> Fig6D saved\n")

# ============================================================================
# 6. 组图: 4 panels (A: circle / B: heatmap / C: bubble / D: pathway)
# ============================================================================
cat("\n--- Assembling composite (4 panels) ---\n")

fig6 <- (p_circle + labs(tag = "A") | p_heatmap + labs(tag = "B")) /
        ((p_bubble + labs(tag = "C")) | (p_pathway + labs(tag = "D"))) +
        plot_layout(heights = c(1, 1.1)) &
        theme(plot.tag = element_text(face = "bold", size = 16))

ggsave(file.path(OUTDIR, "Fig6_Composite_cell_communication.png"), fig6,
       width = 16, height = 14, dpi = 300, bg = "white")
ggsave(file.path(OUTDIR_PDF, "Fig6_Composite_cell_communication.pdf"), fig6,
       width = 16, height = 14, bg = "white")
cat("  -> Fig6 composite saved (4 panels)\n")

cat("\n========================================\n")
cat("  Cell communication analysis complete!\n")
cat("========================================\n")
