##############################################################################
# circos-skill 功能验证脚本
# 用项目真实 PPI + GO 富集数据测试圈图
##############################################################################

suppressPackageStartupMessages({
  library(circlize); library(ggplot2); library(dplyr); library(readr)
  library(viridis); library(ggsci); library(Cairo); library(cowplot); library(stringr)
})
stopifnot(requireNamespace("circlize", quietly=TRUE))
stopifnot(requireNamespace("Cairo", quietly=TRUE))

OUTDIR <- "d:/铁衰老 绝不重蹈覆辙/figures/skill_test"
dir.create(OUTDIR, showWarnings=FALSE, recursive=TRUE)

cat("========================================\n")
cat("  Circos Skill Test\n")
cat("========================================\n\n")

# ---- 1. PPI Chord Diagram ----
cat("--- [Type 1] PPI Chord ---\n")
ppi_edges_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_edges.csv"
ppi_topo_path  <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_ppi_topology.csv"
stopifnot(file.exists(ppi_edges_path), file.exists(ppi_topo_path))

ppi_edges <- read_csv(ppi_edges_path, show_col_types=FALSE)
ppi_topo  <- read_csv(ppi_topo_path, show_col_types=FALSE)
stopifnot(nrow(ppi_edges) > 0, nrow(ppi_topo) > 0)
cat(sprintf("  PPI edges: %d | topology: %d\n", nrow(ppi_edges), nrow(ppi_topo)))
cat(sprintf("  PPI edge cols: %s\n", paste(names(ppi_edges), collapse=", ")))

# 找 source/target/weight 列
src_col <- intersect(c("source","Source","from","node1","Gene_A","gene_a","geneA","A","nodeA"), names(ppi_edges))[1]
tgt_col <- intersect(c("target","Target","to","node2","Gene_B","gene_b","geneB","B","nodeB"), names(ppi_edges))[1]
w_col   <- intersect(c("combined_score","score","weight","confidence","Score","interaction_score"), names(ppi_edges))[1]
stopifnot(!is.na(src_col), !is.na(tgt_col))
if (is.na(w_col)) {
  ppi_edges$weight <- 1
  w_col <- "weight"
}
cat(sprintf("  Using: src=%s tgt=%s weight=%s\n", src_col, tgt_col, w_col))

# Top 25 hub genes by degree
deg_col <- intersect(c("Degree","degree","Degree_Cent"), names(ppi_topo))[1]
gene_col <- intersect(c("Gene","gene","Symbol"), names(ppi_topo))[1]
stopifnot(!is.na(deg_col), !is.na(gene_col))
hub_genes <- ppi_topo %>% arrange(desc(.data[[deg_col]])) %>% head(25) %>% pull(.data[[gene_col]])
ppi_sub <- ppi_edges %>%
  filter(.data[[src_col]] %in% hub_genes & .data[[tgt_col]] %in% hub_genes) %>%
  mutate(weight = as.numeric(.data[[w_col]]))
# 归一化 weight 到 0-1
w_max <- max(ppi_sub$weight, na.rm=TRUE)
if (w_max > 1) ppi_sub$weight <- ppi_sub$weight / w_max
cat(sprintf("  Hub genes: %d | Sub-edges: %d\n", length(hub_genes), nrow(ppi_sub)))

# Build matrix
mat <- matrix(0, nrow=length(hub_genes), ncol=length(hub_genes),
              dimnames=list(hub_genes, hub_genes))
for (i in seq_len(nrow(ppi_sub))) {
  s <- ppi_sub[[src_col]][i]; t <- ppi_sub[[tgt_col]][i]; w <- ppi_sub$weight[i]
  mat[s, t] <- w
  mat[t, s] <- w  # symmetric for undirected PPI
}

# Color by degree
deg_lookup <- setNames(ppi_topo[[deg_col]][ppi_topo[[gene_col]] %in% hub_genes],
                       ppi_topo[[gene_col]][ppi_topo[[gene_col]] %in% hub_genes])
col_vec <- viridis(length(hub_genes), option="D")[order(order(deg_lookup[hub_genes]))]
names(col_vec) <- hub_genes

ppi_png <- file.path(OUTDIR, "circos_ppi_chord_test.png")
ppi_pdf <- file.path(OUTDIR, "circos_ppi_chord_test.pdf")

png(ppi_png, width=9, height=9, units="in", res=300, bg="white")
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(mat, grid.col=col_vec, transparency=0.3,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
title("PPI Hub Genes Chord (Top 25 by Degree)")
circos.clear()
dev.off()

Cairo::CairoPDF(ppi_pdf, width=9, height=9)
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(mat, grid.col=col_vec, transparency=0.3,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
title("PPI Hub Genes Chord (Top 25 by Degree)")
circos.clear()
dev.off()
cat(sprintf("  -> %s (%.0f KB)\n", ppi_png, file.info(ppi_png)$size/1024))

# ---- 2. GO Enrichment Chord ----
cat("\n--- [Type 2] GO Enrichment Chord ---\n")
go_path <- "d:/铁衰老 绝不重蹈覆辙/L2/results/core_go_bp_enrichment.csv"
stopifnot(file.exists(go_path))
go_bp <- read_csv(go_path, show_col_types=FALSE)
stopifnot(nrow(go_bp) > 0)
cat(sprintf("  GO BP: %d rows | cols: %s\n", nrow(go_bp), paste(names(go_bp), collapse=", ")))

# 找列名 (扩展候选 — 兼容 clusterProfiler "Adjusted P-value" 等)
padj_col <- intersect(c("p.adjust","padj","FDR","fdr","qvalue","P.adjust",
                        "Adjusted P-value","Adjusted_Pvalue","adj.P.Val"), names(go_bp))[1]
desc_col <- intersect(c("Description","Term","term","ID"), names(go_bp))[1]
gene_col_go <- intersect(c("geneID","core_enrichment","Genes","genes","Gene"), names(go_bp))[1]
stopifnot(!is.na(padj_col), !is.na(desc_col), !is.na(gene_col_go))
cat(sprintf("  Using: padj=%s desc=%s genes=%s\n", padj_col, desc_col, gene_col_go))

go_top <- go_bp %>%
  filter(!is.na(.data[[padj_col]])) %>%
  arrange(.data[[padj_col]]) %>%
  head(8) %>%
  mutate(Description = make.unique(str_trunc(as.character(.data[[desc_col]]), 40)))

# 解析 gene list (通常是 "gene1/gene2/gene3" 或 "gene1,gene2")
parse_genes <- function(x) {
  x <- as.character(x)
  unlist(strsplit(gsub("^\\s+|\\s+$", "", x), "[/,;]"))
}
all_genes_raw <- unlist(lapply(go_top[[gene_col_go]], parse_genes))
all_genes_raw <- all_genes_raw[all_genes_raw != "" & !is.na(all_genes_raw)]
# 按出现频率取 top 30 基因(避免 chord 圈图 sectors 过多)
gene_freq <- sort(table(all_genes_raw), decreasing=TRUE)
all_genes <- names(gene_freq)[seq_len(min(30, length(gene_freq)))]
cat(sprintf("  Top 8 GO terms | unique genes: %d (filtered to top 30 by freq)\n",
            length(unique(all_genes_raw))))

# Build GO term × gene matrix (binary)
go_mat <- matrix(0, nrow=nrow(go_top), ncol=length(all_genes),
                 dimnames=list(go_top$Description, all_genes))
for (i in seq_len(nrow(go_top))) {
  genes_i <- parse_genes(go_top[[gene_col_go]][i])
  go_mat[i, intersect(genes_i, all_genes)] <- 1
}

go_colors <- viridis(nrow(go_top), option="C")
names(go_colors) <- go_top$Description
gene_colors <- rep("grey70", length(all_genes))
names(gene_colors) <- all_genes

go_png <- file.path(OUTDIR, "circos_go_chord_test.png")
go_pdf <- file.path(OUTDIR, "circos_go_chord_test.pdf")

png(go_png, width=10, height=10, units="in", res=300, bg="white")
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(go_mat, grid.col=c(go_colors, gene_colors), transparency=0.4,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
title("GO BP Enrichment — Gene-Pathway Chord (Top 15)")
circos.clear()
dev.off()

Cairo::CairoPDF(go_pdf, width=10, height=10)
circos.clear()
circos.par(gap.after=2, cell.padding=c(0.02,0,0.02,0))
chordDiagram(go_mat, grid.col=c(go_colors, gene_colors), transparency=0.4,
             annotationTrack=c("grid","name"),
             preAllocateTracks=list(track.height=0.08))
title("GO BP Enrichment — Gene-Pathway Chord (Top 15)")
circos.clear()
dev.off()
cat(sprintf("  -> %s (%.0f KB)\n", go_png, file.info(go_png)$size/1024))

# ---- 验证 ----
cat("\n--- Verification ---\n")
cat(sprintf("  PPI chord: %d x %d hub genes\n", length(hub_genes), length(hub_genes)))
cat(sprintf("  GO chord: %d GO terms x %d genes\n", nrow(go_top), length(all_genes)))
cat("  Real data: PPI edges=", nrow(ppi_edges), " GO BP=", nrow(go_bp), "\n", sep="")
cat("  Circos skill test PASSED.\n")
