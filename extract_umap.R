library(Seurat)
library(readr)

obj <- readRDS("d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/Seurat_sn_MCAO_with_ferroaging_score.rds")
umap <- Embeddings(obj, "umap")
meta <- read_csv("d:/铁衰老 绝不重蹈覆辙/L2/results/GSE233815_sn/cell_metadata_with_ferroaging_score.csv", show_col_types = FALSE)
meta$UMAP_1 <- umap[,1]
meta$UMAP_2 <- umap[,2]
write_csv(meta, "d:/铁衰老 绝不重蹈覆辙/figures/meta_with_umap.csv")
cat("Saved:", nrow(meta), "cells with UMAP\n")
