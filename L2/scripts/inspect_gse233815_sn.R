# Inspect Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds structure.
# Run with Rscript.
library(Seurat)

rds_path <- "data/external/GSE233815/mendeley/Seurat_sn_MCAO_Zucha2023_QCFiltered.Rds"
message("Loading: ", rds_path)
seu <- readRDS(rds_path)

message("\n=== Seurat object summary ===")
print(seu)

message("\n=== Meta.data columns ===")
print(colnames(seu@meta.data))

message("\n=== Condition / Sample / cell_type tables ===")
for (col in c("Condition", "Sample", "cell_type_1", "cell_type_2", "orig.ident")) {
  if (col %in% colnames(seu@meta.data)) {
    message("\n", col, ":")
    print(table(seu@meta.data[[col]]))
  }
}

message("\n=== Assays ===")
print(names(seu@assays))

message("\n=== Default assay ===")
print(DefaultAssay(seu))

message("\n=== Number of features ===")
print(nrow(seu))

message("\n=== First 10 gene names ===")
print(head(rownames(seu), 10))

message("\n=== Save inspection result ===")
saveRDS(seu@meta.data, "data/external/GSE233815/mendeley/sn_meta_data_summary.rds")
message("Meta data saved to data/external/GSE233815/mendeley/sn_meta_data_summary.rds")
