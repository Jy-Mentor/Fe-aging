# ============================================================================
# 铁衰老基因集与功能富集工具函数
# 数据来源: 项目根目录 铁衰老基因.txt (96 基因, 真实文件)
# 人鼠同源基因映射依据: 项目既有 ssgsea_ferroaging_pipeline.R
# ============================================================================

HUMAN_TO_MOUSE_MAP <- list(
  'ABCC1'='Abcc1', 'ACVR1B'='Acvr1b', 'ACSL4'='Acsl4', 'ALOX15'='Alox15',
  'ATF3'='Atf3', 'ATG3'='Atg3', 'BAP1'='Bap1', 'BCL6'='Bcl6', 'BRD7'='Brd7',
  'CAVIN1'='Cavin1', 'CD74'='Cd74', 'CD82'='Cd82', 'CDO1'='Cdo1',
  'COX7A1'='Cox7a1', 'CTSB'='Ctsb', 'CXCL10'='Cxcl10', 'DPEP1'='Dpep1',
  'DPP4'='Dpp4', 'DUOX1'='Duox1', 'DYRK1A'='Dyrk1a', 'E2F1'='E2f1',
  'E2F3'='E2f3', 'EBF3'='Ebf3', 'EDN1'='Edn1', 'EGR1'='Egr1', 'EMP1'='Emp1',
  'EPHA2'='Epha2', 'EPHA4'='Epha4', 'ERN1'='Ern1', 'FBXO31'='Fbxo31',
  'FOSL1'='Fosl1', 'GMFB'='Gmfb', 'HBP1'='Hbp1', 'HERPUD1'='Herpud1',
  'HIF1A'='Hif1a', 'HMGB1'='Hmgb1', 'HMOX1'='Hmox1', 'ICA1'='Ica1',
  'IFNG'='Ifng', 'IGFBP7'='Igfbp7', 'IL1B'='Il1b', 'IL6'='Il6',
  'IRF1'='Irf1', 'IRF7'='Irf7', 'IRF9'='Irf9', 'KDM6B'='Kdm6b',
  'KEAP1'='Keap1', 'KLF6'='Klf6', 'LACTB'='Lactb', 'LCN2'='Lcn2',
  'LGMN'='Lgmn', 'LIFR'='Lifr', 'LOX'='Lox', 'LPCAT3'='Lpcat3',
  'MAP3K14'='Map3k14', 'MAPK1'='Mapk1', 'MAPK14'='Mapk14', 'MCU'='Mcu',
  'MEN1'='Men1', 'MPO'='Mpo', 'NLRP3'='Nlrp3', 'NOX4'='Nox4',
  'NR1D1'='Nr1d1', 'NR2F2'='Nr2f2', 'NUAK2'='Nuak2', 'PADI4'='Padi4',
  'PDE4B'='Pde4b', 'PPP2R2B'='Ppp2r2b', 'PRKD1'='Prkd1', 'PTBP1'='Ptbp1',
  'PTGS2'='Ptgs2', 'RBM3'='Rbm3', 'RUNX3'='Runx3', 'S100A8'='S100a8',
  'SAT1'='Sat1', 'SETD7'='Setd7', 'SLAMF8'='Slamf8', 'SLC1A5'='Slc1a5',
  'SMARCB1'='Smarcb1', 'SMURF2'='Smurf2', 'SNCA'='Snca', 'SOCS1'='Socs1',
  'SOCS2'='Socs2', 'SOD1'='Sod1', 'SP1'='Sp1', 'SPATA2'='Spata2',
  'TBX2'='Tbx2', 'TFRC'='Tfrc', 'TLR4'='Tlr4', 'TNFAIP1'='Tnfaip1',
  'TNFAIP3'='Tnfaip3', 'TXNIP'='Txnip', 'WNT5A'='Wnt5a', 'WWTR1'='Wwtr1',
  'YAP1'='Yap1', 'ZEB1'='Zeb1'
)

load_ferroaging_genes <- function(cfg) {
  fa_path <- file.path(cfg$project$root, cfg$data$ferroaging_genes)
  if (!file.exists(fa_path)) {
    stop("Ferroaging gene file not found: ", fa_path)
  }
  genes <- readLines(fa_path, warn = FALSE)
  genes <- unique(genes[nchar(genes) > 0])
  genes <- genes[genes != ""]
  return(genes)
}

map_human_to_mouse <- function(human_genes, verbose = TRUE) {
  mapped <- sapply(human_genes, function(g) {
    if (g %in% names(HUMAN_TO_MOUSE_MAP)) HUMAN_TO_MOUSE_MAP[[g]] else g
  })
  names(mapped) <- NULL
  if (verbose) {
    n_mapped <- sum(human_genes %in% names(HUMAN_TO_MOUSE_MAP))
    message(sprintf("Mapped %d/%d human genes to mouse orthologs",
                    n_mapped, length(human_genes)))
  }
  return(mapped)
}

intersect_with_seurat <- function(genes, seu, assay = "RNA") {
  avail_genes <- rownames(Seurat::GetAssayData(seu, assay = assay))
  common <- intersect(genes, avail_genes)
  missing <- setdiff(genes, avail_genes)
  if (length(missing) > 0) {
    message(sprintf("  Genes absent in Seurat (%d): %s",
                    length(missing), paste(head(missing, 10), collapse = ", ")))
  }
  message(sprintf("  Intersection: %d / %d", length(common), length(genes)))
  return(list(common = common, missing = missing))
}
