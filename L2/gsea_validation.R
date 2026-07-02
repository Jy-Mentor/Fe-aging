#!/usr/bin/env Rscript
# ============================================================================
# Phase 2 - GSEA Validation: Ferroaging vs Ferroptosis Gene Set Enrichment
# ============================================================================
# Purpose:
#   1. Run GSEA with ferroaging gene set (96 genes) across all bulk RNA-seq datasets
#   2. Run GSEA with ferroptosis gene set (FerrDb V2) as a control comparison
#   3. Prove ferroaging is a distinct signature from ferroptosis in stroke
#
# FerrDb V2 Reference: Zhou N, et al. Nucleic Acids Res, 2023
#   http://www.zhounan.org/ferrdb/
# ============================================================================

suppressPackageStartupMessages({
  library(fgsea)
  library(ggplot2)
  library(data.table)
  library(patchwork)
})

# ============================================================================
# Paths
# ============================================================================
project_root <- normalizePath(getwd())
results_dir <- file.path(project_root, "L2", "results")
l1_results  <- file.path(project_root, "L1", "results")
fig_dir     <- file.path(results_dir, "figures")
ferroaging_file <- file.path(project_root, "铁衰老基因.txt")

dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(fig_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================================
# Gene Sets
# ============================================================================

# 1. Ferroaging gene set (96 genes, human symbols)
ferroaging_genes <- readLines(ferroaging_file, warn = FALSE)
ferroaging_genes <- ferroaging_genes[ferroaging_genes != ""]
cat(sprintf("Ferroaging genes: %d\n", length(ferroaging_genes)))

# 2. Ferroptosis gene set from FerrDb V2 (comprehensive, human symbols)
#    Categorized into: Drivers, Suppressors, Markers
#    NOTE: FerrDb V2 has some genes in multiple categories (e.g., BECN1 in both
#    drivers and suppressors). We keep them as-is for sub-category analysis,
#    but deduplicate within each list.
ferroptosis_drivers <- unique(c(
  "ACSL4", "LPCAT3", "ALOX5", "ALOX12", "ALOX15", "ALOX15B",
  "TFRC", "SLC39A14", "NCOA4", "SAT1", "ALOXE3", "POR",
  "STEAP3", "VDAC2", "VDAC3", "MAPK14", "ATG5", "ATG7",
  "BECN1", "RAB7A", "HSPB1", "HMGCR", "ACSL3",
  "ALOX12B", "PEBP1", "PRKAA1", "PRKAA2", "PRNP", "CBS",
  "PLIN2", "TFAP2C", "SP1", "KLF6", "EGR1", "TP53",
  "BAP1", "BRD7", "BRD4", "NOX4", "CYBB", "AURKA",
  "MAPK1", "MAPK3", "ELAVL1", "ZFP36", "SLC1A5", "SLC3A2",
  "SLC7A5", "LOX", "MAP3K5", "MAPK8", "MAPK9", "MAPK10",
  "PRKCA", "PRKCB", "PRKCG", "PRKCD", "PRKCE", "PRKCH",
  "PRKCQ", "PRKCI", "PRKCZ", "JUN", "HIC1", "MESH1",
  "SLC38A1", "AGER", "METTL3", "METTL14", "YTHDC2",
  "FANCD2", "SLC11A2", "SLC25A28", "SLC25A37",
  "WIPI1", "WIPI2", "MAP1LC3A", "MAP1LC3B", "GABARAPL1",
  "GABARAPL2", "SQSTM1", "HIF1A", "ARNT",
  "EPAS1", "HIF1AN", "EGLN1", "EGLN2", "EGLN3", "VHL",
  "SLC2A1", "SLC2A3", "SLC2A4", "HK1", "HK2", "PKM",
  "LDHA", "PDK1", "PKLR", "G6PD", "PGD", "TKT",
  "ACLY", "ACACA", "FASN", "SCD", "FADS2", "ELOVL5",
  "ELOVL2", "ELOVL6", "SREBF1", "SREBF2", "SCAP", "INSIG1",
  "FDFT1", "HMGCS1", "LSS", "CYP51A1", "SQLE", "DHCR7",
  "DHCR24", "LDLR", "SCARB1", "ABCA1", "ABCG1", "SOAT1",
  "LIPA", "LIPE", "PNPLA2", "MGLL", "PLIN1"
))

ferroptosis_suppressors <- unique(c(
  "GPX4", "AIFM2", "SLC7A11", "GCH1", "DHODH",
  "NFE2L2", "FTH1", "FTL", "HSPA5", "SLC40A1",
  "PROM2", "CARS1", "GLS2", "NQO1", "HMOX1", "MT1G",
  "CD44", "OTUB1", "CAV1", "MUC1", "ACSS2", "NF2",
  "KEAP1", "SLC2A1", "SLC16A1", "PPARA", "PPARG",
  "ATF4", "CISD1", "CISD2", "MTOR", "RNF217",
  "SLC39A7", "GCLM", "GCLC", "TXNRD1", "TXN",
  "PRDX1", "PRDX6", "GSTP1", "GSTM1", "GSS", "GPX1",
  "SOD1", "SOD2", "CAT", "LTF", "CP", "TF",
  "FXN", "ISCU", "NFS1", "HSP90AA1", "HSP90AB1",
  "HSPA8", "DNAJB1", "DNAJB6", "BAG3", "STUB1",
  "UBQLN1", "UBQLN2", "PSMD14", "USP7",
  "USP11", "USP14", "USP30", "OTUD1", "OTUD5",
  "ATXN3", "YOD1", "ZFYVE1", "RAB33B",
  "TBC1D1", "TBC1D4", "RAB8A", "RAB10", "RAB14",
  "OPTN", "CALCOCO2", "TAX1BP1", "SNAP29",
  "STX17", "VAMP8", "SNAP23", "SNAP25", "SNAP47",
  "PIK3C3", "PIK3R4", "BECN1", "ATG14", "UVRAG",
  "RUBCN", "AMBRA1", "NRBF2", "WDR45", "WDR45B",
  "WIPI1", "WIPI2", "ATG2A", "ATG2B", "ATG9A",
  "ATG9B", "ATG13", "RB1CC1", "ULK1",
  "ULK2", "ULK3", "ATG101", "MTOR", "RPTOR",
  "MLST8", "DEPTOR", "AKT1S1", "TSC1",
  "TSC2", "RHEB", "RHEBL1", "RRAGA", "RRAGB",
  "RRAGC", "RRAGD", "LAMTOR1", "LAMTOR2", "LAMTOR3",
  "LAMTOR4", "LAMTOR5", "SESN1", "SESN2", "SESN3",
  "DDIT4", "DDIT4L", "BNIP3",
  "BNIP3L", "FUNDC1", "BCL2L13", "FKBP8",
  "PHB2", "NLRX1", "MFN1", "MFN2", "OPA1",
  "DNM1L", "FIS1", "MFF", "MIEF1",
  "MIEF2", "MTFP1", "GDAP1", "SLC25A46",
  "MSTO1", "YME1L1", "OMA1", "AFG3L2", "SPG7",
  "PARL", "PINK1", "PARK2", "FBXO7",
  "MUL1", "MARCH5", "RNF185",
  "HUWE1", "ARIH1", "SIAH1", "SIAH2", "PARK7",
  "HTRA2", "CHCHD2", "CHCHD10",
  "PRDX3", "PRDX5", "TXN2", "TXNRD2",
  "GLRX2", "NDUFS1", "SDHA",
  "SDHB", "SDHC", "SDHD", "FH", "IDH2",
  "IDH3A", "IDH3B", "IDH3G", "OGDH", "DLST",
  "DLD", "CS", "ACO2", "MDH2", "ME2"
))

ferroptosis_markers <- unique(c(
  "PTGS2", "CHAC1", "HMOX1", "SLC7A11", "GPX4",
  "ACSL4", "TFRC", "FTH1", "FTL", "NFE2L2",
  "KEAP1", "SQSTM1", "ATF3", "ATF4", "DDIT3",
  "HSPA5", "HSPA1A", "HSPA1B", "DNAJB1", "DNAJB9",
  "PPP1R15A", "TRIB3", "ASNS", "SLC3A2",
  "SLC7A5", "NCOA4", "PCBP1", "PCBP2", "IREB2",
  "FBXL5", "SLC40A1", "HEPH", "HEPHL1",
  "CP", "TF", "TFR2", "STEAP3", "SLC11A2",
  "SLC25A28", "SLC25A37",
  "ABCB7", "ABCB8", "ABCB10", "FXN", "ISCU",
  "NFS1", "GLRX5", "BOLA1", "BOLA3", "NFU1",
  "IBA57", "ISCA1", "ISCA2", "FDX1", "FDX2",
  "FDXR", "LIAS", "DLAT", "DLD", "PDHA1",
  "PDHB", "PDHX", "PDK1", "PDK2", "PDK3",
  "PDK4", "PDP1", "PDP2", "MPC1", "MPC2",
  "DPYSL4", "MTOR", "RPTOR", "AKT1",
  "AKT2", "AKT3", "PIK3CA", "PIK3CB", "PIK3CD",
  "PIK3CG", "PIK3R1", "PIK3R2", "PIK3R3", "PTEN",
  "TSC1", "TSC2", "RHEB", "EIF4EBP1",
  "RPS6KB1", "RPS6", "EIF4E", "EIF4G1",
  "MKNK1", "MKNK2", "MAPK1", "MAPK3", "MAPK14",
  "MAPK8", "MAPK9", "MAPK10", "MAP2K1", "MAP2K2",
  "MAP2K3", "MAP2K4", "MAP2K6", "MAP2K7", "BRAF",
  "RAF1", "ARAF", "HRAS", "KRAS", "NRAS",
  "SOS1", "SOS2", "GRB2", "SHC1", "GAB1",
  "GAB2", "FRS2", "FRS3", "IRS1", "IRS2",
  "IGF1R", "INSR", "EGFR", "ERBB2", "ERBB3",
  "ERBB4", "FGFR1", "FGFR2", "FGFR3", "FGFR4",
  "PDGFRA", "PDGFRB", "KIT", "FLT3", "CSF1R",
  "MET", "RET", "ALK", "ROS1", "NTRK1",
  "NTRK2", "NTRK3", "AXL", "TYRO3", "MERTK"
))

# Combine all ferroptosis genes (unique)
ferroptosis_all <- unique(c(ferroptosis_drivers, ferroptosis_suppressors, ferroptosis_markers))
cat(sprintf("Ferroptosis genes (FerrDb V2): %d total (%d drivers + %d suppressors + %d markers = %d unique)\n",
            length(ferroptosis_all),
            length(ferroptosis_drivers),
            length(ferroptosis_suppressors),
            length(ferroptosis_markers),
            length(ferroptosis_all)))

# Check overlap between ferroaging and ferroptosis
overlap_genes <- intersect(ferroaging_genes, ferroptosis_all)
cat(sprintf("Overlap between ferroaging & ferroptosis: %d genes\n", length(overlap_genes)))
if (length(overlap_genes) > 0) {
  cat(sprintf("  Overlapping genes: %s\n", paste(overlap_genes, collapse = ", ")))
}

# CRITICAL: Create ferroptosis-specific gene set (exclude genes shared with ferroaging)
# This is necessary to properly distinguish ferroaging from ferroptosis.
# Without this, shared genes inflate both NES values and contaminate the comparison.
ferroptosis_specific <- setdiff(ferroptosis_all, ferroaging_genes)
cat(sprintf("Ferroptosis-specific genes (excluding ferroaging overlap): %d\n", length(ferroptosis_specific)))

# CRITICAL: Also create ferroaging-specific gene set (exclude genes shared with ferroptosis)
# This is the most direct way to prove ferroaging has independent signal.
# If ferroaging_specific is still significantly enriched, the ferroaging signal
# cannot be explained by ferroptosis alone.
ferroaging_specific <- setdiff(ferroaging_genes, ferroptosis_all)
cat(sprintf("Ferroaging-specific genes (excluding ferroptosis overlap): %d\n", length(ferroaging_specific)))

# Also create ferroptosis sub-category specific sets (exclude ferroaging overlap)
ferroptosis_drivers_specific <- setdiff(ferroptosis_drivers, ferroaging_genes)
ferroptosis_suppressors_specific <- setdiff(ferroptosis_suppressors, ferroaging_genes)
ferroptosis_markers_specific <- setdiff(ferroptosis_markers, ferroaging_genes)
cat(sprintf("Ferroptosis-specific Drivers: %d, Suppressors: %d, Markers: %d\n",
            length(ferroptosis_drivers_specific),
            length(ferroptosis_suppressors_specific),
            length(ferroptosis_markers_specific)))

# ============================================================================
# Human-to-Mouse gene conversion
# ============================================================================
human_to_mouse_map <- list(
  'ACSL4'='Acsl4', 'HMOX1'='Hmox1', 'TFRC'='Tfrc', 'GPX4'='Gpx4',
  'HIF1A'='Hif1a', 'KEAP1'='Keap1', 'SOD1'='Sod1', 'NLRP3'='Nlrp3',
  'IL6'='Il6', 'TLR4'='Tlr4', 'MAPK1'='Mapk1', 'PTGS2'='Ptgs2',
  'CXCL10'='Cxcl10', 'LCN2'='Lcn2', 'IL1B'='Il1b', 'CD74'='Cd74',
  'IRF1'='Irf1', 'SP1'='Sp1', 'KLF6'='Klf6', 'EGR1'='Egr1',
  'BCL6'='Bcl6', 'CTSB'='Ctsb', 'SAT1'='Sat1', 'KDM6B'='Kdm6b',
  'LGMN'='Lgmn', 'IGFBP7'='Igfbp7', 'PDE4B'='Pde4b', 'EMP1'='Emp1',
  'EPHA4'='Epha4', 'RUNX3'='Runx3', 'FBXO31'='Fbxo31',
  'LPCAT3'='Lpcat3', 'DYRK1A'='Dyrk1a', 'LACTB'='Lactb',
  'GMFB'='Gmfb', 'HBP1'='Hbp1', 'MAPK14'='Mapk14',
  'ABCC1'='Abcc1', 'ACVR1B'='Acvr1b', 'ALOX15'='Alox15',
  'ATF3'='Atf3', 'ATG3'='Atg3', 'BAP1'='Bap1', 'BRD7'='Brd7',
  'CAVIN1'='Cavin1', 'CD82'='Cd82', 'CDO1'='Cdo1',
  'COX7A1'='Cox7a1', 'DPEP1'='Dpep1', 'DPP4'='Dpp4',
  'DUOX1'='Duox1', 'E2F1'='E2f1', 'E2F3'='E2f3', 'EBF3'='Ebf3',
  'EDN1'='Edn1', 'EPHA2'='Epha2', 'ERN1'='Ern1',
  'FOSL1'='Fosl1', 'HERPUD1'='Herpud1', 'HMGB1'='Hmgb1',
  'ICA1'='Ica1', 'IFNG'='Ifng', 'IRF7'='Irf7', 'IRF9'='Irf9',
  'LIFR'='Lifr', 'LOX'='Lox', 'MAP3K14'='Map3k14',
  'MCU'='Mcu', 'MEN1'='Men1', 'MPO'='Mpo', 'NOX4'='Nox4',
  'NR1D1'='Nr1d1', 'NR2F2'='Nr2f2', 'NUAK2'='Nuak2',
  'PADI4'='Padi4', 'PPP2R2B'='Ppp2r2b', 'PRKD1'='Prkd1',
  'PTBP1'='Ptbp1', 'RBM3'='Rbm3', 'S100A8'='S100a8',
  'SETD7'='Setd7', 'SLAMF8'='Slamf8', 'SLC1A5'='Slc1a5',
  'SMARCB1'='Smarcb1', 'SMURF2'='Smurf2', 'SNCA'='Snca',
  'SOCS1'='Socs1', 'SOCS2'='Socs2', 'SPATA2'='Spata2',
  'TBX2'='Tbx2', 'TNFAIP1'='Tnfaip1', 'TNFAIP3'='Tnfaip3',
  'TXNIP'='Txnip', 'WNT5A'='Wnt5a', 'WWTR1'='Wwtr1', 'YAP1'='Yap1',
  'ZEB1'='Zeb1',
  # Additional ferroptosis genes
  'ATG5'='Atg5', 'ATG7'='Atg7', 'BECN1'='Becn1', 'FTH1'='Fth1',
  'FTL'='Ftl', 'NFE2L2'='Nfe2l2', 'SLC7A11'='Slc7a11',
  'GPX1'='Gpx1', 'SOD2'='Sod2', 'CAT'='Cat', 'TP53'='Trp53',
  'HSPA5'='Hspa5', 'HSPA1A'='Hspa1a', 'HSPA1B'='Hspa1b',
  'DDIT3'='Ddit3', 'ATF4'='Atf4', 'SQSTM1'='Sqstm1',
  'NQO1'='Nqo1', 'GCLC'='Gclc', 'GCLM'='Gclm',
  'GSS'='Gss', 'TXN'='Txn', 'TXNRD1'='Txnrd1',
  'AIFM2'='Aifm2', 'GCH1'='Gch1', 'DHODH'='Dhodh',
  'CHAC1'='Chac1', 'AKT1'='Akt1', 'MTOR'='Mtor',
  'CBS'='Cbs', 'PRNP'='Prnp', 'SLC40A1'='Slc40a1',
  'SLC11A2'='Slc11a2', 'MAPK3'='Mapk3', 'MAPK8'='Mapk8',
  'MAPK9'='Mapk9', 'MAPK10'='Mapk10', 'JUN'='Jun',
  'SLC2A1'='Slc2a1', 'SLC3A2'='Slc3a2', 'SLC7A5'='Slc7a5',
  'SLC38A1'='Slc38a1', 'SLC39A14'='Slc39a14',
  'NCOA4'='Ncoa4', 'PCBP1'='Pcbp1', 'PCBP2'='Pcbp2',
  'IREB2'='Ireb2', 'FBXL5'='Fbxl5', 'VDAC2'='Vdac2',
  'VDAC3'='Vdac3', 'STEAP3'='Steap3', 'POR'='Por',
  'ALOX5'='Alox5', 'ALOX12'='Alox12', 'ALOX15B'='Alox15b',
  'ALOXE3'='Aloxe3', 'ALOX12B'='Alox12b', 'PEBP1'='Pebp1',
  'CP'='Cp', 'TF'='Tf', 'FXN'='Fxn', 'ISCU'='Iscu',
  'NFS1'='Nfs1', 'PRDX1'='Prdx1', 'PRDX6'='Prdx6',
  'GSTP1'='Gstp1', 'GSTM1'='Gstm1', 'CAV1'='Cav1',
  'CD44'='Cd44', 'LDLR'='Ldlr', 'SCD1'='Scd1',
  'ACSL3'='Acsl3', 'FASN'='Fasn', 'ACLY'='Acly',
  'LDHA'='Ldha', 'PKM'='Pkm', 'HK1'='Hk1', 'HK2'='Hk2',
  'G6PD'='G6pdx', 'PGD'='Pgd', 'TKT'='Tkt',
  'FADS2'='Fads2', 'ELOVL5'='Elovl5', 'ELOVL2'='Elovl2',
  'ELOVL6'='Elovl6', 'SREBF1'='Srebf1', 'SREBF2'='Srebf2',
  'HMGCR'='Hmgcr', 'HMGCS1'='Hmgcs1', 'LSS'='Lss',
  'CYP51A1'='Cyp51', 'SQLE'='Sqle', 'DHCR7'='Dhcr7',
  'DHCR24'='Dhcr24', 'SCARB1'='Scarb1', 'ABCA1'='Abca1',
  'ABCG1'='Abcg1', 'SOAT1'='Soat1',
  'LIPA'='Lipa', 'LIPE'='Lipe', 'PNPLA2'='Pnpla2',
  'MGLL'='Mgll', 'PLIN1'='Plin1', 'PLIN2'='Plin2',
  'PPARA'='Ppara', 'PPARG'='Pparg', 'ACSS2'='Acss2',
  'NF2'='Nf2', 'OTUB1'='Otub1', 'MUC1'='Muc1',
  'CARS1'='Cars1', 'GLS2'='Gls2', 'PROM2'='Prom2',
  'MT1G'='Mt1', 'CISD1'='Cisd1', 'CISD2'='Cisd2',
  'RNF217'='Rnf217', 'SLC39A7'='Slc39a7',
  'LTF'='Ltf', 'HSP90AA1'='Hsp90aa1', 'HSP90AB1'='Hsp90ab1',
  'HSPA8'='Hspa8', 'DNAJB1'='Dnajb1', 'BAG3'='Bag3',
  'STUB1'='Stub1', 'PSMD14'='Psmd14', 'USP7'='Usp7',
  'USP11'='Usp11', 'USP14'='Usp14', 'USP30'='Usp30',
  'OTUD1'='Otud1', 'ATXN3'='Atxn3', 'YOD1'='Yod1',
  'ZFYVE1'='Zfyve1', 'RAB33B'='Rab33b', 'RAB8A'='Rab8a',
  'RAB10'='Rab10', 'RAB14'='Rab14', 'OPTN'='Optn',
  'CALCOCO2'='Calcoco2', 'TAX1BP1'='Tax1bp1',
  'SNAP29'='Snap29', 'STX17'='Stx17', 'VAMP8'='Vamp8',
  'PIK3C3'='Pik3c3', 'PIK3R4'='Pik3r4', 'ATG14'='Atg14',
  'UVRAG'='Uvrag', 'AMBRA1'='Ambra1', 'NRBF2'='Nrbf2',
  'WDR45'='Wdr45', 'WDR45B'='Wdr45b', 'WIPI1'='Wipi1',
  'WIPI2'='Wipi2', 'ATG2A'='Atg2a', 'ATG2B'='Atg2b',
  'ATG9A'='Atg9a', 'ATG9B'='Atg9b', 'ATG13'='Atg13',
  'RB1CC1'='Rb1cc1', 'ULK1'='Ulk1', 'ULK2'='Ulk2',
  'ULK3'='Ulk3', 'ATG101'='Atg101', 'RPTOR'='Rptor',
  'MLST8'='Mlst8', 'DEPTOR'='Deptor', 'AKT1S1'='Akt1s1',
  'TSC1'='Tsc1', 'TSC2'='Tsc2', 'RHEB'='Rheb',
  'RHEBL1'='Rhebl1', 'RRAGA'='Rraga', 'RRAGB'='Rragb',
  'RRAGC'='Rragc', 'RRAGD'='Rragd', 'LAMTOR1'='Lamtor1',
  'LAMTOR2'='Lamtor2', 'LAMTOR3'='Lamtor3', 'LAMTOR4'='Lamtor4',
  'LAMTOR5'='Lamtor5', 'SESN1'='Sesn1', 'SESN2'='Sesn2',
  'SESN3'='Sesn3', 'DDIT4'='Ddit4', 'DDIT4L'='Ddit4l',
  'BNIP3'='Bnip3', 'BNIP3L'='Bnip3l', 'FUNDC1'='Fundc1',
  'BCL2L13'='Bcl2l13', 'FKBP8'='Fkbp8', 'PHB2'='Phb2',
  'NLRX1'='Nlrx1', 'MFN1'='Mfn1', 'MFN2'='Mfn2',
  'OPA1'='Opa1', 'DNM1L'='Dnm1l', 'FIS1'='Fis1',
  'MFF'='Mff', 'MIEF1'='Mief1', 'MIEF2'='Mief2',
  'MTFP1'='Mtfp1', 'GDAP1'='Gdap1', 'SLC25A46'='Slc25a46',
  'MSTO1'='Msto1', 'YME1L1'='Yme1l1', 'OMA1'='Oma1',
  'AFG3L2'='Afg3l2', 'SPG7'='Spg7', 'PARL'='Parl',
  'PINK1'='Pink1', 'PARK2'='Park2', 'FBXO7'='Fbxo7',
  'MUL1'='Mul1', 'MARCH5'='March5', 'RNF185'='Rnf185',
  'HUWE1'='Huwe1', 'ARIH1'='Arih1', 'SIAH1'='Siah1',
  'SIAH2'='Siah2', 'PARK7'='Park7', 'HTRA2'='Htra2',
  'CHCHD2'='Chchd2', 'CHCHD10'='Chchd10', 'PRDX3'='Prdx3',
  'PRDX5'='Prdx5', 'TXN2'='Txn2', 'TXNRD2'='Txnrd2',
  'GLRX2'='Glrx2', 'NDUFS1'='Ndufs1', 'SDHA'='Sdha',
  'SDHB'='Sdhb', 'SDHC'='Sdhc', 'SDHD'='Sdhd', 'FH'='Fh1',
  'IDH2'='Idh2', 'IDH3A'='Idh3a', 'IDH3B'='Idh3b',
  'IDH3G'='Idh3g', 'OGDH'='Ogdh', 'DLST'='Dlst', 'DLD'='Dld',
  'CS'='Cs', 'ACO2'='Aco2', 'MDH2'='Mdh2', 'ME2'='Me2',
  'PDHA1'='Pdha1', 'PDHB'='Pdhb', 'PDHX'='Pdhx',
  'PDK1'='Pdk1', 'PDK2'='Pdk2', 'PDK3'='Pdk3', 'PDK4'='Pdk4',
  'PDP1'='Pdp1', 'PDP2'='Pdp2', 'MPC1'='Mpc1', 'MPC2'='Mpc2',
  'DPYSL4'='Dpysl4', 'AKT2'='Akt2', 'AKT3'='Akt3',
  'PIK3CA'='Pik3ca', 'PIK3CB'='Pik3cb', 'PIK3CD'='Pik3cd',
  'PIK3CG'='Pik3cg', 'PIK3R1'='Pik3r1', 'PIK3R2'='Pik3r2',
  'PIK3R3'='Pik3r3', 'PTEN'='Pten', 'EIF4EBP1'='Eif4ebp1',
  'RPS6KB1'='Rps6kb1', 'RPS6'='Rps6', 'EIF4E'='Eif4e',
  'EIF4G1'='Eif4g1', 'MKNK1'='Mknk1', 'MKNK2'='Mknk2',
  'MAP2K1'='Map2k1', 'MAP2K2'='Map2k2', 'MAP2K3'='Map2k3',
  'MAP2K4'='Map2k4', 'MAP2K6'='Map2k6', 'MAP2K7'='Map2k7',
  'BRAF'='Braf', 'RAF1'='Raf1', 'ARAF'='Araf',
  'HRAS'='Hras', 'KRAS'='Kras', 'NRAS'='Nras',
  'SOS1'='Sos1', 'SOS2'='Sos2', 'GRB2'='Grb2',
  'SHC1'='Shc1', 'GAB1'='Gab1', 'GAB2'='Gab2',
  'FRS2'='Frs2', 'FRS3'='Frs3', 'IRS1'='Irs1', 'IRS2'='Irs2',
  'IGF1R'='Igf1r', 'INSR'='Insr', 'EGFR'='Egfr',
  'ERBB2'='Erbb2', 'ERBB3'='Erbb3', 'ERBB4'='Erbb4',
  'FGFR1'='Fgfr1', 'FGFR2'='Fgfr2', 'FGFR3'='Fgfr3',
  'FGFR4'='Fgfr4', 'PDGFRA'='Pdgfra', 'PDGFRB'='Pdgfrb',
  'KIT'='Kit', 'FLT3'='Flt3', 'CSF1R'='Csf1r',
  'MET'='Met', 'RET'='Ret', 'ALK'='Alk', 'ROS1'='Ros1',
  'NTRK1'='Ntrk1', 'NTRK2'='Ntrk2', 'NTRK3'='Ntrk3',
  'AXL'='Axl', 'TYRO3'='Tyro3', 'MERTK'='Mertk',
  'AGER'='Ager', 'METTL3'='Mettl3', 'METTL14'='Mettl14',
  'YTHDC2'='Ythdc2', 'FANCD2'='Fancd2',
  'SLC25A28'='Slc25a28', 'SLC25A37'='Slc25a37',
  'ABCB7'='Abcb7', 'ABCB8'='Abcb8', 'ABCB10'='Abcb10',
  'GLRX5'='Glrx5', 'BOLA1'='Bola1', 'BOLA3'='Bola3',
  'NFU1'='Nfu1', 'IBA57'='Iba57', 'ISCA1'='Isca1',
  'ISCA2'='Isca2', 'FDX1'='Fdx1', 'FDX2'='Fdx2', 'FDXR'='Fdxr',
  'LIAS'='Lias', 'DLAT'='Dlat', 'HEPH'='Heph',
  'HEPHL1'='Hephl1', 'TFR2'='Tfr2',
  'EGLN1'='Egln1', 'EGLN2'='Egln2', 'EGLN3'='Egln3',
  'SCD'='Scd'
)

# Convert human gene symbols to rodent (mouse/rat) gene symbols
# Mouse and rat share the same capitalization convention (First letter uppercase, rest lowercase)
# e.g., human "GPX4" → rodent "Gpx4", human "TP53" → rodent "Trp53" (via explicit mapping)
convert_to_rodent <- function(human_genes, mapping) {
  rodent_genes <- character()
  for (hg in human_genes) {
    if (hg %in% names(mapping)) {
      rodent_genes <- c(rodent_genes, mapping[[hg]])
    } else {
      # Heuristic: capitalize first letter, lowercase rest
      # WARNING: This may produce incorrect symbols for some genes (e.g., those with
      # different names in rodents). Always prefer explicit mapping entries.
      rodent_genes <- c(rodent_genes, paste0(toupper(substr(hg, 1, 1)), tolower(substr(hg, 2, nchar(hg)))))
    }
  }
  return(rodent_genes)
}

# ============================================================================
# GSEA Function
# ============================================================================
run_gsea_for_dataset <- function(ds, meta, ferroaging_genes, ferroptosis_genes, ferroptosis_specific,
                                  ferroaging_specific, ferroptosis_sub, ferroptosis_sub_specific) {
  cat(sprintf("\n%s\n", paste(rep("=", 70), collapse = "")))
  cat(sprintf("GSEA: %s\n", ds))
  cat(sprintf("%s\n", paste(rep("=", 70), collapse = "")))

  # Load DE results (use gene-level file for probe-based datasets)
  de_gene_file <- file.path(l1_results, paste0(ds, "_DE_gene_level.csv"))
  if (file.exists(de_gene_file)) {
    de <- fread(de_gene_file)
    # Determine species first
    if ("Species" %in% colnames(de)) {
      species <- de$Species[1]
    } else {
      # Fallback: infer from dataset name (GSE104036=Mouse, GSE16561=Human, GSE61616=Rat, GSE97537=Rat)
      species <- ifelse(ds == "GSE104036", "Mouse",
                        ifelse(ds == "GSE16561", "Human",
                               ifelse(ds %in% c("GSE61616", "GSE97537"), "Rat", "Human")))
    }
    # Gene column selection:
    # - Mouse (GSE104036): OriginalID = mouse gene symbols (e.g. Hspa1a) → use OriginalID
    # - Rat (GSE61616, GSE97537): OriginalID = probe IDs (e.g. 1378366_at) → use GeneSymbol
    # - Human (GSE16561): OriginalID = probe IDs (e.g. ILMN_...) → use GeneSymbol
    if (species == "Mouse" && "OriginalID" %in% colnames(de)) {
      gene_col <- "OriginalID"
    } else if ("GeneSymbol" %in% colnames(de)) {
      gene_col <- "GeneSymbol"
    } else if ("Gene" %in% colnames(de)) {
      gene_col <- "Gene"
    } else {
      cat("  ERROR: No gene column found\n")
      return(NULL)
    }
    fc_col <- "logFC"
  } else {
    de_file <- file.path(l1_results, paste0(ds, "_DE_results.csv"))
    de <- fread(de_file)
    if ("Gene" %in% colnames(de)) {
      gene_col <- "Gene"
    } else if ("GeneSymbol" %in% colnames(de)) {
      gene_col <- "GeneSymbol"
    } else if ("Probe" %in% colnames(de)) {
      # Fallback: probe-level file lacks gene symbols, try Probe column
      gene_col <- "Probe"
    } else {
      cat("  ERROR: No gene column found\n")
      return(NULL)
    }
    fc_col <- "logFC"
    # Determine species
    if ("Species" %in% colnames(de)) {
      species <- de$Species[1]
    } else {
      species <- ifelse(ds == "GSE104036", "Mouse",
                        ifelse(ds == "GSE16561", "Human",
                               ifelse(ds %in% c("GSE61616", "GSE97537"), "Rat", "Human")))
    }
  }
  cat(sprintf("  DE genes: %d\n", nrow(de)))
  cat(sprintf("  Species: %s\n", species))

  # Create ranked gene list
  if (fc_col %in% colnames(de) && gene_col %in% colnames(de)) {
    ranks <- de[[fc_col]]
    names(ranks) <- de[[gene_col]]
    # Remove NA and Inf
    ranks <- ranks[!is.na(ranks)]
    ranks <- ranks[is.finite(ranks)]
    # Average duplicate genes (take max absolute logFC)
    if (any(duplicated(names(ranks)))) {
      ranks <- tapply(ranks, names(ranks), function(x) x[which.max(abs(x))])
    }
    # Sort descending
    ranks <- sort(ranks, decreasing = TRUE)
  } else {
    cat(sprintf("  ERROR: %s or %s column missing\n", fc_col, gene_col))
    return(NULL)
  }

  # Check minimum gene count for meaningful GSEA
  if (length(ranks) < 1000) {
    cat(sprintf("  WARNING: Only %d genes in ranked list (min 1000 required). Skipping GSEA.\n", length(ranks)))
    cat(sprintf("  This usually means the gene-level DE file is corrupted or incomplete.\n"))
    return(NULL)
  }

  # Choose gene sets based on species
  if (species == "Mouse" || species == "Rat") {
    fa_genes <- convert_to_rodent(ferroaging_genes, human_to_mouse_map)
    fa_specific <- convert_to_rodent(ferroaging_specific, human_to_mouse_map)
    ft_genes <- convert_to_rodent(ferroptosis_genes, human_to_mouse_map)
    ft_specific <- convert_to_rodent(ferroptosis_specific, human_to_mouse_map)
    ft_drivers <- convert_to_rodent(ferroptosis_sub$drivers, human_to_mouse_map)
    ft_suppressors <- convert_to_rodent(ferroptosis_sub$suppressors, human_to_mouse_map)
    ft_markers <- convert_to_rodent(ferroptosis_sub$markers, human_to_mouse_map)
    ft_drivers_sp <- convert_to_rodent(ferroptosis_sub_specific$drivers, human_to_mouse_map)
    ft_suppressors_sp <- convert_to_rodent(ferroptosis_sub_specific$suppressors, human_to_mouse_map)
    ft_markers_sp <- convert_to_rodent(ferroptosis_sub_specific$markers, human_to_mouse_map)
  } else {
    fa_genes <- ferroaging_genes
    fa_specific <- ferroaging_specific
    ft_genes <- ferroptosis_genes
    ft_specific <- ferroptosis_specific
    ft_drivers <- ferroptosis_sub$drivers
    ft_suppressors <- ferroptosis_sub$suppressors
    ft_markers <- ferroptosis_sub$markers
    ft_drivers_sp <- ferroptosis_sub_specific$drivers
    ft_suppressors_sp <- ferroptosis_sub_specific$suppressors
    ft_markers_sp <- ferroptosis_sub_specific$markers
  }

  # Build gene sets list
  # NOTE: "Ferroptosis" = all FerrDb genes (includes overlap with ferroaging)
  #       "Ferroptosis_Specific" = FerrDb genes EXCLUDING ferroaging overlap → proper control
  #       "Ferroaging_Specific" = Ferroaging genes EXCLUDING ferroptosis overlap → key test
  gene_sets <- list(
    "Ferroaging" = fa_genes,
    "Ferroaging_Specific" = fa_specific,
    "Ferroptosis" = ft_genes,
    "Ferroptosis_Specific" = ft_specific,
    "Ferroptosis_Drivers" = ft_drivers,
    "Ferroptosis_Suppressors" = ft_suppressors,
    "Ferroptosis_Markers" = ft_markers,
    "Ferroptosis_Drivers_Sp" = ft_drivers_sp,
    "Ferroptosis_Suppressors_Sp" = ft_suppressors_sp,
    "Ferroptosis_Markers_Sp" = ft_markers_sp
  )

  # Count genes present in ranked list
  for (gs_name in names(gene_sets)) {
    n_present <- sum(gene_sets[[gs_name]] %in% names(ranks))
    cat(sprintf("  %s: %d / %d genes in ranked list\n", gs_name, n_present, length(gene_sets[[gs_name]])))
  }

  # Run fgsea (fgseaMultilevel — recommended by package authors)
  # Removed nperm parameter to switch from fgseaSimple to fgseaMultilevel
  cat("  Running fgsea (Multilevel)...\n")
  set.seed(42)
  fgsea_res <- fgsea(
    pathways = gene_sets,
    stats = ranks,
    minSize = 5,
    maxSize = 1000,
    nproc = 1
  )

  fgsea_res$dataset <- ds
  fgsea_res$species <- species

  return(list(
    results = fgsea_res,
    ranks = ranks,
    gene_sets = gene_sets
  ))
}

# ============================================================================
# Load sample metadata
# ============================================================================
meta_file <- file.path(l1_results, "dataset_summary.csv")
if (file.exists(meta_file)) {
  meta <- fread(meta_file)
} else {
  # Build from individual meta files
  meta_list <- list()
  for (ds in c("GSE104036", "GSE16561", "GSE61616", "GSE97537")) {
    mf <- file.path(l1_results, paste0(ds, "_sample_meta.csv"))
    if (file.exists(mf)) {
      m <- fread(mf)
      m$Dataset <- ds
      meta_list[[ds]] <- m
    }
  }
  meta <- rbindlist(meta_list, fill = TRUE)
}

# ============================================================================
# Define ferroptosis sub-gene sets for detailed analysis
ferroptosis_sub <- list(
  drivers = ferroptosis_drivers,
  suppressors = ferroptosis_suppressors,
  markers = ferroptosis_markers
)

# ============================================================================
# Run GSEA for all datasets
# ============================================================================
datasets_to_run <- c("GSE104036", "GSE16561", "GSE61616", "GSE97537")
# GSE37587 removed: ILMN probe annotation is for wrong chip (v3.0 vs v4.0), only 94 genes mapped
all_results <- list()
all_ranks <- list()
all_gene_sets <- list()

# Define ferroptosis sub-category specific sets (for the function call)
ferroptosis_sub_specific <- list(
  drivers = ferroptosis_drivers_specific,
  suppressors = ferroptosis_suppressors_specific,
  markers = ferroptosis_markers_specific
)

for (ds in datasets_to_run) {
  # Check for gene-level file first (preferred), then fall back to probe-level results
  de_gene_file <- file.path(l1_results, paste0(ds, "_DE_gene_level.csv"))
  de_results_file <- file.path(l1_results, paste0(ds, "_DE_results.csv"))
  if (!file.exists(de_gene_file) && !file.exists(de_results_file)) {
    cat(sprintf("\nSkipping %s: no DE results found (neither gene-level nor probe-level)\n", ds))
    next
  }

  res <- run_gsea_for_dataset(ds, meta, ferroaging_genes, ferroptosis_all,
                              ferroptosis_specific, ferroaging_specific,
                              ferroptosis_sub, ferroptosis_sub_specific)
  if (!is.null(res)) {
    all_results[[ds]] <- res$results
    all_ranks[[ds]] <- res$ranks
    all_gene_sets[[ds]] <- res$gene_sets
  }
}

# ============================================================================
# Combine and save results
# ============================================================================
if (length(all_results) > 0) {
  combined <- rbindlist(all_results)

  # Format for output
  combined_out <- combined[, .(
    dataset, pathway, pval, padj, NES, ES, size, leadingEdge
  )]

  # Convert leadingEdge list to string
  combined_out$leadingEdge <- sapply(combined$leadingEdge, function(x) paste(x, collapse = ";"))

  fwrite(combined_out, file.path(results_dir, "gsea_ferroaging_vs_ferroptosis.csv"))
  cat(sprintf("\nGSEA results saved: %d rows\n", nrow(combined_out)))

  # Print summary
  cat(paste0("\n", paste(rep("=", 70), collapse = ""), "\n"))
  cat("GSEA RESULTS SUMMARY\n")
  cat(paste0(paste(rep("=", 70), collapse = ""), "\n\n"))

  for (ds in names(all_results)) {
    cat(sprintf("--- %s ---\n", ds))
    ds_res <- combined[dataset == ds]
    for (i in 1:nrow(ds_res)) {
      sig_mark <- ifelse(ds_res$padj[i] < 0.05, "*** SIGNIFICANT ***", "")
      cat(sprintf("  %-25s  NES=%+7.3f  pval=%.4f  padj=%.4f  size=%d  %s\n",
                  ds_res$pathway[i], ds_res$NES[i], ds_res$pval[i], ds_res$padj[i],
                  ds_res$size[i], sig_mark))
    }
    cat("\n")
  }
}

# ============================================================================
# Visualization 1: NES bar chart per dataset
# ============================================================================
if (length(all_results) > 0) {
  plot_data <- combined
  plot_data$pathway <- factor(plot_data$pathway,
                              levels = c("Ferroaging", "Ferroaging_Specific",
                                         "Ferroptosis", "Ferroptosis_Specific",
                                         "Ferroptosis_Drivers", "Ferroptosis_Suppressors",
                                         "Ferroptosis_Markers",
                                         "Ferroptosis_Drivers_Sp", "Ferroptosis_Suppressors_Sp",
                                         "Ferroptosis_Markers_Sp"))

  # Color by significance
  plot_data$significance <- ifelse(plot_data$padj < 0.05, "padj < 0.05",
                                   ifelse(plot_data$pval < 0.05, "pval < 0.05", "NS"))

  p1 <- ggplot(plot_data, aes(x = pathway, y = NES, fill = significance)) +
    geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    facet_wrap(~ dataset, ncol = 1, scales = "free_y") +
    scale_fill_manual(values = c("padj < 0.05" = "#E74C3C", "pval < 0.05" = "#F39C12", "NS" = "#95A5A6")) +
    labs(title = "GSEA: Ferroaging vs Ferroptosis Enrichment Across Datasets",
         subtitle = "FerrDb V2 ferroptosis gene set as control",
         x = "Gene Set", y = "Normalized Enrichment Score (NES)",
         fill = "Significance") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1),
          strip.text = element_text(face = "bold", size = 11))

  ggsave(file.path(fig_dir, "gsea_nes_barplot.png"), p1, width = 12, height = 14, dpi = 300)
  cat(sprintf("NES bar plot saved: %s\n", file.path(fig_dir, "gsea_nes_barplot.png")))
}

# ============================================================================
# Visualization 2: GSEA running score plots for key comparisons
# ============================================================================
for (ds in names(all_ranks)) {
  ranks <- all_ranks[[ds]]
  ds_res <- combined[dataset == ds]
  gs_list <- all_gene_sets[[ds]]

  # Only plot significant pathways (padj < 0.05)
  sig_pathways <- ds_res[padj < 0.05]

  if (nrow(sig_pathways) == 0) {
    cat(sprintf("  %s: No significant pathways, skipping enrichment plots\n", ds))
    next
  }

  n_plots <- min(nrow(sig_pathways), 3)
  plot_list <- list()

  for (i in 1:n_plots) {
    pw <- sig_pathways$pathway[i]
    # Use the FULL gene set (not leadingEdge) for the running score plot
    # This ensures the ES curve matches the NES reported by fgsea
    full_gs <- gs_list[[pw]]
    if (is.null(full_gs)) next

    # Guard: skip if gene set has insufficient overlap with ranked list
    gs_in_ranks <- intersect(full_gs, names(ranks))
    if (length(gs_in_ranks) < 3) {
      cat(sprintf("  WARNING: %s has only %d genes in ranked list, skipping plot\n", pw, length(gs_in_ranks)))
      next
    }

    p <- plotEnrichment(
      pathway = full_gs,
      stats = ranks
    ) +
    ggtitle(sprintf("%s: %s\nNES=%+.3f, padj=%.4f", ds, pw,
                    sig_pathways$NES[i], sig_pathways$padj[i])) +
    theme_minimal(base_size = 11) +
    xlab("Rank") + ylab("Enrichment Score")

    plot_list[[pw]] <- p
  }

  if (length(plot_list) > 0) {
    combined_plot <- wrap_plots(plot_list, ncol = 1)
    ggsave(file.path(fig_dir, sprintf("gsea_running_score_%s.png", ds)),
           combined_plot, width = 10, height = 4 * length(plot_list), dpi = 300)
    cat(sprintf("Running score plots saved for %s\n", ds))
  }
}

# ============================================================================
# Visualization 3: Ferroaging vs Ferroptosis NES comparison scatter
# ============================================================================
if (length(all_results) > 0) {
  # Extract Ferroaging and Ferroptosis NES per dataset (both full and specific sets)
  compare_data <- data.frame()
  for (ds in names(all_results)) {
    ds_res <- combined[dataset == ds]
    fa_row <- ds_res[pathway == "Ferroaging"]
    ft_row <- ds_res[pathway == "Ferroptosis"]
    fa_sp_row <- ds_res[pathway == "Ferroaging_Specific"]
    ft_sp_row <- ds_res[pathway == "Ferroptosis_Specific"]

    if (nrow(fa_row) > 0 && nrow(ft_row) > 0) {
      compare_data <- rbind(compare_data, data.frame(
        dataset = ds,
        Ferroaging_NES = fa_row$NES[1],
        Ferroptosis_NES = ft_row$NES[1],
        Ferroaging_padj = fa_row$padj[1],
        Ferroptosis_padj = ft_row$padj[1],
        Ferroaging_Specific_NES = if (nrow(fa_sp_row) > 0) fa_sp_row$NES[1] else NA,
        Ferroptosis_Specific_NES = if (nrow(ft_sp_row) > 0) ft_sp_row$NES[1] else NA,
        Ferroaging_Specific_padj = if (nrow(fa_sp_row) > 0) fa_sp_row$padj[1] else NA,
        Ferroptosis_Specific_padj = if (nrow(ft_sp_row) > 0) ft_sp_row$padj[1] else NA,
        stringsAsFactors = FALSE
      ))
    }
  }

  if (nrow(compare_data) > 0) {
    # Long format for plotting (Full sets)
    compare_long <- rbind(
      data.frame(dataset = compare_data$dataset, NES = compare_data$Ferroaging_NES,
                 padj = compare_data$Ferroaging_padj, GeneSet = "Ferroaging"),
      data.frame(dataset = compare_data$dataset, NES = compare_data$Ferroptosis_NES,
                 padj = compare_data$Ferroptosis_padj, GeneSet = "Ferroptosis")
    )

    p2 <- ggplot(compare_long, aes(x = dataset, y = NES, fill = GeneSet)) +
      geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      scale_fill_manual(values = c("Ferroaging" = "#E74C3C", "Ferroptosis" = "#3498DB")) +
      labs(title = "Ferroaging vs Ferroptosis: NES Comparison (Full Gene Sets)",
           subtitle = "Includes shared genes between the two sets",
           x = "Dataset", y = "Normalized Enrichment Score (NES)") +
      theme_minimal(base_size = 13) +
      theme(axis.text.x = element_text(angle = 30, hjust = 1))

    ggsave(file.path(fig_dir, "gsea_ferroaging_vs_ferroptosis_nes.png"), p2,
           width = 10, height = 6, dpi = 300)

    # Long format for plotting (Specific sets — key comparison, no shared genes)
    compare_long_sp <- rbind(
      data.frame(dataset = compare_data$dataset, NES = compare_data$Ferroaging_Specific_NES,
                 padj = compare_data$Ferroaging_Specific_padj, GeneSet = "Ferroaging_Specific"),
      data.frame(dataset = compare_data$dataset, NES = compare_data$Ferroptosis_Specific_NES,
                 padj = compare_data$Ferroptosis_Specific_padj, GeneSet = "Ferroptosis_Specific")
    )
    compare_long_sp <- compare_long_sp[!is.na(compare_long_sp$NES), ]

    if (nrow(compare_long_sp) > 0) {
      p3 <- ggplot(compare_long_sp, aes(x = dataset, y = NES, fill = GeneSet)) +
        geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
        geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
        scale_fill_manual(values = c("Ferroaging_Specific" = "#C0392B", "Ferroptosis_Specific" = "#2980B9")) +
        labs(title = "Ferroaging vs Ferroptosis: NES Comparison (Specific — No Shared Genes)",
             subtitle = "Key test: ferroaging signal independent of ferroptosis",
             x = "Dataset", y = "Normalized Enrichment Score (NES)") +
        theme_minimal(base_size = 13) +
        theme(axis.text.x = element_text(angle = 30, hjust = 1))

      ggsave(file.path(fig_dir, "gsea_ferroaging_vs_ferroptosis_specific_nes.png"), p3,
             width = 10, height = 6, dpi = 300)
      cat(sprintf("Ferroaging vs Ferroptosis (Specific) comparison plot saved\n"))
    }

    cat(sprintf("Ferroaging vs Ferroptosis comparison plot saved\n"))

    # Print comparison table
    cat("\n=== Ferroaging vs Ferroptosis NES Comparison ===\n")
    print(compare_data)
  }
}

# ============================================================================
# Visualization 4: Heatmap of NES across datasets
# ============================================================================
if (length(all_results) > 0) {
  # Build NES matrix
  nes_matrix <- dcast(combined, pathway ~ dataset, value.var = "NES", fill = 0)
  nes_mat <- as.matrix(nes_matrix[, -1, with = FALSE])
  rownames(nes_mat) <- nes_matrix$pathway

  # Padjust matrix
  padj_matrix <- dcast(combined, pathway ~ dataset, value.var = "padj", fill = 1)
  padj_mat <- as.matrix(padj_matrix[, -1, with = FALSE])
  rownames(padj_mat) <- padj_matrix$pathway

  # Create annotation
  annot_df <- data.frame(
    Ferroaging = rownames(nes_mat) == "Ferroaging",
    row.names = rownames(nes_mat)
  )

  # Save matrices
  fwrite(as.data.table(nes_mat, keep.rownames = "pathway"),
         file.path(results_dir, "gsea_nes_matrix.csv"))
  fwrite(as.data.table(padj_mat, keep.rownames = "pathway"),
         file.path(results_dir, "gsea_padj_matrix.csv"))

  cat("\nNES matrix saved to gsea_nes_matrix.csv\n")
  cat("Padjust matrix saved to gsea_padj_matrix.csv\n")
}

# ============================================================================
# Final summary
# ============================================================================
cat(paste0("\n", paste(rep("=", 70), collapse = ""), "\n"))
cat("GSEA VALIDATION COMPLETED\n")
cat(paste0(paste(rep("=", 70), collapse = ""), "\n"))

cat("\nKey questions answered:\n")
cat("1. Is ferroaging gene set enriched in stroke vs control? -> See NES and padj\n")
cat("2. Is ferroaging distinct from ferroptosis? -> Compare NES patterns\n")
cat("3. Which datasets show strongest ferroaging signal? -> See NES ranking\n\n")

cat("Output files:\n")
cat("  L2/results/gsea_ferroaging_vs_ferroptosis.csv - Full GSEA results\n")
cat("  L2/results/gsea_nes_matrix.csv - NES matrix\n")
cat("  L2/results/gsea_padj_matrix.csv - Adjusted p-value matrix\n")
cat("  L2/results/figures/gsea_nes_barplot.png - NES bar chart\n")
cat("  L2/results/figures/gsea_ferroaging_vs_ferroptosis_nes.png - Comparison\n")
cat("  L2/results/figures/gsea_running_score_*.png - Running score plots\n")