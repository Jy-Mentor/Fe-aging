#!/usr/bin/env python
"""数据文件审计：真实性、准确性、完整性检查"""
import numpy as np
import pandas as pd
from rdkit import Chem

# === 1. CPI 数据审计 ===
print("=" * 60)
print("1. CPI 数据审计")
print("=" * 60)
cpi = pd.read_csv("d:/铁衰老 绝不重蹈覆辙/L4/results/experimental_actives_detail_cleaned.csv", low_memory=False)
print(f"总行数: {len(cpi)}")
print(f"列名: {list(cpi.columns)}")

for col in ["canonical_smiles", "gene", "organism"]:
    if col in cpi.columns:
        nulls = cpi[col].isna().sum()
        print(f"  {col}: 缺失={nulls}/{len(cpi)} ({100*nulls/len(cpi):.1f}%)")

smiles = cpi["canonical_smiles"].dropna().astype(str)
valid_smiles = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None)
print(f"  SMILES有效性: {valid_smiles}/{len(smiles)} ({100*valid_smiles/len(smiles):.1f}%)")
print(f"  唯一基因: {cpi['gene'].nunique()}")
print(f"  唯一SMILES: {cpi['canonical_smiles'].nunique()}")
print(f"  基因列表: {sorted(cpi['gene'].unique())}")

if "organism" in cpi.columns:
    org_dist = cpi["organism"].value_counts().to_dict()
    print(f"  Organism分布: {org_dist}")

dups = cpi.duplicated(subset=["canonical_smiles", "gene"]).sum()
print(f"  重复CPI对: {dups}")

# 每个基因的CPI数量
gene_counts = cpi["gene"].value_counts()
print(f"  每基因CPI数: min={gene_counts.min()}, max={gene_counts.max()}, mean={gene_counts.mean():.1f}")

# === 2. 化合物特征审计 ===
print()
print("=" * 60)
print("2. 化合物特征审计")
print("=" * 60)
data = np.load("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/compound_features_v31.npz", allow_pickle=True)
feats = data["features"]
smiles_arr = data["smiles"]
print(f"  特征矩阵: {feats.shape}")
print(f"  SMILES数量: {len(smiles_arr)}")
print(f"  特征NaN数: {np.isnan(feats).sum()}")
print(f"  特征Inf数: {np.isinf(feats).sum()}")
print(f"  特征min/max: {feats.min():.4f}/{feats.max():.4f}")
print(f"  特征mean/std: {feats.mean():.4f}/{feats.std():.4f}")

# === 3. 蛋白嵌入审计 ===
print()
print("=" * 60)
print("3. 蛋白嵌入审计")
print("=" * 60)
emb = np.load("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch/esm2_protein_embeddings.npz", allow_pickle=True)
emb_keys = sorted(emb.keys())
print(f"  蛋白数量: {len(emb_keys)}")
dims = [emb[k].shape for k in emb_keys[:3]]
print(f"  嵌入维度: {dims}")

cpi_genes = sorted(cpi["gene"].unique())
missing_emb = [g for g in cpi_genes if g not in emb_keys]
print(f"  CPI基因中有嵌入: {len(cpi_genes) - len(missing_emb)}/{len(cpi_genes)}")
if missing_emb:
    print(f"  缺失嵌入的基因: {missing_emb}")

# === 4. TCM 池审计 ===
print()
print("=" * 60)
print("4. TCM池审计")
print("=" * 60)
tcm = pd.read_csv("d:/铁衰老 绝不重蹈覆辙/L3/results/tcm_compound_pool_tox_filtered_noleak.csv", low_memory=False)
print(f"  化合物数: {len(tcm)}")
print(f"  列名: {list(tcm.columns)}")
tcm_smiles = tcm["SMILES_std"].astype(str)
tcm_valid = sum(1 for s in tcm_smiles if Chem.MolFromSmiles(s) is not None)
print(f"  SMILES有效性: {tcm_valid}/{len(tcm)} ({100*tcm_valid/len(tcm):.1f}%)")

# === 5. 铁衰老基因覆盖 ===
print()
print("=" * 60)
print("5. 铁衰老基因覆盖")
print("=" * 60)
ferroaging = [
    "ABCC1","ACSL4","ACVR1B","ALOX15","ATF3","ATG3","BAP1","BCL6","BRD7","CAVIN1",
    "CD74","CD82","CDO1","COX7A1","CTSB","CXCL10","DPEP1","DPP4","DUOX1","DYRK1A",
    "E2F1","E2F3","EBF3","EDN1","EGR1","EMP1","EPHA2","EPHA4","ERN1","FBXO31",
    "FOSL1","GMFB","HBP1","HERPUD1","HIF1A","HMGB1","HMOX1","ICA1","IFNG","IGFBP7",
    "IL1B","IL6","IRF1","IRF7","IRF9","KDM6B","KEAP1","KLF6","LACTB","LCN2",
    "LGMN","LIFR","LOX","LPCAT3","MAP3K14","MAPK1","MAPK14","MCU","MEN1","MPO",
    "NLRP3","NOX4","NR1D1","NR2F2","NUAK2","PADI4","PDE4B","PPP2R2B","PRKD1","PTBP1",
    "PTGS2","RBM3","RUNX3","S100A8","SAT1","SETD7","SLAMF8","SLC1A5","SMARCB1","SMURF2",
    "SNCA","SOCS1","SOCS2","SOD1","SP1","SPATA2","TBX2","TFRC","TLR4","TNFAIP1",
    "TNFAIP3","TXNIP","WNT5A","WWTR1","YAP1","ZEB1",
]
in_cpi = [g for g in ferroaging if g in cpi_genes]
in_emb = [g for g in ferroaging if g in emb_keys]
print(f"  铁衰老基因总数: {len(ferroaging)}")
print(f"  在CPI数据中: {len(in_cpi)}/{len(ferroaging)}")
print(f"  在ESM-2嵌入中: {len(in_emb)}/{len(ferroaging)}")
only_cpi_not_emb = [g for g in in_cpi if g not in emb_keys]
print(f"  有CPI但无嵌入: {only_cpi_not_emb}")
not_in_cpi = [g for g in ferroaging if g not in cpi_genes]
print(f"  铁衰老基因但不在CPI中: {len(not_in_cpi)}个: {not_in_cpi}")

print()
print("审计完成")