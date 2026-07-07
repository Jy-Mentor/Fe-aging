#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

"""P4输入文件完整性验证"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
L2 = BASE / "L2" / "results"
L3 = BASE / "L3" / "results"
L4 = BASE / "L4" / "results"

errors = []
warnings = []

# 1. L3 compound data
print("=== L3 化合物数据 ===")
cp = pd.read_csv(L3 / "tcm_compound_pool_filtered.csv")
print(f"  tcm_compound_pool_filtered.csv: {cp.shape}")
assert "SMILES_std" in cp.columns, "SMILES_std列缺失"
assert "MOL_ID" in cp.columns, "MOL_ID列缺失"
assert "molecule_name" in cp.columns, "molecule_name列缺失"
print(f"  化合物数: {len(cp)}")

desc = pd.read_csv(L3 / "rdkit_descriptors.csv")
print(f"  rdkit_descriptors.csv: {desc.shape}")
assert len(desc) == len(cp), f"描述符行数({len(desc)}) != 化合物数({len(cp)})"

ecfp4 = np.load(L3 / "ecfp4_fingerprints.npy")
print(f"  ecfp4_fingerprints.npy: {ecfp4.shape}, dtype={ecfp4.dtype}")
assert len(ecfp4) == len(cp), f"ECFP4行数({len(ecfp4)}) != 化合物数({len(cp)})"

maccs = np.load(L3 / "maccs_fingerprints.npy")
print(f"  maccs_fingerprints.npy: {maccs.shape}, dtype={maccs.dtype}")
assert len(maccs) == len(cp), f"MACCS行数({len(maccs)}) != 化合物数({len(cp)})"

# 2. L2 protein data
print("\n=== L2 蛋白质数据 ===")
prot = pd.read_csv(L2 / "target_protein_features.csv")
print(f"  target_protein_features.csv: {prot.shape}")
assert "gene_symbol" in prot.columns, "gene_symbol列缺失"
print(f"  基因数: {len(prot)}")

aac = pd.read_csv(L2 / "protein_descriptors.csv")
print(f"  protein_descriptors.csv: {aac.shape}")
aac_cols = [c for c in aac.columns if c.startswith("AAC_")]
print(f"  AAC_列数: {len(aac_cols)}")
assert len(aac_cols) > 0, "AAC特征列为空"
assert len(aac) == len(prot), f"AAC行数({len(aac)}) != 蛋白数({len(prot)})"

pseaac = pd.read_csv(L2 / "protein_pseaac.csv")
print(f"  protein_pseaac.csv: {pseaac.shape}")
pseaac_cols = [c for c in pseaac.columns if c.startswith("PseAAC_")]
print(f"  PseAAC_列数: {len(pseaac_cols)}")
assert len(pseaac_cols) == 50, f"PseAAC列数({len(pseaac_cols)}) != 50"
assert len(pseaac) == len(prot), f"PseAAC行数({len(pseaac)}) != 蛋白数({len(prot)})"

# 3. L4 active data
print("\n=== L4 活性数据 ===")
for f in ["chembl_active_compounds.csv", "bindingdb_active_compounds.csv"]:
    p = L4 / f
    if p.exists():
        df = pd.read_csv(p)
        print(f"  {f}: EXISTS ({len(df)} rows)")
    else:
        warnings.append(f"  {f}: MISSING (将仅使用内置文献活性数据)")

# 4. Core genes match
print("\n=== 核心基因匹配 ===")
prot_genes = set(prot["gene_symbol"].values)
CORE = [
    "EMP1","SAT1","TLR4","LCN2","EPHA4","CXCL10","KLF6","SP1",
    "CD74","PTGS2","IRF1","FBXO31","LGMN","IGFBP7","IL1B","MAPK1",
    "KDM6B","PDE4B","RUNX3","CTSB","LACTB","LPCAT3","EGR1","BCL6",
    "GMFB","HBP1","SOD1","DYRK1A"
]
PRIORITY = [
    "ACSL4","GPX4","HMOX1","FTH1","FTL","SLC7A11","TFRC",
    "TLR4","PTGS2","IL1B","MAPK1","NFE2L2","TP53","STAT3"
]
all_targets = set(CORE + PRIORITY)
missing = [g for g in all_targets if g not in prot_genes]
if missing:
    errors.append(f"蛋白质数据缺失基因: {missing}")
else:
    print(f"  所有 {len(all_targets)} 个靶标基因均在蛋白数据中")

# 5. SMILES有效性抽检
print("\n=== SMILES有效性抽检 ===")
from rdkit import Chem
null_smiles = cp["SMILES_std"].isna().sum()
print(f"  空SMILES: {null_smiles}")
if null_smiles > 0:
    errors.append(f"空SMILES: {null_smiles}")

# 抽检前100个
valid = 0
for i in range(min(100, len(cp))):
    mol = Chem.MolFromSmiles(cp.iloc[i]["SMILES_std"])
    if mol is not None:
        valid += 1
print(f"  SMILES有效性(前100): {valid}/100")

# 6. Summary
print(f"\n{'='*60}")
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  [ERROR] {e}")
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  [WARN] {w}")

if not errors:
    print("所有输入文件验证通过!")
else:
    print("验证失败, 请修复以上错误后重试")
    exit(1)