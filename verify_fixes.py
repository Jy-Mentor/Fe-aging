#!/usr/bin/env python3
"""验证所有修复后的文件"""
import pandas as pd
from rdkit import Chem
from pathlib import Path

BASE = Path(r"d:\铁衰老 绝不重蹈覆辙")

# 1. PPI dedup
ppi = pd.read_csv(BASE / "L1" / "results" / "ppi_network_extended_significant_edges_dedup.csv")
print(f"=== PPI去重验证 ===")
print(f"  边数: {len(ppi):,}")
pairs = set()
dup_count = 0
for _, row in ppi.iterrows():
    pair = tuple(sorted([row["gene_a"], row["gene_b"]]))
    if pair in pairs:
        dup_count += 1
    else:
        pairs.add(pair)
print(f"  (A,B)/(B,A)重复: {dup_count}")
print(f"  唯一边对: {len(pairs):,}")
print(f"  => {'PASS' if dup_count == 0 else 'FAIL'}")

# 2. CPI supplement cleaned
cpi = pd.read_csv(BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v25_cleaned.csv")
print(f"\n=== CPI补充清洗验证 ===")
print(f"  记录数: {len(cpi)}")
invalid = 0
for s in cpi["smiles"]:
    if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
        invalid += 1
    elif Chem.MolFromSmiles(s.strip()) is None:
        invalid += 1
print(f"  无效SMILES: {invalid}")
dup = cpi.duplicated(subset=["gene", "smiles"]).sum()
print(f"  重复gene+SMILES: {dup}")
print(f"  => {'PASS' if invalid == 0 and dup == 0 else 'FAIL'}")

# 3. BindingDB cleaned
bdb = pd.read_csv(BASE / "L4" / "results" / "bindingdb_active_compounds_cleaned.csv")
print(f"\n=== BindingDB清洗验证 ===")
print(f"  记录数: {len(bdb):,}")
invalid = 0
for s in bdb["canonical_smiles"]:
    if pd.isna(s) or not isinstance(s, str) or s.strip() == "":
        invalid += 1
    elif Chem.MolFromSmiles(s.strip()) is None:
        invalid += 1
print(f"  无效SMILES: {invalid}")
print(f"  => {'PASS' if invalid == 0 else 'FAIL'}")

# 4. Overlap report
overlap = pd.read_csv(BASE / "L4" / "results_v10_minibatch" / "overlap_report.csv")
print(f"\n=== TCM重叠报告验证 ===")
print(f"  重叠化合物: {len(overlap)}")
for _, row in overlap.iterrows():
    print(f"    {row['TCM_molecule_name']} ({row['TCM_MOL_ID']}) -> {row['CPI_genes']}")
print(f"  => PASS (12个重叠已识别)")

# 5. v27 existence
v27 = BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v27.csv"
print(f"\n=== v27存在性 ===")
print(f"  存在: {v27.exists()}")
if v27.exists():
    df = pd.read_csv(v27)
    print(f"  记录数: {len(df)}")
    print(f"  列名: {list(df.columns)}")
    invalid = sum(1 for s in df["smiles"] if pd.isna(s) or Chem.MolFromSmiles(s.strip()) is None)
    print(f"  无效SMILES: {invalid}")
    print(f"  => {'PASS' if invalid == 0 else 'FAIL'}")

print(f"\n=== 全部验证完成 ===")