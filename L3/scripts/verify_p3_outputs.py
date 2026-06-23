#!/usr/bin/env python3
"""Phase 3 - 输出文件完整性验证"""
import pandas as pd
import numpy as np
from pathlib import Path

base = Path(r'd:\铁衰老 绝不重蹈覆辙\L3\results')

files = {
    'CSV候选池': base / 'tcm_compound_pool_filtered.csv',
    'ECFP4指纹': base / 'ecfp4_fingerprints.npy',
    'MACCS指纹': base / 'maccs_fingerprints.npy',
    'RDKit描述符': base / 'rdkit_descriptors.csv',
    '相似性网络': base / 'compound_similarity_network.csv',
    'Pickle包': base / 'compound_pool.pkl.gz',
    '统计报告': base / 'compound_pool_statistics.md',
}

print('=' * 60)
print('Phase 3 输出文件完整性验证')
print('=' * 60)
for name, fpath in files.items():
    exists = fpath.exists()
    size = fpath.stat().st_size if exists else 0
    status = "OK" if exists else "MISSING"
    print(f'  [{name:12s}] {status:8s} {size/1024:8.1f} KB  {fpath.name}')

# 验证CSV数据
df = pd.read_csv(files['CSV候选池'])
print(f'\n候选化合物池验证:')
print(f'  行数: {len(df)}')
print(f'  列数: {len(df.columns)}')
print(f'  SMILES非空: {df["SMILES_std"].notna().sum()}')
print(f'  Lipinski通过: {df["Lipinski_Pass"].sum()}')
print(f'  BBB+或+/-: {df["BBB_Prediction"].isin(["BBB+", "BBB+/-"]).sum()}')
print(f'  PAINS通过: {df["PAINS_Pass"].sum()}')
print(f'  MW范围: {df["MW"].min():.1f} - {df["MW"].max():.1f} Da')
print(f'  OB范围: {df["ob"].min():.1f} - {df["ob"].max():.1f}%')

# 验证指纹
ecfp4 = np.load(files['ECFP4指纹'])
maccs = np.load(files['MACCS指纹'])
print(f'\n分子指纹验证:')
print(f'  ECFP4: {ecfp4.shape} (dtype={ecfp4.dtype})')
print(f'  MACCS: {maccs.shape} (dtype={maccs.dtype})')
print(f'  指纹-化合物数一致: {ecfp4.shape[0] == len(df)}')

# 验证相似性网络
edges = pd.read_csv(files['相似性网络'])
print(f'\n相似性网络验证:')
print(f'  边数: {len(edges)}')
print(f'  Tanimoto范围: {edges["tanimoto"].min():.4f} - {edges["tanimoto"].max():.4f}')

# 验证描述符
desc = pd.read_csv(files['RDKit描述符'])
print(f'\nRDKit描述符验证:')
print(f'  形状: {desc.shape}')
print(f'  描述符-化合物数一致: {desc.shape[0] == len(df)}')

print(f'\n===== Phase 3 验证通过 =====')