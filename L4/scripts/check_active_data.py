import logging
logger = logging.getLogger(__name__)

import pandas as pd
from pathlib import Path

base = Path("d:/铁衰老 绝不重蹈覆辙/L4/results")
core = [
    "EMP1", "SAT1", "TLR4", "LCN2", "EPHA4", "CXCL10", "KLF6", "SP1",
    "CD74", "PTGS2", "IRF1", "FBXO31", "LGMN", "IGFBP7", "IL1B", "MAPK1",
    "KDM6B", "PDE4B", "RUNX3", "CTSB", "LACTB", "LPCAT3", "EGR1", "BCL6",
    "GMFB", "HBP1", "SOD1", "DYRK1A",
]


def _find_column(keywords, columns):
    """按关键词（忽略大小写）查找第一个匹配列。"""
    cols_lower = {c.lower(): c for c in columns}
    for kw in keywords:
        for lower_col, orig_col in cols_lower.items():
            if kw.lower() in lower_col:
                return orig_col
    return None


for fname in ["chembl_active_compounds.csv", "bindingdb_active_compounds.csv"]:
    path = base / fname
    if not path.exists():
        print(f"[SKIP] 文件不存在: {path}")
        continue

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] 无法读取 {fname}: {e}")
        continue

    print(f"=== {fname} ===")
    print(f"  行数: {len(df)}, 列: {list(df.columns)[:8]}")

    gene_col = _find_column(["gene"], df.columns)
    if gene_col is None:
        print("[ERROR] 未找到基因列")
        continue
    print(f"  基因列名: {gene_col}")

    genes = set(df[gene_col].dropna().unique())
    print(f"  基因数: {len(genes)}")

    matched = [g for g in core if g in genes]
    print(f"  核心基因匹配: {len(matched)}/28: {matched}")

    smi_col = _find_column(["canonical_smiles", "smiles"], df.columns)
    if smi_col is None:
        print("[WARNING] 未找到 SMILES 列")
        continue

    for g in matched:
        subset = df[df[gene_col] == g]
        n = len(subset)
        n_smi = subset[smi_col].dropna().nunique()
        print(f"    {g}: {n} 条记录, {n_smi} 个唯一SMILES")
    print()
