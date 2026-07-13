#!/usr/bin/env python3
"""修复TCM池中与CPI训练集和phenotype表型数据集的SMILES重叠（数据泄漏）

Bug-U fix: 不再创建冗余的 noleak 文件。去泄漏检查直接在 tox_filtered 上执行，
若发现泄漏则原地修复并备份。不再生成与源文件逐字节相同的 noleak 副本。
"""
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

L3 = Path("d:/铁衰老 绝不重蹈覆辙/L3/results")
L4 = Path("d:/铁衰老 绝不重蹈覆辙/L4/results")
L4_V10 = Path("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch")


def canon(smi):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def check_and_fix_leaks(tcm_path, cpi_path, pheno_path, inplace=False):
    """检查并修复 TCM 池中的 SMILES 泄漏。

    Args:
        tcm_path: TCM 候选池文件路径
        cpi_path: CPI 训练数据文件路径
        pheno_path: 表型数据集文件路径
        inplace: 是否原地修复（True=覆盖原文件并备份，False=仅报告）

    Returns:
        (n_leaked, n_removed): 泄漏数量和移除数量
    """
    tcm = pd.read_csv(tcm_path, low_memory=False)
    cpi = pd.read_csv(cpi_path, low_memory=False)
    pheno = pd.read_csv(pheno_path, low_memory=False)

    smiles_col = "SMILES_std" if "SMILES_std" in tcm.columns else "canonical_smiles"

    train_smiles = set()
    train_smiles.update(cpi["canonical_smiles"].dropna().astype(str).str.strip().unique())
    if "canonical_smiles" in pheno.columns:
        train_smiles.update(pheno["canonical_smiles"].dropna().astype(str).str.strip().unique())

    raw = tcm[smiles_col].astype(str).str.strip()
    can = raw.apply(canon)
    overlap = raw.isin(train_smiles) | can.isin(train_smiles)
    n_leaked = overlap.sum()

    print(f"File: {tcm_path.name}")
    print(f"  CPI train SMILES: {len(cpi['canonical_smiles'].dropna().unique())}")
    print(f"  Pheno train SMILES: {len(pheno['canonical_smiles'].dropna().unique()) if 'canonical_smiles' in pheno.columns else 0}")
    print(f"  Total unique train SMILES: {len(train_smiles)}")
    print(f"  Leaked TCM SMILES: {n_leaked}")

    n_removed = 0
    if n_leaked > 0 and inplace:
        for _, row in tcm[overlap].iterrows():
            print(f"    Removing: MOL_ID={row['MOL_ID']}, SMILES={row[smiles_col]}")
        # 备份原文件
        backup_path = tcm_path.with_suffix(".csv.bak")
        import shutil
        shutil.copy2(tcm_path, backup_path)
        print(f"  Backup saved: {backup_path}")
        cleaned = tcm[~overlap].copy()
        cleaned.to_csv(tcm_path, index=False)
        n_removed = len(tcm) - len(cleaned)
        print(f"  Fixed: {len(tcm)} -> {len(cleaned)} (removed {n_removed})")
    elif n_leaked > 0 and not inplace:
        print(f"  WARNING: {n_leaked} leaked SMILES found but not removed (inplace=False)")
    else:
        print("  No leaks found")

    return n_leaked, n_removed


cpi_path = L4 / "experimental_actives_detail_cleaned_combined.csv"
pheno_path = L4_V10 / "phenotype_ferroptosis_dataset_v25_clean.csv"

if not cpi_path.exists():
    raise FileNotFoundError(f"CPI 文件不存在: {cpi_path}")
if not pheno_path.exists():
    raise FileNotFoundError(f"表型文件不存在: {pheno_path}")

# Fix v21_Alevel
check_and_fix_leaks(L3 / "tcm_compound_pool_v21_Alevel.csv", cpi_path, pheno_path, inplace=True)

# Check tox_filtered — 原地修复
check_and_fix_leaks(L3 / "tcm_compound_pool_tox_filtered.csv", cpi_path, pheno_path, inplace=True)

# 删除冗余的 noleak 文件（Bug-U: 与 tox_filtered 逐字节相同）
noleak_path = L3 / "tcm_compound_pool_tox_filtered_noleak.csv"
if noleak_path.exists():
    import filecmp
    tox_path = L3 / "tcm_compound_pool_tox_filtered.csv"
    if filecmp.cmp(tox_path, noleak_path, shallow=False):
        noleak_path.unlink()
        print(f"\nDeleted redundant noleak file: {noleak_path} (identical to tox_filtered.csv)")
    else:
        print(f"\nWARNING: noleak differs from tox_filtered — keeping {noleak_path}")