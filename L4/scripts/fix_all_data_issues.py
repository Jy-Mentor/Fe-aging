#!/usr/bin/env python3
"""
数据问题全面修复脚本
基于 validate_data_authenticity.py 的 v25 校验结果，修复以下问题：

P0 (Critical):
  - TCM-CPI数据泄漏: 18个重叠SMILES → 从TCM池中移除
P1 (High):
  - PPI重复边: 107,351条 → 去重（标准化边方向：A<B排序）
  - BDB无效SMILES: 217条 → 删除无效SMILES行
  - CPI_SUPP无效SMILES: 6条 → 删除无效SMILES行
  - CPI_SUPP重复对: 1条 → 去重
P2 (Medium):
  - 疾病基因非人类基因: 11条大鼠基因 → 过滤
  - CPI非数值列: 1,817+18,318条 → 标记但不强制修复（pchembl_value天然含空值）

输出：所有修复后的文件覆盖原文件，备份保存至 *_backup_v25
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_RESULTS = PROJECT_ROOT / "L4" / "results"
L4_V10 = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"

for d in [L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "fix_all_data_issues.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

BACKUP_SUFFIX = f"_backup_v25_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def backup_file(path: Path) -> Path:
    """备份原文件"""
    backup_path = Path(str(path) + BACKUP_SUFFIX)
    if path.exists():
        shutil.copy2(path, backup_path)
        logger.info("备份: %s -> %s", path.name, backup_path.name)
    return backup_path


def is_valid_smiles(smi):
    """校验SMILES有效性"""
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception:
        return False


# ============================================================
# P0: TCM-CPI数据泄漏修复
# ============================================================
def fix_tcm_cpi_leakage():
    """从TCM候选池中移除与CPI训练集重叠的SMILES"""
    logger.info("=" * 60)
    logger.info("[P0] TCM-CPI 数据泄漏修复")
    logger.info("=" * 60)

    tcm_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    cpi_path = L4_RESULTS / "experimental_actives_detail_cleaned_combined.csv"

    if not tcm_path.exists():
        logger.error("TCM文件不存在: %s", tcm_path)
        return False
    if not cpi_path.exists():
        logger.error("CPI文件不存在: %s", cpi_path)
        return False

    backup_file(tcm_path)

    tcm_df = pd.read_csv(tcm_path, low_memory=False)
    cpi_df = pd.read_csv(cpi_path, low_memory=False)

    # 确定SMILES列
    smiles_col = None
    for col in ["SMILES_std", "canonical_smiles", "SMILES", "smiles"]:
        if col in tcm_df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        logger.error("TCM池未找到SMILES列")
        return False

    cpi_smiles_set = set(cpi_df["canonical_smiles"].dropna().astype(str).str.strip().unique())

    # 规范化TCM SMILES
    def canonicalize(smi):
        if pd.isna(smi):
            return None
        s = str(smi).strip()
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    tcm_smiles_raw = tcm_df[smiles_col].astype(str).str.strip()
    tcm_smiles_canon = tcm_smiles_raw.apply(canonicalize)

    # 检测重叠：原始匹配 + canonical匹配
    overlap_mask = tcm_smiles_raw.isin(cpi_smiles_set) | tcm_smiles_canon.isin(cpi_smiles_set)
    n_overlap = overlap_mask.sum()

    if n_overlap == 0:
        logger.info("TCM池与CPI训练集无重叠SMILES，无需修复")
        return True

    logger.warning("发现 %d 个重叠SMILES，将从TCM池中移除", n_overlap)
    overlap_records = tcm_df.loc[overlap_mask, ["MOL_ID", smiles_col]].copy()
    for _, row in overlap_records.iterrows():
        logger.warning("  移除: MOL_ID=%s, SMILES=%s", row["MOL_ID"], row[smiles_col])

    cleaned_df = tcm_df.loc[~overlap_mask].copy()
    cleaned_df.to_csv(tcm_path, index=False)
    logger.info("修复完成: TCM池 %d -> %d 个化合物", len(tcm_df), len(cleaned_df))
    return True


# ============================================================
# P1: PPI重复边去重（significant_edges + supplement）
# ============================================================
def fix_ppi_duplicates():
    """PPI网络重复边去重（标准化边方向：A<B排序）"""
    logger.info("=" * 60)
    logger.info("[P1] PPI 重复边去重")
    logger.info("=" * 60)

    ppi_files = [
        ("significant_edges", L1_RESULTS / "ppi_network_extended_significant_edges.csv"),
        ("supplemented", L1_RESULTS / "ppi_network_supplemented.csv"),
    ]

    all_ok = True
    for name, ppi_path in ppi_files:
        if not ppi_path.exists():
            logger.warning("PPI文件不存在 (%s): %s", name, ppi_path)
            continue

        backup_file(ppi_path)

        df = pd.read_csv(ppi_path, low_memory=False)
        original_n = len(df)
        logger.info("PPI (%s): 原始边数 %d", name, original_n)

        # 标准化边：gene_a < gene_b
        def normalize_edge(row):
            a, b = row["gene_a"], row["gene_b"]
            if a <= b:
                return pd.Series({"gene_a": a, "gene_b": b, "combined_score": row["combined_score"]})
            else:
                return pd.Series({"gene_a": b, "gene_b": a, "combined_score": row["combined_score"]})

        normalized = df.apply(normalize_edge, axis=1)
        n_dup = normalized.duplicated(subset=["gene_a", "gene_b"]).sum()
        logger.info("PPI (%s): 重复边数 %d", name, n_dup)

        if n_dup == 0:
            logger.info("PPI (%s): 无重复边，无需修复", name)
            continue

        # 对于重复边，保留combined_score最高的那条
        normalized_unique = normalized.sort_values("combined_score", ascending=False).drop_duplicates(
            subset=["gene_a", "gene_b"], keep="first"
        )
        n_removed = original_n - len(normalized_unique)
        logger.info("PPI (%s): 移除 %d 条重复边，保留 %d 条", name, n_removed, len(normalized_unique))

        normalized_unique.to_csv(ppi_path, index=False)
        logger.info("PPI (%s): 修复完成 %d -> %d", name, original_n, len(normalized_unique))

    return all_ok


# ============================================================
# P1: BDB无效SMILES修复
# ============================================================
def fix_bdb_invalid_smiles():
    """BindingDB数据中删除无效SMILES行"""
    logger.info("=" * 60)
    logger.info("[P1] BDB 无效SMILES修复")
    logger.info("=" * 60)

    bdb_path = L4_RESULTS / "bindingdb_active_compounds.csv"
    if not bdb_path.exists():
        logger.error("BDB文件不存在: %s", bdb_path)
        return False

    backup_file(bdb_path)

    df = pd.read_csv(bdb_path, low_memory=False)
    original_n = len(df)

    # 确定SMILES列
    smiles_col = None
    for col in ["canonical_smiles", "SMILES", "smiles", "Smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        logger.error("BDB未找到SMILES列, 可用列: %s", list(df.columns))
        return False

    valid_mask = df[smiles_col].apply(is_valid_smiles)
    n_invalid = (~valid_mask).sum()

    if n_invalid == 0:
        logger.info("无无效SMILES，无需修复")
        return True

    logger.warning("发现 %d 条无效SMILES，将删除", n_invalid)
    invalid_samples = df.loc[~valid_mask, smiles_col].head(10).tolist()
    logger.warning("无效SMILES示例: %s", invalid_samples)

    cleaned_df = df.loc[valid_mask].copy()
    cleaned_df.to_csv(bdb_path, index=False)
    logger.info("修复完成: BDB %d -> %d 条", original_n, len(cleaned_df))
    return True


# ============================================================
# P1: CPI_SUPP无效SMILES + 重复修复
# ============================================================
def fix_cpi_supplement():
    """CPI补充数据：删除无效SMILES + 去重"""
    logger.info("=" * 60)
    logger.info("[P1] CPI_SUPP 无效SMILES + 重复修复")
    logger.info("=" * 60)

    supp_path = L4_V10 / "cpi_supplement_v25.csv"
    if not supp_path.exists():
        logger.error("CPI_SUPP文件不存在: %s", supp_path)
        return False

    backup_file(supp_path)

    df = pd.read_csv(supp_path, low_memory=False)
    original_n = len(df)

    # 确定SMILES列
    smiles_col = None
    for col in ["canonical_smiles", "SMILES", "smiles"]:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        logger.error("CPI_SUPP未找到SMILES列, 可用列: %s", list(df.columns))
        return False

    # 1. 删除无效SMILES
    valid_mask = df[smiles_col].apply(is_valid_smiles)
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning("发现 %d 条无效SMILES，将删除", n_invalid)
        invalid_samples = df.loc[~valid_mask, smiles_col].tolist()
        logger.warning("无效SMILES: %s", invalid_samples)
        df = df.loc[valid_mask].copy()

    # 2. 去重 (gene + SMILES)
    if "gene" in df.columns and smiles_col in df.columns:
        dup_mask = df.duplicated(subset=["gene", smiles_col], keep="first")
        n_dup = dup_mask.sum()
        if n_dup > 0:
            logger.warning("发现 %d 条重复 gene+SMILES 对，将去重", n_dup)
            dup_records = df.loc[dup_mask, ["gene", smiles_col]]
            for _, row in dup_records.iterrows():
                logger.warning("  重复: gene=%s, SMILES=%s", row["gene"], row[smiles_col])
            df = df.loc[~dup_mask].copy()

    df.to_csv(supp_path, index=False)
    logger.info("修复完成: CPI_SUPP %d -> %d 条", original_n, len(df))
    return True


# ============================================================
# P2: 疾病基因数据中非人类基因过滤
# ============================================================
def fix_disease_gene_nonhuman():
    """过滤疾病基因数据中的非人类基因（大鼠/小鼠基因）"""
    logger.info("=" * 60)
    logger.info("[P2] 疾病基因非人类基因过滤")
    logger.info("=" * 60)

    dg_path = L4_V10 / "disease_gene_edges.csv"
    if not dg_path.exists():
        logger.error("疾病基因文件不存在: %s", dg_path)
        return False

    backup_file(dg_path)

    df = pd.read_csv(dg_path, low_memory=False)
    original_n = len(df)

    # 大鼠基因模式：以数字结尾的大写字母开头（如 RT1-BA, RT1-CE10）
    # 小鼠基因模式：首字母小写（如 Tnf, Il6）
    # 人类基因模式：全大写字母+数字
    import re

    human_gene_pattern = re.compile(r"^[A-Z][A-Z0-9]*$")

    gene_col = None
    for col in ["gene_symbol", "gene", "Gene"]:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        logger.error("疾病基因未找到基因列, 可用列: %s", list(df.columns))
        return False

    human_mask = df[gene_col].apply(
        lambda g: bool(human_gene_pattern.match(str(g))) if pd.notna(g) else False
    )
    n_nonhuman = (~human_mask).sum()

    if n_nonhuman == 0:
        logger.info("无非人类基因，无需修复")
        return True

    logger.warning("发现 %d 条非人类基因，将删除", n_nonhuman)
    nonhuman_genes = df.loc[~human_mask, gene_col].unique().tolist()
    logger.warning("非人类基因: %s", nonhuman_genes)

    cleaned_df = df.loc[human_mask].copy()
    cleaned_df.to_csv(dg_path, index=False)
    logger.info("修复完成: 疾病基因 %d -> %d 条", original_n, len(cleaned_df))
    return True


# ============================================================
# P2: CPI非数值列处理
# ============================================================
def fix_cpi_non_numeric():
    """CPI数据中standard_value_nM和pchembl_value非数值列处理"""
    logger.info("=" * 60)
    logger.info("[P2] CPI 非数值列处理")
    logger.info("=" * 60)

    cpi_path = L4_RESULTS / "experimental_actives_detail_cleaned_combined.csv"
    if not cpi_path.exists():
        logger.error("CPI文件不存在: %s", cpi_path)
        return False

    backup_file(cpi_path)

    df = pd.read_csv(cpi_path, low_memory=False)

    # standard_value_nM: 非数值行 → 设为NaN（pchembl_value天然可为空）
    if "standard_value_nM" in df.columns:
        numeric_vals = pd.to_numeric(df["standard_value_nM"], errors="coerce")
        n_nonnumeric = numeric_vals.isna().sum() - df["standard_value_nM"].isna().sum()
        if n_nonnumeric > 0:
            logger.warning("standard_value_nM: %d 条非数值 → 设为NaN", n_nonnumeric)
            df["standard_value_nM"] = numeric_vals

    if "pchembl_value" in df.columns:
        numeric_vals = pd.to_numeric(df["pchembl_value"], errors="coerce")
        n_nonnumeric = numeric_vals.isna().sum() - df["pchembl_value"].isna().sum()
        if n_nonnumeric > 0:
            logger.warning("pchembl_value: %d 条非数值 → 设为NaN", n_nonnumeric)
            df["pchembl_value"] = numeric_vals

    df.to_csv(cpi_path, index=False)
    logger.info("CPI非数值列处理完成")
    return True


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("数据问题全面修复")
    logger.info("执行时间: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 60)

    results = {}

    # P0: TCM-CPI泄漏
    results["P0-TCM泄漏"] = fix_tcm_cpi_leakage()

    # P1: PPI重复边
    results["P1-PPI重复"] = fix_ppi_duplicates()

    # P1: BDB无效SMILES
    results["P1-BDB-SMILES"] = fix_bdb_invalid_smiles()

    # P1: CPI_SUPP修复
    results["P1-CPI_SUPP"] = fix_cpi_supplement()

    # P2: 疾病基因
    results["P2-疾病基因"] = fix_disease_gene_nonhuman()

    # P2: CPI非数值
    results["P2-CPI非数值"] = fix_cpi_non_numeric()

    # Summary
    logger.info("=" * 60)
    logger.info("修复总结")
    logger.info("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        logger.info("  [%s] %s", "PASS" if ok else "FAIL", name)
    logger.info("\n  %d/%d 修复成功", passed, total)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())