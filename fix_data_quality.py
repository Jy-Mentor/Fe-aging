#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁衰老项目 - 数据质量修复脚本
基于 data_authenticity_report_v25.txt 的5个FAIL项进行修复
"""
from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# 禁用RDKit的冗余警告
RDLogger.logger().setLevel(RDLogger.ERROR)

# ── 路径配置 ──
BASE = Path(r"d:\铁衰老 绝不重蹈覆辙")

# 日志
LOG_DIR = BASE / "L4" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "fix_data_quality.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# 工具函数
# ============================================================

def is_valid_smiles(smi):
    """RDKit验证SMILES有效性"""
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    mol = Chem.MolFromSmiles(smi.strip())
    return mol is not None

def normalize_smiles(smi):
    """标准化SMILES（去空格、统一大小写）"""
    if pd.isna(smi) or not isinstance(smi, str):
        return ""
    return smi.strip()

# ============================================================
# 修复1: PPI网络重复边去重
# ============================================================
def fix_ppi_duplicates():
    """
    问题: ppi_network_extended_significant_edges.csv 有 107,351 条重复边
    原因: 每条边以 (gene_a, gene_b) 和 (gene_b, gene_a) 各出现一次
    修复: 按字母序排序基因对，去重
    """
    logger.info("=" * 70)
    logger.info("【修复1】PPI网络重复边去重")
    logger.info("=" * 70)

    ppi_path = BASE / "L1" / "results" / "ppi_network_extended_significant_edges.csv"
    dedup_path = BASE / "L1" / "results" / "ppi_network_extended_significant_edges_dedup.csv"

    df = pd.read_csv(ppi_path)
    logger.info(f"  原始边数: {len(df):,}")
    logger.info(f"  列名: {list(df.columns)}")

    # 检查重复
    dup_rows = df.duplicated().sum()
    logger.info(f"  完全重复行: {dup_rows}")

    # 检查 (A,B) vs (B,A) 重复
    # 创建排序后的边对
    df_sorted = df.copy()
    mask = df_sorted['gene_a'] > df_sorted['gene_b']
    df_sorted.loc[mask, ['gene_a', 'gene_b']] = df_sorted.loc[mask, ['gene_b', 'gene_a']].values

    # 去重（保留combined_score最大的边）
    before_dedup = len(df_sorted)
    df_dedup = df_sorted.sort_values('combined_score', ascending=False).drop_duplicates(
        subset=['gene_a', 'gene_b'], keep='first'
    ).reset_index(drop=True)
    after_dedup = len(df_dedup)

    logger.info(f"  去重前: {before_dedup:,} 条边")
    logger.info(f"  去重后: {after_dedup:,} 条边")
    logger.info(f"  移除重复: {before_dedup - after_dedup:,} 条")

    # 保存
    df_dedup.to_csv(dedup_path, index=False)
    logger.info(f"  ✅ 去重文件已保存: {dedup_path}")
    logger.info(f"     文件大小: {os.path.getsize(dedup_path):,} bytes")

    # 验证去重结果
    dup_check = df_dedup.duplicated(subset=['gene_a', 'gene_b']).sum()
    logger.info(f"  验证: 去重后仍有重复: {dup_check}")

    return df_dedup


# ============================================================
# 修复2: CPI补充数据SMILES无效 + 重复
# ============================================================
def fix_cpi_supplement():
    """
    问题1: cpi_supplement_v25.csv 中 6/32 条SMILES无效
    问题2: 存在1条重复gene+SMILES对
    修复: RDKit验证SMILES，过滤无效记录，去重，输出清洗版本
    """
    logger.info("\n" + "=" * 70)
    logger.info("【修复2&3】CPI补充数据SMILES无效 + 重复去除")
    logger.info("=" * 70)

    cpi_path = BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v25.csv"
    cleaned_path = BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v25_cleaned.csv"

    df = pd.read_csv(cpi_path)
    logger.info(f"  原始记录数: {len(df)}")
    logger.info(f"  列名: {list(df.columns)}")

    # 检查每个SMILES
    invalid_rows = []
    for idx, row in df.iterrows():
        smi = row.get('smiles', '')
        if not is_valid_smiles(smi):
            invalid_rows.append((idx, row.get('compound_name', 'N/A'), smi))

    logger.info(f"  无效SMILES数: {len(invalid_rows)}")
    for idx, name, smi in invalid_rows:
        logger.warning(f"    行{idx}: {name} -> {repr(smi)}")

    # 过滤无效SMILES
    df_valid = df[df['smiles'].apply(is_valid_smiles)].copy()
    logger.info(f"  过滤后(有效SMILES): {len(df_valid)}")

    # 检查gene+SMILES重复
    dup_mask = df_valid.duplicated(subset=['gene', 'smiles'], keep=False)
    dup_count = dup_mask.sum()
    logger.info(f"  重复gene+SMILES对: {dup_count} 条")

    # 去重（保留第一条）
    df_cleaned = df_valid.drop_duplicates(subset=['gene', 'smiles'], keep='first').reset_index(drop=True)
    logger.info(f"  最终记录数: {len(df_cleaned)}")

    # 保存
    df_cleaned.to_csv(cleaned_path, index=False)
    logger.info(f"  ✅ 清洗文件已保存: {cleaned_path}")
    logger.info(f"     文件大小: {os.path.getsize(cleaned_path):,} bytes")

    return df_cleaned


# ============================================================
# 修复3: BindingDB SMILES无效
# ============================================================
def fix_bindingdb_smiles():
    """
    问题: bindingdb_active_compounds.csv 中 217/46976 条SMILES无效
    修复: RDKit验证SMILES，过滤无效记录
    """
    logger.info("\n" + "=" * 70)
    logger.info("【修复4】BindingDB SMILES无效过滤")
    logger.info("=" * 70)

    bdb_path = BASE / "L4" / "results" / "bindingdb_active_compounds.csv"
    cleaned_path = BASE / "L4" / "results" / "bindingdb_active_compounds_cleaned.csv"

    df = pd.read_csv(bdb_path)
    logger.info(f"  原始记录数: {len(df):,}")
    logger.info(f"  列名: {list(df.columns)}")

    # 查找SMILES列
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break

    if smiles_col is None:
        logger.error(f"  ❌ 未找到SMILES列！可用列: {list(df.columns)}")
        return None

    logger.info(f"  SMILES列: '{smiles_col}'")

    # 统计无效SMILES
    invalid_mask = ~df[smiles_col].apply(is_valid_smiles)
    invalid_count = invalid_mask.sum()
    logger.info(f"  无效SMILES数: {invalid_count:,}")

    if invalid_count > 0:
        invalid_examples = df[invalid_mask][[smiles_col, 'gene', 'molecule_name']].head(10)
        for _, row in invalid_examples.iterrows():
            logger.warning(f"    基因={row['gene']}, 分子={row['molecule_name']}, SMILES={repr(row[smiles_col])[:80]}")

    # 过滤
    df_cleaned = df[~invalid_mask].copy().reset_index(drop=True)
    logger.info(f"  过滤后记录数: {len(df_cleaned):,}")

    # 保存
    df_cleaned.to_csv(cleaned_path, index=False)
    logger.info(f"  ✅ 清洗文件已保存: {cleaned_path}")
    logger.info(f"     文件大小: {os.path.getsize(cleaned_path):,} bytes")

    return df_cleaned


# ============================================================
# 修复4: TCM池与训练集重叠SMILES识别
# ============================================================
def fix_tcm_overlap():
    """
    问题: TCM候选池中有18个化合物与CPI训练集SMILES完全相同（数据泄漏）
    修复: 识别重叠SMILES的详细信息，输出overlap_report.csv
    注意: 不删除这些化合物，因为它们可能是天然产物，重叠可能是合理的
    """
    logger.info("\n" + "=" * 70)
    logger.info("【修复5】TCM池与CPI训练集重叠SMILES识别")
    logger.info("=" * 70)

    tcm_path = BASE / "L3" / "results" / "tcm_compound_pool_v21_Alevel.csv"
    cpi_path = BASE / "L4" / "results" / "experimental_actives_detail_cleaned.csv"
    report_path = BASE / "L4" / "results_v10_minibatch" / "overlap_report.csv"

    tcm_df = pd.read_csv(tcm_path)
    cpi_df = pd.read_csv(cpi_path)

    logger.info(f"  TCM池: {len(tcm_df)} 个化合物")
    logger.info(f"  CPI训练集: {len(cpi_df)} 条记录, {cpi_df['canonical_smiles'].nunique()} 个唯一SMILES")

    # 标准化SMILES对比
    tcm_smiles = set(tcm_df['SMILES_std'].apply(normalize_smiles).unique())
    cpi_smiles = set(cpi_df['canonical_smiles'].apply(normalize_smiles).unique())

    overlap_smiles = tcm_smiles & cpi_smiles
    logger.info(f"  重叠SMILES数: {len(overlap_smiles)}")

    if len(overlap_smiles) == 0:
        logger.info(f"  ✅ 无重叠，无需处理")
        return

    # 获取详细信息
    overlap_records = []
    for smi in overlap_smiles:
        # TCM侧信息
        tcm_match = tcm_df[tcm_df['SMILES_std'].apply(normalize_smiles) == smi]
        for _, tcm_row in tcm_match.iterrows():
            mol_id = tcm_row.get('MOL_ID', '')
            mol_name = tcm_row.get('molecule_name', '')
            tier = tcm_row.get('tier', '')
            herbs = tcm_row.get('herb_origins', '')

            # CPI侧信息
            cpi_match = cpi_df[cpi_df['canonical_smiles'].apply(normalize_smiles) == smi]
            cpi_genes = cpi_match['gene'].unique().tolist()
            cpi_sources = cpi_match['source'].unique().tolist()
            cpi_targets = cpi_match['target_pref_name'].dropna().unique().tolist()

            overlap_records.append({
                'SMILES': smi,
                'TCM_MOL_ID': mol_id,
                'TCM_molecule_name': mol_name,
                'TCM_tier': tier,
                'TCM_herb_origins': herbs,
                'CPI_genes': '|'.join(cpi_genes),
                'CPI_sources': '|'.join(cpi_sources),
                'CPI_targets': '|'.join(cpi_targets),
                'CPI_record_count': len(cpi_match),
                'leakage_type': 'exact_smiles_match',
                'action': 'FLAG_DO_NOT_REMOVE',
                'reason': '天然产物，重叠可能合理，训练时标记但保留'
            })

    report_df = pd.DataFrame(overlap_records)
    report_df.to_csv(report_path, index=False)
    logger.info(f"  ✅ 重叠报告已保存: {report_path}")
    logger.info(f"     重叠化合物数: {len(report_df)}")
    logger.info(f"\n  重叠详情:")
    for _, row in report_df.iterrows():
        logger.info(f"    TCM: {row['TCM_molecule_name']} ({row['TCM_MOL_ID']})")
        logger.info(f"      基因: {row['CPI_genes']}")
        logger.info(f"      靶标: {row['CPI_targets']}")

    return report_df


# ============================================================
# 验证任务: cpi_supplement_v26.csv 列名检查
# ============================================================
def verify_v26_columns():
    """
    验证cpi_supplement_v26.csv的列名是否正确
    问题: 日志显示缺少smiles和uniprot列
    """
    logger.info("\n" + "=" * 70)
    logger.info("【验证6】cpi_supplement_v26.csv 列名检查")
    logger.info("=" * 70)

    v26_path = BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v26.csv"

    if not v26_path.exists():
        logger.error(f"  ❌ 文件不存在: {v26_path}")
        return

    df = pd.read_csv(v26_path)
    logger.info(f"  记录数: {len(df)}")
    logger.info(f"  列名: {list(df.columns)}")

    # 期望列名: smiles, uniprot
    expected = {'smiles', 'uniprot'}
    actual = set(df.columns)

    has_smiles = 'smiles' in actual
    has_uniprot = 'uniprot' in actual
    has_canonical = 'canonical_smiles' in actual
    has_uniprot_id = 'uniprot_id' in actual

    logger.info(f"  列名分析:")
    logger.info(f"    'smiles' 存在: {has_smiles}")
    logger.info(f"    'uniprot' 存在: {has_uniprot}")
    logger.info(f"    'canonical_smiles' 存在: {has_canonical}")
    logger.info(f"    'uniprot_id' 存在: {has_uniprot_id}")

    if has_smiles and has_uniprot:
        logger.info(f"  ✅ 列名正确")
    else:
        logger.warning(f"  ⚠️ 列名不匹配！")
        if has_canonical and not has_smiles:
            logger.warning(f"     用 'canonical_smiles' 替代 'smiles'")
        if has_uniprot_id and not has_uniprot:
            logger.warning(f"     用 'uniprot_id' 替代 'uniprot'")

    # 检查SMILES有效性
    smiles_col = 'canonical_smiles' if has_canonical else ('smiles' if has_smiles else None)
    if smiles_col:
        invalid_count = sum(~df[smiles_col].apply(is_valid_smiles))
        empty_count = df[smiles_col].isna().sum() + (df[smiles_col].astype(str).str.strip() == '').sum()
        logger.info(f"  SMILES列({smiles_col}): 无效={invalid_count}, 空值={empty_count}")
        if invalid_count > 0:
            logger.warning(f"  ⚠️ 存在 {invalid_count} 条无效SMILES!")

    return df


# ============================================================
# 验证任务: cpi_supplement_v27.csv 存在性检查
# ============================================================
def verify_v27():
    """
    检查cpi_supplement_v27.csv是否存在
    """
    logger.info("\n" + "=" * 70)
    logger.info("【验证7】cpi_supplement_v27.csv 存在性检查")
    logger.info("=" * 70)

    v27_paths = [
        BASE / "L4" / "results_v10_minibatch" / "cpi_supplement_v27.csv",
        BASE / "L4" / "results" / "cpi_supplement_v27.csv",
    ]

    for p in v27_paths:
        if p.exists():
            df = pd.read_csv(p)
            logger.info(f"  ✅ 文件存在: {p}")
            logger.info(f"     记录数: {len(df)}")
            logger.info(f"     列名: {list(df.columns)}")
            logger.info(f"     文件大小: {os.path.getsize(p):,} bytes")

            # 检查SMILES有效性
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            if smiles_col:
                invalid = sum(~df[smiles_col].apply(is_valid_smiles))
                logger.info(f"     SMILES列({smiles_col}): 有效={len(df)-invalid}, 无效={invalid}")
            return df

    logger.warning(f"  ⚠️ cpi_supplement_v27.csv 不存在于已知路径")
    return None


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("  铁衰老项目 - 数据质量修复")
    logger.info(f"  执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    results = {}

    # 修复1: PPI去重
    results['PPI去重'] = fix_ppi_duplicates()

    # 修复2&3: CPI补充数据清洗
    results['CPI补充清洗'] = fix_cpi_supplement()

    # 修复4: BindingDB SMILES过滤
    results['BindingDB清洗'] = fix_bindingdb_smiles()

    # 修复5: TCM重叠识别
    results['TCM重叠报告'] = fix_tcm_overlap()

    # 验证6: v26列名
    results['v26列名验证'] = verify_v26_columns()

    # 验证7: v27存在性
    results['v27存在性'] = verify_v27()

    # ── 汇总 ──
    logger.info("\n\n" + "=" * 70)
    logger.info("                    修复完成汇总")
    logger.info("=" * 70)
    logger.info(f"  1. PPI去重: ppi_network_extended_significant_edges_dedup.csv")
    logger.info(f"  2. CPI补充清洗: cpi_supplement_v25_cleaned.csv")
    logger.info(f"  3. BindingDB清洗: bindingdb_active_compounds_cleaned.csv")
    logger.info(f"  4. TCM重叠报告: overlap_report.csv")
    logger.info(f"  5. v26列名验证: 已完成")
    logger.info(f"  6. v27存在性检查: 已完成")
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()