#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁衰老项目 - 数据完整性校验脚本
验证4个核心数据文件的真实性和完整性
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ── 日志配置 ──
LOG_DIR = Path(r"d:\铁衰老 绝不重蹈覆辙\L4\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "validate_data_integrity.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# ============================================================
# 工具函数
# ============================================================

def is_valid_smiles(smi):
    """检查SMILES字符串是否有效（真实化学结构）"""
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception as e:
        logger.warning(f"SMILES 解析异常: {smi!r}, 错误: {e}")
        return False

def check_file(path):
    """检查文件是否存在且非空"""
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    if os.path.getsize(path) == 0:
        return False, f"文件为空: {path}"
    return True, ""

# ============================================================
# 文件1: CPI数据
# ============================================================

def validate_cpi(path):
    logger.info("=" * 70)
    logger.info("【文件1】CPI数据验证")
    logger.info(f"路径: {path}")
    logger.info("=" * 70)
    
    issues = []
    
    # 0. 文件存在性
    ok, err = check_file(path)
    if not ok:
        logger.error(f"  ❌ 致命: {err}")
        return "ISSUE", [err]
    
    # 1. 读取
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"无法读取CSV: {e}"
        logger.error(f"  ❌ {msg}")
        return "ISSUE", [msg]
    
    logger.info(f"  📊 文件大小: {os.path.getsize(path):,} bytes")
    logger.info(f"  📊 列名: {list(df.columns)}")
    logger.info(f"  📊 总行数: {len(df)}")
    logger.info(f"  📊 前5行:")
    logger.info(df.head(5).to_string(index=False))
    
    # 2. 检查gene列空值
    gene_col = None
    for col in df.columns:
        if col.lower() in ['gene', 'gene_name', 'gene_symbol', 'symbol']:
            gene_col = col
            break
    if gene_col is None:
        issues.append("未找到gene/gene_name列")
        logger.warning(f"  ⚠️ 未找到gene列，现有列: {list(df.columns)}")
    else:
        gene_na = df[gene_col].isna().sum()
        if gene_na > 0:
            issues.append(f"gene列({gene_col})有 {gene_na} 个空值")
            logger.warning(f"  ⚠️ gene列({gene_col})空值数: {gene_na}")
        else:
            logger.info(f"  ✅ gene列({gene_col})无空值")
    
    # 3. 检查canonical_smiles列
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower() or 'smiles' in col.lower():
            smiles_col = col
            break
    if smiles_col is None:
        issues.append("未找到SMILES相关列")
        logger.warning(f"  ⚠️ 未找到SMILES列，现有列: {list(df.columns)}")
    else:
        smiles_na = df[smiles_col].isna().sum()
        logger.info(f"  📊 SMILES列({smiles_col})空值数: {smiles_na}")
        
        # 检查无效SMILES
        invalid_count = 0
        invalid_examples = []
        for idx, smi in df[smiles_col].items():
            if not is_valid_smiles(smi):
                invalid_count += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append((idx, smi))
        
        if invalid_count > 0:
            issues.append(f"SMILES列({smiles_col})有 {invalid_count} 个无效SMILES")
            logger.error(f"  ❌ 无效SMILES数: {invalid_count}")
            for idx, smi in invalid_examples:
                logger.error(f"     行{idx}: {repr(smi)}")
        else:
            logger.info(f"  ✅ 所有SMILES有效")
    
    # 4. 检查重复行
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        issues.append(f"存在 {dup_rows} 行完全重复")
        logger.warning(f"  ⚠️ 完全重复行数: {dup_rows}")
    else:
        logger.info(f"  ✅ 无完全重复行")
    
    # 5. 统计唯一基因数和唯一化合物数
    if gene_col:
        unique_genes = df[gene_col].dropna().nunique()
        logger.info(f"  📊 唯一基因数: {unique_genes}")
    if smiles_col:
        unique_smiles = df[smiles_col].dropna().nunique()
        logger.info(f"  📊 唯一化合物数(SMILES): {unique_smiles}")
    
    # 6. 检查uniprot_id列
    uniprot_col = None
    for col in df.columns:
        if 'uniprot' in col.lower():
            uniprot_col = col
            break
    if uniprot_col is None:
        issues.append("未找到uniprot_id相关列")
        logger.warning(f"  ⚠️ 未找到uniprot_id列，现有列: {list(df.columns)}")
    else:
        uniprot_na = df[uniprot_col].isna().sum()
        uniprot_total = len(df)
        if uniprot_na > 0:
            issues.append(f"uniprot_id列({uniprot_col})有 {uniprot_na}/{uniprot_total} 个空值")
            logger.warning(f"  ⚠️ uniprot_id列({uniprot_col})空值: {uniprot_na}/{uniprot_total}")
        else:
            logger.info(f"  ✅ uniprot_id列({uniprot_col})完整，无空值")
    
    # 7. 检查所有列的空值情况
    logger.info(f"\n  📊 各列空值统计:")
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            logger.info(f"     {col}: {na_count} 空值 ({(na_count/len(df))*100:.1f}%)")
    
    status = "ISSUE" if issues else "OK"
    return status, issues


# ============================================================
# 文件2: PPI网络
# ============================================================

def validate_ppi(path):
    logger.info("\n" + "=" * 70)
    logger.info("【文件2】PPI网络验证")
    logger.info(f"路径: {path}")
    logger.info("=" * 70)
    
    issues = []
    
    ok, err = check_file(path)
    if not ok:
        logger.error(f"  ❌ 致命: {err}")
        return "ISSUE", [err]
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"无法读取CSV: {e}"
        logger.error(f"  ❌ {msg}")
        return "ISSUE", [msg]
    
    logger.info(f"  📊 文件大小: {os.path.getsize(path):,} bytes")
    logger.info(f"  📊 列名: {list(df.columns)}")
    logger.info(f"  📊 总边数(行数): {len(df)}")
    logger.info(f"  📊 前5行:")
    logger.info(df.head(5).to_string(index=False))
    
    # 检查自环边
    self_loop_found = False
    for col_pair in [('source', 'target'), ('protein1', 'protein2'), ('node1', 'node2'),
                      ('gene1', 'gene2'), ('gene_a', 'gene_b'), ('from', 'to'),
                      ('preferredName_A', 'preferredName_B')]:
        if col_pair[0] in df.columns and col_pair[1] in df.columns:
            self_loops = df[df[col_pair[0]] == df[col_pair[1]]]
            if len(self_loops) > 0:
                issues.append(f"存在 {len(self_loops)} 条自环边 (列: {col_pair[0]}={col_pair[1]})")
                logger.error(f"  ❌ 自环边数: {len(self_loops)}")
                self_loop_found = True
            else:
                logger.info(f"  ✅ 无自环边 (列: {col_pair[0]}, {col_pair[1]})")
            break
    else:
        logger.warning(f"  ⚠️ 未找到标准source/target列对，现有列: {list(df.columns)}")
    
    # 统计唯一蛋白数
    protein_ids = set()
    for col in df.columns:
        if col.lower() in ['source', 'target', 'protein1', 'protein2', 'node1', 'node2',
                            'gene1', 'gene2', 'gene_a', 'gene_b', 'from', 'to',
                            'preferredname_a', 'preferredname_b', 'symbol_a', 'symbol_b']:
            protein_ids.update(df[col].dropna().unique())
    
    if protein_ids:
        logger.info(f"  📊 唯一蛋白/基因数: {len(protein_ids)}")
    else:
        logger.warning(f"  ⚠️ 无法自动识别蛋白ID列，手动统计所有列...")
        all_vals = set()
        for col in df.columns:
            if df[col].dtype == object:
                all_vals.update(df[col].dropna().unique())
        logger.info(f"  📊 所有列唯一值总数: {len(all_vals)}")
    
    # 空值检查
    total_na = 0
    for col in df.columns:
        na_count = df[col].isna().sum()
        if na_count > 0:
            issues.append(f"列 '{col}' 有 {na_count} 个空值")
            logger.warning(f"  ⚠️ 列 '{col}' 空值: {na_count}")
            total_na += na_count
    
    if total_na == 0:
        logger.info(f"  ✅ 所有列无空值")
    
    # 重复行检查
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        issues.append(f"存在 {dup_rows} 行完全重复")
        logger.warning(f"  ⚠️ 完全重复行数: {dup_rows}")
    else:
        logger.info(f"  ✅ 无完全重复行")
    
    status = "ISSUE" if issues else "OK"
    return status, issues


# ============================================================
# 文件3: 铁衰老96基因
# ============================================================

def validate_genes(path, cpi_path, ppi_path):
    logger.info("\n" + "=" * 70)
    logger.info("【文件3】铁衰老96基因验证")
    logger.info(f"路径: {path}")
    logger.info("=" * 70)
    
    issues = []
    
    ok, err = check_file(path)
    if not ok:
        logger.error(f"  ❌ 致命: {err}")
        return "ISSUE", [err]
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"无法读取CSV: {e}"
        logger.error(f"  ❌ {msg}")
        return "ISSUE", [msg]
    
    logger.info(f"  📊 文件大小: {os.path.getsize(path):,} bytes")
    logger.info(f"  📊 列名: {list(df.columns)}")
    logger.info(f"  📊 总行数: {len(df)}")
    
    # 尝试找基因列
    gene_col = None
    for col in df.columns:
        if col.lower() in ['gene', 'gene_name', 'gene_symbol', 'symbol', 'name', 'id']:
            gene_col = col
            break
    if gene_col is None:
        gene_col = df.columns[0]  # 默认第一列
    
    genes = df[gene_col].dropna().unique().tolist()
    logger.info(f"  📊 基因数: {len(genes)}")
    logger.info(f"  📊 前20个基因: {genes[:20]}")
    
    if len(genes) != 96:
        issues.append(f"基因数不是96，实际为 {len(genes)}")
        logger.warning(f"  ⚠️ 基因数不符！期望96，实际{len(genes)}")
    else:
        logger.info(f"  ✅ 基因数确认为96")
    
    # 检查是否有重复
    dup_genes = df[gene_col].duplicated().sum()
    if dup_genes > 0:
        issues.append(f"基因列表中有 {dup_genes} 个重复")
        logger.warning(f"  ⚠️ 重复基因数: {dup_genes}")
    else:
        logger.info(f"  ✅ 基因列表无重复")
    
    # 检查与CPI数据的覆盖
    if os.path.exists(cpi_path):
        try:
            cpi_df = pd.read_csv(cpi_path)
            cpi_gene_col = None
            for col in cpi_df.columns:
                if col.lower() in ['gene', 'gene_name', 'gene_symbol', 'symbol']:
                    cpi_gene_col = col
                    break
            if cpi_gene_col:
                cpi_genes = set(cpi_df[cpi_gene_col].dropna().unique())
                genes_set = set(genes)
                cpi_overlap = genes_set & cpi_genes
                cpi_missing = genes_set - cpi_genes
                logger.info(f"  📊 CPI数据覆盖: {len(cpi_overlap)}/{len(genes)} 基因 ({len(cpi_overlap)/len(genes)*100:.1f}%)")
                if cpi_missing:
                    issues.append(f"{len(cpi_missing)} 个基因在CPI数据中缺失: {list(cpi_missing)[:10]}")
                    logger.warning(f"  ⚠️ CPI缺失基因: {list(cpi_missing)[:10]}...")
            else:
                logger.warning(f"  ⚠️ CPI文件中未找到gene列")
        except Exception as e:
            logger.warning(f"  ⚠️ 无法读取CPI文件进行交叉验证: {e}")
    else:
        logger.warning(f"  ⚠️ CPI文件不存在，跳过交叉验证")
    
    # 检查与PPI数据的覆盖
    if os.path.exists(ppi_path):
        try:
            ppi_df = pd.read_csv(ppi_path)
            ppi_genes = set()
            for col in ppi_df.columns:
                if col.lower() in ['source', 'target', 'protein1', 'protein2', 'node1', 'node2',
                                    'gene1', 'gene2', 'gene_a', 'gene_b', 'preferredname_a', 'preferredname_b',
                                    'symbol_a', 'symbol_b']:
                    ppi_genes.update(ppi_df[col].dropna().unique())
            
            if ppi_genes:
                genes_set = set(genes)
                ppi_overlap = genes_set & ppi_genes
                ppi_missing = genes_set - ppi_genes
                logger.info(f"  📊 PPI网络覆盖: {len(ppi_overlap)}/{len(genes)} 基因 ({len(ppi_overlap)/len(genes)*100:.1f}%)")
                if ppi_missing:
                    issues.append(f"{len(ppi_missing)} 个基因在PPI网络中缺失: {list(ppi_missing)[:10]}")
                    logger.warning(f"  ⚠️ PPI缺失基因: {list(ppi_missing)[:10]}...")
            else:
                logger.warning(f"  ⚠️ 无法从PPI文件中提取基因列表")
        except Exception as e:
            logger.warning(f"  ⚠️ 无法读取PPI文件进行交叉验证: {e}")
    else:
        logger.warning(f"  ⚠️ PPI文件不存在，跳过交叉验证")
    
    status = "ISSUE" if issues else "OK"
    return status, issues


# ============================================================
# 文件4: 化合物池
# ============================================================

def validate_compound_pool(path, cpi_path):
    logger.info("\n" + "=" * 70)
    logger.info("【文件4】化合物池验证")
    logger.info(f"路径: {path}")
    logger.info("=" * 70)
    
    issues = []
    
    ok, err = check_file(path)
    if not ok:
        logger.error(f"  ❌ 致命: {err}")
        return "ISSUE", [err]
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        msg = f"无法读取CSV: {e}"
        logger.error(f"  ❌ {msg}")
        return "ISSUE", [msg]
    
    logger.info(f"  📊 文件大小: {os.path.getsize(path):,} bytes")
    logger.info(f"  📊 列名: {list(df.columns)}")
    logger.info(f"  📊 总化合物数(行数): {len(df)}")
    logger.info(f"  📊 前5行:")
    logger.info(df.head(5).to_string(index=False))
    
    # 查找SMILES列
    smiles_col = None
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    
    if smiles_col is None:
        issues.append("未找到SMILES列")
        logger.warning(f"  ⚠️ 未找到SMILES列，现有列: {list(df.columns)}")
    else:
        smiles_na = df[smiles_col].isna().sum()
        logger.info(f"  📊 SMILES列({smiles_col})空值: {smiles_na}/{len(df)}")
        
        # 检查无效SMILES
        invalid_count = 0
        invalid_examples = []
        for idx, smi in df[smiles_col].items():
            if not is_valid_smiles(smi):
                invalid_count += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append((idx, smi))
        
        if invalid_count > 0:
            issues.append(f"SMILES列({smiles_col})有 {invalid_count} 个无效SMILES")
            logger.error(f"  ❌ 无效SMILES数: {invalid_count}")
            for idx, smi in invalid_examples:
                logger.error(f"     行{idx}: {repr(smi)}")
        else:
            logger.info(f"  ✅ 所有SMILES有效")
        
        # 统计唯一化合物
        unique_smiles = df[smiles_col].dropna().nunique()
        logger.info(f"  📊 唯一化合物数(SMILES): {unique_smiles}")
    
    # 检查重复行
    dup_rows = df.duplicated().sum()
    if dup_rows > 0:
        issues.append(f"存在 {dup_rows} 行完全重复")
        logger.warning(f"  ⚠️ 完全重复行数: {dup_rows}")
    else:
        logger.info(f"  ✅ 无完全重复行")
    
    # 检查训练集泄漏 - 与CPI中的SMILES重叠
    if os.path.exists(cpi_path) and smiles_col:
        try:
            cpi_df = pd.read_csv(cpi_path)
            cpi_smiles_col = None
            for col in cpi_df.columns:
                if 'smiles' in col.lower():
                    cpi_smiles_col = col
                    break
            
            if cpi_smiles_col:
                cpi_smiles = set(cpi_df[cpi_smiles_col].dropna().unique())
                pool_smiles = set(df[smiles_col].dropna().unique())
                
                # 标准化比较（strip + uppercase）
                cpi_smiles_norm = set(s.strip().upper() for s in cpi_smiles if isinstance(s, str))
                pool_smiles_norm = set(s.strip().upper() for s in pool_smiles if isinstance(s, str))
                
                overlap = cpi_smiles_norm & pool_smiles_norm
                if overlap:
                    issues.append(f"训练集泄漏！CPI数据与化合物池有 {len(overlap)} 个重叠SMILES")
                    logger.error(f"  ❌ 训练集泄漏！重叠SMILES数: {len(overlap)}")
                    for smi in list(overlap)[:10]:
                        logger.error(f"     {smi}")
                else:
                    logger.info(f"  ✅ 无训练集泄漏（CPI与化合物池SMILES无重叠）")
            else:
                logger.warning(f"  ⚠️ CPI文件中未找到SMILES列，跳过泄漏检查")
        except Exception as e:
            logger.warning(f"  ⚠️ 无法读取CPI文件进行泄漏检查: {e}")
    else:
        if not os.path.exists(cpi_path):
            logger.warning(f"  ⚠️ CPI文件不存在，跳过泄漏检查")
        elif not smiles_col:
            logger.warning(f"  ⚠️ 无SMILES列，跳过泄漏检查")
    
    status = "ISSUE" if issues else "OK"
    return status, issues


# ============================================================
# 主流程
# ============================================================

def main():
    base = r"d:\铁衰老 绝不重蹈覆辙"
    
    files = {
        "CPI数据": os.path.join(base, "L4", "results", "experimental_actives_detail_cleaned.csv"),
        "PPI网络": os.path.join(base, "L1", "results", "ppi_network_extended_significant_edges.csv"),
        "铁衰老96基因": os.path.join(base, "L1", "results", "ferroaging_genes_96.csv"),
        "化合物池": os.path.join(base, "L3", "results", "tcm_compound_pool_v21_Alevel.csv"),
    }
    
    all_results = {}
    
    # 文件1: CPI
    status1, issues1 = validate_cpi(files["CPI数据"])
    all_results["CPI数据"] = (status1, issues1)
    
    # 文件2: PPI
    status2, issues2 = validate_ppi(files["PPI网络"])
    all_results["PPI网络"] = (status2, issues2)
    
    # 文件3: 基因（需要CPI和PPI交叉验证）
    status3, issues3 = validate_genes(files["铁衰老96基因"], files["CPI数据"], files["PPI网络"])
    all_results["铁衰老96基因"] = (status3, issues3)
    
    # 文件4: 化合物池（需要CPI泄漏检查）
    status4, issues4 = validate_compound_pool(files["化合物池"], files["CPI数据"])
    all_results["化合物池"] = (status4, issues4)
    
    # ============================================================
    # 汇总报告
    # ============================================================
    logger.info("\n\n" + "=" * 70)
    logger.info("=" * 70)
    logger.info("                    📋 数据完整性校验报告")
    logger.info("=" * 70)
    logger.info("=" * 70)
    
    for name, (status, issues) in all_results.items():
        icon = "✅" if status == "OK" else "❌"
        logger.info(f"\n{icon} 【{name}】: {status}")
        if issues:
            for i, issue in enumerate(issues, 1):
                logger.info(f"     {i}. {issue}")
        else:
            logger.info(f"     无异常")
    
    # 总体状态
    all_ok = all(s == "OK" for s, _ in all_results.values())
    logger.info(f"\n{'='*70}")
    if all_ok:
        logger.info("🎉 总体状态: ALL OK - 所有文件通过验证")
    else:
        logger.warning("⚠️  总体状态: 存在异常，请检查上述ISSUE")
    logger.info(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    main()