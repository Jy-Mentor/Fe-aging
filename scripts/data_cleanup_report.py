#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据清洗脚本 - 修复数据真实性报告中5个FAIL项和4个WARN项
"""
import pandas as pd
import sys
import os
from datetime import datetime
from rdkit import Chem
from rdkit import RDLogger

# 关闭RDKit噪声日志
RDLogger.DisableLog('rdApp.*')

# ============================================================
# 日志文件
# ============================================================
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cleanup_report.log')

def log(msg, to_stdout=True):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    if to_stdout:
        print(line, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

# ============================================================
# 工具函数：SMILES验证与标准化
# ============================================================
def validate_and_canonicalize_smiles(smiles_str):
    """
    尝试用RDKit标准化SMILES。
    返回 (canonical_smiles, is_valid) 或 (None, False)
    """
    if pd.isna(smiles_str) or str(smiles_str).strip() == '':
        return None, False
    s = str(smiles_str).strip()
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None, False
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical, True
    except Exception:
        return None, False


# ============================================================
# 任务1: CPI补充数据清洗
# ============================================================
def task1_clean_cpi_supplement():
    log("=" * 60)
    log("任务1: CPI补充数据清洗 - cpi_supplement_v25.csv")
    log("=" * 60)

    filepath = r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch\cpi_supplement_v25.csv"
    df = pd.read_csv(filepath)
    n_before = len(df)
    log(f"  原始记录数: {n_before}")

    # 1.1 检查SMILES列
    smiles_col = 'smiles'
    if smiles_col not in df.columns:
        log(f"  ERROR: 找不到SMILES列 '{smiles_col}'，可用列: {list(df.columns)}")
        return

    # 1.2 标准化所有SMILES
    valid_smiles = []
    invalid_count = 0
    invalid_rows = []
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        canonical, is_valid = validate_and_canonicalize_smiles(smi)
        if is_valid:
            valid_smiles.append(canonical)
        else:
            valid_smiles.append(None)
            invalid_count += 1
            invalid_rows.append(idx)
            log(f"    [INVALID SMILES] 行{idx+2}: gene={row.get('gene','?')}, smiles='{str(smi)[:60]}...'")

    df[smiles_col] = valid_smiles
    log(f"  无效SMILES: {invalid_count}/{n_before}")

    # 移除无效SMILES行
    df_valid = df[df[smiles_col].notna()].copy()
    n_after_smiles = len(df_valid)
    log(f"  移除无效SMILES后: {n_after_smiles} (移除{invalid_count}条)")

    # 1.3 检查重复的gene+SMILES对
    dup_cols = ['gene', smiles_col]
    dup_mask = df_valid.duplicated(subset=dup_cols, keep=False)
    n_dup = df_valid[dup_mask].shape[0]
    log(f"  重复gene+SMILES对: {n_dup}条")

    if n_dup > 0:
        log(f"  重复详情:")
        for idx, row in df_valid[dup_mask].iterrows():
            log(f"    行{idx+2}: gene={row['gene']}, smiles={row[smiles_col][:50]}...")

    # 去除重复（保留第一条）
    df_dedup = df_valid.drop_duplicates(subset=dup_cols, keep='first')
    n_after_dedup = len(df_dedup)
    n_removed_dup = n_after_smiles - n_after_dedup
    log(f"  去除重复后: {n_after_dedup} (移除{n_removed_dup}条重复)")

    # 保存
    output_path = filepath.replace('.csv', '_cleaned.csv')
    df_dedup.to_csv(output_path, index=False)
    log(f"  已保存: {output_path}")

    # 汇总
    total_removed = n_before - n_after_dedup
    log(f"  汇总: 原始{n_before} -> 最终{n_after_dedup} (共移除{total_removed}条: {invalid_count}无效SMILES + {n_removed_dup}重复)")
    log("")

    return {
        'task': 'CPI补充数据',
        'file': filepath,
        'output': output_path,
        'n_before': n_before,
        'n_after': n_after_dedup,
        'n_invalid_smiles': invalid_count,
        'n_duplicates': n_removed_dup,
        'n_total_removed': total_removed
    }


# ============================================================
# 任务2: BindingDB数据清洗
# ============================================================
def task2_clean_bindingdb():
    log("=" * 60)
    log("任务2: BindingDB数据清洗 - bindingdb_active_compounds.csv")
    log("=" * 60)

    filepath = r"d:\铁衰老 绝不重蹈覆辙\L4\results\bindingdb_active_compounds.csv"
    df = pd.read_csv(filepath, low_memory=False)
    n_before = len(df)
    log(f"  原始记录数: {n_before}")

    smiles_col = 'canonical_smiles'
    if smiles_col not in df.columns:
        log(f"  ERROR: 找不到SMILES列 '{smiles_col}'，可用列: {list(df.columns)}")
        return

    # 标准化所有SMILES
    valid_smiles = []
    invalid_count = 0
    invalid_summary = []
    for idx, row in df.iterrows():
        smi = row[smiles_col]
        canonical, is_valid = validate_and_canonicalize_smiles(smi)
        if is_valid:
            valid_smiles.append(canonical)
        else:
            valid_smiles.append(None)
            invalid_count += 1
            if invalid_count <= 20:  # 只打印前20条
                log(f"    [INVALID SMILES] 行{idx+2}: gene={row.get('gene','?')}, smiles='{str(smi)[:80]}...'")

    df[smiles_col] = valid_smiles
    log(f"  无效SMILES总数: {invalid_count}/{n_before}")

    # 移除无效SMILES行
    df_valid = df[df[smiles_col].notna()].copy()
    n_after = len(df_valid)
    n_removed = n_before - n_after
    log(f"  移除无效SMILES后: {n_after} (移除{n_removed}条)")

    # 保存
    output_path = filepath.replace('.csv', '_cleaned.csv')
    df_valid.to_csv(output_path, index=False)
    log(f"  已保存: {output_path}")

    log(f"  汇总: 原始{n_before} -> 最终{n_after} (共移除{n_removed}条无效SMILES)")
    log("")

    return {
        'task': 'BindingDB数据',
        'file': filepath,
        'output': output_path,
        'n_before': n_before,
        'n_after': n_after,
        'n_invalid_smiles': invalid_count,
        'n_duplicates': 0,
        'n_total_removed': n_removed
    }


# ============================================================
# 任务3: PPI网络去重
# ============================================================
def task3_dedup_ppi():
    log("=" * 60)
    log("任务3: PPI网络去重 - ppi_network_extended_significant_edges.csv")
    log("=" * 60)

    filepath = r"d:\铁衰老 绝不重蹈覆辙\L1\results\ppi_network_extended_significant_edges.csv"
    df = pd.read_csv(filepath)
    n_before = len(df)
    log(f"  原始记录数: {n_before}")

    # 检查列名
    log(f"  列名: {list(df.columns)}")

    # 对gene_a, gene_b排序，使无向边统一
    cols = ['gene_a', 'gene_b']
    df['sorted_pair'] = df.apply(
        lambda r: tuple(sorted([str(r['gene_a']), str(r['gene_b'])])), axis=1
    )

    # 检查重复
    n_dup = df.duplicated(subset='sorted_pair', keep=False).sum()
    log(f"  重复边(无向)总数: {n_dup}")

    # 按sorted_pair分组，保留combined_score最高的那条
    if 'combined_score' in df.columns:
        df_sorted = df.sort_values('combined_score', ascending=False)
        df_dedup = df_sorted.drop_duplicates(subset='sorted_pair', keep='first')
    else:
        df_dedup = df.drop_duplicates(subset='sorted_pair', keep='first')

    df_dedup = df_dedup.drop(columns=['sorted_pair'])
    n_after = len(df_dedup)
    n_removed = n_before - n_after
    log(f"  去重后: {n_after} (移除{n_removed}条重复边)")

    # 保存
    output_path = filepath.replace('.csv', '_dedup.csv')
    df_dedup.to_csv(output_path, index=False)
    log(f"  已保存: {output_path}")

    log(f"  汇总: 原始{n_before} -> 最终{n_after} (共移除{n_removed}条重复边)")
    log("")

    return {
        'task': 'PPI网络去重',
        'file': filepath,
        'output': output_path,
        'n_before': n_before,
        'n_after': n_after,
        'n_invalid_smiles': 0,
        'n_duplicates': n_removed,
        'n_total_removed': n_removed
    }


# ============================================================
# 任务4: TCM池与训练集SMILES重叠标记
# ============================================================
def task4_check_tcm_leakage():
    log("=" * 60)
    log("任务4: TCM池与训练集SMILES重叠检查")
    log("=" * 60)

    # 读取TCM池
    tcm_path = r"d:\铁衰老 绝不重蹈覆辙\L3\results\tcm_compound_pool_v21_Alevel.csv"
    df_tcm = pd.read_csv(tcm_path)
    n_tcm = len(df_tcm)
    log(f"  TCM池记录数: {n_tcm}")

    # 确定TCM中的SMILES列
    smiles_col_tcm = 'SMILES_std'
    if smiles_col_tcm not in df_tcm.columns:
        log(f"  ERROR: 找不到SMILES列 '{smiles_col_tcm}'，可用列: {list(df_tcm.columns)}")
        return

    # 收集训练集SMILES (从多个来源)
    training_smiles_set = set()
    training_sources = {}

    # 来源1: phenotype_ferroptosis_dataset.csv
    try:
        fp1 = r"d:\铁衰老 绝不重蹈覆辙\L4\results\phenotype_ferroptosis_dataset.csv"
        if os.path.exists(fp1):
            df1 = pd.read_csv(fp1)
            for smi in df1['canonical_smiles'].dropna():
                smi_canon, valid = validate_and_canonicalize_smiles(smi)
                if valid:
                    training_smiles_set.add(smi_canon)
                    training_sources[smi_canon] = 'phenotype_ferroptosis_dataset'
            log(f"  训练集来源1 (phenotype_ferroptosis_dataset): {len(df1)}条记录")
    except Exception as e:
        log(f"  WARNING: 读取phenotype_ferroptosis_dataset失败: {e}")

    # 来源2: chembl_active_compounds.csv
    try:
        fp2 = r"d:\铁衰老 绝不重蹈覆辙\L4\results\chembl_active_compounds.csv"
        if os.path.exists(fp2):
            df2 = pd.read_csv(fp2, low_memory=False)
            smi_col2 = 'canonical_smiles'
            for smi in df2[smi_col2].dropna():
                smi_canon, valid = validate_and_canonicalize_smiles(smi)
                if valid:
                    training_smiles_set.add(smi_canon)
                    if smi_canon not in training_sources:
                        training_sources[smi_canon] = 'chembl_active_compounds'
            log(f"  训练集来源2 (chembl_active_compounds): {len(df2)}条记录")
    except Exception as e:
        log(f"  WARNING: 读取chembl_active_compounds失败: {e}")

    # 来源3: bindingdb_active_compounds_cleaned.csv (如果存在) 或 原始文件
    try:
        fp3_cleaned = r"d:\铁衰老 绝不重蹈覆辙\L4\results\bindingdb_active_compounds_cleaned.csv"
        fp3 = fp3_cleaned if os.path.exists(fp3_cleaned) else r"d:\铁衰老 绝不重蹈覆辙\L4\results\bindingdb_active_compounds.csv"
        if os.path.exists(fp3):
            df3 = pd.read_csv(fp3, low_memory=False)
            smi_col3 = 'canonical_smiles'
            for smi in df3[smi_col3].dropna():
                smi_canon, valid = validate_and_canonicalize_smiles(smi)
                if valid:
                    training_smiles_set.add(smi_canon)
                    if smi_canon not in training_sources:
                        training_sources[smi_canon] = 'bindingdb_active_compounds'
            log(f"  训练集来源3 (bindingdb_active_compounds): {len(df3)}条记录")
    except Exception as e:
        log(f"  WARNING: 读取bindingdb_active_compounds失败: {e}")

    log(f"  训练集唯一SMILES总数: {len(training_smiles_set)}")

    # 标准化TCM池SMILES并比较
    tcm_canonical = []
    tcm_invalid = []
    overlap_indices = []
    overlap_info = []

    for idx, row in df_tcm.iterrows():
        smi = row[smiles_col_tcm]
        canonical, is_valid = validate_and_canonicalize_smiles(smi)
        tcm_canonical.append(canonical)
        if not is_valid:
            tcm_invalid.append(idx)
        elif canonical in training_smiles_set:
            overlap_indices.append(idx)
            overlap_info.append({
                'idx': idx,
                'MOL_ID': row.get('MOL_ID', '?'),
                'molecule_name': row.get('molecule_name', '?'),
                'SMILES_std': smi,
                'canonical_smiles': canonical,
                'training_source': training_sources.get(canonical, 'unknown')
            })

    n_overlap = len(overlap_indices)
    log(f"  TCM池中无效SMILES: {len(tcm_invalid)}")
    log(f"  TCM池与训练集重叠SMILES: {n_overlap}")

    if n_overlap > 0:
        log(f"  重叠详情:")
        for info in overlap_info:
            log(f"    MOL_ID={info['MOL_ID']}, name={info['molecule_name']}, "
                f"training_source={info['training_source']}")
            log(f"      SMILES: {info['SMILES_std'][:80]}")

    # 添加data_leakage_warning列
    df_tcm['data_leakage_warning'] = ''
    for idx in overlap_indices:
        df_tcm.at[idx, 'data_leakage_warning'] = '训练集重叠'
        # 同时添加来源信息
        smi = df_tcm.at[idx, smiles_col_tcm]
        canonical, _ = validate_and_canonicalize_smiles(smi)
        if canonical and canonical in training_sources:
            df_tcm.at[idx, 'data_leakage_warning'] = f'训练集重叠|来源:{training_sources[canonical]}'

    # 保存
    output_path = tcm_path.replace('.csv', '_leakage_checked.csv')
    df_tcm.to_csv(output_path, index=False)
    log(f"  已保存: {output_path}")

    log(f"  汇总: TCM池{n_tcm}条，其中{n_overlap}条与训练集SMILES重叠（已标记，未删除）")
    log("")

    return {
        'task': 'TCM池泄漏检查',
        'file': tcm_path,
        'output': output_path,
        'n_before': n_tcm,
        'n_after': n_tcm,
        'n_invalid_smiles': len(tcm_invalid),
        'n_duplicates': 0,
        'n_overlap': n_overlap,
        'n_total_removed': 0
    }


# ============================================================
# 主函数
# ============================================================
def main():
    log("=" * 60)
    log("铁衰老项目 - 数据清洗脚本启动")
    log(f"运行时间: {datetime.now().isoformat()}")
    log("=" * 60)

    results = []

    # 任务1: CPI补充数据
    try:
        r1 = task1_clean_cpi_supplement()
        if r1:
            results.append(r1)
    except Exception as e:
        log(f"ERROR: 任务1失败: {e}", to_stdout=True)
        import traceback
        log(traceback.format_exc(), to_stdout=True)

    # 任务2: BindingDB数据
    try:
        r2 = task2_clean_bindingdb()
        if r2:
            results.append(r2)
    except Exception as e:
        log(f"ERROR: 任务2失败: {e}", to_stdout=True)
        import traceback
        log(traceback.format_exc(), to_stdout=True)

    # 任务3: PPI网络去重
    try:
        r3 = task3_dedup_ppi()
        if r3:
            results.append(r3)
    except Exception as e:
        log(f"ERROR: 任务3失败: {e}", to_stdout=True)
        import traceback
        log(traceback.format_exc(), to_stdout=True)

    # 任务4: TCM池泄漏检查
    try:
        r4 = task4_check_tcm_leakage()
        if r4:
            results.append(r4)
    except Exception as e:
        log(f"ERROR: 任务4失败: {e}", to_stdout=True)
        import traceback
        log(traceback.format_exc(), to_stdout=True)

    # 输出综合报告
    log("")
    log("=" * 60)
    log("综合修复报告")
    log("=" * 60)
    log("")
    log(f"{'任务':<25} {'原始记录':>10} {'最终记录':>10} {'移除/标记':>10} {'详情':>40}")
    log("-" * 100)

    for r in results:
        detail = ""
        if r.get('n_invalid_smiles', 0) > 0:
            detail += f"无效SMILES:{r['n_invalid_smiles']} "
        if r.get('n_duplicates', 0) > 0:
            detail += f"重复:{r['n_duplicates']} "
        if r.get('n_overlap', 0) > 0:
            detail += f"训练集重叠(已标记):{r['n_overlap']} "
        log(f"{r['task']:<25} {r['n_before']:>10} {r['n_after']:>10} {r['n_total_removed']:>10} {detail:<40}")

    log("")
    log("输出文件:")
    for r in results:
        log(f"  {r['output']}")

    log("")
    log("数据清洗完成。")
    log(f"详细日志: {LOG_FILE}")

    return results


if __name__ == '__main__':
    main()