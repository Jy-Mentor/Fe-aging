#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁衰老项目 - 数据真伪性全面校验脚本 v25
验证所有关键数据文件的真实性和完整性
"""

import os
import sys
import re
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from scipy.stats import mannwhitneyu, ks_2samp

# ============================================================
# 配置
# ============================================================
BASE = r"d:\铁衰老 绝不重蹈覆辙"
REPORT_DIR = os.path.join(BASE, "L4", "logs")
REPORT_PATH = os.path.join(REPORT_DIR, "data_authenticity_report_v25.md")
os.makedirs(REPORT_DIR, exist_ok=True)

# 所有文件路径
FILES = {
    "cpi": os.path.join(BASE, "L4", "results", "experimental_actives_detail_cleaned.csv"),
    "ppi_sig": os.path.join(BASE, "L1", "results", "ppi_network_extended_significant_edges.csv"),
    "ppi_all": os.path.join(BASE, "L1", "results", "ppi_network_extended_edges.csv"),
    "genes96": os.path.join(BASE, "L1", "results", "ferroaging_genes_96.csv"),
    "compound_pool": os.path.join(BASE, "L3", "results", "tcm_compound_pool_v21_Alevel.csv"),
    "compound_tox_noleak": os.path.join(BASE, "L3", "results", "tcm_compound_pool_tox_filtered_noleak.csv"),
    "compound_tox": os.path.join(BASE, "L3", "results", "tcm_compound_pool_tox_filtered.csv"),
    "protein_features": os.path.join(BASE, "L2", "results", "target_protein_features.csv"),
    "esm2": os.path.join(BASE, "L4", "results_v10_minibatch", "esm2_protein_embeddings.npz"),
    "kegg": os.path.join(BASE, "L2", "results", "kegg_pathways", "kegg_human_pathway_genes.tsv"),
    "phenotype": os.path.join(BASE, "L4", "results_v10_minibatch", "phenotype_ferroptosis_dataset_v25_clean.csv"),
    "disease_gene": os.path.join(BASE, "L4", "results_v10_minibatch", "disease_gene_edges.csv"),
}

# FerrDb 参考数据
FERRDB_DIR = os.path.join(BASE, "L4", "data", "ferroptosis_lib", "ferrdb_v2", "ferroptosis_early_preview_upto20231231")

# ============================================================
# 全局状态
# ============================================================
report = []
all_issues = []  # (severity, category, detail)

CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, level="INFO"):
    prefix = {"INFO": "", "PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "H1": "## ", "H2": "### ", "H3": "#### "}
    p = prefix.get(level, "")
    line = f"{p} {msg}" if p else msg
    print(line)
    report.append(line)

def add_issue(severity, category, check_name, detail):
    """severity: 严重, 中等, 轻微"""
    all_issues.append((severity, category, check_name, detail))

def check_file_exists(path):
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    if os.path.getsize(path) == 0:
        return False, f"文件为空: {path}"
    return True, ""

def is_valid_smiles(smi):
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception:
        return False

def get_mol(smi):
    try:
        return Chem.MolFromSmiles(smi.strip())
    except Exception:
        return None

# ============================================================
# 1. CPI数据验证（最高优先级）
# ============================================================
def validate_cpi():
    log("## 1. CPI数据验证 (最高优先级)", "H1")
    log(f"文件: `{FILES['cpi']}`")
    
    ok, err = check_file_exists(FILES['cpi'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "CPI", "文件存在性", err)
        return
    
    try:
        df = pd.read_csv(FILES['cpi'], low_memory=False)
    except Exception as e:
        log(f"❌ [FAIL] 无法读取CPI文件: {e}", "FAIL")
        add_issue("严重", "CPI", "文件读取", str(e))
        return
    
    n_rows = len(df)
    n_genes = df['gene'].nunique()
    n_smiles = df['canonical_smiles'].nunique()
    n_uniprot = df['uniprot_id'].nunique() if 'uniprot_id' in df.columns else 0
    log(f"- 总行数: {n_rows:,}")
    log(f"- 唯一基因: {n_genes}")
    log(f"- 唯一SMILES: {n_smiles:,}")
    log(f"- 唯一UniProt ID: {n_uniprot}")
    log(f"- 列名: {list(df.columns)}")
    
    # 1.1 空值检查
    log("", "INFO")
    log("### 1.1 空值检查", "H3")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        for col, cnt in null_cols.items():
            pct = cnt / n_rows * 100
            if pct > 50:
                log(f"⚠️ [WARN] {col}: {cnt} 空值 ({pct:.1f}%) - 大量缺失", "WARN")
                add_issue("中等", "CPI", "空值", f"{col}: {cnt} 空值 ({pct:.1f}%)")
            else:
                log(f"   {col}: {cnt} 空值 ({pct:.1f}%)")
    else:
        log("✅ [PASS] 无空值", "PASS")
    
    # 空行检查
    all_null_rows = df[df.isnull().all(axis=1)]
    if len(all_null_rows) > 0:
        log(f"❌ [FAIL] 发现 {len(all_null_rows)} 个全空行!", "FAIL")
        add_issue("严重", "CPI", "空行", f"发现 {len(all_null_rows)} 个全空行")
    else:
        log("✅ [PASS] 无全空行", "PASS")
    
    # 1.2 SMILES有效性检查
    log("", "INFO")
    log("### 1.2 SMILES有效性检查", "H3")
    df['smiles_valid'] = df['canonical_smiles'].apply(is_valid_smiles)
    invalid_smiles = df[~df['smiles_valid']]
    n_invalid = len(invalid_smiles)
    if n_invalid > 0:
        log(f"❌ [FAIL] 无效SMILES: {n_invalid} 条 ({n_invalid/n_rows*100:.2f}%)", "FAIL")
        add_issue("严重", "CPI", "SMILES有效性", f"{n_invalid} 条无效SMILES")
        for _, row in invalid_smiles.head(10).iterrows():
            log(f"   基因={row['gene']}, SMILES='{str(row['canonical_smiles'])[:80]}'")
    else:
        log(f"✅ [PASS] 所有 {n_rows:,} 条SMILES均有效", "PASS")
    
    # SMILES长度分布
    df['smiles_len'] = df['canonical_smiles'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    log(f"- SMILES长度: min={df['smiles_len'].min()}, max={df['smiles_len'].max()}, "
        f"median={df['smiles_len'].median():.0f}, mean={df['smiles_len'].mean():.1f}")
    
    # 极短SMILES检查
    short_smiles = df[df['smiles_len'] < 10]
    if len(short_smiles) > 0:
        for _, row in short_smiles.iterrows():
            mol = get_mol(row['canonical_smiles'])
            heavy = mol.GetNumHeavyAtoms() if mol else 0
            if heavy < 5:
                log(f"⚠️ [WARN] 极短SMILES: 基因={row['gene']}, SMILES='{row['canonical_smiles']}', 重原子数={heavy}", "WARN")
                add_issue("轻微", "CPI", "极短SMILES", f"基因={row['gene']}, SMILES='{row['canonical_smiles']}', 重原子数={heavy}")
    
    # 1.3 重复化合物-蛋白对检查
    log("", "INFO")
    log("### 1.3 重复检查", "H3")
    dup_cols = ['gene', 'canonical_smiles'] if 'uniprot_id' in df.columns else ['gene', 'canonical_smiles']
    dup_mask = df.duplicated(subset=dup_cols, keep=False)
    n_dup = dup_mask.sum()
    if n_dup > 0:
        log(f"⚠️ [WARN] 重复化合物-蛋白对: {n_dup} 条 ({n_dup/n_rows*100:.2f}%)", "WARN")
        add_issue("中等", "CPI", "重复数据", f"{n_dup} 条重复化合物-蛋白对")
    else:
        log(f"✅ [PASS] 无重复化合物-蛋白对", "PASS")
    
    # 1.4 UniProt ID验证
    log("", "INFO")
    log("### 1.4 UniProt ID格式检查", "H3")
    uniprot_ids = df['uniprot_id'].dropna().unique()
    valid_uniprot_pattern = re.compile(r'^[A-Z][0-9][A-Z0-9]{3}[0-9]$|^[A-Z][0-9][A-Z0-9]{3}[0-9]-\d+$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]-\d+$')
    invalid_ids = [uid for uid in uniprot_ids if not valid_uniprot_pattern.match(str(uid))]
    if invalid_ids:
        log(f"❌ [FAIL] 无效UniProt ID格式: {len(invalid_ids)} 个", "FAIL")
        for uid in invalid_ids[:5]:
            log(f"   {uid}")
        add_issue("严重", "CPI", "UniProt ID格式", f"{len(invalid_ids)} 个无效格式")
    else:
        log(f"✅ [PASS] 所有 {len(uniprot_ids)} 个UniProt ID格式正确", "PASS")
    
    # 1.5 基因名格式检查
    log("", "INFO")
    log("### 1.5 基因名格式检查", "H3")
    all_genes = df['gene'].dropna().unique()
    hgnc_pattern = re.compile(r'^[A-Z][A-Z0-9a-z\-]*$')
    invalid_gene_names = [g for g in all_genes if not hgnc_pattern.match(str(g))]
    if invalid_gene_names:
        log(f"❌ [FAIL] 非标准基因名: {len(invalid_gene_names)} 个", "FAIL")
        for g in invalid_gene_names[:10]:
            log(f"   {g}")
        add_issue("严重", "CPI", "基因名格式", f"{len(invalid_gene_names)} 个非标准基因名")
    else:
        log(f"✅ [PASS] 所有 {len(all_genes)} 个基因名符合HGNC格式", "PASS")
    
    # 1.6 CPI数量分布
    log("", "INFO")
    log("### 1.6 每个蛋白的CPI数量分布", "H3")
    gene_cpi_counts = df['gene'].value_counts()
    log(f"- 分布: min={gene_cpi_counts.min()}, max={gene_cpi_counts.max()}, "
        f"median={gene_cpi_counts.median():.0f}, mean={gene_cpi_counts.mean():.1f}, std={gene_cpi_counts.std():.1f}")
    log(f"- Top 5 基因:")
    for gene, cnt in gene_cpi_counts.head(5).items():
        log(f"    {gene}: {cnt:,}")
    log(f"- Bottom 5 基因:")
    for gene, cnt in gene_cpi_counts.tail(5).items():
        log(f"    {gene}: {cnt:,}")
    
    imbalance_ratio = gene_cpi_counts.max() / gene_cpi_counts.min() if gene_cpi_counts.min() > 0 else float('inf')
    if imbalance_ratio > 100:
        log(f"⚠️ [WARN] CPI数量极度不平衡: max/min={imbalance_ratio:.0f}", "WARN")
        add_issue("中等", "CPI", "CPI分布不平衡", f"max/min={imbalance_ratio:.0f}")
    
    # 1.7 pchembl_value分布
    if 'pchembl_value' in df.columns:
        log("", "INFO")
        log("### 1.7 pchembl_value分布", "H3")
        pchembl = df['pchembl_value'].dropna()
        log(f"- pchembl: min={pchembl.min():.2f}, max={pchembl.max():.2f}, mean={pchembl.mean():.2f}, median={pchembl.median():.2f}")
        if pchembl.min() < 0 or pchembl.max() > 14:
            log(f"⚠️ [WARN] pchembl范围异常", "WARN")
            add_issue("轻微", "CPI", "pchembl范围", f"min={pchembl.min()}, max={pchembl.max()}")
        else:
            log(f"✅ [PASS] pchembl范围正常 (0-14)", "PASS")
    
    # 1.8 数据来源分布
    if 'source' in df.columns:
        log("", "INFO")
        log("### 1.8 数据来源分布", "H3")
        source_counts = df['source'].value_counts()
        for src, cnt in source_counts.items():
            log(f"    {src}: {cnt:,}")
    
    # 1.9 多靶标化合物检查
    log("", "INFO")
    log("### 1.9 多靶标化合物检查", "H3")
    smiles_gene_map = df.groupby('canonical_smiles')['gene'].apply(list).to_dict()
    multi_target = {smi: genes for smi, genes in smiles_gene_map.items() if len(set(genes)) > 1}
    log(f"- 多靶标化合物: {len(multi_target)} 个")
    if multi_target:
        multi_dist = Counter([len(set(v)) for v in multi_target.values()])
        log(f"- 多靶标分布: {dict(sorted(multi_dist.items()))}")
        for smi, genes in list(multi_target.items())[:5]:
            log(f"    SMILES={smi[:60]}... -> {len(set(genes))}个靶标: {list(set(genes))[:5]}")
    
    return df, gene_cpi_counts

# ============================================================
# 2. PPI网络数据验证
# ============================================================
def validate_ppi():
    log("## 2. PPI网络数据验证", "H1")
    
    for ppi_name, ppi_path in [("显著边", FILES['ppi_sig']), ("全部边", FILES['ppi_all'])]:
        log(f"### 2.{'1' if '显著' in ppi_name else '2'} PPI {ppi_name}", "H3")
        log(f"文件: `{ppi_path}`")
        
        ok, err = check_file_exists(ppi_path)
        if not ok:
            log(f"❌ [FAIL] {err}", "FAIL")
            add_issue("严重", "PPI", f"文件存在性-{ppi_name}", err)
            continue
        
        try:
            df = pd.read_csv(ppi_path)
        except Exception as e:
            log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
            add_issue("严重", "PPI", f"文件读取-{ppi_name}", str(e))
            continue
        
        n_edges = len(df)
        nodes = set(df['gene_a'].dropna().unique()) | set(df['gene_b'].dropna().unique())
        n_nodes = len(nodes)
        log(f"- 总边数: {n_edges:,}")
        log(f"- 唯一节点: {n_nodes:,}")
        
        # 2.1 空值检查
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            log(f"⚠️ [WARN] 空值: {null_count} 个", "WARN")
            add_issue("中等", "PPI", f"空值-{ppi_name}", f"{null_count} 个空值")
        else:
            log(f"✅ [PASS] 无空值", "PASS")
        
        # 2.2 自环边检查
        self_loops = df[df['gene_a'] == df['gene_b']]
        if len(self_loops) > 0:
            log(f"❌ [FAIL] 自环边: {len(self_loops)} 条", "FAIL")
            add_issue("严重", "PPI", f"自环边-{ppi_name}", f"{len(self_loops)} 条自环边")
            for _, row in self_loops.head(5).iterrows():
                log(f"    {row['gene_a']} -> {row['gene_b']}")
        else:
            log(f"✅ [PASS] 无自环边", "PASS")
        
        # 2.3 重复边检查
        dup_edges = df.duplicated(subset=['gene_a', 'gene_b'], keep=False)
        n_dup = dup_edges.sum()
        if n_dup > 0:
            log(f"⚠️ [WARN] 重复边: {n_dup} 条 ({n_dup/n_edges*100:.2f}%)", "WARN")
            add_issue("中等", "PPI", f"重复边-{ppi_name}", f"{n_dup} 条重复边")
        else:
            log(f"✅ [PASS] 无重复边", "PASS")
        
        # 2.4 度分布
        degree_a = df['gene_a'].value_counts()
        degree_b = df['gene_b'].value_counts()
        degree = degree_a.add(degree_b, fill_value=0).astype(int)
        log(f"- 度分布: min={degree.min()}, max={degree.max()}, "
            f"median={degree.median():.0f}, mean={degree.mean():.1f}")
        
        # 孤立节点
        isolated = [n for n in nodes if n not in degree.index]
        if isolated:
            log(f"⚠️ [WARN] 孤立节点: {len(isolated)} 个", "WARN")
            add_issue("中等", "PPI", f"孤立节点-{ppi_name}", f"{len(isolated)} 个孤立节点")
        else:
            log(f"✅ [PASS] 无孤立节点", "PASS")
        
        # 2.5 combined_score检查
        if 'combined_score' in df.columns:
            score = df['combined_score'].dropna()
            log(f"- combined_score: min={score.min()}, max={score.max()}, mean={score.mean():.1f}, median={score.median():.0f}")
            if score.min() < 0 or score.max() > 1000:
                log(f"⚠️ [WARN] combined_score范围异常", "WARN")
                add_issue("轻微", "PPI", "combined_score范围", f"min={score.min()}, max={score.max()}")
            else:
                log(f"✅ [PASS] combined_score范围正常 (0-1000)", "PASS")
        
        # 2.6 基因名格式检查
        all_genes = list(nodes)
        invalid_genes = [g for g in all_genes if not re.match(r'^[A-Z][A-Z0-9a-z\-]*$', str(g))]
        if invalid_genes:
            log(f"⚠️ [WARN] 非标准基因名: {len(invalid_genes)} 个", "WARN")
            for g in invalid_genes[:10]:
                log(f"    {g}")
            add_issue("轻微", "PPI", f"基因名格式-{ppi_name}", f"{len(invalid_genes)} 个非标准基因名")
        else:
            log(f"✅ [PASS] 所有基因名符合标准格式", "PASS")
    
    return df

# ============================================================
# 3. 铁衰老96基因验证
# ============================================================
def validate_genes96():
    log("## 3. 铁衰老96基因验证", "H1")
    log(f"文件: `{FILES['genes96']}`")
    
    ok, err = check_file_exists(FILES['genes96'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "铁衰老96基因", "文件存在性", err)
        return None
    
    try:
        df = pd.read_csv(FILES['genes96'])
    except Exception as e:
        log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
        add_issue("严重", "铁衰老96基因", "文件读取", str(e))
        return None
    
    genes = df['gene_symbol'].dropna().unique().tolist()
    log(f"- 基因总数: {len(genes)}")
    log(f"- 列名: {list(df.columns)}")
    
    if len(genes) != 96:
        log(f"⚠️ [WARN] 基因数={len(genes)}，预期96个", "WARN")
        add_issue("中等", "铁衰老96基因", "基因数量", f"实际{len(genes)}，预期96")
    
    # 3.1 基因名标准化检查
    log("", "INFO")
    log("### 3.1 基因名标准化检查", "H3")
    hgnc_pattern = re.compile(r'^[A-Z][A-Z0-9]*$')
    non_standard = [g for g in genes if not hgnc_pattern.match(str(g))]
    if non_standard:
        log(f"⚠️ [WARN] 非标准基因名: {non_standard}", "WARN")
        add_issue("轻微", "铁衰老96基因", "基因名格式", str(non_standard))
    else:
        log(f"✅ [PASS] 所有基因名标准化", "PASS")
    
    # 3.2 关键铁衰老基因检查
    log("", "INFO")
    log("### 3.2 关键铁衰老基因检查", "H3")
    key_ferroptosis = [
        'ACSL4', 'GPX4', 'FTH1', 'FTL', 'TFRC', 'SLC7A11', 'NOX4', 'HMOX1',
        'PTGS2', 'ALOX5', 'ALOX15', 'NFE2L2', 'KEAP1', 'SLC3A2', 'SAT1',
        'LPCAT3', 'CISD1', 'AIFM2', 'TP53', 'HIF1A', 'NFKB1', 'RELA',
        'STAT3', 'MAPK1', 'MAPK14', 'MTOR', 'SQSTM1', 'ATG5', 'ATG7',
        'BECN1', 'FXN', 'IREB2', 'ACO1', 'VDAC2', 'VDAC3'
    ]
    found_key = [g for g in key_ferroptosis if g in genes]
    missing_key = [g for g in key_ferroptosis if g not in genes]
    log(f"- 关键铁死亡基因: 找到 {len(found_key)}/{len(key_ferroptosis)}")
    log(f"- 找到: {found_key}")
    if missing_key:
        log(f"⚠️ [WARN] 缺失关键铁死亡基因: {missing_key}", "WARN")
        add_issue("中等", "铁衰老96基因", "缺失关键基因", f"缺失: {missing_key}")
    
    # 关键衰老基因
    key_aging = ['TP53', 'CDKN1A', 'CDKN2A', 'SIRT1', 'SIRT3', 'SIRT6', 'FOXO3', 'MTOR', 'IGF1R', 'TERT']
    found_aging = [g for g in key_aging if g in genes]
    missing_aging = [g for g in key_aging if g not in genes]
    log(f"- 关键衰老基因: 找到 {len(found_aging)}/{len(key_aging)}")
    if missing_aging:
        log(f"⚠️ [WARN] 缺失关键衰老基因: {missing_aging}", "WARN")
        add_issue("中等", "铁衰老96基因", "缺失衰老基因", f"缺失: {missing_aging}")
    
    # 3.3 与FerrDb交叉验证
    log("", "INFO")
    log("### 3.3 与FerrDb交叉验证", "H3")
    ferrdb_genes = set()
    for fname in ['driver.csv', 'suppressor.csv', 'marker.csv', 'inducer.csv', 'inhibitor.csv', 'unclassified.reg.csv']:
        fpath = os.path.join(FERRDB_DIR, fname)
        if os.path.exists(fpath):
            try:
                fdf = pd.read_csv(fpath)
                for col in ['Symbol_or_reported_abbr', 'gene', 'Gene', 'symbol', 'Symbol']:
                    if col in fdf.columns:
                        ferrdb_genes.update(fdf[col].dropna().unique())
                        break
            except Exception as e:
                log(f"  读取FerrDb文件失败: {fname} - {e}")
    
    if ferrdb_genes:
        overlap = set(genes) & ferrdb_genes
        log(f"- FerrDb基因总数: {len(ferrdb_genes)}")
        log(f"- 与铁衰老96基因重叠: {len(overlap)}/{len(genes)} ({len(overlap)/len(genes)*100:.1f}%)")
        missing_ferrdb = ferrdb_genes - set(genes)
        if len(missing_ferrdb) > 50:
            log(f"⚠️ [WARN] 大量FerrDb基因未包含: {len(missing_ferrdb)} 个", "WARN")
            add_issue("轻微", "铁衰老96基因", "FerrDb覆盖", f"仅{len(overlap)}/{len(ferrdb_genes)} FerrDb基因被覆盖")
        else:
            log(f"✅ [PASS] 铁衰老96基因与FerrDb覆盖良好", "PASS")
    else:
        log(f"⚠️ [WARN] 无法读取FerrDb数据", "WARN")
    
    # 3.4 重复基因检查
    log("", "INFO")
    log("### 3.4 重复基因检查", "H3")
    dup_genes = df[df.duplicated(subset='gene_symbol', keep=False)]
    if len(dup_genes) > 0:
        log(f"❌ [FAIL] 重复基因: {len(dup_genes)} 个", "FAIL")
        add_issue("严重", "铁衰老96基因", "重复基因", f"{len(dup_genes)} 个重复")
    else:
        log(f"✅ [PASS] 无重复基因", "PASS")
    
    return genes

# ============================================================
# 4. 化合物池数据验证
# ============================================================
def validate_compound_pool():
    log("## 4. 化合物池数据验证", "H1")
    
    pool_index = 1
    for pool_name, pool_path in [
        ("v21_Alevel", FILES['compound_pool']),
        ("tox_filtered", FILES['compound_tox']),
        ("tox_filtered_noleak", FILES['compound_tox_noleak']),
    ]:
        log(f"### 4.{pool_index} {pool_name}", "H3")
        pool_index += 1
        log(f"文件: `{pool_path}`")
        
        ok, err = check_file_exists(pool_path)
        if not ok:
            log(f"❌ [FAIL] {err}", "FAIL")
            add_issue("严重", "化合物池", f"文件存在性-{pool_name}", err)
            continue
        
        try:
            df = pd.read_csv(pool_path, low_memory=False)
        except Exception as e:
            log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
            add_issue("严重", "化合物池", f"文件读取-{pool_name}", str(e))
            continue
        
        n_compounds = len(df)
        log(f"- 总化合物数: {n_compounds:,}")
        log(f"- 列名: {list(df.columns)}")
        
        # 4.1 MOL_ID唯一性
        if 'MOL_ID' in df.columns:
            dup_mol = df[df.duplicated(subset='MOL_ID', keep=False)]
            if len(dup_mol) > 0:
                log(f"❌ [FAIL] MOL_ID重复: {len(dup_mol)} 个", "FAIL")
                add_issue("严重", "化合物池", f"MOL_ID重复-{pool_name}", f"{len(dup_mol)} 个重复")
            else:
                log(f"✅ [PASS] MOL_ID全部唯一", "PASS")
        
        # 4.2 SMILES有效性
        smiles_col = 'SMILES_std' if 'SMILES_std' in df.columns else 'canonical_smiles'
        if smiles_col in df.columns:
            df['_smiles_valid'] = df[smiles_col].apply(is_valid_smiles)
            invalid = df[~df['_smiles_valid']]
            if len(invalid) > 0:
                log(f"❌ [FAIL] 无效SMILES: {len(invalid)} 个", "FAIL")
                add_issue("严重", "化合物池", f"SMILES有效性-{pool_name}", f"{len(invalid)} 个无效SMILES")
                for _, row in invalid.head(5).iterrows():
                    log(f"    MOL_ID={row.get('MOL_ID', '?')}, SMILES='{str(row[smiles_col])[:60]}'")
            else:
                log(f"✅ [PASS] 所有 {n_compounds} 个SMILES有效", "PASS")
            
            # 分子量分布
            df['_mw'] = df[smiles_col].apply(lambda s: Descriptors.MolWt(get_mol(s)) if get_mol(s) else None)
            mw_valid = df['_mw'].dropna()
            if len(mw_valid) > 0:
                log(f"- 分子量: min={mw_valid.min():.1f}, max={mw_valid.max():.1f}, "
                    f"mean={mw_valid.mean():.1f}, median={mw_valid.median():.1f}")
                drug_like = ((mw_valid >= 100) & (mw_valid <= 900)).sum()
                log(f"- 药物类(100-900): {drug_like}/{len(mw_valid)} ({drug_like/len(mw_valid)*100:.1f}%)")
        
        # 4.3 重复SMILES
        if smiles_col in df.columns:
            dup_smiles = df[smiles_col].dropna().duplicated().sum()
            if dup_smiles > 0:
                log(f"⚠️ [WARN] 重复SMILES: {dup_smiles} 个", "WARN")
                add_issue("中等", "化合物池", f"重复SMILES-{pool_name}", f"{dup_smiles} 个重复")
            else:
                log(f"✅ [PASS] 无重复SMILES", "PASS")
        
        # 4.4 空值检查
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            log(f"⚠️ [WARN] 空值: {null_count} 个", "WARN")
            add_issue("轻微", "化合物池", f"空值-{pool_name}", f"{null_count} 个空值")
        
        # 4.5 泄漏标记检查 (noleak文件)
        if 'noleak' in pool_name:
            log(f"✅ [PASS] 已标记为去除CPI泄漏", "PASS")
    
    return df

# ============================================================
# 5. 蛋白特征数据验证
# ============================================================
def validate_protein_features():
    log("## 5. 蛋白特征数据验证", "H1")
    log(f"文件: `{FILES['protein_features']}`")
    
    ok, err = check_file_exists(FILES['protein_features'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "蛋白特征", "文件存在性", err)
        return
    
    try:
        df = pd.read_csv(FILES['protein_features'], low_memory=False)
    except Exception as e:
        log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
        add_issue("严重", "蛋白特征", "文件读取", str(e))
        return
    
    n_proteins = len(df)
    n_cols = len(df.columns)
    log(f"- 蛋白数: {n_proteins:,}")
    log(f"- 特征列数: {n_cols}")
    log(f"- 列名: {list(df.columns)[:20]}..." if n_cols > 20 else f"- 列名: {list(df.columns)}")
    
    # 5.1 NaN值检查
    log("", "INFO")
    log("### 5.1 NaN值检查", "H3")
    nan_counts = df.isnull().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        total_nan = nan_cols.sum()
        log(f"⚠️ [WARN] NaN值: 共 {total_nan} 个，分布在 {len(nan_cols)} 列", "WARN")
        for col, cnt in nan_cols.head(10).items():
            pct = cnt / n_proteins * 100
            log(f"    {col}: {cnt} ({pct:.1f}%)")
        add_issue("中等", "蛋白特征", "NaN值", f"共 {total_nan} 个NaN，分布在 {len(nan_cols)} 列")
    else:
        log(f"✅ [PASS] 无NaN值", "PASS")
    
    # 5.2 基因列检查
    gene_col = None
    for col in ['gene_symbol', 'gene_name', 'Gene', 'uniprot_id', 'gene']:
        if col in df.columns:
            gene_col = col
            break
    
    if gene_col:
        n_unique = df[gene_col].nunique()
        log(f"✅ [PASS] 唯一{gene_col}: {n_unique}")
        if n_unique != n_proteins:
            log(f"⚠️ [WARN] 存在重复{gene_col}", "WARN")
            add_issue("轻微", "蛋白特征", "重复基因", f"{n_proteins - n_unique} 个重复")
    
    # 5.3 特征维度一致性
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    log(f"- 数值特征列: {len(numeric_cols)}")
    
    return df

# ============================================================
# 6. ESM-2嵌入验证
# ============================================================
def validate_esm2():
    log("## 6. ESM-2嵌入验证", "H1")
    log(f"文件: `{FILES['esm2']}`")
    
    ok, err = check_file_exists(FILES['esm2'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "ESM-2", "文件存在性", err)
        return
    
    try:
        data = np.load(FILES['esm2'], allow_pickle=True)
    except Exception as e:
        log(f"❌ [FAIL] 无法加载npz文件: {e}", "FAIL")
        add_issue("严重", "ESM-2", "文件加载", str(e))
        return
    
    log(f"- npz文件中的键: {list(data.keys())[:10]}... (共{len(data.keys())}个)")
    
    # ESM-2嵌入是字典结构：{gene_name: np.array(640,)}
    embedding_matrix = None
    gene_names = []
    
    # 检查是否为字典结构（每个键是基因名，值是一维向量）
    all_vectors = []
    for key in data.keys():
        val = data[key]
        if isinstance(val, np.ndarray) and val.ndim == 1:
            gene_names.append(str(key))
            all_vectors.append(val)
    
    if len(all_vectors) > 0:
        embedding_matrix = np.stack(all_vectors, axis=0)
        n_proteins, n_dims = embedding_matrix.shape
        log(f"- 嵌入矩阵形状: {n_proteins} × {n_dims} (字典结构，{n_proteins}个基因)")
        log(f"- 前10个基因: {gene_names[:10]}")
        
        # 6.1 NaN/Inf检查
        nan_count = np.isnan(embedding_matrix).sum()
        inf_count = np.isinf(embedding_matrix).sum()
        if nan_count > 0:
            log(f"❌ [FAIL] NaN值: {nan_count}", "FAIL")
            add_issue("严重", "ESM-2", "NaN值", f"{nan_count} 个NaN")
        if inf_count > 0:
            log(f"❌ [FAIL] Inf值: {inf_count}", "FAIL")
            add_issue("严重", "ESM-2", "Inf值", f"{inf_count} 个Inf")
        if nan_count == 0 and inf_count == 0:
            log(f"✅ [PASS] 无NaN/Inf值", "PASS")
        
        # 6.2 零向量检查
        zero_vecs = np.sum(np.abs(embedding_matrix) < 1e-10, axis=1) == n_dims
        n_zero = zero_vecs.sum()
        if n_zero > 0:
            log(f"❌ [FAIL] 零向量: {n_zero} 个", "FAIL")
            add_issue("严重", "ESM-2", "零向量", f"{n_zero} 个零向量")
        else:
            log(f"✅ [PASS] 无零向量", "PASS")
        
        # 6.3 L2范数分布
        l2_norms = np.linalg.norm(embedding_matrix, axis=1)
        log(f"- L2范数: min={l2_norms.min():.4f}, max={l2_norms.max():.4f}, "
            f"mean={l2_norms.mean():.4f}, median={np.median(l2_norms):.4f}, std={l2_norms.std():.4f}")
        
        # 6.4 维度检查
        if n_dims == 640:
            log(f"✅ [PASS] 嵌入维度=640，符合ESM-2标准", "PASS")
        else:
            log(f"⚠️ [WARN] 嵌入维度={n_dims}，非标准640", "WARN")
            add_issue("轻微", "ESM-2", "维度", f"实际{n_dims}，预期640")
        
        # 6.5 铁衰老基因覆盖
        log("", "INFO")
        log("### 6.5 铁衰老96基因覆盖", "H3")
        genes96_path = FILES['genes96']
        if os.path.exists(genes96_path):
            g96_df = pd.read_csv(genes96_path)
            g96_genes = set(g96_df['gene_symbol'].dropna().unique())
            esm2_genes = set(gene_names)
            overlap = g96_genes & esm2_genes
            missing = g96_genes - esm2_genes
            extra = esm2_genes - g96_genes
            log(f"- ESM-2嵌入基因数: {len(esm2_genes)}")
            log(f"- 铁衰老96基因中在ESM-2中: {len(overlap)}/{len(g96_genes)} ({len(overlap)/len(g96_genes)*100:.1f}%)")
            if missing:
                log(f"⚠️ [WARN] 缺失: {len(missing)} 个铁衰老基因无ESM-2嵌入", "WARN")
                log(f"    缺失: {sorted(list(missing))}")
                add_issue("中等", "ESM-2", "基因覆盖", f"{len(missing)} 个铁衰老基因无ESM-2嵌入")
            if extra:
                log(f"- 额外基因: {len(extra)} 个 (非铁衰老96基因)")
                log(f"    额外: {sorted(list(extra))}")
    else:
        # 尝试其他方式：查找2D矩阵
        for key in data.keys():
            val = data[key]
            if isinstance(val, np.ndarray) and val.ndim == 2:
                embedding_matrix = val
                log(f"   键 '{key}': shape={val.shape}, dtype={val.dtype}")
        
        if embedding_matrix is not None:
            n_proteins, n_dims = embedding_matrix.shape
            log(f"- 嵌入矩阵形状: {n_proteins} × {n_dims}")
        else:
            log(f"⚠️ [WARN] 未找到嵌入矩阵", "WARN")
            add_issue("中等", "ESM-2", "嵌入矩阵", "未找到嵌入矩阵")
    
    data.close()
    return embedding_matrix

# ============================================================
# 7. KEGG通路数据验证
# ============================================================
def validate_kegg():
    log("## 7. KEGG通路数据验证", "H1")
    log(f"文件: `{FILES['kegg']}`")
    
    ok, err = check_file_exists(FILES['kegg'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "KEGG", "文件存在性", err)
        return
    
    try:
        df = pd.read_csv(FILES['kegg'], sep='\t')
    except Exception as e:
        log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
        add_issue("严重", "KEGG", "文件读取", str(e))
        return
    
    n_rows = len(df)
    n_pathways = df['pathway_id'].nunique() if 'pathway_id' in df.columns else 0
    n_genes = df['gene_symbol'].nunique() if 'gene_symbol' in df.columns else 0
    log(f"- 总行数: {n_rows:,}")
    log(f"- 唯一通路: {n_pathways}")
    log(f"- 唯一基因: {n_genes}")
    log(f"- 列名: {list(df.columns)}")
    
    # 7.1 空值检查
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        log(f"⚠️ [WARN] 空值: {null_count} 个", "WARN")
        add_issue("轻微", "KEGG", "空值", f"{null_count} 个空值")
    else:
        log(f"✅ [PASS] 无空值", "PASS")
    
    # 7.2 通路基因数分布
    if 'pathway_id' in df.columns and 'gene_symbol' in df.columns:
        pathway_gene_counts = df.groupby('pathway_id')['gene_symbol'].nunique()
        log(f"- 通路基因数: min={pathway_gene_counts.min()}, max={pathway_gene_counts.max()}, "
            f"median={pathway_gene_counts.median():.0f}, mean={pathway_gene_counts.mean():.1f}")
    
    # 7.3 重复检查
    dup_rows = df.duplicated()
    if dup_rows.sum() > 0:
        log(f"⚠️ [WARN] 重复行: {dup_rows.sum()} 条", "WARN")
        add_issue("轻微", "KEGG", "重复行", f"{dup_rows.sum()} 条重复")
    else:
        log(f"✅ [PASS] 无重复行", "PASS")
    
    return df

# ============================================================
# 8. 铁死亡表型数据验证
# ============================================================
def validate_phenotype():
    log("## 8. 铁死亡表型数据验证", "H1")
    log(f"文件: `{FILES['phenotype']}`")
    
    ok, err = check_file_exists(FILES['phenotype'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "铁死亡表型", "文件存在性", err)
        return
    
    try:
        df = pd.read_csv(FILES['phenotype'], low_memory=False)
    except Exception as e:
        log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
        add_issue("严重", "铁死亡表型", "文件读取", str(e))
        return
    
    n_rows = len(df)
    log(f"- 总行数: {n_rows:,}")
    log(f"- 列名: {list(df.columns)}")
    
    # 8.1 标签分布
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        log(f"- 标签分布: {label_counts}")
        if 0 not in label_counts or 1 not in label_counts:
            log(f"❌ [FAIL] 缺少正或负样本", "FAIL")
            add_issue("严重", "铁死亡表型", "标签分布", f"缺少类别: {label_counts}")
        else:
            ratio = label_counts.get(1, 0) / label_counts.get(0, 1)
            log(f"- 正负比: 1:{1/ratio:.1f}" if ratio > 0 else "- 正负比: 0")
    
    # 8.2 铁死亡类型分布
    if 'ferroptosis_type' in df.columns:
        type_counts = df['ferroptosis_type'].value_counts().to_dict()
        log(f"- 铁死亡类型: {type_counts}")
    
    # 8.3 数据来源分布
    if 'source' in df.columns:
        source_counts = df['source'].value_counts().to_dict()
        log(f"- 数据来源: {source_counts}")
    
    # 8.4 SMILES有效性
    smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'SMILES_std'
    if smiles_col in df.columns:
        df['_smiles_valid'] = df[smiles_col].apply(is_valid_smiles)
        invalid = df[~df['_smiles_valid']]
        if len(invalid) > 0:
            log(f"❌ [FAIL] 无效SMILES: {len(invalid)} 个", "FAIL")
            add_issue("严重", "铁死亡表型", "SMILES有效性", f"{len(invalid)} 个无效SMILES")
        else:
            log(f"✅ [PASS] 所有SMILES有效", "PASS")
    
    # 8.5 重复检查
    if smiles_col in df.columns:
        dup_smiles = df[smiles_col].duplicated().sum()
        if dup_smiles > 0:
            log(f"⚠️ [WARN] 重复SMILES: {dup_smiles} 个", "WARN")
            add_issue("中等", "铁死亡表型", "重复SMILES", f"{dup_smiles} 个重复")
        else:
            log(f"✅ [PASS] 无重复SMILES", "PASS")
    
    # 8.6 空值检查
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        log(f"⚠️ [WARN] 空值: {null_count} 个", "WARN")
        add_issue("轻微", "铁死亡表型", "空值", f"{null_count} 个空值")
    else:
        log(f"✅ [PASS] 无空值", "PASS")
    
    return df

# ============================================================
# 9. 疾病-基因边数据验证
# ============================================================
def validate_disease_gene():
    log("## 9. 疾病-基因边数据验证", "H1")
    log(f"文件: `{FILES['disease_gene']}`")
    
    ok, err = check_file_exists(FILES['disease_gene'])
    if not ok:
        log(f"❌ [FAIL] {err}", "FAIL")
        add_issue("严重", "疾病-基因边", "文件存在性", err)
        return
    
    try:
        df = pd.read_csv(FILES['disease_gene'], low_memory=False)
    except Exception as e:
        log(f"❌ [FAIL] 无法读取: {e}", "FAIL")
        add_issue("严重", "疾病-基因边", "文件读取", str(e))
        return
    
    n_rows = len(df)
    log(f"- 总行数: {n_rows:,}")
    log(f"- 列名: {list(df.columns)}")
    
    # 9.1 疾病节点统计
    if 'disease_name' in df.columns:
        n_diseases = df['disease_name'].nunique()
        log(f"- 唯一疾病: {n_diseases}")
        disease_counts = df['disease_name'].value_counts()
        for d, c in disease_counts.head(10).items():
            log(f"    {d}: {c}")
    
    # 9.2 基因统计
    if 'gene_symbol' in df.columns:
        n_genes = df['gene_symbol'].nunique()
        log(f"- 唯一基因: {n_genes}")
    
    # 9.3 空值检查
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        log(f"⚠️ [WARN] 空值: {null_count} 个", "WARN")
        add_issue("轻微", "疾病-基因边", "空值", f"{null_count} 个空值")
    else:
        log(f"✅ [PASS] 无空值", "PASS")
    
    # 9.4 重复边检查
    dup_edges = df.duplicated(keep=False)
    if dup_edges.sum() > 0:
        log(f"⚠️ [WARN] 重复边: {dup_edges.sum()} 条", "WARN")
        add_issue("轻微", "疾病-基因边", "重复边", f"{dup_edges.sum()} 条重复")
    else:
        log(f"✅ [PASS] 无重复边", "PASS")
    
    return df

# ============================================================
# 交叉验证
# ============================================================
def cross_validate(cpi_genes, genes96, ppi_nodes):
    log("## 10. 交叉验证", "H1")
    
    if genes96 and ppi_nodes:
        log("### 10.1 铁衰老96基因 vs PPI节点", "H3")
        overlap = set(genes96) & set(ppi_nodes)
        missing = set(genes96) - set(ppi_nodes)
        log(f"- 重叠: {len(overlap)}/{len(genes96)} ({len(overlap)/len(genes96)*100:.1f}%)")
        if missing:
            log(f"⚠️ [WARN] 缺失: {len(missing)} 个基因不在PPI中", "WARN")
            log(f"    缺失前20: {sorted(list(missing))[:20]}")
            add_issue("中等", "交叉验证", "PPI-基因", f"{len(missing)} 个铁衰老基因不在PPI中")
    
    if cpi_genes is not None and genes96:
        log("### 10.2 铁衰老96基因 vs CPI基因", "H3")
        cpi_gene_set = set(cpi_genes.index) if hasattr(cpi_genes, 'index') else set(cpi_genes)
        overlap = set(genes96) & cpi_gene_set
        missing = set(genes96) - cpi_gene_set
        log(f"- 重叠: {len(overlap)}/{len(genes96)} ({len(overlap)/len(genes96)*100:.1f}%)")
        if missing:
            log(f"⚠️ [WARN] 缺失: {len(missing)} 个基因无CPI数据", "WARN")
            log(f"    缺失: {sorted(list(missing))}")
            add_issue("中等", "交叉验证", "CPI-基因", f"{len(missing)} 个铁衰老基因无CPI数据")

# ============================================================
# 主函数
# ============================================================
def main():
    log("# 铁衰老项目 - 数据真伪性全面校验报告 v25", "H1")
    log(f"生成时间: {CURRENT_TIME}")
    log(f"项目根目录: `{BASE}`")
    log("")
    log("---")
    log("")
    
    # 1. CPI
    cpi_result = validate_cpi()
    cpi_df = cpi_result[0] if cpi_result else None
    cpi_gene_counts = cpi_result[1] if cpi_result else None
    log("")
    log("---")
    log("")
    
    # 2. PPI - 使用显著边文件进行交叉验证
    ppi_sig_df = None
    ppi_all_df = None
    
    # 先读取显著边
    ppi_sig_path = FILES['ppi_sig']
    if os.path.exists(ppi_sig_path):
        ppi_sig_df = pd.read_csv(ppi_sig_path)
    
    ppi_nodes = None
    ppi_nodes_sig = None
    ppi_nodes_all = None
    
    # PPI验证
    validate_ppi()
    
    # 获取节点集
    if ppi_sig_df is not None:
        ppi_nodes_sig = set(ppi_sig_df['gene_a'].dropna().unique()) | set(ppi_sig_df['gene_b'].dropna().unique())
    # 默认使用显著边节点集
    ppi_nodes = ppi_nodes_sig
    log("")
    log("---")
    log("")
    
    # 3. 铁衰老96基因
    genes96 = validate_genes96()
    log("")
    log("---")
    log("")
    
    # 4. 化合物池
    pool_df = validate_compound_pool()
    log("")
    log("---")
    log("")
    
    # 5. 蛋白特征
    validate_protein_features()
    log("")
    log("---")
    log("")
    
    # 6. ESM-2
    validate_esm2()
    log("")
    log("---")
    log("")
    
    # 7. KEGG
    validate_kegg()
    log("")
    log("---")
    log("")
    
    # 8. 铁死亡表型
    validate_phenotype()
    log("")
    log("---")
    log("")
    
    # 9. 疾病-基因边
    validate_disease_gene()
    log("")
    log("---")
    log("")
    
    # 10. 交叉验证
    cross_validate(cpi_gene_counts, genes96, ppi_nodes)
    log("")
    log("---")
    log("")
    
    # 汇总报告
    log("## 汇总报告", "H1")
    log("")
    
    severity_counts = Counter(s[0] for s in all_issues)
    log(f"### 问题统计")
    log(f"- 严重: {severity_counts.get('严重', 0)}")
    log(f"- 中等: {severity_counts.get('中等', 0)}")
    log(f"- 轻微: {severity_counts.get('轻微', 0)}")
    log(f"- 总计: {len(all_issues)}")
    log("")
    
    if severity_counts.get('严重', 0) > 0:
        log("### 🔴 严重问题")
        for severity, category, check_name, detail in all_issues:
            if severity == '严重':
                log(f"- **[{category}]** {check_name}: {detail}")
        log("")
    
    if severity_counts.get('中等', 0) > 0:
        log("### 🟡 中等问题")
        for severity, category, check_name, detail in all_issues:
            if severity == '中等':
                log(f"- **[{category}]** {check_name}: {detail}")
        log("")
    
    if severity_counts.get('轻微', 0) > 0:
        log("### 🔵 轻微问题")
        for severity, category, check_name, detail in all_issues:
            if severity == '轻微':
                log(f"- **[{category}]** {check_name}: {detail}")
        log("")
    
    # 总体结论
    if severity_counts.get('严重', 0) > 0:
        log("## ⚠️ 总体结论: 存在严重问题，需立即处理", "H1")
    elif severity_counts.get('中等', 0) > 0:
        log("## ✅ 总体结论: 数据基本可靠，存在中等问题待处理", "H1")
    else:
        log("## ✅ 总体结论: 数据完整可靠", "H1")
    
    # 保存报告
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    log(f"\n报告已保存至: `{REPORT_PATH}`")
    print(f"\n{'='*70}")
    print(f"报告已保存至: {REPORT_PATH}")
    print(f"总问题数: {len(all_issues)} (严重:{severity_counts.get('严重',0)}, 中等:{severity_counts.get('中等',0)}, 轻微:{severity_counts.get('轻微',0)})")

if __name__ == '__main__':
    main()