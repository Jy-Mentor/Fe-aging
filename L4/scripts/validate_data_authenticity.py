#!/usr/bin/env python3
"""
铁衰老项目 - 数据真实性全面校验脚本
验证所有关键数据文件的真实性、完整性和一致性
"""

import os
import sys
import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── RDKit: 用于 SMILES 校验 ──
try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available, SMILES validation will be skipped")

# ============================================================
# 配置
# ============================================================
BASE = r"d:\铁衰老 绝不重蹈覆辙"
REPORT_PATH = os.path.join(BASE, "L4", "results_v10_minibatch", "data_authenticity_report_v25.txt")

# 所有待校验文件路径
FILES = {
    "cpi": os.path.join(BASE, "L4", "results", "experimental_actives_detail_cleaned.csv"),
    "ppi_sig": os.path.join(BASE, "L1", "results", "ppi_network_extended_significant_edges.csv"),
    "ppi_nodes": os.path.join(BASE, "L1", "results", "ppi_network_extended_nodes.csv"),
    "genes96": os.path.join(BASE, "L1", "results", "ferroaging_genes_96.csv"),
    "esm2": os.path.join(BASE, "L4", "results_v10_minibatch", "esm2_protein_embeddings.npz"),
    "kegg": os.path.join(BASE, "L2", "results", "kegg_pathways", "kegg_human_pathway_genes.tsv"),
    "phenotype": os.path.join(BASE, "L4", "results_v10_minibatch", "phenotype_ferroptosis_dataset_v25_clean.csv"),
    "disease_gene": os.path.join(BASE, "L4", "results_v10_minibatch", "disease_gene_edges.csv"),
    "tcm_pool": os.path.join(BASE, "L3", "results", "tcm_compound_pool_v21_Alevel.csv"),
    "cpi_supplement": os.path.join(BASE, "L4", "results_v10_minibatch", "cpi_supplement_v25.csv"),
    "bindingdb": os.path.join(BASE, "L4", "results", "bindingdb_active_compounds.csv"),
    "drugbank": os.path.join(BASE, "L4", "results", "drugbank_active_compounds.csv"),
}

# ============================================================
# 全局状态
# ============================================================
report_lines = []
results_summary = []  # [(check_name, status, details)]
CURRENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, level="INFO"):
    """统一日志输出到控制台和报告"""
    prefix_map = {
        "INFO": "  ",
        "PASS": "  [PASS] ",
        "WARN": "  [WARN] ",
        "FAIL": "  [FAIL] ",
        "H1":  "\n" + "=" * 70,
        "H2":  "\n" + "-" * 50,
        "H3":  "  ",
    }
    prefix = prefix_map.get(level, "  ")
    line = prefix + "\n" + msg if level in ("H1", "H2") else f"{prefix}{msg}"
    print(line)
    report_lines.append(line)


def add_result(name, status, detail):
    """记录检查结果"""
    results_summary.append((name, status, detail))
    if status == "PASS":
        log(f"✓ {detail}", "PASS")
    elif status == "WARN":
        log(f"⚠ {detail}", "WARN")
    elif status == "FAIL":
        log(f"✗ {detail}", "FAIL")


def check_file_exists(path, label):
    """检查文件是否存在且非空"""
    if not os.path.exists(path):
        return False, f"文件不存在: {path}"
    if os.path.getsize(path) == 0:
        return False, f"文件为空: {path}"
    return True, path


def is_valid_smiles(smi):
    """校验 SMILES 字符串有效性"""
    if not HAS_RDKIT:
        return True  # 无 RDKit 时跳过
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        return mol is not None
    except Exception as e:
        logger.warning(f"SMILES 校验异常: {smi!r}, 错误: {e}")
        return False


def is_valid_gene_symbol(symbol):
    """校验基因符号格式: 大写字母+数字"""
    if pd.isna(symbol) or not isinstance(symbol, str):
        return False
    return bool(re.match(r'^[A-Z][A-Z0-9]*$', symbol.strip()))


def is_valid_uniprot_id(uid):
    """校验 UniProt ID 格式"""
    if pd.isna(uid) or not isinstance(uid, str):
        return False
    return bool(re.match(r'^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$', uid.strip()))


# ============================================================
# 1. CPI 数据校验
# ============================================================
def validate_cpi():
    log("│ 1. CPI 数据 (ChEMBL + BindingDB + DrugBank)", "H1")

    ok, path = check_file_exists(FILES["cpi"], "CPI")
    if not ok:
        add_result("CPI-文件存在性", "FAIL", path)
        return

    try:
        df = pd.read_csv(FILES["cpi"], low_memory=False)
        log(f"  文件: {os.path.basename(FILES['cpi'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
        log(f"  列数: {len(df.columns)}", "INFO")
    except Exception as e:
        add_result("CPI-读取", "FAIL", f"无法读取文件: {e}")
        return

    # 1.1 必需列检查
    required_cols = ["gene", "uniprot_id", "canonical_smiles", "standard_type",
                     "standard_value_nM", "pchembl_value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        add_result("CPI-必需列", "FAIL", f"缺少列: {missing_cols}")
    else:
        add_result("CPI-必需列", "PASS", f"所有 {len(required_cols)} 个必需列均存在")

    # 1.2 SMILES 有效性校验
    if "canonical_smiles" in df.columns:
        smiles_col = "canonical_smiles"
        valid_smiles = df[smiles_col].apply(is_valid_smiles)
        n_valid = valid_smiles.sum()
        n_invalid = (~valid_smiles).sum()
        if n_invalid > 0:
            add_result("CPI-SMILES有效性", "FAIL",
                       f"{n_invalid}/{len(df)} 条 SMILES 无效")
            # 打印前几个无效 SMILES
            bad = df[~valid_smiles][smiles_col].head(10).tolist()
            log(f"    前10个无效SMILES: {bad}", "WARN")
        else:
            add_result("CPI-SMILES有效性", "PASS",
                       f"全部 {n_valid:,} 条 SMILES 有效")

    # 1.3 重复行检查 (gene + SMILES + standard_type + standard_value_nM)
    if all(c in df.columns for c in ["gene", "canonical_smiles", "standard_type", "standard_value_nM"]):
        dup_key = df.duplicated(subset=["gene", "canonical_smiles", "standard_type", "standard_value_nM"])
        n_dup = dup_key.sum()
        if n_dup > 0:
            add_result("CPI-重复行", "FAIL",
                       f"存在 {n_dup:,} 条重复 (gene+SMILES+type+value)")
        else:
            add_result("CPI-重复行", "PASS", "无重复行")

    # 1.4 UniProt ID 格式校验
    if "uniprot_id" in df.columns:
        valid_uid = df["uniprot_id"].apply(is_valid_uniprot_id)
        n_valid = valid_uid.sum()
        n_invalid = (~valid_uid).sum()
        if n_invalid > 0:
            bad_uids = df[~valid_uid]["uniprot_id"].dropna().unique().tolist()[:10]
            add_result("CPI-UniProt格式", "WARN",
                       f"{n_invalid}/{len(df)} 条 UniProt ID 格式异常, 示例: {bad_uids}")
        else:
            add_result("CPI-UniProt格式", "PASS",
                       f"全部 {n_valid:,} UniProt ID 格式有效")

    # 1.5 standard_value_nM 数值性
    if "standard_value_nM" in df.columns:
        try:
            numeric_vals = pd.to_numeric(df["standard_value_nM"], errors="coerce")
            n_nan = numeric_vals.isna().sum()
            if n_nan > 0:
                add_result("CPI-standard_value_nM", "WARN",
                           f"{n_nan}/{len(df)} 条 standard_value_nM 非数值")
            else:
                add_result("CPI-standard_value_nM", "PASS",
                           "全部 standard_value_nM 为数值")
        except Exception as e:
            logger.error("standard_value_nM 转换失败: %s", e, exc_info=True)
            add_result("CPI-standard_value_nM", "FAIL", f"standard_value_nM 转换失败: {e}")

    # 1.6 pchembl_value 数值性
    if "pchembl_value" in df.columns:
        try:
            numeric_vals = pd.to_numeric(df["pchembl_value"], errors="coerce")
            n_nan = numeric_vals.isna().sum()
            if n_nan > 0:
                add_result("CPI-pchembl_value", "WARN",
                           f"{n_nan}/{len(df)} 条 pchembl_value 非数值 (可能为空值)")
            else:
                add_result("CPI-pchembl_value", "PASS",
                           "全部 pchembl_value 为数值")
        except Exception as e:
            logger.error("pchembl_value 转换失败: %s", e, exc_info=True)
            add_result("CPI-pchembl_value", "FAIL", f"pchembl_value 转换失败: {e}")

    # 1.7 基因符号格式校验
    if "gene" in df.columns:
        valid_gene = df["gene"].apply(is_valid_gene_symbol)
        n_invalid = (~valid_gene).sum()
        if n_invalid > 0:
            bad_genes = df[~valid_gene]["gene"].dropna().unique().tolist()[:10]
            add_result("CPI-基因符号", "WARN",
                       f"{n_invalid}/{len(df)} 条基因符号格式异常, 示例: {bad_genes}")
        else:
            add_result("CPI-基因符号", "PASS", "全部基因符号格式有效")

    # 1.8 统计信息
    log(f"  唯一基因数: {df['gene'].nunique() if 'gene' in df.columns else 'N/A'}", "INFO")
    log(f"  唯一SMILES数: {df['canonical_smiles'].nunique() if 'canonical_smiles' in df.columns else 'N/A'}", "INFO")
    log(f"  数据来源: {df['source'].value_counts().to_dict() if 'source' in df.columns else 'N/A'}", "INFO")

    return df


# ============================================================
# 2. PPI 网络边校验
# ============================================================
def validate_ppi():
    log("│ 2. PPI 网络 (STRING 扩展网络)", "H1")

    ok, path = check_file_exists(FILES["ppi_sig"], "PPI edges")
    if not ok:
        add_result("PPI-文件存在性", "FAIL", path)
        return None, None

    try:
        df = pd.read_csv(FILES["ppi_sig"])
        log(f"  文件: {os.path.basename(FILES['ppi_sig'])}", "INFO")
        log(f"  边数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("PPI-读取", "FAIL", f"无法读取: {e}")
        return None, None

    # 2.1 必需列
    required_cols = ["gene_a", "gene_b", "combined_score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        add_result("PPI-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("PPI-必需列", "PASS", f"所有 {len(required_cols)} 个必需列均存在")

    # 2.2 combined_score 数值性和范围
    if "combined_score" in df.columns:
        try:
            scores = pd.to_numeric(df["combined_score"], errors="coerce")
            n_nan = scores.isna().sum()
            if n_nan > 0:
                add_result("PPI-combined_score", "FAIL", f"{n_nan} 条 combined_score 非数值")
            else:
                # 检查范围 [0, 1000]
                s_min, s_max = scores.min(), scores.max()
                if s_min < 0 or s_max > 1000:
                    add_result("PPI-combined_score范围", "WARN",
                               f"combined_score 范围 [{s_min:.1f}, {s_max:.1f}], 超出 [0, 1000]")
                else:
                    add_result("PPI-combined_score", "PASS",
                               f"combined_score 范围 [{s_min:.1f}, {s_max:.1f}] 在 [0, 1000] 内")
        except Exception as e:
            logger.error("combined_score 转换失败: %s", e, exc_info=True)
            add_result("PPI-combined_score", "FAIL", f"无法转换 combined_score: {e}")

    # 2.3 自环检查
    if "gene_a" in df.columns and "gene_b" in df.columns:
        self_loops = df[df["gene_a"] == df["gene_b"]]
        n_self = len(self_loops)
        if n_self > 0:
            add_result("PPI-自环", "FAIL", f"存在 {n_self:,} 条自环边")
        else:
            add_result("PPI-自环", "PASS", "无自环边")

    # 2.4 重复边检查
    if "gene_a" in df.columns and "gene_b" in df.columns:
        # 标准化边: 按字母序排序 gene_a, gene_b
        def normalize_edge(row):
            return tuple(sorted([row["gene_a"], row["gene_b"]]))
        normalized = df.apply(normalize_edge, axis=1)
        n_dup = normalized.duplicated().sum()
        if n_dup > 0:
            add_result("PPI-重复边", "FAIL", f"存在 {n_dup:,} 条重复边")
        else:
            add_result("PPI-重复边", "PASS", "无重复边")

    # 2.5 提取所有唯一的 PPI 节点
    if "gene_a" in df.columns and "gene_b" in df.columns:
        nodes = set(df["gene_a"].dropna().unique()) | set(df["gene_b"].dropna().unique())
        log(f"  唯一节点数: {len(nodes):,}", "INFO")
        log(f"  唯一基因符号: {len(nodes):,}", "INFO")
    else:
        nodes = set()

    return df, nodes


# ============================================================
# 3. Ferroaging 96 基因校验
# ============================================================
def validate_ferroaging_genes():
    log("│ 3. Ferroaging 96 基因", "H1")

    ok, path = check_file_exists(FILES["genes96"], "Ferroaging genes")
    if not ok:
        add_result("GENES96-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["genes96"])
        log(f"  文件: {os.path.basename(FILES['genes96'])}", "INFO")
        log(f"  记录数: {len(df)}", "INFO")
    except Exception as e:
        add_result("GENES96-读取", "FAIL", f"无法读取: {e}")
        return None

    # 3.1 基因列
    gene_col = "gene_symbol" if "gene_symbol" in df.columns else "gene"
    if gene_col not in df.columns:
        add_result("GENES96-列名", "FAIL", f"找不到基因列, 可用列: {list(df.columns)}")
        return None

    genes = df[gene_col].dropna().unique().tolist()
    n_genes = len(genes)
    n_dups = df[gene_col].duplicated().sum()

    # 3.2 数量检查 (期望 96)
    if n_genes == 96:
        add_result("GENES96-数量", "PASS", f"恰好 {n_genes} 个唯一基因")
    elif n_genes == 100:
        add_result("GENES96-数量", "WARN", f"共 {n_genes} 个唯一基因 (可能是100基因版本, 包含4个补充)")
    else:
        add_result("GENES96-数量", "WARN", f"共 {n_genes} 个唯一基因 (期望 96)")

    # 3.3 重复检查
    if n_dups > 0:
        add_result("GENES96-重复", "FAIL", f"存在 {n_dups} 条重复基因")
    else:
        add_result("GENES96-重复", "PASS", "无重复基因")

    # 3.4 基因符号格式校验
    invalid_genes = [g for g in genes if not is_valid_gene_symbol(g)]
    if invalid_genes:
        add_result("GENES96-基因格式", "FAIL",
                   f"{len(invalid_genes)} 个基因符号格式无效: {invalid_genes[:10]}")
    else:
        add_result("GENES96-基因格式", "PASS", f"全部 {n_genes} 个基因符号格式有效")

    log(f"  基因列表 (前20): {genes[:20]}", "INFO")
    return set(genes)


# ============================================================
# 4. ESM-2 嵌入校验
# ============================================================
def validate_esm2():
    log("│ 4. ESM-2 蛋白质嵌入", "H1")

    ok, path = check_file_exists(FILES["esm2"], "ESM-2 embeddings")
    if not ok:
        add_result("ESM2-文件存在性", "FAIL", path)
        return None

    try:
        data = np.load(FILES["esm2"], allow_pickle=True)
        keys = list(data.keys())
        log(f"  文件: {os.path.basename(FILES['esm2'])}", "INFO")
        log(f"  蛋白质数量: {len(keys)}", "INFO")
    except Exception as e:
        add_result("ESM2-读取", "FAIL", f"无法读取: {e}")
        return None

    # 4.1 维度检查
    dims = set()
    n_nan_total = 0
    n_inf_total = 0
    bad_keys = []

    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray):
            dims.add(arr.shape)
            if np.isnan(arr).any():
                n_nan_total += 1
                bad_keys.append(f"{k}(NaN)")
            if np.isinf(arr).any():
                n_inf_total += 1
                bad_keys.append(f"{k}(Inf)")

    # 检查维度
    if len(dims) == 1:
        dim = list(dims)[0]
        if len(dim) == 1 and dim[0] == 640:
            add_result("ESM2-维度", "PASS", f"全部 {len(keys)} 个蛋白质嵌入维度为 {dim[0]}")
        else:
            add_result("ESM2-维度", "FAIL", f"维度异常: {dim}")
    else:
        add_result("ESM2-维度", "FAIL", f"维度不一致: {dims}")

    # 4.2 NaN/Inf 检查
    if n_nan_total > 0:
        add_result("ESM2-NaN", "FAIL", f"{n_nan_total} 个蛋白质嵌入含 NaN")
    else:
        add_result("ESM2-NaN", "PASS", "无 NaN 值")

    if n_inf_total > 0:
        add_result("ESM2-Inf", "FAIL", f"{n_inf_total} 个蛋白质嵌入含 Inf")
    else:
        add_result("ESM2-Inf", "PASS", "无 Inf 值")

    if bad_keys:
        log(f"  异常蛋白质: {bad_keys[:10]}", "WARN")

    return set(keys)


# ============================================================
# 5. KEGG 通路校验
# ============================================================
def validate_kegg():
    log("│ 5. KEGG 通路数据", "H1")

    ok, path = check_file_exists(FILES["kegg"], "KEGG pathways")
    if not ok:
        add_result("KEGG-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["kegg"], sep="\t")
        log(f"  文件: {os.path.basename(FILES['kegg'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
        log(f"  列: {list(df.columns)}", "INFO")
    except Exception as e:
        add_result("KEGG-读取", "FAIL", f"无法读取: {e}")
        return None

    # 5.1 必需列
    required = ["pathway_id", "pathway_name", "pathway_class", "gene_symbol"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_result("KEGG-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("KEGG-必需列", "PASS", f"所有 {len(required)} 个必需列均存在")

    # 5.2 pathway_id 格式 (hsaXXXXX)
    if "pathway_id" in df.columns:
        pids = df["pathway_id"].dropna().unique()
        valid_mask = pd.Series(pids).apply(lambda x: bool(re.match(r'^hsa\d{5}$', str(x))))
        invalid = pids[~valid_mask]
        if len(invalid) > 0:
            add_result("KEGG-pathway_id格式", "WARN",
                       f"{len(invalid)} 个 pathway_id 格式异常, 示例: {list(invalid)[:10]}")
        else:
            add_result("KEGG-pathway_id格式", "PASS",
                       f"全部 {len(pids)} 个 pathway_id 格式为 hsaXXXXX")
        log(f"  唯一通路数: {len(pids)}", "INFO")

    # 5.3 重复 gene-pathway 对
    if "pathway_id" in df.columns and "gene_symbol" in df.columns:
        dup = df.duplicated(subset=["pathway_id", "gene_symbol"])
        n_dup = dup.sum()
        if n_dup > 0:
            add_result("KEGG-重复对", "FAIL", f"存在 {n_dup:,} 条重复 gene-pathway 对")
        else:
            add_result("KEGG-重复对", "PASS", "无重复 gene-pathway 对")

    # 5.4 基因符号格式
    if "gene_symbol" in df.columns:
        valid_gene = df["gene_symbol"].apply(is_valid_gene_symbol)
        n_invalid = (~valid_gene).sum()
        if n_invalid > 0:
            bad = df[~valid_gene]["gene_symbol"].dropna().unique().tolist()[:10]
            add_result("KEGG-基因格式", "WARN",
                       f"{n_invalid}/{len(df)} 条基因符号格式异常, 示例: {bad}")
        else:
            add_result("KEGG-基因格式", "PASS", "全部基因符号格式有效")

    log(f"  唯一基因数: {df['gene_symbol'].nunique() if 'gene_symbol' in df.columns else 'N/A'}", "INFO")
    return df


# ============================================================
# 6. 铁死亡表型数据校验
# ============================================================
def validate_phenotype():
    log("│ 6. 铁死亡表型数据集", "H1")

    ok, path = check_file_exists(FILES["phenotype"], "Phenotype data")
    if not ok:
        add_result("PHENO-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["phenotype"])
        log(f"  文件: {os.path.basename(FILES['phenotype'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("PHENO-读取", "FAIL", f"无法读取: {e}")
        return None

    # 6.1 必需列
    required = ["canonical_smiles", "label", "ferroptosis_type", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_result("PHENO-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("PHENO-必需列", "PASS", f"所有 {len(required)} 个必需列均存在")

    # 6.2 SMILES 有效性
    if "canonical_smiles" in df.columns:
        valid_smiles = df["canonical_smiles"].apply(is_valid_smiles)
        n_invalid = (~valid_smiles).sum()
        if n_invalid > 0:
            add_result("PHENO-SMILES有效性", "FAIL",
                       f"{n_invalid}/{len(df)} 条 SMILES 无效")
            bad = df[~valid_smiles]["canonical_smiles"].head(10).tolist()
            log(f"    前10个无效SMILES: {bad}", "WARN")
        else:
            add_result("PHENO-SMILES有效性", "PASS",
                       f"全部 {valid_smiles.sum():,} 条 SMILES 有效")

    # 6.3 label 检查 (应为 0/1)
    if "label" in df.columns:
        labels = df["label"].dropna().unique()
        if set(labels) <= {0, 1}:
            add_result("PHENO-label", "PASS",
                       f"label 值仅为 0/1, 分布: {df['label'].value_counts().to_dict()}")
        else:
            add_result("PHENO-label", "FAIL", f"label 包含非 0/1 值: {labels}")

    # 6.4 重复 SMILES 检查 (同一 SMILES 不应有冲突 label)
    if "canonical_smiles" in df.columns and "label" in df.columns:
        # 找到有多个不同 label 的 SMILES
        grouped = df.groupby("canonical_smiles")["label"].nunique()
        conflict_smiles = grouped[grouped > 1]
        n_conflict = len(conflict_smiles)
        if n_conflict > 0:
            add_result("PHENO-冲突标签", "FAIL",
                       f"{n_conflict} 个 SMILES 有冲突的 label 值")
        else:
            add_result("PHENO-冲突标签", "PASS",
                       "无冲突标签的 SMILES")

    # 6.5 ferroptosis_type 分布
    if "ferroptosis_type" in df.columns:
        type_counts = df["ferroptosis_type"].value_counts().to_dict()
        log(f"  ferroptosis_type 分布: {type_counts}", "INFO")

    # 6.6 source 分布
    if "source" in df.columns:
        source_counts = df["source"].value_counts().to_dict()
        log(f"  source 分布: {source_counts}", "INFO")

    return df


# ============================================================
# 7. 疾病基因边校验
# ============================================================
def validate_disease_gene():
    log("│ 7. 疾病-基因关联边", "H1")

    ok, path = check_file_exists(FILES["disease_gene"], "Disease gene edges")
    if not ok:
        add_result("DG-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["disease_gene"])
        log(f"  文件: {os.path.basename(FILES['disease_gene'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("DG-读取", "FAIL", f"无法读取: {e}")
        return None

    # 7.1 必需列
    required = ["disease_name", "disease_type", "gene_symbol", "evidence", "source"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_result("DG-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("DG-必需列", "PASS", f"所有 {len(required)} 个必需列均存在")

    # 7.2 基因符号格式
    if "gene_symbol" in df.columns:
        valid_gene = df["gene_symbol"].apply(is_valid_gene_symbol)
        n_invalid = (~valid_gene).sum()
        if n_invalid > 0:
            bad = df[~valid_gene]["gene_symbol"].dropna().unique().tolist()[:10]
            add_result("DG-基因格式", "WARN",
                       f"{n_invalid}/{len(df)} 条基因符号格式异常, 示例: {bad}")
        else:
            add_result("DG-基因格式", "PASS", "全部基因符号格式有效")

    # 7.3 疾病类型分布
    if "disease_type" in df.columns:
        type_counts = df["disease_type"].value_counts().to_dict()
        log(f"  疾病类型分布: {type_counts}", "INFO")

    # 7.4 统计
    if "disease_name" in df.columns:
        log(f"  唯一疾病: {df['disease_name'].nunique()}", "INFO")
    if "gene_symbol" in df.columns:
        log(f"  唯一基因: {df['gene_symbol'].nunique()}", "INFO")

    return df


# ============================================================
# 8. TCM 候选化合物池校验
# ============================================================
def validate_tcm_pool():
    log("│ 8. TCM 候选化合物池", "H1")

    ok, path = check_file_exists(FILES["tcm_pool"], "TCM pool")
    if not ok:
        add_result("TCM-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["tcm_pool"])
        log(f"  文件: {os.path.basename(FILES['tcm_pool'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("TCM-读取", "FAIL", f"无法读取: {e}")
        return None

    # 8.1 SMILES_std 列存在
    if "SMILES_std" not in df.columns:
        add_result("TCM-SMILES列", "FAIL", f"缺少 SMILES_std 列, 可用: {list(df.columns)}")
        return None

    # 8.2 SMILES 有效性
    valid_smiles = df["SMILES_std"].apply(is_valid_smiles)
    n_valid = valid_smiles.sum()
    n_invalid = (~valid_smiles).sum()
    if n_invalid > 0:
        add_result("TCM-SMILES有效性", "FAIL",
                   f"{n_invalid}/{len(df)} 条 SMILES 无效")
        bad = df[~valid_smiles]["SMILES_std"].head(10).tolist()
        log(f"    前10个无效SMILES: {bad}", "WARN")
    else:
        add_result("TCM-SMILES有效性", "PASS", f"全部 {n_valid:,} 条 SMILES 有效")

    # 8.3 MOL_ID 唯一性
    if "MOL_ID" in df.columns:
        n_dup = df["MOL_ID"].duplicated().sum()
        if n_dup > 0:
            add_result("TCM-MOL_ID重复", "FAIL", f"存在 {n_dup} 个重复 MOL_ID")
        else:
            add_result("TCM-MOL_ID重复", "PASS", f"全部 {len(df):,} 个 MOL_ID 唯一")

    # 8.4 统计
    if "tier" in df.columns:
        tier_counts = df["tier"].value_counts().to_dict()
        log(f"  优先级分布: {tier_counts}", "INFO")

    log(f"  唯一 SMILES: {df['SMILES_std'].nunique() if 'SMILES_std' in df.columns else 'N/A'}", "INFO")
    return df


# ============================================================
# 9. CPI 补充数据校验
# ============================================================
def validate_cpi_supplement():
    log("│ 9. CPI 补充数据", "H1")

    ok, path = check_file_exists(FILES["cpi_supplement"], "CPI supplement")
    if not ok:
        add_result("CPI_SUPP-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["cpi_supplement"])
        log(f"  文件: {os.path.basename(FILES['cpi_supplement'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("CPI_SUPP-读取", "FAIL", f"无法读取: {e}")
        return None

    # 9.1 SMILES 有效性
    smiles_col = "smiles" if "smiles" in df.columns else "canonical_smiles"
    if smiles_col in df.columns:
        valid_smiles = df[smiles_col].apply(is_valid_smiles)
        n_invalid = (~valid_smiles).sum()
        if n_invalid > 0:
            add_result("CPI_SUPP-SMILES有效性", "FAIL",
                       f"{n_invalid}/{len(df)} 条 SMILES 无效")
        else:
            add_result("CPI_SUPP-SMILES有效性", "PASS",
                       f"全部 {valid_smiles.sum():,} 条 SMILES 有效")
    else:
        add_result("CPI_SUPP-SMILES有效性", "FAIL", f"找不到 SMILES 列, 可用: {list(df.columns)}")

    # 9.2 重复 gene+SMILES 对
    gene_col = "gene" if "gene" in df.columns else None
    if gene_col and smiles_col in df.columns:
        dup = df.duplicated(subset=[gene_col, smiles_col])
        n_dup = dup.sum()
        if n_dup > 0:
            add_result("CPI_SUPP-重复", "FAIL",
                       f"存在 {n_dup} 条重复 gene+SMILES 对")
        else:
            add_result("CPI_SUPP-重复", "PASS", "无重复 gene+SMILES 对")

    # 9.3 基因符号格式
    if gene_col:
        valid_gene = df[gene_col].apply(is_valid_gene_symbol)
        n_invalid = (~valid_gene).sum()
        if n_invalid > 0:
            bad = df[~valid_gene][gene_col].dropna().unique().tolist()[:10]
            add_result("CPI_SUPP-基因格式", "WARN",
                       f"{n_invalid}/{len(df)} 条基因符号异常, 示例: {bad}")
        else:
            add_result("CPI_SUPP-基因格式", "PASS", "全部基因符号格式有效")

    return df


# ============================================================
# 10. BindingDB 数据校验
# ============================================================
def validate_bindingdb():
    log("│ 10. BindingDB 数据", "H1")

    ok, path = check_file_exists(FILES["bindingdb"], "BindingDB")
    if not ok:
        add_result("BDB-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["bindingdb"], low_memory=False)
        log(f"  文件: {os.path.basename(FILES['bindingdb'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("BDB-读取", "FAIL", f"无法读取: {e}")
        return None

    # 10.1 必需列
    required = ["gene", "uniprot_id", "canonical_smiles", "standard_type", "standard_value_nM"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_result("BDB-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("BDB-必需列", "PASS", f"所有 {len(required)} 个必需列均存在")

    # 10.2 SMILES 有效性
    if "canonical_smiles" in df.columns:
        valid = df["canonical_smiles"].apply(is_valid_smiles)
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            add_result("BDB-SMILES有效性", "FAIL",
                       f"{n_invalid}/{len(df)} 条 SMILES 无效")
        else:
            add_result("BDB-SMILES有效性", "PASS",
                       f"全部 {valid.sum():,} 条 SMILES 有效")

    log(f"  唯一基因: {df['gene'].nunique() if 'gene' in df.columns else 'N/A'}", "INFO")
    return df


# ============================================================
# 11. DrugBank 数据校验
# ============================================================
def validate_drugbank():
    log("│ 11. DrugBank 数据", "H1")

    ok, path = check_file_exists(FILES["drugbank"], "DrugBank")
    if not ok:
        add_result("DBK-文件存在性", "FAIL", path)
        return None

    try:
        df = pd.read_csv(FILES["drugbank"])
        log(f"  文件: {os.path.basename(FILES['drugbank'])}", "INFO")
        log(f"  记录数: {len(df):,}", "INFO")
    except Exception as e:
        add_result("DBK-读取", "FAIL", f"无法读取: {e}")
        return None

    # 11.1 必需列
    required = ["gene", "uniprot_id", "drugbank_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        add_result("DBK-必需列", "FAIL", f"缺少列: {missing}")
    else:
        add_result("DBK-必需列", "PASS", f"所有 {len(required)} 个必需列均存在")

    # 11.2 drugbank_id 格式 (DBXXXXX)
    if "drugbank_id" in df.columns:
        valid_ids = df["drugbank_id"].dropna().apply(
            lambda x: bool(re.match(r'^DB\d{5}$', str(x))))
        n_invalid = (~valid_ids).sum()
        if n_invalid > 0:
            bad = df[~valid_ids]["drugbank_id"].dropna().unique().tolist()[:10]
            add_result("DBK-drugbank_id格式", "WARN",
                       f"{n_invalid}/{len(df)} 条 drugbank_id 格式异常, 示例: {bad}")
        else:
            add_result("DBK-drugbank_id格式", "PASS",
                       "全部 drugbank_id 格式为 DBXXXXX")

    log(f"  唯一基因: {df['gene'].nunique() if 'gene' in df.columns else 'N/A'}", "INFO")
    log(f"  唯一药物: {df['drugbank_id'].nunique() if 'drugbank_id' in df.columns else 'N/A'}", "INFO")
    return df


# ============================================================
# 12. 跨文件一致性校验
# ============================================================
def validate_cross_consistency(genes96, ppi_nodes, esm2_genes, cpi_df, phenotype_df, tcm_df, dis_gene_df, cpi_supp_df):
    log("│ 12. 跨文件一致性校验", "H1")

    if genes96 is None:
        add_result("CROSS-genes96", "FAIL", "无 ferroaging 基因数据, 跳过跨文件校验")
        return
    log(f"  Ferroaging 基因集: {len(genes96)} 个基因", "INFO")

    # 12.1 PPI 网络覆盖
    if ppi_nodes is not None:
        ppi_overlap = genes96 & ppi_nodes
        missing_in_ppi = genes96 - ppi_nodes
        if missing_in_ppi:
            add_result("CROSS-PPI覆盖", "WARN",
                       f"{len(ppi_overlap)}/{len(genes96)} 个基因在PPI网络中, "
                       f"缺失: {sorted(missing_in_ppi)[:10]}...")
        else:
            add_result("CROSS-PPI覆盖", "PASS",
                       f"全部 {len(genes96)} 个基因在PPI网络中")

    # 12.2 ESM-2 嵌入覆盖
    if esm2_genes is not None:
        esm2_overlap = genes96 & esm2_genes
        missing_in_esm2 = genes96 - esm2_genes
        if missing_in_esm2:
            add_result("CROSS-ESM2覆盖", "WARN",
                       f"{len(esm2_overlap)}/{len(genes96)} 个基因有ESM-2嵌入, "
                       f"缺失: {sorted(missing_in_esm2)[:10]}...")
        else:
            add_result("CROSS-ESM2覆盖", "PASS",
                       f"全部 {len(genes96)} 个基因有ESM-2嵌入")

    # 12.3 疾病基因引用 ferroaging 基因
    if dis_gene_df is not None and "gene_symbol" in dis_gene_df.columns:
        dis_genes = set(dis_gene_df["gene_symbol"].dropna().unique())
        overlap = dis_genes & genes96
        non_overlap = dis_genes - genes96
        if non_overlap:
            add_result("CROSS-疾病基因覆盖", "WARN",
                       f"{len(overlap)}/{len(dis_genes)} 个疾病关联基因在ferroaging96中, "
                       f"非ferroaging: {len(non_overlap)} 个")
        else:
            add_result("CROSS-疾病基因覆盖", "PASS",
                       "全部疾病关联基因在ferroaging96中")

    # 12.4 CPI 基因的 UniProt ID 映射
    if cpi_df is not None and "gene" in cpi_df.columns and "uniprot_id" in cpi_df.columns:
        cpi_genes = cpi_df["gene"].dropna().unique()
        cpi_with_uniprot = cpi_df.dropna(subset=["uniprot_id"])["gene"].nunique()
        if cpi_with_uniprot < len(cpi_genes):
            add_result("CROSS-CPI-UniProt映射", "WARN",
                       f"{cpi_with_uniprot}/{len(cpi_genes)} 个CPI基因有UniProt ID")
        else:
            add_result("CROSS-CPI-UniProt映射", "PASS",
                       f"全部 {len(cpi_genes)} 个CPI基因有UniProt ID")

    # 12.5 数据泄漏检查: TCM SMILES vs 训练集 SMILES
    if tcm_df is not None and phenotype_df is not None:
        if "SMILES_std" in tcm_df.columns and "canonical_smiles" in phenotype_df.columns:
            # 标准化 SMILES 比较
            tcm_smiles = set(tcm_df["SMILES_std"].dropna().apply(lambda x: x.strip()))
            train_smiles = set(phenotype_df["canonical_smiles"].dropna().apply(lambda x: x.strip()))
            overlap = tcm_smiles & train_smiles
            if overlap:
                add_result("CROSS-TCM数据泄漏", "FAIL",
                           f"TCM池与训练集有 {len(overlap)} 个重叠SMILES (数据泄漏!)")
                log(f"    重叠SMILES: {list(overlap)[:10]}", "WARN")
            else:
                add_result("CROSS-TCM数据泄漏", "PASS",
                           "TCM池与训练集无SMILES重叠 (无数据泄漏)")

            # 也检查与 CPI 补充数据的重叠
            if cpi_supp_df is not None:
                supp_smiles_col = "smiles" if "smiles" in cpi_supp_df.columns else "canonical_smiles"
                if supp_smiles_col in cpi_supp_df.columns:
                    supp_smiles = set(cpi_supp_df[supp_smiles_col].dropna().apply(lambda x: x.strip()))
                    tcm_supp_overlap = tcm_smiles & supp_smiles
                    if tcm_supp_overlap:
                        add_result("CROSS-TCM-CPI补充泄漏", "WARN",
                                   f"TCM池与CPI补充数据有 {len(tcm_supp_overlap)} 个重叠SMILES")
                    else:
                        add_result("CROSS-TCM-CPI补充泄漏", "PASS",
                                   "TCM池与CPI补充数据无SMILES重叠")

    # 12.6 CPI 数据中基因来源统计
    if cpi_df is not None and "source" in cpi_df.columns:
        source_counts = cpi_df["source"].value_counts().to_dict()
        log(f"  CPI数据来源分布: {source_counts}", "INFO")


# ============================================================
# 主函数
# ============================================================
def main():
    log("=" * 70)
    log("    铁衰老项目 - 数据真实性全面校验", "INFO")
    log(f"    执行时间: {CURRENT_TIME}", "INFO")
    log(f"    RDKit可用: {HAS_RDKIT}", "INFO")
    log("=" * 70)

    # ── 单文件校验 ──
    log("第一部分: 单文件逐项校验", "H1")
    cpi_df = validate_cpi()
    ppi_df, ppi_nodes = validate_ppi()
    genes96 = validate_ferroaging_genes()
    esm2_genes = validate_esm2()
    kegg_df = validate_kegg()
    phenotype_df = validate_phenotype()
    dis_gene_df = validate_disease_gene()
    tcm_df = validate_tcm_pool()
    cpi_supp_df = validate_cpi_supplement()
    bindingdb_df = validate_bindingdb()
    drugbank_df = validate_drugbank()

    # ── 跨文件一致性校验 ──
    log("\n\n第二部分: 跨文件一致性校验", "H1")
    validate_cross_consistency(
        genes96, ppi_nodes, esm2_genes,
        cpi_df, phenotype_df, tcm_df, dis_gene_df, cpi_supp_df
    )

    # ── 汇总 ──
    log("\n\n" + "=" * 70)
    log("    校验结果汇总", "H1")
    log("=" * 70)

    n_pass = sum(1 for _, s, _ in results_summary if s == "PASS")
    n_warn = sum(1 for _, s, _ in results_summary if s == "WARN")
    n_fail = sum(1 for _, s, _ in results_summary if s == "FAIL")

    log(f"\n{'检查项':<45s} {'状态':<8s} {'详情'}", "INFO")
    log("-" * 100, "INFO")
    for name, status, detail in results_summary:
        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(status, "?")
        log(f"  {name:<43s} {status_icon} {status:<5s} {detail}", "INFO")

    log("-" * 100, "INFO")
    log(f"  总计: {len(results_summary)} 项检查", "INFO")
    log(f"  通过: {n_pass} ({100*n_pass/max(1,len(results_summary)):.1f}%)", "PASS")
    if n_warn > 0:
        log(f"  警告: {n_warn} ({100*n_warn/max(1,len(results_summary)):.1f}%)", "WARN")
    if n_fail > 0:
        log(f"  失败: {n_fail} ({100*n_fail/max(1,len(results_summary)):.1f}%)", "FAIL")

    log(f"\n  结论: {'数据真实性校验通过' if n_fail == 0 else '存在需要修复的问题'}", "INFO")
    log("=" * 70)

    # ── 保存报告 ──
    try:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n报告已保存至: {REPORT_PATH}")
    except Exception as e:
        print(f"\n无法保存报告: {e}")

    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)