#!/usr/bin/env python3
"""
模型输入数据修补与验证脚本
==========================
针对 Phase 4 GAT+HGT 管线的输入数据进行全面检查与修补：
  1. CPI 清洗后数据质量复核（DtypeWarning 处理、空值检查）
  2. TCM 候选池与 CPI 训练集重叠检测与去泄漏
  3. 蛋白特征表完整性检查与缺失基因报告
  4. KEGG 通路注释核对（使用 L2 已下载的 KEGG）
  5. PPI 网络核对（使用 L1 扩展后的 PPI）

输出：
  L4/results/model_input_repair_report.json
  L3/results/tcm_compound_pool_tox_filtered_noleak.csv
  L4/logs/repair_model_inputs.log
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_RESULTS = PROJECT_ROOT / "L4" / "results"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"

for d in [L4_LOGS, L4_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "repair_model_inputs.log"
REPORT_FILE = L4_RESULTS / "model_input_repair_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> str:
    """用 RDKit 将 SMILES 规范化为 canonical 形式；无法解析返回空字符串。"""
    if pd.isna(smiles):
        return ""
    s = str(smiles).strip()
    if not s:
        return ""
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def load_ferroaging_genes() -> Set[str]:
    """加载铁衰老基因集。"""
    path = L1_RESULTS / "ferroaging_genes_96.csv"
    if not path.exists():
        logger.error("铁衰老基因集不存在: %s", path)
        return set()
    df = pd.read_csv(path)
    genes = set(df["gene_symbol"].dropna().astype(str).str.upper().unique())
    logger.info("铁衰老基因集: %d 个基因", len(genes))
    return genes


def check_cpi_data() -> Dict[str, Any]:
    """检查 CPI 清洗后数据。"""
    path = L4_RESULTS / "experimental_actives_detail_cleaned.csv"
    logger.info("=" * 60)
    logger.info("[1/5] CPI 数据检查")
    logger.info("=" * 60)

    if not path.exists():
        logger.error("CPI 清洗文件不存在: %s", path)
        return {"status": "ERROR", "error": "file_not_found"}

    # 使用 low_memory=False 避免 DtypeWarning
    df = pd.read_csv(path, low_memory=False)
    report = {
        "file": str(path),
        "rows": int(len(df)),
        "genes": int(df["gene"].nunique()),
        "unique_smiles": int(df["canonical_smiles"].nunique()),
        "required_columns_present": True,
        "null_smiles": 0,
        "invalid_smiles": 0,
        "duplicate_gene_smiles": 0,
        "per_target_counts": {},
        "warnings": [],
    }

    required = ["gene", "canonical_smiles", "uniprot_id"]
    for col in required:
        if col not in df.columns:
            report["required_columns_present"] = False
            report["warnings"].append(f"缺少必需列: {col}")

    null_mask = df["canonical_smiles"].isna() | (df["canonical_smiles"].astype(str).str.strip() == "")
    report["null_smiles"] = int(null_mask.sum())
    if report["null_smiles"] > 0:
        report["warnings"].append(f"存在 {report['null_smiles']} 条空 SMILES")

    dup = df.duplicated(subset=["gene", "canonical_smiles"], keep=False).sum()
    report["duplicate_gene_smiles"] = int(dup)
    if dup > 0:
        report["warnings"].append(f"存在 {dup} 条 (gene, SMILES) 重复")

    # 验证 canonical_smiles 是否可被 RDKit 解析
    invalid = 0
    sample_invalid: List[str] = []
    for smi in df["canonical_smiles"].dropna().unique()[:5000]:
        if canonicalize_smiles(smi) == "":
            invalid += 1
            if len(sample_invalid) < 5:
                sample_invalid.append(str(smi))
    report["invalid_smiles_sample_check"] = {
        "checked_unique": int(min(5000, df["canonical_smiles"].nunique())),
        "invalid": invalid,
        "samples": sample_invalid,
    }
    if invalid > 0:
        report["warnings"].append(f"抽样发现 {invalid} 条不可解析 SMILES")

    gene_counts = df["gene"].value_counts()
    report["per_target_counts"] = {g: int(c) for g, c in gene_counts.head(20).items()}
    report["sparse_targets"] = sorted(gene_counts[gene_counts < 10].index.tolist())
    report["low_count_targets"] = sorted(gene_counts[(gene_counts >= 10) & (gene_counts < 50)].index.tolist())

    report["status"] = "OK" if not report["warnings"] else "WARN"
    logger.info("CPI 检查完成: %d 行, %d 基因, %d 唯一 SMILES, 状态=%s",
                report["rows"], report["genes"], report["unique_smiles"], report["status"])
    if report["warnings"]:
        for w in report["warnings"]:
            logger.warning("  - %s", w)
    return report


def repair_tcm_pool(cpi_df: pd.DataFrame) -> Dict[str, Any]:
    """检查并修复 TCM 候选池与 CPI 训练集的数据泄漏。"""
    input_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    output_path = L3_RESULTS / "tcm_compound_pool_tox_filtered_noleak.csv"

    logger.info("=" * 60)
    logger.info("[2/5] TCM 候选池泄漏检查")
    logger.info("=" * 60)

    if not input_path.exists():
        logger.error("TCM 候选池不存在: %s", input_path)
        return {"status": "ERROR", "error": "file_not_found"}

    tcm_df = pd.read_csv(input_path, low_memory=False)
    original_n = len(tcm_df)

    # 标准化 SMILES 列名
    smiles_col = None
    for col in ["SMILES_std", "canonical_smiles", "SMILES", "smiles"]:
        if col in tcm_df.columns:
            smiles_col = col
            break

    report = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_compounds": int(original_n),
        "smiles_column": smiles_col,
        "overlaps": [],
        "removed": 0,
        "remaining": int(original_n),
        "status": "OK",
    }

    if smiles_col is None:
        report["status"] = "ERROR"
        report["error"] = "no_smiles_column"
        logger.error("TCM 池未找到 SMILES 列")
        return report

    # 规范化 CPI SMILES
    cpi_smiles_set = set(cpi_df["canonical_smiles"].dropna().astype(str).str.strip().unique())

    # 规范化 TCM SMILES 并检测重叠
    tcm_smiles_raw = tcm_df[smiles_col].astype(str).str.strip()
    tcm_smiles_canonical = tcm_smiles_raw.apply(canonicalize_smiles)

    # 记录原始无法解析的 TCM SMILES
    unparseable = (tcm_smiles_canonical == "").sum()
    if unparseable > 0:
        logger.warning("TCM 池中有 %d 条 SMILES 无法被 RDKit 解析", unparseable)
        report["unparseable_smiles"] = int(unparseable)

    # 重叠检测：直接字符串匹配 + canonical 匹配
    overlap_mask = tcm_smiles_raw.isin(cpi_smiles_set) | tcm_smiles_canonical.isin(cpi_smiles_set)
    overlap_indices = tcm_df[overlap_mask].index.tolist()

    if overlap_indices:
        overlap_records = tcm_df.loc[overlap_mask, ["MOL_ID", smiles_col]].copy()
        overlap_records["canonical_smiles"] = overlap_records[smiles_col].apply(canonicalize_smiles)
        report["overlaps"] = overlap_records.to_dict(orient="records")
        report["removed"] = int(overlap_mask.sum())
        report["remaining"] = int(original_n - overlap_mask.sum())
        report["status"] = "REPAIRED"

        cleaned_df = tcm_df.loc[~overlap_mask].copy()
        cleaned_df.to_csv(output_path, index=False)
        logger.info("移除 %d 个与 CPI 训练集重叠的 TCM 化合物，已保存至 %s",
                    report["removed"], output_path)
    else:
        # 无重叠也保存一份，保持接口一致
        tcm_df.to_csv(output_path, index=False)
        logger.info("TCM 候选池无泄漏，已复制至 %s", output_path)

    logger.info("TCM 池: 原始 %d -> 剩余 %d", original_n, report["remaining"])
    return report


def check_protein_features(warm_targets: Set[str]) -> Dict[str, Any]:
    """检查蛋白特征表完整性。"""
    path = L2_RESULTS / "target_protein_features.csv"
    logger.info("=" * 60)
    logger.info("[3/5] 蛋白特征表检查")
    logger.info("=" * 60)

    if not path.exists():
        logger.error("蛋白特征文件不存在: %s", path)
        return {"status": "ERROR", "error": "file_not_found"}

    df = pd.read_csv(path, low_memory=False)
    genes_with_features = set(df["gene_symbol"].dropna().astype(str).str.upper().unique())

    report = {
        "file": str(path),
        "total_proteins": int(len(genes_with_features)),
        "missing_from_warm_targets": sorted(warm_targets - genes_with_features),
        "missing_count": int(len(warm_targets - genes_with_features)),
        "warm_targets_covered": sorted(warm_targets & genes_with_features),
        "warm_targets_covered_count": int(len(warm_targets & genes_with_features)),
        "feature_columns": [c for c in df.columns if c not in ["uniprot_id", "gene_symbol", "sequence"]],
        "warnings": [],
    }

    # 检查关键数值列是否全为 0/NaN
    numeric_cols = ["n_domains", "n_ptms", "n_transmembrane", "length"]
    zero_cols = []
    for col in numeric_cols:
        if col in df.columns and (df[col].fillna(0) == 0).all():
            zero_cols.append(col)
    if zero_cols:
        report["warnings"].append(f"以下列全为 0/NaN: {zero_cols}")

    report["status"] = "OK" if not report["missing_from_warm_targets"] and not zero_cols else "WARN"
    logger.info("蛋白特征: %d 个蛋白", report["total_proteins"])
    logger.info("温靶标覆盖: %d/%d", report["warm_targets_covered_count"], len(warm_targets))
    if report["missing_from_warm_targets"]:
        logger.warning("缺失蛋白特征的温靶标: %s", report["missing_from_warm_targets"])
    if zero_cols:
        logger.warning("全零特征列: %s", zero_cols)
    return report


def check_kegg_pathways() -> Dict[str, Any]:
    """检查 L2 已下载的 KEGG 通路注释。"""
    path = L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
    logger.info("=" * 60)
    logger.info("[4/5] KEGG 通路注释检查")
    logger.info("=" * 60)

    if not path.exists():
        logger.error("KEGG 通路文件不存在: %s", path)
        return {"status": "ERROR", "error": "file_not_found"}

    df = pd.read_csv(path, sep="\t", low_memory=False)
    report = {
        "file": str(path),
        "rows": int(len(df)),
        "pathways": int(df["pathway_id"].nunique()),
        "genes_with_pathway": int(df["gene_symbol"].nunique()),
        "required_columns_present": all(c in df.columns for c in ["pathway_id", "gene_symbol"]),
        "sample": df.head(3).to_dict(orient="records"),
        "status": "OK",
    }

    if not report["required_columns_present"]:
        report["status"] = "ERROR"
        report["warnings"] = ["缺少必需列 pathway_id 或 gene_symbol"]
    else:
        report["status"] = "OK"

    logger.info("KEGG: %d 条记录, %d 通路, %d 基因", report["rows"], report["pathways"], report["genes_with_pathway"])
    return report


def check_ppi_network() -> Dict[str, Any]:
    """检查扩展后的 PPI 网络。"""
    extended_path = L1_RESULTS / "ppi_network_extended_edges.csv"
    original_path = L1_RESULTS / "ppi_network_edges.csv"

    logger.info("=" * 60)
    logger.info("[5/5] PPI 网络检查")
    logger.info("=" * 60)

    report = {
        "original_file": str(original_path),
        "extended_file": str(extended_path),
        "original_edges": 0,
        "extended_edges": 0,
        "extended_nodes": 0,
        "extended_score_range": {},
        "status": "OK",
    }

    if original_path.exists():
        df_orig = pd.read_csv(original_path, low_memory=False)
        report["original_edges"] = int(len(df_orig))
        logger.info("原始 PPI 文件: %d 条边", report["original_edges"])

    if not extended_path.exists():
        report["status"] = "ERROR"
        report["error"] = "extended_file_not_found"
        logger.error("扩展 PPI 网络不存在: %s", extended_path)
        return report

    df_ext = pd.read_csv(extended_path, low_memory=False)
    report["extended_edges"] = int(len(df_ext))
    report["extended_nodes"] = int(pd.concat([df_ext["gene_a"], df_ext["gene_b"]]).nunique())
    report["extended_score_range"] = {
        "min": float(df_ext["combined_score"].min()),
        "max": float(df_ext["combined_score"].max()),
        "mean": float(df_ext["combined_score"].mean()),
    }
    logger.info("扩展 PPI 网络: %d 条边, %d 个节点", report["extended_edges"], report["extended_nodes"])
    logger.info("combined_score 范围: %.1f - %.1f", report["extended_score_range"]["min"], report["extended_score_range"]["max"])
    return report


def main():
    logger.info("=" * 60)
    logger.info("模型输入数据修补与验证")
    logger.info("=" * 60)

    ferroaging_genes = load_ferroaging_genes()

    # 1. CPI
    cpi_report = check_cpi_data()

    # 2. TCM pool leak repair (need CPI df)
    cpi_path = L4_RESULTS / "experimental_actives_detail_cleaned.csv"
    cpi_df = pd.read_csv(cpi_path, low_memory=False) if cpi_path.exists() else pd.DataFrame()
    tcm_report = repair_tcm_pool(cpi_df)

    # 3. Protein features
    # warm targets = genes that have CPI data AND are in ferroaging list
    warm_targets = set()
    if not cpi_df.empty and "gene" in cpi_df.columns:
        warm_targets = set(cpi_df["gene"].dropna().astype(str).str.upper().unique()) & ferroaging_genes
    protein_report = check_protein_features(warm_targets)

    # 4. KEGG
    kegg_report = check_kegg_pathways()

    # 5. PPI
    ppi_report = check_ppi_network()

    # Aggregate report
    full_report = {
        "summary": {
            "ferroaging_genes": len(ferroaging_genes),
            "warm_targets": sorted(warm_targets),
            "warm_target_count": len(warm_targets),
            "overall_status": "OK",
        },
        "cpi": cpi_report,
        "tcm_pool": tcm_report,
        "protein_features": protein_report,
        "kegg_pathways": kegg_report,
        "ppi_network": ppi_report,
    }

    # Determine overall status
    statuses = [r.get("status", "OK") for r in [cpi_report, tcm_report, protein_report, kegg_report, ppi_report]]
    if any(s == "ERROR" for s in statuses):
        full_report["summary"]["overall_status"] = "ERROR"
    elif any(s in ("WARN", "REPAIRED") for s in statuses):
        full_report["summary"]["overall_status"] = "WARN/REPAIRED"
    else:
        full_report["summary"]["overall_status"] = "OK"

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    logger.info("=" * 60)
    logger.info("报告已保存: %s", REPORT_FILE)
    logger.info("总体状态: %s", full_report["summary"]["overall_status"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
