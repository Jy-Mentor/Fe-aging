#!/usr/bin/env python3
"""
CPI 数据清洗脚本
================
对 L4/results/experimental_actives_detail.csv 进行清洗：
  1. 剔除 canonical_smiles 为空/无效的条目
  2. 用 RDKit 将 SMILES 规范化为 canonical 形式
  3. 剔除 standard_relation != '=' 的非精确活性记录
  4. 按 (gene, canonical_smiles) 去重，保留第一条
  5. 识别并可选剔除稀疏靶标（<10 条 CPI）
  6. 输出清洗后的 CPI 文件与清洗报告

运行：
    python L4/scripts/clean_cpi_data.py
输出：
    L4/results/experimental_actives_detail_cleaned.csv
    L4/results/cpi_cleaning_report.json
    L4/logs/clean_cpi_data.log
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(Path("L4/logs/clean_cpi_data.log"), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
INPUT = BASE / "L4" / "results" / "experimental_actives_detail.csv"
OUTPUT = BASE / "L4" / "results" / "experimental_actives_detail_cleaned.csv"
REPORT = BASE / "L4" / "results" / "cpi_cleaning_report.json"

# 默认保留稀疏靶标但记录；设为 True 可剔除
DROP_SPARSE_TARGETS = False
SPARSE_THRESHOLD = 10


def canonicalize_smiles(smiles: str) -> str:
    """用 RDKit 将 SMILES 转为 canonical 形式；无法解析返回空字符串。"""
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


def main():
    logger.info("=" * 60)
    logger.info("CPI 数据清洗")
    logger.info("=" * 60)

    if not INPUT.exists():
        logger.error("CPI 文件不存在: %s", INPUT)
        sys.exit(1)

    df = pd.read_csv(INPUT, low_memory=False)
    n_total = len(df)
    logger.info("原始 CPI 数据: %d 行, %d 基因, %d 唯一 SMILES",
                n_total, df["gene"].nunique(), df["canonical_smiles"].nunique())

    report = {
        "input_file": str(INPUT),
        "output_file": str(OUTPUT),
        "initial_rows": int(n_total),
        "initial_genes": int(df["gene"].nunique()),
        "initial_unique_smiles": int(df["canonical_smiles"].nunique()),
        "steps": [],
    }

    # 1. 删除空 SMILES
    null_mask = df["canonical_smiles"].isna() | (df["canonical_smiles"].astype(str).str.strip() == "")
    n_null = null_mask.sum()
    df = df[~null_mask].copy()
    report["steps"].append({
        "step": "drop_null_smiles",
        "dropped": int(n_null),
        "remaining": int(len(df)),
    })
    logger.info("[1/5] 删除空 SMILES: %d 行, 剩余 %d 行", n_null, len(df))

    # 2. 规范化为 canonical SMILES 并剔除无效 SMILES
    logger.info("[2/5] 规范化 SMILES 并检测无效项...")
    df["canonical_smiles_original"] = df["canonical_smiles"].astype(str).str.strip()
    df["canonical_smiles"] = df["canonical_smiles_original"].apply(canonicalize_smiles)
    invalid_mask = df["canonical_smiles"] == ""
    n_invalid = invalid_mask.sum()

    # 记录部分无效 SMILES 示例
    invalid_examples = df.loc[invalid_mask, "canonical_smiles_original"].head(10).tolist()
    df = df[~invalid_mask].copy()
    report["steps"].append({
        "step": "canonicalize_and_drop_invalid_smiles",
        "dropped": int(n_invalid),
        "remaining": int(len(df)),
        "invalid_examples": invalid_examples,
    })
    logger.info("  无效/不可解析 SMILES: %d 行, 剩余 %d 行", n_invalid, len(df))

    # 3. 剔除 standard_relation != '=' 的记录
    if "standard_relation" in df.columns:
        non_eq_mask = df["standard_relation"].fillna("=") != "="
        n_non_eq = non_eq_mask.sum()
        if n_non_eq > 0:
            df = df[~non_eq_mask].copy()
            logger.info("[3/5] 剔除 standard_relation != '=' 记录: %d 行, 剩余 %d 行",
                        n_non_eq, len(df))
        else:
            logger.info("[3/5] 无 standard_relation != '=' 记录")
        report["steps"].append({
            "step": "drop_non_exact_relation",
            "dropped": int(n_non_eq),
            "remaining": int(len(df)),
        })
    else:
        logger.info("[3/5] 无 standard_relation 列，跳过")
        report["steps"].append({
            "step": "drop_non_exact_relation",
            "dropped": 0,
            "remaining": int(len(df)),
            "note": "column_missing",
        })

    # 4. 按 (gene, canonical_smiles) 去重
    n_before_dup = len(df)
    df = df.drop_duplicates(subset=["gene", "canonical_smiles"], keep="first").copy()
    n_dup = n_before_dup - len(df)
    report["steps"].append({
        "step": "deduplicate_gene_smiles",
        "dropped": int(n_dup),
        "remaining": int(len(df)),
    })
    logger.info("[4/5] 按 (gene, SMILES) 去重: %d 行, 剩余 %d 行", n_dup, len(df))

    # 5. 稀疏靶标处理
    gene_counts = df["gene"].value_counts()
    sparse_targets = gene_counts[gene_counts < SPARSE_THRESHOLD].index.tolist()
    low_count_targets = gene_counts[(gene_counts >= SPARSE_THRESHOLD) & (gene_counts < 50)].index.tolist()

    logger.info("[5/5] 稀疏靶标 (<10): %d 个 — %s", len(sparse_targets), sparse_targets)
    logger.info("        低样本靶标 (10-49): %d 个", len(low_count_targets))

    if DROP_SPARSE_TARGETS and sparse_targets:
        n_before_sparse = len(df)
        df = df[~df["gene"].isin(sparse_targets)].copy()
        n_sparse_dropped = n_before_sparse - len(df)
        logger.info("  已剔除稀疏靶标: %d 行, 剩余 %d 行", n_sparse_dropped, len(df))
    else:
        n_sparse_dropped = 0
        logger.info("  保留稀疏靶标（未剔除），将在报告中标记")

    report["sparse_targets"] = {
        "threshold": SPARSE_THRESHOLD,
        "n_sparse": len(sparse_targets),
        "sparse_genes": sorted(sparse_targets),
        "n_low_count": len(low_count_targets),
        "low_count_genes": sorted(low_count_targets),
        "dropped": int(n_sparse_dropped),
    }

    # 6. 保存清洗后文件
    # 删除临时列
    df = df.drop(columns=["canonical_smiles_original"], errors="ignore")
    df.to_csv(OUTPUT, index=False)
    logger.info("已保存清洗后 CPI: %s", OUTPUT)

    # 7. 最终统计
    final_stats = {
        "final_rows": int(len(df)),
        "final_genes": int(df["gene"].nunique()),
        "final_unique_smiles": int(df["canonical_smiles"].nunique()),
        "per_target_counts": {g: int(c) for g, c in gene_counts.head(20).items()},
    }
    report["final"] = final_stats

    logger.info("=" * 60)
    logger.info("清洗完成")
    logger.info("  原始: %d 行", n_total)
    logger.info("  最终: %d 行 (%d 基因, %d 唯一 SMILES)",
                final_stats["final_rows"], final_stats["final_genes"],
                final_stats["final_unique_smiles"])
    logger.info("=" * 60)

    # 保存报告
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("清洗报告: %s", REPORT)


if __name__ == "__main__":
    main()
