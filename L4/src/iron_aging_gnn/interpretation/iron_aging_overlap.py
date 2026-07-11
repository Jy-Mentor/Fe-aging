"""铁衰老基因集交集分析：预测靶点与铁衰老基因集的交集分析。

参考:
  - 铁衰老基因集来源: L1/results/ferroaging_genes_96.csv (WGCNA + RRA 整合)
  - FerrDb: 铁死亡数据库 (driver/suppressor/marker/inducer)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """从当前文件向上查找项目根目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "L1").exists() and (parent / "L4").exists():
            return parent
    return current.parents[4]


def load_iron_aging_genes(
    gene_set_path: Optional[Path] = None,
) -> set[str]:
    """加载铁衰老 96 基因集。

    Args:
        gene_set_path: 铁衰老基因集 CSV 路径，默认从 L1 读取

    Returns:
        {gene_symbol, ...} 大写基因符号集合
    """
    if gene_set_path is None:
        project_root = _find_project_root()
        gene_set_path = project_root / "L1" / "results" / "ferroaging_genes_96.csv"

    if not gene_set_path.exists():
        logger.error(f"铁衰老基因集文件不存在: {gene_set_path}")
        return set()

    df = pd.read_csv(gene_set_path)
    if "gene_symbol" not in df.columns:
        logger.error(f"铁衰老基因集缺少 gene_symbol 列: {df.columns.tolist()}")
        return set()

    genes = {g.upper() for g in df["gene_symbol"].dropna().astype(str)}
    logger.info(f"铁衰老基因集加载完成: {len(genes)} 个基因")
    return genes


def load_ferroptosis_gene_sets(
    ferroptosis_dir: Optional[Path] = None,
) -> dict[str, set[str]]:
    """加载铁死亡数据库 (FerrDb) 基因分类。

    Args:
        ferroptosis_dir: FerrDb 数据目录路径

    Returns:
        {"driver": {gene, ...}, "suppressor": {gene, ...}, ...}
    """
    if ferroptosis_dir is None:
        project_root = _find_project_root()
        ferroptosis_dir = (project_root / "L4" / "data" / "ferroptosis_lib"
                           / "ferrdb_v2" / "ferroptosis_early_preview_upto20231231")

    if not ferroptosis_dir.exists():
        logger.warning(f"FerrDb 数据目录不存在: {ferroptosis_dir}")
        return {}

    gene_sets: dict[str, set[str]] = {}
    category_files = {
        "driver": "driver.csv",
        "suppressor": "suppressor.csv",
        "marker": "marker.csv",
        "inducer": "inducer.csv",
        "inhibitor": "inhibitor.csv",
    }

    for category, filename in category_files.items():
        filepath = ferroptosis_dir / filename
        if not filepath.exists():
            logger.warning(f"FerrDb 文件不存在: {filepath}")
            continue
        df = pd.read_csv(filepath)
        gene_col = "Symbol_or_reported_abbr"
        if gene_col not in df.columns:
            logger.warning(f"FerrDb {category} 文件缺少基因列")
            continue
        genes = {g.upper() for g in df[gene_col].dropna().astype(str)}
        gene_sets[category] = genes

    logger.info(f"FerrDb 基因集加载完成: {len(gene_sets)} 个类别")
    return gene_sets


def compute_iron_aging_overlap(
    predicted_genes: list[str],
    iron_aging_genes: Optional[set[str]] = None,
    ferroptosis_sets: Optional[dict[str, set[str]]] = None,
    score_threshold: Optional[float] = None,
    predicted_scores: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """计算预测靶点与铁衰老/铁死亡基因集的交集。

    Args:
        predicted_genes: 预测靶点基因列表
        iron_aging_genes: 铁衰老 96 基因集，默认自动加载
        ferroptosis_sets: 铁死亡分类基因集，默认自动加载
        score_threshold: 预测分数阈值，低于此值的基因不纳入交集
        predicted_scores: {gene: score} 预测分数映射

    Returns:
        DataFrame: 交集分析结果表
    """
    if iron_aging_genes is None:
        iron_aging_genes = load_iron_aging_genes()
    if ferroptosis_sets is None:
        ferroptosis_sets = load_ferroptosis_gene_sets()

    pred_set = {g.upper() for g in predicted_genes}
    if predicted_scores:
        score_map = {k.upper(): v for k, v in predicted_scores.items()}
    else:
        score_map = {g: 1.0 for g in pred_set}

    if score_threshold is not None:
        pred_set = {g for g in pred_set if score_map.get(g, 0) >= score_threshold}

    results = []

    # 铁衰老 96 基因集交集
    ia_overlap = pred_set & iron_aging_genes
    for gene in sorted(ia_overlap):
        results.append({
            "gene": gene,
            "gene_set": "iron_aging_96",
            "category": "铁衰老核心基因",
            "prediction_score": score_map.get(gene, None),
        })

    # 铁死亡分类交集
    for category, gene_set in ferroptosis_sets.items():
        overlap = pred_set & gene_set
        category_names = {
            "driver": "铁死亡驱动基因",
            "suppressor": "铁死亡抑制基因",
            "marker": "铁死亡标志基因",
            "inducer": "铁死亡诱导基因",
            "inhibitor": "铁死亡抑制因子",
        }
        cat_name = category_names.get(category, category)
        for gene in sorted(overlap):
            results.append({
                "gene": gene,
                "gene_set": f"ferroptosis_{category}",
                "category": cat_name,
                "prediction_score": score_map.get(gene, None),
            })

    if not results:
        logger.warning("预测靶点与铁衰老/铁死亡基因集无交集")
        return pd.DataFrame(columns=["gene", "gene_set", "category", "prediction_score"])

    df = pd.DataFrame(results)
    n_ia = len(ia_overlap)
    n_ferr = len(df[df["gene_set"] != "iron_aging_96"]["gene"].unique())
    n_total_targets = len(pred_set)
    logger.info(f"铁衰老交集分析完成: {n_total_targets} 预测靶点中, "
                f"{n_ia} 个铁衰老核心基因, {n_ferr} 个铁死亡相关基因")
    return df


__all__ = [
    "load_iron_aging_genes",
    "load_ferroptosis_gene_sets",
    "compute_iron_aging_overlap",
]