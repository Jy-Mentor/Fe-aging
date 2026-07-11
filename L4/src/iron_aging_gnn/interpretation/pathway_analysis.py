"""通路富集分析：基于 KEGG 数据库的 Fisher 精确检验通路富集。

参考:
  - Kanehisa et al. (2021) "KEGG: integrating viruses and cellular organisms", Nucleic Acids Research
  - Subramanian et al. (2005) "Gene set enrichment analysis", PNAS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """从当前文件向上查找项目根目录。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "L1").exists() and (parent / "L4").exists():
            return parent
    return current.parents[4]


def load_kegg_pathway_genes(
    kegg_path: Optional[Path] = None,
) -> dict[str, set[str]]:
    """加载 KEGG 通路→基因集合映射。

    Args:
        kegg_path: KEGG 通路基因 TSV 文件路径，默认从 L2 读取

    Returns:
        {pathway_id: {gene_symbol, ...}} 映射
    """
    if kegg_path is None:
        project_root = _find_project_root()
        kegg_path = project_root / "L2" / "results" / "kegg_pathways" / "kegg_human_pathway_genes.tsv"

    if not kegg_path.exists():
        logger.error(f"KEGG 通路文件不存在: {kegg_path}")
        return {}

    df = pd.read_csv(kegg_path, sep="\t", low_memory=False)
    if not {"pathway_id", "gene_symbol"}.issubset(df.columns):
        logger.error(f"KEGG 文件缺少必需列 (pathway_id, gene_symbol): {df.columns.tolist()}")
        return {}

    pathway_to_genes: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        pid = str(row["pathway_id"]).strip()
        gene = str(row["gene_symbol"]).strip().upper()
        if not pid or not gene:
            continue
        if pid not in pathway_to_genes:
            pathway_to_genes[pid] = set()
        pathway_to_genes[pid].add(gene)

    logger.info(f"KEGG 通路加载完成: {len(pathway_to_genes)} 通路, "
                f"{df['gene_symbol'].nunique()} 基因")
    return pathway_to_genes


def load_pathway_annotations(
    annot_path: Optional[Path] = None,
) -> dict[str, dict[str, str]]:
    """加载 KEGG 通路注释信息（名称、类别）。

    Args:
        annot_path: 通路注释 CSV 文件路径

    Returns:
        {pathway_id: {"name": ..., "category": ..., "subcategory": ...}}
    """
    if annot_path is None:
        project_root = _find_project_root()
        annot_path = (project_root / "L4" / "data" / "github_sources" / "CPIExtract"
                      / "KEGG_enrichment" / "cache" / "kegg_pathway_annotation_official.csv")

    if not annot_path.exists():
        logger.warning(f"KEGG 通路注释文件不存在: {annot_path}")
        return {}

    df = pd.read_csv(annot_path)
    annotations: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        hsa_id = str(row.get("hsa_id", "")).strip()
        if not hsa_id:
            continue
        annotations[hsa_id] = {
            "name": str(row.get("Term_clean", row.get("Term_base", ""))).strip(),
            "category": str(row.get("kegg_category", "")).strip(),
            "subcategory": str(row.get("kegg_subcategory", "")).strip(),
        }
    logger.info(f"KEGG 通路注释加载完成: {len(annotations)} 条")
    return annotations


def pathway_enrichment(
    query_genes: list[str],
    background_genes: Optional[list[str]] = None,
    pathway_to_genes: Optional[dict[str, set[str]]] = None,
    pval_threshold: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Fisher 精确检验通路富集分析。

    Args:
        query_genes: 查询基因列表（如预测靶点）
        background_genes: 背景基因列表（默认使用 KEGG 中所有基因）
        pathway_to_genes: 通路→基因映射，默认自动加载
        pval_threshold: 显著性阈值
        method: 多重检验校正方法 ("fdr_bh", "bonferroni", "fdr_by")

    Returns:
        DataFrame: 通路富集结果表，按 adjusted p-value 排序
    """
    if pathway_to_genes is None:
        pathway_to_genes = load_kegg_pathway_genes()

    if not pathway_to_genes:
        logger.error("通路数据为空，无法进行富集分析")
        return pd.DataFrame()

    query_set = {g.upper() for g in query_genes}
    if background_genes is None:
        background_set = set().union(*pathway_to_genes.values())
    else:
        background_set = {g.upper() for g in background_genes}

    query_set = query_set & background_set
    n_query = len(query_set)
    if n_query == 0:
        logger.warning("查询基因与背景基因无交集")
        return pd.DataFrame()

    n_bg = len(background_set)
    results = []

    for pathway_id, pathway_genes in pathway_to_genes.items():
        pathway_genes_in_bg = pathway_genes & background_set
        n_pathway = len(pathway_genes_in_bg)
        if n_pathway == 0:
            continue

        overlap = query_set & pathway_genes_in_bg
        n_overlap = len(overlap)
        if n_overlap == 0:
            continue

        # Fisher 精确检验
        a = n_overlap
        b = n_pathway - n_overlap
        c = n_query - n_overlap
        d = n_bg - n_pathway - c
        if d < 0:
            d = 0
        table = [[a, b], [c, d]]
        _, pval = fisher_exact(table, alternative="greater")

        results.append({
            "pathway_id": pathway_id,
            "n_overlap": n_overlap,
            "n_pathway": n_pathway,
            "n_query": n_query,
            "n_background": n_bg,
            "overlap_genes": ",".join(sorted(overlap)),
            "p_value": pval,
            "fold_enrichment": (n_overlap / n_query) / (n_pathway / n_bg) if n_pathway / n_bg > 0 else 0.0,
        })

    if not results:
        logger.warning("未找到任何富集通路")
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 多重检验校正
    pvals = df["p_value"].values
    _, adjusted, _, _ = multipletests(pvals, alpha=pval_threshold, method=method)
    df["adjusted_p_value"] = adjusted
    df["significant"] = df["adjusted_p_value"] < pval_threshold

    df = df.sort_values("adjusted_p_value")

    # 添加通路注释
    annotations = load_pathway_annotations()
    if annotations:
        df["pathway_name"] = df["pathway_id"].map(
            lambda pid: annotations.get(pid, {}).get("name", "")
        )
        df["pathway_category"] = df["pathway_id"].map(
            lambda pid: annotations.get(pid, {}).get("category", "")
        )
    else:
        df["pathway_name"] = ""
        df["pathway_category"] = ""

    n_sig = df["significant"].sum()
    logger.info(f"通路富集完成: {len(df)} 条通路, {n_sig} 条显著 (p_adj < {pval_threshold})")
    return df


__all__ = [
    "load_kegg_pathway_genes",
    "load_pathway_annotations",
    "pathway_enrichment",
]