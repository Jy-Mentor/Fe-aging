"""生物学解释管道：化合物→靶点→通路→铁衰老机制→CIRI 保护链条。

整合通路富集、铁衰老交集分析，输出从分子到表型的可解释链条。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .iron_aging_overlap import compute_iron_aging_overlap, load_ferroptosis_gene_sets, load_iron_aging_genes
from .pathway_analysis import load_kegg_pathway_genes, load_pathway_annotations, pathway_enrichment

logger = logging.getLogger(__name__)


def explain_predictions(
    prediction_df: pd.DataFrame,
    target_genes: list[str],
    score_threshold: Optional[float] = None,
    top_k_compounds: int = 20,
    top_k_pathways: int = 15,
    pathway_pval_threshold: float = 0.05,
    output_dir: Optional[Path] = None,
) -> dict[str, pd.DataFrame]:
    """生成完整的生物学解释：化合物→靶点→通路→铁衰老→CIRI 保护。

    对预测结果中的 top-k 化合物，分析其靶点与铁衰老基因集和 KEGG 通路的关联。

    Args:
        prediction_df: 预测结果 DataFrame，包含 SMILES 列和基因得分列
        target_genes: 目标基因列表（列名）
        score_threshold: 化合物-基因得分阈值，低于此值不纳入解释
        top_k_compounds: 分析前 k 个化合物
        top_k_pathways: 输出前 k 个显著通路
        pathway_pval_threshold: 通路富集显著性阈值
        output_dir: 输出目录（可选，保存 CSV 文件）

    Returns:
        dict: {
            "compound_targets": 化合物-靶点表,
            "pathway_enrichment": 通路富集结果表,
            "iron_aging_overlap": 铁衰老交集表,
            "explanation_chains": 解释链条表,
        }
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载基础数据
    logger.info("=== 生物学解释管道启动 ===")
    iron_aging_genes = load_iron_aging_genes()
    ferroptosis_sets = load_ferroptosis_gene_sets()
    pathway_to_genes = load_kegg_pathway_genes()
    load_pathway_annotations()

    if not iron_aging_genes:
        logger.warning("铁衰老基因集未加载，交集分析将跳过")
    if not pathway_to_genes:
        logger.warning("KEGG 通路数据未加载，富集分析将跳过")

    # 2. 提取 top-k 化合物
    gene_cols = [g for g in target_genes if g in prediction_df.columns]
    if not gene_cols:
        logger.error(f"预测结果中未找到目标基因列: {target_genes}")
        return {}

    # 计算每个化合物的平均得分
    mean_scores = prediction_df[gene_cols].mean(axis=1)
    prediction_df = prediction_df.copy()
    prediction_df["mean_score"] = mean_scores
    top_df = prediction_df.nlargest(top_k_compounds, "mean_score")

    logger.info(f"Top-{top_k_compounds} 化合物平均得分范围: "
                f"[{top_df['mean_score'].min():.4f}, {top_df['mean_score'].max():.4f}]")

    # 3. 化合物-靶点表
    compound_targets = []
    all_predicted_genes: set[str] = set()
    gene_scores: dict[str, list[float]] = {}

    for _, row in top_df.iterrows():
        mol_id = row.get("MOL_ID", row.get("molecule_name", "unknown"))
        smi = row.get("SMILES", "")
        for gene in gene_cols:
            score = row[gene]
            if score_threshold is not None and score < score_threshold:
                continue
            compound_targets.append({
                "compound_id": mol_id,
                "SMILES": smi,
                "target_gene": gene,
                "prediction_score": score,
            })
            all_predicted_genes.add(gene.upper())
            if gene not in gene_scores:
                gene_scores[gene] = []
            gene_scores[gene].append(score)

    ct_df = pd.DataFrame(compound_targets)
    logger.info(f"化合物-靶点表: {len(ct_df)} 条记录, "
                f"{len(all_predicted_genes)} 个唯一靶点基因")

    # 4. 通路富集分析
    gene_mean_scores = {g: sum(s) / len(s) for g, s in gene_scores.items()}
    sorted_genes = sorted(gene_mean_scores, key=gene_mean_scores.get, reverse=True)

    enrichment_df = pd.DataFrame()
    if pathway_to_genes:
        enrichment_df = pathway_enrichment(
            query_genes=sorted_genes,
            pathway_to_genes=pathway_to_genes,
            pval_threshold=pathway_pval_threshold,
        )
        if not enrichment_df.empty:
            enrichment_df = enrichment_df.head(top_k_pathways)

    # 5. 铁衰老交集分析
    overlap_df = pd.DataFrame()
    if iron_aging_genes:
        overlap_df = compute_iron_aging_overlap(
            predicted_genes=sorted_genes,
            iron_aging_genes=iron_aging_genes,
            ferroptosis_sets=ferroptosis_sets,
            predicted_scores=gene_mean_scores,
        )

    # 6. 构建解释链条
    chains = _build_explanation_chains(
        ct_df=ct_df,
        enrichment_df=enrichment_df,
        overlap_df=overlap_df,
        iron_aging_genes=iron_aging_genes,
    )

    # 7. 输出
    if output_dir is not None:
        ct_df.to_csv(output_dir / "compound_targets.csv", index=False)
        if not enrichment_df.empty:
            enrichment_df.to_csv(output_dir / "pathway_enrichment.csv", index=False)
        if not overlap_df.empty:
            overlap_df.to_csv(output_dir / "iron_aging_overlap.csv", index=False)
        if not chains.empty:
            chains.to_csv(output_dir / "explanation_chains.csv", index=False)
        logger.info(f"解释结果已保存至: {output_dir}")

    return {
        "compound_targets": ct_df,
        "pathway_enrichment": enrichment_df,
        "iron_aging_overlap": overlap_df,
        "explanation_chains": chains,
    }


def _build_explanation_chains(
    ct_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    iron_aging_genes: set[str],
) -> pd.DataFrame:
    """构建化合物→靶点→通路→铁衰老→CIRI 解释链条。

    Args:
        ct_df: 化合物-靶点表
        enrichment_df: 通路富集结果表
        overlap_df: 铁衰老交集表
        iron_aging_genes: 铁衰老基因集

    Returns:
        DataFrame: 解释链条
    """
    chains = []

    # 构建基因→通路映射
    gene_to_pathways: dict[str, list[str]] = {}
    if not enrichment_df.empty and "overlap_genes" in enrichment_df.columns:
        for _, row in enrichment_df.iterrows():
            if not row.get("significant", False):
                continue
            pid = row["pathway_id"]
            pname = row.get("pathway_name", pid)
            genes = str(row["overlap_genes"]).split(",")
            for g in genes:
                g = g.strip().upper()
                if g not in gene_to_pathways:
                    gene_to_pathways[g] = []
                gene_to_pathways[g].append(f"{pname} ({pid})")

    # 铁衰老基因
    ia_genes = iron_aging_genes if iron_aging_genes else set()

    # 铁死亡分类
    gene_to_ferroptosis: dict[str, str] = {}
    if not overlap_df.empty:
        for _, row in overlap_df.iterrows():
            gene = row["gene"].upper()
            if row["gene_set"] != "iron_aging_96":
                gene_to_ferroptosis[gene] = row["category"]

    # 构建链条
    for _, row in ct_df.iterrows():
        gene = row["target_gene"].upper()
        compound = row["compound_id"]
        score = row["prediction_score"]

        is_ia = gene in ia_genes
        ferro_cat = gene_to_ferroptosis.get(gene, "")
        pathways = gene_to_pathways.get(gene, [])

        if not pathways:
            if is_ia or ferro_cat:
                chains.append({
                    "compound": compound,
                    "SMILES": row["SMILES"],
                    "target_gene": gene,
                    "prediction_score": score,
                    "is_iron_aging": is_ia,
                    "is_ferroptosis_related": bool(ferro_cat),
                    "ferroptosis_category": ferro_cat,
                    "enriched_pathways": "",
                    "explanation": _format_explanation(gene, is_ia, ferro_cat, []),
                })
            continue

        for p in pathways:
            chains.append({
                "compound": compound,
                "SMILES": row["SMILES"],
                "target_gene": gene,
                "prediction_score": score,
                "is_iron_aging": is_ia,
                "is_ferroptosis_related": bool(ferro_cat),
                "ferroptosis_category": ferro_cat,
                "enriched_pathways": p,
                "explanation": _format_explanation(gene, is_ia, ferro_cat, [p]),
            })

    if not chains:
        return pd.DataFrame()

    return pd.DataFrame(chains)


def _format_explanation(
    gene: str,
    is_iron_aging: bool,
    ferroptosis_category: str,
    pathways: list[str],
) -> str:
    """格式化单条解释文本。

    输出格式: "{gene} → [铁衰老核心基因/铁死亡XX基因] → {通路1, 通路2} → CIRI保护"
    """
    parts = [gene]

    if is_iron_aging:
        parts.append("铁衰老核心基因")
    if ferroptosis_category:
        parts.append(ferroptosis_category)

    if pathways:
        parts.append("→".join(pathways))

    parts.append("→ CIRI 保护")
    return " | ".join(parts)


__all__ = [
    "explain_predictions",
]