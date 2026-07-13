#!/usr/bin/env python3
"""
Rank-sum 排名融合排序（主方案）— 替代 TOPSIS 和原加权线性组合
=============================================================
方法：
  1. 构建决策矩阵 (quality, coverage, network)
     - quality = 0.5 * avg_score + 0.5 * (max_score ** 2)  [非线性变换，增强高置信优势]
     - coverage = n_strong / n_tasks  [强相互作用: >0.8, Huang et al. 2025]
     - network = sum(score * hub_weight[gene]) / max_network_score
  2. 主方案：Rank-sum 对每个准则单独排名后加总（越小越好）
  3. 对比方案：TOPSIS（熵权法）、Borda（等权排名融合）
  4. 敏感性分析：权重扰动 + 多方法交叉验证

关键参考文献：
  - Rank-sum / Borda: de Borda (1781); Chang et al. (2013) Front Genet
  - Consensus scoring: Moshawih et al. (2024) J Cheminform 16:62
  - 强相互作用阈值: Huang et al. (2025), 0.8 阈值
  - TOPSIS: Hwang & Yoon (1981) "Multiple Attribute Decision Making", Springer
  - 熵权法: Shannon (1948) Bell System Technical Journal 27:379-423

输出：
  L4/results_v6_toxfiltered/tcm_top_candidates_topsis.csv
  L4/results_v6_toxfiltered/topsis_sensitivity_analysis.json
  L4/results_v6_toxfiltered/topsis_vs_original_comparison.csv
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v6_toxfiltered"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"

L4_LOGS.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "rank_topsis.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据加载
# ============================================================

def load_predictions() -> pd.DataFrame:
    """加载 GAT+HGT 预测结果"""
    path = L4_RESULTS / "tcm_predictions_full_v6_tox.csv"
    if not path.exists():
        logger.error(f"预测文件不存在: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"加载预测: {len(df)} 化合物, {df.shape[1]} 列")
    return df


def load_ppi_hub_weights() -> dict[str, float]:
    """加载扩展 PPI hub 权重 (Degree Centrality)

    优先使用 L1/results/ppi_network_extended_hub_genes.csv（7225 基因，铁衰老种子
    在 STRING PPI 中扩展 1-2 层邻居）；回退到 ppi_hub_genes.csv（28 基因，旧版）。
    """
    extended_path = L1_RESULTS / "ppi_network_extended_hub_genes.csv"
    path = extended_path if extended_path.exists() else (L1_RESULTS / "ppi_hub_genes.csv")
    if not path.exists():
        logger.warning(f"PPI hub文件不存在: {path}, 使用默认权重")
        return {}

    df = pd.read_csv(path)
    weights = {}
    max_degree = df["Degree_Centrality"].max()
    for _, row in df.iterrows():
        gene = str(row["Gene"]).strip().upper()
        w = float(row["Degree_Centrality"]) / max_degree if max_degree > 0 else 0.0
        weights[gene] = w

    logger.info(f"PPI hub权重: {len(weights)} 个基因, "
                f"range=[{min(weights.values()):.3f}, {max(weights.values()):.3f}]")
    return weights


# ============================================================
# 决策矩阵构建
# ============================================================

def build_decision_matrix(
    pred_df: pd.DataFrame,
    target_genes: list[str],
    hub_weights: dict[str, float],
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    构建决策矩阵

    准则:
      1. quality = 0.5 * avg_score + 0.5 * (max_score ** 2)  [非线性变换]
         文献依据: 平方变换增强高置信单靶标活性优势，
         类似虚拟筛选中对高docking score的指数加权策略
      2. coverage = n_strong / n_tasks  [强相互作用: >0.8]
         文献依据: Huang et al. (2025), 0.8 为强相互作用阈值
      3. network = sum(score * hub_weight[gene]) / max_network_score
         文献依据: 网络药理学 hub 基因中心性加权
    """
    gene_cols = [g for g in target_genes if g in pred_df.columns]
    scores = pred_df[gene_cols].values
    n_tasks = len(gene_cols)

    logger.info(f"决策矩阵: {len(pred_df)} 化合物 × {n_tasks} 靶标")

    # 准则1: quality — 非线性变换 (max_score² 增强高置信优势)
    avg_score = np.nanmean(scores, axis=1)
    max_score = np.nanmax(scores, axis=1)
    quality = 0.5 * avg_score + 0.5 * (max_score ** 2)

    # 准则2: coverage (强相互作用: >0.8, Huang et al. 2025)
    n_strong = np.nansum(scores > 0.8, axis=1)
    coverage = n_strong / n_tasks

    # 准则3: network (hub基因中心性加权)
    network = np.zeros(len(pred_df))
    for i in range(len(pred_df)):
        net_sum = 0.0
        for j, gene in enumerate(gene_cols):
            w = hub_weights.get(gene, 0.0)
            s = scores[i, j]
            if not np.isnan(s):
                net_sum += s * w
        network[i] = net_sum

    max_network = network.max()
    if max_network > 0:
        network = network / max_network

    # 记录中间值
    info_df = pd.DataFrame({
        "quality": quality,
        "coverage": coverage,
        "network": network,
        "avg_score": avg_score,
        "max_score": max_score,
        "n_strong": n_strong,
    })

    criteria_names = ["quality", "coverage", "network"]

    # 构建决策矩阵
    matrix = np.column_stack([quality, coverage, network])

    return matrix, criteria_names, info_df


# ============================================================
# 排名方法
# ============================================================

def rank_sum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank-sum 排名融合（主方案）

    对每个准则单独排名，加总排名得分。
    分数越低越好（排名越靠前）。

    文献依据:
      - de Borda (1781)
      - Chang et al. (2013) Front Genet: RankSum 在meta-analysis中表现优异
      - Moshawih et al. (2024) J Cheminform: 共识评分优于单一方法

    返回:
      rank_sum_scores: (n_samples,) rank-sum得分
      rank_quality: (n_samples,) quality准则排名
      rank_coverage: (n_samples,) coverage准则排名
      rank_network: (n_samples,) network准则排名
    """
    n, m = matrix.shape
    ranks = np.zeros((n, m))

    for j in range(m):
        # 降序排列（所有准则都是越大越好）
        order = np.argsort(-matrix[:, j])
        for rank, idx in enumerate(order):
            ranks[idx, j] = rank + 1  # 1-based rank

    rank_sum_scores = ranks.sum(axis=1)
    return rank_sum_scores, ranks[:, 0], ranks[:, 1], ranks[:, 2]


def weighted_rank_sum(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    加权 Rank-sum 排名融合

    对每个准则排名后，按权重加权求和。

    参数:
      matrix: (n_samples, n_criteria)
      weights: (n_criteria,) 权重向量

    返回:
      weighted_rank_sum: (n_samples,)
    """
    n, m = matrix.shape
    ranks = np.zeros((n, m))

    for j in range(m):
        order = np.argsort(-matrix[:, j])
        for rank, idx in enumerate(order):
            ranks[idx, j] = rank + 1

    weighted = ranks @ weights  # 加权求和
    return weighted


def entropy_weight(matrix: np.ndarray) -> np.ndarray:
    """
    熵权法计算客观权重

    参数:
      matrix: (n_samples, n_criteria) 决策矩阵 (已正向化)

    返回:
      weights: (n_criteria,) 权重向量
    """
    n, m = matrix.shape

    for j in range(m):
        if matrix[:, j].min() < 0:
            matrix[:, j] -= matrix[:, j].min()

    col_sum = matrix.sum(axis=0)
    col_sum[col_sum == 0] = 1e-10
    p = matrix / col_sum

    e = np.zeros(m)
    for j in range(m):
        pj = p[:, j]
        pj = pj[pj > 0]
        if len(pj) == 0:
            e[j] = 1.0
        else:
            e[j] = -np.sum(pj * np.log(pj)) / np.log(n)

    d = 1.0 - e
    if d.sum() == 0:
        return np.ones(m) / m
    w = d / d.sum()

    return w


def topsis(matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """
    TOPSIS 多准则决策（对比方案）

    参数:
      matrix: (n_samples, n_criteria)
      weights: (n_criteria,) 权重向量 (None则使用熵权)

    返回:
      closeness: (n_samples,) 相对贴近度 [0, 1]
    """
    n, m = matrix.shape

    if weights is None:
        weights = entropy_weight(matrix.copy())

    norm = np.sqrt(np.sum(matrix ** 2, axis=0))
    norm[norm == 0] = 1e-10
    r = matrix / norm

    v = r * weights

    a_plus = np.max(v, axis=0)
    a_minus = np.min(v, axis=0)

    s_plus = np.sqrt(np.sum((v - a_plus) ** 2, axis=1))
    s_minus = np.sqrt(np.sum((v - a_minus) ** 2, axis=1))

    denom = s_plus + s_minus
    denom[denom == 0] = 1e-10
    closeness = s_minus / denom

    return closeness


def borda_rank(matrix: np.ndarray) -> np.ndarray:
    """
    Borda 排名融合（对比方案，等价于等权 Rank-sum）

    返回:
      borda_scores: (n_samples,) Borda得分
    """
    scores, _, _, _ = rank_sum(matrix)
    return scores


# ============================================================
# 敏感性分析
# ============================================================

def sensitivity_analysis_rank_sum(
    matrix: np.ndarray,
    criteria_names: list[str],
    compounds: list[str],
    n_top: int = 50,
    n_perturbations: int = 100,
) -> dict:
    """
    敏感性分析（针对 Rank-sum 主方案）

    策略:
      1. 加权 Rank-sum 扰动: 随机生成权重, 检查排名稳定性
      2. 评分扰动: 对决策矩阵值加噪声, 检查排名变化
      3. 多方法交叉验证: 比较 Rank-sum vs TOPSIS vs Borda

    返回:
      sensitivity_results: Dict
    """
    np.random.seed(42)
    n, m = matrix.shape

    # === 1. 加权 Rank-sum 扰动 ===
    base_rank_sum, _, _, _ = rank_sum(matrix)
    base_order = np.argsort(base_rank_sum)
    base_top50 = set(base_order[:n_top])

    jaccard_weighted = []
    rank_variations = {i: [] for i in range(n)}

    for _ in range(n_perturbations):
        # 随机生成权重 (Dirichlet分布)
        w = np.random.dirichlet(np.ones(m) * 3)
        wr = weighted_rank_sum(matrix, w)
        order = np.argsort(wr)
        top50 = set(order[:n_top])

        jac = len(top50 & base_top50) / len(top50 | base_top50) if (top50 | base_top50) else 0
        jaccard_weighted.append(jac)

        for rank, idx in enumerate(order):
            rank_variations[idx].append(rank + 1)

    # === 2. 评分扰动 (决策矩阵值加噪声) ===
    jaccard_noise = []
    noise_rank_variations = {i: [] for i in range(n)}

    for _ in range(n_perturbations):
        # 对决策矩阵加 ±5% 噪声
        noise = np.random.uniform(-0.05, 0.05, matrix.shape)
        noisy_matrix = np.clip(matrix + noise * matrix.std(axis=0), 0, None)
        rs, _, _, _ = rank_sum(noisy_matrix)
        order = np.argsort(rs)
        top50 = set(order[:n_top])

        jac = len(top50 & base_top50) / len(top50 | base_top50) if (top50 | base_top50) else 0
        jaccard_noise.append(jac)

        for rank, idx in enumerate(order):
            noise_rank_variations[idx].append(rank + 1)

    # === 3. 多方法交叉验证 ===
    # TOPSIS
    ew = entropy_weight(matrix.copy())
    closeness = topsis(matrix, ew)
    topsis_order = set(np.argsort(-closeness)[:n_top])

    # Borda (等权rank-sum)
    borda = borda_rank(matrix)
    borda_order = set(np.argsort(borda)[:n_top])

    # 交叉重叠
    jac_rs_topsis = len(base_top50 & topsis_order) / len(base_top50 | topsis_order)
    jac_rs_borda = len(base_top50 & borda_order) / len(base_top50 | borda_order)
    jac_topsis_borda = len(topsis_order & borda_order) / len(topsis_order | borda_order)

    # === 4. 稳定性指标 ===
    top10_stable = set()
    top20_stable = set()
    top50_stable = set()

    for idx in range(n):
        ranks = np.array(rank_variations[idx])
        if np.all(ranks <= 10):
            top10_stable.add(idx)
        if np.all(ranks <= 20):
            top20_stable.add(idx)
        if np.all(ranks <= 50):
            top50_stable.add(idx)

    # 石竹烯排名变化
    bcp_ranks = []
    for i, comp in enumerate(compounds):
        if comp in ["MOL_BCP", "beta-caryophyllene", "Caryophyllene"]:
            bcp_ranks = rank_variations[i]
            break

    results = {
        "method": "rank_sum",
        "n_perturbations": n_perturbations,
        "criteria": criteria_names,
        # 加权扰动
        "weighted_perturbation": {
            "jaccard_mean": float(np.mean(jaccard_weighted)),
            "jaccard_std": float(np.std(jaccard_weighted)),
            "jaccard_min": float(np.min(jaccard_weighted)),
            "jaccard_max": float(np.max(jaccard_weighted)),
        },
        # 评分噪声扰动
        "noise_perturbation": {
            "jaccard_mean": float(np.mean(jaccard_noise)),
            "jaccard_std": float(np.std(jaccard_noise)),
            "jaccard_min": float(np.min(jaccard_noise)),
            "jaccard_max": float(np.max(jaccard_noise)),
        },
        # 多方法交叉验证
        "cross_validation": {
            "rank_sum_vs_topsis_jaccard": float(jac_rs_topsis),
            "rank_sum_vs_borda_jaccard": float(jac_rs_borda),
            "topsis_vs_borda_jaccard": float(jac_topsis_borda),
        },
        # 稳定性
        "n_stable_top10": len(top10_stable),
        "n_stable_top20": len(top20_stable),
        "n_stable_top50": len(top50_stable),
        # 熵权法权重
        "entropy_weights": {name: float(w) for name, w in zip(criteria_names, ew, strict=False)},
    }

    if bcp_ranks:
        results["bcp_rank"] = {
            "min": int(np.min(bcp_ranks)),
            "max": int(np.max(bcp_ranks)),
            "mean": float(np.mean(bcp_ranks)),
            "std": float(np.std(bcp_ranks)),
        }

    return results


# ============================================================
# 对比分析
# ============================================================

def compare_methods(
    result_df: pd.DataFrame,
    original_df: pd.DataFrame,
    top_n: int = 100,
) -> pd.DataFrame:
    """
    对比多种排名方法: Rank-sum vs TOPSIS vs 原始排名
    """
    old_cols = ["MOL_ID", "molecule_name", "composite_score", "rank"]
    if not all(c in original_df.columns for c in old_cols):
        logger.warning("原始排名列缺失，跳过对比")
        return pd.DataFrame()

    old_df = original_df[old_cols].copy()
    old_df = old_df.rename(columns={
        "composite_score": "old_composite_score",
        "rank": "old_rank",
    })

    new_df = result_df[[
        "MOL_ID", "molecule_name",
        "rank_sum", "rank_sum_rank",
        "topsis_score", "topsis_rank",
        "borda_score", "borda_rank",
    ]].copy()

    merged = new_df.merge(old_df, on=["MOL_ID", "molecule_name"], how="inner")
    merged["rank_change_vs_old"] = merged["old_rank"] - merged["rank_sum_rank"]
    merged["rank_change_vs_topsis"] = merged["topsis_rank"] - merged["rank_sum_rank"]

    merged = merged.sort_values("rank_sum_rank").head(top_n)
    return merged


# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Rank-sum 排名融合排序（主方案）")
    logger.info("=" * 60)

    # 1. 加载数据
    pred_df = load_predictions()
    hub_weights = load_ppi_hub_weights()

    # 确定靶标列
    gene_cols = [c for c in pred_df.columns
                 if c not in ["rank", "MOL_ID", "molecule_name", "SMILES_std", "SMILES",
                              "composite_score", "avg_score", "max_score", "n_hits",
                              "n_high", "consistency", "n_targets", "top_targets"]]
    logger.info(f"靶标列: {len(gene_cols)} 个 — {gene_cols[:10]}...")

    # 2. 构建决策矩阵 (非线性 quality)
    matrix, criteria_names, info_df = build_decision_matrix(
        pred_df, gene_cols, hub_weights
    )
    logger.info(f"决策矩阵: {matrix.shape}")
    logger.info(f"准则: {criteria_names}")
    for j, name in enumerate(criteria_names):
        col = matrix[:, j]
        logger.info(f"  {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
                     f"min={col.min():.4f}, max={col.max():.4f}")

    # 3. 主方案: Rank-sum 排名融合
    rank_sum_scores, rank_quality, rank_coverage, rank_network = rank_sum(matrix)
    logger.info(f"Rank-sum得分: mean={rank_sum_scores.mean():.1f}, "
                 f"std={rank_sum_scores.std():.1f}, "
                 f"min={rank_sum_scores.min():.0f}, max={rank_sum_scores.max():.0f}")

    # 4. 对比方案: TOPSIS (熵权法)
    ew = entropy_weight(matrix.copy())
    logger.info(f"熵权法权重: {dict(zip(criteria_names, ew.round(4), strict=False))}")
    closeness = topsis(matrix, ew)
    logger.info(f"TOPSIS贴近度: mean={closeness.mean():.4f}, "
                 f"std={closeness.std():.4f}")

    # 5. 对比方案: Borda (等权rank-sum)
    borda_scores = borda_rank(matrix)
    logger.info(f"Borda得分: mean={borda_scores.mean():.1f}, std={borda_scores.std():.1f}")

    # 6. 构建输出 DataFrame
    result_df = pred_df[["MOL_ID", "molecule_name", "SMILES"]].copy()

    # 基础指标
    result_df["avg_score"] = info_df["avg_score"]
    result_df["max_score"] = info_df["max_score"]
    result_df["n_strong"] = info_df["n_strong"].astype(int)
    result_df["n_targets"] = len(gene_cols)

    # 决策准则
    result_df["quality"] = info_df["quality"]
    result_df["coverage"] = info_df["coverage"]
    result_df["network"] = info_df["network"]

    # 排名列
    result_df["rank_quality"] = rank_quality.astype(int)
    result_df["rank_coverage"] = rank_coverage.astype(int)
    result_df["rank_network"] = rank_network.astype(int)

    # 主方案: Rank-sum
    result_df["rank_sum"] = rank_sum_scores
    result_df["topsis_score"] = closeness
    result_df["borda_score"] = borda_scores

    # === 排名计算 ===
    # Rank-sum 排名 (主方案)
    rs_order = np.argsort(rank_sum_scores)
    rs_rank_arr = np.zeros(len(result_df), dtype=int)
    for rank, idx in enumerate(rs_order):
        rs_rank_arr[idx] = rank + 1
    result_df["rank_sum_rank"] = rs_rank_arr

    # TOPSIS 排名
    topsis_order = np.argsort(-closeness)
    topsis_rank_arr = np.zeros(len(result_df), dtype=int)
    for rank, idx in enumerate(topsis_order):
        topsis_rank_arr[idx] = rank + 1
    result_df["topsis_rank"] = topsis_rank_arr

    # Borda 排名
    borda_order = np.argsort(borda_scores)
    borda_rank_arr = np.zeros(len(result_df), dtype=int)
    for rank, idx in enumerate(borda_order):
        borda_rank_arr[idx] = rank + 1
    result_df["borda_rank"] = borda_rank_arr

    # 使用 Rank-sum 作为主排名
    result_df = result_df.sort_values("rank_sum_rank").reset_index(drop=True)
    result_df["rank"] = range(1, len(result_df) + 1)

    # top_targets
    gene_cols_in_df = [g for g in gene_cols if g in pred_df.columns]
    scores = pred_df[gene_cols_in_df].values
    top_targets_list = []
    for i in range(len(result_df)):
        mol_id = result_df.loc[i, "MOL_ID"]
        orig_idx = pred_df[pred_df["MOL_ID"] == mol_id].index[0]
        gene_scores = [(g, scores[orig_idx][j]) for j, g in enumerate(gene_cols_in_df)]
        gene_scores.sort(key=lambda x: x[1], reverse=True)
        top5 = gene_scores[:5]
        top_targets_list.append(", ".join([f"{g}({s:.3f})" for g, s in top5]))
    result_df["top_targets"] = top_targets_list

    # Top 500
    top_df = result_df.head(500).copy()

    # 7. 保存结果
    result_df.to_csv(L4_RESULTS / "tcm_predictions_full_topsis.csv", index=False)
    top_df.to_csv(L4_RESULTS / "tcm_top_candidates_topsis.csv", index=False)
    logger.info(f"全量预测: {L4_RESULTS / 'tcm_predictions_full_topsis.csv'}")
    logger.info(f"Top 500: {L4_RESULTS / 'tcm_top_candidates_topsis.csv'}")

    # 8. 敏感性分析
    compounds = result_df["MOL_ID"].tolist()
    sensitivity = sensitivity_analysis_rank_sum(matrix, criteria_names, compounds)
    with open(L4_RESULTS / "topsis_sensitivity_analysis.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)
    logger.info(f"敏感性分析: {L4_RESULTS / 'topsis_sensitivity_analysis.json'}")

    # 9. 多方法对比
    comparison = compare_methods(result_df, pred_df)
    if not comparison.empty:
        comparison.to_csv(L4_RESULTS / "topsis_vs_original_comparison.csv", index=False)
        logger.info(f"多方法对比: {L4_RESULTS / 'topsis_vs_original_comparison.csv'}")

    # 10. 打印摘要
    logger.info("=" * 60)
    logger.info("Rank-sum Top 10:")
    for i in range(min(10, len(top_df))):
        row = top_df.iloc[i]
        logger.info(f"  {row['rank']:3d}. {row['MOL_ID']:12s} {row['molecule_name'][:40]:40s} "
                     f"rank_sum={row['rank_sum']:.0f} "
                     f"Q={row['quality']:.4f} C={row['coverage']:.4f} N={row['network']:.4f}")

    # 石竹烯
    if "MOL_BCP" in result_df["MOL_ID"].values:
        bcp = result_df[result_df["MOL_ID"] == "MOL_BCP"].iloc[0]
        logger.info("---")
        logger.info("beta-caryophyllene (MOL_BCP):")
        logger.info(f"  Rank-sum排名: {bcp['rank_sum_rank']}/{len(result_df)}")
        logger.info(f"  rank_sum={bcp['rank_sum']:.0f} "
                     f"(rank_Q={bcp['rank_quality']:.0f}, "
                     f"rank_C={bcp['rank_coverage']:.0f}, "
                     f"rank_N={bcp['rank_network']:.0f})")
        logger.info(f"  quality={bcp['quality']:.4f} coverage={bcp['coverage']:.4f} network={bcp['network']:.4f}")
        logger.info(f"  avg_score={bcp['avg_score']:.4f} max_score={bcp['max_score']:.4f} n_strong={bcp['n_strong']}")
        logger.info(f"  TOPSIS排名: {bcp['topsis_rank']}/{len(result_df)}")
        logger.info(f"  Borda排名: {bcp['borda_rank']}/{len(result_df)}")
        if "composite_score" in pred_df.columns:
            old_rank = pred_df[pred_df["MOL_ID"] == "MOL_BCP"]["rank"].values[0]
            logger.info(f"  原始排名: {old_rank}/{len(pred_df)}")
        logger.info(f"  Top靶标: {bcp['top_targets']}")

    logger.info("---")
    logger.info("敏感性分析:")
    logger.info(f"  加权扰动 Jaccard: mean={sensitivity['weighted_perturbation']['jaccard_mean']:.4f} "
                 f"± {sensitivity['weighted_perturbation']['jaccard_std']:.4f}")
    logger.info(f"  噪声扰动 Jaccard: mean={sensitivity['noise_perturbation']['jaccard_mean']:.4f} "
                 f"± {sensitivity['noise_perturbation']['jaccard_std']:.4f}")
    logger.info(f"  Rank-sum vs TOPSIS Jaccard: {sensitivity['cross_validation']['rank_sum_vs_topsis_jaccard']:.4f}")
    logger.info(f"  Rank-sum vs Borda Jaccard: {sensitivity['cross_validation']['rank_sum_vs_borda_jaccard']:.4f}")
    logger.info(f"  Top10稳定: {sensitivity['n_stable_top10']} 个化合物")
    logger.info(f"  Top50稳定: {sensitivity['n_stable_top50']} 个化合物")
    if "bcp_rank" in sensitivity:
        bcp_r = sensitivity["bcp_rank"]
        logger.info(f"  BCP排名范围: [{bcp_r['min']}, {bcp_r['max']}], mean={bcp_r['mean']:.1f} ± {bcp_r['std']:.1f}")

    logger.info("=" * 60)
    logger.info("完成")


if __name__ == "__main__":
    main()