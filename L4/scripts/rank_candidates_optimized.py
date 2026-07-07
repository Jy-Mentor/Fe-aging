import logging
logger = logging.getLogger(__name__)

"""
优化候选化合物排序脚本

目标：
1. 解决当前综合评分偏向“靶标数量多”的问题；
2. 引入铁衰老 PPI hub 基因网络中心性；
3. 用熵权法客观确定权重；
4. 特别提升石竹烯（beta-caryophyllene, MOL_BCP）等“精准少靶标”化合物的排名。

输入：
- L4/results_v6/tcm_predictions_full_v6.csv（v6 完整预测结果）
- L1/results/ppi_hub_genes.csv（铁衰老 PPI hub 基因中心性）

输出：
- L4/results_v6_optimized/tcm_predictions_full_optimized.csv
- L4/results_v6_optimized/tcm_top_candidates_optimized.csv
- L4/results_v6_optimized/ranking_comparison_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path("d:/铁衰老 绝不重蹈覆辙")
INPUT_PRED = BASE_DIR / "L4/results_v6/tcm_predictions_full_v6.csv"
INPUT_HUB = BASE_DIR / "L1/results/ppi_hub_genes.csv"
OUTPUT_DIR = BASE_DIR / "L4/results_v6_optimized"


def load_hub_genes(path: Path) -> tuple[set[str], dict[str, float]]:
    """加载 PPI hub 基因，返回基因集合与中心性权重。"""
    hub_df = pd.read_csv(path)
    hub_df['Gene'] = hub_df['Gene'].astype(str).str.strip()
    # 使用 Hub_Rank 的倒数作为中心性权重；Hub_Rank 越小越核心
    hub_weight = dict(zip(hub_df['Gene'], 1.0 / hub_df['Hub_Rank'].astype(float), strict=False))
    return set(hub_df['Gene']), hub_weight


def compute_quality_score(scores: np.ndarray) -> float:
    """质量维度：平均活性 + 最大活性的非线性奖励。"""
    return 0.7 * float(scores.mean()) + 0.3 * float(scores.max())


def compute_coverage_score(scores: np.ndarray, threshold: float = 0.7, cap: int = 5) -> float:
    """精准覆盖维度：高置信命中数，cap 到 5，避免盲目奖励多靶标。"""
    n_high = int((scores > threshold).sum())
    return min(n_high, cap) / cap


def compute_network_score(
    scores: np.ndarray,
    target_cols: list[str],
    hub_weight: dict[str, float],
    max_possible: float,
) -> float:
    """网络中心性维度：命中 hub 基因的加权活性得分。"""
    weighted_sum = sum(
        scores[i] * hub_weight.get(target_cols[i], 0.0)
        for i in range(len(target_cols))
        if target_cols[i] in hub_weight
    )
    return weighted_sum / max_possible


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """使用熵权法客观计算三维指标权重。"""
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    p = X_norm / (X_norm.sum(axis=0) + 1e-10)
    e = -np.sum(p * np.log(p + 1e-10), axis=0) / np.log(len(X) + 1e-10)
    weights = (1.0 - e) / (1.0 - e).sum()
    return weights


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    df = pd.read_csv(INPUT_PRED)
    hub_genes, hub_weight = load_hub_genes(INPUT_HUB)

    # 2. 识别靶标列
    meta_cols = {
        'rank', 'MOL_ID', 'molecule_name', 'SMILES', 'avg_score', 'max_score',
        'n_hits', 'n_high', 'n_targets', 'consistency', 'composite_score', 'top_targets'
    }
    target_cols = [c for c in df.columns if c not in meta_cols]

    print(f"化合物数: {len(df)}")
    print(f"靶标列数: {len(target_cols)}")
    print(f"Hub 基因数: {len(hub_genes)}")
    print(f"靶标列中属于 hub 的基因: {len(set(target_cols) & hub_genes)}")

    # 3. 计算网络分母（理论最大加权得分）
    max_hub_score = sum(hub_weight.get(t, 0.0) for t in target_cols)
    print(f"网络维度最大可能得分: {max_hub_score:.4f}")

    # 4. 计算三维指标
    quality_list = []
    coverage_list = []
    network_list = []

    for idx, row in df.iterrows():
        scores = np.array([row[t] for t in target_cols])
        quality_list.append(compute_quality_score(scores))
        coverage_list.append(compute_coverage_score(scores))
        network_list.append(compute_network_score(scores, target_cols, hub_weight, max_hub_score))

    df['quality_score'] = quality_list
    df['coverage_score'] = coverage_list
    df['network_score'] = network_list

    # 5. 熵权法确定权重
    X = df[['quality_score', 'coverage_score', 'network_score']].values
    weights = entropy_weights(X)
    print(f"\n熵权法权重:")
    print(f"  quality_score:  {weights[0]:.4f}")
    print(f"  coverage_score: {weights[1]:.4f}")
    print(f"  network_score:  {weights[2]:.4f}")

    # 6. 计算优化后综合得分
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
    df['composite_score_optimized'] = (X_norm * weights).sum(axis=1)

    # 7. 重新排序
    df_old = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
    df_old['rank_old'] = np.arange(1, len(df_old) + 1)

    df_new = df.sort_values('composite_score_optimized', ascending=False).reset_index(drop=True)
    df_new['rank_new'] = np.arange(1, len(df_new) + 1)

    # 合并新旧排名
    rank_map_old = dict(zip(df_old['MOL_ID'], df_old['rank_old'], strict=False))
    rank_map_new = dict(zip(df_new['MOL_ID'], df_new['rank_new'], strict=False))

    df['rank_old'] = df['MOL_ID'].map(rank_map_old)
    df['rank_new'] = df['MOL_ID'].map(rank_map_new)
    df['rank_change'] = df['rank_old'] - df['rank_new']  # 正值表示排名上升

    # 8. 输出完整结果
    output_full = OUTPUT_DIR / 'tcm_predictions_full_optimized.csv'
    df.to_csv(output_full, index=False)
    print(f"\n完整预测结果已保存: {output_full}")

    # 9. 输出 Top 候选
    top_cols = [
        'rank_new', 'MOL_ID', 'molecule_name', 'composite_score_optimized',
        'quality_score', 'coverage_score', 'network_score',
        'avg_score', 'max_score', 'n_hits', 'n_high', 'n_targets',
        'top_targets', 'rank_old', 'rank_change'
    ]
    top_df = df_new[top_cols].copy()
    top_df.rename(columns={'rank_new': 'rank'}, inplace=True)
    output_top = OUTPUT_DIR / 'tcm_top_candidates_optimized.csv'
    top_df.head(50).to_csv(output_top, index=False)
    print(f"Top50 候选已保存: {output_top}")

    # 10. 特别关注石竹烯
    bcp = df[df['MOL_ID'] == 'MOL_BCP']
    print("\n" + "=" * 60)
    print("石竹烯（beta-caryophyllene）排名对比")
    print("=" * 60)
    if not bcp.empty:
        b = bcp.iloc[0]
        print(f"  旧排名: {int(b['rank_old'])}")
        print(f"  新排名: {int(b['rank_new'])}")
        print(f"  排名上升: {int(b['rank_change'])} 位")
        print(f"  旧综合得分: {b['composite_score']:.4f}")
        print(f"  新综合得分: {b['composite_score_optimized']:.4f}")
        print(f"  quality_score:  {b['quality_score']:.4f}")
        print(f"  coverage_score: {b['coverage_score']:.4f}")
        print(f"  network_score:  {b['network_score']:.4f}")
    else:
        print("  未在预测结果中找到 MOL_BCP")

    # 11. 生成对比报告
    report_lines = [
        "# 候选化合物排序优化报告",
        "",
        "## 1. 优化目标",
        "- 解决原评分公式偏向多靶标化合物的问题；",
        "- 引入铁衰老 PPI 网络 hub 基因中心性；",
        "- 使用熵权法客观确定权重；",
        "- 提升精准少靶标化合物（如石竹烯）的排名。",
        "",
        "## 2. 三维评分指标",
        "",
        "| 维度 | 说明 | 计算方式 |",
        "|------|------|----------|",
        "| quality_score | 整体活性质量 | 0.7 * avg_score + 0.3 * max_score |",
        "| coverage_score | 高置信靶标精准覆盖 | min(n_high, 5) / 5 |",
        "| network_score | 铁衰老 hub 基因命中加权得分 | sum(score_i * hub_weight_i) / max_possible |",
        "",
        "## 3. 熵权法权重",
        "",
        f"- quality_score:  {weights[0]:.4f}",
        f"- coverage_score: {weights[1]:.4f}",
        f"- network_score:  {weights[2]:.4f}",
        "",
        "## 4. 石竹烯排名变化",
        "",
    ]

    if not bcp.empty:
        b = bcp.iloc[0]
        report_lines.extend([
            f"- 旧排名: {int(b['rank_old'])}",
            f"- 新排名: {int(b['rank_new'])}",
            f"- 排名上升: {int(b['rank_change'])} 位",
            f"- 旧综合得分: {b['composite_score']:.4f}",
            f"- 新综合得分: {b['composite_score_optimized']:.4f}",
            f"- quality_score: {b['quality_score']:.4f}",
            f"- coverage_score: {b['coverage_score']:.4f}",
            f"- network_score: {b['network_score']:.4f}",
            "",
            "## 5. 石竹烯命中 hub 基因",
            "",
        ])
        for t in target_cols:
            if t in hub_weight and b[t] > 0.5:
                report_lines.append(f"- {t}: 预测概率 {b[t]:.4f}, hub 权重 {hub_weight[t]:.4f}")
        report_lines.append("")

    report_lines.extend([
        "## 6. Top 10 候选化合物",
        "",
        top_df.head(10).to_markdown(index=False),
        "",
        "## 7. 排名上升最多的 10 个化合物",
        "",
    ])

    biggest_risers = df.nl