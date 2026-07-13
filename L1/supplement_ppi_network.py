#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充铁衰老 PPI 网络：
1. 识别 ppi_network_extended_significant_edges.csv 中缺失的铁衰老基因
2. 从 ppi_network_extended_edges.csv 补充（combined_score >= 400 的边）
3. 从原始 STRING PPI 补充低置信度边（combined_score >= 150）
4. 对仍缺失的基因，作为孤立节点记录警告
5. 输出补充后的 PPI 网络
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(r'd:\铁衰老 绝不重蹈覆辙\logs\supplement_ppi_network.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 路径
ROOT = Path(r'd:\铁衰老 绝不重蹈覆辙')
FERRO_GENES_PATH = ROOT / 'L1' / 'results' / 'ferroaging_genes_96.csv'
DEDUP_PATH = ROOT / 'L1' / 'results' / 'ppi_network_extended_significant_edges.csv'
EXTENDED_PATH = ROOT / 'L1' / 'results' / 'ppi_network_extended_edges.csv'
SIGNIFICANT_PATH = ROOT / 'L1' / 'results' / 'ppi_network_extended_significant_edges.csv'
STRING_PPI_PATH = Path(r'C:\Users\Jy-Mentor-7\Desktop\9606蛋白质\9606_human_ppi_symbol.txt')
OUT_DIR = ROOT / 'L1' / 'results'

LOW_SCORE_THRESHOLD = 150  # 低置信度阈值
HIGH_SCORE_THRESHOLD = 400  # 高置信度阈值


def load_ferro_genes():
    """加载铁衰老96基因"""
    df = pd.read_csv(FERRO_GENES_PATH)
    genes = set(df['gene_symbol'].dropna().astype(str).str.strip().str.upper())
    logger.info(f"铁衰老96基因: {len(genes)} 个")
    return genes


def get_ppi_genes(ppi_path):
    """从PPI边文件中提取所有基因"""
    df = pd.read_csv(ppi_path, low_memory=False)
    genes = set(df['gene_a'].astype(str).str.strip().str.upper()) | \
            set(df['gene_b'].astype(str).str.strip().str.upper())
    return genes, df


def main():
    logger.info("=" * 70)
    logger.info("开始补充铁衰老 PPI 网络")
    logger.info("=" * 70)

    # 步骤1: 加载铁衰老基因和当前PPI网络
    ferro_genes = load_ferro_genes()

    # 当前 dedup PPI
    if not DEDUP_PATH.exists():
        logger.error(f"dedup PPI 文件不存在: {DEDUP_PATH}")
        sys.exit(1)
    dedup_genes, dedup_df = get_ppi_genes(DEDUP_PATH)
    logger.info(f"dedup PPI 网络: {len(dedup_df)} 条边, {len(dedup_genes)} 个基因")

    # 识别缺失基因
    covered = ferro_genes & dedup_genes
    missing = ferro_genes - dedup_genes
    logger.info(f"PPI 覆盖: {len(covered)} / {len(ferro_genes)}")
    logger.info(f"缺失: {len(missing)} 个基因")
    logger.info(f"缺失基因列表: {sorted(missing)}")

    # ============================================================
    # 步骤2: 从 ppi_network_extended_edges.csv 补充
    # ============================================================
    logger.info("-" * 60)
    logger.info("步骤2: 从扩展网络补充 (combined_score >= 400)")

    extended_genes, extended_df = get_ppi_genes(EXTENDED_PATH)
    logger.info(f"扩展网络: {len(extended_df)} 条边, {len(extended_genes)} 个基因")

    # 在扩展网络中存在的缺失基因
    missing_in_extended = missing & extended_genes
    still_missing_extended = missing - extended_genes
    logger.info(f"扩展网络中可补充: {len(missing_in_extended)} 个基因")
    logger.info(f"扩展网络中仍缺失: {len(still_missing_extended)} 个基因")
    if len(missing_in_extended) > 0:
        logger.info(f"可补充基因: {sorted(missing_in_extended)}")

    # 收集补充边：dedup边 + 扩展网络中涉及缺失基因的边
    supplement_edges_high = dedup_df.copy()

    # 筛选扩展网络中涉及缺失基因的边
    missing_in_extended_upper = set(g.upper() for g in missing_in_extended)
    extra_edges_extended = extended_df[
        extended_df['gene_a'].astype(str).str.strip().str.upper().isin(missing_in_extended_upper) |
        extended_df['gene_b'].astype(str).str.strip().str.upper().isin(missing_in_extended_upper)
    ]
    logger.info(f"扩展网络中涉及缺失基因的边: {len(extra_edges_extended)} 条")

    if len(extra_edges_extended) > 0:
        supplement_edges_high = pd.concat(
            [supplement_edges_high, extra_edges_extended], ignore_index=True
        )

    # 去重
    supplement_edges_high = supplement_edges_high.drop_duplicates(
        subset=['gene_a', 'gene_b'], keep='first'
    ).reset_index(drop=True)

    supplement_genes_high = set(supplement_edges_high['gene_a'].astype(str).str.strip().str.upper()) | \
                            set(supplement_edges_high['gene_b'].astype(str).str.strip().str.upper())
    covered_high = ferro_genes & supplement_genes_high
    missing_high = ferro_genes - supplement_genes_high
    logger.info(f"补充后(高置信度): {len(covered_high)} / {len(ferro_genes)} 个基因覆盖")
    logger.info(f"补充后(高置信度)仍缺失: {len(missing_high)} 个基因")

    # ============================================================
    # 步骤3: 从原始 STRING PPI 补充低置信度边
    # ============================================================
    logger.info("-" * 60)
    logger.info("步骤3: 从原始 STRING PPI 补充低置信度边 (combined_score >= 150)")

    low_edges = []

    if STRING_PPI_PATH.exists():
        logger.info(f"读取原始 STRING PPI: {STRING_PPI_PATH}")
        chunk_count = 0
        total_rows = 0
        for chunk in pd.read_csv(STRING_PPI_PATH, sep='\t', chunksize=500_000):
            chunk_count += 1
            total_rows += len(chunk)
            # 筛选低置信度边 (150 <= score < 400)
            low_chunk = chunk[
                (chunk['combined_score'] >= LOW_SCORE_THRESHOLD) &
                (chunk['combined_score'] < HIGH_SCORE_THRESHOLD)
            ]
            if len(low_chunk) == 0:
                continue
            # 检查是否涉及仍缺失的基因
            missing_upper = set(g.upper() for g in missing_high)
            mask = (
                low_chunk['gene_a'].astype(str).str.strip().str.upper().isin(missing_upper) |
                low_chunk['gene_b'].astype(str).str.strip().str.upper().isin(missing_upper)
            )
            relevant = low_chunk[mask]
            if len(relevant) > 0:
                low_edges.append(relevant)
            if chunk_count % 20 == 0:
                logger.info(f"  已处理 {chunk_count} 个 chunks ({total_rows:,} 行), "
                           f"累计低置信度补充边: {sum(len(e) for e in low_edges)}")

        if low_edges:
            low_edges_df = pd.concat(low_edges, ignore_index=True)
            low_edges_df = low_edges_df.drop_duplicates(
                subset=['gene_a', 'gene_b'], keep='first'
            ).reset_index(drop=True)
            logger.info(f"原始 STRING 中低置信度补充边: {len(low_edges_df)} 条")
            logger.info(f"补充边分数范围: {low_edges_df['combined_score'].min():.0f} - "
                       f"{low_edges_df['combined_score'].max():.0f}")
        else:
            low_edges_df = pd.DataFrame(columns=['gene_a', 'gene_b', 'combined_score'])
            logger.info("原始 STRING 中未找到低置信度补充边")
    else:
        logger.warning(f"原始 STRING PPI 文件不存在: {STRING_PPI_PATH}")
        low_edges_df = pd.DataFrame(columns=['gene_a', 'gene_b', 'combined_score'])

    # ============================================================
    # 步骤4: 合并所有边并输出最终补充网络
    # ============================================================
    logger.info("-" * 60)
    logger.info("步骤4: 合并所有边，输出最终补充 PPI 网络")

    # 合并 high + low 边
    all_edges = pd.concat([supplement_edges_high, low_edges_df], ignore_index=True)
    all_edges = all_edges.drop_duplicates(subset=['gene_a', 'gene_b'], keep='first').reset_index(drop=True)

    # 统计最终覆盖
    all_genes = set(all_edges['gene_a'].astype(str).str.strip().str.upper()) | \
                set(all_edges['gene_b'].astype(str).str.strip().str.upper())
    final_covered = ferro_genes & all_genes
    final_missing = ferro_genes - all_genes

    logger.info(f"最终补充网络: {len(all_edges)} 条边, {len(all_genes)} 个基因")
    logger.info(f"铁衰老基因覆盖: {len(final_covered)} / {len(ferro_genes)}")
    logger.info(f"仍缺失: {len(final_missing)} 个基因")

    # ============================================================
    # 步骤5: 为仍缺失的基因添加孤立节点
    # ============================================================
    logger.info("-" * 60)
    logger.info("步骤5: 处理孤立节点")

    isolated_edges = []
    if len(final_missing) > 0:
        logger.warning(f"以下 {len(final_missing)} 个基因无任何 PPI 数据，将作为孤立节点记录:")
        for g in sorted(final_missing):
            logger.warning(f"  - {g} (无 PPI 边，将作为孤立节点)")
            # 添加自环边作为占位（标记为孤立节点，combined_score=0）
            isolated_edges.append({
                'gene_a': g,
                'gene_b': g,
                'combined_score': 0
            })

    isolated_df = pd.DataFrame(isolated_edges, columns=['gene_a', 'gene_b', 'combined_score'])

    # 最终网络 = 所有边 + 孤立节点标记
    final_edges = pd.concat([all_edges, isolated_df], ignore_index=True) if len(isolated_df) > 0 else all_edges

    # 保存
    output_path = OUT_DIR / 'ppi_network_supplemented.csv'
    final_edges.to_csv(output_path, index=False)
    logger.info(f"补充后 PPI 网络已保存: {output_path}")
    logger.info(f"最终: {len(final_edges)} 条边（含 {len(isolated_df)} 个孤立节点标记）")

    # ============================================================
    # 步骤6: 生成补充报告
    # ============================================================
    logger.info("-" * 60)
    logger.info("步骤6: 生成补充报告")

    # 统计各来源的边数
    n_original_dedup = len(dedup_df)
    n_extended_extra = len(extra_edges_extended)
    n_low_confidence = len(low_edges_df)
    n_isolated = len(isolated_df)

    report_lines = [
        "=" * 70,
        "铁衰老 PPI 网络补充报告",
        "=" * 70,
        f"铁衰老96基因总数: {len(ferro_genes)}",
        f"原始 dedup PPI 覆盖: {len(covered)} 个基因 ({n_original_dedup} 条边)",
        f"",
        f"--- 补充情况 ---",
        f"从扩展网络补充 (score>=400): {len(missing_in_extended)} 个基因, {n_extended_extra} 条边",
        f"从 STRING 低置信度补充 (150<=score<400): {len(low_edges_df)} 条边",
        f"涉及低置信度边补充的基因: {len(missing_high - final_missing)} 个",
        f"",
        f"--- 最终结果 ---",
        f"最终覆盖: {len(final_covered)} / {len(ferro_genes)} 个基因",
        f"最终边数: {len(all_edges)} 条（不含孤立节点标记）",
        f"孤立节点: {len(final_missing)} 个基因",
        f"",
    ]

    if len(final_missing) > 0:
        report_lines.append(f"孤立节点列表: {', '.join(sorted(final_missing))}")
        report_lines.append("")
        report_lines.append("注意: 孤立节点仍有 ESM-2 嵌入和 KEGG 通路信息，可在下游分析中作为特征节点使用。")

    report_lines.append("=" * 70)

    for line in report_lines:
        logger.info(line)

    # 保存报告
    report_path = OUT_DIR / 'ppi_supplement_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info(f"补充报告已保存: {report_path}")

    return final_edges, final_covered, final_missing


if __name__ == '__main__':
    final_edges, covered, missing = main()