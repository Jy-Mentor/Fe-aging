#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩展铁衰老 PPI 网络：
1. 以 ferroaging_genes_96 为种子基因
2. 在 STRING 人类全 PPI 中扩展 1 层邻居
3. 用整合 DEGs (RRA) 筛选差异显著模块
4. 输出扩展后的 PPI 网络到 L1/results
"""

import os
import sys
import time
import logging
from pathlib import Path
from collections import defaultdict, deque

import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import hypergeom, fisher_exact

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(r'd:\铁衰老 绝不重蹈覆辙\logs\expand_ppi_network.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 路径
ROOT = Path(r'd:\铁衰老 绝不重蹈覆辙')
PPI_FILE = Path(r'C:\Users\Jy-Mentor-7\Desktop\9606蛋白质\9606_human_ppi_symbol.txt')
SEED_FILE = ROOT / 'L1' / 'results' / 'ferroaging_genes_96.csv'
DEG_FILE = ROOT / 'L1' / 'results' / 'RRA_gene_level_integrated.csv'
OUT_DIR = ROOT / 'L1' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMBINED_SCORE_THRESHOLD = 400  # STRING 常用置信度阈值
DEG_TOP_N = 2000  # 取 RRA 中 MedianRank 最小的 top N 作为显著 DEG
MODULE_DEG_FDR = 0.05
MIN_MODULE_SIZE = 5


def load_seed_genes():
    """加载铁衰老种子基因集。"""
    df = pd.read_csv(SEED_FILE)
    seeds = set(df['gene_symbol'].dropna().astype(str).str.strip().unique())
    logger.info(f'加载种子基因: {len(seeds)} 个')
    return seeds


def load_degs():
    """加载整合 DEGs (RRA)。"""
    df = pd.read_csv(DEG_FILE)
    # GeneSymbol 可能包含多基因注释，取第一个
    df['gene_symbol'] = df['GeneSymbol'].astype(str).str.split(' /// ').str[0].str.strip()
    df = df.sort_values('MedianRank', ascending=True)
    deg_top = df.head(DEG_TOP_N)
    degs = set(deg_top['gene_symbol'].unique())
    logger.info(f'加载显著 DEGs (top {DEG_TOP_N}): {len(degs)} 个')
    return df, degs


def read_ppi_chunked(path, score_threshold=COMBINED_SCORE_THRESHOLD, chunksize=500_000):
    """分块读取 STRING PPI 文件。"""
    for i, chunk in enumerate(pd.read_csv(path, sep='\t', chunksize=chunksize)):
        chunk = chunk[chunk['combined_score'] >= score_threshold]
        yield i, chunk


def expand_layers(seeds, ppi_file, max_layer=1):
    """分轮扫描 PPI 文件扩展 1 层邻居，并单独收集完整边集。

    注意：最后一层扫描时 current_nodes 尚未包含本层新节点，
    因此同层新节点之间的边会遗漏。修复方法：在所有节点确定后，
    再单独扫描一次 PPI，收集最终节点集内部的所有边。
    """
    current_nodes = set(seeds)
    layers = {0: set(seeds)}

    for layer in range(1, max_layer + 1):
        logger.info(f'开始扩展第 {layer} 层邻居...')
        new_neighbors = set()
        for i, chunk in read_ppi_chunked(ppi_file):
            mask = chunk['gene_a'].isin(current_nodes) | chunk['gene_b'].isin(current_nodes)
            sub = chunk[mask]
            if len(sub) == 0:
                continue
            # 向量化收集新邻居
            new_neighbors.update(sub.loc[~sub['gene_a'].isin(current_nodes), 'gene_a'])
            new_neighbors.update(sub.loc[~sub['gene_b'].isin(current_nodes), 'gene_b'])
            if i % 20 == 0:
                logger.info(f'  已处理 {i+1} 个 chunks, 当前层新邻居 {len(new_neighbors)}')
        current_nodes = current_nodes | new_neighbors
        layers[layer] = new_neighbors
        logger.info(f'第 {layer} 层完成: 新邻居 {len(new_neighbors)}, 累计节点 {len(current_nodes)}')

    # 最终扫描：补全最终节点集内部的全部边（尤其是 L2-L2 边）
    logger.info('最终扫描收集完整边集...')
    edges = []
    for i, chunk in read_ppi_chunked(ppi_file):
        sub = chunk[chunk['gene_a'].isin(current_nodes) & chunk['gene_b'].isin(current_nodes)]
        if len(sub) > 0:
            edges.append(sub)
        if i % 20 == 0:
            logger.info(f'  已处理 {i+1} 个 chunks')
    edges_df = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(columns=['gene_a', 'gene_b', 'combined_score'])
    logger.info(f'完整边集: {len(edges_df)} 条')

    return current_nodes, layers, edges_df


def _multi_source_shortest_path_length(G, sources):
    """多源无权 BFS 最短距离。兼容旧版/新版 NetworkX。"""
    dist = {}
    queue = deque()
    for s in sources:
        if s in G and s not in dist:
            dist[s] = 0
            queue.append(s)
    while queue:
        node = queue.popleft()
        d = dist[node]
        for neighbor in G.neighbors(node):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                queue.append(neighbor)
    return dist


def build_subgraph(nodes, edges_df, seeds, degs):
    """构建扩展后的子图，并标记节点属性。"""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for _, row in edges_df.iterrows():
        a, b, w = row['gene_a'], row['gene_b'], row['combined_score']
        if a in nodes and b in nodes:
            G.add_edge(a, b, weight=float(w))

    # 节点属性
    for n in G.nodes():
        G.nodes[n]['is_seed'] = n in seeds
        G.nodes[n]['is_deg'] = n in degs

    # 用无权 BFS 从所有种子节点同时计算最短距离（层数）
    seed_nodes = set(seeds) & set(G.nodes())
    lengths = _multi_source_shortest_path_length(G, seed_nodes)
    for n in G.nodes():
        if n in seed_nodes:
            G.nodes[n]['layer'] = 0
        else:
            G.nodes[n]['layer'] = lengths.get(n, -1)

    logger.info(f'子图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边')
    return G


def identify_significant_modules(G, seeds, degs, universe_size=None):
    """用 Louvain 社区检测识别模块，并用 DEGs 做富集显著性筛选。"""
    if universe_size is None:
        universe_size = G.number_of_nodes()

    deg_nodes = set(degs) & set(G.nodes())
    seed_nodes = set(seeds) & set(G.nodes())
    K = len(deg_nodes)  # 网络中 DEG 总数

    # Louvain 社区检测（带权重）
    import community as community_louvain
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    comm_to_nodes = defaultdict(set)
    for n, c in partition.items():
        comm_to_nodes[c].add(n)

    modules = []
    for comm_id, nodes in comm_to_nodes.items():
        if len(nodes) < MIN_MODULE_SIZE:
            continue
        mod_degs = deg_nodes & nodes
        mod_seeds = seed_nodes & nodes
        k = len(mod_degs)
        n = len(nodes)

        # 超几何检验：从 universe_size 个基因中抽 n 个，其中 K 个是 DEG，命中 k 个
        if n > universe_size or K > universe_size or k > n:
            pval = 1.0
        else:
            pval = hypergeom.sf(k - 1, universe_size, K, n) if k > 0 else 1.0

        modules.append({
            'module_id': comm_id + 1,
            'size': n,
            'n_deg': k,
            'deg_ratio': k / n if n > 0 else 0,
            'n_seed': len(mod_seeds),
            'seed_ratio': len(mod_seeds) / n if n > 0 else 0,
            'p_value': pval,
            'genes': ';'.join(sorted(nodes))
        })

    if not modules:
        logger.warning('未识别到大小 >= %d 的模块', MIN_MODULE_SIZE)
        return pd.DataFrame()

    mod_df = pd.DataFrame(modules)
    from statsmodels.stats.multitest import multipletests
    reject, pvals_corrected, _, _ = multipletests(mod_df['p_value'].fillna(1.0), method='fdr_bh')
    mod_df['fdr_bh'] = pvals_corrected
    mod_df['significant'] = (mod_df['fdr_bh'] < MODULE_DEG_FDR) & (mod_df['n_deg'] >= 1)
    mod_df = mod_df.sort_values(['significant', 'deg_ratio', 'size'], ascending=[False, False, False])
    logger.info(f'识别模块: {len(mod_df)} 个, 显著模块 (FDR<{MODULE_DEG_FDR}): {mod_df["significant"].sum()} 个')
    return mod_df


def main():
    logger.info('=' * 60)
    logger.info('开始扩展铁衰老 PPI 网络')
    logger.info('=' * 60)

    if not PPI_FILE.exists():
        logger.error(f'PPI 文件不存在: {PPI_FILE}')
        sys.exit(1)

    seeds = load_seed_genes()
    deg_df, degs = load_degs()

    t0 = time.time()
    nodes, layers, edges_df = expand_layers(seeds, PPI_FILE, max_layer=1)
    logger.info(f'网络扩展耗时: {time.time()-t0:.1f}s')
    logger.info(f'各层节点数: ' + ', '.join([f'L{k}={len(v)}' for k, v in sorted(layers.items())]))
    logger.info(f'扩展后总边数 (score>={COMBINED_SCORE_THRESHOLD}): {len(edges_df)}')

    # 保存完整扩展网络
    edges_df.to_csv(OUT_DIR / 'ppi_network_extended_edges.csv', index=False)

    # 构建图
    G = build_subgraph(nodes, edges_df, seeds, degs)

    # 保存节点信息
    node_records = []
    for n in G.nodes():
        node_records.append({
            'Gene': n,
            'Degree': G.degree(n),
            'is_seed': G.nodes[n].get('is_seed', False),
            'is_deg': G.nodes[n].get('is_deg', False),
            'layer': G.nodes[n].get('layer', -1)
        })
    node_df = pd.DataFrame(node_records)
    node_df['Degree_Centrality'] = node_df['Degree'] / (node_df.shape[0] - 1) if node_df.shape[0] > 1 else 0
    node_df = node_df.sort_values('Degree', ascending=False)
    node_df.to_csv(OUT_DIR / 'ppi_network_extended_nodes.csv', index=False)

    # 对大图避免计算 betweenness/closeness（O(n*m) 太慢）
    # 仅用 degree centrality 做 hub 排序；可选对 top 500 节点近似 closeness
    node_df['Hub_Rank'] = node_df['Degree_Centrality'].rank(ascending=False, method='min').astype(int)
    hub_df = node_df.sort_values(['Degree_Centrality', 'Degree'], ascending=[False, False])
    hub_df.to_csv(OUT_DIR / 'ppi_network_extended_hub_genes.csv', index=False)

    # 识别显著模块
    mod_df = identify_significant_modules(G, seeds, degs)
    if not mod_df.empty:
        mod_df.to_csv(OUT_DIR / 'ppi_network_extended_modules.csv', index=False)

    # 保存 DEG 筛选后的子网络
    if not mod_df.empty:
        sig_genes = set()
        for gene_str in mod_df[mod_df['significant']]['genes']:
            sig_genes.update(gene_str.split(';'))
        sig_edges = edges_df[(edges_df['gene_a'].isin(sig_genes)) & (edges_df['gene_b'].isin(sig_genes))]
        sig_edges.to_csv(OUT_DIR / 'ppi_network_extended_significant_edges.csv', index=False)
        logger.info(f'显著模块子网: {len(sig_genes)} 节点, {len(sig_edges)} 边')

    logger.info('扩展 PPI 网络完成，结果保存到 L1/results/')


if __name__ == '__main__':
    main()
