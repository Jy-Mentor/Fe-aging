#!/usr/bin/env python3
"""
GSE104036 (Mouse MCAO) 三项专项分析管线
=========================================
1. 最短路径分析 —— 核心PPI子网络中关键基因间调控距离
2. 铁衰老 GSVA/ssGSEA 通路活性评分
3. 免疫浸润分析 —— 免疫细胞丰度 + 铁衰老-炎症相关性
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import pearsonr, mannwhitneyu

ROOT = Path(r'd:\铁衰老 绝不重蹈覆辙')
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = ROOT / 'L2' / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'shortest_path_gsva_immune.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 文件路径
# ============================================================
CORE_PPI_EDGES = RESULT_DIR / 'core_ppi_edges.csv'
CORE_PPI_TOPOLOGY = RESULT_DIR / 'core_ppi_topology.csv'
FINAL_CORE_GENES = RESULT_DIR / 'final_core_gene_set.csv'
FERRORAGING_96 = ROOT / 'L1' / 'results' / 'ferroaging_genes_96.csv'
EXPR_MATRIX = ROOT / 'L1' / 'results' / 'GSE104036_expression_matrix.csv'
SAMPLE_META = ROOT / 'L1' / 'results' / 'GSE104036_sample_meta.csv'

# ============================================================
# 人鼠基因转换表（与已有R脚本保持一致）
# ============================================================
HUMAN_TO_MOUSE_MAP = {
    'ABCC1': 'Abcc1', 'ACVR1B': 'Acvr1b', 'ACSL4': 'Acsl4', 'ALOX15': 'Alox15',
    'ATF3': 'Atf3', 'ATG3': 'Atg3', 'BAP1': 'Bap1', 'BCL6': 'Bcl6',
    'BRD7': 'Brd7', 'CAVIN1': 'Cavin1', 'CD74': 'Cd74', 'CD82': 'Cd82',
    'CDO1': 'Cdo1', 'COX7A1': 'Cox7a1', 'CTSB': 'Ctsb', 'CXCL10': 'Cxcl10',
    'DPEP1': 'Dpep1', 'DPP4': 'Dpp4', 'DUOX1': 'Duox1', 'DYRK1A': 'Dyrk1a',
    'E2F1': 'E2f1', 'E2F3': 'E2f3', 'EBF3': 'Ebf3', 'EDN1': 'Edn1',
    'EGR1': 'Egr1', 'EMP1': 'Emp1', 'EPHA2': 'Epha2', 'EPHA4': 'Epha4',
    'ERN1': 'Ern1', 'FBXO31': 'Fbxo31', 'FOSL1': 'Fosl1', 'GMFB': 'Gmfb',
    'HBP1': 'Hbp1', 'HERPUD1': 'Herpud1', 'HIF1A': 'Hif1a', 'HMGB1': 'Hmgb1',
    'HMOX1': 'Hmox1', 'ICA1': 'Ica1', 'IFNG': 'Ifng', 'IGFBP7': 'Igfbp7',
    'IL1B': 'Il1b', 'IL6': 'Il6', 'IRF1': 'Irf1', 'IRF7': 'Irf7',
    'IRF9': 'Irf9', 'KDM6B': 'Kdm6b', 'KEAP1': 'Keap1', 'KLF6': 'Klf6',
    'LACTB': 'Lactb', 'LCN2': 'Lcn2', 'LGMN': 'Lgmn', 'LIFR': 'Lifr',
    'LOX': 'Lox', 'LPCAT3': 'Lpcat3', 'MAP3K14': 'Map3k14', 'MAPK1': 'Mapk1',
    'MAPK14': 'Mapk14', 'MCU': 'Mcu', 'MEN1': 'Men1', 'MPO': 'Mpo',
    'NLRP3': 'Nlrp3', 'NOX4': 'Nox4', 'NR1D1': 'Nr1d1', 'NR2F2': 'Nr2f2',
    'NUAK2': 'Nuak2', 'PADI4': 'Padi4', 'PDE4B': 'Pde4b', 'PPP2R2B': 'Ppp2r2b',
    'PRKD1': 'Prkd1', 'PTBP1': 'Ptbp1', 'PTGS2': 'Ptgs2', 'RBM3': 'Rbm3',
    'RUNX3': 'Runx3', 'S100A8': 'S100a8', 'SAT1': 'Sat1', 'SETD7': 'Setd7',
    'SLAMF8': 'Slamf8', 'SLC1A5': 'Slc1a5', 'SMARCB1': 'Smarcb1', 'SMURF2': 'Smurf2',
    'SNCA': 'Snca', 'SOCS1': 'Socs1', 'SOCS2': 'Socs2', 'SOD1': 'Sod1',
    'SP1': 'Sp1', 'SPATA2': 'Spata2', 'TBX2': 'Tbx2', 'TFRC': 'Tfrc',
    'TLR4': 'Tlr4', 'TNFAIP1': 'Tnfaip1', 'TNFAIP3': 'Tnfaip3', 'TXNIP': 'Txnip',
    'WNT5A': 'Wnt5a', 'WWTR1': 'Wwtr1', 'YAP1': 'Yap1', 'ZEB1': 'Zeb1'
}

# ============================================================
# 小鼠免疫细胞特征基因集（基于 Bindea 2013 + Newman 2015 免疫签名，小鼠同源转换）
# ============================================================
MOUSE_IMMUNE_SIGNATURES = {
    'CD8_T_cells': ['Cd8a', 'Cd8b1', 'Cd3d', 'Cd3e', 'Cd3g', 'Gzmb', 'Prf1', 'Nkg7'],
    'CD4_T_cells': ['Cd4', 'Cd3d', 'Cd3e', 'Cd3g', 'Il7r', 'Cd28', 'Icos'],
    'Treg_cells': ['Foxp3', 'Il2ra', 'Ctla4', 'Ikzf2', 'Tnfrsf18', 'Tnfrsf4'],
    'NK_cells': ['Nkg7', 'Klrb1c', 'Ncr1', 'Klrk1', 'Gzma', 'Prf1', 'Xcl1'],
    'B_cells': ['Cd19', 'Cd79a', 'Cd79b', 'Ms4a1', 'Pax5', 'Blk', 'Fcmr'],
    'Monocytes': ['Cd14', 'Ccr2', 'Csf1r', 'Itgam', 'Ly6c2', 'Fn1', 'Sell'],
    'Macrophages_M1': ['Nos2', 'Il12a', 'Il12b', 'Tnf', 'Cd86', 'Cd80', 'Ccl3', 'Il23a'],
    'Macrophages_M2': ['Arg1', 'Mrc1', 'Cd163', 'Il10', 'Retnla', 'Chil3', 'Tgfb1'],
    'Neutrophils': ['Ly6g', 'Mmp9', 'Cxcr2', 'Csf3r', 'Itgam', 'S100a8', 'S100a9'],
    'Dendritic_cells': ['Itgax', 'Flt3', 'Ccr7', 'Cd83', 'Ccl17', 'Ccl22', 'Batf3'],
    'Microglia': ['Tmem119', 'P2ry12', 'Cx3cr1', 'Trem2', 'Hexb', 'Cst3', 'Fcrls'],
    'Astrocytes': ['Gfap', 'Aqp4', 'Slc1a3', 'Slc1a2', 'Aldh1l1', 'S100b', 'Aldoc'],
}

# 关键炎症因子基因（小鼠）
MOUSE_INFLAMMATION_GENES = [
    'Il1b', 'Il6', 'Tnf', 'Cxcl10', 'Ccl2', 'Ptgs2', 'Nos2',
    'Il10', 'Tgfb1', 'Ifng', 'Icam1', 'Vcam1', 'Mmp9',
    'Hif1a', 'Hmox1', 'Nfkb1', 'Rela', 'Stat3'
]

# 核心Hub基因列表（用于最短路径分析）
CORE_HUB_GENES = ['TP53', 'EGFR', 'STAT3', 'HIF1A', 'MTOR', 'EP300', 'TNF',
                  'IL6', 'SIRT1', 'TLR4', 'MAPK1', 'NFE2L2', 'PTGS2', 'BAX',
                  'CASP3', 'BECN1', 'KRAS', 'PTEN', 'FOXO3']

# 铁衰老标记基因（用于最短路径，从96基因集中选）
FERRORAGING_MARKER_GENES = ['HIF1A', 'HMOX1', 'TFRC', 'ACSL4', 'PTGS2',
                            'IL6', 'IL1B', 'CXCL10', 'SOD1', 'HMGB1',
                            'NLRP3', 'KEAP1', 'SAT1', 'LCN2', 'S100A8',
                            'CD74', 'TLR4', 'IFNG', 'MAPK1', 'MAPK14']


def human_to_mouse(human_genes):
    """将人类基因转换为小鼠同源基因符号。"""
    result = []
    for g in human_genes:
        if g in HUMAN_TO_MOUSE_MAP:
            result.append(HUMAN_TO_MOUSE_MAP[g])
        else:
            # 默认规则：首字母大写 + 剩余小写
            result.append(g.capitalize())
    return result


# ============================================================
# Part 1: 最短路径分析
# ============================================================
def shortest_path_analysis():
    """在核心PPI子网络中执行最短路径搜索。"""
    logger.info('=' * 60)
    logger.info('Part 1: 最短路径分析')
    logger.info('=' * 60)

    # 加载PPI网络
    edge_df = pd.read_csv(CORE_PPI_EDGES)
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['Gene_A'], row['Gene_B'])

    # 加载拓扑数据
    topo_df = pd.read_csv(CORE_PPI_TOPOLOGY)

    logger.info('核心PPI子网络: %d 节点, %d 边', G.number_of_nodes(), G.number_of_edges())

    # 确定在网络中存在的hub基因和铁衰老标记基因
    hub_in_network = [g for g in CORE_HUB_GENES if g in G]
    ferro_in_network = [g for g in FERRORAGING_MARKER_GENES if g in G]
    logger.info('Hub基因在网络中: %d/%d', len(hub_in_network), len(CORE_HUB_GENES))
    logger.info('铁衰老标记基因在网络中: %d/%d', len(ferro_in_network), len(FERRORAGING_MARKER_GENES))

    # --- 1a. Hub基因间的成对最短路径 ---
    logger.info('\n--- Hub基因间最短路径 ---')
    hub_pair_results = []
    for i in range(len(hub_in_network)):
        for j in range(i + 1, len(hub_in_network)):
            g1, g2 = hub_in_network[i], hub_in_network[j]
            try:
                path = nx.shortest_path(G, source=g1, target=g2)
                dist = len(path) - 1
                hub_pair_results.append({
                    'Gene_A': g1, 'Gene_B': g2,
                    'Shortest_Path_Length': dist,
                    'Path': ' -> '.join(path)
                })
            except nx.NetworkXNoPath:
                hub_pair_results.append({
                    'Gene_A': g1, 'Gene_B': g2,
                    'Shortest_Path_Length': np.nan,
                    'Path': 'No path'
                })

    hub_pair_df = pd.DataFrame(hub_pair_results).sort_values('Shortest_Path_Length')
    hub_pair_df.to_csv(RESULT_DIR / 'shortest_path_hub_pairs.csv', index=False, encoding='utf-8-sig')
    valid_pairs = hub_pair_df.dropna(subset=['Shortest_Path_Length'])
    logger.info('Hub基因对: %d 对, 平均距离 = %.2f, 最小距离 = %d, 最大距离 = %d',
                len(valid_pairs), valid_pairs['Shortest_Path_Length'].mean(),
                int(valid_pairs['Shortest_Path_Length'].min()),
                int(valid_pairs['Shortest_Path_Length'].max()))
    logger.info('最短的5对:')
    for _, row in valid_pairs.head(5).iterrows():
        logger.info('  %s - %s: %d步 - %s', row['Gene_A'], row['Gene_B'],
                    int(row['Shortest_Path_Length']), row['Path'])

    # --- 1b. Hub基因与铁衰老标记基因间的最短路径 ---
    logger.info('\n--- Hub → 铁衰老标记 最短路径 ---')
    hub_to_ferro_results = []
    for hub in hub_in_network:
        for fg in ferro_in_network:
            if hub == fg:
                continue
            try:
                path = nx.shortest_path(G, source=hub, target=fg)
                dist = len(path) - 1
                hub_to_ferro_results.append({
                    'Hub_Gene': hub,
                    'Ferroaging_Gene': fg,
                    'Distance': dist,
                    'Path': ' -> '.join(path)
                })
            except nx.NetworkXNoPath:
                hub_to_ferro_results.append({
                    'Hub_Gene': hub,
                    'Ferroaging_Gene': fg,
                    'Distance': np.nan,
                    'Path': 'No path'
                })

    hub_ferro_df = pd.DataFrame(hub_to_ferro_results).sort_values('Distance')
    hub_ferro_df.to_csv(RESULT_DIR / 'shortest_path_hub_to_ferroaging.csv', index=False, encoding='utf-8-sig')

    # 每个铁衰老基因到最近Hub的平均距离
    ferro_to_hub_stats = []
    for fg in ferro_in_network:
        fg_dists = hub_ferro_df[hub_ferro_df['Ferroaging_Gene'] == fg]['Distance'].dropna()
        if len(fg_dists) > 0:
            ferro_to_hub_stats.append({
                'Ferroaging_Gene': fg,
                'Min_Distance_to_Hub': fg_dists.min(),
                'Mean_Distance_to_Hub': fg_dists.mean(),
                'Closest_Hub': hub_ferro_df[(hub_ferro_df['Ferroaging_Gene'] == fg) &
                                            (hub_ferro_df['Distance'] == fg_dists.min())]['Hub_Gene'].iloc[0]
            })
    ferro_hub_stats_df = pd.DataFrame(ferro_to_hub_stats).sort_values('Mean_Distance_to_Hub')
    ferro_hub_stats_df.to_csv(RESULT_DIR / 'shortest_path_ferroaging_to_hub_stats.csv', index=False, encoding='utf-8-sig')
    logger.info('铁衰老基因到Hub平均距离: %.2f ± %.2f',
                ferro_hub_stats_df['Mean_Distance_to_Hub'].mean(),
                ferro_hub_stats_df['Mean_Distance_to_Hub'].std())
    logger.info('最接近Hub的铁衰老基因 (距离<=2):')
    close_genes = ferro_hub_stats_df[ferro_hub_stats_df['Min_Distance_to_Hub'] <= 2]
    for _, row in close_genes.iterrows():
        logger.info('  %s → %s (距离=%d, 平均=%.2f)',
                    row['Ferroaging_Gene'], row['Closest_Hub'],
                    int(row['Min_Distance_to_Hub']), row['Mean_Distance_to_Hub'])

    # --- 1c. 铁死亡直接靶点与铁衰老标记基因 ---
    logger.info('\n--- 铁死亡直接靶点 ↔ 铁衰老标记最短路径 ---')
    # 加载核心基因集获取Part A
    core_df = pd.read_csv(FINAL_CORE_GENES)
    part_a_genes = set(core_df[core_df['Source'] == 'PartA_CIRI_Caryophyllene']['Gene'])
    direct_targets_in_network = [g for g in part_a_genes if g in G]

    if direct_targets_in_network:
        for dtg in direct_targets_in_network:
            dtg_dists = []
            for fg in ferro_in_network:
                if dtg == fg:
                    continue
                try:
                    path = nx.shortest_path(G, source=dtg, target=fg)
                    dtg_dists.append(len(path) - 1)
                except nx.NetworkXNoPath:
                    pass
            if dtg_dists:
                logger.info('  %s → 铁衰老基因: 距离 %.2f ± %.2f (min=%d, max=%d)',
                            dtg, np.mean(dtg_dists), np.std(dtg_dists),
                            min(dtg_dists), max(dtg_dists))
    else:
        logger.info('  无Part A基因在PPI网络中')

    # --- 1d. 网络全局属性 ---
    logger.info('\n--- PPI子网络全局属性 ---')
    if nx.is_connected(G):
        avg_shortest_path = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
        logger.info('网络连通: 是')
    else:
        components = list(nx.connected_components(G))
        largest_cc = G.subgraph(max(components, key=len))
        avg_shortest_path = nx.average_shortest_path_length(largest_cc)
        diameter = nx.diameter(largest_cc)
        logger.info('网络连通: 否 (%d个连通分量, 最大分量=%d节点)',
                    len(components), largest_cc.number_of_nodes())

    logger.info('平均最短路径: %.2f', avg_shortest_path)
    logger.info('网络直径: %d', diameter)
    clustering_coef = nx.average_clustering(G)
    logger.info('平均聚类系数: %.4f', clustering_coef)
    logger.info('网络密度: %.4f', nx.density(G))

    return G


# ============================================================
# 手工 ssGSEA 实现 (基于 Barbie et al. 2009, Nature)
# ============================================================
def ssgsea_manual(data, gene_set, alpha=0.25):
    """
    Single-sample GSEA (ssGSEA) 纯 numpy 实现。
    参考: Barbie et al. 2009 (Nature, 462:108-112); GSVA (Hanzelmann 2013).

    算法:
    1. 每样本独立: 基因按表达值降序排列 (最高表达排第1位)
    2. 位置权重 = rank_position^alpha (pos=1..N, default alpha=0.25)
    3. running_sum = cumsum(P_hit - P_miss)
       P_hit(i) = Σ_{j≤i, gene_j∈S} pos_weight(j) / Σ_{gene∈S} pos_weight
       P_miss(i) = Σ_{j≤i, gene_j∉S} 1 / (N - |S|)
    4. Score = max(running_sum)  # >0 表示基因集富集于高表达区

    data: genes × samples pd.DataFrame
    gene_set: data.index 中的基因名 list
    alpha: 权重指数 (default 0.25, per Barbie 2009)
    返回: pd.Series (index=sample, value=ssGSEA score)
    """
    genes = data.index.values
    N = len(genes)
    in_set = np.isin(genes, np.array(gene_set))
    N_set = int(in_set.sum())
    if N_set < 2:
        raise ValueError(f'Gene set has only {N_set} genes in data, need >= 2')

    position_weights = np.arange(1, N + 1, dtype=float) ** alpha
    P_N = float(N - N_set)

    scores = {}
    for sample in data.columns:
        order = np.argsort(data[sample].values)[::-1]
        sorted_in_set = in_set[order].astype(float)

        P_G = float(np.sum(sorted_in_set * position_weights))
        if P_G == 0:
            scores[sample] = 0.0
            continue

        hit = sorted_in_set * position_weights / P_G
        miss = (1.0 - sorted_in_set) / P_N
        scores[sample] = float(np.max(np.cumsum(hit - miss)))

    return pd.Series(scores)


# ============================================================
# Part 2: 铁衰老 GSVA/ssGSEA 通路活性评分
# ============================================================
def ferroaging_ssgsea_analysis():
    """使用手工 ssGSEA 计算铁衰老通路活性评分。"""
    logger.info('\n' + '=' * 60)
    logger.info('Part 2: 铁衰老 ssGSEA 通路活性评分 (GSE104036)')
    logger.info('=' * 60)

    # 加载铁衰老96基因集
    fa96 = pd.read_csv(FERRORAGING_96)
    ferroaging_human = fa96['gene_symbol'].tolist()
    logger.info('铁衰老基因集: %d human genes', len(ferroaging_human))

    # 加载表达矩阵
    expr_df = pd.read_csv(EXPR_MATRIX, index_col=0)
    meta_df = pd.read_csv(SAMPLE_META)

    # CPM + log2 标准化
    lib_sizes = expr_df.sum(axis=0)
    cpm = expr_df.div(lib_sizes, axis=1) * 1e6
    log2cpm = np.log2(cpm + 1)

    # 人→鼠基因转换
    ferroaging_mouse = human_to_mouse(ferroaging_human)
    present_mouse = [g for g in ferroaging_mouse if g in log2cpm.index]
    logger.info('  GSE104036中可匹配铁衰老基因: %d/%d', len(present_mouse), len(ferroaging_mouse))

    if len(present_mouse) < 5:
        logger.error('铁衰老基因覆盖率不足，无法进行ssGSEA')
        return None, None

    # 缺失基因日志
    missing = [g for g in ferroaging_mouse if g not in log2cpm.index]
    if missing:
        logger.info('  缺失基因: %s', ', '.join(missing))

    # ssGSEA (手工实现)
    logger.info('  执行ssGSEA (手工实现)...')
    score_series = ssgsea_manual(log2cpm, present_mouse, alpha=0.25)

    sample_ids = score_series.index.tolist()
    scores_df = pd.DataFrame({
        'Sample': sample_ids,
        'Ferroaging_Score': score_series.values
    })

    # 合并元数据
    scores_df['group'] = scores_df['Sample'].map(
        dict(zip(meta_df['sample'], meta_df['group'])))
    scores_df['time'] = scores_df['Sample'].map(
        dict(zip(meta_df['sample'], meta_df['time'])))
    scores_df['tissue'] = scores_df['Sample'].map(
        dict(zip(meta_df['sample'], meta_df['tissue'])))

    score_vals = score_series.values
    logger.info('ssGSEA评分统计 (%d 基因集):', len(present_mouse))
    logger.info('  Range: %.4f - %.4f', score_vals.min(), score_vals.max())
    logger.info('  Mean: %.4f ± %.4f', score_vals.mean(), score_vals.std())

    # 分组统计 + 统计检验
    from scipy.stats import mannwhitneyu
    logger.info('\n分组比较 (Mann-Whitney U):')
    groups = scores_df.groupby('group')['Ferroaging_Score']
    for grp_name, grp_data in groups:
        logger.info('  %s: %.4f ± %.4f (n=%d)',
                    grp_name, grp_data.mean(), grp_data.std(), len(grp_data))

    comparisons = [('Ipsilateral', 'Sham'), ('Ipsilateral', 'Contralateral'),
                   ('Contralateral', 'Sham')]
    stats_results = []
    for g1_name, g2_name in comparisons:
        g1_vals = scores_df[scores_df['group'] == g1_name]['Ferroaging_Score'].values
        g2_vals = scores_df[scores_df['group'] == g2_name]['Ferroaging_Score'].values
        if len(g1_vals) >= 2 and len(g2_vals) >= 2:
            stat, p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            n1, n2 = len(g1_vals), len(g2_vals)
            s_pooled = np.sqrt(((n1 - 1) * np.var(g1_vals, ddof=1) +
                                (n2 - 1) * np.var(g2_vals, ddof=1)) / (n1 + n2 - 2))
            d = (np.mean(g1_vals) - np.mean(g2_vals)) / s_pooled if s_pooled > 0 else 0
            hedges_g = d * (1 - 3 / (4 * (n1 + n2) - 9))
            stats_results.append({
                'Group1': g1_name, 'Group2': g2_name,
                'Mean1': np.mean(g1_vals), 'Mean2': np.mean(g2_vals),
                'Cohens_d': d, 'Hedges_g': hedges_g,
                'MannWhitney_U': stat, 'P_value': p,
                'N1': n1, 'N2': n2
            })
            logger.info('  %s vs %s: d=%.3f, p=%.2e', g1_name, g2_name, d, p)

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(RESULT_DIR / 'ferroaging_ssgsea_stats_GSE104036.csv', index=False, encoding='utf-8-sig')

    # 时序分析 (仅Ipsilateral)
    ipsi_scores = scores_df[scores_df['group'] == 'Ipsilateral'].copy()
    time_order = ['0hr', '3hr', '6hr', '12hr', '24hr']
    logger.info('\nIpsilateral时序趋势:')
    for t in time_order:
        t_scores = ipsi_scores[ipsi_scores['time'] == t]['Ferroaging_Score']
        if len(t_scores) > 0:
            logger.info('  %s: %.4f ± %.4f (n=%d)',
                        t, t_scores.mean(), t_scores.std(), len(t_scores))

    # 时序相关分析
    ipsi_non_sham = ipsi_scores[ipsi_scores['time'] != '0hr'].copy()
    if len(ipsi_non_sham) >= 5:
        ipsi_non_sham['time_num'] = ipsi_non_sham['time'].apply(
            lambda t: int(t.replace('hr', '')) if pd.notna(t) else 0)
        from scipy.stats import spearmanr
        rho, p_rho = spearmanr(ipsi_non_sham['time_num'], ipsi_non_sham['Ferroaging_Score'])
        logger.info('Ipsilateral时序Spearman ρ: %.3f, p=%.2e', rho, p_rho)

    scores_df.to_csv(RESULT_DIR / 'ferroaging_ssgsea_scores_GSE104036.csv', index=False, encoding='utf-8-sig')
    return scores_df, stats_df


# ============================================================
# Part 3: 免疫浸润分析
# ============================================================
def immune_infiltration_analysis(scores_df):
    """基于手工ssGSEA的免疫浸润分析 + 炎症因子相关性。"""
    logger.info('\n' + '=' * 60)
    logger.info('Part 3: 免疫浸润分析 (基于特征基因集手工ssGSEA)')
    logger.info('=' * 60)

    # 加载表达矩阵
    expr_df = pd.read_csv(EXPR_MATRIX, index_col=0)
    lib_sizes = expr_df.sum(axis=0)
    cpm = expr_df.div(lib_sizes, axis=1) * 1e6
    log2cpm = np.log2(cpm + 1)

    # 3a. 免疫细胞丰度评分
    logger.info('--- 免疫细胞 ssGSEA 评分 ---')
    immune_enrichment = {}
    for cell_type, marker_genes in MOUSE_IMMUNE_SIGNATURES.items():
        present_markers = [g for g in marker_genes if g in log2cpm.index]
        if len(present_markers) < 3:
            logger.warning('  %s: 标记基因不足 (%d), 跳过', cell_type, len(present_markers))
            continue

        try:
            vals = ssgsea_manual(log2cpm, present_markers, alpha=0.25)
            immune_enrichment[cell_type] = vals.values
            logger.info('  %-20s: %.4f ± %.4f (markers=%d/%d)',
                        cell_type, np.mean(vals.values), np.std(vals.values),
                        len(present_markers), len(marker_genes))
        except Exception as e:
            logger.warning('  %s: ssGSEA失败 - %s', cell_type, e)

    if not immune_enrichment:
        logger.error('无免疫细胞类型可评分')
        return None

    # 构建免疫细胞丰度矩阵
    sample_ids = log2cpm.columns.tolist()
    immune_df = pd.DataFrame(immune_enrichment, index=sample_ids)
    immune_df.index.name = 'Sample'
    immune_df.to_csv(RESULT_DIR / 'immune_cell_scores_GSE104036.csv', encoding='utf-8-sig')

    # 3b. 炎症因子表达提取
    logger.info('\n--- 炎症因子表达 ---')
    meta_df = pd.read_csv(SAMPLE_META)
    inflam_present = [g for g in MOUSE_INFLAMMATION_GENES if g in log2cpm.index]
    logger.info('可检测炎症因子: %d/%d - %s',
                len(inflam_present), len(MOUSE_INFLAMMATION_GENES),
                ', '.join(inflam_present))

    inflam_expr = log2cpm.loc[inflam_present].T
    inflam_expr.to_csv(RESULT_DIR / 'inflammation_expression_GSE104036.csv', encoding='utf-8-sig')

    # 3c. Pearson相关性: 铁衰老评分 vs 免疫丰度
    logger.info('\n--- 铁衰老评分 vs 免疫细胞丰度 (Pearson) ---')
    if scores_df is None:
        logger.warning('无铁衰老评分数据，跳过相关性分析')
        return immune_df

    scores_series = scores_df.set_index('Sample')['Ferroaging_Score']

    immune_cor_results = []
    for cell_type in immune_df.columns:
        common_samples = list(set(immune_df.index) & set(scores_series.index))
        if len(common_samples) < 5:
            continue
        im_vals = immune_df.loc[common_samples, cell_type].values
        sc_vals = scores_series.loc[common_samples].values
        r, p = pearsonr(im_vals, sc_vals)
        immune_cor_results.append({
            'Cell_Type': cell_type,
            'Pearson_R': r,
            'P_value': p,
            'N_samples': len(common_samples)
        })
        logger.info('  %-24s r=%+.3f p=%.3e', cell_type, r, p)

    immune_cor_df = pd.DataFrame(immune_cor_results).sort_values('Pearson_R', ascending=False)
    immune_cor_df.to_csv(RESULT_DIR / 'immune_ferroaging_correlation_GSE104036.csv',
                         index=False, encoding='utf-8-sig')

    # 3d. Pearson相关性: 铁衰老评分 vs 炎症因子
    logger.info('\n--- 铁衰老评分 vs 炎症因子 (Pearson) ---')
    inflam_cor_results = []
    for gene in inflam_present:
        common_samples = list(set(inflam_expr.index) & set(scores_series.index))
        if len(common_samples) < 5:
            continue
        gene_vals = inflam_expr.loc[common_samples, gene].values
        sc_vals = scores_series.loc[common_samples].values
        r, p = pearsonr(gene_vals, sc_vals)
        inflam_cor_results.append({
            'Gene': gene,
            'Pearson_R': r,
            'P_value': p,
            'N_samples': len(common_samples)
        })

    inflam_cor_df = pd.DataFrame(inflam_cor_results).sort_values('Pearson_R', ascending=False)
    inflam_cor_df.to_csv(RESULT_DIR / 'inflammation_ferroaging_correlation_GSE104036.csv',
                         index=False, encoding='utf-8-sig')

    logger.info('显著正相关 (p<0.05):')
    sig_pos = inflam_cor_df[(inflam_cor_df['Pearson_R'] > 0) & (inflam_cor_df['P_value'] < 0.05)]
    for _, row in sig_pos.iterrows():
        logger.info('  %-10s r=%+.3f p=%.3e', row['Gene'], row['Pearson_R'], row['P_value'])
    logger.info('显著负相关 (p<0.05):')
    sig_neg = inflam_cor_df[(inflam_cor_df['Pearson_R'] < 0) & (inflam_cor_df['P_value'] < 0.05)]
    for _, row in sig_neg.iterrows():
        logger.info('  %-10s r=%+.3f p=%.3e', row['Gene'], row['Pearson_R'], row['P_value'])

    # 3e. Ipsilateral vs Sham 免疫丰度差异
    logger.info('\n--- Ipsilateral vs Sham 免疫差异 ---')
    gsm = scores_df.groupby('group')['Sample'].apply(list).to_dict()
    ipsi_samples = gsm.get('Ipsilateral', [])
    sham_samples = gsm.get('Sham', [])

    ipsi_immune_diff = []
    for cell_type in immune_df.columns:
        ipsi_vals = immune_df.loc[[s for s in ipsi_samples if s in immune_df.index], cell_type].values
        sham_vals = immune_df.loc[[s for s in sham_samples if s in immune_df.index], cell_type].values
        if len(ipsi_vals) >= 2 and len(sham_vals) >= 2:
            try:
                stat_u, p_u = mannwhitneyu(ipsi_vals, sham_vals, alternative='two-sided')
                fc = np.mean(ipsi_vals) - np.mean(sham_vals)
                ipsi_immune_diff.append({
                    'Cell_Type': cell_type,
                    'Ipsilateral_Mean': np.mean(ipsi_vals),
                    'Sham_Mean': np.mean(sham_vals),
                    'Delta': fc,
                    'P_value': p_u
                })
                logger.info('  %-24s Δ=%.3f p=%.3e', cell_type, fc, p_u)
            except Exception:
                pass

    ipsi_diff_df = pd.DataFrame(ipsi_immune_diff).sort_values('Delta', ascending=False)
    ipsi_diff_df.to_csv(RESULT_DIR / 'immune_ipsilateral_vs_sham_GSE104036.csv',
                        index=False, encoding='utf-8-sig')

    return immune_df


# ============================================================
# Main
# ============================================================
def main():
    logger.info('=' * 70)
    logger.info('GSE104036 三项专项分析管线')
    logger.info('数据集: Mouse MCAO RNA-seq, 27 samples (3 Sham, 12 Contralateral, 12 Ipsilateral)')
    logger.info('=' * 70)

    t_start = time.time()

    # Part 1: 最短路径分析
    G = shortest_path_analysis()

    # Part 2: 铁衰老 ssGSEA
    scores_df, stats_df = ferroaging_ssgsea_analysis()

    # Part 3: 免疫浸润分析
    immune_df = immune_infiltration_analysis(scores_df)

    # 汇总
    elapsed = time.time() - t_start
    logger.info('\n' + '=' * 70)
    logger.info('分析完成! 耗时: %.1f 秒', elapsed)
    logger.info('输出文件保存至: %s', RESULT_DIR)
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
