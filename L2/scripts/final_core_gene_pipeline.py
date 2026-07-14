#!/usr/bin/env python3
"""
最终核心基因集确定与多维度验证管线
=============================================
核心基因集 = (CIRI-铁衰老候选基因 ∩ 石竹烯高置信度靶点) ∪ 网络邻近扩展的铁死亡关键基因

步骤:
  1. 加载并整合多源真实基因数据
  2. 确定最终核心基因集
  3. STRING PPI 网络构建 + 拓扑参数 (Degree, Betweenness)
  4. GO/KEGG 富集分析 (调用 R clusterProfiler)
  5. MCODE 紧密连接模块识别
  6. WGCNA 共表达模块验证
"""

import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import hypergeom

ROOT = Path(r'd:\铁衰老 绝不重蹈覆辙')
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR = ROOT / 'L2' / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'final_core_gene_pipeline.log', mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

CARY_FILE = Path(r'C:\Users\Jy-Mentor-7\Desktop\申请书\石竹烯 人.txt')
FERRO_FILE = Path(r'C:\Users\Jy-Mentor-7\Desktop\申请书\铁死亡驱动基因集.txt')
CIRI_CAND_FILE = ROOT / 'L2' / 'results' / 'ciri_ferroaging_lasso_candidates.csv'
STRING_LINKS = ROOT / '9606.protein.links.v12.0.txt'
STRING_ALIASES = ROOT / '9606.protein.aliases.v12.0.txt'
CORE_GENES_FILE = ROOT / 'L1' / 'results' / 'core_genes_final.csv'
FERRORAGING_96_FILE = ROOT / 'L1' / 'results' / 'ferroaging_genes_96.csv'
WGCNA_DIR = ROOT / 'L1' / 'results' / 'wgcna_GSE16561'
ILMN_PROBE_FILE = ROOT / 'L1' / 'results' / 'ILMN_probe_to_gene.csv'
RSCRIPT_EXE = r'C:\R\R-4.5.2\bin\Rscript.exe'
STRING_SCORE_THRESHOLD = 700
MCODE_MIN_SIZE = 4
MCODE_MAX_DEPTH = 100


def load_genes_from_txt(path):
    """从纯文本文件加载基因列表（每行一个基因）。"""
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        genes = {line.strip() for line in f if line.strip()}
    logger.info('从 %s 加载: %d 个基因', path.name, len(genes))
    return genes


def load_ciri_candidates():
    """从 LASSO 筛选结果加载 CIRI-铁衰老候选基因。"""
    df = pd.read_csv(CIRI_CAND_FILE)
    genes = df['Gene_Human'].dropna().astype(str).str.strip().tolist()
    logger.info('CIRI-铁衰老候选基因: %d 个 - %s', len(genes), genes)
    return set(genes)


def build_string_gene_map():
    """构建 STRING protein ID -> gene symbol 映射。"""
    logger.info('构建 STRING ID -> Gene Symbol 映射...')
    id_to_gene = {}
    priority = ['Ensembl_HGNC_symbol', 'Ensembl_HGNC', 'HGNC', 'BioMart_HGNC_symbol']
    with open(STRING_ALIASES, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            sid, alias, source = parts[0], parts[1], parts[2]
            if source in priority:
                if sid not in id_to_gene or priority.index(source) < priority.index(id_to_gene[sid][1]):
                    id_to_gene[sid] = (alias, source)
    result = {k: v[0] for k, v in id_to_gene.items()}
    logger.info('STRING ID 映射: %d 条', len(result))
    return result


def load_string_ppi(id_to_gene):
    """加载 STRING PPI 网络：返回邻接表和所有基因集合。"""
    logger.info('加载 STRING PPI (combined_score > %d)...', STRING_SCORE_THRESHOLD)
    edges = defaultdict(set)
    all_genes = set()
    line_count = 0
    with open(STRING_LINKS, 'r') as f:
        next(f)
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                logger.info('  已处理 %dM 行...', line_count // 1000000)
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            score = float(parts[2])
            if score < STRING_SCORE_THRESHOLD:
                continue
            ga = id_to_gene.get(parts[0])
            gb = id_to_gene.get(parts[1])
            if not ga or not gb or ga == gb:
                continue
            edges[ga].add(gb)
            edges[gb].add(ga)
            all_genes.add(ga)
            all_genes.add(gb)
    n_edges = sum(len(v) for v in edges.values()) // 2
    logger.info('PPI 网络: %d 个基因, %d 条边', len(all_genes), n_edges)
    return edges, all_genes


def compute_network_proximity_ferroptosis(cary_targets, ferro_genes, ppi_edges, ppi_all_genes):
    """计算石竹烯靶点与铁死亡基因的网络邻近关系。

    返回:
      proximity_ferro: 铁死亡基因中与任一石竹烯靶点距离 ≤ 1 的基因集合
      direct_targets: 铁死亡基因 ∩ 石竹烯靶点 (直接药靶)
      proximity_stats: 富集统计
    """
    cary_in_ppi = cary_targets & ppi_all_genes
    ferro_in_ppi = ferro_genes & ppi_all_genes

    # 一阶邻居：石竹烯靶点的所有直接邻居
    cary_neighbors = set()
    for t in cary_in_ppi:
        cary_neighbors.update(ppi_edges.get(t, set()))
    cary_neighbors -= cary_in_ppi

    # 铁死亡基因中，属于石竹烯直接靶点或一阶邻居的
    direct_targets = ferro_in_ppi & cary_in_ppi
    neighbor_ferro = ferro_in_ppi & cary_neighbors
    proximity_ferro = direct_targets | neighbor_ferro

    # 超几何检验
    M = len(ppi_all_genes)
    n_ferro = len(ferro_in_ppi)
    N_proximity = len(cary_in_ppi) + len(cary_neighbors)
    k_prox = len(proximity_ferro & (cary_in_ppi | cary_neighbors))

    p_val = 1.0
    if k_prox > 0 and n_ferro > 0 and N_proximity > 0 and M > 0:
        p_val = 0.0
        for i in range(k_prox, min(n_ferro, N_proximity) + 1):
            log_p = (math.lgamma(n_ferro + 1) - math.lgamma(i + 1) - math.lgamma(n_ferro - i + 1) +
                     math.lgamma(M - n_ferro + 1) - math.lgamma(N_proximity - i + 1) -
                     math.lgamma(M - n_ferro - N_proximity + i + 1) -
                     (math.lgamma(M + 1) - math.lgamma(N_proximity + 1) - math.lgamma(M - N_proximity + 1)))
            p_val += math.exp(log_p)
        p_val = min(p_val, 1.0)

    logger.info('石竹烯靶点在PPI中: %d, 一阶邻居: %d', len(cary_in_ppi), len(cary_neighbors))
    logger.info('铁死亡基因在PPI中: %d', n_ferro)
    logger.info('直接靶点 (铁死亡 ∩ 石竹烯): %d 个', len(direct_targets))
    logger.info('邻近铁死亡基因 (一阶邻居): %d 个', len(neighbor_ferro))
    logger.info('汇聚: %d 个, 超几何 p=%.2e', len(proximity_ferro), p_val)

    return proximity_ferro, direct_targets, {'p_value': p_val, 'n_direct': len(direct_targets),
                                              'n_neighbor': len(neighbor_ferro), 'n_total': len(proximity_ferro)}


def compute_final_core_gene_set(ciri_candidates, cary_targets, proximity_ferro, ppi_edges, ppi_all_genes):
    """确定最终核心基因集。

    公式: 最终核心基因集 = (CIRI候选 ∩ 石竹烯靶点) ∪ 网络邻近扩展的铁死亡关键基因
    """
    cary_in_ppi = cary_targets & ppi_all_genes
    ciri_in_ppi = ciri_candidates & ppi_all_genes

    # Part A: CIRI-铁衰老候选基因 ∩ 石竹烯高置信度靶点
    part_a = ciri_in_ppi & cary_in_ppi
    logger.info('Part A (CIRI ∩ 石竹烯): %d 个 - %s', len(part_a), sorted(part_a))

    # Part B: 网络邻近扩展的铁死亡关键基因
    part_b = proximity_ferro
    logger.info('Part B (邻近铁死亡关键基因): %d 个', len(part_b))

    # 合并
    core_genes = part_a | part_b
    logger.info('最终核心基因集: %d 个基因', len(core_genes))

    return core_genes, part_a, part_b


def build_core_ppi_network(core_genes, ppi_edges):
    """为核心基因集构建 STRING PPI 子网络，计算拓扑参数。"""
    logger.info('构建核心基因 PPI 网络...')
    G = nx.Graph()
    core_in_ppi = {g for g in core_genes if g in ppi_edges}

    for g in core_in_ppi:
        G.add_node(g)
        for neighbor in ppi_edges[g]:
            if neighbor in core_in_ppi and g < neighbor:
                G.add_edge(g, neighbor)

    logger.info('核心 PPI: %d 节点, %d 边', G.number_of_nodes(), G.number_of_edges())

    if G.number_of_nodes() == 0:
        logger.warning('核心基因集中无基因在 STRING PPI 中，返回空网络')
        return G, pd.DataFrame()

    # Degree
    degree_dict = dict(G.degree())

    # Betweenness centrality
    logger.info('计算 Betweenness centrality...')
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)

    # Closeness centrality
    closeness_dict = nx.closeness_centrality(G)

    # Eigenvector centrality
    try:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-5)
    except nx.PowerIterationFailedConvergence:
        logger.warning('Eigenvector 不收敛，使用近似值')
        eigenvector_dict = {n: 0.0 for n in G.nodes()}

    node_df = pd.DataFrame({
        'Gene': list(G.nodes()),
        'Degree': [degree_dict[n] for n in G.nodes()],
        'Betweenness': [betweenness_dict[n] for n in G.nodes()],
        'Closeness': [closeness_dict[n] for n in G.nodes()],
        'Eigenvector': [eigenvector_dict[n] for n in G.nodes()]
    })
    node_df = node_df.sort_values('Degree', ascending=False).reset_index(drop=True)
    node_df['Degree_Rank'] = node_df['Degree'].rank(ascending=False, method='min').astype(int)
    node_df['Betweenness_Rank'] = node_df['Betweenness'].rank(ascending=False, method='min').astype(int)

    logger.info('拓扑参数计算完成')
    logger.info('Top 10 Hub 基因 (按 Degree): %s',
                node_df.head(10)[['Gene', 'Degree', 'Betweenness']].to_string(index=False))

    return G, node_df


def run_mcode(G, min_size=MCODE_MIN_SIZE):
    """识别 PPI 网络中的紧密连接模块。

    使用 NetworkX greedy_modularity_communities 结合子图密度过滤，
    实现类似 MCODE 的效果：识别高度内聚的功能性亚簇。
    """
    logger.info('运行模块识别 (min_size=%d)...', min_size)
    if G.number_of_nodes() < min_size:
        logger.warning('图过小 (n=%d)，模块识别跳过', G.number_of_nodes())
        return []

    from networkx.algorithms.community import greedy_modularity_communities

    communities = list(greedy_modularity_communities(G, weight=None, resolution=1.2))
    logger.info('Greedy modularity 识别到 %d 个社区', len(communities))

    modules = []
    visited = set()
    for i, comm in enumerate(communities):
        if len(comm) < min_size:
            continue
        subgraph = G.subgraph(comm)
        density = nx.density(subgraph)
        if density < 0.05:
            continue
        seed = max(comm, key=lambda n: G.degree(n))
        modules.append({
            'module_id': len(modules) + 1,
            'genes': sorted(comm),
            'size': len(comm),
            'density': round(density, 4),
            'seed_gene': seed,
            'seed_degree': G.degree(seed)
        })
        visited.update(comm)

    logger.info('过滤后得到 %d 个模块 (min_size=%d, min_density=0.05)', len(modules), min_size)
    for m in modules:
        logger.info('  Module %d: size=%d, density=%.3f, seed=%s, genes=%s...',
                    m['module_id'], m['size'], m['density'], m['seed_gene'], m['genes'][:5])
    return modules


def run_go_kegg_enrichment(core_gene_list):
    """通过 gseapy (Enrichr API) 进行 GO/KEGG 富集分析。"""
    import gseapy as gp

    gene_list = sorted(core_gene_list)
    logger.info('通过 gseapy/Enrichr API 进行 GO/KEGG 富集 (%d 个基因)...', len(gene_list))

    gene_set_libraries = {
        'GO_Biological_Process_2023': 'core_go_bp_enrichment.csv',
        'GO_Molecular_Function_2023': 'core_go_mf_enrichment.csv',
        'GO_Cellular_Component_2023': 'core_go_cc_enrichment.csv',
        'KEGG_2021_Human': 'core_kegg_enrichment.csv',
        'WikiPathway_2023_Human': 'core_wikipathway_enrichment.csv',
        'Reactome_2022': 'core_reactome_enrichment.csv'
    }

    for library, filename in gene_set_libraries.items():
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=library,
                organism='human',
                outdir=None,
                no_plot=True,
                cutoff=1.0  # 不设阈值，获取全部结果
            )
            if enr is None or enr.results is None or len(enr.results) == 0:
                logger.info('%s: 无显著富集项', library)
                continue

            df = enr.results.copy()
            df = df.sort_values('Adjusted P-value')
            df.to_csv(RESULT_DIR / filename, index=False, encoding='utf-8-sig')
            logger.info('%s: %d 个富集项, Top: %s (p.adj=%.2e)',
                        library, len(df),
                        df.iloc[0]['Term'] if len(df) > 0 else 'None',
                        df.iloc[0]['Adjusted P-value'] if len(df) > 0 else np.nan)
        except Exception as e:
            logger.warning('%s: 富集分析失败 - %s', library, e)


def validate_wgcna(core_genes):
    """利用已有 WGCNA 结果验证核心基因的模块身份和基因显著性。"""
    logger.info('WGCNA 共表达模块验证...')

    # 加载探针-基因映射
    ilmn_df = pd.read_csv(ILMN_PROBE_FILE)
    probe_to_gene = dict(zip(ilmn_df['Probe'], ilmn_df['GeneSymbol']))

    # 加载模块分配
    gm_df = pd.read_csv(WGCNA_DIR / 'gene_module_assignment.csv')

    def map_probe(pid):
        if pid in probe_to_gene and pd.notna(probe_to_gene[pid]) and probe_to_gene[pid].strip():
            return probe_to_gene[pid].strip().upper()
        return None

    gm_df['GeneSymbol'] = gm_df['Gene'].apply(map_probe)
    gm_mapped = gm_df[gm_df['GeneSymbol'].notna()].copy()

    # 加载模块-性状相关性
    mt_cor = pd.read_csv(WGCNA_DIR / 'module_trait_correlation.csv')
    mt_pval = pd.read_csv(WGCNA_DIR / 'module_trait_pvalue.csv')

    # 确定性状列名
    cor_cols = [c for c in mt_cor.columns if c != 'Module']
    trait_col = cor_cols[0] if cor_cols else 'Stroke'
    pval_cols = [c for c in mt_pval.columns if c != 'Module']
    pval_col = pval_cols[0] if pval_cols else trait_col

    mt_cor[trait_col] = pd.to_numeric(mt_cor[trait_col], errors='coerce')
    mt_pval[pval_col] = pd.to_numeric(mt_pval[pval_col], errors='coerce')
    mt_cor['abs_cor'] = mt_cor[trait_col].abs()
    mt_cor['pval'] = mt_pval[pval_col]

    # 找出与性状显著相关的模块 (|cor|>0.3, p<0.05)
    sig_modules = mt_cor[(mt_cor['abs_cor'] > 0.3) & (mt_cor['pval'] < 0.05)]
    sig_module_names = set()
    for _, row in sig_modules.iterrows():
        mn = row['Module']
        if mn.startswith('ME'):
            sig_module_names.add(mn[2:])
    logger.info('WGCNA 显著模块: %s (n=%d)', sorted(sig_module_names), len(sig_module_names))

    # 加载 MM 和 GS
    mm_df = pd.read_csv(WGCNA_DIR / 'module_membership.csv')
    mm_df['GeneSymbol'] = mm_df['Gene'].apply(map_probe)
    mm_df = mm_df[mm_df['GeneSymbol'].notna()]

    gs_df = pd.read_csv(WGCNA_DIR / 'gene_significance.csv')
    gs_df['GeneSymbol'] = gs_df['Gene'].apply(map_probe)
    gs_df = gs_df[gs_df['GeneSymbol'].notna()]

    # 对每个核心基因，找其所属的 WGCNA 模块
    gene_module_map = {}
    for _, row in gm_mapped.iterrows():
        g = row['GeneSymbol']
        if g not in gene_module_map:
            gene_module_map[g] = row['Module']

    # 为每个基因找最高 MM 的模块
    mm_cols = [c for c in mm_df.columns if c not in ('Gene', 'GeneSymbol')]
    gs_col = trait_col if trait_col in gs_df.columns else gs_df.columns[0]

    results = []
    for gene in sorted(core_genes):
        gene_upper = gene.upper()

        # 模块身份
        wgcna_module = gene_module_map.get(gene_upper, 'grey')

        # 最高 MM 及其模块
        gene_mm = mm_df[mm_df['GeneSymbol'] == gene_upper]
        best_mm_val = np.nan
        best_mm_mod = 'grey'
        if len(gene_mm) > 0:
            mm_vals = gene_mm[mm_cols].iloc[0]
            best_mm_idx = mm_vals.abs().idxmax()
            best_mm_val = mm_vals[best_mm_idx]
            best_mm_mod = best_mm_idx

        # GS
        gene_gs = gs_df[gs_df['GeneSymbol'] == gene_upper]
        gs_val = float(gene_gs[gs_col].iloc[0]) if len(gene_gs) > 0 else np.nan

        # 是否在显著模块中
        in_sig_module = wgcna_module in sig_module_names

        results.append({
            'Gene': gene,
            'WGCNA_Module': wgcna_module,
            'Best_MM_Module': best_mm_mod,
            'Best_MM': best_mm_val,
            'GS': gs_val,
            'In_Significant_Module': in_sig_module,
            'MM_High_GS_High': (not np.isnan(best_mm_val) and best_mm_val > 0.5 and
                               not np.isnan(gs_val) and abs(gs_val) > 0.1)
        })

    wgcna_val_df = pd.DataFrame(results)
    wgcna_val_df = wgcna_val_df.sort_values('Best_MM', ascending=False, na_position='last')

    n_validated = wgcna_val_df['In_Significant_Module'].sum()
    n_high = wgcna_val_df['MM_High_GS_High'].sum()
    logger.info('核心基因 WGCNA 验证: %d/%d 位于显著共表达模块, %d 具有高MM+高GS',
                n_validated, len(results), n_high)
    for _, row in wgcna_val_df.iterrows():
        logger.info('  %-12s Module=%-12s MM=%.3f GS=%.3f Sig=%s High=%s',
                    row['Gene'], row['WGCNA_Module'], row['Best_MM'], row['GS'],
                    row['In_Significant_Module'], row['MM_High_GS_High'])

    return wgcna_val_df


def check_pathway_enrichment(core_genes):
    """检查核心基因是否显著富集于铁死亡、衰老、炎症、缺血相关通路。"""
    ferro_pathway_genes = {
        'Ferroptosis': {'GPX4', 'ACSL4', 'SLC7A11', 'TFRC', 'FTH1', 'NFE2L2', 'HMOX1', 'PTGS2',
                        'SAT1', 'LOX', 'ALOX5', 'ALOX12', 'ALOX15', 'VDAC2', 'VDAC1', 'GLS2',
                        'LPCAT3', 'DPP4', 'CARS1', 'CS', 'RPL8', 'IREB2', 'ATP5MC3', 'TFR1',
                        'STEAP1', 'SLC11A2', 'NCOA4', 'PCBP1', 'PCBP2', 'MAP1LC3A', 'ATG5',
                        'ATG7', 'BECN1', 'SQSTM1', 'HMGB1', 'KEAP1', 'CDKN2A', 'TP53',
                        'CAV1', 'NOX1', 'NOX4', 'PRNP', 'CHAC1', 'ABCC1', 'AIFM2', 'FSP1',
                        'SLC39A14', 'SLC39A8', 'SLC40A1', 'CP', 'TF', 'FXN', 'ISCU', 'NFE2L1',
                        'MTOR', 'PRKAA1', 'PRKAA2', 'AMPK', 'STAT3', 'NFKB1', 'RELA',
                        'EGFR', 'KRAS', 'HRAS', 'NRAS', 'BRAF', 'MAPK1', 'MAPK3',
                        'BCL2', 'BAX', 'BID', 'CASP3', 'PARP1', 'ATG13', 'ATG3', 'ATG4D',
                        'NCOA4', 'TXNIP', 'HIF1A', 'EGLN2', 'VHL', 'EPAS1', 'SIRT1',
                        'BRD4', 'BRD7', 'MDM2', 'CDKN1A', 'GDF15', 'LACTB', 'MPO',
                        'CYBB', 'NOX3', 'DUOX1', 'DUOX2', 'POR', 'FMO1', 'NQO1'},
        'Cellular_Senescence': {'CDKN2A', 'CDKN1A', 'TP53', 'RB1', 'LMNB1', 'HMGB1', 'HMGB2',
                                'IL6', 'IL1B', 'TNF', 'CXCL8', 'CCL2', 'CXCL10', 'MMP1',
                                'MMP3', 'MMP9', 'IGFBP3', 'IGFBP7', 'TGFB1', 'TGFB2',
                                'SERPINE1', 'CDKN2B', 'H2AFX', 'CHEK1', 'ATM', 'ATR',
                                'SIRT1', 'SIRT6', 'FOXO3', 'FOXO4', 'NFE2L2', 'GATA4',
                                'MAPK14', 'PTGS2', 'BCL2', 'BCL6', 'E2F1', 'E2F3',
                                'SP1', 'EGR1', 'IRF1', 'STAT3', 'TLR4', 'NFKB1',
                                'KDM6B', 'EHMT2', 'SETD7', 'SUV39H1', 'CBX5',
                                'LGMN', 'CTSB', 'CTSD', 'GBA', 'TREM2', 'APOE',
                                'CD74', 'HLA-DRA', 'CDKN1C', 'DEC1', 'GLB1', 'WRN',
                                'BLM', 'RECQL4', 'TERF2', 'TERT', 'P16', 'P21',
                                'RB', 'RBL1', 'RBL2', 'EZH2', 'BMI1', 'SUZ12',
                                'SAT1', 'ODC1', 'SRM', 'SMS', 'AMD1', 'PAOX',
                                'CD38', 'NAMPT', 'NNMT', 'SASP', 'CXCL1', 'CXCL2'},
        'Neuroinflammation_Ischemia': {'TLR4', 'NFKB1', 'RELA', 'IL1B', 'IL6', 'TNF', 'CXCL10',
                                       'CCL2', 'ICAM1', 'VCAM1', 'MMP9', 'PTGS2', 'NOS2',
                                       'HIF1A', 'VEGFA', 'BDNF', 'NGF', 'GDNF', 'BCL2',
                                       'BAX', 'CASP3', 'CASP8', 'CASP9', 'PARP1', 'AIFM1',
                                       'SOD1', 'SOD2', 'CAT', 'GPX1', 'GSR', 'TXN',
                                       'HMOX1', 'NFE2L2', 'KEAP1', 'MAPK1', 'MAPK3', 'MAPK8',
                                       'MAPK9', 'MAPK14', 'AKT1', 'MTOR', 'PIK3CA',
                                       'EGFR', 'SRC', 'STAT3', 'JAK2', 'EPO', 'EPOR',
                                       'GRIN1', 'GRIN2A', 'GRIN2B', 'GRIA1', 'GRIA2',
                                       'DLG4', 'SYN1', 'SYP', 'GAP43', 'MAP2',
                                       'GFAP', 'AQP4', 'SLC1A2', 'SLC1A3', 'CLDN5',
                                       'OCLN', 'TJP1', 'CDH5', 'PECAM1', 'ENG',
                                       'NOTCH1', 'NOTCH2', 'DLL4', 'JAG1', 'WNT3A', 'WNT5A',
                                       'SHH', 'PTCH1', 'GLI1', 'BMP2', 'BMP4', 'BMP7',
                                       'APP', 'PSEN1', 'MAPT', 'SNCA', 'TARDBP', 'FUS',
                                       'ATXN1', 'ATXN3', 'HTT', 'PRNP', 'TREM2', 'TYROBP',
                                       'ITGAM', 'ITGB2', 'CSF1R', 'CX3CR1', 'P2RY12',
                                       'TMEM119', 'HEXB', 'SPP1', 'LPL', 'CD68', 'AIF1'}
    }

    core_set = {g.upper() for g in core_genes}
    results = []
    for pathway_name, pathway_gene_set in ferro_pathway_genes.items():
        pathway_upper = {g.upper() for g in pathway_gene_set}
        overlap = core_set & pathway_upper
        results.append({
            'Pathway_Category': pathway_name,
            'Overlap_Count': len(overlap),
            'Overlap_Genes': ';'.join(sorted(overlap)),
            'Total_Core': len(core_set),
            'Pathway_Size': len(pathway_upper)
        })

    pw_df = pd.DataFrame(results)
    logger.info('通路富集检查:')
    for _, row in pw_df.iterrows():
        logger.info('  %s: %d/%d 核心基因命中 - %s',
                    row['Pathway_Category'], row['Overlap_Count'],
                    row['Total_Core'], row['Overlap_Genes'][:80])
    return pw_df


def main():
    logger.info('=' * 70)
    logger.info('最终核心基因集确定与多维度验证管线')
    logger.info('=' * 70)

    t_start = time.time()

    # ========== Step 1: 加载基因数据 ==========
    logger.info('[Step 1/7] 加载多源基因数据...')
    ciri_candidates = load_ciri_candidates()
    cary_targets = load_genes_from_txt(CARY_FILE)
    ferro_genes = load_genes_from_txt(FERRO_FILE)
    id_to_gene = build_string_gene_map()
    ppi_edges, ppi_all_genes = load_string_ppi(id_to_gene)

    # ========== Step 2: 确定最终核心基因集 ==========
    logger.info('[Step 2/7] 计算核心基因集...')
    proximity_ferro, direct_ferro_targets, prox_stats = compute_network_proximity_ferroptosis(
        cary_targets, ferro_genes, ppi_edges, ppi_all_genes)
    core_genes, part_a, part_b = compute_final_core_gene_set(
        ciri_candidates, cary_targets, proximity_ferro, ppi_edges, ppi_all_genes)

    logger.info('')
    logger.info('=== 最终核心基因集 (%d 个基因) ===' , len(core_genes))
    logger.info('%s', sorted(core_genes))
    logger.info('  Part A (CIRI ∩ 石竹烯): %d 个 - %s', len(part_a), sorted(part_a))
    logger.info('  Part B (邻近铁死亡): %d 个', len(part_b))

    # ========== Step 3: STRING PPI 网络 + 拓扑分析 ==========
    logger.info('[Step 3/7] 构建核心基因 PPI 网络并计算拓扑参数...')
    G_core, node_df = build_core_ppi_network(core_genes, ppi_edges)

    # ========== Step 4: MCODE 模块识别 ==========
    logger.info('[Step 4/7] MCODE 模块识别...')
    mcode_modules = run_mcode(G_core)

    # ========== Step 5: GO/KEGG 富集分析 ==========
    logger.info('[Step 5/7] GO/KEGG 富集分析 (Enrichr API)...')
    run_go_kegg_enrichment(sorted(core_genes))

    # ========== Step 6: 通路富集检查 ==========
    logger.info('[Step 6/7] 铁死亡/衰老/炎症/缺血通路富集检查...')
    pw_df = check_pathway_enrichment(core_genes)

    # ========== Step 7: WGCNA 验证 ==========
    logger.info('[Step 7/7] WGCNA 共表达模块验证...')
    wgcna_val_df = validate_wgcna(core_genes)

    # ========== 保存结果 ==========
    logger.info('保存所有结果...')

    # 核心基因集
    core_df = pd.DataFrame({
        'Gene': sorted(core_genes),
        'Source': ['PartA_CIRI_Caryophyllene' if g in part_a else 'PartB_Ferroptosis_Proximity'
                   for g in sorted(core_genes)]
    })
    core_df.to_csv(RESULT_DIR / 'final_core_gene_set.csv', index=False, encoding='utf-8-sig')

    # PPI 网络边
    if G_core.number_of_edges() > 0:
        edge_list = [(u, v) for u, v in G_core.edges()]
        edge_df = pd.DataFrame(edge_list, columns=['Gene_A', 'Gene_B'])
        edge_df.to_csv(RESULT_DIR / 'core_ppi_edges.csv', index=False, encoding='utf-8-sig')

    # 拓扑参数
    if len(node_df) > 0:
        node_df.to_csv(RESULT_DIR / 'core_ppi_topology.csv', index=False, encoding='utf-8-sig')

    # MCODE 模块
    if mcode_modules:
        mcode_rows = []
        for m in mcode_modules:
            mcode_rows.append({
                'Module_ID': m['module_id'],
                'Size': m['size'],
                'Density': m['density'],
                'Seed_Gene': m['seed_gene'],
                'Genes': ';'.join(m['genes'])
            })
        pd.DataFrame(mcode_rows).to_csv(RESULT_DIR / 'core_mcode_modules.csv', index=False, encoding='utf-8-sig')

    # 通路检查
    pw_df.to_csv(RESULT_DIR / 'core_pathway_enrichment_check.csv', index=False, encoding='utf-8-sig')

    # WGCNA 验证
    wgcna_val_df.to_csv(RESULT_DIR / 'core_wgcna_validation.csv', index=False, encoding='utf-8-sig')

    # 汇总 JSON
    summary = {
        'final_core_gene_count': len(core_genes),
        'core_genes': sorted(core_genes),
        'part_a_ciri_caryophyllene': sorted(part_a),
        'part_a_count': len(part_a),
        'part_b_ferroptosis_proximity_count': len(part_b),
        'ppi_nodes': G_core.number_of_nodes(),
        'ppi_edges': G_core.number_of_edges(),
        'mcode_modules': len(mcode_modules),
        'mcode_module_sizes': [m['size'] for m in mcode_modules],
        'wgcna_genes_in_sig_module': int(wgcna_val_df['In_Significant_Module'].sum()),
        'wgcna_genes_high_mm_gs': int(wgcna_val_df['MM_High_GS_High'].sum()),
        'network_proximity_p_value': prox_stats['p_value'],
        'ferroptosis_direct_targets': len(direct_ferro_targets),
        'top_hubs': node_df.head(10)[['Gene', 'Degree', 'Betweenness']].to_dict('records') if len(node_df) > 0 else [],
        'pathway_check': pw_df.to_dict('records')
    }
    with open(RESULT_DIR / 'core_gene_pipeline_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    logger.info('')
    logger.info('=' * 70)
    logger.info('管线完成! 耗时: %.1f 秒', elapsed)
    logger.info('最终核心基因集: %d 个', len(core_genes))
    logger.info('PPI 节点: %d, 边: %d', G_core.number_of_nodes(), G_core.number_of_edges())
    logger.info('MCODE 模块: %d 个', len(mcode_modules))
    logger.info('WGCNA 显著模块基因: %d/%d', wgcna_val_df['In_Significant_Module'].sum(), len(wgcna_val_df))
    logger.info('结果保存至: %s', RESULT_DIR)
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
