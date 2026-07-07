"""图构建：通路邻居预计算 + 同质图/异质图构建 + 邻接表预计算

v24: 支持疾病节点（GSE61616）和铁衰老96基因全蛋白集扩展
v23-topo: 可选 PPI 拓扑负样本预计算
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import HeteroData

from ..data.features import build_compound_features

logger = logging.getLogger(__name__)


def build_pathway_neighbors(
    gene_to_pathways: dict[str, list[str]],
    gene_to_idx: dict[str, int],
    n_compounds: int,
) -> dict[int, set]:
    """预计算同通路蛋白邻居（用于中度负样本采样）

    Returns:
        prot_to_path_neighbors: {蛋白局部索引: set(同通路其他蛋白局部索引)}
    """
    pathway_to_genes: dict[str, set] = defaultdict(set)
    for gene, paths in gene_to_pathways.items():
        if gene not in gene_to_idx:
            continue
        g_idx = gene_to_idx[gene] - n_compounds
        if g_idx < 0:
            continue
        for p in paths:
            pathway_to_genes[p].add(g_idx)

    prot_to_path_neighbors: dict[int, set] = defaultdict(set)
    for genes in pathway_to_genes.values():
        for g in genes:
            prot_to_path_neighbors[g].update(genes - {g})

    return prot_to_path_neighbors


def build_graphs_and_adj(
    cpi_df: pd.DataFrame,
    ppi_df: pd.DataFrame,
    gene_to_pathways: dict[str, list[str]],
    prot_feat: dict[str, np.ndarray],
    disease_df: pd.DataFrame | None = None,
    ferro96_genes_file: str | Path | None = None,
    use_topology_neg: bool = False,
    topo_neighbors_top_k: int = 50,
    use_esm_similarity_neg: bool = False,
    esm_similarity_top_k: int = 50,
    use_compound_similarity_edges: bool = True,
    comp_sim_threshold: float = 0.7,
    comp_sim_top_k: int = 10,
):
    """构建同质图 + 异质图 + 邻接表（可选疾病节点和拓扑/ESM-2负样本）

    加载铁衰老96基因集，确保所有核心基因作为蛋白节点（用于zero-shot预测）。
    预计算基于PPI拓扑和ESM-2余弦相似度的负样本邻居（可选，默认关闭）。
    """
    # 化合物索引
    all_smiles = sorted(cpi_df["canonical_smiles"].unique())
    smi_to_idx = {s: i for i, s in enumerate(all_smiles)}
    n_compounds = len(all_smiles)

    # 蛋白索引
    ppi_genes = set()
    for _, row in ppi_df.iterrows():
        ppi_genes.add(str(row["source"]).strip().upper())
        ppi_genes.add(str(row["target"]).strip().upper())
    # 确保铁衰老96基因全部作为蛋白节点，即使无CPI/PPI数据（用于zero-shot预测）
    ferro96_genes = set()
    if ferro96_genes_file is not None:
        ferro96_path = Path(ferro96_genes_file)
        if ferro96_path.exists():
            ferro96_genes = set(pd.read_csv(ferro96_path)["gene_symbol"].dropna().astype(str).str.upper().unique())
            logger.info(f"铁衰老96基因集加载: {len(ferro96_genes)} 个")
    all_genes = sorted(set(cpi_df["gene"].unique()) | set(prot_feat.keys()) | ppi_genes | ferro96_genes)
    gene_to_idx = {g: i + n_compounds for i, g in enumerate(all_genes)}
    n_proteins = len(all_genes)
    logger.info(f"总蛋白节点 = {n_proteins} (CPI={cpi_df['gene'].nunique()}, "
                f"PPI网络={len(ppi_genes)}, 铁衰老96={len(ferro96_genes)})")

    # 化合物特征
    logger.info(f"  computing compound features ({n_compounds} compounds)...")
    comp_feat, _, _, _ = build_compound_features(all_smiles)

    # 化合物-化合物 Tanimoto 相似性边 — 解决冷启动验证时化合物节点孤立问题
    comp_sim_edges: list[tuple[int, int]] = []
    if use_compound_similarity_edges and n_compounds > 1:
        logger.info(f"  computing compound-compound similarity edges (threshold={comp_sim_threshold}, top_k={comp_sim_top_k})...")
        try:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
            ecfp4_fps = []
            for smi in all_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    ecfp4_fps.append(None)
                else:
                    ecfp4_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            n_comp_sim_edges = 0
            for i in range(n_compounds):
                if ecfp4_fps[i] is None:
                    continue
                sims = []
                for j in range(n_compounds):
                    if i >= j or ecfp4_fps[j] is None:
                        continue
                    sim = DataStructs.TanimotoSimilarity(ecfp4_fps[i], ecfp4_fps[j])
                    if sim >= comp_sim_threshold:
                        sims.append((j, sim))
                sims.sort(key=lambda x: x[1], reverse=True)
                for j, _sim in sims[:comp_sim_top_k]:
                    comp_sim_edges.append((i, j))
                    comp_sim_edges.append((j, i))
                    n_comp_sim_edges += 2
            logger.info(f"  compound-compound similarity edges: {n_comp_sim_edges} (threshold={comp_sim_threshold})")
        except Exception as e:
            logger.warning(f"  compound similarity edge computation failed ({e}), skipping")

    # 蛋白特征
    prot_feat_dim = next(iter(prot_feat.values())).shape[0] if prot_feat else 20
    prot_esm_dim = prot_feat_dim  # 原始 ESM-2 维度，供独立投影器使用
    prot_matrix = np.zeros((n_proteins, prot_feat_dim), dtype=np.float32)
    n_no_feat = 0
    for gene, idx_offset in gene_to_idx.items():
        idx = idx_offset - n_compounds
        if gene in prot_feat:
            prot_matrix[idx] = prot_feat[gene]
        else:
            prot_matrix[idx] = 0.0
            n_no_feat += 1
    if n_no_feat > 0:
        logger.warning(f"  无蛋白特征基因（缺失特征，已用零填充）: {n_no_feat}")

    # 通路隶属关系特征 — 为 SAGE 模型添加通路信息，弥补蛋白冷启动场景下拓扑缺失
    # 将每个蛋白的 KEGG 通路隶属关系编码为 one-hot 向量，拼接到 ESM-2 嵌入后
    all_pathways = sorted({pid for paths in gene_to_pathways.values() for pid in paths})
    pathway_to_idx = {p: i for i, p in enumerate(all_pathways)}
    n_pathways_feat = len(all_pathways)

    if n_pathways_feat > 0:
        pathway_feat = np.zeros((n_proteins, n_pathways_feat), dtype=np.float32)
        n_proteins_with_path = 0
        for gene, paths in gene_to_pathways.items():
            if gene in gene_to_idx:
                p_idx = gene_to_idx[gene] - n_compounds
                for pid in paths:
                    if pid in pathway_to_idx:
                        pathway_feat[p_idx, pathway_to_idx[pid]] = 1.0
                        n_proteins_with_path += 1
        # 去重计数（每个蛋白只计一次）
        n_proteins_with_path = int((pathway_feat.sum(axis=1) > 0).sum())
        logger.info(f"  通路特征 (one-hot): {n_proteins_with_path}/{n_proteins} 蛋白有通路信息, "
                    f"n_pathways={n_pathways_feat}")

        # 拼接通路特征到蛋白特征矩阵
        prot_matrix = np.concatenate([prot_matrix, pathway_feat], axis=1)
        prot_feat_dim = prot_matrix.shape[1]  # 更新蛋白特征维度
    else:
        logger.warning("  通路特征为空，跳过通路信息添加")

    # 统一维度
    feat_dim = max(comp_feat.shape[1], prot_feat_dim)
    if feat_dim != comp_feat.shape[1]:
        comp_feat = np.pad(comp_feat, ((0, 0), (0, feat_dim - comp_feat.shape[1])), mode="constant")
    if feat_dim != prot_feat_dim:
        prot_matrix = np.pad(prot_matrix, ((0, 0), (0, feat_dim - prot_feat_dim)), mode="constant")

    x = torch.from_numpy(np.vstack([comp_feat, prot_matrix]))
    logger.info(f"同质图: {n_compounds + n_proteins} 节点, feat_dim={feat_dim}, prot_raw_dim={prot_feat_dim}")

    # ======== 邻接表构建 ========
    # 同质图邻接表（用于 GAT 分支采样）
    homo_adj = defaultdict(list)  # node_idx -> [neighbor_indices]

    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            src = smi_to_idx[smi]
            dst = gene_to_idx[gene]
            homo_adj[src].append(dst)
            homo_adj[dst].append(src)

    n_ppi_edges = 0
    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            si = gene_to_idx[src]
            ti = gene_to_idx[tgt]
            homo_adj[si].append(ti)
            homo_adj[ti].append(si)
            n_ppi_edges += 1

    n_comp_sim_adj = 0
    for src, dst in comp_sim_edges:
        if src < n_compounds and dst < n_compounds:
            homo_adj[src].append(dst)
            n_comp_sim_adj += 1

    logger.info(f"同质图邻接: {len(homo_adj)} 节点, {n_ppi_edges} PPI 边, {n_comp_sim_adj} 化合物相似性边")

    # 异质图邻接表（用于 HGT 分支采样）
    hetero_adj = {
        ("compound", "interacts", "protein"): defaultdict(list),
        ("compound", "similar_to", "compound"): defaultdict(list),
        ("protein", "ppi", "protein"): defaultdict(list),
        ("protein", "belongs_to", "pathway"): defaultdict(list),
        ("protein", "associated_with", "disease"): defaultdict(list),
        ("disease", "involves", "protein"): defaultdict(list),
    }

    # 疾病节点（GSE61616 Ferroaging DEGs）
    disease_names = []
    disease_to_idx = {}
    n_diseases = 0
    if disease_df is not None and not disease_df.empty:
        disease_names = sorted(disease_df["disease_name"].unique())
        disease_to_idx = {d: i for i, d in enumerate(disease_names)}
        n_diseases = len(disease_names)
        logger.info(f"  疾病节点: {n_diseases} 个, 疾病-蛋白边来源={len(disease_df)} 条")

    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in smi_to_idx and gene in gene_to_idx:
            hetero_adj[("compound", "interacts", "protein")][smi_to_idx[smi]].append(
                gene_to_idx[gene] - n_compounds)

    for src, dst in comp_sim_edges:
        if src < n_compounds and dst < n_compounds:
            hetero_adj[("compound", "similar_to", "compound")][src].append(dst)

    for _, row in ppi_df.iterrows():
        src = str(row["source"]).strip().upper()
        tgt = str(row["target"]).strip().upper()
        if src in gene_to_idx and tgt in gene_to_idx:
            hetero_adj[("protein", "ppi", "protein")][gene_to_idx[src] - n_compounds].append(
                gene_to_idx[tgt] - n_compounds)

    for gene, paths in gene_to_pathways.items():
        if gene in gene_to_idx:
            p_idx = gene_to_idx[gene] - n_compounds
            for pid in paths:
                hetero_adj[("protein", "belongs_to", "pathway")][p_idx].append(pid)

    # 疾病-蛋白边构建（蛋白局部索引 <-> 疾病整数索引）
    if disease_df is not None and not disease_df.empty:
        for _, row in disease_df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            d_name = str(row["disease_name"]).strip()
            if gene in gene_to_idx and d_name in disease_to_idx:
                p_idx = gene_to_idx[gene] - n_compounds
                d_idx = disease_to_idx[d_name]
                hetero_adj[("protein", "associated_with", "disease")][p_idx].append(d_idx)
                hetero_adj[("disease", "involves", "protein")][d_idx].append(p_idx)

    # 通路索引（已在特征构建阶段计算，此处复用）
    n_pathways = n_pathways_feat

    # 通路ID完全数值化 — 将邻接表中的字符串通路ID转为整数索引，消除字符串匹配开销
    new_pt_adj = defaultdict(list)
    for prot_idx, path_list in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for pid in path_list:
            if pid in pathway_to_idx:
                new_pt_adj[prot_idx].append(pathway_to_idx[pid])
    hetero_adj[("protein", "belongs_to", "pathway")] = new_pt_adj
    logger.info(f"  通路ID数值化完成: {len(new_pt_adj)} 蛋白 → {n_pathways} 通路")

    # 预计算同通路蛋白邻居（用于中度负样本采样）
    prot_to_path_neighbors = build_pathway_neighbors(gene_to_pathways, gene_to_idx, n_compounds)
    logger.info(f"  同通路蛋白邻居: {len(prot_to_path_neighbors)} 蛋白")

    # 异质图数据（用于 HGT 全图验证）
    hetero_data = HeteroData()
    hetero_data["compound"].x = torch.from_numpy(comp_feat)
    hetero_data["protein"].x = torch.from_numpy(prot_matrix)
    hetero_data["pathway"].x = torch.zeros(max(n_pathways, 1), 1, dtype=torch.float32)
    hetero_data["pathway"].n_pathways = n_pathways

    # CPI 边
    cpi_edges = [[], []]
    for src, dsts in hetero_adj[("compound", "interacts", "protein")].items():
        for dst in dsts:
            cpi_edges[0].append(src)
            cpi_edges[1].append(dst)
    hetero_data["compound", "interacts", "protein"].edge_index = torch.tensor(cpi_edges, dtype=torch.long)

    # 化合物-化合物相似性边
    comp_sim_edges_tensor = [[], []]
    for src, dsts in hetero_adj[("compound", "similar_to", "compound")].items():
        for dst in dsts:
            comp_sim_edges_tensor[0].append(src)
            comp_sim_edges_tensor[1].append(dst)
    hetero_data["compound", "similar_to", "compound"].edge_index = torch.tensor(
        comp_sim_edges_tensor, dtype=torch.long
    ) if comp_sim_edges_tensor[0] else torch.zeros((2, 0), dtype=torch.long)
    # 反向边（用于双向消息传递）
    rev_comp_sim = [comp_sim_edges_tensor[1][:], comp_sim_edges_tensor[0][:]]
    hetero_data["compound", "rev_similar_to", "compound"].edge_index = torch.tensor(
        rev_comp_sim, dtype=torch.long
    ) if rev_comp_sim[0] else torch.zeros((2, 0), dtype=torch.long)

    # PPI 边
    ppi_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "ppi", "protein")].items():
        for dst in dsts:
            ppi_edges[0].append(src)
            ppi_edges[1].append(dst)
    hetero_data["protein", "ppi", "protein"].edge_index = torch.tensor(ppi_edges, dtype=torch.long)

    # 通路边（通路ID已数值化，dst 已是整数，无需再次转换）
    pt_edges = [[], []]
    for src, dsts in hetero_adj[("protein", "belongs_to", "pathway")].items():
        for dst in dsts:
            pt_edges[0].append(src)
            pt_edges[1].append(dst)
    hetero_data["protein", "belongs_to", "pathway"].edge_index = torch.tensor(pt_edges, dtype=torch.long)
    rev_pt = [pt_edges[1][:], pt_edges[0][:]]
    hetero_data["pathway", "includes", "protein"].edge_index = torch.tensor(rev_pt, dtype=torch.long)

    # 疾病边加入异质图
    if n_diseases > 0:
        pd_edges = [[], []]
        for src, dsts in hetero_adj[("protein", "associated_with", "disease")].items():
            for dst in dsts:
                pd_edges[0].append(src)
                pd_edges[1].append(dst)
        hetero_data["protein", "associated_with", "disease"].edge_index = torch.tensor(pd_edges, dtype=torch.long)
        rev_pd = [pd_edges[1][:], pd_edges[0][:]]
        hetero_data["disease", "involves", "protein"].edge_index = torch.tensor(rev_pd, dtype=torch.long)
        hetero_data["disease"].x = torch.zeros(n_diseases, 1, dtype=torch.float32)
        logger.info(f"disease edges = {len(pd_edges[0])}")

    logger.info(f"异质图: compound({n_compounds}) protein({n_proteins}) pathway({n_pathways}) disease({n_diseases}) | "
                f"CPI={len(cpi_edges[0])} PPI={len(ppi_edges[0])} Pathway={len(pt_edges[0])} "
                f"CompSim={len(comp_sim_edges_tensor[0])}")

    # Opt1: 预计算全图同质边索引，验证/预测直接复用（速度提升 10x+）
    homo_edge_list = []
    for node in range(n_compounds + n_proteins):
        for nbr in homo_adj.get(node, []):
            homo_edge_list.append([node, nbr])
    if homo_edge_list:
        homo_edge_index = torch.tensor(homo_edge_list, dtype=torch.long).t().contiguous()
    else:
        homo_edge_index = torch.zeros((2, 0), dtype=torch.long)
    logger.info(f"预计算全图边索引: {homo_edge_index.shape[1]} 条边")

    # 预计算基于PPI拓扑的负样本邻居（可选，默认关闭以避免训练启动开销）
    prot_to_topo_medium_neighbors: dict[int, set] | None = None
    prot_to_topo_hard_neighbors: dict[int, set] | None = None
    if use_topology_neg:
        active_genes = {str(g).strip().upper() for g in cpi_df["gene"].dropna().unique()}
        logger.info(
            f"预计算PPI拓扑负样本 (active_genes={len(active_genes)}, top_k={topo_neighbors_top_k}) ..."
        )
        try:
            from .topology_negative_sampling import (
                TopologyNegativeSampler,
                build_topology_hard_neighbors,
                build_topology_medium_neighbors,
            )
            sampler = TopologyNegativeSampler(ppi_df)
            prot_to_topo_medium_neighbors = build_topology_medium_neighbors(
                ppi_df,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                active_genes=active_genes,
                top_k=topo_neighbors_top_k,
                sampler=sampler,
            )
            prot_to_topo_hard_neighbors = build_topology_hard_neighbors(
                ppi_df,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                active_genes=active_genes,
                top_k=topo_neighbors_top_k,
                sampler=sampler,
            )
        except ImportError as e:
            logger.warning("初始化 TopologyNegativeSampler 失败，回退到无拓扑负样本: %s", e)
        except Exception:
            logger.exception("初始化 TopologyNegativeSampler 失败，回退到无拓扑负样本")

    # v41: 预计算基于 ESM-2 余弦相似度的难负样本邻居（可选，默认关闭）
    prot_to_esm_hard_neighbors: dict[int, set] | None = None
    if use_esm_similarity_neg:
        # 使用 prot_feat 中的 ESM-2 嵌入（已由 load_protein_features 加载）
        active_genes = {str(g).strip().upper() for g in cpi_df["gene"].dropna().unique()}
        logger.info(
            f"v41: 预计算ESM-2余弦相似度难负样本 "
            f"(active_genes={len(active_genes)}, top_k={esm_similarity_top_k}) ..."
        )
        try:
            from .esm_similarity_negative_sampling import (
                build_esm_similarity_hard_neighbors,
            )
            prot_to_esm_hard_neighbors = build_esm_similarity_hard_neighbors(
                esm2_embeddings=prot_feat,
                gene_to_idx=gene_to_idx,
                n_compounds=n_compounds,
                ppi_df=ppi_df,
                active_genes=active_genes,
                top_k=esm_similarity_top_k,
            )
        except ImportError as e:
            logger.warning(
                "v41: 初始化 ESM-2 相似度负样本失败，回退到无此负样本: %s", e
            )
        except Exception:
            logger.exception(
                "v41: 初始化 ESM-2 相似度负样本失败，回退到无此负样本"
            )

    return {
        "x": x,
        "feat_dim": feat_dim,
        "prot_feat_dim": prot_feat_dim,
        "prot_esm_dim": prot_esm_dim,
        "n_compounds": n_compounds,
        "n_proteins": n_proteins,
        "smi_to_idx": smi_to_idx,
        "gene_to_idx": gene_to_idx,
        "homo_adj": homo_adj,
        "homo_edge_index": homo_edge_index,
        "hetero_adj": hetero_adj,
        "hetero_data": hetero_data,
        "n_pathways": n_pathways,
        "n_diseases": n_diseases,
        "disease_to_idx": disease_to_idx,
        "prot_to_path_neighbors": prot_to_path_neighbors,
        "prot_to_topo_medium_neighbors": prot_to_topo_medium_neighbors,
        "prot_to_topo_hard_neighbors": prot_to_topo_hard_neighbors,
        "prot_to_esm_hard_neighbors": prot_to_esm_hard_neighbors,  # v41: ESM-2余弦相似度难负样本
    }