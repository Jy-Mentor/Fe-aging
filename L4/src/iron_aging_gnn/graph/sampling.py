"""子图采样：DropEdge 正则化 + 同质图/异质图邻居采样"""

from __future__ import annotations

import logging
import random

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def drop_edge(edge_index: torch.Tensor, p: float = 0.15) -> torch.Tensor:
    """DropEdge 正则化：随机丢弃 p 比例的边，缓解过拟合与过平滑

    参考: Rong et al. (2020) "DropEdge: Towards Deep Graph Neural Networks", ICLR
    """
    if p <= 0 or edge_index.shape[1] <= 1:
        return edge_index
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask]


def sample_homo_subgraph(
    seed_compounds: list[int],
    homo_adj: dict[int, list[int]],
    num_neighbors: list[int] | None = None,
    seed: int | None = None,
) -> tuple[list[int], dict[int, int], torch.Tensor]:
    """GraphSAGE 风格邻居采样：固定每层邻居数，避免邻居爆炸"""
    if num_neighbors is None:
        num_neighbors = [32, 16]
    if seed is not None:
        random.seed(seed)
    nodes = set(seed_compounds)
    frontier = set(seed_compounds)

    for hop_neighbors in num_neighbors:
        next_frontier = set()
        for node in frontier:
            nbrs = homo_adj.get(node, [])
            if len(nbrs) > hop_neighbors:
                nbrs = random.sample(nbrs, hop_neighbors)
            next_frontier.update(nbrs)
        nodes.update(next_frontier)
        frontier = next_frontier

    node_list = sorted(nodes)
    node_to_local = {n: i for i, n in enumerate(node_list)}

    # 向量化边构建，替换双重 Python 循环
    # 仅保留两端点都在 node_list 中的边
    src_list, dst_list = [], []
    for node in node_list:
        nbrs = homo_adj.get(node, [])
        if nbrs:
            for nbr in nbrs:
                if nbr in node_to_local:
                    src_list.append(node)
                    dst_list.append(nbr)

    if src_list:
        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)
        src_local = torch.tensor([node_to_local[int(s)] for s in src_t], dtype=torch.long)
        dst_local = torch.tensor([node_to_local[int(d)] for d in dst_t], dtype=torch.long)
        edge_index = torch.stack([src_local, dst_local])
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return node_list, node_to_local, edge_index


def sample_hetero_subgraph(
    seed_compounds: list[int],
    hetero_adj: dict,
    num_neighbors: list[int] | None = None,
    seed: int | None = None,
    seed_proteins: list[int] | None = None,
    add_seed_cpi_edges: bool = False,
) -> tuple[HeteroData, list[int], list[int], list[int], list[int], dict[int, int], dict[int, int], dict[int, int]]:
    """异质图手动邻居采样。

    seed_proteins 参数允许将指定蛋白（如验证蛋白）作为孤立节点纳入子图，
    用于 OOM 降级 mini-batch 验证。

    add_seed_cpi_edges: 冷启动验证时，seed_compounds 在 hetero_adj 中无 CPI 出边，
        导致 HGT 中化合物节点孤立。若开启，则在子图中临时添加 seed_compounds -> seed_proteins
        的 CPI 边，仅用于消息传递，不用于标签构造。
    """
    if num_neighbors is None:
        num_neighbors = [32, 16]
    if seed is not None:
        random.seed(seed)
    compounds = set(seed_compounds)
    proteins = set(seed_proteins) if seed_proteins else set()
    pathways = set()
    diseases = set()

    # 1-hop: 化合物 → 蛋白
    cpi_adj = hetero_adj[("compound", "interacts", "protein")]
    for c in seed_compounds:
        if c in cpi_adj:
            nbrs = cpi_adj[c]
            if len(nbrs) > num_neighbors[0]:
                nbrs = random.sample(nbrs, num_neighbors[0])
            proteins.update(nbrs)

    # 2-hop: 蛋白 → 蛋白 + 蛋白 → 通路 + 蛋白 → 疾病
    ppi_adj = hetero_adj[("protein", "ppi", "protein")]
    pt_adj = hetero_adj[("protein", "belongs_to", "pathway")]
    pd_adj = hetero_adj.get(("protein", "associated_with", "disease"), {})
    for p in list(proteins):
        if p in ppi_adj:
            nbrs = ppi_adj[p]
            if len(nbrs) > num_neighbors[1]:
                nbrs = random.sample(nbrs, num_neighbors[1])
            proteins.update(nbrs)
        if p in pt_adj:
            nbrs = pt_adj[p]
            if len(nbrs) > num_neighbors[1]:
                nbrs = random.sample(nbrs, num_neighbors[1])
            pathways.update(nbrs)
        if p in pd_adj:
            nbrs = pd_adj[p]
            if len(nbrs) > num_neighbors[1]:
                nbrs = random.sample(nbrs, num_neighbors[1])
            diseases.update(nbrs)

    comp_sorted = sorted(compounds)
    prot_sorted = sorted(proteins)
    path_sorted = sorted(pathways)
    disease_sorted = sorted(diseases)
    comp_map = {c: i for i, c in enumerate(comp_sorted)}
    prot_map = {p: i for i, p in enumerate(prot_sorted)}
    path_map = {p: i for i, p in enumerate(path_sorted)}
    disease_map = {d: i for i, d in enumerate(disease_sorted)}
    path_global = list(path_sorted)
    disease_global = list(disease_sorted)

    sg = HeteroData()
    sg._comp_sorted = comp_sorted
    sg._prot_map = prot_map
    sg._path_global = path_global
    sg._disease_global = disease_global

    def _build_edges(et, src_map, dst_map):
        sl, dl = [], []
        for s, ds in hetero_adj.get(et, {}).items():
            if s in src_map:
                for d in ds:
                    if d in dst_map:
                        sl.append(src_map[s])
                        dl.append(dst_map[d])
        if sl:
            return torch.tensor([sl, dl], dtype=torch.long)
        return torch.zeros((2, 0), dtype=torch.long)

    sg["compound", "interacts", "protein"].edge_index = _build_edges(
        ("compound", "interacts", "protein"), comp_map, prot_map)

    # v40-fix: 冷启动化合物在 hetero_adj 中无 CPI 出边，验证时临时添加 seed -> candidate 蛋白边，
    # 仅用于 HGT 消息传递，不用于标签构造。
    if add_seed_cpi_edges and seed_proteins:
        extra_sl, extra_dl = [], []
        for c in seed_compounds:
            c_local = comp_map.get(c)
            if c_local is None:
                continue
            for p in seed_proteins:
                p_local = prot_map.get(p)
                if p_local is not None:
                    extra_sl.append(c_local)
                    extra_dl.append(p_local)
        if extra_sl:
            existing = sg["compound", "interacts", "protein"].edge_index
            extra = torch.tensor([extra_sl, extra_dl], dtype=torch.long)
            sg["compound", "interacts", "protein"].edge_index = torch.cat([existing, extra], dim=1)

    # v40-fix: HGTConv 需要双向消息传递，手动构建 CPI 反向边
    cpi_sl, cpi_dl = sg["compound", "interacts", "protein"].edge_index.tolist()
    if cpi_sl:
        sg["protein", "rev_interacts", "compound"].edge_index = torch.tensor(
            [cpi_dl, cpi_sl], dtype=torch.long)
    else:
        sg["protein", "rev_interacts", "compound"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    sg["protein", "ppi", "protein"].edge_index = _build_edges(
        ("protein", "ppi", "protein"), prot_map, prot_map)
    # v40-fix: 无向 PPI 需要双向边
    ppi_sl, ppi_dl = sg["protein", "ppi", "protein"].edge_index.tolist()
    if ppi_sl:
        sg["protein", "rev_ppi", "protein"].edge_index = torch.tensor(
            [ppi_dl, ppi_sl], dtype=torch.long)
    else:
        sg["protein", "rev_ppi", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    sg["protein", "belongs_to", "pathway"].edge_index = _build_edges(
        ("protein", "belongs_to", "pathway"), prot_map, path_map)
    sg["protein", "associated_with", "disease"].edge_index = _build_edges(
        ("protein", "associated_with", "disease"), prot_map, disease_map)

    # 手动构建反向边（通路→蛋白），_build_edges 无法处理反向映射
    sl_rev, dl_rev = [], []
    for p_global, pathway_ids in hetero_adj.get(("protein", "belongs_to", "pathway"), {}).items():
        if p_global in prot_map:
            for pid in pathway_ids:
                if pid in path_map:
                    sl_rev.append(path_map[pid])   # 通路为源
                    dl_rev.append(prot_map[p_global])  # 蛋白为目标
    if sl_rev:
        sg["pathway", "includes", "protein"].edge_index = torch.tensor(
            [sl_rev, dl_rev], dtype=torch.long)
    else:
        sg["pathway", "includes", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # v24: 手动构建疾病→蛋白反向边
    sl_rev_d, dl_rev_d = [], []
    for p_global, disease_ids in hetero_adj.get(("protein", "associated_with", "disease"), {}).items():
        if p_global in prot_map:
            for did in disease_ids:
                if did in disease_map:
                    sl_rev_d.append(disease_map[did])
                    dl_rev_d.append(prot_map[p_global])
    if sl_rev_d:
        sg["disease", "involves", "protein"].edge_index = torch.tensor(
            [sl_rev_d, dl_rev_d], dtype=torch.long)
    else:
        sg["disease", "involves", "protein"].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # v24: disease节点特征（无原始特征，用零向量，模型内disease_embed生成嵌入）
    sg["disease"].x = torch.zeros(len(disease_sorted), 1, dtype=torch.float32)

    return sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map
