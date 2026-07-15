"""验证安全图构建：训练/验证图隔离，防止数据泄露"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


def build_val_safe_homo_edge_index(
    homo_edge_index: torch.Tensor,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set | None = None,
) -> torch.Tensor:
    """v18: 构建严格验证安全的同质图边索引

    为真实评估蛋白冷启动泛化能力，必须让验证蛋白在图中完全孤立：
      - 移除所有一端是验证集化合物的边
      - 移除所有一端是验证集蛋白的边（包括 PPI、CPI）

    Args:
        homo_edge_index: (2, E) 全图同质边索引
        n_compounds: 化合物节点数
        val_comp_set: 验证集化合物全局索引集合
        val_prot_set: 验证集蛋白局部索引集合（0-based，相对于 n_compounds）

    Returns:
        (2, E') 过滤后的边索引
    """
    if not val_comp_set and not val_prot_set:
        return homo_edge_index
    src = homo_edge_index[0]
    dst = homo_edge_index[1]

    # 全局索引集合转张量，用于向量化 isin
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=homo_edge_index.device)

    # 移除所有涉及验证集化合物的边
    src_in_val_comp = torch.isin(src, val_comp_tensor)
    dst_in_val_comp = torch.isin(dst, val_comp_tensor)
    remove = src_in_val_comp | dst_in_val_comp

    # 移除所有涉及验证集蛋白的边（局部索引 -> 全局索引）
    if val_prot_set:
        val_prot_global = torch.tensor(
            sorted(p + n_compounds for p in val_prot_set),
            dtype=torch.long, device=homo_edge_index.device)
        src_in_val_prot = torch.isin(src, val_prot_global)
        dst_in_val_prot = torch.isin(dst, val_prot_global)
        remove = remove | src_in_val_prot | dst_in_val_prot

    mask = ~remove
    n_removed = (~mask).sum().item()
    logger.info(f"  严格验证安全同质图: 移除 {n_removed} 条边 (val_comp + val_prot 全部边), "
                f"保留 {mask.sum().item()} 条边")
    return homo_edge_index[:, mask]


def build_val_safe_hetero_data(
    hetero_data: HeteroData,
    val_comp_set: set,
    val_prot_set: set | None = None,
) -> HeteroData:
    """v18: 构建严格验证安全的异质图

    移除所有涉及验证集化合物或验证集蛋白的边，确保验证蛋白在异质图中完全孤立。

    Args:
        hetero_data: 全图异质图数据
        val_comp_set: 验证集化合物全局索引集合
        val_prot_set: 验证集蛋白局部索引集合（0-based，相对于化合物数）

    Returns:
        过滤后的异质图数据
    """
    hetero_data_val = HeteroData()
    for node_type in hetero_data.node_types:
        hetero_data_val[node_type].x = hetero_data[node_type].x.clone()
        if node_type == "pathway" and hasattr(hetero_data["pathway"], "n_pathways"):
            hetero_data_val["pathway"].n_pathways = hetero_data["pathway"].n_pathways

    # v25-fix: 预计算过滤张量，避免每个 edge_type 重复 sorted+torch.tensor
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long) if val_comp_set else None
    val_prot_tensor = torch.tensor(sorted(val_prot_set), dtype=torch.long) if val_prot_set else None

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        src_type, rel, dst_type = edge_type
        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)

        # 1) 涉及验证集化合物的 CPI 边：compound -> protein
        if edge_type == ("compound", "interacts", "protein") and val_comp_tensor is not None:
            val_comp_dev = val_comp_tensor.to(edge_index.device)
            keep_mask = keep_mask & (~torch.isin(edge_index[0], val_comp_dev))

        # 2) 涉及验证集蛋白的边（PPI / protein-pathway / pathway-protein / protein-disease / disease-protein）
        if val_prot_tensor is not None:
            val_prot_dev = val_prot_tensor.to(edge_index.device)
            if src_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[0], val_prot_dev))
            if dst_type == "protein":
                keep_mask = keep_mask & (~torch.isin(edge_index[1], val_prot_dev))

        n_removed = (~keep_mask).sum().item()
        if n_removed > 0:
            logger.info(f"  严格验证安全异质图: 移除 {edge_type} {n_removed} 条边, "
                        f"保留 {keep_mask.sum().item()} 条边")
        hetero_data_val[edge_type].edge_index = edge_index[:, keep_mask]

    return hetero_data_val


def build_val_comp_cold_homo_edge_index(
    homo_edge_index: torch.Tensor,
    val_comp_set: set,
) -> torch.Tensor:
    """v18: 构建化合物冷启动验证同质图

    化合物冷启动评估中，验证化合物未在训练中出现，因此移除其所有 CPI 边。
    但蛋白（包括验证蛋白）之间的 PPI 边和通路边保留，因为蛋白侧不是冷启动对象。
    """
    if not val_comp_set:
        return homo_edge_index
    src = homo_edge_index[0]
    dst = homo_edge_index[1]
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long, device=homo_edge_index.device)
    remove = torch.isin(src, val_comp_tensor) | torch.isin(dst, val_comp_tensor)
    mask = ~remove
    n_removed = (~mask).sum().item()
    logger.info(f"  化合物冷启动验证同质图: 移除 {n_removed} 条边 (仅 val_comp), 保留 {mask.sum().item()} 条边")
    return homo_edge_index[:, mask]


def build_val_comp_cold_hetero_data(
    hetero_data,
    val_comp_set: set,
) -> HeteroData:
    """v18: 构建化合物冷启动验证异质图

    仅移除验证集化合物相关的 CPI 边，保留所有蛋白-蛋白/蛋白-通路/通路-蛋白边。
    """
    hetero_data_val = HeteroData()
    for node_type in hetero_data.node_types:
        hetero_data_val[node_type].x = hetero_data[node_type].x.clone()
        if node_type == "pathway" and hasattr(hetero_data["pathway"], "n_pathways"):
            hetero_data_val["pathway"].n_pathways = hetero_data["pathway"].n_pathways

    # v25-fix: 预计算 val_comp_tensor，避免每个 edge_type 重复 sorted+torch.tensor
    val_comp_tensor = torch.tensor(sorted(val_comp_set), dtype=torch.long) if val_comp_set else None

    for edge_type in hetero_data.edge_types:
        edge_index = hetero_data[edge_type].edge_index
        if edge_type == ("compound", "interacts", "protein") and val_comp_tensor is not None:
            val_comp_dev = val_comp_tensor.to(edge_index.device)
            keep_mask = ~torch.isin(edge_index[0], val_comp_dev)
        else:
            keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool, device=edge_index.device)
        n_removed = (~keep_mask).sum().item()
        if n_removed > 0:
            n_kept = keep_mask.sum().item()
            logger.info(f"  化合物冷启动验证异质图: 移除 {edge_type} {n_removed} 条边, 保留 {n_kept} 条边")
        hetero_data_val[edge_type].edge_index = edge_index[:, keep_mask]
    return hetero_data_val


def build_train_safe_homo_adj(
    homo_adj: dict,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set | None = None,
) -> dict:
    """v18: 构建训练安全同质邻接表

    为真实蛋白冷启动评估，训练图必须对验证蛋白完全不可见：
      - 移除所有涉及验证集化合物的边
      - 移除所有涉及验证集蛋白的边（包括 CPI、PPI）

    这样验证蛋白在训练阶段从未参与任何消息传递，确保冷启动指标反映
    模型对全新蛋白的泛化能力。
    """
    train_adj = defaultdict(list)
    val_prot_global = {p + n_compounds for p in val_prot_set} if val_prot_set else set()
    val_nodes = set(val_comp_set) | val_prot_global
    for src, dsts in homo_adj.items():
        if src in val_nodes:
            continue
        for dst in dsts:
            if dst not in val_nodes:
                train_adj[src].append(dst)
    return train_adj


def build_train_safe_hetero_adj(
    hetero_adj: dict,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set | None = None,
) -> dict:
    """v70-fix: 构建训练安全异质邻接表（通用化过滤）

    从训练图中彻底移除验证集化合物/蛋白相关的所有异质边，
    确保冷启动验证的严格性（inductive setting）。

    通用过滤规则（基于边类型的源/目标节点类型）：
      - 源类型为 compound 且 src ∈ val_comp_set → 移除整条源记录
      - 源类型为 protein  且 src ∈ val_prot_set → 移除整条源记录
      - 目标类型为 compound 且 dst ∈ val_comp_set → 移除该目标
      - 目标类型为 protein  且 dst ∈ val_prot_set → 移除该目标
      - pathway / disease 节点不属于验证集，不过滤

    参考:
      - PyG RandomLinkSplit transform (inductive setting)
      - GBAN-DA (Lv et al. 2026, PMID:41687302) "rigorous cold pair splits"
      - CFM-DTI (Li et al. 2026, PMID:42341701) ESM2 + GNN 冷启动评估
    """
    val_prot_set = val_prot_set or set()
    train_adj = {}
    for et, adj in hetero_adj.items():
        src_type, _rel, dst_type = et
        new_adj = defaultdict(list)
        for src, dsts in adj.items():
            # 源端过滤
            if src_type == "compound" and src in val_comp_set:
                continue
            if src_type == "protein" and src in val_prot_set:
                continue
            # 目标端过滤
            for dst in dsts:
                if dst_type == "compound" and dst in val_comp_set:
                    continue
                if dst_type == "protein" and dst in val_prot_set:
                    continue
                new_adj[src].append(dst)
        train_adj[et] = new_adj
    return train_adj


def build_val_safe_hetero_adj(
    hetero_adj: dict,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set | None = None,
) -> dict:
    """v18: 构建验证安全异质邻接表

    用于 HGT OOM 降级时的 mini-batch 验证，确保子图采样不引入验证蛋白边。
    与训练安全邻接表语义相同（均移除验证集化合物/蛋白相关边），但单独命名
    便于后续区分验证/训练采样策略。
    """
    # 验证安全邻接表与训练安全邻接表在当前设定下等价
    return build_train_safe_hetero_adj(hetero_adj, n_compounds, val_comp_set, val_prot_set)


def build_val_comp_cold_hetero_adj(
    hetero_adj,
    n_compounds: int,
    val_comp_set: set,
    val_prot_set: set = None,
):
    """v59: 构建严格化合物冷启动验证异质邻接表

    彻底移除验证集化合物的所有 CPI 边（无论目标蛋白属于训练集还是验证集），
    确保验证化合物仅依靠自身特征（encode_compound）和全局蛋白拓扑（PPI/通路/疾病）
    进行预测，符合真实冷启动场景。

    蛋白侧所有拓扑（PPI / 通路 / 疾病）保持完整。
    """
    val_adj = {}
    val_prot_set = set() if val_prot_set is None else val_prot_set
    for et, adj in hetero_adj.items():
        new_adj = defaultdict(list)
        for src, dsts in adj.items():
            if et == ("compound", "interacts", "protein") and src in val_comp_set:
                # v59: 严格冷启动 — 完全移除验证化合物的所有 CPI 边
                continue
            else:
                new_adj[src].extend(dsts)
        val_adj[et] = new_adj
    return val_adj
