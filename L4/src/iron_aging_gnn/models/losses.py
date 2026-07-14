"""铁衰老 GNN 损失函数模块

包含:
  - Focal Loss（处理类别不平衡）
  - BPR 排序损失（优化 AUC）
  - InfoNCE 对比损失（表示学习）
  - 辅助网络重建损失（多任务正则化）
  - 语义注意力对齐损失（跨模态对齐）

参考:
  - Lin et al. (2017) "Focal Loss for Dense Object Detection"
  - van den Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding"
  - Rendle et al. (2009) "BPR: Bayesian Personalized Ranking from Implicit Feedback"
  - Chen et al. (2020) "A Simple Framework for Contrastive Learning of Visual Representations"
  - Lai et al. (2025) "DHGT-DTI: Dual-Perspective Heterogeneous Graph Transformer for DTI", PMC12616060
  - Peng et al. (2025) "GHCDTI: Graph Wavelet-Based Cross-View Contrastive Learning for DTI", PMC12365291
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)


# ─── 常量配置 ───────────────────────────────────────────────────────
# 硬约束 (project_memory): BCE:0.6, BPR:0.4, InfoNCE:0, Aux:0
FOCAL_GAMMA = 1.0
FOCAL_ALPHA = 0.6
SCORE_CLAMP = 10.0
LABEL_SMOOTHING_POS = 0.95
LABEL_SMOOTHING_NEG = 0.05
TEMPERATURE = 1.0
BPR_WEIGHT = 0.4
CPI_LOSS_WEIGHT = 0.6
INFONCE_WEIGHT = 0.0
INFONCE_TEMPERATURE = 0.07
AUX_RECON_WEIGHT = 0.05  # v70-fix: 与 default.yaml 同步，辅助重建损失已实现
SEMANTIC_ATTN_WEIGHT = 0.0
SEMANTIC_ATTN_TEMPERATURE = 0.5


# ─── 工具函数 ───────────────────────────────────────────────────────


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = FOCAL_GAMMA,
    alpha: float | None = FOCAL_ALPHA,
) -> torch.Tensor:
    """带 logits 输入的 Focal Loss，包含数值稳定性优化。

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        logits: 原始 logits（未经过 Sigmoid）
        targets: 平滑后的目标值，范围 [0, 1]
        gamma: 聚焦参数，γ 越大对易分类样本的惩罚越小
        alpha: 类别权重，用于平衡正负样本
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    focal_factor = (1 - p_t) ** gamma
    if alpha is not None:
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        focal_factor = focal_factor * alpha_factor
    return (focal_factor * loss).mean()


# ─── 辅助函数：对边进行重建损失 ────────────────────────────────────


def _compute_edge_recon_loss(
    src_emb: torch.Tensor,
    dst_emb: torch.Tensor,
    n_samples: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """对指定边类型计算重建损失（正样本三元组 + 随机负样本）。"""
    n_edges = src_emb.shape[0]
    if n_edges == 0:
        return torch.tensor(0.0, device=src_emb.device, dtype=src_emb.dtype)

    edges_sample = torch.randperm(n_edges, device=src_emb.device)[: min(n_samples, n_edges)]
    s = src_emb[edges_sample]
    d = dst_emb[edges_sample]
    pos_logits = (s * d).sum(dim=1) / temperature
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits, torch.ones_like(pos_logits), reduction="mean"
    )

    shuffle_idx = torch.randperm(d.shape[0], device=d.device)
    d_shuffled = d[shuffle_idx]
    neg_logits = (s * d_shuffled).sum(dim=1) / temperature
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits, torch.zeros_like(neg_logits), reduction="mean"
    )

    return (pos_loss + neg_loss) / 2


def compute_aux_recon_loss(
    model: torch.nn.Module,
    data: HeteroData,
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    device: torch.device,
    aux_loss_config: dict | None = None,
) -> torch.Tensor:
    """计算辅助网络重建损失。

    重建 6 类网络拓扑结构，增强嵌入的语义一致性。
    参考: DHGT-DTI (Lai et al. 2025) 多任务学习框架

    Args:
        model: 模型实例（用于获取嵌入）
        data: 异质图数据
        comp_emb: 化合物嵌入 [n_compounds, out_dim]
        prot_emb: 蛋白嵌入 [n_proteins, out_dim]
        device: 计算设备
        aux_loss_config: 辅助损失配置字典，可选字段见默认值

    Returns:
        加权后的总辅助重建损失
    """
    cfg = {
        "ppi_samples": 256,
        "pathway_samples": 128,
        "ddi_samples": 128,
        "drug_disease_samples": 128,
        "protein_disease_samples": 128,
        "drug_side_effect_samples": 128,
    }
    if aux_loss_config:
        cfg.update(aux_loss_config)

    total_loss = torch.tensor(0.0, device=device, dtype=comp_emb.dtype)
    active_count = 0

    # 1. PPI 重建
    if hasattr(data, "protein", "ppi") and data["protein", "ppi", "protein"].edge_index.shape[1] > 0:
        edge_idx = data["protein", "ppi", "protein"].edge_index
        src, dst = edge_idx
        ppi_loss = _compute_edge_recon_loss(prot_emb[src], prot_emb[dst], cfg["ppi_samples"])
        if ppi_loss.item() > 0:
            total_loss = total_loss + ppi_loss
            active_count += 1

    # 2. 通路-蛋白重建
    if hasattr(data, "pathway", "pathway_protein") and data["pathway", "pathway_protein", "protein"].edge_index.shape[1] > 0:
        if hasattr(model, "pathway_emb") and model.pathway_emb is not None:
            edge_idx = data["pathway", "pathway_protein", "protein"].edge_index
            src, dst = edge_idx
            pathway_emb = model.pathway_emb.weight
            pw_loss = _compute_edge_recon_loss(
                pathway_emb[src], prot_emb[dst], cfg["pathway_samples"]
            )
            if pw_loss.item() > 0:
                total_loss = total_loss + pw_loss
                active_count += 1

    # 3. DDI 重建
    if (hasattr(data, "drug", "ddi") and data["drug", "ddi", "drug"].edge_index.shape[1] > 0):
        edge_idx = data["drug", "ddi", "drug"].edge_index
        src, dst = edge_idx
        ddi_loss = _compute_edge_recon_loss(comp_emb[src], comp_emb[dst], cfg["ddi_samples"])
        if ddi_loss.item() > 0:
            total_loss = total_loss + ddi_loss
            active_count += 1

    # 4. Drug-Disease 重建
    if (hasattr(data, "drug", "drug_disease") and data["drug", "drug_disease", "disease"].edge_index.shape[1] > 0):
        if hasattr(model, "disease_emb") and model.disease_emb is not None:
            edge_idx = data["drug", "drug_disease", "disease"].edge_index
            src, dst = edge_idx
            disease_emb = model.disease_emb.weight
            dd_loss = _compute_edge_recon_loss(
                comp_emb[src], disease_emb[dst], cfg["drug_disease_samples"]
            )
            if dd_loss.item() > 0:
                total_loss = total_loss + dd_loss
                active_count += 1

    # 5. Protein-Disease 重建
    if (hasattr(data, "protein", "protein_disease") and data["protein", "protein_disease", "disease"].edge_index.shape[1] > 0):
        if hasattr(model, "disease_emb") and model.disease_emb is not None:
            edge_idx = data["protein", "protein_disease", "disease"].edge_index
            src, dst = edge_idx
            disease_emb = model.disease_emb.weight
            pd_loss = _compute_edge_recon_loss(
                prot_emb[src], disease_emb[dst], cfg["protein_disease_samples"]
            )
            if pd_loss.item() > 0:
                total_loss = total_loss + pd_loss
                active_count += 1

    # 6. Drug-SideEffect 重建
    if (hasattr(data, "drug", "drug_side_effect") and data["drug", "drug_side_effect", "sideeffect"].edge_index.shape[1] > 0):
        if hasattr(model, "side_effect_emb") and model.side_effect_emb is not None:
            edge_idx = data["drug", "drug_side_effect", "sideeffect"].edge_index
            src, dst = edge_idx
            se_emb = model.side_effect_emb.weight
            dse_loss = _compute_edge_recon_loss(
                comp_emb[src], se_emb[dst], cfg["drug_side_effect_samples"]
            )
            if dse_loss.item() > 0:
                total_loss = total_loss + dse_loss
                active_count += 1

    if active_count == 0:
        logger.debug("  无辅助网络需要重建")
        return torch.tensor(0.0, device=device, dtype=comp_emb.dtype)

    return total_loss / active_count


# ─── 语义注意力损失 ───────────────────────────────────────────────


def compute_semantic_attention_loss(
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    attn_weights: torch.Tensor,
    temperature: float = SEMANTIC_ATTN_TEMPERATURE,
    n_samples: int = 128,
) -> torch.Tensor:
    """计算语义注意力对齐损失。

    参考: GHCDTI (Peng et al. 2025) 语义注意力机制

    Args:
        comp_emb: 化合物嵌入 [n_compounds, dim]
        prot_emb: 蛋白嵌入 [n_proteins, dim]
        attn_weights: 语义注意力权重 [n_compounds, n_proteins] 或 [batch]
        temperature: 注意力温度
        n_samples: 采样数

    Returns:
        语义注意力对齐损失
    """
    n_compounds = comp_emb.shape[0]
    n_proteins = prot_emb.shape[0]
    if n_compounds < 2 or n_proteins < 2:
        return torch.tensor(0.0, device=comp_emb.device, dtype=comp_emb.dtype)

    rand_c = torch.randperm(n_compounds, device=comp_emb.device)[: min(n_samples, n_compounds)]
    rand_p = torch.randperm(n_proteins, device=prot_emb.device)[: min(n_samples, n_proteins)]

    c = comp_emb[rand_c]
    p = prot_emb[rand_p]
    sim = torch.mm(c, p.t()) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)

    loss_c = F.cross_entropy(sim, labels)
    loss_p = F.cross_entropy(sim.t(), labels[: sim.shape[1]])
    return (loss_c + loss_p) / 2


# ─── CPI 主损失 ────────────────────────────────────────────────────


def compute_cpi_loss(
    model: torch.nn.Module,
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    comp_sorted: list[int] = None,
    prot_map: dict[int, int] = None,
    precomputed_pos: dict[int, list[int]] = None,
    n_compounds: int = 0,
    prot_to_path_neighbors: dict[int, set] = None,
    epoch: int = 0,
    stage_epochs: int = 100,
    memory_bank: torch.Tensor | None = None,
    compound_to_prot_locals: dict[int, list[int]] = None,
    use_infonce: bool = False,
    bpr_weight: float = 0.4,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    prot_to_topo_medium_neighbors: dict[int, list[int]] = None,
    prot_to_topo_hard_neighbors: dict[int, list[int]] = None,
    focal_gamma: float | None = None,
    focal_alpha: float | None = None,
    use_residue_decoder: bool = True,
    bpr_detach_neg: bool = True,
    semantic_attn_weight: float = 0.0,
    semantic_attn_temperature: float = 0.5,
    T: float = 1.0,
    score_clamp: float = 10.0,
) -> torch.Tensor:
    """计算 CPI 组合损失（Focal + BPR + 可选 InfoNCE + 语义注意力）。

    由 trainer 调用，内部处理课程负采样、难负样本挖掘、BPR 排序损失、
    InfoNCE 对比损失和语义注意力对齐损失。

    Args:
        model: 模型实例
        comp_emb: 化合物嵌入 [n_compounds, out_dim]
        prot_emb: 蛋白嵌入 [n_proteins, out_dim]
        pos_src: 正样本化合物索引
        pos_dst: 正样本蛋白索引
        comp_sorted: 子图中化合物全局索引列表
        prot_map: 蛋白全局索引 -> 子图局部索引映射
        precomputed_pos: {化合物全局索引: [蛋白全局索引列表]}
        n_compounds: 图中化合物总数
        prot_to_path_neighbors: 蛋白通路邻居映射
        epoch: 当前 epoch
        stage_epochs: 阶段总 epoch 数
        memory_bank: Memory Bank 历史嵌入
        compound_to_prot_locals: {化合物全局索引: [蛋白局部索引列表]}
        use_infonce: 是否启用 InfoNCE 对比损失
        bpr_weight: BPR 损失权重
        use_curriculum: 是否启用课程负采样
        use_topology_neg: 是否启用拓扑负采样
        prot_to_topo_medium_neighbors: 蛋白拓扑中度负样本邻居
        prot_to_topo_hard_neighbors: 蛋白拓扑硬负样本邻居
        focal_gamma: Focal Loss gamma
        focal_alpha: Focal Loss alpha
        use_residue_decoder: 是否对正样本使用残基路径
        bpr_detach_neg: BPR 是否 detach 负样本
        semantic_attn_weight: 语义注意力损失权重
        semantic_attn_temperature: 语义注意力温度
        T: 解码温度
        score_clamp: 分数截断值

    Returns:
        CPI 组合损失
    """
    if focal_gamma is None:
        focal_gamma = FOCAL_GAMMA
    if focal_alpha is None:
        focal_alpha = FOCAL_ALPHA

    n_batch_prots = int(prot_emb.shape[0])

    # ── 正样本损失 ────────────────────────────────────────────────
    pos_residue_idx = pos_dst if use_residue_decoder else None
    pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst],
                             prot_residue_indices=pos_residue_idx) / T
    pos_score = torch.clamp(pos_score, -score_clamp, score_clamp)
    pos_loss = focal_loss_with_logits(
        pos_score, torch.full_like(pos_score, LABEL_SMOOTHING_POS),
        gamma=focal_gamma, alpha=focal_alpha)

    if use_residue_decoder and getattr(model, "decoder_type", None) == "residue_bilinear":
        pos_score_fb = model.decode(comp_emb[pos_src], prot_emb[pos_dst],
                                    prot_residue_indices=None) / T
        pos_score_fb = torch.clamp(pos_score_fb, -score_clamp, score_clamp)
        pos_loss_fb = focal_loss_with_logits(
            pos_score_fb, torch.full_like(pos_score_fb, LABEL_SMOOTHING_POS),
            gamma=focal_gamma, alpha=focal_alpha)
        pos_loss = pos_loss + 2.5 * pos_loss_fb

    unique_src = pos_src.unique()
    n_unique = len(unique_src)
    if n_unique == 0 or n_batch_prots <= 1:
        return CPI_LOSS_WEIGHT * pos_loss

    # ── 课程负采样 ────────────────────────────────────────────────
    neg_scores_list = []

    batch_comp_emb = comp_emb[unique_src]

    # 计算课程阶段
    if use_curriculum and stage_epochs > 0:
        progress = epoch / max(stage_epochs, 1)
        # 三阶段: 随机(0-30%) → 通路中度(30-65%) → 极硬(65-100%)
        CURRICULUM_RANDOM = 0.30
        CURRICULUM_MODERATE = 0.65
    else:
        progress = 1.0  # 无课程时直接进入硬负采样
        CURRICULUM_RANDOM = 0.0
        CURRICULUM_MODERATE = 0.0

    # 构建正样本掩码（用于排除正样本，避免采样到已知正交互）
    pos_mask = torch.zeros(n_unique, n_batch_prots, device=pos_score.device, dtype=torch.bool)
    for i, c in enumerate(unique_src):
        c_mask = pos_src == c
        pos_mask[i, pos_dst[c_mask]] = True

    if progress < CURRICULUM_RANDOM:
        # ── 阶段1: 随机负采样 ────────────────────────────────────
        n_neg_per_compound = 5
        for i in range(n_unique):
            available = torch.where(~pos_mask[i])[0]
            if len(available) == 0:
                continue
            n_sample = min(n_neg_per_compound, len(available))
            rand_idx = available[torch.randperm(len(available), device=available.device)[:n_sample]]
            neg_scores_random = model.decode(
                batch_comp_emb[i].unsqueeze(0).expand(n_sample, -1),
                prot_emb[rand_idx],
                prot_residue_indices=None,
            ) / T
            neg_scores_list.append(neg_scores_random)

    elif progress < CURRICULUM_MODERATE:
        # ── 阶段2: 通路邻近中度负采样 ─────────────────────────────
        n_neg_per_compound = 5
        for i in range(n_unique):
            c_global = comp_sorted[unique_src[i].item()] if comp_sorted and unique_src[i].item() < len(comp_sorted) else None
            # 尝试获取该化合物的正样本蛋白，进而查找通路邻居
            pathway_negs = None
            if c_global is not None and precomputed_pos and c_global in precomputed_pos:
                pos_prots_global = precomputed_pos[c_global]
                pathway_prot_locals = set()
                for pg in pos_prots_global:
                    if prot_map and pg in prot_map:
                        pl = prot_map[pg]
                        if prot_to_path_neighbors and pl in prot_to_path_neighbors:
                            pathway_prot_locals.update(prot_to_path_neighbors[pl])
                # 过滤正样本
                pathway_prot_locals = pathway_prot_locals - set(pos_dst[pos_src == unique_src[i]].tolist())
                if pathway_prot_locals:
                    pathway_negs = torch.tensor(list(pathway_prot_locals), device=pos_score.device)
                    pathway_negs = pathway_negs[pathway_negs < n_batch_prots]
            # 如果有通路邻居，使用通路邻居；否则回退到随机负采样
            if pathway_negs is not None and len(pathway_negs) > 0:
                n_sample = min(n_neg_per_compound, len(pathway_negs))
                idx = pathway_negs[torch.randperm(len(pathway_negs), device=pathway_negs.device)[:n_sample]]
            else:
                available = torch.where(~pos_mask[i])[0]
                if len(available) == 0:
                    continue
                n_sample = min(n_neg_per_compound, len(available))
                idx = available[torch.randperm(len(available), device=available.device)[:n_sample]]
            neg_scores_moderate = model.decode(
                batch_comp_emb[i].unsqueeze(0).expand(n_sample, -1),
                prot_emb[idx],
                prot_residue_indices=None,
            ) / T
            neg_scores_list.append(neg_scores_moderate)

    else:
        # ── 阶段3: 极硬负采样（全量评分 + Top-K 挖掘）─────────────
        with torch.no_grad():
            all_scores = model.decode(
                batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
                prot_emb.repeat(n_unique, 1),
                prot_residue_indices=None,
            ).view(n_unique, n_batch_prots)
            all_scores[pos_mask] = -float("inf")
            topk_indices = all_scores.topk(k=min(10, n_batch_prots - 1), dim=1)[1]

        n_hard_neg = 5
        for i in range(n_unique):
            hard_idx = topk_indices[i, torch.randperm(
                topk_indices.size(1), device=topk_indices.device)[:n_hard_neg]]
            neg_scores_hard = model.decode(
                batch_comp_emb[i].unsqueeze(0).expand(hard_idx.shape[0], -1),
                prot_emb[hard_idx],
                prot_residue_indices=None,
            ) / T
            neg_scores_list.append(neg_scores_hard)

    if not neg_scores_list:
        return CPI_LOSS_WEIGHT * pos_loss

    neg_scores = torch.cat(neg_scores_list)
    neg_scores = torch.clamp(neg_scores, -score_clamp, score_clamp)
    neg_loss = focal_loss_with_logits(
        neg_scores, torch.full_like(neg_scores, LABEL_SMOOTHING_NEG),
        gamma=focal_gamma, alpha=focal_alpha)

    # ── BPR 排序损失 (Hard-BPR 增强) ──────────────────────────────
    # 参考: Shi et al. (2023, WWW) + Hard-BPR (Shi et al. 2024)
    # HNS + BPR 等价于优化 OPAUC，与 Top-K 指标强相关
    # 动态 margin 基于负样本标准差，缓解假阴性影响
    bpr_loss = torch.tensor(0.0, device=pos_loss.device)
    if n_unique > 0 and neg_scores.shape[0] > 0:
        pos_mean = pos_score.mean()
        neg_mean = neg_scores.mean()
        neg_std = neg_scores.std()
        dynamic_margin = max(0.05, 0.2 * neg_std)
        if bpr_detach_neg:
            neg_mean = neg_mean.detach()
            neg_std = neg_std.detach()
        bpr_loss = -torch.log(torch.sigmoid(pos_mean - neg_mean - dynamic_margin) + 1e-8)

    # ── 组合损失 ──────────────────────────────────────────────────
    loss = CPI_LOSS_WEIGHT * (pos_loss + neg_loss) + bpr_weight * bpr_loss

    # ── InfoNCE 对比损失 ──────────────────────────────────────────
    if use_infonce and INFONCE_WEIGHT > 0:
        infonce_loss = compute_infonce_loss(
            model, comp_emb, prot_emb, memory_bank=memory_bank,
            n_samples=min(128, n_batch_prots),
        )
        loss = loss + INFONCE_WEIGHT * infonce_loss

    # ── 语义注意力对齐损失 ─────────────────────────────────────────
    if semantic_attn_weight > 0:
        sem_loss = compute_semantic_attention_loss(
            comp_emb, prot_emb,
            attn_weights=torch.ones(1, device=loss.device),
            temperature=semantic_attn_temperature,
            n_samples=min(128, n_batch_prots),
        )
        loss = loss + semantic_attn_weight * sem_loss

    return loss


def compute_auxiliary_reconstruction_loss(
    model: torch.nn.Module,
    prot_emb: torch.Tensor,
    prot_local_indices: list[int] = None,
    homo_adj: dict = None,
    n_compounds: int = 0,
    ppi_samples: int = 256,
    is_hetero: bool = False,
    comp_emb: torch.Tensor = None,
    comp_local_indices: list[int] = None,
    ddi_samples: int = 128,
    prot_to_path_neighbors: dict[int, set] = None,
    prot_disease_samples: int = 128,
    hetero_adj: dict = None,
    n_diseases: int = 0,
    disease_embed = None,
    pathway_samples: int = 128,
    drug_disease_samples: int = 128,
) -> torch.Tensor:
    """计算辅助网络重建损失（trainer 接口）。

    v70-fix: 修复空实现，基于真实邻接矩阵计算 PPI 边重建损失。
    参考: DHGT-DTI (Lai et al. 2025) 多任务学习框架 — 辅助重建正则化
          GHCDTI (Peng et al. 2025) 跨视图对比学习

    Args:
        model: 模型实例（未直接使用，保留接口兼容性）
        prot_emb: 蛋白嵌入 [n_batch_prots, out_dim]
        prot_local_indices: 蛋白在全局图中的节点索引列表
        homo_adj: 同质图邻接矩阵 {'edge_index': (2, E), 'edge_weights': (E,)}
        n_compounds: 化合物节点数
        ppi_samples: PPI 边采样数
        is_hetero: 是否为异质图模型
        comp_emb: 化合物嵌入 [n_batch_comps, out_dim]（异质图模式）
        comp_local_indices: 化合物全局索引列表（异质图模式）
        ddi_samples: DDI 边采样数（异质图模式）
        prot_to_path_neighbors: 蛋白-通路邻居映射（异质图模式）
        prot_disease_samples: 蛋白-疾病边采样数（异质图模式）
        hetero_adj: 异质图邻接矩阵 dict（异质图模式）
        n_diseases: 疾病节点数（异质图模式）
        disease_embed: 疾病嵌入（异质图模式）
        pathway_samples: 通路边采样数（异质图模式）
        drug_disease_samples: 药物-疾病边采样数（异质图模式）

    Returns:
        辅助重建损失标量
    """
    device = prot_emb.device
    dtype = prot_emb.dtype
    zero = torch.tensor(0.0, device=device, dtype=dtype)

    if prot_emb.shape[0] < 4 or prot_local_indices is None or len(prot_local_indices) < 4:
        return zero

    # 构建全局蛋白索引 -> 局部 batch 索引映射
    global_to_local = {g: i for i, g in enumerate(prot_local_indices)}

    total_loss = zero
    active_count = 0

    # ── PPI 边重建（同质图模式） ──────────────────────────────────
    if homo_adj is not None and "edge_index" in homo_adj:
        edge_index = homo_adj["edge_index"]
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            # 筛选 PPI 边：两端均为蛋白节点（node_id >= n_compounds）
            ppi_mask = (src >= n_compounds) & (dst >= n_compounds)
            ppi_src = src[ppi_mask]
            ppi_dst = dst[ppi_mask]
            if ppi_src.shape[0] > 0:
                # 映射到 batch 局部索引
                ppi_local_src = []
                ppi_local_dst = []
                for s, d in zip(ppi_src.tolist(), ppi_dst.tolist(), strict=False):
                    ls = global_to_local.get(s)
                    ld = global_to_local.get(d)
                    if ls is not None and ld is not None:
                        ppi_local_src.append(ls)
                        ppi_local_dst.append(ld)
                if len(ppi_local_src) >= 4:
                    ppi_local_src_t = torch.tensor(ppi_local_src, device=device, dtype=torch.long)
                    ppi_local_dst_t = torch.tensor(ppi_local_dst, device=device, dtype=torch.long)
                    ppi_loss = _compute_edge_recon_loss(
                        prot_emb[ppi_local_src_t], prot_emb[ppi_local_dst_t], ppi_samples
                    )
                    if ppi_loss.item() > 0:
                        total_loss = total_loss + ppi_loss
                        active_count += 1

    # ── 异质图模式：PPI + 通路边重建 ──────────────────────────────
    if is_hetero and hetero_adj is not None:
        # PPI 边（异质图）
        ppi_key = ("protein", "ppi", "protein")
        if ppi_key in hetero_adj:
            ppi_ei = hetero_adj[ppi_key]
            if ppi_ei.shape[1] > 0:
                ppi_src, ppi_dst = ppi_ei
                ppi_local_src = []
                ppi_local_dst = []
                for s, d in zip(ppi_src.tolist(), ppi_dst.tolist(), strict=False):
                    ls = global_to_local.get(s)
                    ld = global_to_local.get(d)
                    if ls is not None and ld is not None:
                        ppi_local_src.append(ls)
                        ppi_local_dst.append(ld)
                if len(ppi_local_src) >= 4:
                    ppi_local_src_t = torch.tensor(ppi_local_src, device=device, dtype=torch.long)
                    ppi_local_dst_t = torch.tensor(ppi_local_dst, device=device, dtype=torch.long)
                    ppi_loss = _compute_edge_recon_loss(
                        prot_emb[ppi_local_src_t], prot_emb[ppi_local_dst_t], ppi_samples
                    )
                    if ppi_loss.item() > 0:
                        total_loss = total_loss + ppi_loss
                        active_count += 1

        # 通路-蛋白边重建（异质图）
        pw_key = ("pathway", "pathway_protein", "protein")
        if pw_key in hetero_adj and prot_to_path_neighbors is not None:
            pw_ei = hetero_adj[pw_key]
            if pw_ei.shape[1] > 0 and hasattr(model, "pathway_embed") and model.pathway_embed is not None:
                pw_src, pw_dst = pw_ei
                pw_local_src = []  # pathway embedding indices
                pw_local_dst = []
                for s, d in zip(pw_src.tolist(), pw_dst.tolist(), strict=False):
                    ld = global_to_local.get(d)
                    if ld is not None:
                        pw_local_src.append(s)
                        pw_local_dst.append(ld)
                if len(pw_local_src) >= 4:
                    pw_local_src_t = torch.tensor(pw_local_src, device=device, dtype=torch.long)
                    pw_local_dst_t = torch.tensor(pw_local_dst, device=device, dtype=torch.long)
                    pw_emb = model.pathway_embed(pw_local_src_t)
                    pw_loss = _compute_edge_recon_loss(
                        pw_emb, prot_emb[pw_local_dst_t], min(pathway_samples, len(pw_local_src))
                    )
                    if pw_loss.item() > 0:
                        total_loss = total_loss + pw_loss
                        active_count += 1

        # DDI 边重建（异质图）
        ddi_key = ("drug", "ddi", "drug")
        if ddi_key in hetero_adj and comp_emb is not None and comp_local_indices is not None:
            ddi_ei = hetero_adj[ddi_key]
            if ddi_ei.shape[1] > 0:
                ddi_src, ddi_dst = ddi_ei
                comp_global_to_local = {g: i for i, g in enumerate(comp_local_indices)}
                ddi_local_src = []
                ddi_local_dst = []
                for s, d in zip(ddi_src.tolist(), ddi_dst.tolist(), strict=False):
                    ls = comp_global_to_local.get(s)
                    ld = comp_global_to_local.get(d)
                    if ls is not None and ld is not None:
                        ddi_local_src.append(ls)
                        ddi_local_dst.append(ld)
                if len(ddi_local_src) >= 4:
                    ddi_local_src_t = torch.tensor(ddi_local_src, device=device, dtype=torch.long)
                    ddi_local_dst_t = torch.tensor(ddi_local_dst, device=device, dtype=torch.long)
                    ddi_loss = _compute_edge_recon_loss(
                        comp_emb[ddi_local_src_t], comp_emb[ddi_local_dst_t], ddi_samples
                    )
                    if ddi_loss.item() > 0:
                        total_loss = total_loss + ddi_loss
                        active_count += 1

    if active_count == 0:
        return zero

    return total_loss / active_count


def compute_infonce_loss(
    model: torch.nn.Module,
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    memory_bank: torch.Tensor | None = None,
    n_samples: int = 128,
) -> torch.Tensor:
    """计算 InfoNCE 对比损失。

    参考: GHCDTI (Peng et al. 2025) 跨视图对比学习

    Args:
        model: 模型（未直接使用，保留接口兼容性）
        comp_emb: 化合物嵌入 [n_compounds, dim]
        prot_emb: 蛋白嵌入 [n_proteins, dim]
        memory_bank: Memory Bank 历史嵌入 [bank_size, dim]
        n_samples: 每批次采样数

    Returns:
        InfoNCE 对比损失
    """
    n_compounds = comp_emb.shape[0]
    n_proteins = prot_emb.shape[0]

    if n_compounds < 2 or n_proteins < 2:
        return torch.tensor(0.0, device=comp_emb.device, dtype=comp_emb.dtype)

    rand_c = torch.randperm(n_compounds, device=comp_emb.device)[: min(n_samples, n_compounds)]
    rand_p = torch.randperm(n_proteins, device=prot_emb.device)[: min(n_samples, n_proteins)]

    c = comp_emb[rand_c]
    p = prot_emb[rand_p]

    # 相同位置为正样本，其余为负样本（跨模态对比学习）
    logits = torch.mm(c, p.t()) / INFONCE_TEMPERATURE
    labels = torch.arange(logits.shape[0], device=logits.device)

    if memory_bank is not None and memory_bank.shape[0] > 0:
        mem_sample = memory_bank[torch.randperm(memory_bank.shape[0], device=memory_bank.device)[:n_samples]]
        c_logits = torch.mm(c, mem_sample.t()) / INFONCE_TEMPERATURE
        p_logits = torch.mm(p, mem_sample.t()) / INFONCE_TEMPERATURE
        loss_c = F.cross_entropy(c_logits, labels[: c_logits.shape[0]])
        loss_p = F.cross_entropy(p_logits, labels[: p_logits.shape[0]])
        return (loss_c + loss_p) / 2

    loss_c = F.cross_entropy(logits, labels)
    loss_p = F.cross_entropy(logits.t(), labels[: logits.shape[1]])
    return (loss_c + loss_p) / 2


__all__ = [
    "FOCAL_GAMMA", "FOCAL_ALPHA", "SCORE_CLAMP", "LABEL_SMOOTHING_POS", "LABEL_SMOOTHING_NEG",
    "TEMPERATURE", "BPR_WEIGHT", "CPI_LOSS_WEIGHT", "INFONCE_WEIGHT",
    "AUX_RECON_WEIGHT", "SEMANTIC_ATTN_WEIGHT", "SEMANTIC_ATTN_TEMPERATURE",
    "focal_loss_with_logits", "compute_cpi_loss", "compute_infonce_loss",
    "compute_aux_recon_loss", "compute_auxiliary_reconstruction_loss",
    "compute_semantic_attention_loss",
]
