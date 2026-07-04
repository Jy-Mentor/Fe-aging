"""损失函数模块 — Focal Loss + InfoNCE 对比损失

.. deprecated::
    此模块中的 `compute_cpi_loss` 已废弃，不再与主脚本保持同步。
    实际逻辑已迁移至 ``L4/scripts/phase4_v10_minibatch.py`` 中的
    ``_compute_cpi_loss``（v37+），其包含以下本模块缺失的特性：
      - ``compound_to_prot_locals`` 向量化 mask 预计算（v31）
      - ``_get_residue_indices`` 辅助函数 + ``prot_residue_indices`` 参数（v33）
      - ``ResidueAwareBilinearDecoder`` 的 OOM 降级处理（v37）
      - BPR 负样本的残基注意力路径（v37）

    本模块保留 ``focal_loss_with_logits`` 与 ``infonce_loss`` 两个基础函数，
    供其他模块独立使用。

参考:
  - Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV
  - Oord et al. (2018) "Representation Learning with Contrastive Predictive Coding", arXiv
  - He et al. (2020) "MoCo", CVPR
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .memory_bank import MemoryBank

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 模块级常量 ----
MASK_VAL = -1e9                  # 掩码值（屏蔽无效候选）
EPS = 1e-8                       # 数值稳定 epsilon
EPS_SMALL = 1e-10                # 小数 epsilon（用于 multinomial 分母保护）
SCORE_CLAMP = 10                 # 分数裁剪范围 [-10, 10]
LABEL_SMOOTHING_POS = 0.9        # 正样本标签平滑目标
LABEL_SMOOTHING_NEG = 0.1        # 负样本标签平滑目标
CPI_LOSS_WEIGHT = 0.6            # CPI 正负样本损失权重（Focal Loss 部分）
INFONCE_WEIGHT = 0.1             # InfoNCE 对比损失权重
INFONCE_WARMUP_RATIO = 0.15      # InfoNCE 预热占阶段 epoch 的比例
INFONCE_MEM_SAMPLE = 256         # InfoNCE 从 Memory Bank 采样数
INFONCE_TEMPERATURE = 0.07       # InfoNCE 对比损失温度
CURRICULUM_PHASE1 = 0.3          # 阶段1（随机负样本）的 epoch 比例
CURRICULUM_PHASE2 = 0.7          # 阶段2（中度负样本）开始的比例
MEDIUM_NEG_RATIO = 0.3           # 中度负样本占 unique 化合物的比例
HARD_NEG_RATIO = 0.1             # 极硬负样本占 unique 化合物的比例

# NaN/Inf 批次计数器（模块级，避免函数属性污染）
_nan_batch_count: int = 0
_MAX_NAN_BATCHES = 5


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    reduction: str = "mean",
) -> torch.Tensor:
    """Focal Loss：自动降低易分类样本的损失权重，聚焦困难样本

    参考: Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)  # p_t = sigmoid(logits) if target=1, else 1-sigmoid(logits)
    focal_weight = (1 - pt) ** gamma
    if alpha > 0:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_weight = alpha_t * focal_weight
    loss = focal_weight * bce
    return loss.mean() if reduction == "mean" else loss.sum()


def infonce_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    memory_scores: torch.Tensor | None = None,
    temperature: float = 0.07,
    memory_weight: float = 0.3,
) -> torch.Tensor:
    """InfoNCE 对比损失：最大化正样本对相似度，最小化负样本对相似度

    L = -log( exp(pos/τ) / (exp(pos/τ) + Σ exp(neg/τ) + Σ exp(mem/τ)) )

    当 memory_scores 不为 None 时，将 memory bank 中的全局负样本
    纳入分母，迫使决策边界更精确。

    参考:
      - Oord et al. (2018) "Representation Learning with Contrastive
        Predictive Coding", arXiv.
      - He et al. (2020) "MoCo", CVPR.

    Args:
        pos_scores: (N,) 正样本对得分
        neg_scores: (N, K) 或 (N,) 负样本对得分
        memory_scores: (N, M) 或 None，memory bank 全局负样本得分
        temperature: 温度参数 τ（默认 0.07）
        memory_weight: memory bank 负样本权重（0~1）

    Returns:
        scalar loss
    """
    pos = pos_scores / temperature
    neg = neg_scores / temperature

    if neg.dim() == 1:
        neg = neg.unsqueeze(1)

    # 分母：正样本 + 所有负样本
    denominator = torch.cat([pos.unsqueeze(1), neg], dim=1)

    if memory_scores is not None and memory_scores.numel() > 0:
        mem = memory_scores / temperature
        if mem.dim() == 1:
            mem = mem.unsqueeze(1)
        # 加权融合 memory bank 负样本
        denominator = torch.cat([denominator, memory_weight * mem], dim=1)

    loss = -pos + torch.logsumexp(denominator, dim=1)
    return loss.mean()


def compute_cpi_loss(
    model,
    comp_emb: torch.Tensor,
    prot_emb: torch.Tensor,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    comp_sorted: list[int],
    prot_map: dict[int, int],
    precomputed_pos: dict[int, list[int]],
    n_compounds: int,
    prot_to_path_neighbors: dict[int, set] | None,
    epoch: int,
    stage_epochs: int,
    memory_bank: MemoryBank,
    use_infonce: bool = False,  # v21: 消融实验结论 — 移除 InfoNCE 提升 SAGE +75%, HGT +23%
    bpr_weight: float = 0.4,
    use_curriculum: bool = True,  # v20: 消融实验开关
    use_topology_neg: bool = False,  # v23-topo: 是否使用PPI拓扑驱动的难负样本
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    prot_to_topo_medium_neighbors: dict[int, set] | None = None,
    prot_to_topo_hard_neighbors: dict[int, set] | None = None,
) -> torch.Tensor:
    """.. deprecated:: v25
        此函数已废弃。实际逻辑请使用 ``L4/scripts/phase4_v10_minibatch.py``
        中的 ``_compute_cpi_loss``（v37+），该版本新增了 ``compound_to_prot_locals``
        向量化 mask、``_get_residue_indices`` 残基索引辅助、以及 OOM 降级处理。
    """
    n_batch_prots = prot_emb.shape[0]
    T = model.temperature

    # 正样本
    pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst]) / T
    pos_score = torch.clamp(pos_score, -SCORE_CLAMP, SCORE_CLAMP)
    pos_loss = focal_loss_with_logits(
        pos_score, torch.full_like(pos_score, LABEL_SMOOTHING_POS), gamma=focal_gamma, alpha=focal_alpha)

    unique_src = pos_src.unique()
    n_unique = len(unique_src)
    if n_unique == 0 or n_batch_prots <= 1:
        return pos_loss

    batch_comp_emb = comp_emb[unique_src]
    all_scores = model.decode(
        batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
        prot_emb.repeat(n_unique, 1)
    ).reshape(n_unique, n_batch_prots) / T

    # 正样本 mask
    mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
    for i, src_local in enumerate(unique_src):
        src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
        if src_global >= 0 and src_global in precomputed_pos:
            for p_global in precomputed_pos[src_global]:
                p_local = p_global - n_compounds
                if p_local in prot_map:
                    p_idx = prot_map[p_local]
                    if p_idx < n_batch_prots:
                        mask[i, p_idx] = MASK_VAL

    # 课程阶段判定（按当前阶段总 epoch 计算）
    # v20: use_curriculum=False 时始终使用随机负样本
    if use_curriculum:
        curriculum_phase = epoch / stage_epochs
        if curriculum_phase < CURRICULUM_PHASE1:
            n_medium = n_hard = 0
        elif curriculum_phase < CURRICULUM_PHASE2:
            n_medium = int(n_unique * MEDIUM_NEG_RATIO)
            n_hard = 0
        else:
            n_hard = int(n_unique * HARD_NEG_RATIO)
            n_medium = 0
    else:
        n_medium = n_hard = 0

    # 初始化 hard_neg_scores 为随机负样本（排除正样本 + 保护全零行）
    hard_neg_scores = torch.zeros(n_unique, device=DEVICE)
    valid_mask = (mask == 0).float()
    row_sum = valid_mask.sum(dim=1)
    safe_rows = row_sum > 0
    if safe_rows.any():
        valid_mask = valid_mask / (row_sum.unsqueeze(1) + EPS_SMALL)
        # v25-fix: 仅对 safe_rows 采样，避免全零行触发 RuntimeError
        rand_dst = torch.multinomial(valid_mask[safe_rows], 1).squeeze(-1)
        hard_neg_scores[safe_rows] = model.decode(comp_emb[unique_src[safe_rows]], prot_emb[rand_dst]) / T
        hard_neg_scores = torch.clamp(hard_neg_scores, -SCORE_CLAMP, SCORE_CLAMP)

    # Phase 2: 中度负样本
    # v23-topo: 支持基于PPI拓扑的中度负样本，与通路共现策略互斥
    medium_neighbor_dict = prot_to_path_neighbors
    if use_topology_neg and prot_to_topo_medium_neighbors is not None:
        medium_neighbor_dict = prot_to_topo_medium_neighbors

    if n_medium > 0 and medium_neighbor_dict is not None and n_batch_prots > 2:
        medium_neg_scores = hard_neg_scores.clone()
        medium_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
        for i, src_local in enumerate(unique_src):
            src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
            if src_global < 0 or src_global not in precomputed_pos:
                continue
            topo_neighbors: set = set()
            for p_global in precomputed_pos[src_global]:
                p_local = p_global - n_compounds
                if p_local >= 0 and p_local in medium_neighbor_dict:
                    topo_neighbors.update(medium_neighbor_dict[p_local])
            if not topo_neighbors:
                continue
            batch_neighbor_positions = []
            for pn in topo_neighbors:
                if pn in prot_map:
                    pi = prot_map[pn]
                    if 0 <= pi < n_batch_prots and mask[i, pi] == 0:
                        batch_neighbor_positions.append(pi)
            if batch_neighbor_positions:
                bi_t = torch.tensor(batch_neighbor_positions, device=DEVICE)
                neighbor_scores = all_scores[i, bi_t]
                best_idx = neighbor_scores.argmax()
                medium_neg_scores[i] = torch.clamp(neighbor_scores[best_idx], -SCORE_CLAMP, SCORE_CLAMP)
                medium_found[i] = True

        medium_candidates = torch.where(medium_found)[0]
        if len(medium_candidates) > 0:
            n_actual = min(n_medium, len(medium_candidates))
            perm = torch.randperm(len(medium_candidates), device=DEVICE)
            hard_neg_scores[medium_candidates[perm[:n_actual]]] = medium_neg_scores[medium_candidates[perm[:n_actual]]]

    # Phase 3: 极硬负样本
    # v23-topo: 支持基于PPI拓扑的难负样本（共同邻居/高Jaccard）
    if n_hard > 0:
        if use_topology_neg and prot_to_topo_hard_neighbors is not None:
            hard_neg_scores_topo = hard_neg_scores.clone()
            hard_found = torch.zeros(n_unique, dtype=torch.bool, device=DEVICE)
            for i, src_local in enumerate(unique_src):
                src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
                if src_global < 0 or src_global not in precomputed_pos:
                    continue
                topo_hard: set = set()
                for p_global in precomputed_pos[src_global]:
                    p_local = p_global - n_compounds
                    if p_local >= 0 and p_local in prot_to_topo_hard_neighbors:
                        topo_hard.update(prot_to_topo_hard_neighbors[p_local])
                if not topo_hard:
                    continue
                batch_hard_positions = []
                for pn in topo_hard:
                    if pn in prot_map:
                        pi = prot_map[pn]
                        if 0 <= pi < n_batch_prots and mask[i, pi] == 0:
                            batch_hard_positions.append(pi)
                if batch_hard_positions:
                    bi_t = torch.tensor(batch_hard_positions, device=DEVICE)
                    hard_neighbor_scores = all_scores[i, bi_t]
                    best_idx = hard_neighbor_scores.argmax()
                    hard_neg_scores_topo[i] = torch.clamp(hard_neighbor_scores[best_idx], -SCORE_CLAMP, SCORE_CLAMP)
                    hard_found[i] = True

            hard_candidates = torch.where(hard_found)[0]
            if len(hard_candidates) > 0:
                n_actual = min(n_hard, len(hard_candidates))
                perm = torch.randperm(len(hard_candidates), device=DEVICE)
                selected = hard_candidates[perm[:n_actual]]
                hard_neg_scores[selected] = hard_neg_scores_topo[selected]
        else:
            hard_neg_idx = (all_scores + mask).argmax(dim=1)
            hard_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
            hard_scores = torch.clamp(hard_scores, -SCORE_CLAMP, SCORE_CLAMP)
            hard_candidates = torch.randperm(n_unique, device=DEVICE)[:n_hard]
            hard_neg_scores[hard_candidates] = hard_scores[hard_candidates]

    # Focal Loss
    neg_loss = focal_loss_with_logits(
        hard_neg_scores, torch.full_like(hard_neg_scores, LABEL_SMOOTHING_NEG), gamma=focal_gamma, alpha=focal_alpha)

    # 正样本对 -> unique_src 位置映射（用于 BPR 和 InfoNCE）
    src_to_pos = {s.item(): i for i, s in enumerate(unique_src)}
    pos_indices = torch.tensor(
        [src_to_pos[s.item()] for s in pos_src],
        device=DEVICE, dtype=torch.long
    )

    # BPR 损失
    pair_mask = mask[pos_indices]
    bpr_valid_mask = (pair_mask == 0).float()
    bpr_row_sum = bpr_valid_mask.sum(dim=1)
    bpr_safe = bpr_row_sum > 0
    bpr_neg_scores = torch.zeros(len(pos_src), device=DEVICE)
    if bpr_safe.any():
        bpr_valid_mask[bpr_safe] = bpr_valid_mask[bpr_safe] / bpr_row_sum[bpr_safe].unsqueeze(1)
        # v25-fix: 仅对 bpr_safe 行采样，避免全零行触发 RuntimeError
        bpr_neg_dst = torch.multinomial(bpr_valid_mask[bpr_safe], 1).squeeze(-1)
        bpr_neg_scores[bpr_safe] = all_scores[pos_indices[bpr_safe], bpr_neg_dst]
    # v25-fix: 对unsafe行使用(all_scores + mask)中最低分蛋白作为替代负样本
    # mask 已将正样本设为 MASK_VAL=-1e9，确保 min 不会选到正样本
    bpr_unsafe = ~bpr_safe
    if bpr_unsafe.any():
        bpr_neg_scores[bpr_unsafe] = (all_scores[pos_indices[bpr_unsafe]] + mask[pos_indices[bpr_unsafe]]).min(dim=1).values
    bpr_loss = -torch.log(torch.sigmoid(pos_score - bpr_neg_scores) + EPS).mean()

    loss = CPI_LOSS_WEIGHT * (pos_loss + neg_loss) + bpr_weight * bpr_loss

    # InfoNCE
    # v20-fix: 原 epoch > 50 在预训练(10) + 微调(15) 周期下永不触发，改为按阶段 epoch 比例触发
    infonce_warmup = max(2, int(stage_epochs * INFONCE_WARMUP_RATIO))
    if use_infonce and epoch > infonce_warmup and memory_bank.size() > 0 and len(pos_indices) > 0:
        n_mem = min(INFONCE_MEM_SAMPLE, memory_bank.size())
        mem_emb = memory_bank.sample(n_mem)
        if mem_emb.shape[0] > 0:
            pos_idx_sub = pos_indices[:len(pos_score)]
            mem_scores = model.decode(
                comp_emb[unique_src[pos_idx_sub]].unsqueeze(1).expand(-1, n_mem, -1).reshape(-1, model.out_dim),
                mem_emb.repeat(len(pos_idx_sub), 1)
            ).reshape(len(pos_idx_sub), n_mem)
            infonce = infonce_loss(
                pos_score[:len(pos_idx_sub)] * T,
                hard_neg_scores[pos_idx_sub] * T,
                memory_scores=mem_scores, temperature=INFONCE_TEMPERATURE,
            )
            loss = loss + INFONCE_WEIGHT * infonce

    if torch.isnan(loss) or torch.isinf(loss):
        global _nan_batch_count
        _nan_batch_count += 1
        if _nan_batch_count >= _MAX_NAN_BATCHES:
            raise RuntimeError(
                f"compute_cpi_loss 连续 {_nan_batch_count} 个 batch 产生 NaN/Inf loss，"
                f"可能是梯度爆炸或数据异常，请检查学习率、模型初始化或输入数据")
        logger.warning(
            f"compute_cpi_loss 产生 NaN/Inf loss（连续 {_nan_batch_count}/{_MAX_NAN_BATCHES}），"
            f"返回零损失以保护训练")
        return torch.tensor(0.0, device=loss.device, requires_grad=False)
    else:
        _nan_batch_count = 0  # 正常 batch 重置计数器

    return loss