"""损失函数模块 — Focal Loss + InfoNCE 对比损失 + CPI 损失 + 辅助重建损失

本模块提供：
  - ``focal_loss_with_logits``: Focal Loss 基础函数
  - ``infonce_loss``: InfoNCE 对比损失基础函数
  - ``_CpiLossState``: CPI 损失运行期状态（OOM/NaN 跟踪）
  - ``compute_cpi_loss``: CPI 损失（v67，与主脚本同步）
  - ``compute_auxiliary_reconstruction_loss``: 辅助网络重建损失（v67）

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
INFONCE_MEM_SAMPLE = 256         # InfoNCE 从 Memory Bank 采样数（须与 config memory_bank.infonce_mem_sample 一致）
INFONCE_TEMPERATURE = 0.07       # InfoNCE 对比损失温度
CURRICULUM_PHASE1 = 0.3          # 阶段1（随机负样本）的 epoch 比例
CURRICULUM_PHASE2 = 0.7          # 阶段2（中度负样本）开始的比例
MEDIUM_NEG_RATIO = 0.3           # 中度负样本占 unique 化合物的比例
HARD_NEG_RATIO = 0.1             # 极硬负样本占 unique 化合物的比例


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


class _CpiLossState:
    """compute_cpi_loss 运行期状态，替代模块级全局变量。"""

    def __init__(self) -> None:
        self.nan_batch_counter: int = 0
        self.pos_oom_counter: int = 0
        self.hard_neg_oom_counter: int = 0
        self.bpr_oom_counter: int = 0


# 默认共享状态：训练器不注入 _state 时仍可跨 batch 累计 OOM/NaN 次数。
_default_cpi_loss_state = _CpiLossState()


# [Ref: 3] Focal Loss: Lin et al. (2017) ICCV (α=0.75, γ=2.0)
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
    compound_to_prot_locals: dict[int, list[int]] | None = None,
    use_infonce: bool = False,
    bpr_weight: float = 0.4,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    prot_to_topo_medium_neighbors: dict[int, set] | None = None,
    prot_to_topo_hard_neighbors: dict[int, set] | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    _state: _CpiLossState | None = None,
    use_residue_decoder: bool = True,
    bpr_detach_neg: bool = True,
) -> torch.Tensor:
    """v67: 共享的 CPI 损失计算（Focal + BPR + 课程负采样）— InfoNCE 默认关闭

    Args:
        model: 拥有 decode() 与 temperature 属性的模型
        comp_emb: (n_batch_compounds, out_dim) 化合物嵌入
        prot_emb: (n_batch_proteins, out_dim) 蛋白嵌入
        pos_src: (n_pos,) 正样本对的化合物局部索引
        pos_dst: (n_pos,) 正样本对的蛋白局部索引
        comp_sorted: 局部化合物索引 -> 全局化合物索引
        prot_map: 局部蛋白索引 (p_global - n_compounds) -> batch 蛋白索引
        precomputed_pos: 全局化合物 -> 正样本蛋白全局索引集合
        n_compounds: 化合物总数，用于蛋白全局/局部索引转换
        prot_to_path_neighbors: 蛋白 -> 同通路蛋白局部索引集合
        epoch: 当前 epoch（用于课程阶段判定）
        stage_epochs: 当前阶段总 epoch 数
        memory_bank: MemoryBank 实例
        compound_to_prot_locals: 向量化 mask 预计算映射表
        use_infonce: 是否启用 InfoNCE（预训练阶段可关闭）
        use_topology_neg: 为True时，优先使用PPI拓扑邻居替代通路邻居。
        prot_to_topo_medium_neighbors: 蛋白 -> 拓扑中度负样本局部索引集合。
        prot_to_topo_hard_neighbors: 蛋白 -> 拓扑难负样本局部索引集合。
        _state: 运行期状态实例；未提供时自动创建。
        use_residue_decoder: 为False时跳过残基解码器（预训练阶段降显存）。
        bpr_detach_neg: BPR负样本是否detach。预训练True（显存保护），微调False（优化负样本嵌入）。

    Returns:
        loss 标量张量
    """
    state = _state if _state is not None else _default_cpi_loss_state
    n_batch_prots = prot_emb.shape[0]
    T = model.temperature

    # 诊断断言，确保 prot_map 键类型正确
    if prot_map:
        assert all(isinstance(k, int) for k in prot_map.keys()),             "compute_cpi_loss: prot_map 键必须是整数局部蛋白索引"
        assert all(0 <= v < n_batch_prots for v in prot_map.values()),             f"compute_cpi_loss: prot_map 值越界，应在 [0, {n_batch_prots}) 内"

    # 构建 batch 蛋白位置 -> 图蛋白局部索引的逆映射，用于 residue_bilinear 解码器。
    # residue_bilinear 的 _prot_to_residue_idx 以图蛋白局部索引（0~n_proteins-1）为键。
    prot_inv_map = {v: k for k, v in prot_map.items()} if prot_map else {}

    def _get_residue_indices(batch_positions: torch.Tensor) -> torch.Tensor | None:
        """将 batch 蛋白位置转换为图蛋白局部索引，供 residue_bilinear 查找残基特征。"""
        if not prot_inv_map:
            return None
        return torch.tensor(
            [prot_inv_map.get(p.item(), -1) for p in batch_positions],
            device=batch_positions.device, dtype=torch.long
        )

    # 正样本（预训练阶段跳过残基解码器以节省显存）
    pos_residue_idx = _get_residue_indices(pos_dst) if use_residue_decoder else None
    try:
        pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst], prot_residue_indices=pos_residue_idx) / T
    except torch.cuda.OutOfMemoryError as e:
        # 正样本 OOM 降级为 fast bilinear
        state.pos_oom_counter += 1
        logger.warning(
            f"compute_cpi_loss: pos 残基路径 OOM（连续 {state.pos_oom_counter} 次），"
            f"降级为 fast bilinear: {e}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pos_score = model.decode(comp_emb[pos_src], prot_emb[pos_dst], prot_residue_indices=None) / T
    pos_score = torch.clamp(pos_score, -SCORE_CLAMP, SCORE_CLAMP)
    pos_loss = focal_loss_with_logits(
        pos_score, torch.full_like(pos_score, LABEL_SMOOTHING_POS), gamma=focal_gamma, alpha=focal_alpha)

    # v61-fix: 当训练走残基路径时，额外对 fast bilinear 路径施加正样本监督，
    # 避免验证阶段统一使用 fast bilinear 时因该路径未被训练而 AUC 崩溃。
    if use_residue_decoder and model.decoder_type == "residue_bilinear":
        pos_score_fb = model.decode(comp_emb[pos_src], prot_emb[pos_dst], prot_residue_indices=None) / T
        pos_score_fb = torch.clamp(pos_score_fb, -SCORE_CLAMP, SCORE_CLAMP)
        pos_loss_fb = focal_loss_with_logits(
            pos_score_fb, torch.full_like(pos_score_fb, LABEL_SMOOTHING_POS), gamma=focal_gamma, alpha=focal_alpha)
        pos_loss = pos_loss + pos_loss_fb

    unique_src = pos_src.unique()
    n_unique = len(unique_src)
    if n_unique == 0 or n_batch_prots <= 1:
        return pos_loss

    batch_comp_emb = comp_emb[unique_src]
    # 全蛋白 pair-matrix 规模极大，残基注意力会导致 8GB 显存图内存爆炸；
    # 此处退化为 GNN 全局嵌入点积，仅对正样本/难负样本启用残基解码器。
    # v46: 用 torch.no_grad() 包裹，all_scores 仅用于 hard neg 选择，不需要梯度
    with torch.no_grad():
        all_scores = model.decode(
            batch_comp_emb.unsqueeze(1).expand(-1, n_batch_prots, -1).reshape(-1, model.out_dim),
            prot_emb.repeat(n_unique, 1),
            prot_residue_indices=None,
        ).reshape(n_unique, n_batch_prots) / T

    # 向量化正样本mask构建 — 使用预计算 compound_to_prot_locals 映射表
    mask = torch.zeros(n_unique, n_batch_prots, device=DEVICE)
    src_indices = []
    dst_indices = []
    if compound_to_prot_locals is not None:
        for i, src_local in enumerate(unique_src):
            src_local_val = src_local.item()
            if src_local_val < len(comp_sorted):
                src_global = comp_sorted[src_local_val]
                if src_global >= 0 and src_global in compound_to_prot_locals:
                    for p_local in compound_to_prot_locals[src_global]:
                        if p_local in prot_map:
                            p_idx = prot_map[p_local]
                            if p_idx < n_batch_prots:
                                src_indices.append(i)
                                dst_indices.append(p_idx)
    else:
        # 回退：未提供预计算映射表时使用原始双循环
        for i, src_local in enumerate(unique_src):
            src_global = comp_sorted[src_local.item()] if src_local.item() < len(comp_sorted) else -1
            if src_global >= 0 and src_global in precomputed_pos:
                for p_global in precomputed_pos[src_global]:
                    p_local = p_global - n_compounds
                    if p_local in prot_map:
                        p_idx = prot_map[p_local]
                        if p_idx < n_batch_prots:
                            mask[i, p_idx] = MASK_VAL
    if src_indices:
        mask.index_put_(
            (torch.tensor(src_indices, device=DEVICE, dtype=torch.long),
             torch.tensor(dst_indices, device=DEVICE, dtype=torch.long)),
            torch.tensor(MASK_VAL, device=DEVICE)
        )

    # v60-fix: 课程阶段判定按累积 epoch 比例计算。
    # trainer 传入的 epoch 已为 epoch + cumulative_epoch_offset，stage_epochs 为总 epoch 数
    # （pretrain + finetune），避免微调阶段重置回随机负采样。
    # use_curriculum=False 时始终使用随机负样本
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
        # 仅对 safe_rows 采样，避免全零行触发 RuntimeError
        rand_dst = torch.multinomial(valid_mask[safe_rows], 1).squeeze(-1)
        # 随机负样本尝试使用残基路径，OOM 时降级为 fast bilinear
        # v49-fix: 随机负样本同样尊重 use_residue_decoder，避免预训练阶段只更新残基路径参数
        # 而正样本走 fast bilinear 导致残基路径无约束、破坏训练稳定性。
        rand_residue_idx = _get_residue_indices(rand_dst) if use_residue_decoder else None
        try:
            hard_neg_scores[safe_rows] = model.decode(
                comp_emb[unique_src[safe_rows]], prot_emb[rand_dst],
                prot_residue_indices=rand_residue_idx,
            ) / T
        except torch.cuda.OutOfMemoryError as e:
            state.hard_neg_oom_counter += 1
            logger.warning(
                f"compute_cpi_loss: hard_neg 残基路径 OOM（连续 {state.hard_neg_oom_counter} 次），"
                f"降级为 fast bilinear: {e}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            hard_neg_scores[safe_rows] = model.decode(
                comp_emb[unique_src[safe_rows]], prot_emb[rand_dst],
                prot_residue_indices=None,
            ) / T
        hard_neg_scores = torch.clamp(hard_neg_scores, -SCORE_CLAMP, SCORE_CLAMP)

    # Phase 2: 中度负样本
    # 支持基于PPI拓扑的中度负样本，与通路共现策略互斥
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
    # 支持基于PPI拓扑的难负样本（共同邻居/高Jaccard）
    # v43: 额外引入 Memory Bank 全局蛋白嵌入作为跨 batch 的极硬负样本。
    hard_neg_from_memory = torch.zeros(n_unique, device=DEVICE)
    has_memory_neg = False
    if n_hard > 0 and memory_bank.size() > 0:
        n_mem = min(256, memory_bank.size())
        mem_emb = memory_bank.sample(n_mem)
        if mem_emb.shape[0] > 0:
            # 计算当前 batch 化合物与全局 memory bank 蛋白的得分
            mem_scores = model.decode(
                batch_comp_emb.unsqueeze(1).expand(-1, mem_emb.shape[0], -1).reshape(-1, model.out_dim),
                mem_emb.repeat(n_unique, 1),
                prot_residue_indices=None,
            ).reshape(n_unique, -1)
            mem_hard_scores, _ = mem_scores.max(dim=1)
            hard_neg_from_memory = torch.clamp(mem_hard_scores, -SCORE_CLAMP, SCORE_CLAMP)
            has_memory_neg = True

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

            if has_memory_neg:
                # 取拓扑难负样本与 memory bank 难负样本中的更高分者
                hard_neg_scores_topo = torch.maximum(hard_neg_scores_topo, hard_neg_from_memory)
                hard_found = hard_found | (hard_neg_from_memory > hard_neg_scores)

            hard_candidates = torch.where(hard_found)[0]
            if len(hard_candidates) > 0:
                n_actual = min(n_hard, len(hard_candidates))
                perm = torch.randperm(len(hard_candidates), device=DEVICE)
                selected = hard_candidates[perm[:n_actual]]
                hard_neg_scores[selected] = hard_neg_scores_topo[selected]
        else:
            hard_neg_idx = (all_scores + mask).argmax(dim=1)
            hard_scores = all_scores[torch.arange(n_unique, device=DEVICE), hard_neg_idx]
            if has_memory_neg:
                hard_scores = torch.maximum(hard_scores, hard_neg_from_memory)
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

    # BPR 损失 — v60: 每个正样本对独立采样负样本，避免同一化合物的多个
    # 正样本对共享同一个 hardest 负样本，增强排序训练的多样性。
    # v60-fix: bpr_weight=0 时完全跳过 BPR 计算，避免无效开销和潜在死锁。
    bpr_loss = torch.tensor(0.0, device=DEVICE)
    if bpr_weight > 0:
        valid_counts = (mask[pos_indices] == 0).sum(dim=1)
        has_valid = valid_counts > 0
        bpr_neg_scores = torch.zeros(len(pos_src), device=DEVICE)
        if has_valid.any():
            # 向量化：对每条正样本对，在其有效负样本上按温度加权 softmax 采样。
            bpr_score_matrix = all_scores[pos_indices[has_valid]]  # (n_valid, n_batch_prots)
            bpr_mask_matrix = mask[pos_indices[has_valid]]         # 正样本位置为 MASK_VAL
            bpr_weights = torch.softmax((bpr_score_matrix + bpr_mask_matrix) / T, dim=1)
            # 防止极端数值导致概率全零
            bpr_weights = torch.clamp(bpr_weights, min=EPS_SMALL)
            bpr_weights = bpr_weights / bpr_weights.sum(dim=1, keepdim=True)
            neg_idx = torch.multinomial(bpr_weights, 1).squeeze(-1)  # (n_valid,)
            bpr_neg_scores[has_valid] = all_scores[pos_indices[has_valid], neg_idx]
        if (~has_valid).any():
            # 无有效负样本时回退到 hardest 负样本得分
            bpr_neg_scores[~has_valid] = hard_neg_scores[pos_indices[~has_valid]]
        # BPR 负样本 detach 控制 — 预训练时 detach（显存保护/粗粒度），微调时保留梯度（优化负样本嵌入）
        bpr_neg_for_loss = bpr_neg_scores.detach() if bpr_detach_neg else bpr_neg_scores
        bpr_loss = -torch.log(torch.sigmoid(pos_score - bpr_neg_for_loss) + EPS).mean()

    loss = CPI_LOSS_WEIGHT * (pos_loss + neg_loss) + bpr_weight * bpr_loss

    # InfoNCE
    # 原 epoch > 50 在预训练(10) + 微调(15) 周期下永不触发，改为按阶段 epoch 比例触发
    infonce_warmup = max(2, int(stage_epochs * INFONCE_WARMUP_RATIO))
    if use_infonce and epoch > infonce_warmup and memory_bank.size() > 0 and len(pos_indices) > 0:
        n_mem = min(INFONCE_MEM_SAMPLE, memory_bank.size())
        mem_emb = memory_bank.sample(n_mem)
        if mem_emb.shape[0] > 0:
            pos_idx_sub = pos_indices[:len(pos_score)]
            mem_scores = model.decode(
                comp_emb[unique_src[pos_idx_sub]].unsqueeze(1).expand(-1, n_mem, -1).reshape(-1, model.out_dim),
                mem_emb.repeat(len(pos_idx_sub), 1),
                prot_residue_indices=None,  # Memory Bank 无对应蛋白索引，residue_bilinear 退化为点积
            ).reshape(len(pos_idx_sub), n_mem)
            infonce = infonce_loss(
                pos_score[:len(pos_idx_sub)] * T,
                hard_neg_scores[pos_idx_sub] * T,
                memory_scores=mem_scores, temperature=INFONCE_TEMPERATURE,
            )
            loss = loss + INFONCE_WEIGHT * infonce

    if torch.isnan(loss) or torch.isinf(loss):
        state.nan_batch_counter += 1
        if state.nan_batch_counter >= 5:
            raise RuntimeError(
                f"compute_cpi_loss 连续 {state.nan_batch_counter} 个 batch 产生 NaN/Inf loss，"
                f"可能是梯度爆炸或数据异常，请检查学习率、模型初始化或输入数据")
        logger.warning(
            f"compute_cpi_loss 产生 NaN/Inf loss（连续 {state.nan_batch_counter}/5），"
            f"返回零损失以保护训练")
        return torch.tensor(0.0, device=loss.device, requires_grad=False)
    else:
        state.nan_batch_counter = 0  # 正常 batch 重置计数器

    return loss


# v67: 辅助网络重建损失函数
def compute_auxiliary_reconstruction_loss(*, model, prot_emb, prot_local_indices, homo_adj,
    n_compounds, ppi_samples=256, prot_to_path_neighbors=None, is_hetero=False,
    comp_emb=None, comp_local_indices=None, ddi_samples=128,
    prot_disease_samples=128, hetero_adj=None, n_diseases=0, disease_embed=None):
    device = prot_emb.device
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0
    if ppi_samples > 0 and len(prot_local_indices) >= 2:
        n_local = len(prot_local_indices)
        n_ppi = min(ppi_samples, n_local * (n_local - 1) // 2)
        pos_pairs, neg_pairs = [], []
        prot_set = set(prot_local_indices)
        for i_idx, pi in enumerate(prot_local_indices):
            for pj in homo_adj.get(pi, []):
                if pj in prot_set and pi < pj:
                    try: pos_pairs.append((i_idx, prot_local_indices.index(pj)))
                    except ValueError: pass
        import random as _r; rng = _r.Random(42); attempts = 0
        while len(neg_pairs) < n_ppi and attempts < n_ppi * 10:
            a, b = rng.randint(0, n_local-1), rng.randint(0, n_local-1)
            attempts += 1
            if a >= b or (a,b) in pos_pairs or (b,a) in pos_pairs or (a,b) in neg_pairs: continue
            neg_pairs.append((a,b))
        if pos_pairs and neg_pairs:
            n = min(len(pos_pairs), len(neg_pairs))
            pp = torch.tensor(pos_pairs[:n], dtype=torch.long, device=device)
            pn = torch.tensor(neg_pairs[:n], dtype=torch.long, device=device)
            ps = (prot_emb[pp[:,0]] * prot_emb[pp[:,1]]).sum(dim=1)
            ns = (prot_emb[pn[:,0]] * prot_emb[pn[:,1]]).sum(dim=1)
            total_loss = total_loss + F.binary_cross_entropy_with_logits(torch.cat([ps,ns]), torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]))
            n_terms += 1
    if ddi_samples > 0 and comp_emb is not None and comp_local_indices is not None and len(comp_local_indices) >= 2:
        n_lc = len(comp_local_indices)
        import random as _r2; rng2 = _r2.Random(42)
        n_ddi = min(ddi_samples, n_lc*(n_lc-1)//2)
        dpairs = [(rng2.randint(0,n_lc-1), rng2.randint(0,n_lc-1)) for _ in range(n_ddi*2)]
        dpairs = [(a,b) for a,b in dpairs if a < b][:n_ddi]
        if dpairs:
            di = torch.tensor(dpairs, dtype=torch.long, device=device)
            ds = (comp_emb[di[:,0]] * comp_emb[di[:,1]]).sum(dim=1)
            dc = F.cosine_similarity(comp_emb[di[:,0]], comp_emb[di[:,1]], dim=1)
            dl = (dc > 0.5).float().detach()
            if dl.sum() > 0 and (1-dl).sum() > 0:
                total_loss = total_loss + F.binary_cross_entropy_with_logits(ds, dl)
                n_terms += 1
    if is_hetero and prot_disease_samples > 0 and hetero_adj is not None and n_diseases > 0:
        n_lp = len(prot_local_indices)
        pedges = hetero_adj.get(("protein","associated_with","disease"), {})
        import random as _r3; rng3 = _r3.Random(42)
        pos_pd = [(li, d) for li, gp in enumerate(prot_local_indices) for d in pedges.get(gp,[]) if d < n_diseases]
        ns = min(prot_disease_samples, n_lp * n_diseases)
        neg_pd, att = [], 0
        while len(neg_pd) < ns and att < ns*10:
            p, d = rng3.randint(0,n_lp-1), rng3.randint(0,n_diseases-1)
            att += 1
            if (p,d) in pos_pd or (p,d) in neg_pd: continue
            neg_pd.append((p,d))
        if pos_pd and neg_pd:
            n = min(len(pos_pd), len(neg_pd))
            pp = torch.tensor(pos_pd[:n], dtype=torch.long, device=device)
            pn = torch.tensor(neg_pd[:n], dtype=torch.long, device=device)
            de = disease_embed if disease_embed is not None else model.disease_embed(torch.arange(n_diseases, device=device))
            ps = (prot_emb[pp[:,0]] * de[pp[:,1]]).sum(dim=1)
            ns = (prot_emb[pn[:,0]] * de[pn[:,1]]).sum(dim=1)
            total_loss = total_loss + F.binary_cross_entropy_with_logits(torch.cat([ps,ns]), torch.cat([torch.ones_like(ps), torch.zeros_like(ns)]))
            n_terms += 1
    if n_terms == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / n_terms