"""Pipeline 验证函数 — SAGE / HGT / HGT mini-batch / SimpleHGN 验证

所有函数从 phase4_v10_minibatch.py 提取，通过依赖注入接收配置参数，
消除对模块级全局常量的依赖。
"""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import torch

from ..evaluation.metrics import compute_early_enrichment_metrics, compute_ranking_metrics
from ..graph.sampling import sample_hetero_subgraph
from ..models import HGTLinkPredictor, SimpleHGNLinkPredictor

logger = logging.getLogger(__name__)


def _compute_ranking_metrics(score_matrix, valid_pos_list, ks=(10, 20, 50)):
    """委托至标准化 metrics 模块计算 Precision@K / Recall@K / Hit@K / NDCG@K / EF。"""
    return compute_ranking_metrics(score_matrix, valid_pos_list, ks=ks)


# ============================================================================
# validate_sage
# ============================================================================

def validate_sage(model, x, homo_edge_index, val_compounds, all_compound_to_pos, n_compounds,
                  device, score_clamp, hard_neg_top_k, rand_neg_top_k, mask_val,
                  neg_ratio: int = 100,
                  return_embeddings: bool = False):
    """v63: SAGE 验证 — 批量 MLP 解码器评分，避免 Python 循环反复 forward

    return_embeddings=True 时返回 (metrics_dict, node_emb_tensor)，
    供调用方复用全图嵌入，避免重复 GPU 前向传播（修复问题 A）。

    论文引用:
      - GraphSAGE: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs", NeurIPS.
      - 药物筛选评估: Rifaioglu et al. (2021) "Recent applications of deep learning and machine
        intelligence on in silico drug discovery", Briefings in Bioinformatics."""
    with torch.no_grad():
        x_dev = x.to(device)
        edge_index = homo_edge_index.to(device)
        node_emb = model(x_dev, edge_index)  # n_compounds=None 使用 self.n_compounds
        # v64: 全图前向传播完成后立即释放输入特征矩阵和边索引，
        # 减少峰值显存，避免与后续双维度分块评分争抢显存。
        del x_dev, edge_index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        prot_emb = node_emb[n_compounds:]
        comp_emb = node_emb[:n_compounds]
        T = model.temperature
        n_prots = prot_emb.shape[0]

        val_compounds_list = list(val_compounds)
        n_val = len(val_compounds_list)
        if n_val == 0:
            r = {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}
            return (r, node_emb) if return_embeddings else r

        # v58: 双维度分块 — 同时沿化合物和蛋白维度分块，避免 RTX 5060 8GB 显存碎片化 OOM。
        # 化合物 128 × 蛋白 6846 = 876K pairs，单次 expand+reshape 产生 ~450MB 中间张量，
        # 加上后续 linear 层分配，在 WDDM 驱动下极易触发碎片化分配失败。
        comp_sub = comp_emb[val_compounds_list]  # (n_val, d)
        comp_batch = 32
        prot_batch = 1024
        score_chunks = []
        for c_start in range(0, n_val, comp_batch):
            c_end = min(c_start + comp_batch, n_val)
            sub_comp = comp_sub[c_start:c_end]
            n_c = c_end - c_start
            row_chunks = []
            for p_start in range(0, n_prots, prot_batch):
                p_end = min(p_start + prot_batch, n_prots)
                sub_prot = prot_emb[p_start:p_end]
                n_p = p_end - p_start
                # 显式 repeat 替代 expand+reshape，避免非连续视图触发隐式拷贝
                comp_exp = sub_comp.repeat_interleave(n_p, dim=0)
                prot_exp = sub_prot.repeat(n_c, 1)
                sub_scores = model.decode(
                    comp_exp, prot_exp, prot_residue_indices=None
                ).reshape(n_c, n_p) / T
                row_chunks.append(sub_scores)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            score_chunks.append(torch.cat(row_chunks, dim=1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score_matrix = torch.cat(score_chunks, dim=0)  # (n_val, n_prots)

        # v51: 训练/验证 decoder 路径一致性 — 正样本使用残基注意力，负样本使用 fast bilinear。
        # 验证阶段全 pair-matrix 仍用快速双线性打分（高效），仅对正样本重新计算残基注意力分数，
        # 避免 47M+ 对全部走残基路径导致验证不可接受地慢，同时消除正样本的分布偏移。
        use_residue_in_val = (
            hasattr(model, "decoder")
            and model.decoder.__class__.__name__ == "ResidueAwareBilinearDecoder"
        )
        pos_key_to_residue_score: dict[tuple[int, int], float] = {}
        if use_residue_in_val:
            pos_src_list, pos_dst_list = [], []
            for idx, src in enumerate(val_compounds_list):
                pos_set = all_compound_to_pos.get(src, set())
                valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_prots]
                for p in valid_pos:
                    pos_src_list.append(idx)
                    pos_dst_list.append(p)
            if pos_src_list:
                pos_src_t = torch.tensor(pos_src_list, device=device, dtype=torch.long)
                pos_dst_t = torch.tensor(pos_dst_list, device=device, dtype=torch.long)
                pos_comp_emb = comp_sub[pos_src_t]
                pos_prot_emb = prot_emb[pos_dst_t]
                try:
                    pos_residue_scores = model.decode(
                        pos_comp_emb, pos_prot_emb, prot_residue_indices=pos_dst_t
                    ).reshape(-1) / T
                    pos_residue_scores = torch.clamp(pos_residue_scores, -score_clamp, score_clamp)
                    for s, d, score in zip(pos_src_list, pos_dst_list, pos_residue_scores, strict=False):
                        pos_key_to_residue_score[(int(s), int(d))] = torch.sigmoid(score).item()
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"_validate_sage: 正样本残基路径 OOM，验证回退到 fast bilinear: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        y_true, y_score = [], []
        valid_pos_list = []  # 收集 per-compound 正样本局部索引，用于排名指标
        n_valid = 0
        for idx, src in enumerate(val_compounds_list):
            pos_set = all_compound_to_pos.get(src, set())
            # pos_set 存全局索引，转为局部索引
            valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_prots]
            if not valid_pos:
                continue
            n_valid += 1
            valid_pos_list.append(valid_pos)

            scores = score_matrix[idx]

            # 正样本（residue_bilinear 时使用残基注意力分数）
            for p in valid_pos:
                y_true.append(1)
                key = (idx, p)
                if key in pos_key_to_residue_score:
                    y_score.append(pos_key_to_residue_score[key])
                else:
                    y_score.append(torch.sigmoid(scores[p]).item())

            # 固定 1:neg_ratio 随机负采样（文献标准做法，与训练时课程负采样解耦）
            # 每个正样本对应 neg_ratio 个随机负样本，确保 AUC/AUPR 可对比
            neg_mask = torch.ones(n_prots, device=device, dtype=torch.bool)
            for p in valid_pos:
                neg_mask[p] = False
            neg_candidates = torch.where(neg_mask)[0]
            n_neg_target = len(valid_pos) * neg_ratio
            n_neg = min(n_neg_target, len(neg_candidates))
            if n_neg > 0:
                neg_idx = neg_candidates[torch.randperm(len(neg_candidates), device=device)[:n_neg]]
                for ri in neg_idx:
                    y_true.append(0)
                    y_score.append(torch.sigmoid(scores[ri]).item())

        if len(y_true) < 2 or len(set(y_true)) < 2:
            r = {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}
            return (r, node_emb) if return_embeddings else r

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)
        nan_mask = np.isnan(y_score_arr) | np.isinf(y_score_arr)
        if nan_mask.any():
            logger.warning(f"_validate_sage: 验证分数含 {nan_mask.sum()} 个 NaN/Inf，已过滤")
            valid_idx = ~nan_mask
            y_true_arr = y_true_arr[valid_idx]
            y_score_arr = y_score_arr[valid_idx]
            if len(y_true_arr) < 2 or len(set(y_true_arr)) < 2:
                r = {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}
                return (r, node_emb) if return_embeddings else r

        # v50: 使用标准化指标模块统一计算 AUC/AUPR/EF/ROCE/BEDROC
        result = compute_early_enrichment_metrics(
            y_true_arr, y_score_arr,
            score_matrix=score_matrix, valid_pos_list=valid_pos_list,
        )
        result["n_valid_compounds"] = n_valid
        return (result, node_emb) if return_embeddings else result


# ============================================================================
# validate_hgt
# ============================================================================

def validate_hgt(
    model: HGTLinkPredictor,
    hetero_data,
    val_compounds: list[int],
    all_compound_to_pos: dict[int, set],
    n_compounds: int,
    n_proteins: int,
    device,
    score_clamp,
    hetero_adj: dict | None = None,
    return_embeddings: bool = False,
    neg_ratio: int = 100,
) -> dict[str, float]:
    """v62: HGT 全图前向验证 — 一次全图前向传播，然后批量解码。

    修复 v31 的 mini-batch 子图采样导致的化合物孤立问题：
    - mini-batch 子图中验证化合物无 CPI 边，嵌入退化为纯特征投影
    - 蛋白质通过 PPI/通路/疾病边有结构信息
    - 化合物与蛋白质嵌入不在同一表示空间，AUC/AUPR 系统性崩塌

    v62 改为全图前向传播（异构全图 < 40MB，6.8GB GPU 绰绰有余）：
    - 使用 hetero_data.edge_index_dict（val CPI 边已移除）做一次全图编码
    - 所有节点（含验证化合物）通过 PPI 网络间接参与消息传递
    - 与 SAGE 验证逻辑一致，化合物与蛋白质嵌入在同一表示空间
    """
    model.eval()
    with torch.no_grad():
        # 1) 全图前向传播
        hetero_data = hetero_data.to(device)
        hgt_out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        comp_emb = hgt_out["compound"]  # (n_compounds, d)
        prot_emb = hgt_out["protein"]   # (n_proteins, d)
        T = model.temperature

        val_compounds_list = list(val_compounds)
        n_val = len(val_compounds_list)
        if n_val == 0:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": 0}

        # 2) 双维度分块批量解码（与 SAGE 验证一致）
        comp_sub = comp_emb[val_compounds_list]  # (n_val, d)
        comp_batch = 32
        prot_batch = 1024
        score_chunks = []
        for c_start in range(0, n_val, comp_batch):
            c_end = min(c_start + comp_batch, n_val)
            sub_comp = comp_sub[c_start:c_end]
            n_c = c_end - c_start
            row_chunks = []
            for p_start in range(0, n_proteins, prot_batch):
                p_end = min(p_start + prot_batch, n_proteins)
                sub_prot = prot_emb[p_start:p_end]
                n_p = p_end - p_start
                comp_exp = sub_comp.repeat_interleave(n_p, dim=0)
                prot_exp = sub_prot.repeat(n_c, 1)
                sub_scores = model.decode(
                    comp_exp, prot_exp, prot_residue_indices=None
                ).reshape(n_c, n_p) / T
                row_chunks.append(sub_scores)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            score_chunks.append(torch.cat(row_chunks, dim=1))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score_matrix = torch.cat(score_chunks, dim=0)  # (n_val, n_proteins)

        # 3) 正样本残基注意力分数（与 SAGE 验证一致）
        use_residue_in_val = (
            hasattr(model, "decoder")
            and model.decoder.__class__.__name__ == "ResidueAwareBilinearDecoder"
        )
        pos_key_to_residue_score: dict[tuple[int, int], float] = {}
        if use_residue_in_val:
            pos_src_list, pos_dst_list = [], []
            for idx, src in enumerate(val_compounds_list):
                pos_set = all_compound_to_pos.get(src, set())
                valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_proteins]
                for p in valid_pos:
                    pos_src_list.append(idx)
                    pos_dst_list.append(p)
            if pos_src_list:
                pos_src_t = torch.tensor(pos_src_list, device=device, dtype=torch.long)
                pos_dst_t = torch.tensor(pos_dst_list, device=device, dtype=torch.long)
                pos_comp_emb = comp_sub[pos_src_t]
                pos_prot_emb = prot_emb[pos_dst_t]
                try:
                    pos_residue_scores = model.decode(
                        pos_comp_emb, pos_prot_emb, prot_residue_indices=pos_dst_t
                    ).reshape(-1) / T
                    pos_residue_scores = torch.clamp(pos_residue_scores, -score_clamp, score_clamp)
                    for s, d, score in zip(pos_src_list, pos_dst_list, pos_residue_scores, strict=False):
                        pos_key_to_residue_score[(int(s), int(d))] = torch.sigmoid(score).item()
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"_validate_hgt: 正样本残基路径 OOM，验证回退到 fast bilinear: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 4) Per-compound 评分与固定 1:neg_ratio 随机负采样
        y_true, y_score = [], []
        all_batch_ranking = []
        n_valid = 0
        for idx, src in enumerate(val_compounds_list):
            pos_set = all_compound_to_pos.get(src, set())
            valid_pos = [p - n_compounds for p in pos_set if n_compounds <= p < n_compounds + n_proteins]
            if not valid_pos:
                continue
            n_valid += 1
            scores = score_matrix[idx]

            # 正样本
            for p in valid_pos:
                y_true.append(1)
                if (idx, p) in pos_key_to_residue_score:
                    y_score.append(pos_key_to_residue_score[(idx, p)])
                else:
                    y_score.append(torch.sigmoid(scores[p]).item())

            # 固定 1:neg_ratio 随机负采样（文献标准做法，与训练时课程负采样解耦）
            neg_mask = torch.ones(n_proteins, device=device, dtype=torch.bool)
            neg_mask[torch.tensor(valid_pos, device=device, dtype=torch.long)] = False
            neg_candidates = torch.where(neg_mask)[0]
            n_neg_target = len(valid_pos) * neg_ratio
            n_neg = min(n_neg_target, len(neg_candidates))
            if n_neg > 0:
                neg_idx = neg_candidates[torch.randperm(len(neg_candidates), device=device)[:n_neg]]
                for ri in neg_idx:
                    y_true.append(0)
                    y_score.append(torch.sigmoid(scores[ri]).item())

            # Per-compound 排名指标
            valid_pos_batch = valid_pos
            batch_ranking = _compute_ranking_metrics(
                scores.unsqueeze(0), [valid_pos_batch]
            )
            all_batch_ranking.append(batch_ranking)

        if len(y_true) < 2 or len(set(y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid}

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)

        # 诊断日志
        try:
            pos_scores = y_score_arr[y_true_arr == 1]
            neg_scores = y_score_arr[y_true_arr == 0]
            if pos_scores.size and neg_scores.size:
                logger.info(
                    f"  [HGT val diag] n_pos={len(pos_scores)} n_neg={len(neg_scores)} "
                    f"pos={pos_scores.mean():.4f}±{pos_scores.std():.4f} "
                    f"neg={neg_scores.mean():.4f}±{neg_scores.std():.4f} "
                    f"gap={(pos_scores.mean() - neg_scores.mean()):.4f}"
                )
        except Exception as e:
            logger.warning(f"  [HGT val diag] 诊断打印异常: {e}")

        result = compute_early_enrichment_metrics(y_true_arr, y_score_arr)
        if all_batch_ranking:
            precision_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("precision@")}
            recall_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("recall@")}
            hit_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("hit@")}
            ndcg_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("ndcg@")}
            ef_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("ef@")}
            n_batches = len(all_batch_ranking)
            for batch_r in all_batch_ranking:
                for k, v in batch_r.items():
                    if k in precision_sums:
                        precision_sums[k] += v
                    elif k in recall_sums:
                        recall_sums[k] += v
                    elif k in hit_sums:
                        hit_sums[k] += v
                    elif k in ndcg_sums:
                        ndcg_sums[k] += v
                    elif k in ef_sums:
                        ef_sums[k] += v
            for k in precision_sums:
                result[k] = precision_sums[k] / n_batches
            for k in recall_sums:
                result[k] = recall_sums[k] / n_batches
            for k in hit_sums:
                result[k] = hit_sums[k] / n_batches
            for k in ndcg_sums:
                result[k] = ndcg_sums[k] / n_batches
            for k in ef_sums:
                result[k] = ef_sums[k] / n_batches
        result["n_valid_compounds"] = n_valid
        return result


# ============================================================================
# validate_hgt_minibatch
# ============================================================================

def validate_hgt_minibatch(
    model: HGTLinkPredictor,
    hetero_data,
    hetero_adj: dict,
    val_compounds: list[int],
    all_compound_to_pos: dict[int, set],
    n_compounds: int,
    n_proteins: int,
    device,
    hgt_val_use_residue_for_pos: bool = True,
    hgt_val_num_neighbors: list[int] | None = None,
    num_neighbors: list[int] = None,
    val_batch_size: int = 256,
    hgt_val_subgraph_cache: dict | None = None,
) -> dict[str, float]:
    """v52-fix2: HGT mini-batch 验证进一步加速。

    对验证化合物分批采样异质子图，在各子图内计算得分后全局聚合。
    - val_batch_size 提升至 256，减少子图采样次数。
    - 验证阶段 num_neighbors 降至 [8, 4]，在保留 2 阶结构信息的同时大幅降低 CPU 采样开销。
    - 每 batch 负样本候选池 target_pool 降至 512，进一步缩小子图规模。
    - 首次采样后缓存子图，后续验证直接复用，避免重复采样。
    """
    if hgt_val_subgraph_cache is None:
        hgt_val_subgraph_cache = {}
    if num_neighbors is None:
        num_neighbors = [8, 4]
    if hgt_val_num_neighbors is None:
        hgt_val_num_neighbors = [64, 32]
    cache_key_shape = (val_batch_size, len(val_compounds), tuple(num_neighbors))
    model.eval()
    with torch.no_grad():
        T = model.temperature
        all_y_true, all_y_score = [], []
        all_batch_ranking = []  # 收集每个 batch 的排名指标
        residue_pos_scores_diag = []  # v61: 残基注意力正样本分数诊断（不参与主指标）
        n_valid_compounds = 0

        for batch_start in range(0, len(val_compounds), val_batch_size):
            batch_seeds = val_compounds[batch_start:batch_start + val_batch_size]
            cache_key = (cache_key_shape, batch_start)
            t_batch_start = time.time()
            logger.info(f"  HGT val batch {batch_start}/{len(val_compounds)} starting")

            cached = hgt_val_subgraph_cache.get(cache_key)
            if cached is not None:
                sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = cached
            else:
                # 化合物冷启动验证中，验证化合物在 val_hetero_adj 中已移除 CPI 边，
                # 必须显式将正样本蛋白与随机负样本蛋白作为 seed_proteins 纳入子图，
                # 否则子图仅含孤立验证化合物，AUC/AUPR 会恒为 0.5。
                candidate_proteins = set()
                for s in batch_seeds:
                    for p_global in all_compound_to_pos.get(s, set()):
                        p_local = p_global - n_compounds
                        if 0 <= p_local < n_proteins:
                            candidate_proteins.add(p_local)
                # 补充随机负样本，保证每个 batch 有足够候选蛋白（上限 512）
                if candidate_proteins:
                    all_prot_set = set(range(n_proteins))
                    neg_pool = list(all_prot_set - candidate_proteins)
                    if neg_pool:
                        rng = random.Random(42 + batch_start)
                        target_pool = 512
                        n_neg_sample = min(target_pool - len(candidate_proteins), len(neg_pool))
                        candidate_proteins.update(rng.sample(neg_pool, n_neg_sample))
                seed_proteins = sorted(candidate_proteins)

                # 化合物冷启动验证中禁止临时添加 seed->candidate CPI 边。
                # 这些边会在 HGT 消息传递中造成信息泄漏（化合物嵌入吸收候选蛋白特征），
                # 导致模型在验证时变相"看到答案"，训练/验证分布不一致，AUC 被严重压低。
                # 保持化合物节点孤立，使其嵌入退化为 encode_compound(x)，才是公平的冷启动评估。
                t_sample_start = time.time()
                sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = sample_hetero_subgraph(
                    batch_seeds, hetero_adj, num_neighbors, seed=42, seed_proteins=seed_proteins,
                    add_seed_cpi_edges=False)
                hgt_val_subgraph_cache[cache_key] = (
                    sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map
                )
                logger.info(
                    f"  HGT val batch {batch_start}/{len(val_compounds)} sampled "
                    f"(compounds={len(comp_sorted)}, proteins={len(prot_sorted)}, pathways={len(path_sorted)}, "
                    f"diseases={len(disease_sorted)}) in {time.time() - t_sample_start:.2f}s"
                )

            if not prot_sorted:
                continue

            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted)].to(device)
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted)].to(device)
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=device)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=device)
            # 疾病节点嵌入
            if disease_sorted:
                disease_global_tensor = torch.tensor(sg._disease_global, device=device).unsqueeze(-1)
                sg["disease"].x = disease_global_tensor
            else:
                sg["disease"].x = torch.zeros(0, 1, device=device)

            sg = sg.to(device)
            hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]
            n_batch_prots = prot_emb.shape[0]
            # 构建 局部索引 -> 全局蛋白索引 映射，供 prot_residue_indices 使用
            prot_inv_map_local = {v: k for k, v in prot_map.items()}

            batch_scores = []  # per-compound scores for ranking
            batch_valid_pos = []  # per-compound valid_pos for ranking

            for _bi, s in enumerate(batch_seeds):
                if s not in comp_map:
                    continue
                comp_local = comp_map[s]

                pos_set = all_compound_to_pos.get(s, set())
                valid_pos = []
                for p_global in pos_set:
                    p_local = p_global - n_compounds
                    if p_local in prot_map:
                        valid_pos.append(prot_map[p_local])
                if not valid_pos:
                    continue
                n_valid_compounds += 1

                # v61-fix: 验证阶段统一使用 fast bilinear 分数作为候选排序与 AUC/AUPR 计算，
                # 避免正样本残基分数与负样本 fast bilinear 分数分布不一致导致 AUC/AUPR 虚高。
                # 若 hgt_val_use_residue_for_pos=True，正样本在 y_score 中仍使用残基分数（与训练一致），
                # 但排名指标与硬负样本选择仍基于统一分数，保证评估可解释性。
                valid_pos_tensor = torch.tensor(valid_pos, device=device, dtype=torch.long)

                # 1) 全候选蛋白 fast bilinear 打分（统一候选分数与排序）
                scores = model.decode(
                    comp_emb[comp_local:comp_local+1].expand(n_batch_prots, -1), prot_emb,
                    prot_residue_indices=None,
                ) / T

                # 2) 正样本单独走残基注意力路径（仅在配置启用时参与 y_score）
                if hgt_val_use_residue_for_pos:
                    pos_global_indices = torch.tensor(
                        [prot_inv_map_local[p] for p in valid_pos],
                        device=device, dtype=torch.long,
                    )
                    pos_scores_residue = model.decode(
                        comp_emb[comp_local:comp_local+1].expand(len(valid_pos), -1),
                        prot_emb[valid_pos_tensor],
                        prot_residue_indices=pos_global_indices,
                    ) / T
                else:
                    pos_scores_residue = None

                batch_scores.append(scores.cpu())  # 收集 per-compound 得分（fast bilinear 统一）
                batch_valid_pos.append(valid_pos)  # 收集 per-compound 正样本索引
                if pos_scores_residue is not None:
                    for idx, ps in zip(valid_pos_tensor, pos_scores_residue, strict=False):
                        all_y_true.append(1)
                        all_y_score.append(torch.sigmoid(ps).item())
                        residue_pos_scores_diag.append(torch.sigmoid(ps).item())
                else:
                    for idx in valid_pos_tensor:
                        all_y_true.append(1)
                        all_y_score.append(torch.sigmoid(scores[idx]).item())

                n_hard = min(5, n_batch_prots - len(valid_pos))
                if n_hard > 0:
                    mask = torch.zeros(n_batch_prots, device=device)
                    for p in valid_pos:
                        mask[p] = -1e9
                    _, hard_indices = (scores + mask).topk(n_hard)
                    for hi in hard_indices:
                        if hi.item() < n_batch_prots:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[hi]).item())

                n_rand = min(5, n_batch_prots - len(valid_pos))
                if n_rand > 0:
                    rand_mask = torch.ones(n_batch_prots, device=device)
                    for p in valid_pos:
                        rand_mask[p] = 0
                    # 排除已选中的硬负样本，避免重复采样导致 AUC/AUPR 虚高
                    if n_hard > 0:
                        for hi in hard_indices:
                            if hi.item() < n_batch_prots:
                                rand_mask[hi] = 0
                    rand_candidates = torch.where(rand_mask > 0)[0]
                    if len(rand_candidates) > 0:
                        n_sample = min(n_rand, len(rand_candidates))
                        rand_idx = rand_candidates[torch.randperm(len(rand_candidates), device=device)[:n_sample]]
                        for ri in rand_idx:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[ri]).item())
                # v45 fix: 增加更多随机负样本以平衡硬负样本偏置
                n_rand_extra = min(100, n_batch_prots - len(valid_pos) - n_hard)
                if n_rand_extra > 0:
                    extra_rand_mask = torch.ones(n_batch_prots, device=device)
                    for p in valid_pos:
                        extra_rand_mask[p] = 0
                    if n_hard > 0:
                        for hi in hard_indices:
                            if hi.item() < n_batch_prots:
                                extra_rand_mask[hi] = 0
                    extra_candidates = torch.where(extra_rand_mask > 0)[0]
                    if len(extra_candidates) > 0:
                        n_sample = min(n_rand_extra, len(extra_candidates))
                        extra_rand_idx = extra_candidates[torch.randperm(len(extra_candidates), device=device)[:n_sample]]
                        for ri in extra_rand_idx:
                            all_y_true.append(0)
                            all_y_score.append(torch.sigmoid(scores[ri]).item())

            # 计算 per-batch 排名指标
            if batch_scores and batch_valid_pos:
                batch_score_matrix = torch.stack(batch_scores, dim=0)
                batch_ranking = _compute_ranking_metrics(batch_score_matrix, batch_valid_pos)
                all_batch_ranking.append(batch_ranking)

            logger.info(
                f"  HGT val batch {batch_start}/{len(val_compounds)} done in {time.time() - t_batch_start:.2f}s"
            )

        if len(all_y_true) < 2 or len(set(all_y_true)) < 2:
            return {"auc": 0.5, "aupr": 0.5, "n_valid_compounds": n_valid_compounds}

        y_true_arr = np.array(all_y_true)
        y_score_arr = np.array(all_y_score)

        # HGT 验证 logit 分布诊断（用于排查 AUC≈0.5 / AUPR 异常低 / AUC=1.0 虚高）
        try:
            pos_scores = y_score_arr[y_true_arr == 1]
            neg_scores = y_score_arr[y_true_arr == 0]
            if pos_scores.size and neg_scores.size:
                diag_msg = (
                    f"  [HGT val diag] n_pos={len(pos_scores)} n_neg={len(neg_scores)} "
                    f"pos={pos_scores.mean():.4f}±{pos_scores.std():.4f} "
                    f"neg={neg_scores.mean():.4f}±{neg_scores.std():.4f} "
                    f"gap={(pos_scores.mean() - neg_scores.mean()):.4f}"
                )
                if residue_pos_scores_diag:
                    r_arr = np.array(residue_pos_scores_diag)
                    diag_msg += f" | residue_pos={r_arr.mean():.4f}±{r_arr.std():.4f}"
                logger.info(diag_msg)
            else:
                logger.info(
                    f"  [HGT val diag] 样本缺失: n_pos={len(pos_scores)} n_neg={len(neg_scores)}"
                )
        except Exception as e:
            logger.warning(f"  [HGT val diag] 诊断打印异常: {e}")

        # v50: 使用标准化指标模块统一计算 AUC/AUPR/EF/ROCE/BEDROC
        # HGT mini-batch 各 batch 蛋白候选集不同，排名指标按 batch 独立计算后平均
        result = compute_early_enrichment_metrics(
            y_true_arr, y_score_arr,
        )
        if all_batch_ranking:
            precision_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("precision@")}
            recall_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("recall@")}
            hit_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("hit@")}
            ndcg_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("ndcg@")}
            ef_sums = {k: 0.0 for k in all_batch_ranking[0] if k.startswith("ef@")}
            n_batches = len(all_batch_ranking)
            for batch_r in all_batch_ranking:
                for k, v in batch_r.items():
                    if k in precision_sums:
                        precision_sums[k] += v
                    elif k in recall_sums:
                        recall_sums[k] += v
                    elif k in hit_sums:
                        hit_sums[k] += v
                    elif k in ndcg_sums:
                        ndcg_sums[k] += v
                    elif k in ef_sums:
                        ef_sums[k] += v
            for sums in (precision_sums, recall_sums, hit_sums, ndcg_sums, ef_sums):
                for k, v in sums.items():
                    result[k] = v / n_batches
        result["n_valid_compounds"] = n_valid_compounds
        # v59: 验证结束后清空 HGT 子图缓存，避免长训练或多次验证导致内存泄漏。
        # 每次验证采样成本较低（~0.5s/epoch），清理缓存换取内存安全。
        hgt_val_subgraph_cache.clear()
        return result


# ============================================================================
# validate_simplehgn
# ============================================================================

def validate_simplehgn(
    model: SimpleHGNLinkPredictor,
    hetero_data,
    val_compounds: list[int],
    all_compound_to_pos: dict[int, set],
    n_compounds: int,
    n_proteins: int,
    device,
    score_clamp,
    hetero_adj: dict | None = None,
    neg_ratio: int = 100,
) -> dict[str, float]:
    """SimpleHGN 全图前向验证，委托给 validate_hgt（v62 全图版本）。

    SimpleHGN 与 HGT 共享相同的 forward(x_dict, edge_index_dict) 接口
    和 decode() 解码逻辑，验证流程完全一致。

    签名与 validate_hgt 一致，适配 train_hgt 的调用方式：
        train_hgt 调用: validate_hgt_fn(model, hd, val_compounds, ..., hetero_adj=val_hetero_adj)
    """
    return validate_hgt(
        model, hetero_data, val_compounds,
        all_compound_to_pos, n_compounds, n_proteins,
        device, score_clamp,
        hetero_adj=hetero_adj,
        neg_ratio=neg_ratio)