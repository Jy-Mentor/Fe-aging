"""Pipeline 预测函数 — HGT / SimpleHGN / TCM 集成预测

所有函数从 phase4_v10_minibatch.py 提取，通过依赖注入接收配置参数，
消除对模块级全局常量的依赖。
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..data.features import build_compound_features
from ..graph.sampling import sample_hetero_subgraph
from ..models import HGTLinkPredictor, SAGELinkPredictor, SimpleHGNLinkPredictor

logger = logging.getLogger(__name__)


def predict_hgt_scores(
    hgt_model: HGTLinkPredictor,
    graphs: dict,
    tcm_feat: torch.Tensor,
    target_local_indices: torch.Tensor,
    n_targets: int,
    device,
    hgt_val_num_neighbors: list[int] | None = None,
) -> torch.Tensor:
    """HGT 目标蛋白评分：优先全图推理，OOM 时降级为 mini-batch。"""
    n_compounds = graphs["n_compounds"]
    n_pathways = graphs.get("n_pathways", 0)
    hetero_data = graphs["hetero_data"]
    hgt_data_dev = hetero_data.to(device)
    hgt_data_dev["pathway"].x = hgt_model.pathway_embed(
        torch.arange(max(n_pathways, 1), device=device))
    hgt_x_dict_full = {k: v.clone() for k, v in hgt_data_dev.x_dict.items()}

    hgt_model.eval()
    with torch.no_grad():
        try:
            hgt_out = hgt_model(hgt_x_dict_full, hgt_data_dev.edge_index_dict)
            hgt_prot_emb = hgt_out["protein"]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            logger.warning("  HGT 全图预测 OOM，降级为 mini-batch 蛋白推理")
            hgt_prot_emb = torch.zeros(
                n_compounds + graphs["n_proteins"], hgt_model.out_dim, device=device)
            hgt_prot_emb[n_compounds:] = predict_hgt_target_proteins_minibatch(
                hgt_model, graphs, target_local_indices, device, hgt_val_num_neighbors)

        hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
        hgt_T = hgt_model.temperature
        n_tcm = hgt_tcm_emb.shape[0]
        hgt_tcm_exp = hgt_tcm_emb.unsqueeze(1).expand(-1, n_targets, -1).reshape(-1, hgt_tcm_emb.shape[-1])
        hgt_prot_exp = hgt_prot_emb[target_local_indices].unsqueeze(0).expand(n_tcm, -1, -1).reshape(-1, hgt_prot_emb.shape[-1])
        # v58: TCM 预测阶段使用 fast bilinear 路径（prot_residue_indices=None），
        # 与 SAGE 预测保持一致。残基注意力路径（max_residue_batch=4）对 7 万+ 对
        # 预测耗时爆炸（约 18K 次 chunk 调用），且 TCM 化合物无残基偏好先验。
        return torch.sigmoid(
            hgt_model.decode(hgt_tcm_exp, hgt_prot_exp, prot_residue_indices=None) / hgt_T
        ).reshape(n_tcm, n_targets)


def predict_hgt_target_proteins_minibatch(
    hgt_model: HGTLinkPredictor,
    graphs: dict,
    target_local_indices: torch.Tensor,
    device,
    hgt_val_num_neighbors: list[int] | None = None,
    num_neighbors: list[int] | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """HGT mini-batch 目标蛋白嵌入推理（OOM 安全）"""
    if num_neighbors is None:
        num_neighbors = hgt_val_num_neighbors if hgt_val_num_neighbors is not None else [64, 32]
    hetero_data = graphs["hetero_data"]
    hetero_adj = graphs["hetero_adj"]
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]

    cpi_adj = hetero_adj[("compound", "interacts", "protein")]
    prot_to_compounds = defaultdict(list)
    for c_global, p_locals in cpi_adj.items():
        for p_local in p_locals:
            if 0 <= p_local < n_proteins:
                prot_to_compounds[p_local].append(c_global)

    full_prot_emb = torch.zeros(n_proteins, hgt_model.out_dim, device=device)
    target_list = target_local_indices.cpu().tolist()
    missing_targets = set()

    hgt_model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(target_list), batch_size):
            batch_targets = target_list[batch_start:batch_start + batch_size]
            seed_compounds = sorted({c for p in batch_targets for c in prot_to_compounds.get(p, [])})
            if not seed_compounds:
                seed_compounds = [0]
            try:
                sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = sample_hetero_subgraph(
                    seed_compounds, hetero_adj, num_neighbors=num_neighbors, seed=42,
                    seed_proteins=batch_targets, add_seed_cpi_edges=False,
                )
            except Exception as e:
                logger.warning(f"  HGT mini-batch 子图采样失败 (targets={batch_start}-{batch_start + len(batch_targets)}): {e}")
                missing_targets.update(batch_targets)
                continue
            if not prot_sorted:
                missing_targets.update(batch_targets)
                continue

            comp_tensor = torch.tensor(comp_sorted)
            if seed_compounds == [0] and 0 not in cpi_adj:
                sg["compound"].x = torch.zeros(len(comp_sorted), hetero_data["compound"].x.shape[1], device=device)
            else:
                sg["compound"].x = hetero_data["compound"].x[comp_tensor].to(device)
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted)].to(device)
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=device)
                path_global_tensor = torch.clamp(path_global_tensor, min=0, max=hgt_model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = hgt_model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, hgt_model.pathway_embed.embedding_dim, device=device)
            if disease_sorted:
                disease_global_tensor = torch.tensor(sg._disease_global, device=device).unsqueeze(-1)
                sg["disease"].x = disease_global_tensor
            else:
                sg["disease"].x = torch.zeros(0, 1, device=device)

            sg = sg.to(device)
            hgt_out = hgt_model(sg.x_dict, sg.edge_index_dict)
            batch_prot_emb = hgt_out["protein"]
            for p_local in batch_targets:
                if p_local in prot_map:
                    full_prot_emb[p_local] = batch_prot_emb[prot_map[p_local]]
                else:
                    missing_targets.add(p_local)

    if missing_targets:
        logger.warning(f"  HGT mini-batch 推理缺失 {len(missing_targets)} 个目标蛋白嵌入，已置零: {sorted(missing_targets)[:10]}")

    return full_prot_emb[target_local_indices]


def predict_simplehgn_scores(
    simplehgn_model: SimpleHGNLinkPredictor,
    graphs: dict,
    tcm_feat: torch.Tensor,
    target_local_indices: torch.Tensor,
    n_targets: int,
    device,
    hgt_val_num_neighbors: list[int] | None = None,
) -> torch.Tensor:
    """SimpleHGN 目标蛋白评分，委托给 predict_hgt_scores（共享 forward/decode 接口）。"""
    return predict_hgt_scores(
        simplehgn_model, graphs, tcm_feat, target_local_indices, n_targets,
        device, hgt_val_num_neighbors,
    )


def predict_tcm(
    sage_model: SAGELinkPredictor,
    hgt_model: HGTLinkPredictor | None,
    graphs: dict,
    tcm_smiles: list[str],
    target_genes: list[str],
    compound_stats: tuple,
    device,
    diversity_penalty: float = 0.3,  # 默认值 0.1→0.3 与全局 DIVERSITY_PENALTY 一致
    mc_samples: int = 0,
    tcm_feat_precomputed: torch.Tensor | None = None,
    tree_predictions: pd.DataFrame | None = None,  # 树模型预测分数
    tree_weight: float = 0.6,  # 树模型集成权重
    sage_w: float = 0.5,  # v56: SAGE 集成权重（动态或等权回退）
    hgt_w: float = 0.5,  # v56: HGT 集成权重
    simplehgn_model: SimpleHGNLinkPredictor | None = None,
    simplehgn_w: float = 0.0,
    hgt_val_num_neighbors: list[int] | None = None,
) -> pd.DataFrame:
    """v40: SAGE + HGT + SimpleHGN + 树模型四方集成预测 — 动态加权 + 多样性约束 + MC Dropout

    Args:
        diversity_penalty: 余弦相似度惩罚系数（0~1，越大越惩罚相似预测）
        mc_samples: MC Dropout 采样次数（0=禁用，推荐30）
        tree_predictions: 树模型预测 DataFrame (MOL_ID, SMILES, gene, score)
        tree_weight: 树模型在最终集成中的权重（0~1）
        sage_w: SAGE 分支集成权重（v56: 基于验证 AUPR 动态计算）
        hgt_w: HGT 分支集成权重
        simplehgn_model: SimpleHGN 模型（可选，为 None 时跳过 SimpleHGN 分支）
        simplehgn_w: SimpleHGN 分支集成权重
    """
    n_iterations = max(1, mc_samples)
    use_mc = mc_samples > 0

    if use_mc:
        sage_model.train()  # 保持 Dropout 开启，无梯度
        if hgt_model is not None:
            hgt_model.train()
        if simplehgn_model is not None:
            simplehgn_model.train()
    else:
        sage_model.eval()
        if hgt_model is not None:
            hgt_model.eval()
        if simplehgn_model is not None:
            simplehgn_model.eval()

    if tcm_feat_precomputed is not None:
        tcm_feat = tcm_feat_precomputed.to(device)
    else:
        tcm_feat_raw, _, _, _ = build_compound_features(tcm_smiles, stats=compound_stats)
        feat_dim = graphs["feat_dim"]
        if tcm_feat_raw.shape[1] < feat_dim:
            tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
        tcm_feat = torch.from_numpy(tcm_feat_raw).to(device)

    x_dev = graphs["x"].to(device)
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    gene_to_idx = graphs["gene_to_idx"]
    homo_edge_index = graphs["homo_edge_index"]

    # v56: 动态集成权重 — 基于验证 AUPR 加权
    use_simplehgn = simplehgn_model is not None and simplehgn_w > 0
    if use_simplehgn:
        logger.info(f"  集成权重: SAGE={sage_w:.3f}, HGT={hgt_w:.3f}, SimpleHGN={simplehgn_w:.3f}（基于验证 AUPR 动态加权）")
    elif sage_w == 0.5 and hgt_w == 0.5:
        logger.info(f"  集成权重: SAGE={sage_w:.3f}, HGT={hgt_w:.3f}（等权回退）")
    else:
        logger.info(f"  集成权重: SAGE={sage_w:.3f}, HGT={hgt_w:.3f}（基于验证 AUPR 动态加权）")

    # 预构建基因→蛋白局部索引映射
    gene_index_map = []  # [(gene, local_p_idx), ...]
    for gene in target_genes:
        if gene in gene_to_idx:
            p_idx = gene_to_idx[gene]
            local_p_idx = p_idx - n_compounds
            if 0 <= local_p_idx < n_proteins:
                gene_index_map.append((gene, local_p_idx))
            else:
                gene_index_map.append((gene, -1))
        else:
            gene_index_map.append((gene, -1))

    all_sage_scores_mc = []  # (n_iter, n_tcm, n_genes)
    all_hgt_scores_mc = []
    all_simplehgn_scores_mc = []

    for _it in range(n_iterations):
        try:
            with torch.no_grad():
                # SAGE: 原生归纳式推理
                # 分别处理 — 全图编码蛋白嵌入 + encode_compound 编码 TCM 化合物
                edge_index = homo_edge_index.to(device)
                node_emb = sage_model(x_dev, edge_index)  # 全图（原化合物+蛋白）
                sage_prot_emb = node_emb[n_compounds:]
                sage_tcm_emb = sage_model.encode_compound(tcm_feat)  # TCM 化合物（无CPI边，仅投影+卷积）
                sage_T = sage_model.temperature

                # v45: 仅对目标基因对应的蛋白打分，避免 residue_bilinear 在全蛋白集上耗时爆炸。
                valid_gene_indices = [(j, lp) for j, (_, lp) in enumerate(gene_index_map) if lp >= 0]
                target_local_indices = torch.tensor(
                    [lp for _, lp in valid_gene_indices], dtype=torch.long, device=device)

                # SAGE 向量化评分: (n_tcm, n_target_prots)
                n_tcm_sage = sage_tcm_emb.shape[0]
                n_targets = target_local_indices.shape[0]
                sage_tcm_exp = sage_tcm_emb.unsqueeze(1).expand(-1, n_targets, -1).reshape(-1, sage_tcm_emb.shape[-1])
                sage_prot_exp = sage_prot_emb[target_local_indices].unsqueeze(0).expand(n_tcm_sage, -1, -1).reshape(-1, sage_prot_emb.shape[-1])
                # v55-fix: TCM 预测阶段使用 fast bilinear 路径，避免对 7 万+ 化合物-蛋白对
                # 逐对计算残基注意力导致预测耗时爆炸。训练/验证仍通过 validate_sage 走残基路径。
                sage_target_scores = torch.sigmoid(
                    sage_model.decode(sage_tcm_exp, sage_prot_exp) / sage_T
                ).reshape(n_tcm_sage, n_targets)

                if hgt_model is not None:
                    hgt_target_scores = predict_hgt_scores(
                        hgt_model, graphs, tcm_feat, target_local_indices, n_targets,
                        device, hgt_val_num_neighbors,
                    )
                else:
                    hgt_target_scores = None

                if use_simplehgn:
                    simplehgn_target_scores = predict_simplehgn_scores(
                        simplehgn_model, graphs, tcm_feat, target_local_indices, n_targets,
                        device, hgt_val_num_neighbors,
                    )
                else:
                    simplehgn_target_scores = None

                # 提取目标基因的分数
                iter_sage = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=device)
                iter_hgt = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=device)
                iter_simplehgn = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=device)
                for score_col, (target_col, _local_p_idx) in enumerate(valid_gene_indices):
                    iter_sage[:, target_col] = sage_target_scores[:, score_col]
                    if hgt_target_scores is not None:
                        iter_hgt[:, target_col] = hgt_target_scores[:, score_col]
                    if simplehgn_target_scores is not None:
                        iter_simplehgn[:, target_col] = simplehgn_target_scores[:, score_col]
                all_sage_scores_mc.append(iter_sage.cpu())
                all_hgt_scores_mc.append(iter_hgt.cpu())
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                raise
            logger.warning(f"  predict_tcm MC 迭代 {_it + 1}/{n_iterations} OOM: {e}")
            torch.cuda.empty_cache()
            if _it == 0:
                logger.error("  predict_tcm 首次迭代即 OOM，无法继续预测，请减小 mc_samples 或降级到 CPU")
                raise
            logger.warning(f"  predict_tcm OOM 降级: 跳过迭代 {_it + 1}，使用前 {len(all_sage_scores_mc)} 次结果")
            break

    # ---- 聚合 MC 迭代结果 ----
    if use_mc:
        sage_stack = torch.stack(all_sage_scores_mc, dim=0)  # (n_iter, n_tcm, n_genes)
        hgt_stack = torch.stack(all_hgt_scores_mc, dim=0)

        sage_mean = sage_stack.mean(dim=0)  # (n_tcm, n_genes)
        sage_std = sage_stack.std(dim=0)
        hgt_mean = hgt_stack.mean(dim=0)
        hgt_std = hgt_stack.std(dim=0)

        if use_simplehgn and all_simplehgn_scores_mc:
            simplehgn_stack = torch.stack(all_simplehgn_scores_mc, dim=0)
            simplehgn_mean = simplehgn_stack.mean(dim=0)
            simplehgn_std = simplehgn_stack.std(dim=0)
        else:
            simplehgn_mean = torch.full_like(sage_mean, 0.5)
            simplehgn_std = None

        logger.info(f"  MC Dropout ({mc_samples} 次): SAGE 均值范围 [{sage_mean.min():.4f}, {sage_mean.max():.4f}], "
                    f"平均不确定度 {sage_std.mean():.4f}")
    else:
        sage_mean = all_sage_scores_mc[0]
        hgt_mean = all_hgt_scores_mc[0]
        if use_simplehgn and all_simplehgn_scores_mc:
            simplehgn_mean = all_simplehgn_scores_mc[0]
        else:
            simplehgn_mean = torch.full_like(sage_mean, 0.5)
        sage_std = hgt_std = simplehgn_std = None

    # 多样性约束 — 在分支均值上应用
    # 原始公式 diversity_factor = 1 - penalty * (1 - delta) 会惩罚一致性、奖励分歧，与集成学习直觉相反。
    # 修正为：模型越一致（delta→0），越信任集成分数；越分歧（delta→1），越向 0.5 收缩表示不确定。
    # 三分支时取平均 pairwise delta
    delta_sh = torch.abs(sage_mean - simplehgn_mean)  # SAGE vs SimpleHGN
    delta_hh = torch.abs(hgt_mean - simplehgn_mean)   # HGT vs SimpleHGN
    delta = (delta_sh + delta_hh) / 2.0  # SimpleHGN 与 SAGE/HGT 的平均分歧
    # 使用函数参数 diversity_penalty 而非全局常量 DIVERSITY_PENALTY
    # 原代码直接引用全局常量，导致函数参数完全无效
    diversity_factor = 1.0 - diversity_penalty * delta
    weighted_scores = sage_w * sage_mean + hgt_w * hgt_mean + simplehgn_w * simplehgn_mean
    final_scores = weighted_scores * diversity_factor + 0.5 * (1.0 - diversity_factor)

    # 树模型集成 — 将树模型预测与 GNN 集成分数加权融合
    tree_scores_tensor = None
    if tree_predictions is not None and len(tree_predictions) > 0:
        # 构建 (SMILES, gene) → score 的查找表
        tree_lookup = {}
        for _, row in tree_predictions.iterrows():
            raw_score = float(row["score"])
            # v44: 若树模型分数超出 [0,1]，视为 logit 进行 sigmoid 校准并告警。
            if raw_score < 0.0 or raw_score > 1.0:
                logger.warning(
                    f"树模型分数超出概率范围 (score={raw_score:.4f})，自动 sigmoid 校准"
                )
                raw_score = float(1.0 / (1.0 + math.exp(-raw_score)))
                raw_score = max(0.0, min(1.0, raw_score))
            tree_lookup[(str(row["SMILES"]), str(row["gene"]))] = raw_score
        # 构建与 final_scores 同形状的张量
        tree_scores_tensor = torch.full_like(final_scores, 0.5)
        tree_matched = 0
        for i, smi in enumerate(tcm_smiles):
            for j, (gene, _) in enumerate(gene_index_map):
                key = (smi, gene)
                if key in tree_lookup:
                    tree_scores_tensor[i, j] = tree_lookup[key]
                    tree_matched += 1
        logger.info(f"  树模型集成: 匹配 {tree_matched}/{final_scores.numel()} 对, "
                    f"权重 tree={tree_weight:.2f} GNN={1-tree_weight:.2f}")
        # 三方融合: GNN 集成分数 × (1-tree_weight) + 树模型分数 × tree_weight
        final_scores = (1 - tree_weight) * final_scores + tree_weight * tree_scores_tensor

    # 按基因维度计算余弦相似度并取均值，避免全局展平丢失基因特异性信息
    # sage_mean/hgt_mean: (n_tcm, n_genes)
    per_gene_cos = []
    for g in range(sage_mean.shape[1]):
        sg = sage_mean[:, g]
        hg = hgt_mean[:, g]
        per_gene_cos.append(F.cosine_similarity(sg.unsqueeze(0), hg.unsqueeze(0)).item())
    cos_sim = float(np.mean(per_gene_cos))
    if use_simplehgn:
        per_gene_cos_sh = []
        for g in range(sage_mean.shape[1]):
            sg = sage_mean[:, g]
            shg = simplehgn_mean[:, g]
            per_gene_cos_sh.append(F.cosine_similarity(sg.unsqueeze(0), shg.unsqueeze(0)).item())
        cos_sim_sh = float(np.mean(per_gene_cos_sh))
        per_gene_cos_hh = []
        for g in range(hgt_mean.shape[1]):
            hg = hgt_mean[:, g]
            shg = simplehgn_mean[:, g]
            per_gene_cos_hh.append(F.cosine_similarity(hg.unsqueeze(0), shg.unsqueeze(0)).item())
        cos_sim_hh = float(np.mean(per_gene_cos_hh))
        logger.info(f"  分支余弦相似度: SAGE-HGT={cos_sim:.4f}, SAGE-SimpleHGN={cos_sim_sh:.4f}, HGT-SimpleHGN={cos_sim_hh:.4f} (越低越好，表示分支互补性强)")
    else:
        logger.info(f"  分支余弦相似度: {cos_sim:.4f} (越低越好，表示分支互补性强)")

    # 构建结果 DataFrame
    results = []
    for i, smi in enumerate(tcm_smiles):
        row = {"MOL_ID": f"TCM_{i}", "molecule_name": "", "SMILES": smi}
        for j, (gene, _) in enumerate(gene_index_map):
            row[gene] = final_scores[i, j].item()
            if use_mc and sage_std is not None and hgt_std is not None:
                # MC 不确定性：取分支标准差的均值作为该对的不确定度
                if use_simplehgn and simplehgn_std is not None:
                    row[f"{gene}_uncertainty"] = ((sage_std[i, j] + hgt_std[i, j] + simplehgn_std[i, j]) / 3).item()
                else:
                    row[f"{gene}_uncertainty"] = ((sage_std[i, j] + hgt_std[i, j]) / 2).item()
        if use_mc and sage_std is not None and hgt_std is not None:
            # 聚合不确定性指标
            if use_simplehgn and simplehgn_std is not None:
                pair_uncertainties = (sage_std[i] + hgt_std[i] + simplehgn_std[i]) / 3
            else:
                pair_uncertainties = (sage_std[i] + hgt_std[i]) / 2
            row["mean_uncertainty"] = pair_uncertainties.mean().item()
            row["max_uncertainty"] = pair_uncertainties.max().item()
        results.append(row)

    return pd.DataFrame(results)