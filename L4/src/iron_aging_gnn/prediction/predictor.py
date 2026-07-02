"""预测模块：SAGE + HGT 集成预测 — 动态权重 + 多样性约束 + MC Dropout

v25: 从 phase4_v10_minibatch.py 提取 predict_tcm 函数
v26-fix: HGT MC Dropout 预准备 — 将 hetero_data.to(DEVICE) 和 pathway_embed 移出循环

参考:
  - Zhou et al. (2021) "Diver"
  - Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..models import HGTLinkPredictor, SAGELinkPredictor

logger = logging.getLogger(__name__)

# ---- v25 模块级常量（与主脚本 phase4_v10_minibatch.py 保持一致） ----
DEFAULT_AUPR = 0.5
DIVERSITY_PENALTY = 0.1


def predict_tcm(
    sage_model: SAGELinkPredictor,
    hgt_model: HGTLinkPredictor | None,
    graphs: dict,
    tcm_smiles: list[str],
    target_genes: list[str],
    compound_stats: tuple,
    device: torch.device,
    sage_prot_aupr: float = 0.5,
    hgt_prot_aupr: float = 0.5,
    diversity_penalty: float = 0.1,
    mc_samples: int = 0,
    tcm_feat_precomputed: torch.Tensor | None = None,
    # 外部注入的化合物特征构建函数
    build_compound_features_fn=None,
) -> pd.DataFrame:
    """v17: SAGE + HGT 集成预测 — 动态权重 + 多样性约束 + MC Dropout

    v17 改进:
      - 基于蛋白冷启动 AUPR 动态调整 SAGE/HGT 权重
      - 余弦相似度多样性惩罚：鼓励两个分支利用不同信号
      - MC Dropout 不确定性估计：mc_samples>0 时保持 Dropout 开启，
        重复 mc_samples 次前向，输出均值 + 标准差
      - 参考: Zhou et al. (2021) "Diver";
              Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"

    Args:
        sage_model: SAGE 链接预测模型
        hgt_model: HGT 链接预测模型（可为 None）
        graphs: 图数据字典
        tcm_smiles: TCM 化合物 SMILES 列表
        target_genes: 目标基因列表
        compound_stats: 化合物特征统计信息（均值/标准差）
        device: 计算设备
        sage_prot_aupr: SAGE 蛋白冷启动 AUPR
        hgt_prot_aupr: HGT 蛋白冷启动 AUPR
        diversity_penalty: 余弦相似度惩罚系数（0~1，越大越惩罚相似预测）
        mc_samples: MC Dropout 采样次数（0=禁用，推荐30）
        tcm_feat_precomputed: 预计算的 TCM 化合物特征（可选）
        build_compound_features_fn: 化合物特征构建函数（从主脚本注入）

    Returns:
        pd.DataFrame: 预测结果表
    """
    if build_compound_features_fn is None:
        raise ValueError("predict_tcm 需要注入 build_compound_features_fn 参数。")

    n_iterations = max(1, mc_samples)
    use_mc = mc_samples > 0

    if use_mc:
        sage_model.train()
        if hgt_model is not None:
            hgt_model.train()
    else:
        sage_model.eval()
        if hgt_model is not None:
            hgt_model.eval()

    if tcm_feat_precomputed is not None:
        tcm_feat = tcm_feat_precomputed.to(device)
    else:
        tcm_feat_raw, _, _, _ = build_compound_features_fn(tcm_smiles, stats=compound_stats)
        feat_dim = graphs["feat_dim"]
        if tcm_feat_raw.shape[1] < feat_dim:
            tcm_feat_raw = np.pad(tcm_feat_raw, ((0, 0), (0, feat_dim - tcm_feat_raw.shape[1])), mode="constant")
        tcm_feat = torch.from_numpy(tcm_feat_raw).to(device)

    x_dev = graphs["x"].to(device)
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    gene_to_idx = graphs["gene_to_idx"]
    homo_edge_index = graphs["homo_edge_index"]

    # v17: 动态集成权重 — 基于蛋白冷启动 AUPR
    total_aupr = sage_prot_aupr + hgt_prot_aupr
    if total_aupr > 0:
        sage_w = sage_prot_aupr / total_aupr
        hgt_w = hgt_prot_aupr / total_aupr
    else:
        sage_w = hgt_w = 0.5
    logger.info(f"  集成权重: SAGE={sage_w:.3f} (prot_aupr={sage_prot_aupr:.3f}), "
                f"HGT={hgt_w:.3f} (prot_aupr={hgt_prot_aupr:.3f})")

    # 预构建基因→蛋白局部索引映射
    gene_index_map = []
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

    all_sage_scores_mc = []
    all_hgt_scores_mc = []

    # v26-fix: HGT MC Dropout 预准备
    hgt_data_dev = None
    hgt_x_dict_full = None
    if hgt_model is not None:
        hgt_data_dev = graphs["hetero_data"].to(device)
        n_pathways = graphs["n_pathways"]
        hgt_data_dev["pathway"].x = hgt_model.pathway_embed(
            torch.arange(max(n_pathways, 1), device=device))
        hgt_x_dict_full = {k: v.clone() for k, v in hgt_data_dev.x_dict.items()}

    for _it in range(n_iterations):
        with torch.no_grad():
            edge_index = homo_edge_index.to(device)
            node_emb = sage_model(x_dev, edge_index)
            sage_prot_emb = node_emb[n_compounds:]
            sage_tcm_emb = sage_model.encode_compound(tcm_feat)
            sage_T = sage_model.temperature

            n_tcm_sage = sage_tcm_emb.shape[0]
            n_prots_all = sage_prot_emb.shape[0]
            sage_tcm_exp = sage_tcm_emb.unsqueeze(1).expand(-1, n_prots_all, -1).reshape(-1, sage_tcm_emb.shape[-1])
            sage_prot_exp = sage_prot_emb.unsqueeze(0).expand(n_tcm_sage, -1, -1).reshape(-1, sage_prot_emb.shape[-1])
            sage_all_scores = torch.sigmoid(
                sage_model.decode(sage_tcm_exp, sage_prot_exp) / sage_T
            ).reshape(n_tcm_sage, n_prots_all)

            if hgt_model is not None:
                hgt_out = hgt_model(hgt_x_dict_full, hgt_data_dev.edge_index_dict)
                hgt_prot_emb = hgt_out["protein"]
                hgt_tcm_emb = hgt_model.encode_compound(tcm_feat)
                hgt_T = hgt_model.temperature

                n_tcm = hgt_tcm_emb.shape[0]
                n_prots_all = hgt_prot_emb.shape[0]
                hgt_tcm_exp = hgt_tcm_emb.unsqueeze(1).expand(-1, n_prots_all, -1).reshape(-1, hgt_tcm_emb.shape[-1])
                hgt_prot_exp = hgt_prot_emb.unsqueeze(0).expand(n_tcm, -1, -1).reshape(-1, hgt_prot_emb.shape[-1])
                hgt_all_scores = torch.sigmoid(
                    hgt_model.decode(hgt_tcm_exp, hgt_prot_exp) / hgt_T
                ).reshape(n_tcm, n_prots_all)
            else:
                hgt_all_scores = None

            iter_sage = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=device)
            iter_hgt = torch.full((len(tcm_smiles), len(target_genes)), 0.5, device=device)
            for j, local_p_idx in [(j, lp) for j, (_, lp) in enumerate(gene_index_map) if lp >= 0]:
                iter_sage[:, j] = sage_all_scores[:, local_p_idx]
                if hgt_all_scores is not None:
                    iter_hgt[:, j] = hgt_all_scores[:, local_p_idx]
            all_sage_scores_mc.append(iter_sage.cpu())
            all_hgt_scores_mc.append(iter_hgt.cpu())

    # ---- 聚合 MC 迭代结果 ----
    if use_mc:
        sage_stack = torch.stack(all_sage_scores_mc, dim=0)
        hgt_stack = torch.stack(all_hgt_scores_mc, dim=0)

        sage_mean = sage_stack.mean(dim=0)
        sage_std = sage_stack.std(dim=0)
        hgt_mean = hgt_stack.mean(dim=0)
        hgt_std = hgt_stack.std(dim=0)

        logger.info(f"  MC Dropout ({mc_samples} 次): SAGE 均值范围 [{sage_mean.min():.4f}, {sage_mean.max():.4f}], "
                    f"平均不确定度 {sage_std.mean():.4f}")
    else:
        sage_mean = all_sage_scores_mc[0]
        hgt_mean = all_hgt_scores_mc[0]
        sage_std = hgt_std = None

    # v19-fix: 多样性约束
    delta = torch.abs(sage_mean - hgt_mean)
    diversity_factor = 1.0 - DIVERSITY_PENALTY * delta
    weighted_scores = sage_w * sage_mean + hgt_w * hgt_mean
    final_scores = weighted_scores * diversity_factor + 0.5 * (1.0 - diversity_factor)

    # v26-fix: 按基因维度计算余弦相似度
    per_gene_cos = []
    for g in range(sage_mean.shape[1]):
        sg = sage_mean[:, g]
        hg = hgt_mean[:, g]
        per_gene_cos.append(F.cosine_similarity(sg.unsqueeze(0), hg.unsqueeze(0)).item())
    cos_sim = float(np.mean(per_gene_cos))
    logger.info(f"  分支余弦相似度: {cos_sim:.4f} (越低越好，表示分支互补性强)")

    # 构建结果 DataFrame
    results = []
    for i, smi in enumerate(tcm_smiles):
        row = {"MOL_ID": f"TCM_{i}", "molecule_name": "", "SMILES": smi}
        for j, (gene, _) in enumerate(gene_index_map):
            row[gene] = final_scores[i, j].item()
            if use_mc:
                row[f"{gene}_uncertainty"] = ((sage_std[i, j] + hgt_std[i, j]) / 2).item()
        if use_mc:
            pair_uncertainties = (sage_std[i] + hgt_std[i]) / 2
            row["mean_uncertainty"] = pair_uncertainties.mean().item()
            row["max_uncertainty"] = pair_uncertainties.max().item()
        results.append(row)

    return pd.DataFrame(results)


__all__ = [
    "predict_tcm",
]