#!/usr/bin/env python3
"""
互补性分析 (Complementarity Analysis)
基于 v21 训练结果，量化 SAGE 与 HGT 双分支的互补性：
1. 错误重叠矩阵 — 两分支在验证集上的错误 Venn 图
2. CKA 相似度 — 蛋白嵌入相似度随训练 epoch 变化
3. 按蛋白度分层性能 — 高低连接蛋白上的分支差异
4. 与单分支消融的嵌入相似度对比
"""
from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

# ── 路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_SCRIPTS = PROJECT_ROOT / "L4" / "scripts"
L4_SRC = PROJECT_ROOT / "L4" / "src"
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(L4_SRC))
sys.path.insert(0, str(L4_SCRIPTS))
import phase4_v10_minibatch as p4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "complementarity.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("complementarity")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cka_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered Kernel Alignment (CKA) between two embedding matrices."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    # Linear kernel: HSIC = ||Y^T X||_F^2
    hsic = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    var_x = np.linalg.norm(X.T @ X, ord="fro")
    var_y = np.linalg.norm(Y.T @ Y, ord="fro")
    if var_x < 1e-10 or var_y < 1e-10:
        return 0.0
    return hsic / (var_x * var_y)


def get_protein_degrees(homo_adj: dict[int, list[int]], n_compounds: int, all_proteins: list[int]) -> dict[int, int]:
    """计算每个蛋白的 PPI 度"""
    degrees = {}
    for p in all_proteins:
        global_p = p + n_compounds
        degrees[p] = len(homo_adj.get(global_p, []))
    return degrees


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("互补性分析 — SAGE vs HGT 双分支")
    logger.info("=" * 60)

    # ── 加载数据 ──
    logger.info(">>> 加载数据")
    cpi_df = p4.load_cpi_data()
    ppi_df = p4.load_ppi_network()
    gene_to_pathways = p4.load_kegg_pathways()
    prot_feat, gene_to_seq = p4.load_protein_features()

    warm_targets = sorted(set(cpi_df["gene"].unique()) & set(p4.ALL_FERRORAGING_GENES))
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()
    logger.info(f"  温靶标: {len(warm_targets)} 个")

    logger.info(">>> 构建图")
    graphs = p4.build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)

    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    homo_adj = graphs["homo_adj"]

    # ── 拆分 ──
    all_compounds = sorted(graphs["smi_to_idx"].values())
    all_proteins = sorted({
        graphs["gene_to_idx"][g] - n_compounds
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= n_compounds
    })
    import random
    random.seed(42)
    random.shuffle(all_compounds)
    random.shuffle(all_proteins)

    n_train_comp = int(len(all_compounds) * 0.85)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        gene = row["gene"]
        if gene in graphs["gene_to_idx"]:
            cpi_proteins.add(graphs["gene_to_idx"][gene] - n_compounds)
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

    n_val_cpi = max(1, int(len(cpi_proteins) * 0.20))
    n_train_cpi = len(cpi_proteins) - n_val_cpi
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * 0.20))
    n_train_non_cpi = len(non_cpi_proteins) - n_val_non_cpi

    cpi_proteins_list = list(cpi_proteins)
    random.shuffle(cpi_proteins_list)
    random.shuffle(non_cpi_proteins)

    train_proteins = set(cpi_proteins_list[:n_train_cpi]) | set(non_cpi_proteins[:n_train_non_cpi])
    val_proteins = set(cpi_proteins_list[n_train_cpi:]) | set(non_cpi_proteins[n_train_non_cpi:])

    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    val_comp_set = set(val_compounds)
    logger.info(f"  拆分: train_comp={len(train_compounds)}, val_comp={len(val_compounds)}, "
                f"train_prot={len(train_proteins)}, val_prot={len(val_proteins)}")

    # ── 构建验证安全图 ──
    homo_edge_index_val = p4._build_val_comp_cold_homo_edge_index(graphs["homo_edge_index"], val_comp_set)
    hetero_data_val = p4._build_val_comp_cold_hetero_data(graphs["hetero_data"], val_comp_set)
    homo_edge_index_prot_cold = p4._build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], n_compounds, val_comp_set, val_proteins)
    hetero_data_prot_cold = p4._build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)
    hetero_adj = p4._build_val_safe_hetero_adj(
        graphs["hetero_adj"], n_compounds, val_comp_set, val_proteins)

    # ── 初始化模型并加载 v21 权重 ──
    logger.info(">>> 初始化模型并加载 v21 权重")
    sage_model = p4.SAGELinkPredictor(
        comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],
        n_compounds=n_compounds, hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
        n_pathways=graphs["n_pathways"])
    hgt_node_feat_dims = {
        "compound": graphs["feat_dim"], "protein": graphs["prot_esm_dim"],
        "pathway": 1, "pathway_count": graphs["n_pathways"],
    }
    hgt_model = p4.HGTLinkPredictor(
        hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
        dropout=0.5, metadata=graphs["hetero_data"].metadata(),
        compound_feat_dim=graphs["feat_dim"], node_feat_dims=hgt_node_feat_dims)

    # 加载 v23 最佳检查点
    sage_ckpt = L4_RESULTS / "sage_best_v23.pt"
    hgt_ckpt = L4_RESULTS / "hgt_best_v23.pt"
    if sage_ckpt.exists():
        sage_model.load_state_dict(torch.load(sage_ckpt, map_location=DEVICE, weights_only=True))
        logger.info("  SAGE v23 权重已加载")
    else:
        logger.warning("  SAGE v23 权重不存在，使用随机初始化")
    if hgt_ckpt.exists():
        hgt_model.load_state_dict(torch.load(hgt_ckpt, map_location=DEVICE, weights_only=True))
        logger.info("  HGT v23 权重已加载")
    else:
        logger.warning("  HGT v23 权重不存在，使用随机初始化")

    sage_model = sage_model.to(DEVICE).eval()
    hgt_model = hgt_model.to(DEVICE).eval()

    x = graphs["x"].to(DEVICE)
    homo_ei_val = homo_edge_index_val.to(DEVICE)
    homo_ei_prot_cold = homo_edge_index_prot_cold.to(DEVICE)

    # ============================================================
    # 分析 1: 错误重叠矩阵
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info(">>> 分析 1: 错误重叠矩阵")
    logger.info("=" * 60)

    # 收集两分支在验证集上的预测（化合物冷启动）
    sage_preds = {}
    hgt_preds = {}
    y_true = {}

    threshold = 0.5

    with torch.no_grad():
        # SAGE
        sage_emb = sage_model.encoder(x, homo_ei_val)[:n_compounds]
        sage_prot_emb = sage_model.encoder(x, homo_ei_val)[n_compounds:]
        for c in val_compounds:
            c_emb = sage_emb[c].unsqueeze(0)
            pos_set = compound_to_pos.get(c, set())
            for p in val_proteins:
                p_global = p + n_compounds
                p_local = p_global - n_compounds
                if p_local < 0 or p_local >= sage_prot_emb.shape[0]:
                    continue
                p_emb = sage_prot_emb[p_local].unsqueeze(0)
                score = torch.sigmoid(sage_model.decoder(c_emb, p_emb)).item()
                key = (c, p)
                sage_preds[key] = score
                y_true[key] = 1.0 if p_global in compound_to_pos.get(c, set()) else 0.0

        torch.cuda.empty_cache()

        # HGT
        hgt_comp_emb = None
        hgt_prot_emb = None
        hgt_hetero = hetero_data_val.to(DEVICE)
        hgt_out = hgt_model(hgt_hetero.x_dict, hgt_hetero.edge_index_dict)
        hgt_comp_emb = hgt_out["compound"][:n_compounds]
        hgt_prot_emb = hgt_out["protein"]

        for c in val_compounds:
            if c >= hgt_comp_emb.shape[0]:
                continue
            c_emb = hgt_comp_emb[c].unsqueeze(0)
            pos_set = compound_to_pos.get(c, set())
            for p in val_proteins:
                if p >= hgt_prot_emb.shape[0]:
                    continue
                p_emb = hgt_prot_emb[p].unsqueeze(0)
                score = torch.sigmoid(hgt_model.decode(c_emb, p_emb)).item()
                key = (c, p)
                hgt_preds[key] = score

    # 计算错误重叠
    common_keys = set(sage_preds.keys()) & set(hgt_preds.keys())
    sage_errors = set()
    hgt_errors = set()
    sage_correct = set()
    hgt_correct = set()

    for key in common_keys:
        true = y_true[key]
        sage_pred = 1.0 if sage_preds[key] >= threshold else 0.0
        hgt_pred = 1.0 if hgt_preds[key] >= threshold else 0.0

        if sage_pred == true:
            sage_correct.add(key)
        else:
            sage_errors.add(key)

        if hgt_pred == true:
            hgt_correct.add(key)
        else:
            hgt_errors.add(key)

    both_correct = sage_correct & hgt_correct
    both_error = sage_errors & hgt_errors
    sage_only_error = sage_errors - hgt_errors
    hgt_only_error = hgt_errors - sage_errors

    total = len(common_keys)
    logger.info(f"  总评估对: {total}")
    logger.info(f"  两分支均正确: {len(both_correct)} ({len(both_correct)/total*100:.1f}%)")
    logger.info(f"  两分支均错误: {len(both_error)} ({len(both_error)/total*100:.1f}%)")
    logger.info(f"  仅 SAGE 错误: {len(sage_only_error)} ({len(sage_only_error)/total*100:.1f}%)")
    logger.info(f"  仅 HGT 错误: {len(hgt_only_error)} ({len(hgt_only_error)/total*100:.1f}%)")
    logger.info(f"  互补比 (非重叠错误/总错误): "
                f"{(len(sage_only_error)+len(hgt_only_error))/max(len(sage_errors|hgt_errors),1)*100:.1f}%")

    # ── 按蛋白度分层 ──
    logger.info("\n" + "=" * 60)
    logger.info(">>> 分析 2: 按蛋白度分层性能")
    logger.info("=" * 60)

    protein_degrees = get_protein_degrees(homo_adj, n_compounds, all_proteins)
    low_deg = {p for p in val_proteins if protein_degrees.get(p, 0) <= 5}
    mid_deg = {p for p in val_proteins if 6 <= protein_degrees.get(p, 0) <= 20}
    high_deg = {p for p in val_proteins if protein_degrees.get(p, 0) > 20}

    logger.info(f"  低度蛋白 (0-5): {len(low_deg)}")
    logger.info(f"  中度蛋白 (6-20): {len(mid_deg)}")
    logger.info(f"  高度蛋白 (>20): {len(high_deg)}")

    for tag, prot_set in [("低度", low_deg), ("中度", mid_deg), ("高度", high_deg)]:
        if not prot_set:
            continue
        sage_y_true, sage_y_pred = [], []
        hgt_y_true, hgt_y_pred = [], []
        for key in common_keys:
            c, p = key
            if p in prot_set:
                sage_y_pred.append(sage_preds[key])
                hgt_y_pred.append(hgt_preds[key])
                sage_y_true.append(y_true[key])
                hgt_y_true.append(y_true[key])

        if len(set(sage_y_true)) > 1:
            sage_auc = roc_auc_score(sage_y_true, sage_y_pred)
            sage_aupr = average_precision_score(sage_y_true, sage_y_pred)
            hgt_auc = roc_auc_score(hgt_y_true, hgt_y_pred)
            hgt_aupr = average_precision_score(hgt_y_true, hgt_y_pred)
            logger.info(f"  [{tag}] SAGE: auc={sage_auc:.4f}, aupr={sage_aupr:.4f}")
            logger.info(f"  [{tag}] HGT:  auc={hgt_auc:.4f}, aupr={hgt_aupr:.4f}")
        else:
            logger.info(f"  [{tag}] 正/负样本单一，跳过评估")

    # ── CKA 相似度 ──
    logger.info("\n" + "=" * 60)
    logger.info(">>> 分析 3: CKA 相似度 (SAGE vs HGT 蛋白嵌入)")
    logger.info("=" * 60)

    with torch.no_grad():
        sage_emb_all = sage_model.encoder(x, homo_ei_val).cpu().numpy()
        hgt_out_all = hgt_model(hgt_hetero.x_dict, hgt_hetero.edge_index_dict)
        hgt_emb_all = hgt_out_all["protein"].cpu().numpy()

    # 只取验证集中的蛋白
    val_prot_list = sorted(val_proteins)
    sage_val_emb = sage_emb_all[n_compounds:][val_prot_list]
    hgt_val_emb = hgt_emb_all[val_prot_list]

    cka = cka_similarity(sage_val_emb, hgt_val_emb)
    logger.info(f"  SAGE-HGT 蛋白嵌入 CKA (线性核): {cka:.4f}")

    # 计算余弦相似度分布
    sage_norm = sage_val_emb / (np.linalg.norm(sage_val_emb, axis=1, keepdims=True) + 1e-10)
    hgt_norm = hgt_val_emb / (np.linalg.norm(hgt_val_emb, axis=1, keepdims=True) + 1e-10)
    cos_sims = np.sum(sage_norm * hgt_norm, axis=1)
    logger.info(f"  余弦相似度: mean={cos_sims.mean():.4f}, std={cos_sims.std():.4f}, "
                f"min={cos_sims.min():.4f}, max={cos_sims.max():.4f}")

    # ── 保存分析结果 ──
    results = {
        "total_pairs": total,
        "both_correct": len(both_correct),
        "both_error": len(both_error),
        "sage_only_error": len(sage_only_error),
        "hgt_only_error": len(hgt_only_error),
        "complementarity_ratio": (len(sage_only_error) + len(hgt_only_error)) / max(len(sage_errors | hgt_errors), 1),
        "cka_similarity": cka,
        "cos_mean": cos_sims.mean(),
        "cos_std": cos_sims.std(),
    }
    pd.DataFrame([results]).to_csv(L4_RESULTS / "complementarity_analysis.csv", index=False)
    logger.info(f"\n结果已保存: {L4_RESULTS / 'complementarity_analysis.csv'}")

    total_time = time.time() - start_time
    logger.info(f"\n互补性分析完成！总耗时 {total_time / 60:.1f} 分钟")


if __name__ == "__main__":
    main()