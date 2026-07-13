#!/usr/bin/env python3
"""
ESM-2 + MLP 基线分类器
目的：验证蛋白冷启动 AUPR 是否可以被纯序列模型（无 GNN）完全解释。
设计：冻结 ESM-2 提取蛋白嵌入 + 化合物指纹（ECFP4+MACCS+RDKit）→ 3 层 MLP 二分类。
使用与 v21 完全相同的冷启动拆分和评估协议。
"""
from __future__ import annotations

import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "esm2_mlp_baseline.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("esm2_mlp")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESM2MLPClassifier(nn.Module):
    """3 层 MLP: [compound_fingerprint + ESM2_embedding] → 二分类"""

    def __init__(self, comp_dim: int, prot_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.comp_proj = nn.Sequential(
            nn.Linear(comp_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, comp_feat: torch.Tensor, prot_feat: torch.Tensor) -> torch.Tensor:
        c = self.comp_proj(comp_feat)
        p = self.prot_proj(prot_feat)
        combined = torch.cat([c, p], dim=-1)
        return self.classifier(combined).squeeze(-1)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal Loss，与 v21 一致"""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def evaluate(
    model: nn.Module,
    comp_feats: np.ndarray,
    prot_feats: np.ndarray,
    compound_to_pos: dict[int, set],
    eval_compounds: list[int],
    eval_proteins: set[int],
    n_compounds: int,
    n_proteins: int,
    prot_global_to_local: dict[int, int],
    tag: str = "",
) -> dict:
    """评估化合物冷启动 + 蛋白冷启动"""
    model.eval()
    comp_t = torch.tensor(comp_feats, dtype=torch.float32, device=DEVICE)
    prot_t = torch.tensor(prot_feats, dtype=torch.float32, device=DEVICE)

    # ── 化合物冷启动 ──
    y_true, y_pred = [], []
    for c in eval_compounds:
        pos_set = compound_to_pos.get(c, set())
        if not pos_set:
            continue
        c_emb = comp_t[c].unsqueeze(0)
        for p in eval_proteins:
            p_local = prot_global_to_local.get(p, -1)
            if p_local < 0:
                continue
            p_emb = prot_t[p_local].unsqueeze(0)
            with torch.no_grad():
                score = torch.sigmoid(model(c_emb, p_emb)).item()
            y_pred.append(score)
            y_true.append(1.0 if p in pos_set else 0.0)

    val_auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
    val_aupr = average_precision_score(y_true, y_pred)
    logger.info(f"  [{tag}] 化合物冷启动: n_valid={len(eval_compounds)}, auc={val_auc:.4f}, aupr={val_aupr:.4f}")

    return {"auc": val_auc, "aupr": val_aupr}


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ESM-2 + MLP 基线分类器")
    logger.info("=" * 60)

    # ── 加载数据 ──
    logger.info(">>> 加载数据")
    cpi_df = p4.load_cpi_data()
    prot_feat_dict, gene_to_seq = p4.load_protein_features(use_esm2=True)
    warm_targets = sorted(set(cpi_df["gene"].unique()) & set(p4.ALL_FERRORAGING_GENES))
    logger.info(f"  温靶标: {len(warm_targets)} 个")
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    # ── 构建基因到索引的映射 ──
    # 使用与 v21 相同的图结构来获得一致的索引
    ppi_df = p4.load_ppi_network()
    gene_to_pathways = p4.load_kegg_pathways()
    graphs = p4.build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat_dict)

    n_compounds = graphs["n_compounds"]
    gene_to_idx = graphs["gene_to_idx"]
    warm_genes = [g for g in warm_targets if g in gene_to_idx]
    logger.info(f"  温靶标基因（在图中有索引）: {len(warm_genes)} 个")

    # ── 提取 ESM-2 蛋白嵌入 ──
    prot_esm_dim = graphs["prot_esm_dim"]
    prot_esm_embeddings = np.zeros((n_compounds, prot_esm_dim), dtype=np.float32)  # dummy for compounds
    prot_local_to_gene = {}

    # 构造蛋白嵌入矩阵: 只包含 warm_targets 中的蛋白
    warm_prot_local = []
    for g in warm_genes:
        g_idx = gene_to_idx[g]
        p_local = g_idx - n_compounds
        if p_local >= 0:
            warm_prot_local.append(p_local)
            prot_local_to_gene[p_local] = g

    logger.info(f"  温靶标蛋白局部索引: {len(warm_prot_local)} 个")

    # 蛋白嵌入: 使用 ESM-2 特征
    prot_feat_matrix = np.zeros((len(warm_prot_local), prot_esm_dim), dtype=np.float32)
    prot_global_to_local = {}
    for i, p_local in enumerate(warm_prot_local):
        g = prot_local_to_gene[p_local]
        prot_global_to_local[p_local] = i
        if g in prot_feat_dict:
            prot_feat_matrix[i] = prot_feat_dict[g][:prot_esm_dim].astype(np.float32)

    # ── 提取化合物指纹 ──
    all_smiles = list(graphs["smi_to_idx"].keys())
    smi_to_idx = graphs["smi_to_idx"]
    logger.info(f"  计算化合物特征 ({len(all_smiles)} 个)...")
    comp_feat_matrix, cp_mean, cp_std, cp_col_mean = p4.build_compound_features(all_smiles)
    logger.info(f"  化合物特征维度: {comp_feat_matrix.shape[1]}")

    # ── 数据拆分（与 v21 完全一致） ──
    all_compounds = sorted(smi_to_idx.values())
    random.seed(42)
    random.shuffle(all_compounds)

    n_train_comp = int(len(all_compounds) * 0.85)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    # 蛋白拆分
    all_proteins = list(warm_prot_local)
    random.shuffle(all_proteins)

    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        g = row["gene"]
        if g in gene_to_idx:
            p_local = gene_to_idx[g] - n_compounds
            if p_local >= 0 and p_local in warm_prot_local:
                cpi_proteins.add(p_local)

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

    # 预计算正样本
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        g = row["gene"]
        if smi in smi_to_idx and g in gene_to_idx:
            p_local = gene_to_idx[g] - n_compounds
            if p_local >= 0 and p_local in warm_prot_local:
                compound_to_pos[smi_to_idx[smi]].add(p_local)

    logger.info(f"  拆分: train_comp={len(train_compounds)}, val_comp={len(val_compounds)}, "
                f"train_prot={len(train_proteins)}, val_prot={len(val_proteins)}")

    # ── 构建模型 ──
    comp_dim = comp_feat_matrix.shape[1]
    model = ESM2MLPClassifier(comp_dim=comp_dim, prot_dim=prot_esm_dim, hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    logger.info(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # ── 训练 ──
    n_epochs = 50
    batch_size = 512
    patience = 10
    best_val_aupr = 0.0
    best_state = None
    patience_counter = 0

    comp_t = torch.tensor(comp_feat_matrix, dtype=torch.float32, device=DEVICE)
    prot_t = torch.tensor(prot_feat_matrix, dtype=torch.float32, device=DEVICE)

    logger.info(f">>> 开始训练 ({n_epochs} epochs, batch_size={batch_size})")

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        # 准备训练样本
        train_pos_pairs = []
        for c in train_compounds:
            pos_set = compound_to_pos.get(c, set())
            for p in pos_set:
                if p in train_proteins:
                    train_pos_pairs.append((c, p))

        if len(train_pos_pairs) == 0:
            logger.warning(f"  Epoch {epoch}: 无正样本对，跳过")
            continue

        # 每个正样本配 10 个负样本
        train_neg_pairs = []
        for c, p in train_pos_pairs:
            pos_set = compound_to_pos.get(c, set())
            neg_candidates = [bp for bp in train_proteins if bp not in pos_set]
            neg_samples = random.sample(neg_candidates, 10) if len(neg_candidates) > 10 else neg_candidates
            for np_ in neg_samples:
                train_neg_pairs.append((c, np_))

        all_train_pairs = train_pos_pairs + train_neg_pairs
        all_labels = [1.0] * len(train_pos_pairs) + [0.0] * len(train_neg_pairs)
        random.shuffle(list(zip(all_train_pairs, all_labels, strict=False)))

        # Mini-batch 训练
        indices = list(range(len(all_train_pairs)))
        random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_comp = [all_train_pairs[j][0] for j in batch_idx]
            batch_prot = [all_train_pairs[j][1] for j in batch_idx]
            batch_labels = torch.tensor([all_labels[j] for j in batch_idx], dtype=torch.float32, device=DEVICE)

            c_emb = comp_t[batch_comp]
            p_emb = prot_t[[prot_global_to_local[p] for p in batch_prot]]

            logits = model(c_emb, p_emb)
            loss = focal_loss(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # ── 验证 ──
        if epoch % 2 == 0:
            val_metrics = evaluate(
                model, comp_feat_matrix, prot_feat_matrix,
                compound_to_pos, val_compounds, val_proteins,
                n_compounds, len(warm_prot_local),
                prot_global_to_local, tag=f"epoch {epoch}"
            )
            val_aupr = val_metrics["aupr"]

            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            logger.info(f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | val_auc={val_metrics['auc']:.4f} | val_aupr={val_aupr:.4f}")

            if patience_counter >= patience:
                logger.info(f"  早停 (epoch {epoch})")
                break
        else:
            logger.info(f"  Epoch {epoch:3d} | loss={avg_loss:.4f}")

    # ── 加载最佳模型 ──
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  加载最优 checkpoint (val_aupr={best_val_aupr:.4f})")

    # ── 最终评估 ──
    logger.info("=" * 60)
    logger.info(">>> 最终评估")

    # 化合物冷启动
    val_metrics = evaluate(
        model, comp_feat_matrix, prot_feat_matrix,
        compound_to_pos, val_compounds, val_proteins,
        n_compounds, len(warm_prot_local),
        prot_global_to_local, tag="FINAL"
    )

    # 蛋白冷启动
    model.eval()
    y_true_pc, y_pred_pc = [], []
    for c in val_compounds:
        pos_set = compound_to_pos.get(c, set())
        if not pos_set:
            continue
        c_emb = comp_t[c].unsqueeze(0)
        for p in val_proteins:
            if p not in prot_global_to_local:
                continue
            p_emb = prot_t[prot_global_to_local[p]].unsqueeze(0)
            with torch.no_grad():
                score = torch.sigmoid(model(c_emb, p_emb)).item()
            y_pred_pc.append(score)
            y_true_pc.append(1.0 if p in pos_set else 0.0)

    prot_auc = roc_auc_score(y_true_pc, y_pred_pc) if len(set(y_true_pc)) > 1 else 0.5
    prot_aupr = average_precision_score(y_true_pc, y_pred_pc)
    logger.info(f"  蛋白冷启动: auc={prot_auc:.4f}, aupr={prot_aupr:.4f}")

    # ── 保存结果 ──
    results = {
        "model": "ESM2+MLP",
        "val_auc": val_metrics["auc"],
        "val_aupr": val_metrics["aupr"],
        "prot_auc": prot_auc,
        "prot_aupr": prot_aupr,
    }
    pd.DataFrame([results]).to_csv(L4_RESULTS / "model_performance_esm2_mlp_v27.csv", index=False)
    logger.info(f"  结果已保存: {L4_RESULTS / 'model_performance_esm2_mlp_v27.csv'}")

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"ESM-2 + MLP 基线完成！总耗时 {total_time / 60:.1f} 分钟")
    logger.info(f"  化合物冷启动: auc={val_metrics['auc']:.4f}, aupr={val_metrics['aupr']:.4f}")
    logger.info(f"  蛋白冷启动: auc={prot_auc:.4f}, aupr={prot_aupr:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()