#!/usr/bin/env python3
"""
Phase 4 v5: Multi-Task Positive-Unlabeled Neural Network (MT-PUNN)
==================================================================
叙事逻辑重构：
  从 v4.5 的"每个靶标独立二分类、以其他靶标活性为伪阴性"，
  转向"多任务阳性-未标记学习（Positive-Unlabeled Learning）"：
  所有 CIRI/铁死亡靶标共享一个分子编码器，在统一化学空间中联合学习，
  未标记样本不再被强制标记为阴性，而是作为 U 样本参与 nnPU 风险估计。

核心设计：
  1. 输入：化合物 ECFP + MACCS + RDKit 2D 描述符。
  2. 共享编码器：3 层 MLP（含残差与 Dropout）。
  3. N 个任务特定头（N = len(ALL_TARGET_GENES)），每个靶标一个 sigmoid 输出。
  4. 损失：每个有数据靶标独立计算 nnPU 损失后平均；无数据靶标不参与。
  5. 验证：Murcko 骨架切分，评估 held-out 阳性 vs 未标记样本的 AUC/AUPR。
  6. 候选排序：保留 v4.5 的多维度综合得分，但基于 MT-PUNN 预测概率。

关键参考：
  - 分子表示：
    - ECFP4: Rogers & Hahn, "Extended-Connectivity Fingerprints",
      J. Chem. Inf. Model. 2010, 50(5):742-754, doi:10.1021/ci100050t
    - MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys
    - RDKit 2D descriptors: Landrum G., RDKit open-source cheminformatics,
      https://github.com/rdkit/rdkit
  - 验证切分：
    - Murcko scaffold: Bemis & Murcko, "The Properties of Known Drugs. 1.
      Molecular Frameworks", J. Med. Chem. 1996, doi:10.1021/jm9602928
  - nnPU 损失: Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator",
    NeurIPS 2017, arXiv:1703.00593; 官方实现 https://github.com/kiryor/nnPUlearning
  - DTI 深度学习方法:
    - DeepPurpose: Huang et al., "DeepPurpose: a deep learning library for
      drug-target interaction prediction", Bioinformatics 2020,
      doi:10.1093/bioinformatics/btaa1005; GitHub:
      https://github.com/kexinhuang12345/DeepPurpose
    - MolTrans: Huang et al., "MolTrans: Molecular Interaction Transformer for
      drug-target interaction prediction", Bioinformatics 2021,
      doi:10.1093/bioinformatics/btaa880; GitHub:
      https://github.com/kexinhuang12345/MolTrans
  - 多靶标 PU 药物发现: Hao et al., "Developing a Semi-Supervised Approach Using a
    PU-Learning-Based Data Augmentation Strategy for Multitarget Drug Discovery",
    Int. J. Mol. Sci. 2024, doi:10.3390/ijms25158239
  - TCM 数据库: Ru et al., 2014, doi:10.1021/ci4005517; Wang et al., 2024,
    doi:10.3389/fphar.2024.1303693

输出：
  L4/results_v5/model_performance_v5.csv
  L4/results_v5/tcm_predictions_full_v5.csv
  L4/results_v5/tcm_top_candidates_v5.csv
  L4/results_v5/enrichment_analysis_v5.csv
  L4/results_v5/training_metrics_v5.json
  L4/results_v5/phase4_report_v5.md
  L4/logs/phase4_model_pipeline_v5.log
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# 复用 v4.5 中经过验证的数据加载与工具函数
sys.path.insert(0, str(Path(__file__).parent))
import phase4_model_pipeline_v4_5 as v45

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
warnings.filterwarnings("ignore", category=FutureWarning, module="rdkit")
warnings.filterwarnings("ignore", message=".*MorganGenerator.*")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results_v5"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "phase4_model_pipeline_v5.log"

# 清除 v4.5 导入时可能注册的 handler，避免日志写入 v4.5 文件
logger = logging.getLogger(__name__)
root = logging.getLogger()
for h in list(root.handlers):
    root.removeHandler(h)
for h in list(logger.handlers):
    logger.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

ALL_TARGET_GENES = v45.ALL_TARGET_GENES
CORE_GENES = v45.CORE_GENES
PRIORITY_TARGETS = v45.PRIORITY_TARGETS

# ============================================================
# 化合物特征工程：ECFP + MACCS + RDKit 描述符
# ============================================================
RDKIT_DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumHeteroatoms",
    "NumValenceElectrons",
    "NHOHCount",
    "NOCount",
    "RingCount",
    "FractionCSP3",
    "BalabanJ",
]


def _compute_maccs(smiles_iter):
    """计算 MACCS 指纹（167 bits），无效分子返回零向量。"""
    fps = []
    for smi in smiles_iter:
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            fps.append(np.zeros(167, dtype=np.float32))
            continue
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            arr = np.zeros(167, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
        except Exception:
            fps.append(np.zeros(167, dtype=np.float32))
    return np.array(fps, dtype=np.float32)


def _compute_rdkit_descriptors(smiles_iter):
    """计算 RDKit 2D 描述符，无效分子返回 NaN（后续用训练集均值填充）。"""
    desc_funcs = {name: getattr(Descriptors, name) for name in RDKIT_DESCRIPTOR_NAMES}
    rows = []
    for smi in smiles_iter:
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception:
            mol = None
        if mol is None:
            rows.append([np.nan] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        vals = []
        for name in RDKIT_DESCRIPTOR_NAMES:
            try:
                vals.append(float(desc_funcs[name](mol)))
            except Exception:
                vals.append(np.nan)
        rows.append(vals)
    return np.array(rows, dtype=np.float32)


def build_compound_features(smiles_list, ecfp4=None, stats=None):
    """
    拼接化合物特征：
    - ECFP4 和 MACCS 指纹保持二值性（不标准化）
    - RDKit 2D 描述符进行 Z-score 标准化

    参数：
      smiles_list: SMILES 列表。
      ecfp4: 预计算的 ECFP4 指纹（可选）。
      stats: 预计算的 (desc_mean, desc_std, desc_col_mean) 元组（可选）。
             若提供则用于标准化/填充，否则基于当前数据计算。

    返回：
      features, desc_mean, desc_std, desc_col_mean
    """
    logger.info("  计算 MACCS 指纹...")
    maccs = _compute_maccs(smiles_list)  # 保持二值性
    logger.info("  计算 RDKit 2D 描述符...")
    desc = _compute_rdkit_descriptors(smiles_list)

    # 仅对连续值描述符进行标准化，指纹特征保持原值
    if stats is None:
        # 描述符缺失值用列均值填充，无穷大截断
        desc_col_mean = np.nanmean(desc, axis=0)
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(desc_col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)

        # Z-score 标准化（仅对描述符）
        desc_mean = desc.mean(axis=0)
        desc_std = desc.std(axis=0) + 1e-8
        desc = (desc - desc_mean) / desc_std
    else:
        desc_mean, desc_std, desc_col_mean = stats
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(desc_col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        desc = (desc - desc_mean) / (desc_std + 1e-8)

    # 拼接特征：指纹保持原值（0/1），描述符已标准化
    if ecfp4 is not None:
        features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)
    else:
        features = np.hstack([maccs, desc]).astype(np.float32)
    return features, desc_mean, desc_std, desc_col_mean


# ============================================================
# 多任务 PU 标签矩阵构建
# ============================================================
def build_multitask_labels(compound_smiles, active_df, all_genes):
    """
    构建多任务标签矩阵。

    参数：
      compound_smiles: 用于训练/预测的化合物 SMILES 列表（去重）。
      active_df: 真实正样本 DataFrame，必须包含 gene, canonical_smiles。
      all_genes: 全部 38 个靶标基因列表。

    返回：
      Y: (n_compounds, n_genes) 标签矩阵，1=已知阳性，0=未标记。
      gene_to_col: gene -> 列索引。
    """
    n_compounds = len(compound_smiles)
    n_genes = len(all_genes)
    gene_to_col = {g: i for i, g in enumerate(all_genes)}

    smiles_to_idx = {smi: i for i, smi in enumerate(compound_smiles)}

    Y = np.zeros((n_compounds, n_genes), dtype=np.int8)
    for _, row in active_df.iterrows():
        gene = row["gene"]
        smi = row["canonical_smiles"]
        if gene not in gene_to_col or smi not in smiles_to_idx:
            continue
        Y[smiles_to_idx[smi], gene_to_col[gene]] = 1

    pos_per_gene = Y.sum(axis=0)
    logger.info(f"  多任务标签矩阵: {n_compounds} 化合物 × {n_genes} 靶标")
    for g, col in gene_to_col.items():
        n_pos = int(pos_per_gene[col])
        if n_pos > 0:
            logger.info(f"    {g}: {n_pos} 个已知阳性")
    return Y, gene_to_col


# ============================================================
# Scaffold 切分
# ============================================================
def scaffold_split(smiles_list, n_splits=5, seed=RANDOM_SEED):
    """
    按 Murcko 骨架分组，返回每折的 train/val 索引。
    返回 list of (train_idx, val_idx) 元组。
    """
    rng = np.random.default_rng(seed)
    scaffolds = {}
    for i, smi in enumerate(smiles_list):
        scaf = v45._compute_scaffold(smi)
        scaffolds.setdefault(scaf, []).append(i)

    scaffold_items = list(scaffolds.items())
    rng.shuffle(scaffold_items)

    folds = [([], []) for _ in range(n_splits)]
    for scaf, idxs in scaffold_items:
        fold_id = rng.integers(0, n_splits)
        for f in range(n_splits):
            if f == fold_id:
                folds[f][1].extend(idxs)
            else:
                folds[f][0].extend(idxs)

    result = []
    for train, val in folds:
        result.append((np.array(train, dtype=int), np.array(val, dtype=int)))
    return result


# ============================================================
# PyTorch 模型：Multi-Task Positive-Unlabeled Neural Network
# ============================================================
class MTPUNN(nn.Module):
    """共享编码器 + N 个任务特定头的多任务神经网络（含投影残差连接）。"""

    def __init__(self, input_dim: int, hidden_dims=(512, 256, 128), dropout=0.3, n_tasks=38):
        super().__init__()
        self.input_dim = input_dim
        self.n_tasks = n_tasks
        self.hidden_dims = hidden_dims

        self.blocks = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
            # 维度变化时使用线性投影，保证残差可加
            if prev != h:
                self.skip_projections.append(nn.Linear(prev, h))
            else:
                self.skip_projections.append(nn.Identity())
            prev = h

        # 任务特定头
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_tasks)])

    def forward(self, x):
        h = x
        for block, skip in zip(self.blocks, self.skip_projections):
            h = block(h) + skip(h)
        logits = torch.cat([head(h) for head in self.heads], dim=1)
        return torch.sigmoid(logits)


# ============================================================
# nnPU 损失
# ============================================================
def nnpu_loss(y_pred, y_true, prior, task_mask, eps=1e-7, gamma=1.0):
    """
    非负风险估计（nnPU）损失。
    
    参考：Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NeurIPS 2017
    
    公式：
      unbiased_risk = π_p * L_p + L_u - π_p * L_{p→n}
      clipped_risk = max(unbiased_risk, 0)  # 非负裁剪
    
    其中：
      - L_p: 正样本风险（预测为负的损失）
      - L_u: 未标记样本风险（预测为负的损失）
      - L_{p→n}: 正样本被当作负样本时的风险
    
    参数：
      y_pred: (batch, n_tasks) 预测概率。
      y_true: (batch, n_tasks) 标签，1=阳性，0=未标记。
      prior: (n_tasks,) 每个任务的类别先验 π_p。
      task_mask: (n_tasks,) bool，True 表示该任务有数据参与损失。
      eps: 数值稳定性常数。
      gamma: 非负风险裁剪参数（原始论文建议值为1）。

    返回：
      标量损失。
    """
    pos = (y_true == 1).float()
    unl = (y_true == 0).float()

    losses = []
    for t in range(y_pred.shape[1]):
        if not task_mask[t]:
            continue
        n_pos = pos[:, t].sum()
        n_unl = unl[:, t].sum()
        if n_pos == 0 or n_unl == 0:
            continue

        p_t = y_pred[:, t]
        prior_t = prior[t]

        # 正样本风险：-log(p(y=1|x)) for positive samples
        pos_risk = (pos[:, t] * -torch.log(p_t + eps)).sum() / n_pos
        
        # 未标记样本风险：-log(1-p(y=1|x)) for unlabeled samples
        neg_risk = (unl[:, t] * -torch.log(1.0 - p_t + eps)).sum() / n_unl
        
        # 正样本被误分类为负样本的风险（用于无偏估计）
        pos_as_neg_risk = (pos[:, t] * -torch.log(1.0 - p_t + eps)).sum() / n_pos

        # nnPU 无偏风险估计
        unbiased_risk = prior_t * pos_risk + neg_risk - prior_t * pos_as_neg_risk
        
        # 非负裁剪：避免负风险估计（关键创新点）
        clipped_risk = torch.max(unbiased_risk, torch.tensor(0.0, device=unbiased_risk.device))
        
        # 加权损失
        losses.append(clipped_risk * gamma)

    if not losses:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    return torch.stack(losses).mean()


# ============================================================
# 训练与评估
# ============================================================
def train_mtpunn(X_train, Y_train, X_val, Y_val, priors, task_mask,
                 hidden_dims=(512, 256, 128), dropout=0.3,
                 batch_size=512, epochs=200, lr=1e-3, patience=20,
                 device=None):
    """训练 MT-PUNN 并返回最佳模型与训练历史。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  使用设备: {device}")

    n_tasks = Y_train.shape[1]
    model = MTPUNN(X_train.shape[1], hidden_dims=hidden_dims,
                   dropout=dropout, n_tasks=n_tasks).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(Y_train.astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(Y_val.astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )

    priors_tensor = torch.from_numpy(priors).float().to(device)
    task_mask_tensor = torch.from_numpy(task_mask).bool().to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    nan_skip_count = 0
    max_nan_skips = 50  # 连续跳过 NaN 损失的最大次数
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        batch_nan_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = nnpu_loss(pred, yb, priors_tensor, task_mask_tensor)
            
            # 处理异常损失值
            if torch.isnan(loss):
                nan_skip_count += 1
                batch_nan_count += 1
                if nan_skip_count <= max_nan_skips:
                    logger.debug(f"  Epoch {epoch}: NaN loss encountered, skipping batch ({nan_skip_count}/{max_nan_skips})")
                    continue
                else:
                    logger.warning(f"  Epoch {epoch}: Too many NaN losses ({nan_skip_count}), breaking training early")
                    break
            elif torch.isinf(loss):
                logger.debug(f"  Epoch {epoch}: Inf loss encountered, skipping batch")
                continue
            
            # 零损失是有效状态，不应跳过
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if nan_skip_count > max_nan_skips:
            break
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        if batch_nan_count > 0:
            logger.info(f"  Epoch {epoch}: {batch_nan_count} batches skipped due to NaN/Inf loss")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = nnpu_loss(pred, yb, priors_tensor, task_mask_tensor)
                if not torch.isnan(loss):
                    val_losses.append(loss.item())
        avg_val_loss = np.mean(val_losses) if val_losses else float("inf")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Epoch {epoch:03d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        if epochs_no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def evaluate_per_task(model, X, Y, task_mask, device, max_unl_for_auc=2000, seed=RANDOM_SEED):
    """
    对每个任务计算 AUC/AUPR：阳性 vs 随机采样的未标记子集。
    仅返回有阳性的任务指标。
    """
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        pred = model(X_t).cpu().numpy()

    results = {}
    for t in range(Y.shape[1]):
        if not task_mask[t]:
            continue
        y_true = Y[:, t]
        pos_idx = np.where(y_true == 1)[0]
        if len(pos_idx) == 0:
            continue
        unl_idx = np.where(y_true == 0)[0]
        if len(unl_idx) == 0:
            continue
        if len(unl_idx) > max_unl_for_auc:
            unl_idx = rng.choice(unl_idx, size=max_unl_for_auc, replace=False)

        eval_idx = np.concatenate([pos_idx, unl_idx])
        y_eval = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(unl_idx))])
        p_eval = pred[eval_idx, t]

        try:
            auc = roc_auc_score(y_eval, p_eval)
            aupr = average_precision_score(y_eval, p_eval)
        except Exception:
            auc = np.nan
            aupr = np.nan
        results[t] = {"auc": float(auc), "aupr": float(aupr), "n_pos": int(len(pos_idx))}
    return results, pred


# ============================================================
# 候选排序与富集分析
# ============================================================
def rank_candidates(pred_tcm, tcm_data, all_genes, gene_to_col,
                    top_k=50, score_threshold=0.5, high_threshold=0.7):
    """基于 MT-PUNN 预测概率对 TCM 候选化合物进行综合排序。"""
    n_compounds = pred_tcm.shape[0]
    n_tasks = pred_tcm.shape[1]

    rows = []
    for i in range(n_compounds):
        scores = pred_tcm[i]
        n_hits = int((scores > score_threshold).sum())
        n_high = int((scores > high_threshold).sum())
        avg_score = float(scores.mean())
        max_score = float(scores.max())
        consistency = float(1.0 - scores.std())

        # 综合得分：与 v4.5 保持一致，便于横向对比
        composite = (
            0.30 * avg_score
            + 0.20 * max_score
            + 0.20 * (n_hits / n_tasks)
            + 0.20 * (n_high / n_tasks)
            + 0.10 * consistency
        )

        # Top 5 靶标
        top_idx = np.argsort(scores)[::-1][:5]
        top_targets = ", ".join([f"{all_genes[idx]}({scores[idx]:.3f})" for idx in top_idx])

        rows.append({
            "rank": 0,
            "MOL_ID": tcm_data["mol_ids"][i],
            "molecule_name": tcm_data["names"][i],
            "SMILES": tcm_data["smiles"][i],
            "composite_score": composite,
            "avg_score": avg_score,
            "max_score": max_score,
            "n_hits": n_hits,
            "n_high": n_high,
            "n_targets": n_tasks,
            "consistency": consistency,
            "top_targets": top_targets,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df.head(top_k)


def compute_enrichment(pred_tcm, tcm_data, Y_tcm, all_genes, gene_to_col,
                       percentiles=(1, 5, 10)):
    """
    仅对 TCM 中真实出现的阳性计算富集因子。
    返回 DataFrame。
    """
    rows = []
    n_compounds = pred_tcm.shape[0]
    for gene in all_genes:
        col = gene_to_col[gene]
        pos_idx = np.where(Y_tcm[:, col] == 1)[0]
        if len(pos_idx) == 0:
            continue
        baseline_rate = len(pos_idx) / n_compounds
        scores = pred_tcm[:, col]
        sorted_idx = np.argsort(scores)[::-1]
        for pct in percentiles:
            n_top = max(1, int(np.round(n_compounds * pct / 100)))
            top_idx = sorted_idx[:n_top]
            n_hits = len(set(top_idx) & set(pos_idx))
            ef = (n_hits / n_top) / baseline_rate if baseline_rate > 0 else 0.0
            rows.append({
                "gene": gene,
                "top_percent": pct,
                "n_top": n_top,
                "n_hits": n_hits,
                "n_pos_tcm": len(pos_idx),
                "baseline_rate": round(baseline_rate, 5),
                "enrichment_factor": round(ef, 2),
            })
    return pd.DataFrame(rows)


# ============================================================
# 报告生成
# ============================================================
def _df_to_markdown(df):
    """简单 DataFrame -> Markdown 表格，无需 tabulate。"""
    if df.empty:
        return "- 无数据"
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join([" --- " for _ in cols]) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join([str(v) for v in row]) + " |")
    return "\n".join(lines)


def generate_report(perf_df, top_df, enrich_df, history, n_params,
                    total_time, output_path, n_tcm, n_tasks):
    """生成 Markdown 报告。"""
    lines = [
        "# Phase 4 v5: MT-PUNN 训练报告",
        "",
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总耗时: {total_time:.1f} 分钟",
        "",
        "## 1. 架构与叙事",
        "- 模型: Multi-Task Positive-Unlabeled Neural Network (MT-PUNN)",
        "- 输入: 化合物 ECFP4 + MACCS + RDKit 2D 描述符",
        "- 共享编码器: 3 层 MLP（BatchNorm + Dropout）",
        f"- 输出头: {n_tasks} 个靶标独立 sigmoid（铁衰老差异表达基因集）",
        "- 损失: nnPU（非负风险估计），未标记样本不再强制为阴性",
        "- 验证: Murcko 骨架切分",
        "",
        "## 2. 模型规模",
        f"- 可训练参数: {n_params:,}",
        f"- 最终训练损失: {history['train_loss'][-1]:.4f}",
        f"- 最终验证损失: {history['val_loss'][-1]:.4f}",
        "",
        "## 3. 靶标级 CV 性能",
        _df_to_markdown(perf_df),
        "",
        "## 4. 富集因子",
        _df_to_markdown(enrich_df),
        "",
        "## 5. Top 候选化合物",
        _df_to_markdown(top_df.head(20)),
        "",
        "## 6. 与 v4.5 的关键差异",
        "- v4.5: 每个靶标独立训练浅层 ML，以其他靶标活性为伪阴性；",
        "- v5: 所有靶标共享深度编码器，采用 PU 学习，不将未标记样本强制标记为阴性；",
        "- v5 的 AUC/AUPR 使用未标记子集计算，通常低于 v4.5，但更贴近真实筛选场景。",
        "",
        "## 7. 局限",
        "- 类别先验 π_p 为保守估计，可能影响 PU 风险校准；",
        "- 未使用真实 inactive 或 decoy，EF 仍为乐观估计；",
        "- 深层模型在小样本靶标上可能不如 v4.5 稳定；",
        f"- TCM 候选池（L3 输出）包含 {n_tcm} 个唯一 SMILES，已做 SMILES 去重与名称-SMILES 一致性校验（剔除 MW 偏差过大的条目）。",
        "",
        "## 8. 关键参考",
        "### 分子表示与化学信息学工具",
        "- Rogers & Hahn (2010) Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50(5):742-754. doi:10.1021/ci100050t",
        "- MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys",
        "- RDKit: Landrum G., open-source cheminformatics toolkit, https://github.com/rdkit/rdkit",
        "- Murcko scaffold: Bemis & Murcko (1996) The Properties of Known Drugs. 1. Molecular Frameworks. J. Med. Chem. doi:10.1021/jm9602928",
        "",
        "### 阳性-未标记学习（PU Learning）",
        "- Kiryo et al. (2017) Positive-Unlabeled Learning with Non-Negative Risk Estimator. NeurIPS. arXiv:1703.00593",
        "- nnPU 官方实现: https://github.com/kiryor/nnPUlearning",
        "- Hao et al. (2024) PU-Learning-Based Data Augmentation for Multitarget Drug Discovery. Int. J. Mol. Sci. doi:10.3390/ijms25158239",
        "",
        "### 药物-靶标相互作用深度学习方法",
        "- DeepPurpose: Huang et al. (2020) DeepPurpose: a deep learning library for drug-target interaction prediction. Bioinformatics. doi:10.1093/bioinformatics/btaa1005; https://github.com/kexinhuang12345/DeepPurpose",
        "- MolTrans: Huang et al. (2021) MolTrans: Molecular Interaction Transformer for drug-target interaction prediction. Bioinformatics. doi:10.1093/bioinformatics/btaa880; https://github.com/kexinhuang12345/MolTrans",
        "",
        "### TCM 数据库",
        "- TCMSP: Ru et al. (2014) TCMSP: A Database of Systems Pharmacology for Drug Discovery from Herbal Medicines. J. Chem. Inf. Model. doi:10.1021/ci4005517",
        "- Wang et al. (2024) A critical assessment of Traditional Chinese Medicine databases. Front. Pharmacol. doi:10.3389/fphar.2024.1303693",
    ]
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 主流程
# ============================================================
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 v5: MT-PUNN 训练开始")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    compound_data = v45.load_compound_data()
    protein_data = v45.load_protein_data()
    # MT-PUNN 仅使用化合物特征，不使用蛋白质特征
    # 活性数据按 ALL_TARGET_GENES 过滤（96个铁衰老靶标）
    active_df = v45.load_activity_data(ALL_TARGET_GENES)

    # 2. 构建训练化合物池：所有已知活性 SMILES + TCM 化合物
    active_smiles = active_df["canonical_smiles"].unique().tolist()
    tcm_smiles = compound_data["smiles"].tolist()
    all_smiles = sorted(set(active_smiles + tcm_smiles))
    logger.info(f"训练化合物池: {len(all_smiles)} 个唯一 SMILES（活性 {len(active_smiles)} + TCM {len(tcm_smiles)}）")

    # 3. 计算 ECFP4 与增强特征（基于训练池拟合统计量）
    logger.info("=" * 60)
    logger.info("[v5] 计算化合物增强特征")
    logger.info("=" * 60)
    _, ecfp4_all = v45._compute_ecfp4(all_smiles)
    features_all, feat_mean, feat_std, feat_col_mean = build_compound_features(
        all_smiles, ecfp4=ecfp4_all
    )
    logger.info(f"  特征维度: {features_all.shape[1]}（ECFP4 2048 + MACCS 167 + RDKit {len(RDKIT_DESCRIPTOR_NAMES)}）")

    # 4. 构建多任务标签矩阵
    Y, gene_to_col = build_multitask_labels(all_smiles, active_df, ALL_TARGET_GENES)
    n_tasks = Y.shape[1]

    # 任务掩码与类别先验
    task_mask = np.array([Y[:, t].sum() > 0 for t in range(n_tasks)], dtype=bool)
    priors = np.array([min(0.5, Y[:, t].sum() / len(all_smiles)) for t in range(n_tasks)], dtype=np.float32)
    logger.info(f"  有数据任务数: {task_mask.sum()}/{n_tasks}")

    # 5. Scaffold 切分（5 折中第 0 折作为验证，其余训练）
    logger.info("=" * 60)
    logger.info("[v5] Murcko 骨架切分")
    logger.info("=" * 60)
    folds = scaffold_split(all_smiles, n_splits=5, seed=RANDOM_SEED)
    train_idx, val_idx = folds[0]
    logger.info(f"  训练集: {len(train_idx)}, 验证集: {len(val_idx)}")

    # 6. 训练 MT-PUNN
    logger.info("=" * 60)
    logger.info("[v5] 训练 MT-PUNN")
    logger.info("=" * 60)
    model, history = train_mtpunn(
        features_all[train_idx], Y[train_idx],
        features_all[val_idx], Y[val_idx],
        priors, task_mask,
        hidden_dims=(512, 256, 128), dropout=0.3,
        batch_size=512, epochs=200, lr=1e-3, patience=20,
        device=device,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  模型参数: {n_params:,}")

    # 7. 评估验证集
    logger.info("=" * 60)
    logger.info("[v5] 验证集评估")
    logger.info("=" * 60)
    val_results, _ = evaluate_per_task(model, features_all[val_idx], Y[val_idx],
                                       task_mask, device)

    perf_rows = []
    for t, res in val_results.items():
        gene = ALL_TARGET_GENES[t]
        perf_rows.append({
            "gene": gene,
            "val_auc": round(res["auc"], 4),
            "val_aupr": round(res["aupr"], 4),
            "n_pos": res["n_pos"],
            "prior": round(priors[t], 4),
        })
    perf_df = pd.DataFrame(perf_rows).sort_values("val_auc", ascending=False)
    logger.info(f"  可评估靶标: {len(perf_df)}")
    logger.info(f"  平均 val_auc: {perf_df['val_auc'].mean():.4f}")
    logger.info(f"  平均 val_aupr: {perf_df['val_aupr'].mean():.4f}")

    # 8. 在完整 TCM 化合物集上预测（避免训练池去重导致 TCM 丢失）
    logger.info("=" * 60)
    logger.info("[v5] TCM 预测")
    logger.info("=" * 60)
    _, ecfp4_tcm = v45._compute_ecfp4(compound_data["smiles"])
    features_tcm, _, _, _ = build_compound_features(
        compound_data["smiles"],
        ecfp4=ecfp4_tcm,
        stats=(feat_mean, feat_std, feat_col_mean),
    )
    Y_tcm, _ = build_multitask_labels(
        compound_data["smiles"].tolist(), active_df, ALL_TARGET_GENES
    )

    model.eval()
    with torch.no_grad():
        pred_tcm = model(torch.from_numpy(features_tcm).to(device)).cpu().numpy()

    # 9. 候选排序
    top_candidates = rank_candidates(pred_tcm, compound_data, ALL_TARGET_GENES, gene_to_col)

    # 10. 富集分析
    enrich_df = compute_enrichment(pred_tcm, compound_data, Y_tcm, ALL_TARGET_GENES, gene_to_col)

    # 11. 保存完整预测
    pred_df = pd.DataFrame(pred_tcm, columns=ALL_TARGET_GENES)
    pred_df.insert(0, "MOL_ID", compound_data["mol_ids"])
    pred_df.insert(1, "molecule_name", compound_data["names"])
    pred_df.insert(2, "SMILES", compound_data["smiles"])

    # 12. 保存输出
    L4_RESULTS.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(L4_RESULTS / "model_performance_v5.csv", index=False)
    pred_df.to_csv(L4_RESULTS / "tcm_predictions_full_v5.csv", index=False)
    top_candidates.to_csv(L4_RESULTS / "tcm_top_candidates_v5.csv", index=False)
    enrich_df.to_csv(L4_RESULTS / "enrichment_analysis_v5.csv", index=False)

    metrics = {
        "n_compounds": len(all_smiles),
        "n_tcm": len(tcm_smiles),
        "n_tasks": int(n_tasks),
        "n_trainable_tasks": int(task_mask.sum()),
        "n_params": int(n_params),
        "mean_val_auc": float(perf_df["val_auc"].mean()),
        "mean_val_aupr": float(perf_df["val_aupr"].mean()),
        "history": history,
    }
    with open(L4_RESULTS / "training_metrics_v5.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    total_time = (time.time() - start_time) / 60.0
    n_tcm = len(tcm_smiles)
    generate_report(perf_df, top_candidates, enrich_df, history, n_params,
                    total_time, L4_RESULTS / "phase4_report_v5.md", n_tcm, n_tasks)

    logger.info("=" * 60)
    logger.info("Phase 4 v5: MT-PUNN 训练完成")
    logger.info(f"  总耗时: {total_time:.1f} 分钟")
    logger.info(f"  输出目录: {L4_RESULTS}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception(f"v5 训练失败: {e}")
        raise
