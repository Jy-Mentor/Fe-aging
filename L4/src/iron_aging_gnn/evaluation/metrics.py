"""标准化 CPI/DTI 评估指标实现。

参考:
  - Bender et al. (2021) "A practical guide to large-scale docking", Nature Protocols.
  - Truchon & Bayly (2007) "Evaluating virtual screening methods", J. Chem. Inf. Model.
  - Zhao et al. (2009) "A statistical framework to evaluate virtual screening", BMC Bioinformatics.
  - Rifaioglu et al. (2021) "Recent applications of deep learning on in silico drug discovery",
    Briefings in Bioinformatics.

设计原则:
  1. 所有指标对退化情况（单类、全同分数、空输入）返回明确默认值，不抛异常。
  2. AUC/AUPR 提供可选的 bootstrap 置信区间，便于实验对比的统计显著性判断。
  3. EF 采用 per-compound 平均，避免高频化合物主导全局指标。
  4. ROCE 在目标 FPR 未达时使用保守外推，并提供平滑处理。
  5. 排名指标（Precision@K / Recall@K / Hit@K / NDCG@K）统一从预计算得分矩阵计算。
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def _is_degenerate(y_true: np.ndarray, y_score: np.ndarray) -> bool:
    """检查输入是否足以计算可靠指标。"""
    if y_true is None or y_score is None:
        return True
    if len(y_true) < 2 or len(y_score) < 2:
        return True
    if len(set(y_true)) < 2:
        return True
    if not np.isfinite(y_score).all():
        return True
    # 所有预测分数相同（含退化的常数输出）时，AUC/ROCE 无区分能力
    if len(np.unique(y_score)) <= 1:
        return True
    return False


def safe_roc_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    default: float = 0.5,
) -> float:
    """安全计算 ROC-AUC，退化情况返回 default。"""
    if _is_degenerate(y_true, y_score):
        return float(default)
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"safe_roc_auc_score 失败: {exc}")
        return float(default)


def safe_average_precision_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    default: float = 0.5,
) -> float:
    """安全计算 AUPR，退化情况返回 default。"""
    if _is_degenerate(y_true, y_score):
        return float(default)
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"safe_average_precision_score 失败: {exc}")
        return float(default)


def _bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """使用 bootstrap 估计指标置信区间。"""
    n = len(y_true)
    rng = np.random.RandomState(rng_seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]
        if len(set(y_t)) < 2:
            continue
        scores.append(metric_fn(y_t, y_s))
    if not scores:
        return {"mean": metric_fn(y_true, y_score), "low": np.nan, "high": np.nan}
    alpha = 1.0 - ci
    return {
        "mean": float(np.mean(scores)),
        "low": float(np.percentile(scores, 100 * alpha / 2)),
        "high": float(np.percentile(scores, 100 * (1 - alpha / 2))),
    }


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """计算成对二分类指标（AUC/AUPR）并可选输出 bootstrap 置信区间。

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数（概率或 logit 均可，AUC/AUPR 对单调变换不敏感）
        bootstrap: 是否计算 95% 置信区间
        n_bootstrap: bootstrap 重采样次数
        rng_seed: 随机种子

    Returns:
        dict: 包含 auc, aupr 及可选 ci 的指标字典
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # 过滤 NaN/Inf 分数，避免 sklearn 异常
    nan_mask = ~(np.isnan(y_score) | np.isinf(y_score))
    if (~nan_mask).any():
        logger.warning(
            f"compute_pairwise_metrics: 过滤 {(~nan_mask).sum()} 个 NaN/Inf 分数"
        )
        y_true = y_true[nan_mask]
        y_score = y_score[nan_mask]

    result: dict[str, float] = {
        "auc": safe_roc_auc_score(y_true, y_score),
        "aupr": safe_average_precision_score(y_true, y_score),
    }

    if bootstrap and len(set(y_true)) >= 2 and len(y_true) >= 10:
        auc_ci = _bootstrap_metric(
            y_true, y_score, roc_auc_score, n_bootstrap, rng_seed
        )
        aupr_ci = _bootstrap_metric(
            y_true, y_score, average_precision_score, n_bootstrap, rng_seed
        )
        result["auc_ci_low"] = auc_ci["low"]
        result["auc_ci_high"] = auc_ci["high"]
        result["aupr_ci_low"] = aupr_ci["low"]
        result["aupr_ci_high"] = aupr_ci["high"]

    return result


def compute_ranking_metrics(
    score_matrix: torch.Tensor,
    valid_pos_list: list[list[int]],
    ks: Iterable[int] = (10, 20, 50),
    fractions: Iterable[float] = (0.01, 0.05),
) -> dict[str, float]:
    """从预计算得分矩阵计算排名指标。

    指标:
      - precision@K, recall@K, hit@K, ndcg@K
      - EF@X% (per-compound 平均，符合虚拟筛选惯例)

    EF 公式（per-compound）:
        EF@X%_i = hits_i / (X% * n_pos_i)
    对全部验证化合物取平均。

    Args:
        score_matrix: (n_val, n_candidates) 原始分数张量
        valid_pos_list: 每个化合物的正样本局部列索引列表
        ks: Precision/Recall/Hit/NDCG 的 K 值
        fractions: EF 计算的顶部比例

    Returns:
        dict: 排名指标字典
    """
    if score_matrix.numel() == 0 or not valid_pos_list:
        return {}

    n_candidates = score_matrix.shape[1]
    ks = sorted({k for k in ks if 1 <= k <= n_candidates})
    fractions = sorted({f for f in fractions if 0 < f < 1})

    precision_at_k = {k: [] for k in ks}
    recall_at_k = {k: [] for k in ks}
    hit_at_k = {k: [] for k in ks}
    ndcg_at_k = {k: [] for k in ks}
    ef_values = {f: [] for f in fractions}

    score_matrix_cpu = score_matrix.cpu()

    for comp_idx, valid_pos in enumerate(valid_pos_list):
        if not valid_pos:
            continue
        n_pos = len(valid_pos)
        valid_pos_set = set(valid_pos)

        scores = score_matrix_cpu[comp_idx]
        _, sorted_indices = torch.sort(scores, descending=True)
        sorted_indices_cpu = sorted_indices.cpu().tolist()

        for k in ks:
            k_actual = min(k, n_candidates)
            top_k = sorted_indices_cpu[:k_actual]
            hits = sum(1 for p in top_k if p in valid_pos_set)
            precision_at_k[k].append(hits / k_actual)
            recall_at_k[k].append(hits / n_pos)
            hit_at_k[k].append(float(hits > 0))
            # NDCG@K
            dcg = 0.0
            for rank, idx in enumerate(top_k, start=1):
                if idx in valid_pos_set:
                    dcg += 1.0 / np.log2(rank + 1)
            idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(n_pos, k_actual) + 1))
            ndcg_at_k[k].append(dcg / idcg if idcg > 0 else 0.0)

        for frac in fractions:
            top_n = max(1, int(n_candidates * frac))
            hits_frac = sum(1 for p in sorted_indices_cpu[:top_n] if p in valid_pos_set)
            ef_values[frac].append(hits_frac / (frac * n_pos) if n_pos > 0 else 1.0)

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    result: dict[str, float] = {}
    for k in ks:
        result[f"precision@{k}"] = _mean(precision_at_k[k])
        result[f"recall@{k}"] = _mean(recall_at_k[k])
        result[f"hit@{k}"] = _mean(hit_at_k[k])
        result[f"ndcg@{k}"] = _mean(ndcg_at_k[k])

    for frac in fractions:
        pct = int(frac * 100)
        result[f"ef@{pct}%"] = _mean(ef_values[frac])

    return result


def compute_roce(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_levels: Iterable[float] = (0.005, 0.01, 0.02, 0.05),
) -> dict[str, float]:
    """计算 ROCE (ROC Enrichment) — 早期富集指标。

    ROCE@f = TPR(f) / f，其中 f 为目标假阳性率。
    使用 sklearn roc_curve 提取 FPR-TPR，对目标 FPR 做线性插值。
    若目标 FPR 超过实际最大 FPR，则使用最后 TPR（保守估计，避免虚高）。

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数
        fpr_levels: 目标 FPR 列表，默认 0.5%, 1%, 2%, 5%

    Returns:
        dict: ROCE@X%
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if _is_degenerate(y_true, y_score):
        return {f"ROCE@{int(f * 100)}%": 0.0 for f in fpr_levels}

    result = {}
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    for fp_rate in fpr_levels:
        if fp_rate <= 0:
            result[f"ROCE@{int(fp_rate * 100)}%"] = 0.0
            continue

        idx = np.searchsorted(fpr, fp_rate)
        if idx == 0:
            tpr_at = float(tpr[0])
        elif idx >= len(fpr):
            # 目标 FPR 超过曲线范围：使用最大 FPR 处 TPR（保守）
            tpr_at = float(tpr[-1])
        else:
            f_low, f_high = fpr[idx - 1], fpr[idx]
            t_low, t_high = tpr[idx - 1], tpr[idx]
            if f_high - f_low < 1e-12:
                tpr_at = float(t_low)
            else:
                tpr_at = float(
                    t_low + (fp_rate - f_low) * (t_high - t_low) / (f_high - f_low)
                )
        roce = tpr_at / fp_rate
        result[f"ROCE@{int(fp_rate * 100)}%"] = float(roce)

    return result


def compute_bedroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alpha: float = 20.0,
) -> float:
    """BEDROC (Boltzmann-Enhanced Discrimination of ROC) 早期富集指标。

    直接委托 RDKit CalcBEDROC 实现，避免手写公式偏差。
    参考: Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488-508.

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数
        alpha: 早期富集权重，默认 20.0

    Returns:
        float: BEDROC 值
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    n = len(y_true)
    n_act = int(np.sum(y_true))
    if n_act == 0:
        return 0.0
    if n_act == n:
        return 1.0

    order = np.argsort(y_score)[::-1]
    scores = [
        [float(y_score[order[i]]), bool(y_true[order[i]])]
        for i in range(n)
    ]
    try:
        return float(CalcBEDROC(scores, col=1, alpha=alpha))
    except Exception as exc:
        logger.warning(f"compute_bedroc 失败: {exc}")
        return 0.0


def compute_early_enrichment_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    score_matrix: torch.Tensor | None = None,
    valid_pos_list: list[list[int]] | None = None,
    ks: Iterable[int] = (10, 20, 50),
    ef_fractions: Iterable[float] = (0.01, 0.05),
    roce_fpr_levels: Iterable[float] = (0.005, 0.01, 0.02, 0.05),
    bedroc_alpha: float = 20.0,
) -> dict[str, float]:
    """统一计算早期富集与排名指标。

    Args:
        y_true: 成对标签
        y_score: 成对预测分数
        score_matrix: 可选的 (n_compounds, n_candidates) 得分矩阵
        valid_pos_list: 可选的每个化合物正样本索引
        ks: Precision/Recall/Hit/NDCG 的 K 值
        ef_fractions: EF 分数列表
        roce_fpr_levels: ROCE FPR 水平
        bedroc_alpha: BEDROC 权重

    Returns:
        dict: 合并后的指标字典
    """
    result: dict[str, float] = {}

    pairwise = compute_pairwise_metrics(y_true, y_score)
    result.update(pairwise)

    if score_matrix is not None and valid_pos_list:
        ranking = compute_ranking_metrics(score_matrix, valid_pos_list, ks, ef_fractions)
        result.update(ranking)

    roce = compute_roce(y_true, y_score, roce_fpr_levels)
    result.update(roce)

    result["BEDROC"] = compute_bedroc(y_true, y_score, bedroc_alpha)
    return result
