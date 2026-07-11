"""标准化验证协议：固定比例负采样、分层k折交叉验证、多种子评估、结果归档。

使评估结果与领域文献可比，符合 DTI/CPI 虚拟筛选标准实践。

参考:
  - Bender et al. (2021) "A practical guide to large-scale docking", Nature Protocols.
  - Rifaioglu et al. (2021) "Recent applications of deep learning on in silico drug discovery",
    Briefings in Bioinformatics.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .metrics import (
    compute_bedroc,
    compute_pairwise_metrics,
    compute_ranking_metrics,
    compute_roce,
)

logger = logging.getLogger(__name__)


def fixed_ratio_negative_sampling(
    positive_pairs: list[tuple[int, int]],
    all_compounds: list[int],
    all_proteins: list[int],
    ratio: int = 100,
    seed: int = 42,
    exclude_known_positives: bool = True,
) -> list[tuple[int, int]]:
    """固定比例负采样：为每个正样本生成 ratio 个负样本。

    Args:
        positive_pairs: [(comp_idx, prot_idx), ...] 正样本对
        all_compounds: 全部化合物索引列表
        all_proteins: 全部蛋白索引列表
        ratio: 正:负比例，默认 100（1:100，符合虚拟筛选文献惯例）
        seed: 随机种子
        exclude_known_positives: 是否排除已知正样本

    Returns:
        [(comp_idx, prot_idx), ...] 负样本对列表
    """
    rng = np.random.RandomState(seed)
    pos_set = set(positive_pairs) if exclude_known_positives else set()
    compounds = np.array(all_compounds)
    proteins = np.array(all_proteins)

    negative_pairs: list[tuple[int, int]] = []
    max_attempts = ratio * len(positive_pairs) * 10
    attempts = 0

    while len(negative_pairs) < ratio * len(positive_pairs) and attempts < max_attempts:
        comp = int(rng.choice(compounds))
        prot = int(rng.choice(proteins))
        pair = (comp, prot)
        if pair not in pos_set and pair not in negative_pairs:
            negative_pairs.append(pair)
        attempts += 1

    if len(negative_pairs) < ratio * len(positive_pairs):
        logger.warning(
            f"固定比例负采样: 仅生成 {len(negative_pairs)}/{ratio * len(positive_pairs)} 个负样本 "
            f"(尝试 {attempts} 次后耗尽有效组合)"
        )

    return negative_pairs


def stratified_kfold_split(
    compounds: list[int],
    compound_to_pos: dict[int, set[int]],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """分层 k 折交叉验证：按化合物分层，确保每折正样本比例相近。

    Args:
        compounds: 全部化合物索引
        compound_to_pos: {comp_idx: {prot_idx, ...}} 正样本映射
        n_folds: k 折数
        seed: 随机种子

    Returns:
        [(train_compounds, val_compounds), ...] 每折的训练/验证化合物列表
    """
    rng = np.random.RandomState(seed)
    # 按正样本数分层
    strata = defaultdict(list)
    for comp in compounds:
        n_pos = len(compound_to_pos.get(comp, set()))
        if n_pos == 0:
            strata[0].append(comp)
        elif n_pos <= 5:
            strata["low"].append(comp)
        elif n_pos <= 20:
            strata["mid"].append(comp)
        else:
            strata["high"].append(comp)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for stratum_comps in strata.values():
        shuffled = list(stratum_comps)
        rng.shuffle(shuffled)
        for i, comp in enumerate(shuffled):
            folds[i % n_folds].append(comp)

    splits: list[tuple[list[int], list[int]]] = []
    for i in range(n_folds):
        val = folds[i]
        train = [c for j in range(n_folds) if j != i for c in folds[j]]
        splits.append((train, val))

    logger.info(f"分层 {n_folds} 折交叉验证: strata={dict((k, len(v)) for k, v in strata.items())}")
    return splits


def evaluate_with_multiple_seeds(
    evaluate_fn: Callable[[int], dict[str, float]],
    seeds: list[int] = (42, 123, 456),
) -> dict[str, dict[str, float]]:
    """多随机种子评估：多次运行评估函数，输出均值±标准差。

    Args:
        evaluate_fn: 评估函数，接收 seed 参数，返回 {metric_name: value}
        seeds: 种子列表

    Returns:
        {
            "mean": {metric: mean_value},
            "std": {metric: std_value},
            "per_seed": {seed: {metric: value}},
        }
    """
    per_seed: dict[int, dict[str, float]] = {}
    all_metrics: dict[str, list[float]] = defaultdict(list)

    for seed in seeds:
        logger.info(f"多种子评估: seed={seed}")
        metrics = evaluate_fn(seed)
        per_seed[seed] = metrics
        for k, v in metrics.items():
            all_metrics[k].append(v)

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    for k, values in all_metrics.items():
        mean_metrics[k] = float(np.mean(values))
        std_metrics[k] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    return {
        "mean": mean_metrics,
        "std": std_metrics,
        "per_seed": {str(s): m for s, m in per_seed.items()},
    }


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    score_matrix: torch.Tensor | None = None,
    valid_pos_list: list[list[int]] | None = None,
    config: dict | None = None,
) -> dict[str, float]:
    """标准化模型评估：计算全套指标。

    Args:
        y_true: (N,) 真实标签
        y_score: (N,) 预测分数
        score_matrix: 可选 (n_compounds, n_candidates) 得分矩阵
        valid_pos_list: 可选每个化合物的正样本索引
        config: 评估配置字典（来自 evaluation.yaml）

    Returns:
        {metric_name: value} 指标字典
    """
    cfg = config or {}
    metrics_cfg = cfg.get("metrics", {})
    bootstrap_cfg = metrics_cfg.get("bootstrap", {})

    do_bootstrap = bootstrap_cfg.get("enabled", False)
    n_bootstrap = bootstrap_cfg.get("n_iterations", 1000)

    pairwise = compute_pairwise_metrics(
        y_true, y_score,
        bootstrap=do_bootstrap,
        n_bootstrap=n_bootstrap,
    )

    result: dict[str, float] = dict(pairwise)

    roce = compute_roce(y_true, y_score)
    result.update(roce)

    result["BEDROC_20"] = compute_bedroc(y_true, y_score, alpha=20.0)
    result["BEDROC_160.9"] = compute_bedroc(y_true, y_score, alpha=160.9)

    if score_matrix is not None and valid_pos_list:
        ranking = compute_ranking_metrics(
            score_matrix, valid_pos_list,
            ks=(10, 50),
            fractions=(0.01, 0.05),
        )
        result.update(ranking)

    logger.info(
        f"评估完成: AUC={result.get('auc', 0):.4f}, "
        f"AUPR={result.get('aupr', 0):.4f}, "
        f"BEDROC_20={result.get('BEDROC_20', 0):.4f}"
    )
    return result


def archive_results(
    results: dict,
    output_dir: str | Path = "runs/evaluations",
    prefix: str = "eval",
    config: dict | None = None,
) -> Path:
    """归档评估结果到 JSON 和 CSV 文件。

    Args:
        results: 评估结果字典
        output_dir: 输出目录
        prefix: 文件名前缀
        config: 归档配置

    Returns:
        归档输出目录路径
    """
    cfg = config or {}
    archiving_cfg = cfg.get("archiving", {})
    output_dir = Path(archiving_cfg.get("output_dir", output_dir))
    include_timestamp = archiving_cfg.get("include_timestamp", True)
    fmt = archiving_cfg.get("format", "json")

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
    ts_part = f"_{timestamp}" if timestamp else ""

    saved_files: list[Path] = []

    if fmt in ("json", "both"):
        json_path = output_dir / f"{prefix}{ts_part}.json"
        serializable = _make_serializable(results)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
        saved_files.append(json_path)
        logger.info(f"评估结果已归档 JSON: {json_path}")

    if fmt in ("csv", "both"):
        csv_path = output_dir / f"{prefix}{ts_part}.csv"
        _save_as_csv(results, csv_path)
        saved_files.append(csv_path)
        logger.info(f"评估结果已归档 CSV: {csv_path}")

    return output_dir


def _make_serializable(obj):
    """递归转换 numpy/torch 类型为 Python 原生类型。"""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _save_as_csv(results: dict, path: Path) -> None:
    """将扁平化指标字典保存为 CSV。"""
    flat: dict[str, float] = {}
    for k, v in results.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (int, float, np.floating, np.integer)):
                    flat[f"{k}_{sub_k}"] = float(sub_v)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            flat[str(k)] = float(v)

    with open(path, "w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in sorted(flat.items()):
            f.write(f"{k},{v:.6f}\n")


class ValidationProtocol:
    """标准化验证协议：统一管理负采样、交叉验证、多种子评估和结果归档。

    用法:
        protocol = ValidationProtocol(config)
        protocol.setup(positive_pairs, all_compounds, all_proteins)

        # 固定比例负采样评估
        results = protocol.evaluate_with_negative_sampling(
            model_predict_fn, ratio=100
        )

        # 多随机种子评估
        results = protocol.evaluate_multi_seed(model_predict_fn)

        # 归档
        protocol.archive(results)
    """

    def __init__(self, config: dict | None = None):
        """初始化验证协议。

        Args:
            config: 评估配置字典（来自 evaluation.yaml）
        """
        self.config = config or {}
        self.positive_pairs: list[tuple[int, int]] = []
        self.all_compounds: list[int] = []
        self.all_proteins: list[int] = []
        self._rng = np.random.RandomState(
            self.config.get("negative_sampling", {}).get("seed", 42)
        )

    def setup(
        self,
        positive_pairs: list[tuple[int, int]],
        all_compounds: list[int],
        all_proteins: list[int],
    ) -> None:
        """设置验证数据。

        Args:
            positive_pairs: [(comp_idx, prot_idx), ...] 正样本对
            all_compounds: 全部化合物索引
            all_proteins: 全部蛋白索引
        """
        self.positive_pairs = positive_pairs
        self.all_compounds = all_compounds
        self.all_proteins = all_proteins

    def sample_negatives(self, ratio: int = 100, seed: int | None = None) -> list[tuple[int, int]]:
        """生成固定比例负样本。

        Args:
            ratio: 正:负比例
            seed: 随机种子，None 使用配置默认值

        Returns:
            负样本对列表
        """
        ns_cfg = self.config.get("negative_sampling", {})
        if seed is None:
            seed = ns_cfg.get("seed", 42)
        exclude = ns_cfg.get("exclude_known_positives", True)

        return fixed_ratio_negative_sampling(
            self.positive_pairs,
            self.all_compounds,
            self.all_proteins,
            ratio=ratio,
            seed=seed,
            exclude_known_positives=exclude,
        )

    def cross_validation_splits(self, n_folds: int = 5) -> list[tuple[list[int], list[int]]]:
        """生成分层 k 折交叉验证划分。

        Args:
            n_folds: k 折数

        Returns:
            [(train_compounds, val_compounds), ...]
        """
        cv_cfg = self.config.get("cross_validation", {})
        n_folds = cv_cfg.get("n_folds", n_folds)
        seed = cv_cfg.get("seed", 42)

        compound_to_pos: dict[int, set[int]] = defaultdict(set)
        for comp, prot in self.positive_pairs:
            compound_to_pos[comp].add(prot)

        return stratified_kfold_split(
            self.all_compounds, compound_to_pos, n_folds=n_folds, seed=seed
        )

    def evaluate_with_negative_sampling(
        self,
        predict_fn: Callable[[list[tuple[int, int]]], tuple[np.ndarray, np.ndarray]],
        ratio: int | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """固定比例负采样评估。

        Args:
            predict_fn: 预测函数，接收 [(comp, prot), ...]，返回 (y_true, y_score)
            ratio: 正:负比例，None 使用配置默认值
            seed: 随机种子

        Returns:
            指标字典
        """
        ns_cfg = self.config.get("negative_sampling", {})
        if ratio is None:
            ratio = ns_cfg.get("default_ratio", 100)

        negatives = self.sample_negatives(ratio=ratio, seed=seed)
        all_pairs = list(self.positive_pairs) + negatives
        y_true = np.array(
            [1] * len(self.positive_pairs) + [0] * len(negatives),
            dtype=np.float32,
        )

        y_true_pred, y_score = predict_fn(all_pairs)

        return evaluate_model_predictions(y_true, y_score, config=self.config)

    def evaluate_multi_seed(
        self,
        predict_fn: Callable[[int], dict[str, float]],
        seeds: list[int] | None = None,
    ) -> dict[str, dict[str, float]]:
        """多随机种子评估。

        Args:
            predict_fn: 评估函数，接收 seed，返回 {metric: value}
            seeds: 种子列表，None 使用配置默认值

        Returns:
            {"mean": {...}, "std": {...}, "per_seed": {...}}
        """
        ms_cfg = self.config.get("multi_seed", {})
        if seeds is None:
            seeds = ms_cfg.get("seeds", [42, 123, 456])

        return evaluate_with_multiple_seeds(predict_fn, seeds)

    def archive(self, results: dict, prefix: str = "eval") -> Path:
        """归档评估结果。

        Args:
            results: 评估结果字典
            prefix: 文件名前缀

        Returns:
            归档输出目录
        """
        return archive_results(results, prefix=prefix, config=self.config)