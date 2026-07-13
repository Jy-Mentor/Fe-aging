"""冷启动评估模块：Cold Drug 与 Cold Target 实体级交叉验证。

参考:
  - GHCDTI (2025): Cold Drug AUC=0.920, AUPR=0.778; Cold Target AUC=0.881, AUPR=0.691
  - 10 折实体级交叉验证，测试实体在训练/验证中完全排除
  - 负样本按 10:1 比例生成，排除高相似度假阴性
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


class ColdStartEvaluator:
    """冷启动评估器：Cold Drug / Cold Target 实体级交叉验证。

    遵循 GHCDTI (2025) 评估协议：
      - 按实体（药物或靶点）将 CPI 对划分为训练/测试集
      - 测试实体在训练期间完全不可见
      - 10 折实体级交叉验证
      - 负样本 10:1 比例，排除高相似度假阴性

    用法:
        evaluator = ColdStartEvaluator(n_folds=10, negative_ratio=10, seed=42)
        results = evaluator.run_cold_drug_evaluation(cpi_pairs, drug_ids, predict_fn)
        results = evaluator.run_cold_target_evaluation(cpi_pairs, protein_ids, predict_fn)
    """

    def __init__(
        self,
        n_folds: int = 10,
        negative_ratio: int = 10,
        seed: int = 42,
        similarity_threshold: float = 0.8,
        drug_similarity_fn: Callable[[int, int], float] | None = None,
        protein_similarity_fn: Callable[[int, int], float] | None = None,
    ):
        """初始化冷启动评估器。

        Args:
            n_folds: 交叉验证折数，默认 10
            negative_ratio: 正:负样本比例，默认 10
            seed: 随机种子
            similarity_threshold: 高相似度阈值，用于排除假阴性（默认 0.8）
            drug_similarity_fn: 药物相似度函数 (drug_id, drug_id) -> float
            protein_similarity_fn: 蛋白质相似度函数 (protein_id, protein_id) -> float
        """
        self.n_folds = n_folds
        self.negative_ratio = negative_ratio
        self.seed = seed
        self.similarity_threshold = similarity_threshold
        self.drug_similarity_fn = drug_similarity_fn
        self.protein_similarity_fn = protein_similarity_fn
        self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # 实体级划分
    # ------------------------------------------------------------------

    def cold_drug_split(
        self,
        cpi_pairs: list[tuple[int, int]],
        drug_ids: list[int],
        test_ratio: float = 0.2,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """按药物实体划分训练/测试集。

        测试药物在训练集中完全不可见。

        Args:
            cpi_pairs: [(drug_id, protein_id), ...] CPI 正样本对
            drug_ids: 所有药物 ID 列表
            test_ratio: 测试药物比例，默认 0.2

        Returns:
            (train_pairs, test_pairs): 训练集和测试集 CPI 对
        """
        n_test = max(1, int(len(drug_ids) * test_ratio))
        shuffled = list(drug_ids)
        self._rng.shuffle(shuffled)
        test_drugs = set(shuffled[:n_test])

        train_pairs: list[tuple[int, int]] = []
        test_pairs: list[tuple[int, int]] = []
        for drug, protein in cpi_pairs:
            if drug in test_drugs:
                test_pairs.append((drug, protein))
            else:
                train_pairs.append((drug, protein))

        logger.info(
            f"Cold Drug 划分: {len(train_pairs)} train / {len(test_pairs)} test 对, "
            f"{len(drug_ids) - len(test_drugs)} train / {len(test_drugs)} test 药物"
        )
        return train_pairs, test_pairs

    def cold_target_split(
        self,
        cpi_pairs: list[tuple[int, int]],
        protein_ids: list[int],
        test_ratio: float = 0.2,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """按蛋白质实体划分训练/测试集。

        测试蛋白质在训练集中完全不可见。

        Args:
            cpi_pairs: [(drug_id, protein_id), ...] CPI 正样本对
            protein_ids: 所有蛋白质 ID 列表
            test_ratio: 测试蛋白质比例，默认 0.2

        Returns:
            (train_pairs, test_pairs): 训练集和测试集 CPI 对
        """
        n_test = max(1, int(len(protein_ids) * test_ratio))
        shuffled = list(protein_ids)
        self._rng.shuffle(shuffled)
        test_proteins = set(shuffled[:n_test])

        train_pairs: list[tuple[int, int]] = []
        test_pairs: list[tuple[int, int]] = []
        for drug, protein in cpi_pairs:
            if protein in test_proteins:
                test_pairs.append((drug, protein))
            else:
                train_pairs.append((drug, protein))

        logger.info(
            f"Cold Target 划分: {len(train_pairs)} train / {len(test_pairs)} test 对, "
            f"{len(protein_ids) - len(test_proteins)} train / {len(test_proteins)} test 蛋白"
        )
        return train_pairs, test_pairs

    def entity_level_kfold(
        self,
        cpi_pairs: list[tuple[int, int]],
        entity_ids: list[int],
        entity_extractor: Callable[[tuple[int, int]], int],
    ) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]:
        """实体级 k 折交叉验证划分。

        将实体划分为 n_folds 份，每折以一份实体为测试集，其余为训练集。

        Args:
            cpi_pairs: CPI 正样本对列表
            entity_ids: 实体 ID 列表（药物或蛋白质）
            entity_extractor: 从 CPI 对中提取实体 ID 的函数

        Returns:
            [(train_pairs, test_pairs), ...] 每折的训练/测试对
        """
        shuffled = list(entity_ids)
        self._rng.shuffle(shuffled)
        fold_size = len(shuffled) // self.n_folds

        folds: list[list[tuple[int, int]]] = []
        for fold_idx in range(self.n_folds):
            start = fold_idx * fold_size
            if fold_idx == self.n_folds - 1:
                end = len(shuffled)
            else:
                end = start + fold_size
            test_entities = set(shuffled[start:end])

            test_pairs: list[tuple[int, int]] = []
            train_pairs: list[tuple[int, int]] = []
            for pair in cpi_pairs:
                if entity_extractor(pair) in test_entities:
                    test_pairs.append(pair)
                else:
                    train_pairs.append(pair)
            folds.append((train_pairs, test_pairs))

        logger.info(f"实体级 {self.n_folds} 折交叉验证: {len(entity_ids)} 个实体")
        return folds

    # ------------------------------------------------------------------
    # 负采样
    # ------------------------------------------------------------------

    def sample_negatives(
        self,
        positive_pairs: list[tuple[int, int]],
        all_drugs: list[int],
        all_proteins: list[int],
        exclude_similar: bool = True,
    ) -> list[tuple[int, int]]:
        """10:1 固定比例负采样，可选排除高相似度假阴性。

        Args:
            positive_pairs: 正样本对列表
            all_drugs: 所有药物 ID 列表
            all_proteins: 所有蛋白质 ID 列表
            exclude_similar: 是否排除高相似度（药物或蛋白）的假阴性

        Returns:
            负样本对列表
        """
        pos_set = set(positive_pairs)
        drugs_arr = np.array(all_drugs)
        proteins_arr = np.array(all_proteins)

        n_target = self.negative_ratio * len(positive_pairs)
        negative_pairs: list[tuple[int, int]] = []
        max_attempts = n_target * 10
        attempts = 0

        while len(negative_pairs) < n_target and attempts < max_attempts:
            drug = int(self._rng.choice(drugs_arr))
            protein = int(self._rng.choice(proteins_arr))
            pair = (drug, protein)
            attempts += 1
            if pair in pos_set or pair in negative_pairs:
                continue
            if exclude_similar and self._is_similar_false_negative(
                pair, positive_pairs
            ):
                continue
            negative_pairs.append(pair)

        if len(negative_pairs) < n_target:
            logger.warning(
                f"负采样: 仅生成 {len(negative_pairs)}/{n_target} 个负样本 "
                f"(尝试 {attempts} 次)"
            )

        return negative_pairs

    def _is_similar_false_negative(
        self,
        candidate: tuple[int, int],
        positive_pairs: list[tuple[int, int]],
    ) -> bool:
        """检查候选负样本是否与已知正样本高度相似（可能为假阴性）。

        Args:
            candidate: 候选负样本 (drug_id, protein_id)
            positive_pairs: 已知正样本对

        Returns:
            True 如果候选负样本应被排除（与正样本高度相似）
        """
        cand_drug, cand_protein = candidate
        for pos_drug, pos_protein in positive_pairs:
            if self.drug_similarity_fn is not None and cand_protein == pos_protein:
                sim = self.drug_similarity_fn(cand_drug, pos_drug)
                if sim >= self.similarity_threshold:
                    return True
            if self.protein_similarity_fn is not None and cand_drug == pos_drug:
                sim = self.protein_similarity_fn(cand_protein, pos_protein)
                if sim >= self.similarity_threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> dict[str, float]:
        """计算 AUC 和 AUPR。

        Args:
            y_true: 真实标签 (0/1)
            y_score: 预测分数

        Returns:
            {"auc": float, "aupr": float}
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_score = np.asarray(y_score, dtype=np.float64)

        valid = ~(np.isnan(y_score) | np.isinf(y_score))
        if not valid.all():
            n_filtered = (~valid).sum()
            logger.warning(f"evaluate: 过滤 {n_filtered} 个 NaN/Inf 分数")
            y_true = y_true[valid]
            y_score = y_score[valid]

        if len(np.unique(y_true)) < 2:
            logger.warning("evaluate: 单一类别，返回默认值 0.5")
            return {"auc": 0.5, "aupr": 0.5}

        try:
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            logger.warning("AUC 计算失败，返回 0.5", exc_info=True)
            auc = 0.5

        try:
            aupr = float(average_precision_score(y_true, y_score))
        except Exception:
            logger.warning("AUPR 计算失败，返回 0.5", exc_info=True)
            aupr = 0.5

        return {"auc": auc, "aupr": aupr}

    # ------------------------------------------------------------------
    # 完整评估流程
    # ------------------------------------------------------------------

    def run_cold_drug_evaluation(
        self,
        cpi_pairs: list[tuple[int, int]],
        drug_ids: list[int],
        protein_ids: list[int],
        predict_fn: Callable[[list[tuple[int, int]]], np.ndarray],
        test_ratio: float = 0.2,
        use_kfold: bool = False,
    ) -> dict[str, float]:
        """Cold Drug 完整评估流程。

        Args:
            cpi_pairs: [(drug_id, protein_id), ...] CPI 正样本对
            drug_ids: 所有药物 ID 列表
            protein_ids: 所有蛋白质 ID 列表
            predict_fn: 预测函数，接收 [(drug_id, protein_id), ...]，返回预测分数数组
            test_ratio: 测试药物比例（use_kfold=False 时有效）
            use_kfold: 是否使用 k 折交叉验证（否则单次划分）

        Returns:
            {"auc": float, "aupr": float} 或 k 折均值 {"auc_mean": ..., "auc_std": ..., ...}
        """
        if use_kfold:
            folds = self.entity_level_kfold(
                cpi_pairs, drug_ids,
                entity_extractor=lambda pair: pair[0],
            )
            return self._run_kfold_evaluation(folds, protein_ids, predict_fn, "Cold Drug")

        train_pairs, test_pairs = self.cold_drug_split(
            cpi_pairs, drug_ids, test_ratio=test_ratio
        )
        return self._run_single_evaluation(
            train_pairs, test_pairs, drug_ids, protein_ids, predict_fn, "Cold Drug"
        )

    def run_cold_target_evaluation(
        self,
        cpi_pairs: list[tuple[int, int]],
        drug_ids: list[int],
        protein_ids: list[int],
        predict_fn: Callable[[list[tuple[int, int]]], np.ndarray],
        test_ratio: float = 0.2,
        use_kfold: bool = False,
    ) -> dict[str, float]:
        """Cold Target 完整评估流程。

        Args:
            cpi_pairs: [(drug_id, protein_id), ...] CPI 正样本对
            drug_ids: 所有药物 ID 列表
            protein_ids: 所有蛋白质 ID 列表
            predict_fn: 预测函数，接收 [(drug_id, protein_id), ...]，返回预测分数数组
            test_ratio: 测试蛋白质比例（use_kfold=False 时有效）
            use_kfold: 是否使用 k 折交叉验证（否则单次划分）

        Returns:
            {"auc": float, "aupr": float} 或 k 折均值 {"auc_mean": ..., "auc_std": ..., ...}
        """
        if use_kfold:
            folds = self.entity_level_kfold(
                cpi_pairs, protein_ids,
                entity_extractor=lambda pair: pair[1],
            )
            return self._run_kfold_evaluation(folds, protein_ids, predict_fn, "Cold Target")

        train_pairs, test_pairs = self.cold_target_split(
            cpi_pairs, protein_ids, test_ratio=test_ratio
        )
        return self._run_single_evaluation(
            train_pairs, test_pairs, drug_ids, protein_ids, predict_fn, "Cold Target"
        )

    def _run_single_evaluation(
        self,
        train_pairs: list[tuple[int, int]],
        test_pairs: list[tuple[int, int]],
        drug_ids: list[int],
        protein_ids: list[int],
        predict_fn: Callable[[list[tuple[int, int]]], np.ndarray],
        task_name: str,
    ) -> dict[str, float]:
        """单次冷启动评估。"""
        test_negatives = self.sample_negatives(test_pairs, drug_ids, protein_ids)
        all_test_pairs = test_pairs + test_negatives
        y_true = np.array(
            [1] * len(test_pairs) + [0] * len(test_negatives), dtype=np.float64
        )
        y_score = predict_fn(all_test_pairs)

        metrics = self.evaluate(y_true, y_score)
        logger.info(
            f"{task_name} 评估: AUC={metrics['auc']:.4f}, AUPR={metrics['aupr']:.4f}, "
            f"正样本={len(test_pairs)}, 负样本={len(test_negatives)}"
        )
        return metrics

    def _run_kfold_evaluation(
        self,
        folds: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]],
        protein_ids: list[int],
        predict_fn: Callable[[list[tuple[int, int]]], np.ndarray],
        task_name: str,
    ) -> dict[str, float]:
        """k 折交叉验证评估。"""
        all_drugs = list({d for fold in folds for pair in fold[0] + fold[1] for d in [pair[0]]})
        auc_list: list[float] = []
        aupr_list: list[float] = []

        for fold_idx, (train_pairs, test_pairs) in enumerate(folds):
            fold_metrics = self._run_single_evaluation(
                train_pairs, test_pairs, all_drugs, protein_ids,
                predict_fn, f"{task_name} fold-{fold_idx + 1}"
            )
            auc_list.append(fold_metrics["auc"])
            aupr_list.append(fold_metrics["aupr"])

        auc_arr = np.array(auc_list)
        aupr_arr = np.array(aupr_list)
        return {
            "auc_mean": float(np.mean(auc_arr)),
            "auc_std": float(np.std(auc_arr, ddof=1)),
            "aupr_mean": float(np.mean(aupr_arr)),
            "aupr_std": float(np.std(aupr_arr, ddof=1)),
            "auc_per_fold": auc_list,
            "aupr_per_fold": aupr_list,
        }


# ------------------------------------------------------------------
# 便捷函数
# ------------------------------------------------------------------

def cold_drug_masks(
    cpi_pairs: list[tuple[int, int]],
    drug_ids: list[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 Cold Drug 训练/测试 mask。

    Args:
        cpi_pairs: [(drug_id, protein_id), ...]
        drug_ids: 所有药物 ID
        test_ratio: 测试药物比例
        seed: 随机种子

    Returns:
        (train_mask, test_mask): 布尔数组，长度等于 cpi_pairs
    """
    evaluator = ColdStartEvaluator(seed=seed)
    train_pairs, test_pairs = evaluator.cold_drug_split(
        cpi_pairs, drug_ids, test_ratio=test_ratio
    )
    test_set = set(test_pairs)
    n = len(cpi_pairs)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for i, pair in enumerate(cpi_pairs):
        if pair in test_set:
            test_mask[i] = True
        else:
            train_mask[i] = True
    return train_mask, test_mask


def cold_target_masks(
    cpi_pairs: list[tuple[int, int]],
    protein_ids: list[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 Cold Target 训练/测试 mask。

    Args:
        cpi_pairs: [(drug_id, protein_id), ...]
        protein_ids: 所有蛋白质 ID
        test_ratio: 测试蛋白质比例
        seed: 随机种子

    Returns:
        (train_mask, test_mask): 布尔数组，长度等于 cpi_pairs
    """
    evaluator = ColdStartEvaluator(seed=seed)
    train_pairs, test_pairs = evaluator.cold_target_split(
        cpi_pairs, protein_ids, test_ratio=test_ratio
    )
    test_set = set(test_pairs)
    n = len(cpi_pairs)
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for i, pair in enumerate(cpi_pairs):
        if pair in test_set:
            test_mask[i] = True
        else:
            train_mask[i] = True
    return train_mask, test_mask