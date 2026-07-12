"""评估指标测试

测试 AUC/AUPR/ROCE/BEDROC/EF/NDCG 等指标计算正确性。
"""

import numpy as np
import torch


class TestPairwiseMetrics:
    """AUC/AUPR 成对指标测试"""

    def test_perfect_prediction(self):
        from iron_aging_gnn.evaluation.metrics import compute_pairwise_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = compute_pairwise_metrics(y_true, y_score)
        assert result["auc"] == 1.0
        assert result["aupr"] == 1.0

    def test_random_prediction(self):
        from iron_aging_gnn.evaluation.metrics import compute_pairwise_metrics
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = compute_pairwise_metrics(y_true, y_score)
        assert result["auc"] == 0.5
        assert result["aupr"] == 0.5

    def test_degenerate_single_class(self):
        from iron_aging_gnn.evaluation.metrics import compute_pairwise_metrics
        y_true = np.array([0, 0, 0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = compute_pairwise_metrics(y_true, y_score)
        assert result["auc"] == 0.5
        assert result["aupr"] == 0.5

    def test_with_bootstrap(self):
        from iron_aging_gnn.evaluation.metrics import compute_pairwise_metrics
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = rng.rand(100)
        result = compute_pairwise_metrics(y_true, y_score, bootstrap=True, n_bootstrap=100)
        assert "auc_ci_low" in result
        assert "auc_ci_high" in result
        assert "aupr_ci_low" in result
        assert "aupr_ci_high" in result


class TestRankingMetrics:
    """排名指标测试"""

    def test_basic(self):
        from iron_aging_gnn.evaluation.metrics import compute_ranking_metrics
        score_matrix = torch.tensor([
            [0.9, 0.1, 0.5, 0.8, 0.3],
            [0.2, 0.9, 0.4, 0.1, 0.7],
        ])
        valid_pos_list = [[0, 3], [1, 4]]
        result = compute_ranking_metrics(score_matrix, valid_pos_list, ks=(3, 5))
        assert "precision@3" in result
        assert "recall@3" in result
        assert "hit@3" in result
        assert "ndcg@3" in result
        assert result["hit@3"] == 1.0  # 前3名都命中

    def test_empty_input(self):
        from iron_aging_gnn.evaluation.metrics import compute_ranking_metrics
        result = compute_ranking_metrics(torch.tensor([]), [])
        assert result == {}


class TestROCE:
    """ROCE 早期富集测试"""

    def test_perfect_enrichment(self):
        from iron_aging_gnn.evaluation.metrics import compute_roce
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.linspace(0, 0.3, 50), np.linspace(0.7, 1.0, 50)])
        result = compute_roce(y_true, y_score)
        assert "ROCE@1%" in result
        assert "ROCE@5%" in result
        assert result["ROCE@1%"] > 0

    def test_degenerate(self):
        from iron_aging_gnn.evaluation.metrics import compute_roce
        y_true = np.array([0, 0, 0])
        y_score = np.array([0.1, 0.2, 0.3])
        result = compute_roce(y_true, y_score)
        assert result["ROCE@1%"] == 0.0


class TestBEDROC:
    """BEDROC 测试"""

    def test_perfect(self):
        from iron_aging_gnn.evaluation.metrics import compute_bedroc
        n = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.concatenate([np.linspace(0, 0.3, 50), np.linspace(0.7, 1.0, 50)])
        result = compute_bedroc(y_true, y_score)
        assert result > 0.5

    def test_all_negative(self):
        from iron_aging_gnn.evaluation.metrics import compute_bedroc
        y_true = np.zeros(50)
        y_score = np.random.rand(50)
        result = compute_bedroc(y_true, y_score)
        assert result == 0.0