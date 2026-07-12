"""图模块测试：DropEdge 正则化、头尾节点划分、训练/验证集拆分

测试 iron_aging_gnn.graph.sampling 和 iron_aging_gnn.graph.split 中的关键函数。
"""

import pytest
import torch


class TestDropEdge:
    """DropEdge 正则化：随机丢弃边以缓解过拟合与过平滑"""

    @pytest.fixture(autouse=True)
    def setup(self):
        pass

    def test_drop_edge_reduces_count(self):
        """drop_edge(p=0.5) 应丢弃约一半的边"""
        from iron_aging_gnn.graph.sampling import drop_edge

        edge_index = torch.randint(0, 100, (2, 1000))
        dropped = drop_edge(edge_index, p=0.5)

        assert dropped.dim() == 2
        assert dropped.shape[0] == 2
        assert dropped.shape[1] > 0
        drop_ratio = 1.0 - dropped.shape[1] / edge_index.shape[1]
        assert 0.3 < drop_ratio < 0.7, f"drop_ratio={drop_ratio:.3f}，期望在 0.3~0.7"

    def test_drop_edge_zero_prob(self):
        """p=0 时不应丢弃任何边，返回原张量"""
        from iron_aging_gnn.graph.sampling import drop_edge

        edge_index = torch.randint(0, 50, (2, 200))
        dropped = drop_edge(edge_index, p=0.0)

        assert torch.equal(dropped, edge_index)

    def test_drop_edge_single_edge(self):
        """单条边 + p>0 时，边界情况：直接返回原张量"""
        from iron_aging_gnn.graph.sampling import drop_edge

        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        dropped = drop_edge(edge_index, p=0.5)

        assert torch.equal(dropped, edge_index)

    def test_drop_edge_shape_preserved(self):
        """drop_edge 输出形状第 0 维始终为 2"""
        from iron_aging_gnn.graph.sampling import drop_edge

        edge_index = torch.randint(0, 200, (2, 500))
        dropped = drop_edge(edge_index, p=0.3)

        assert dropped.shape[0] == 2

    def test_drop_edge_deterministic_with_seed(self):
        """相同 seed 下的结果应一致（通过 torch 全局 seed 控制）"""
        from iron_aging_gnn.graph.sampling import drop_edge

        edge_index = torch.randint(0, 100, (2, 200))

        torch.manual_seed(42)
        d1 = drop_edge(edge_index, p=0.3)
        torch.manual_seed(42)
        d2 = drop_edge(edge_index, p=0.3)

        assert torch.equal(d1, d2)


class TestSplitHeadTailNodes:
    """头尾节点划分：社区感知评分 + 头节点欠采样"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.train_compounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.compound_to_pos = {
            0: {10, 11, 12, 13, 14},
            1: {10, 11, 12, 13},
            2: {10, 11, 12},
            3: {10, 11},
            4: {10},
            5: {15, 16, 17, 18, 19, 20},
            6: {21},
            7: {22, 23},
            8: {24, 25, 26},
            9: set(),
        }

    def test_returns_two_lists(self):
        """split_head_tail_nodes 返回两个 list"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        pretrain, tail = split_head_tail_nodes(
            self.train_compounds,
            self.compound_to_pos,
            head_ratio=0.2,
            lambda_hhi=1.0,
            seed=42,
        )

        assert isinstance(pretrain, list)
        assert isinstance(tail, list)

    def test_head_ratio_controls_tail_count(self):
        """head_ratio 越小，tail 越大（更多节点被视为尾节点）"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        _, tail_small_head = split_head_tail_nodes(
            self.train_compounds,
            self.compound_to_pos,
            head_ratio=0.2,
            lambda_hhi=1.0,
            seed=42,
        )
        _, tail_large_head = split_head_tail_nodes(
            self.train_compounds,
            self.compound_to_pos,
            head_ratio=0.6,
            lambda_hhi=1.0,
            seed=42,
        )

        assert len(tail_small_head) >= len(tail_large_head)

    def test_tail_preserved_in_pretrain(self):
        """尾节点应全部保留在 pretrain_compounds 中"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        pretrain, tail = split_head_tail_nodes(
            self.train_compounds,
            self.compound_to_pos,
            head_ratio=0.2,
            lambda_hhi=1.0,
            seed=42,
        )

        for t in tail:
            assert t in pretrain, f"尾节点 {t} 未出现在 pretrain_compounds 中"

    def test_pretrain_length_consistent(self):
        """pretrain = tail + 欠采样头节点，长度应 <= 原始长度"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        pretrain, tail = split_head_tail_nodes(
            self.train_compounds,
            self.compound_to_pos,
            head_ratio=0.2,
            lambda_hhi=1.0,
            seed=42,
        )

        assert len(pretrain) <= len(self.train_compounds)
        assert len(tail) <= len(pretrain)

    def test_seed_reproducibility(self):
        """相同 seed 产生相同结果"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        p1, t1 = split_head_tail_nodes(
            self.train_compounds, self.compound_to_pos,
            head_ratio=0.2, seed=42,
        )
        p2, t2 = split_head_tail_nodes(
            self.train_compounds, self.compound_to_pos,
            head_ratio=0.2, seed=42,
        )

        assert p1 == p2
        assert t1 == t2

    def test_head_undersample_ratio(self):
        """head_undersample_ratio 控制头节点保留比例"""
        from iron_aging_gnn.graph.split import split_head_tail_nodes

        pretrain_keep_more, _ = split_head_tail_nodes(
            self.train_compounds, self.compound_to_pos,
            head_ratio=0.2, head_undersample_ratio=0.9, seed=42,
        )
        pretrain_keep_less, _ = split_head_tail_nodes(
            self.train_compounds, self.compound_to_pos,
            head_ratio=0.2, head_undersample_ratio=0.3, seed=42,
        )

        assert len(pretrain_keep_more) >= len(pretrain_keep_less)


class TestSplitTrainVal:
    """训练/验证集拆分：化合物随机拆分 + 蛋白分层拆分"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.all_compounds = list(range(100))
        self.all_proteins = list(range(100, 200))
        self.cpi_proteins = set(range(100, 150))

    def test_returns_correct_types(self):
        """split_train_val 返回两个 list + 两个 set"""
        from iron_aging_gnn.graph.split import split_train_val

        train_c, val_c, train_p, val_p = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            val_compound_ratio=0.2,
            val_protein_ratio=0.2,
            seed=42,
        )

        assert isinstance(train_c, list)
        assert isinstance(val_c, list)
        assert isinstance(train_p, set)
        assert isinstance(val_p, set)

    def test_compound_ratio_respected(self):
        """化合物拆分比例应大致正确"""
        from iron_aging_gnn.graph.split import split_train_val

        val_ratio = 0.2
        train_c, val_c, _, _ = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            val_compound_ratio=val_ratio,
            val_protein_ratio=0.2,
            seed=42,
        )

        expected_val = int(len(self.all_compounds) * val_ratio)
        expected_train = len(self.all_compounds) - expected_val

        assert len(val_c) == expected_val
        assert len(train_c) == expected_train

    def test_no_overlap_compounds(self):
        """训练集和验证集化合物不应重叠"""
        from iron_aging_gnn.graph.split import split_train_val

        train_c, val_c, _, _ = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            seed=42,
        )

        assert set(train_c).isdisjoint(set(val_c))

    def test_no_overlap_proteins(self):
        """训练集和验证集蛋白不应重叠"""
        from iron_aging_gnn.graph.split import split_train_val

        _, _, train_p, val_p = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            seed=42,
        )

        assert train_p.isdisjoint(val_p)

    def test_all_compounds_covered(self):
        """所有化合物必须出现在 train 或 val 中"""
        from iron_aging_gnn.graph.split import split_train_val

        train_c, val_c, _, _ = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            seed=42,
        )

        covered = set(train_c) | set(val_c)
        assert covered == set(self.all_compounds)

    def test_seed_reproducibility(self):
        """相同 seed 产生相同拆分结果"""
        from iron_aging_gnn.graph.split import split_train_val

        result1 = split_train_val(
            self.all_compounds, self.all_proteins,
            self.cpi_proteins, seed=42,
        )
        result2 = split_train_val(
            self.all_compounds, self.all_proteins,
            self.cpi_proteins, seed=42,
        )

        assert result1[0] == result2[0]
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]
        assert result1[3] == result2[3]

    def test_cpi_proteins_in_val(self):
        """验证集应包含有 CPI 交互的蛋白"""
        from iron_aging_gnn.graph.split import split_train_val

        _, _, _, val_p = split_train_val(
            self.all_compounds,
            self.all_proteins,
            self.cpi_proteins,
            val_protein_ratio=0.3,
            seed=42,
        )

        val_cpi = val_p & self.cpi_proteins
        assert len(val_cpi) > 0, "验证集应包含至少一个 CPI 蛋白"

    def test_different_seeds_different_splits(self):
        """不同 seed 产生不同拆分"""
        from iron_aging_gnn.graph.split import split_train_val

        _, val_c1, _, _ = split_train_val(
            self.all_compounds, self.all_proteins,
            self.cpi_proteins, seed=42,
        )
        _, val_c2, _, _ = split_train_val(
            self.all_compounds, self.all_proteins,
            self.cpi_proteins, seed=99,
        )

        assert val_c1 != val_c2
