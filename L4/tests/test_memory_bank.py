"""MemoryBank 单元测试

测试 FIFO 嵌入队列的初始化、更新、采样、容量环绕等行为。
"""

import pytest
import torch

from iron_aging_gnn.models.memory_bank import MemoryBank


class TestMemoryBankInit:
    """初始化测试"""

    def test_default_init(self):
        """默认参数初始化"""
        bank = MemoryBank()
        assert bank.max_size == 8192
        assert bank.out_dim == 64
        assert bank.device == "cpu"
        assert bank.ptr == 0
        assert bank.full is False
        assert bank.bank.shape == (8192, 64)

    def test_custom_init(self):
        """自定义参数初始化"""
        bank = MemoryBank(max_size=1024, out_dim=128, device="cpu")
        assert bank.max_size == 1024
        assert bank.out_dim == 128
        assert bank.device == "cpu"
        assert bank.ptr == 0
        assert bank.full is False
        assert bank.bank.shape == (1024, 128)

    def test_bank_initialized_with_zeros(self):
        """bank 初始值全为零"""
        bank = MemoryBank(max_size=100, out_dim=32)
        assert torch.all(bank.bank == 0.0)


class TestMemoryBankUpdate:
    """update() 与 size() 测试"""

    @pytest.mark.parametrize("n_embeddings", [1, 5, 10, 50])
    def test_update_increases_size(self, n_embeddings):
        """更新后 size() 反映嵌入数量"""
        bank = MemoryBank(max_size=100, out_dim=16)
        emb = torch.randn(n_embeddings, 16)
        bank.update(emb)
        assert bank.size() == n_embeddings
        assert bank.ptr == n_embeddings

    def test_update_multiple_batches(self):
        """多次更新后 size 累加"""
        bank = MemoryBank(max_size=100, out_dim=8)
        for i in range(3):
            emb = torch.randn(10, 8)
            bank.update(emb)
        assert bank.size() == 30
        assert bank.ptr == 30

    def test_update_empty_embeddings(self):
        """零个嵌入更新时状态不变"""
        bank = MemoryBank(max_size=100, out_dim=8)
        emb = torch.randn(10, 8)
        bank.update(emb)
        ptr_before = bank.ptr
        size_before = bank.size()
        bank.update(torch.zeros(0, 8))
        assert bank.ptr == ptr_before
        assert bank.size() == size_before

    def test_update_fills_to_capacity(self):
        """填满容量后 full=True, size()==max_size"""
        bank = MemoryBank(max_size=20, out_dim=4)
        emb = torch.randn(20, 4)
        bank.update(emb)
        assert bank.full is True
        assert bank.size() == 20
        assert bank.ptr == 0  # 恰好填满，ptr 回到 0

    def test_update_wraps_around(self):
        """超过容量时环绕写入，始终 full=True, size()==max_size"""
        bank = MemoryBank(max_size=20, out_dim=4)
        # 先填满
        bank.update(torch.randn(20, 4))
        assert bank.full is True
        # 再写入，环绕
        bank.update(torch.randn(5, 4))
        assert bank.full is True
        assert bank.size() == 20
        assert bank.ptr == 5

    def test_update_wraps_large_batch(self):
        """单次写入超过容量，截断后保留最后 max_size 个"""
        bank = MemoryBank(max_size=10, out_dim=4)
        emb = torch.randn(25, 4)
        bank.update(emb)
        assert bank.full is True
        assert bank.size() == 10
        assert bank.ptr == 0  # 截断后 (0 + 10) % 10

    def test_update_detaches_embeddings(self):
        """update 存储的是 detached 副本，不保留梯度"""
        emb = torch.randn(5, 8, requires_grad=True)
        bank = MemoryBank(max_size=10, out_dim=8)
        bank.update(emb)
        # bank 中的值不应有 grad
        assert bank.bank[:5].grad is None
        assert not bank.bank[:5].requires_grad


class TestMemoryBankSample:
    """sample() 测试"""

    def test_sample_shape(self):
        """采样返回正确形状"""
        bank = MemoryBank(max_size=100, out_dim=32)
        bank.update(torch.randn(50, 32))
        sampled = bank.sample(10)
        assert sampled.shape == (10, 32)

    def test_sample_all_available(self):
        """采样全部可用嵌入"""
        bank = MemoryBank(max_size=50, out_dim=16)
        bank.update(torch.randn(30, 16))
        sampled = bank.sample(30)
        assert sampled.shape == (30, 16)

    def test_sample_caps_at_available(self):
        """请求超过可用数量时，返回可用数量（不抛异常）"""
        bank = MemoryBank(max_size=100, out_dim=8)
        bank.update(torch.randn(5, 8))
        sampled = bank.sample(20)
        assert sampled.shape == (5, 8)

    def test_sample_caps_at_max_size(self):
        """填满后请求超过 max_size，返回 max_size 个"""
        bank = MemoryBank(max_size=10, out_dim=8)
        bank.update(torch.randn(10, 8))
        sampled = bank.sample(50)
        assert sampled.shape == (10, 8)

    def test_sample_empty_bank(self):
        """空 bank 采样返回零行张量"""
        bank = MemoryBank(max_size=50, out_dim=16)
        sampled = bank.sample(10)
        assert sampled.shape == (0, 16)

    def test_sample_randomness(self):
        """两次采样可能返回不同索引（验证随机性）"""
        bank = MemoryBank(max_size=100, out_dim=4)
        bank.update(torch.randn(100, 4))
        torch.manual_seed(0)
        s1 = bank.sample(50)
        torch.manual_seed(1)
        s2 = bank.sample(50)
        # 不同种子采样的结果不应完全相同
        assert not torch.allclose(s1, s2)


class TestMemoryBankVariedConfigs:
    """不同维度和容量配置测试"""

    @pytest.mark.parametrize("max_size,out_dim", [
        (1, 1),
        (10, 2),
        (100, 64),
        (500, 128),
        (2048, 256),
    ])
    def test_init_and_update(self, max_size, out_dim):
        """不同配置下初始化与更新正常"""
        bank = MemoryBank(max_size=max_size, out_dim=out_dim)
        n = min(5, max_size)
        emb = torch.randn(n, out_dim)
        bank.update(emb)
        assert bank.size() == n
        sampled = bank.sample(n)
        assert sampled.shape == (n, out_dim)

    def test_dimension_1(self):
        """dim=1 边界情况"""
        bank = MemoryBank(max_size=10, out_dim=1)
        emb = torch.randn(5, 1)
        bank.update(emb)
        assert bank.size() == 5
        sampled = bank.sample(3)
        assert sampled.shape == (3, 1)

    def test_capacity_1(self):
        """capacity=1 边界情况"""
        bank = MemoryBank(max_size=1, out_dim=8)
        emb = torch.randn(1, 8)
        bank.update(emb)
        assert bank.size() == 1
        sampled = bank.sample(1)
        assert sampled.shape == (1, 8)
        # 再次写入，环绕
        bank.update(torch.randn(1, 8))
        assert bank.size() == 1
        assert bank.full is True


class TestMemoryBankFIFOWrap:
    """FIFO 环绕写入正确性测试"""

    def test_wrap_overwrites_oldest(self):
        """环绕后最老的嵌入被覆盖"""
        bank = MemoryBank(max_size=5, out_dim=2)
        # 写入 [0,1,2,3,4]
        first_batch = torch.tensor([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])
        bank.update(first_batch)
        # 再写入 [5,6]，应覆盖 [0,1]
        second_batch = torch.tensor([
            [5.0, 5.0],
            [6.0, 6.0],
        ])
        bank.update(second_batch)
        # bank 现在应包含 [5,6,2,3,4]，ptr=2
        expected = torch.tensor([
            [5.0, 5.0],
            [6.0, 6.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ])
        assert torch.allclose(bank.bank, expected)

    def test_wrap_single_batch_exceeds_capacity(self):
        """单次写入超过容量，保留最后 max_size 个"""
        bank = MemoryBank(max_size=3, out_dim=1)
        emb = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        bank.update(emb)
        # 截断保留最后 3 个：[3,4,5]，ptr=0
        expected = torch.tensor([[3.0], [4.0], [5.0]])
        assert torch.allclose(bank.bank, expected)
        assert bank.ptr == 0
        assert bank.full is True


class TestMemoryBankEdgeCases:
    """边界与异常情况测试"""

    def test_size_zero_after_init(self):
        """初始化后 size() 为 0"""
        bank = MemoryBank(max_size=100, out_dim=16)
        assert bank.size() == 0

    def test_sample_zero_from_empty(self):
        """空 bank 采样 0 个返回空张量"""
        bank = MemoryBank(max_size=50, out_dim=8)
        sampled = bank.sample(0)
        assert sampled.shape == (0, 8)

    def test_sample_zero_from_filled(self):
        """有数据的 bank 采样 0 个返回空张量"""
        bank = MemoryBank(max_size=10, out_dim=4)
        bank.update(torch.randn(5, 4))
        sampled = bank.sample(0)
        assert sampled.shape == (0, 4)

    def test_full_flag_after_fill(self):
        """恰好填满时 full=True"""
        bank = MemoryBank(max_size=10, out_dim=4)
        bank.update(torch.randn(5, 4))
        assert bank.full is False
        bank.update(torch.randn(5, 4))
        assert bank.full is True

    def test_full_flag_after_overfill(self):
        """超过容量后 full=True"""
        bank = MemoryBank(max_size=10, out_dim=4)
        bank.update(torch.randn(12, 4))
        assert bank.full is True

    def test_device_cpu(self):
        """验证张量在 CPU 上"""
        bank = MemoryBank(max_size=10, out_dim=4, device="cpu")
        emb = torch.randn(3, 4)
        bank.update(emb)
        sampled = bank.sample(2)
        assert sampled.device.type == "cpu"