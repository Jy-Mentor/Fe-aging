"""损失函数模块单元测试

测试 focal_loss_with_logits, infonce_loss, _CpiLossState 等。
"""

import numpy as np
import torch


class TestFocalLossWithLogits:
    """Focal Loss 基础函数测试"""

    def test_perfect_prediction_low_loss(self):
        """完美预测时损失应接近零"""
        from iron_aging_gnn.models.losses import focal_loss_with_logits
        logits = torch.tensor([5.0, -5.0, 5.0, -5.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.75)
        assert loss.item() < 0.01

    def test_worst_prediction_high_loss(self):
        """完全错误预测时损失应较高"""
        from iron_aging_gnn.models.losses import focal_loss_with_logits
        logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = focal_loss_with_logits(logits, targets, gamma=2.0, alpha=0.75)
        assert loss.item() > 1.0

    def test_gamma_zero_equals_bce(self):
        """gamma=0, alpha=0.5 时等价于标准 BCE（alpha_t=0.5 对正负样本等权）"""
        from iron_aging_gnn.models.losses import focal_loss_with_logits
        logits = torch.tensor([1.0, -1.0, 0.5, -0.5])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss_focal = focal_loss_with_logits(logits, targets, gamma=0.0, alpha=0.5)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        assert abs(loss_focal.item() * 2 - bce.item()) < 1e-5

    def test_alpha_weights_positive_class(self):
        """alpha=0.75 时正样本损失贡献更大（验证 alpha_t 权重方向正确）"""
        from iron_aging_gnn.models.losses import focal_loss_with_logits
        logits_pos = torch.tensor([0.0])
        targets_pos = torch.tensor([1.0])
        loss_pos = focal_loss_with_logits(logits_pos, targets_pos, gamma=0.0, alpha=0.75)
        logits_neg = torch.tensor([0.0])
        targets_neg = torch.tensor([0.0])
        loss_neg = focal_loss_with_logits(logits_neg, targets_neg, gamma=0.0, alpha=0.75)
        assert loss_pos.item() > loss_neg.item()

    def test_gradient_flow(self):
        """梯度可以正常反向传播"""
        from iron_aging_gnn.models.losses import focal_loss_with_logits
        logits = torch.tensor([0.5, -0.5], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])
        loss = focal_loss_with_logits(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


class TestInfoNCELoss:
    """InfoNCE 对比损失测试"""

    def test_positive_only(self):
        """正样本远大于负样本（温度=1.0）时损失应大于0"""
        from iron_aging_gnn.models.losses import infonce_loss
        pos = torch.tensor([2.0, 3.0])
        neg = torch.tensor([[0.1], [0.2]])
        loss = infonce_loss(pos, neg, temperature=1.0)
        assert loss.item() > 0

    def test_high_pos_low_neg(self):
        """高正样本相似度 + 低负样本相似度时损失小"""
        from iron_aging_gnn.models.losses import infonce_loss
        pos = torch.tensor([5.0])
        neg = torch.tensor([[-5.0, -5.0, -5.0]])
        loss = infonce_loss(pos, neg, temperature=1.0)
        assert loss.item() < 0.01

    def test_low_pos_high_neg(self):
        """低正样本相似度 + 高负样本相似度时损失大"""
        from iron_aging_gnn.models.losses import infonce_loss
        pos = torch.tensor([-5.0])
        neg = torch.tensor([[5.0, 5.0, 5.0]])
        loss = infonce_loss(pos, neg, temperature=1.0)
        assert loss.item() > 10.0

    def test_with_memory_scores(self):
        """memory bank 负样本纳入分母"""
        from iron_aging_gnn.models.losses import infonce_loss
        pos = torch.tensor([1.0])
        neg = torch.tensor([[-1.0]])
        mem = torch.tensor([[2.0]])
        loss_no_mem = infonce_loss(pos, neg, temperature=1.0)
        loss_with_mem = infonce_loss(pos, neg, memory_scores=mem, temperature=1.0)
        assert loss_with_mem > loss_no_mem

    def test_gradient_flow(self):
        """梯度可以正常反向传播"""
        from iron_aging_gnn.models.losses import infonce_loss
        pos = torch.tensor([0.5], requires_grad=True)
        neg = torch.tensor([[0.1, 0.2]], requires_grad=True)
        loss = infonce_loss(pos, neg)
        loss.backward()
        assert pos.grad is not None
        assert not torch.isnan(pos.grad).any()


class TestCpiLossState:
    """CPI 损失状态管理测试"""

    def test_initial_state(self):
        from iron_aging_gnn.models.losses import _CpiLossState
        state = _CpiLossState()
        assert state.nan_batch_counter == 0
        assert state.pos_oom_counter == 0
        assert state.hard_neg_oom_counter == 0
        assert state.bpr_oom_counter == 0

    def test_counter_increment(self):
        from iron_aging_gnn.models.losses import _CpiLossState
        state = _CpiLossState()
        state.nan_batch_counter += 1
        state.pos_oom_counter += 1
        assert state.nan_batch_counter == 1
        assert state.pos_oom_counter == 1

    def test_independent_instances(self):
        from iron_aging_gnn.models.losses import _CpiLossState
        s1 = _CpiLossState()
        s2 = _CpiLossState()
        s1.nan_batch_counter = 5
        assert s2.nan_batch_counter == 0

    def test_default_state_exists(self):
        from iron_aging_gnn.models.losses import _default_cpi_loss_state
        assert _default_cpi_loss_state is not None
