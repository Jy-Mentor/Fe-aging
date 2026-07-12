"""随机种子可复现性测试

验证两次相同种子运行产生完全一致的结果。
"""

import random

import numpy as np
import torch
import torch.nn as nn


def _run_training(seed: int):
    """运行一个简化的训练循环，返回 loss 列表。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []

    for _ in range(5):
        x = torch.randn(16, 64)
        y = torch.randn(16, 1)
        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def test_identical_seeds_same_result():
    """两次相同种子运行应产生完全一致的 loss。"""
    losses_1 = _run_training(42)
    losses_2 = _run_training(42)
    assert losses_1 == losses_2, f"种子42两次运行不一致: {losses_1} != {losses_2}"


def test_different_seeds_different_result():
    """不同种子运行应产生不同的 loss。"""
    losses_1 = _run_training(42)
    losses_2 = _run_training(123)
    # 至少第一个 batch 的 loss 应该不同（非常高的概率）
    assert losses_1 != losses_2, "不同种子产生相同结果，随机性可能未生效"


def test_utils_seed_module():
    """验证 utils/seed.py 模块可正常导入和设置种子。"""
    from iron_aging_gnn.utils.seed import set_seed
    set_seed(42)
    x1 = torch.randn(5)
    set_seed(42)
    x2 = torch.randn(5)
    assert torch.equal(x1, x2), "set_seed 后随机数不一致"


def test_utils_reproducibility_import():
    """验证 reproducibility.py 模块可正常导入。"""
    from iron_aging_gnn.utils.reproducibility import (
        generate_reproducibility_manifest,
        save_reproducibility_manifest,
        export_environment_fingerprint,
    )
    assert callable(generate_reproducibility_manifest)
    assert callable(save_reproducibility_manifest)
    assert callable(export_environment_fingerprint)