"""测试随机种子与可复现性工具。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from L4.src.iron_aging_gnn.utils.seed import seed_worker, set_seed


def test_set_seed_fixes_python_numpy_torch():
    """设置种子后，Python / NumPy / PyTorch 随机数应可复现。"""
    set_seed(123, deterministic=True)
    py1 = [random.random() for _ in range(5)]
    np1 = np.random.rand(5).tolist()
    torch1 = torch.rand(5).tolist()

    set_seed(123, deterministic=True)
    py2 = [random.random() for _ in range(5)]
    np2 = np.random.rand(5).tolist()
    torch2 = torch.rand(5).tolist()

    assert py1 == py2
    assert np1 == np2
    assert torch1 == torch2


def test_set_seed_sets_environment():
    """set_seed 应设置 PYTHONHASHSEED 环境变量。"""
    set_seed(42)
    assert os.environ.get("PYTHONHASHSEED") == "42"


def test_seed_worker_deterministic():
    """seed_worker 应基于 worker_id 和 base_seed 生成确定性种子。"""
    seed_worker(0, base_seed=42)
    r0a = random.randint(0, 10_000_000)

    seed_worker(0, base_seed=42)
    r0b = random.randint(0, 10_000_000)

    seed_worker(1, base_seed=42)
    r1 = random.randint(0, 10_000_000)

    assert r0a == r0b
    assert r0a != r1
