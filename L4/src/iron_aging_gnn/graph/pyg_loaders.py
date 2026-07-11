"""PyG 标准 Loader 封装：NeighborLoader (同构图) + HGTLoader (异构图)。

提供与自定义采样器兼容的接口，通过配置开关切换。
保留自定义采样器作为 fallback，确保向后兼容。

用法:
    from iron_aging_gnn.graph.pyg_loaders import create_homo_loader, create_hetero_loader

    # 同构图
    loader = create_homo_loader(data, input_nodes=seed_compounds, num_neighbors=[16, 8])

    # 异构图
    loader = create_hetero_loader(hetero_data, input_nodes=("compound", seed_compounds),
                                  num_neighbors={"compound": [16, 8], "protein": [16, 8]})

参考:
  - PyG NeighborLoader: https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html
  - PyG HGTLoader: https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader

logger = logging.getLogger(__name__)


def create_homo_loader(
    data: Data,
    input_nodes: torch.Tensor | list[int],
    num_neighbors: list[int] | None = None,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> NeighborLoader:
    """创建同构图 NeighborLoader。

    Args:
        data: PyG Data 对象（同构图）
        input_nodes: 种子节点索引（训练化合物）
        num_neighbors: 每层采样邻居数，默认 [16, 8]
        batch_size: 每个 batch 的种子节点数
        shuffle: 是否 shuffle
        num_workers: 数据加载 worker 数
        **kwargs: 传递给 NeighborLoader 的额外参数

    Returns:
        NeighborLoader 实例
    """
    if num_neighbors is None:
        num_neighbors = [16, 8]

    if isinstance(input_nodes, list):
        input_nodes = torch.tensor(input_nodes, dtype=torch.long)

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=min(batch_size, len(input_nodes)),
        input_nodes=input_nodes,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    logger.info(
        f"NeighborLoader 创建: input_nodes={len(input_nodes)}, "
        f"num_neighbors={num_neighbors}, batch_size={batch_size}"
    )
    return loader


def create_hetero_loader(
    data: HeteroData,
    input_nodes: tuple[str, torch.Tensor | list[int]],
    num_neighbors: dict[str, list[int]] | None = None,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> HGTLoader:
    """创建异构图 HGTLoader。

    Args:
        data: PyG HeteroData 对象
        input_nodes: (node_type, indices) 种子节点
        num_neighbors: {node_type: [samples_per_layer]} 每层每类型采样邻居数
        batch_size: 每个 batch 的种子节点数
        shuffle: 是否 shuffle
        num_workers: 数据加载 worker 数
        **kwargs: 传递给 HGTLoader 的额外参数

    Returns:
        HGTLoader 实例
    """
    if num_neighbors is None:
        num_neighbors = {
            "compound": [16, 8],
            "protein": [16, 8],
            "pathway": [4, 2],
            "disease": [2, 1],
        }

    node_type, indices = input_nodes
    if isinstance(indices, list):
        indices = torch.tensor(indices, dtype=torch.long)

    loader = HGTLoader(
        data,
        num_samples=num_neighbors,
        batch_size=min(batch_size, len(indices)),
        input_nodes=(node_type, indices),
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
    logger.info(
        f"HGTLoader 创建: input_nodes=({node_type}, {len(indices)}), "
        f"num_neighbors={num_neighbors}, batch_size={batch_size}"
    )
    return loader


def benchmark_loader_throughput(
    loader,
    n_batches: int = 20,
    warmup: int = 3,
    device: torch.device | None = None,
) -> dict[str, float]:
    """基准测试 Loader 吞吐量。

    Args:
        loader: NeighborLoader 或 HGTLoader 实例
        n_batches: 测试 batch 数
        warmup: 预热 batch 数
        device: 目标设备，None 则使用 CPU

    Returns:
        {
            "batches_per_second": float,
            "samples_per_second": float,
            "avg_batch_time_ms": float,
            "total_time_s": float,
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预热
    for i, batch in enumerate(loader):
        if i >= warmup:
            break
        if isinstance(batch, HeteroData):
            batch = batch.to(device)
        else:
            batch = batch.to(device)

    # 计时
    total_samples = 0
    times: list[float] = []
    t0 = time.perf_counter()

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        t_batch_start = time.perf_counter()
        if isinstance(batch, HeteroData):
            batch = batch.to(device)
            if hasattr(batch, "num_nodes"):
                total_samples += sum(batch.num_nodes.values())
            elif "compound" in batch:
                total_samples += batch["compound"].x.shape[0]
        else:
            batch = batch.to(device)
            total_samples += batch.x.shape[0] if hasattr(batch, "x") else 0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_batch_end = time.perf_counter()
        times.append((t_batch_end - t_batch_start) * 1000)

    total_time = time.perf_counter() - t0
    n_actual = len(times)

    result = {
        "batches_per_second": n_actual / total_time if total_time > 0 else 0.0,
        "samples_per_second": total_samples / total_time if total_time > 0 else 0.0,
        "avg_batch_time_ms": sum(times) / n_actual if n_actual > 0 else 0.0,
        "total_time_s": total_time,
    }

    logger.info(
        f"Loader 基准测试: {result['batches_per_second']:.1f} batches/s, "
        f"{result['samples_per_second']:.0f} samples/s, "
        f"avg {result['avg_batch_time_ms']:.1f} ms/batch"
    )
    return result


def compare_loaders(
    custom_loader_fn,
    pyg_loader_fn,
    custom_loader_kwargs: dict | None = None,
    pyg_loader_kwargs: dict | None = None,
    n_batches: int = 20,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """对比自定义 Loader 与 PyG Loader 的吞吐量。

    Args:
        custom_loader_fn: 自定义 Loader 创建函数
        pyg_loader_fn: PyG Loader 创建函数
        custom_loader_kwargs: 自定义 Loader 参数
        pyg_loader_kwargs: PyG Loader 参数
        n_batches: 测试 batch 数
        device: 目标设备

    Returns:
        {"custom": {...}, "pyg": {...}, "speedup": float}
    """
    custom_loader_kwargs = custom_loader_kwargs or {}
    pyg_loader_kwargs = pyg_loader_kwargs or {}

    logger.info("=" * 50)
    logger.info("Loader 吞吐量对比")
    logger.info("=" * 50)

    custom_loader = custom_loader_fn(**custom_loader_kwargs)
    custom_result = benchmark_loader_throughput(custom_loader, n_batches=n_batches, device=device)

    pyg_loader = pyg_loader_fn(**pyg_loader_kwargs)
    pyg_result = benchmark_loader_throughput(pyg_loader, n_batches=n_batches, device=device)

    speedup = pyg_result["batches_per_second"] / max(custom_result["batches_per_second"], 1e-6)
    logger.info(
        f"PyG Loader 加速比: {speedup:.2f}x "
        f"(custom: {custom_result['batches_per_second']:.1f} batches/s, "
        f"pyg: {pyg_result['batches_per_second']:.1f} batches/s)"
    )

    return {
        "custom": custom_result,
        "pyg": pyg_result,
        "speedup": speedup,
    }