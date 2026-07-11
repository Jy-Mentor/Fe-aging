"""Pipeline 工具函数 — GPU 监控、OOM 处理、梯度检查、张量诊断"""

from __future__ import annotations

import logging
import time
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

def get_prot_feat_dim(prot_feat) -> int | None:
    """提取蛋白特征维度，兼容 numpy ndarray / torch Tensor / dict。"""
    if hasattr(prot_feat, "shape") and len(prot_feat.shape) >= 2:
        return int(prot_feat.shape[-1])
    if isinstance(prot_feat, dict):
        for v in prot_feat.values():
            if hasattr(v, "shape") and len(v.shape) >= 1:
                return int(v.shape[-1])
    return None

def check_gpu_memory(min_free_gb: float = 1.0) -> bool:
    """检查 GPU 可用显存是否充足。"""
    if not torch.cuda.is_available():
        return True
    free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
    return free_mem >= min_free_gb

def log_gpu_memory(tag: str = "") -> None:
    """记录 GPU 显存使用情况。"""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / (1024 ** 3)
    total_gb = total_mem / (1024 ** 3)
    logger.info(f"  GPU Memory [{tag}]: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, free={free_gb:.2f}GB/{total_gb:.2f}GB")

def log_step_time(start_time: float, step_name: str) -> float:
    """记录步骤耗时并返回当前时间。"""
    elapsed = time.time() - start_time
    logger.info(f"  [{step_name}] 耗时: {elapsed:.2f}s")
    return time.time()

def handle_oom_and_retry(model, optimizer, scaler, batch_seeds, loss_fn, max_retries=3):
    """OOM 降级重试：减小 batch 后重试前向传播。"""
    if not torch.cuda.is_available():
        raise RuntimeError("OOM on CPU — 不应发生")
    for attempt in range(max_retries):
        try:
            torch.cuda.empty_cache()
            reduced_batch = batch_seeds[: max(1, len(batch_seeds) // (2 ** attempt))]
            return loss_fn(reduced_batch)
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"  OOM 重试 {attempt+1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                raise
    return None

def check_gradient_norm(model: nn.Module, warn_threshold: float = 100.0) -> float:
    """计算模型总梯度范数，超过阈值时警告。"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    if total_norm > warn_threshold:
        logger.warning(f"梯度范数异常: {total_norm:.1f} > {warn_threshold:.1f}")
    return total_norm