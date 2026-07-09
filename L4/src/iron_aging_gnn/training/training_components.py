"""训练组件模块 — 可复用的训练辅助类

Classes:
    Validator: 验证逻辑、早停、最佳模型管理
    MemoryBankManager: Memory Bank 更新与全局刷新
    GradientMonitor: 梯度检查与裁剪
    LRSchedulerFactory: 学习率调度器工厂
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn

from ..models.memory_bank import MemoryBank

logger = logging.getLogger(__name__)


class Validator:
    """验证器 — 封装验证逻辑、早停与最佳模型管理。

    支持化合物冷启动验证（AUC / AUPR），可选表型分类 AUC。
    """

    def __init__(
        self,
        validate_fn,
        patience: int = 15,
        val_freq: int = 2,
        pretrain_val_freq: int = 5,
    ):
        """初始化验证器。

        Args:
            validate_fn: 验证函数，签名 (model, x, edge_index, val_compounds,
                         compound_to_pos, n_compounds) -> dict
            patience: 早停耐心值
            val_freq: 微调阶段验证频率（每 N epoch）
            pretrain_val_freq: 预训练阶段验证频率
        """
        self.validate_fn = validate_fn
        self.patience = patience
        self.val_freq = val_freq
        self.pretrain_val_freq = pretrain_val_freq

        self.best_val_auc: float = 0.0
        self.best_val_aupr: float = 0.0
        self.best_state: dict | None = None
        self.patience_counter: int = 0

    def should_validate(self, epoch: int, is_pretrain: bool = False) -> bool:
        """判断是否需要验证。"""
        freq = self.pretrain_val_freq if is_pretrain else self.val_freq
        return epoch % freq == 0

    def validate_sage(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        val_compounds: list[int],
        compound_to_pos: dict,
        n_compounds: int,
        use_amp: bool = True,
    ) -> dict:
        """SAGE 验证 — 全图前向 + 验证函数。"""
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            metrics = self.validate_fn(
                model, x, edge_index,
                val_compounds, compound_to_pos, n_compounds,
            )
        model.train()
        return metrics

    def update_best(self, current_aupr: float, current_auc: float = 0.0) -> bool:
        """更新最佳指标并返回是否为新最佳。

        v38: 早停基于 val_aupr（化合物冷启动），替代原蛋白冷启动 AUPR。
        """
        if current_auc > self.best_val_auc:
            self.best_val_auc = current_auc

        if current_aupr > self.best_val_aupr:
            self.best_val_aupr = current_aupr
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            return False

    def capture_best_state(self, model: nn.Module) -> None:
        """捕获当前模型参数为最佳状态。"""
        self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def load_best_state(self, model: nn.Module) -> bool:
        """加载最佳模型参数。返回是否成功。"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            return True
        return False

    def should_stop_early(self) -> bool:
        """判断是否应该早停。"""
        return self.patience_counter >= self.patience

    def reset(self) -> None:
        """重置验证器状态（用于新训练阶段）。"""
        self.best_val_auc = 0.0
        self.best_val_aupr = 0.0
        self.best_state = None
        self.patience_counter = 0

    def get_best_entry(self, history: list[dict]) -> dict:
        """从历史记录中获取最佳条目。"""
        if not history:
            return {"auc": 0.0, "aupr": 0.0}
        return max(history, key=lambda x: x.get("aupr", 0))


class MemoryBankManager:
    """Memory Bank 管理器 — 封装更新与全局刷新。

    全局刷新在每 N epoch 执行一次，将全图训练蛋白嵌入推入 Memory Bank。
    """

    def __init__(
        self,
        memory_bank_size: int = 8192,
        out_dim: int = 64,
        device: str = "cuda",
        refresh_freq: int = 5,
    ):
        """初始化 Memory Bank 管理器。

        Args:
            memory_bank_size: Memory Bank 最大容量。
            out_dim: 蛋白嵌入维度。
            device: 张量存储设备。
            refresh_freq: 全局刷新频率（每 N epoch）。
        """
        self.memory_bank = MemoryBank(
            max_size=memory_bank_size,
            out_dim=out_dim,
            device=device,
        )
        self.refresh_freq = refresh_freq
        self.memory_bank_size = memory_bank_size
        self.out_dim = out_dim
        self.device = device

    def update(self, prot_emb: torch.Tensor) -> None:
        """更新 Memory Bank（每个 batch 调用）。"""
        self.memory_bank.update(prot_emb.detach())

    def should_refresh(self, epoch: int) -> bool:
        """判断是否需要全局刷新。"""
        return epoch % self.refresh_freq == 0

    def refresh_global_sage(
        self,
        model: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        n_compounds: int,
        val_proteins: set | None = None,
        use_amp: bool = True,
    ) -> None:
        """SAGE 全局刷新 — 全图前向，收集训练蛋白嵌入。"""
        model.eval()
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                # v54-fix: 调用方现在传入 CPU 张量，刷新前再移动到 GPU
                x_dev = x.to(self.device)
                edge_index_dev = edge_index.to(self.device)
                full_node_emb = model(x_dev, edge_index_dev, n_compounds=n_compounds)
                # v64: 全图前向传播完成后立即释放输入特征矩阵和边索引，减少峰值显存
                del x_dev, edge_index_dev
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                full_prot_emb = full_node_emb[n_compounds:]
                if val_proteins is not None and len(val_proteins) > 0:
                    train_prot_mask = torch.ones(
                        full_prot_emb.shape[0], dtype=torch.bool,
                        device=full_prot_emb.device,
                    )
                    train_prot_mask[list(val_proteins)] = False
                    full_prot_emb = full_prot_emb[train_prot_mask]
                self.memory_bank = MemoryBank(
                    max_size=self.memory_bank_size,
                    out_dim=self.out_dim,
                    device=self.device,
                )
                self.memory_bank.update(full_prot_emb)
            logger.info(f"  SAGE Memory Bank 全局刷新: {self.memory_bank.size()} 训练蛋白嵌入")
        finally:
            model.train()

    def refresh_global_hgt(
        self,
        model: nn.Module,
        hetero_data,
        val_proteins: set | None = None,
        use_amp: bool = True,
    ) -> None:
        """HGT 全局刷新 — 全图前向，收集训练蛋白嵌入。"""
        model.eval()
        try:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
                hgt_out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
                full_prot_emb = hgt_out["protein"]
                if val_proteins is not None and len(val_proteins) > 0:
                    train_prot_mask = torch.ones(
                        full_prot_emb.shape[0], dtype=torch.bool,
                        device=full_prot_emb.device,
                    )
                    train_prot_mask[list(val_proteins)] = False
                    full_prot_emb = full_prot_emb[train_prot_mask]
                self.memory_bank = MemoryBank(
                    max_size=self.memory_bank_size,
                    out_dim=self.out_dim,
                    device=self.device,
                )
                self.memory_bank.update(full_prot_emb)
            logger.info(f"  HGT Memory Bank 全局刷新: {self.memory_bank.size()} 训练蛋白嵌入")
        finally:
            model.train()


class GradientMonitor:
    """梯度监控器 — 梯度范数检查与裁剪。"""

    def __init__(self, grad_clip_norm: float = 1.0, warn_threshold: float = 100.0):
        """初始化梯度监控器。

        Args:
            grad_clip_norm: 梯度裁剪最大范数。
            warn_threshold: 梯度范数警告阈值。
        """
        self.grad_clip_norm = grad_clip_norm
        self.warn_threshold = warn_threshold

    def check_and_clip(self, model: nn.Module, scaler: torch.amp.GradScaler | None = None, optimizer: torch.optim.Optimizer | None = None) -> float:
        """检查梯度范数并按 grad_clip_norm 裁剪（CPU 计算避免 GPU OOM）。

        Args:
            model: 待检查的模型。
            scaler: AMP GradScaler 实例（可选）。若提供则先 unscale。
            optimizer: 优化器实例（可选）。scaler.unscale_ 需要。

        Returns:
            梯度总范数。
        """
        if scaler is not None:
            scaler.unscale_(optimizer if optimizer is not None else model.parameters())

        # 先处理 NaN/Inf
        n_sanitized = self._sanitize_nan_gradients(model)
        if n_sanitized > 0:
            logger.warning(f"梯度 NaN/Inf 已清零: {n_sanitized} 个参数")

        # CPU 安全梯度裁剪 — 避免 clip_grad_norm_ 的 GPU 级联范数开销
        total_norm = self._safe_clip_grad_norm_(model, self.grad_clip_norm)

        if total_norm > self.warn_threshold and not math.isnan(total_norm):
            logger.warning(f"梯度范数异常: {total_norm:.1f} > {self.warn_threshold:.1f}")

        return total_norm

    @staticmethod
    def _safe_clip_grad_norm_(model: nn.Module, max_norm: float) -> float:
        """CPU 安全梯度裁剪 — 逐参数计算范数，避免 GPU 大张量级联。

        Args:
            model: 待裁剪的模型。
            max_norm: 最大梯度范数。

        Returns:
            梯度总范数。
        """
        total_sq_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm_sq = p.grad.data.detach().cpu().pow(2).sum().item()
                total_sq_norm += param_norm_sq
        total_norm = total_sq_norm ** 0.5
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale)
        return total_norm

    @staticmethod
    def _sanitize_nan_gradients(model: nn.Module) -> int:
        """将梯度中的 NaN/Inf 值清零，防止优化器状态损坏。

        Returns:
            被清零的参数数量。
        """
        n_sanitized = 0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    p.grad = torch.where(
                        torch.isnan(p.grad) | torch.isinf(p.grad),
                        torch.zeros_like(p.grad),
                        p.grad,
                    )
                    n_sanitized += 1
        return n_sanitized

    def _compute_norm(self, model: nn.Module) -> float:
        """计算模型梯度总范数（CPU 计算以避免 GPU OOM）。"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.detach().cpu().norm(2).item()
                total_norm += param_norm ** 2
        return total_norm ** 0.5


class LRSchedulerFactory:
    """学习率调度器工厂 — 创建 warmup + cosine 衰减调度器。"""

    @staticmethod
    def create_cosine_warmup(
        optimizer: torch.optim.Optimizer,
        epochs: int,
        warmup_ratio: float = 0.05,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """创建 warmup + cosine 衰减调度器。

        Args:
            optimizer: 优化器实例。
            epochs: 总 epoch 数。
            warmup_ratio: warmup 占比。

        Returns:
            LambdaLR 调度器。
        """
        warmup_epochs = max(1, int(epochs * warmup_ratio))
        warmup_epochs = min(warmup_epochs, epochs)

        def lr_lambda(e: int) -> float:
            if warmup_epochs > 0 and e < warmup_epochs:
                return e / warmup_epochs
            if epochs <= warmup_epochs:
                return 1.0
            progress = (e - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress)) * 1.0 + 1e-6

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def create_linear_decay(
        optimizer: torch.optim.Optimizer,
        epochs: int,
        decay_rate: float = 0.5,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """创建线性衰减调度器（用于预训练阶段）。

        Args:
            optimizer: 优化器实例。
            epochs: 总 epoch 数。
            decay_rate: 最终衰减到初始学习率的比例。

        Returns:
            LambdaLR 调度器。
        """
        def lr_lambda(e: int) -> float:
            return 1.0 - decay_rate * (e / epochs)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def create_plateau(
        optimizer: torch.optim.Optimizer,
        patience: int = 2,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        mode: str = "max",
        metric_name: str = "aupr",
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        """创建 ReduceLROnPlateau 调度器（用于微调阶段抑制过拟合）。

        Args:
            optimizer: 优化器实例。
            patience: 验证指标未提升的耐心 epoch 数。
            factor: 学习率衰减系数。
            min_lr: 最小学习率。
            mode: "max" 表示指标越大越好（AUC/AUPR），"min" 表示越小越好（loss）。
            metric_name: 监控的指标名称（仅用于日志）。

        Returns:
            ReduceLROnPlateau 调度器。
        """
        logger.info(
            f"  微调阶段使用 ReduceLROnPlateau: mode={mode}, patience={patience}, "
            f"factor={factor}, min_lr={min_lr}, metric={metric_name}"
        )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
        )