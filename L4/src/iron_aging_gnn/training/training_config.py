from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

"""训练配置数据类 — 封装 train_sage / train_hgt 的 30+ 参数

使用方式:
    config = TrainingConfig()                    # 全默认值
    config = TrainingConfig(epochs=50, lr=1e-3)  # 覆盖部分参数
    config = TrainingConfig.from_config(cfg)     # 从 pydantic Config 构建
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..utils.config import Config as PydanticConfig


@dataclass
class TrainingConfig:
    """训练配置 — 封装所有 train_sage / train_hgt 共用参数。

    字段按功能分组，默认值与原有函数签名默认值保持一致。
    """

    # ============================================================
    # 基础训练参数
    # ============================================================
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    batch_size: int = 256
    random_seed: int = 42

    # ============================================================
    # 图采样参数
    # ============================================================
    num_neighbors: list[int] = field(default_factory=lambda: [32, 16])

    # ============================================================
    # 两阶段训练参数
    # ============================================================
    two_stage: bool = False
    pretrain_epochs: int = 0
    pretrain_lr: float | None = None
    head_ratio: float = 0.2
    lambda_hhi: float = 1.0
    head_undersample_ratio: float = 0.6
    pretrain_lr_multiplier: float = 1.5
    pretrain_lr_decay: float = 0.5

    # ============================================================
    # 损失函数参数
    # ============================================================
    use_infonce: bool = False
    use_bpr: bool = True
    bpr_weight: float = 0.4
    use_curriculum: bool = True
    use_topology_neg: bool = False
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75

    # ============================================================
    # 表型多任务参数
    # ============================================================
    pheno_lambda: float = 0.3
    pheno_compound_indices: list[int] | None = None
    pheno_labels: list[int] | None = None

    # ============================================================
    # 正则化 / 优化器参数
    # ============================================================
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.05

    # ============================================================
    # DropEdge 参数
    # ============================================================
    dropedge_ppi: float = 0.15
    dropedge_pathway: float = 0.10
    dropedge_cpi: float = 0.0

    # ============================================================
    # Memory Bank 参数
    # ============================================================
    memory_bank_size: int = 8192

    # ============================================================
    # 其他
    # ============================================================
    use_amp: bool = False

    @property
    def pheno_enabled(self) -> bool:
        """表型分类任务是否启用。"""
        return (
            self.pheno_compound_indices is not None
            and self.pheno_labels is not None
            and len(self.pheno_compound_indices) > 0
            and len(self.pheno_labels) > 0
        )

    @classmethod
    def from_config(cls, cfg: PydanticConfig) -> TrainingConfig:
        """从 pydantic Config 对象构建 TrainingConfig。

        Args:
            cfg: 项目 pydantic 配置实例。

        Returns:
            TrainingConfig 实例，字段值取自 cfg 对应子配置。
        """
        return cls(
            # 基础训练
            epochs=cfg.sage.epochs,
            lr=cfg.sage.lr,
            patience=cfg.sage.patience,
            batch_size=cfg.sage.batch_size,
            random_seed=cfg.random_seed,
            num_neighbors=list(cfg.sage.num_neighbors),
            # 两阶段训练
            two_stage=cfg.sage.two_stage,
            pretrain_epochs=cfg.sage.pretrain_epochs,
            pretrain_lr=cfg.sage.pretrain_lr,
            head_ratio=cfg.two_stage.head_ratio,
            lambda_hhi=cfg.two_stage.lambda_hhi,
            head_undersample_ratio=cfg.two_stage.head_undersample_ratio,
            pretrain_lr_multiplier=cfg.two_stage.pretrain_lr_multiplier,
            pretrain_lr_decay=cfg.two_stage.pretrain_lr_decay,
            # 损失函数
            use_infonce=False,
            use_bpr=True,
            bpr_weight=cfg.loss.bpr_weight,
            use_curriculum=True,
            use_topology_neg=False,
            focal_gamma=cfg.loss.focal_gamma,
            focal_alpha=cfg.loss.focal_alpha,
            # 表型多任务
            pheno_lambda=cfg.training.pheno_lambda,
            # 正则化
            weight_decay=cfg.training.weight_decay,
            grad_clip_norm=cfg.training.grad_clip_norm,
            warmup_ratio=cfg.training.warmup_ratio,
            # DropEdge
            dropedge_ppi=cfg.training.dropedge_ppi,
            dropedge_pathway=cfg.training.dropedge_pathway,
            dropedge_cpi=0.0,
            # Memory Bank
            memory_bank_size=cfg.memory_bank.memory_bank_size,
            # 其他
            use_amp=False,
        )

    def to_dict(self) -> dict:
        """转换为字典，便于传递给函数。"""
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "num_neighbors": self.num_neighbors,
            "random_seed": self.random_seed,
            "two_stage": self.two_stage,
            "pretrain_epochs": self.pretrain_epochs,
            "pretrain_lr": self.pretrain_lr,
            "head_ratio": self.head_ratio,
            "lambda_hhi": self.lambda_hhi,
            "head_undersample_ratio": self.head_undersample_ratio,
            "pretrain_lr_multiplier": self.pretrain_lr_multiplier,
            "pretrain_lr_decay": self.pretrain_lr_decay,
            "use_infonce": self.use_infonce,
            "use_bpr": self.use_bpr,
            "bpr_weight": self.bpr_weight,
            "use_curriculum": self.use_curriculum,
            "use_topology_neg": self.use_topology_neg,
            "focal_gamma": self.focal_gamma,
            "focal_alpha": self.focal_alpha,
            "pheno_lambda": self.pheno_lambda,
            "pheno_compound_indices": self.pheno_compound_indices,
            "pheno_labels": self.pheno_labels,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip_norm,
            "warmup_ratio": self.warmup_ratio,
            "dropedge_ppi": self.dropedge_ppi,
            "dropedge_pathway": self.dropedge_pathway,
            "dropedge_cpi": self.dropedge_cpi,
            "memory_bank_size": self.memory_bank_size,
            "use_amp": self.use_amp,
        }