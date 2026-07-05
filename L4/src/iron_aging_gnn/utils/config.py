"""铁衰老 GNN 项目配置系统
=======================
基于 pydantic 的严格类型配置类，支持从 YAML 文件加载配置并合并默认值。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _get_default_project_root() -> Path:
    """自动推断项目根目录：config.py 位于 L4/src/iron_aging_gnn/utils/，
    项目根目录为 L4 的父目录。
    """
    return Path(__file__).resolve().parent.parent.parent.parent.parent


_DEFAULT_FERRORAGING_GENES: list[str] = sorted([
    "ABCC1", "ACSL4", "ACVR1B", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
    "BRD7", "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
    "DPEP1", "DPP4", "DUOX1", "DYRK1A", "E2F1", "E2F3", "EBF3", "EDN1",
    "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1", "FBXO31", "FOSL1", "GMFB",
    "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1", "ICA1", "IFNG", "IGFBP7",
    "IL1B", "IL6", "IRF1", "IRF7", "IRF9", "KDM6B", "KEAP1", "KLF6",
    "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3", "MAP3K14", "MAPK1",
    "MAPK14", "MCU", "MEN1", "MPO", "NLRP3", "NOX4", "NR1D1", "NR2F2",
    "NUAK2", "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2", "RBM3",
    "RUNX3", "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5", "SMARCB1", "SMURF2",
    "SNCA", "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2", "TBX2", "TFRC",
    "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP", "WNT5A", "WWTR1", "YAP1", "ZEB1",
])


class PathConfig(BaseModel):
    """路径配置 — 所有路径均相对于 project_root 或为绝对路径。"""

    project_root: Path = Field(default_factory=_get_default_project_root)
    l1_results: Path = Field(default_factory=lambda: Path("L1/results"))
    l2_results: Path = Field(default_factory=lambda: Path("L2/results"))
    l3_results: Path = Field(default_factory=lambda: Path("L3/results"))
    l4_root: Path = Field(default_factory=lambda: Path("L4"))
    l4_results: Path = Field(default_factory=lambda: Path("L4/results_v10_minibatch"))
    l4_logs: Path = Field(default_factory=lambda: Path("L4/logs"))

    def resolve(self, base: Path | None = None) -> PathConfig:
        """将所有相对路径解析为绝对路径。"""
        root = base or self.project_root
        return PathConfig(
            project_root=root,
            l1_results=root / self.l1_results if not self.l1_results.is_absolute() else self.l1_results,
            l2_results=root / self.l2_results if not self.l2_results.is_absolute() else self.l2_results,
            l3_results=root / self.l3_results if not self.l3_results.is_absolute() else self.l3_results,
            l4_root=root / self.l4_root if not self.l4_root.is_absolute() else self.l4_root,
            l4_results=root / self.l4_results if not self.l4_results.is_absolute() else self.l4_results,
            l4_logs=root / self.l4_logs if not self.l4_logs.is_absolute() else self.l4_logs,
        )


class ModelConfig(BaseModel):
    """模型架构参数 — SAGE 和 HGT 共用。"""

    hidden_dim: int = Field(default=64, ge=16, le=512, description="隐藏层维度")
    out_dim: int = Field(default=64, ge=16, le=512, description="输出嵌入维度")
    num_layers: int = Field(default=2, ge=1, le=6, description="GNN 卷积层数")
    dropout: float = Field(default=0.5, ge=0.0, le=0.9, description="Dropout 比率")
    num_heads: int = Field(default=2, ge=1, le=16, description="HGT 注意力头数")
    # 投影器 Dropout（独立于主 Dropout，用于细粒度正则化控制）
    prot_proj_dropout: float = Field(default=0.4, ge=0.0, le=0.9, description="蛋白特征投影器外部 Dropout")
    prot_proj_inner_dropout: float = Field(default=0.3, ge=0.0, le=0.9, description="蛋白特征投影器内部 Dropout")
    pathway_proj_dropout: float = Field(default=0.3, ge=0.0, le=0.9, description="通路投影器 Dropout")
    pheno_head_dropout: float = Field(default=0.3, ge=0.0, le=0.9, description="表型分类头 Dropout")
    score_clamp: float = Field(default=10.0, ge=1.0, description="分数裁剪范围 [-score_clamp, score_clamp]")
    decoder_type: str = Field(default="mlp", description="解码器类型：mlp / dot / bilinear")
    temperature: float = Field(default=5.0, gt=0.0, description="解码温度系数 T，固定为 5.0 不参与梯度")


class SageConfig(BaseModel):
    """SAGE 分支训练超参数。"""

    epochs: int = Field(default=15, ge=1, description="训练轮数")
    lr: float = Field(default=5e-4, gt=0.0, description="学习率")
    patience: int = Field(default=5, ge=1, description="早停耐心值")
    batch_size: int = Field(default=256, ge=1, description="批次大小")
    num_neighbors: list[int] = Field(default=[32, 16], description="邻域采样邻居数（每层）")
    two_stage: bool = Field(default=True, description="是否启用两阶段迁移学习")
    pretrain_epochs: int = Field(default=10, ge=1, description="预训练轮数")
    pretrain_lr: float = Field(default=7.5e-4, gt=0.0, description="预训练学习率")


class HgtConfig(BaseModel):
    """HGT 分支训练超参数。"""

    epochs: int = Field(default=15, ge=1, description="训练轮数")
    lr: float = Field(default=1e-3, gt=0.0, description="学习率")
    patience: int = Field(default=5, ge=1, description="早停耐心值")
    batch_size: int = Field(default=128, ge=1, description="批次大小")
    num_neighbors: list[int] = Field(default=[32, 16], description="邻域采样邻居数（每层）")
    two_stage: bool = Field(default=True, description="是否启用两阶段迁移学习")
    pretrain_epochs: int = Field(default=10, ge=1, description="预训练轮数")
    pretrain_lr: float = Field(default=1.5e-3, gt=0.0, description="预训练学习率")


class TwoStageConfig(BaseModel):
    """两阶段迁移学习配置。"""

    head_ratio: float = Field(
        default=0.2, ge=0.05, le=0.5,
        description="头节点比例（度排名前 head_ratio 的节点）",
    )
    lambda_hhi: float = Field(
        default=1.0, ge=0.0, le=2.0,
        description="HHI 评分中拓扑集中度的权重",
    )
    head_undersample_ratio: float = Field(
        default=0.6, ge=0.1, le=1.0,
        description="头节点欠采样比例（保留比例）",
    )
    infonce_warmup_ratio: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="InfoNCE 预热比例（当前阶段总 epoch 的比例）",
    )
    pretrain_lr_multiplier: float = Field(
        default=1.5, ge=1.0, le=5.0,
        description="预训练学习率相对于主学习率的倍数",
    )
    pretrain_lr_decay: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="预训练线性衰减最终比例",
    )


class CurriculumConfig(BaseModel):
    """课程负采样配置。

    三阶段课程：
    - 阶段1 (前30% epoch): 仅随机负采样
    - 阶段2 (中40% epoch): 中度负采样（通路邻近蛋白作为负样本）
    - 阶段3 (后30% epoch): 极硬负采样（最高分非正样本作为负样本）
    """

    random_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="仅随机负采样阶段的 epoch 占比")
    moderate_ratio: float = Field(default=0.4, ge=0.0, le=1.0, description="中度负采样阶段的 epoch 占比")
    hard_ratio: float = Field(default=0.3, ge=0.0, le=1.0, description="极硬负采样阶段的 epoch 占比")
    moderate_pathway_weight: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="中度负采样阶段中，通路邻近负样本的化合物占比",
    )
    medium_neg_ratio: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="中度负样本占 unique 化合物的比例",
    )
    hard_neg_ratio: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="极硬负样本占 unique 化合物的比例",
    )


class LossConfig(BaseModel):
    """损失函数配置。"""

    focal_alpha: float = Field(default=0.75, ge=0.0, le=1.0, description="Focal Loss α 参数")
    focal_gamma: float = Field(default=2.0, ge=0.0, le=5.0, description="Focal Loss γ 参数")
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.5, description="标签平滑系数")
    label_smoothing_pos: float = Field(default=0.9, ge=0.5, le=1.0, description="正样本标签平滑目标")
    label_smoothing_neg: float = Field(default=0.1, ge=0.0, le=0.5, description="负样本标签平滑目标")
    bce_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="BCE 损失权重")
    bpr_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="BPR 排序损失权重")
    infonce_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="InfoNCE 对比损失权重")
    temperature: float = Field(default=5.0, gt=0.0, description="Sigmoid 温度参数 T")
    infonce_temperature: float = Field(default=0.07, gt=0.0, description="InfoNCE 对比损失温度 τ")


class ValidationConfig(BaseModel):
    """验证与预测配置。"""

    compound_split_ratio: float = Field(
        default=0.85, ge=0.5, le=1.0,
        description="化合物冷启动训练/验证拆分比例（训练比例）",
    )
    protein_cold_split_ratio: float = Field(
        default=0.50, ge=0.0, le=0.5,
        description="蛋白冷启动验证拆分比例（验证蛋白占比）",
    )
    mc_samples: int = Field(default=30, ge=0, le=100, description="MC Dropout 采样次数（0=禁用）")
    diversity_penalty: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="多样性惩罚系数（余弦相似度惩罚）",
    )
    hard_neg_top_k: int = Field(default=5, ge=1, description="验证时硬负样本采样数")
    rand_neg_top_k: int = Field(default=5, ge=1, description="验证时随机负样本采样数")
    prot_cold_neg_ratio: int = Field(default=10, ge=1, description="蛋白冷启动负样本相对正样本的倍数")
    val_batch_size: int = Field(default=512, ge=1, description="验证时批量评分")
    hgt_val_batch_size: int = Field(default=64, ge=1, description="HGT mini-batch 验证时每批化合物数")
    hgt_val_num_neighbors: list[int] = Field(default=[64, 32], description="HGT mini-batch 验证时邻居采样数")
    val_freq: int = Field(default=2, ge=1, description="验证频率（每 N epoch）")
    pretrain_val_freq: int = Field(default=5, ge=1, description="预训练阶段验证频率")
    mem_refresh_freq: int = Field(default=5, ge=1, description="Memory Bank 全局刷新频率")
    default_aupr: float = Field(default=0.5, ge=0.0, le=1.0, description="默认 AUPR（无验证数据时）")


class ESM2Config(BaseModel):
    """ESM-2 蛋白质语言模型配置。"""

    use_esm2: bool = Field(default=True, description="是否使用 ESM-2 预训练嵌入")
    model_name: str = Field(
        default="facebook/esm2_t30_150M_UR50D",
        description="HuggingFace ESM-2 模型名称",
    )
    esm_batch_size: int = Field(default=4, ge=1, description="ESM-2 推理批次大小")
    esm_max_len: int = Field(default=1022, ge=1, description="ESM-2 最大序列长度（含特殊 token 则为 1022 aa）")


class TrainingConfig(BaseModel):
    """共享训练超参数 — SAGE 和 HGT 共用。"""

    weight_decay: float = Field(default=1e-4, ge=0.0, le=0.1, description="AdamW 权重衰减")
    grad_clip_norm: float = Field(default=1.0, gt=0.0, description="梯度裁剪最大范数")
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=0.5, description="LR Warmup 占训练总 epoch 的比例")
    pheno_lambda: float = Field(default=0.05, ge=0.0, le=1.0, description="表型分类损失权重")
    dropedge_ppi: float = Field(default=0.15, ge=0.0, le=1.0, description="PPI 边 DropEdge 概率")
    dropedge_pathway: float = Field(default=0.10, ge=0.0, le=1.0, description="通路边 DropEdge 概率")
    flag_step: float = Field(default=0.01, ge=0.0, description="Gaussian Feature Augmentation 扰动幅度")


class MemoryBankConfig(BaseModel):
    """Memory Bank 配置。"""

    memory_bank_size: int = Field(default=8192, ge=1, description="Memory Bank 最大容量")
    infonce_mem_sample: int = Field(default=256, ge=1, description="InfoNCE 从 Memory Bank 采样数")


class PredictionConfig(BaseModel):
    """预测与排序配置。"""

    top_n_candidates: int = Field(default=500, ge=1, description="输出候选化合物数")
    warm_targets_top_n: int = Field(default=5, ge=1, description="top targets 中 warm 靶标数")
    zs_targets_top_n: int = Field(default=3, ge=1, description="top targets 中 zero-shot 靶标数")
    composite_avg_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="composite_score 中加权平均分权重")
    composite_max_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="composite_score 中加权最高分权重")
    composite_hits_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="composite_score 中加权命中数权重")
    ferro_factor_base: float = Field(default=0.7, ge=0.0, le=1.0, description="铁死亡概率融合因子基数")
    zs_bonus_max: float = Field(default=0.05, ge=0.0, le=1.0, description="zero-shot bonus 上限")
    tree_ensemble_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="树模型集成权重")


class NumericalConfig(BaseModel):
    """数值常量配置。"""

    mask_val: float = Field(default=-1e9, description="掩码值（屏蔽无效候选）")
    eps: float = Field(default=1e-8, gt=0.0, description="数值稳定 epsilon")
    eps_small: float = Field(default=1e-10, gt=0.0, description="小数 epsilon（用于 multinomial 分母保护）")


class NegativeSamplingConfig(BaseModel):
    """难负样本配置（PPI拓扑 + ESM-2结构相似性）。"""

    use_topology_neg: bool = Field(default=False, description="是否启用PPI拓扑难负样本")
    use_esm_similarity_neg: bool = Field(default=False, description="是否启用ESM-2余弦相似度难负样本")
    topo_neighbors_top_k: int = Field(default=50, ge=1, description="拓扑负样本每个蛋白保留候选数")
    esm_similarity_top_k: int = Field(default=50, ge=1, description="ESM-2相似度负样本每个蛋白保留候选数")


class Config(BaseModel):
    """铁衰老 GNN 项目总配置。

    使用方式：
        # 方式1：默认配置
        config = Config()

        # 方式2：从 YAML 加载
        config = load_config("configs/default.yaml")

        # 方式3：从 YAML 加载并覆盖部分参数
        config = load_config("configs/custom.yaml")
    """

    random_seed: int = Field(default=42, ge=0, description="全局随机种子")
    paths: PathConfig = Field(default_factory=PathConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sage: SageConfig = Field(default_factory=SageConfig)
    hgt: HgtConfig = Field(default_factory=HgtConfig)
    two_stage: TwoStageConfig = Field(default_factory=TwoStageConfig)
    curriculum: CurriculumConfig = Field(default_factory=CurriculumConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    esm2: ESM2Config = Field(default_factory=ESM2Config)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    memory_bank: MemoryBankConfig = Field(default_factory=MemoryBankConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    negative_sampling: NegativeSamplingConfig = Field(default_factory=NegativeSamplingConfig)
    ferrogenesis_genes: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_FERRORAGING_GENES),
        description="铁衰老靶标基因列表",
    )
    ferrogenesis_genes_csv: str | None = Field(
        default=None,
        description="铁衰老靶标基因 CSV 文件路径（相对于 project_root）",
    )

    def get_resolved_paths(self) -> PathConfig:
        """返回解析后的绝对路径配置。"""
        return self.paths.resolve()

    def get_l4_results_dir(self) -> Path:
        """返回 L4 结果目录（自动创建）。"""
        p = self.paths.resolve()
        d = p.l4_results
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_l4_logs_dir(self) -> Path:
        """返回 L4 日志目录（自动创建）。"""
        p = self.paths.resolve()
        d = p.l4_logs
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(config_path: str | None = None) -> Config:
    """从 YAML 文件加载配置，未指定的字段使用默认值。

    Args:
        config_path: YAML 配置文件路径。为 None 时返回全默认配置。

    Returns:
        Config: 合并后的配置实例。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。
        yaml.YAMLError: YAML 格式错误时抛出。
    """
    if config_path is None:
        return Config()

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    # utf-8-sig 自动去除 BOM，避免 Windows 下 BOM 导致 yaml 解析失败
    with config_file.open(encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return Config()

    # 逐层合并：对每个子配置节点，展开为 kwargs 传入子类
    merged = _deep_merge_defaults(raw)

    return Config(**merged)


def _deep_merge_defaults(raw: dict) -> dict:
    """将 YAML 原始数据与默认配置子类合并，支持嵌套结构。

    YAML 中的路径字段如为相对路径字符串，则保持为字符串；
    在 Config 创建时由 pydantic 自动转换为 Path。

    Args:
        raw: 从 YAML 加载的原始字典。

    Returns:
        合并后的字典，可直接传入 Config(**merged)。
    """
    # 子配置映射：YAML 键名 -> 默认配置类
    sub_configs = {
        "paths": PathConfig,
        "model": ModelConfig,
        "sage": SageConfig,
        "hgt": HgtConfig,
        "two_stage": TwoStageConfig,
        "curriculum": CurriculumConfig,
        "loss": LossConfig,
        "validation": ValidationConfig,
        "esm2": ESM2Config,
        "training": TrainingConfig,
        "memory_bank": MemoryBankConfig,
        "prediction": PredictionConfig,
        "numerical": NumericalConfig,
    }

    merged = {}
    for key, value in raw.items():
        if key in sub_configs and isinstance(value, dict):
            # 子配置：用默认值填充缺失字段
            default_cls = sub_configs[key]
            default_dict = default_cls().model_dump()
            default_dict.update(value)
            merged[key] = default_dict
        else:
            merged[key] = value

    return merged
