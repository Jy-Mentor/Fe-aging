# 铁衰老 GNN — 中药单体脑缺血再灌注药物重定位

基于异构图神经网络（Heterogeneous GNN）的铁衰老-中药单体-脑缺血再灌注（CIRI）药物重定位系统。

## 项目简介

本项目构建了一个四模态异质图（化合物-蛋白-通路-疾病），利用 **SAGE + HGT + SimpleHGN + RGCN** 四分支集成模型预测中药单体活性成分对 CIRI 相关靶标的潜在治疗作用。核心创新包括：

- **残基感知双线性注意力解码器**（ResidueAwareBilinearDecoder）：在化合物-蛋白交互预测中引入蛋白残基级 ESM-2 嵌入，实现残基-原子级交互建模
- **四分支异构图集成**：同构图拓扑（SAGE）+ 异构图语义（HGT）+ 边类型感知（SimpleHGN）+ 关系卷积（RGCN）
- **铁死亡表型辅助任务**：多任务联合训练，增强化合物表征的铁死亡特异性
- **课程负采样 + Memory Bank**：从随机负采样逐步过渡到拓扑难负样本

## 目录结构

```
L4/
├── entry/                  # 入口脚本（训练/评估/预测/图构建）
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── build_graph.py
│   └── run_pipeline.py
├── src/iron_aging_gnn/     # 核心模块
│   ├── data/               # 数据加载与常量
│   ├── evaluation/         # 评估指标（AUC/AUPR/ROCE/BEDROC/EF/NDCG）
│   ├── graph/              # 图构建、采样、负采样、分割
│   ├── models/             # 模型定义（SAGE/HGT/SimpleHGN/RGCN + 解码器）
│   ├── prediction/         # TCM 预测推理
│   ├── training/           # 训练器、配置、组件
│   └── utils/              # 工具（配置、设备、日志、种子、可复现性）
├── scripts/                # 数据处理与实验脚本（80+）
├── configs/                # 配置文件
│   ├── default.yaml        # 主配置
│   ├── default_v30.yaml    # v30 配置快照
│   └── default_v28_backup.yaml
├── docs/                   # 文档
│   ├── cpi_dti_gnn_optimization_research_report_v2.md
│   ├── cpi_accuracy_optimization_report_v50.md
│   ├── cpi_dti_gnn_optimization_feasibility_assessment.md
│   ├── domain_adaptation_design.md
│   └── v24_quality_audit.md
├── tests/                  # 单元测试
├── requirements.txt        # Python 依赖
├── environment.yml         # Conda 环境
├── pyproject.toml          # 项目配置
└── README.md               # 本文件
```

## 快速开始

### 环境要求

- Python >= 3.10
- CUDA >= 12.0（GPU 训练）
- **显存**: 至少 8GB（RTX 50 系 8GB 模式下需注意显存管理）
- 操作系统: Windows / Linux

### 安装

```bash
# 克隆项目
cd L4

# 方式一：pip 安装
pip install -r requirements.txt

# 方式二：Conda 安装
conda env create -f environment.yml
conda activate iron_aging_gnn

# 方式三：开发模式安装
pip install -e .
```

### 数据准备

确保以下数据目录存在（由 L1/L2/L3 阶段生成）：

```
L1/results/    # 铁衰老核心基因集、PPI 网络
L2/results/    # 蛋白特征（ESM-2）、通路注释
L3/results/    # TCM 化合物库（SMILES、描述符）
```

### 训练

```bash
# 完整训练（SAGE + HGT + SimpleHGN + RGCN 四分支）
python entry/train.py --config configs/default.yaml

# 仅训练 SAGE 分支
python entry/train.py --config configs/default.yaml --model sage

# 两阶段迁移学习
python entry/train.py --config configs/default.yaml --two-stage
```

### 评估

```bash
# 评估已训练的模型
python entry/evaluate.py --checkpoint results_v10_minibatch/sage_best.pt

# 完整评估（含早期富集指标）
python entry/evaluate.py --checkpoint results_v10_minibatch/sage_best.pt --full
```

### 预测

```bash
# TCM 候选化合物预测
python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --tcm-pool L3/results/tcm_pool.csv

# 带不确定性估计（MC Dropout）
python entry/predict.py --checkpoint results_v10_minibatch/sage_best.pt --mc-samples 30
```

### 一键流水线

```bash
python entry/run_pipeline.py --config configs/default.yaml
```

## 模型架构

| 分支 | 类型 | 编码器 | 特点 |
|------|------|--------|------|
| SAGE | 同构图 | SAGEConv | 拓扑感知，AUPR ~0.80（最优） |
| HGT | 异构图 | HGTConv | 类型特定注意力 + 门控 |
| SimpleHGN | 异构图 | GATv2Conv + edge_embed | 边类型感知，无门控，稳定性好 |
| RGCN | 异构图 | RGCNConv | 关系特定权重聚合 |

四个分支共享 **ResidueAwareBilinearDecoder**（基于 GraphBAN 双线性注意力），支持四种解码模式：
- `mlp`: 拼接后 MLP
- `dot`: 点积
- `bilinear`: 低秩双线性
- `residue_bilinear`: 残基感知双线性注意力（推荐）

## 评估指标

- **AUROC / AUPR**: 分类性能
- **ROCE** (ROC Enrichment): 早期富集（0.5%/1%/2%/5% FPR）
- **BEDROC**: Boltzmann 增强判别
- **EF@X%**: 富集因子
- **Precision@K / Recall@K / NDCG@K**: 排名指标

## 硬件说明

- 当前在 RTX 50 系 8GB 显存上运行，需注意:
  - `torch.cuda.empty_cache()` 每 10 个 batch 调用一次（非每 batch）
  - 残基特征默认驻留在 CPU，按需搬入 GPU
  - 验证阶段使用全图前向传播（HGT/SimpleHGN），及时释放张量

## 引用

- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW
- Hamilton et al. (2017) "GraphSAGE", NeurIPS
- Hadipour et al. (2025) "GraphBAN", Nature Communications
- Schlichtkrull et al. (2018) "RGCN", ESWC
- Lv et al. (2021) "SimpleHGN", KDD
- Rives et al. (2021) "ESM-2", PNAS

## 常见问题

**Q: 训练时 CUDA OOM 怎么办？**
A: 减小 `batch_size`（config 中 `sage.batch_size` / `hgt.batch_size`），或减少 `num_neighbors`。

**Q: HGT 验证 AUPR = 1.0？**
A: 检查 config 中 `validation.hgt_val_use_residue_for_pos` 是否为 `false`。正样本残基分数与负样本 fast bilinear 分数分布不一致会导致虚高。

**Q: 如何复现 SAGE AUPR ~0.80？**
A: 使用 `configs/default.yaml`，`decoder_type: residue_bilinear`，`temperature: 1.0`，`random_seed: 42`。

## 更多文档

- [CPI/DTI GNN 优化研究报告](docs/cpi_dti_gnn_optimization_research_report_v2.md)
- [CPI 精度优化报告 v50](docs/cpi_accuracy_optimization_report_v50.md)
- [GNN 优化可行性评估](docs/cpi_dti_gnn_optimization_feasibility_assessment.md)
- [领域自适应设计](docs/domain_adaptation_design.md)
- [v24 质量审计](docs/v24_quality_audit.md)