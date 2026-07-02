# 铁衰老 GNN 研究模块 — Code Wiki

> 项目：基于拓扑-语义双分支图神经网络的铁衰老药物靶标预测  
> 版本：v2.1.0  
> 最后更新：2026-06-28

---

## 1. 项目概述

本仓库实现了一套面向铁衰老（iron aging）中药单体筛选的端到端计算流程，核心由四个层级（L1–L4）组成：

- **L1**：从 GEO 转录组数据出发，识别铁衰老差异表达基因与核心基因集。
- **L2**：蛋白特征工程（AAC / PseAAC / ESM-2）、单细胞评分、GSEA / GSVA 验证。
- **L3**：中药化合物库构建（TCMSP + PubChem SMILES 校验）、毒性过滤、化合物池综合评分。
- **L4**：基于 **GraphSAGE** 与 **Heterogeneous Graph Transformer (HGT)** 的双分支图神经网络，预测化合物-蛋白相互作用（CPI），并在严格蛋白冷启动设定下评估模型泛化能力。

---

## 2. 仓库结构

```text
.
├── L1/                     # 转录组分析与核心基因识别
│   ├── parse_geo_datasets.py
│   ├── core_gene_intersection.py
│   ├── expand_ppi_network.py
│   ├── go_kegg_enrichment.R
│   ├── wgcna_analysis.R
│   └── results/            # DE 结果、核心基因、PPI 网络
├── L2/                     # 蛋白特征与多组学验证
│   ├── protein_features.py
│   ├── sc_pipeline.py
│   ├── bulk_gsva.R
│   ├── add_missing_protein_features.py
│   └── results/            # 蛋白特征、GSEA、单细胞评分
├── L3/                     # 中药化合物库构建
│   ├── scripts/
│   │   ├── build_comprehensive_pool.py
│   │   ├── fix_smiles.py
│   │   ├── toxicity_filter.py
│   │   └── prepare_model_input_pool.py
│   ├── data/               # TCMSP 原始数据、PubChem 缓存
│   └── results/            # 化合物池、指纹、描述符
├── L4/                     # GNN 模型与预测
│   ├── src/iron_aging_gnn/ # 核心 Python 包
│   │   ├── data/           # 特征工程、加载、自检
│   │   ├── graph/          # 图构建、采样、拆分、验证安全图
│   │   ├── models/         # SAGE / HGT / Loss / MemoryBank
│   │   ├── utils/          # 配置、设备、日志、种子、可复现性
│   │   ├── prediction/
│   │   └── training/
│   ├── scripts/
│   │   ├── phase4_v10_minibatch.py   # 主训练脚本（v23）
│   │   ├── validate_p4_inputs_v3.py  # 输入验证
│   │   ├── ablation_study.py
│   │   ├── stat_significance.py
│   │   └── complementarity_analysis.py
│   ├── configs/default.yaml
│   ├── results_v10_minibatch/        # 当前主结果目录
│   └── results_v4* ~ v6* /           # 历史结果目录（保留用于追溯）
├── tests/                  # 单元测试（新增）
├── pyproject.toml          # 项目元数据、依赖、工具配置
├── requirements.txt
├── requirements-dev.txt
├── ruff.toml
├── pytest.ini
└── CODE_WIKI.md            # 本文档
```

---

## 3. 核心模块职责

### 3.1 `L4/src/iron_aging_gnn/data/`

| 文件 | 职责 |
|------|------|
| `constants.py` | 铁衰老基因列表、RDKit 描述符名、ECFP4 位数、默认随机种子 |
| `features.py` | 化合物特征（ECFP4 + MACCS + RDKit 描述符）、AAC / ESM-2 蛋白特征计算 |
| `loader.py` | 数据集加载与 CPI / PPI 数据预处理 |
| `self_check.py` | 训练前管线自检：SMILES 有效性、CPI 重复、特征 NaN、温靶标数量 |

### 3.2 `L4/src/iron_aging_gnn/graph/`

| 文件 | 职责 |
|------|------|
| `build.py` | 构建同质图 / 异质图、邻接表、通路 one-hot 特征、预计算全图边索引 |
| `sampling.py` | DropEdge、同质图 / 异质图邻居采样、种子固定 |
| `split.py` | 头尾节点划分（近似 HHI）、化合物 / 蛋白冷启动拆分 |
| `validation_graphs.py` | 构建验证安全图：常规验证图、化合物冷启动图、蛋白冷启动图 |

### 3.3 `L4/src/iron_aging_gnn/models/`

| 文件 | 职责 |
|------|------|
| `sage.py` | `SAGELinkPredictor`：GraphSAGE + 投影层 + MLP 解码器 |
| `hgt.py` | `HGTLinkPredictor`：HGTConv + 节点自适应门控 + 双线性解码器 |
| `losses.py` | Focal Loss、InfoNCE、共享 CPI 损失（Focal + BPR + 课程负采样） |
| `memory_bank.py` | 跨 batch 蛋白嵌入存储，供全局困难负样本采样 |

### 3.4 `L4/src/iron_aging_gnn/utils/`

| 文件 | 职责 |
|------|------|
| `config.py` | 基于 Pydantic v2 的层级配置系统，支持 YAML 加载与默认值合并 |
| `seed.py` | 全局随机种子固定，CuDNN / cuBLAS 确定性配置，DataLoader worker 种子 |
| `reproducibility.py` | 生成实验复现清单：Git 版本、依赖版本、数据校验和、配置快照 |
| `logging.py` | 统一 logger 工厂（文件 + 控制台） |
| `device.py` | 自动 GPU / CPU 设备选择 |

---

## 4. 关键类与函数说明

### 4.1 配置系统

```python
from L4.src.iron_aging_gnn.utils.config import Config, load_config

# 默认配置
cfg = Config()

# 从 YAML 加载（未指定字段使用默认值）
cfg = load_config("L4/configs/default.yaml")

# 解析后的绝对路径
paths = cfg.get_resolved_paths()
```

配置项覆盖：模型架构、SAGE/HGT 训练超参、两阶段迁移学习、课程负采样、损失函数、验证与预测、ESM-2 等。

### 4.2 随机种子与可复现性

```python
from L4.src.iron_aging_gnn.utils.seed import set_seed
from L4.src.iron_aging_gnn.utils.reproducibility import (
    generate_reproducibility_manifest,
    save_reproducibility_manifest,
)

set_seed(42, deterministic=True)

manifest = generate_reproducibility_manifest(
    project_root=".",
    config_path="L4/configs/default.yaml",
    data_files=["L3/results/tcm_compound_pool_filtered.csv"],
    seed=42,
)
save_reproducibility_manifest(manifest, "L4/results_v10_minibatch/reproducibility_manifest.json")
```

### 4.3 化合物特征

```python
from L4.src.iron_aging_gnn.data.features import build_compound_features

features, mean, std, col_mean = build_compound_features(["CCO", "c1ccccc1"])
```

### 4.4 图构建

```python
from L4.src.iron_aging_gnn.graph.build import build_graphs_and_adj

graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)
# 返回: x, feat_dim, prot_feat_dim, prot_esm_dim, n_compounds, n_proteins,
#       smi_to_idx, gene_to_idx, homo_adj, homo_edge_index,
#       hetero_adj, hetero_data, n_pathways, prot_to_path_neighbors
```

### 4.5 模型

```python
from L4.src.iron_aging_gnn.models.sage import SAGELinkPredictor
from L4.src.iron_aging_gnn.models.hgt import HGTLinkPredictor

sage = SAGELinkPredictor(
    comp_feat_dim=2048 + 167 + len(RDKIT_DESCRIPTOR_NAMES),
    prot_feat_dim=640,
    n_compounds=n_compounds,
    hidden_dim=64,
    out_dim=64,
    num_layers=2,
    dropout=0.5,
    n_pathways=n_pathways,
)

hgt = HGTLinkPredictor(
    hidden_dim=64,
    out_dim=64,
    num_heads=2,
    num_layers=2,
    dropout=0.5,
    metadata=(node_types, edge_types),
    compound_feat_dim=2048 + 167 + len(RDKIT_DESCRIPTOR_NAMES),
    node_feat_dims={"protein": 640, "pathway_count": n_pathways},
)
```

---

## 5. 数据流

```text
GEO 原始数据
    │
    ▼
L1/results/ferroaging_genes_96.csv ────────────────────┐
    │                                                   │
    ▼                                                   │
L2/results/target_protein_features.csv                 │
L2/results/protein_descriptors.csv                     │
L2/results/protein_pseaac.csv                          │
    │                                                   │
    ▼                                                   │
TCMSP 原始数据 + PubChem SMILES 校验                    │
    │                                                   │
    ▼                                                   │
L3/results/tcm_compound_pool_filtered.csv              │
L3/results/ecfp4_fingerprints.npy                      │
L3/results/rdkit_descriptors.csv                       │
    │                                                   │
    ▼                                                   │
L4/scripts/phase4_v10_minibatch.py ◄───────────────────┘
    │
    ├── 构建同质图 / 异质图
    ├── SAGE / HGT 双分支训练（两阶段迁移学习）
    ├── 严格蛋白冷启动验证
    └── 输出：model_performance_v23.csv / tcm_top_candidates_v23.csv
```

---

## 6. 运行方式

### 6.1 环境安装

```bash
# 仅运行依赖
pip install -r requirements.txt

# 开发依赖（含 ruff / pytest / mypy）
pip install -r requirements-dev.txt

# 或直接以可编辑模式安装
pip install -e ".[dev]"
```

### 6.2 输入验证

```bash
python L4/scripts/validate_p4_inputs_v3.py
```

输出：`L4/logs/input_validation_report_v3.json`、`L4/logs/input_checksums_v3.json`。

### 6.3 主训练流程

```bash
python L4/scripts/phase4_v10_minibatch.py
```

可通过环境变量覆盖默认配置（未来将进一步扩展 CLI 参数支持）：

```bash
# 示例：关闭 ESM-2 以使用 AAC 特征
# 修改 L4/configs/default.yaml 中 esm2.use_esm2: false 后运行
```

### 6.4 质量门禁

```bash
# 静态检查
ruff check .

# 单元测试
pytest

# 输入验证
python L4/scripts/validate_p4_inputs_v3.py
```

---

## 7. 测试体系

测试位于 `tests/` 目录，使用 pytest：

| 测试文件 | 覆盖内容 |
|----------|----------|
| `test_seed.py` | 随机种子固定、worker 种子独立性 |
| `test_config.py` | Pydantic 配置加载、合并、校验、路径解析 |
| `test_features.py` | 化合物特征形状与有效性、AAC 归一化 |
| `test_split.py` | 头尾节点划分、训练/验证集不相交性 |
| `test_reproducibility.py` | 复现清单字段、配置文件与数据校验和、保存 |

运行全部测试：

```bash
pytest -q
```

---

## 8. 学术规范与可复现性

本项目遵循以下学术规范实践：

1. **随机种子固定**：`utils.seed.set_seed` 固定 Python / NumPy / PyTorch，并启用 CuDNN deterministic 模式。
2. **配置快照**：`utils.config.load_config` 基于 Pydantic 严格校验，避免参数漂移。
3. **数据校验和**：`validate_p4_inputs_v3.py` 与 `reproducibility.py` 对关键输入文件生成 SHA-256 校验和。
4. **实验清单**：每次训练可生成 `reproducibility_manifest.json`，记录 Git commit、依赖版本、数据文件哈希、随机种子。
5. **严格验证协议**：区分常规验证图与蛋白冷启动验证图，避免信息泄漏；负样本排除正样本与 2-hop 邻居。
6. **消融实验与统计显著性**：`ablation_study.py` 与 `stat_significance.py` 支持系统化实验与 paired t-test。

---

## 9. 已知限制与待办

- `phase4_v10_minibatch.py` 仍为单文件大脚本，未来可进一步拆分为 `training/` 子模块。
- 主脚本 CLI 参数支持有限，当前主要通过 `L4/configs/default.yaml` 配置。
- `L4/results_v4* ~ v6*` 为历史结果目录，保留用于论文实验追溯；新实验统一输出到 `L4/results_v10_minibatch/`。
- `L4/papers/` 为文献下载与解析脚本集合，不属于核心训练流程。

---

## 10. 贡献指南

1. 修改代码前请先阅读相关模块文档与测试。
2. 新增功能必须附带单元测试。
3. 提交前运行 `ruff check .` 与 `pytest`。
4. 训练实验需附带 `reproducibility_manifest.json`。
5. 禁止在代码中使用 `try-except: pass` 静默吞异常；所有异常应记录日志并向上传播。
