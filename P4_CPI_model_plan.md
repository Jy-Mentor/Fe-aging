﻿﻿﻿# Phase 4：CPI 模型训练与中药单体-核心靶标穷举预测

> **核心定位：** Phase 4 是本项目的最关键阶段。使用公开数据库（ChEMBL/BindingDB/DrugBank）的实验验证CPI数据训练归纳式图神经网络模型，然后对核心靶标 × 中药单体做穷举预测。

---

## 一、核心原理澄清

### 1.1 训练数据 ≠ 核心靶标数据

| 概念 | 来源 | 规模 |
|------|------|------|
| **训练集** | ChEMBL/BindingDB/DrugBank 中所有人类蛋白的实验验证CPI数据 | >10,000 对 |
| **预测目标** | Phase 1 筛选出的核心靶标（ACSL4、HMOX1 等 5-15 个） | 5-15 个蛋白 |
| **候选化合物** | Phase 3 构建的中药单体库 | ~5,000 个单体 |

训练数据来自公开数据库，**不依赖核心靶标的已知配体**。核心靶标只有 1 个也不影响模型训练。

### 1.2 冷启动问题

核心靶标（如 ACSL4、HMOX1）在 ChEMBL 训练集中可能只有极少量甚至没有已知配体。模型训练时没见过这些靶标的结合模式，预测时可能不准。

**这正是选择 CPI-IGAE（归纳式 GNN）的原因：**
- 通过**配体集表征蛋白**（蛋白 = 其已知配体的集合）
- 归纳式图聚合器使模型能泛化到训练集未见过的蛋白
- 结合蛋白序列嵌入（ESM-2）、PPI 网络关系增强泛化能力

---

## 二、方法基础：CPI-IGAE

### 2.1 论文来源

万晓喆 (2022). 基于图神经网络的化合物-蛋白质相互作用研究. 中国科学院上海药物研究所博士学位论文.

### 2.2 核心创新

1. **加权同质图：** 将异质的化合物-蛋白关系图转化为同质图，用配体集表征蛋白，有利于图中信息交互
2. **归纳式图神经网络：** 使用 GraphSAGE 风格的邻居采样和聚合，赋予模型处理冷启动问题的能力
3. **端到端学习：** 直接从加权同质图中学习节点嵌入，通过点积解码预测化合物-蛋白相互作用

### 2.3 仓库地址

https://github.com/wanxiaozhe/CPI-IGAE

### 2.4 技术栈

- DGL (Deep Graph Library) + PyTorch
- GraphSAGE 归纳式图卷积
- 邻居采样 + 负采样训练
- 蛋白 = 配体集表征 + 氨基酸组成 + ESM-2 嵌入

---

## 三、Phase 4 工作流

### 步骤 4.1：从 ChEMBL 收集 CPI 训练数据

**数据来源：**
- ChEMBL v34（REST API）：筛选 `standard_type` in {IC50, Ki, Kd}，`standard_relation` = '='，`standard_value` <= 10,000 nM (10 uM)
- 靶标限定：Homo sapiens，`target_type` = 'SINGLE PROTEIN'
- 提取字段：化合物 ChEMBL ID、SMILES、靶标 UniProt ID、活性值、活性类型

**数据量估算：**
- ChEMBL 中 IC50/Ki/Kd <= 10uM 的人类蛋白-化合物对 > 100,000 条
- 去重后预计 50,000-80,000 对

**输出文件：**
- `P4/data/chembl_cpi_raw.csv`：原始CPI数据
- `P4/data/chembl_cpi_processed.csv`：去重、标准化后的训练数据
- `P4/data/compound_smiles.csv`：化合物SMILES字典
- `P4/data/protein_sequences.fasta`：靶标蛋白序列

### 步骤 4.2：构建加权同质图

**图构建方法（CPI-IGAE方式）：**

```
节点 = 化合物 + 蛋白（所有节点同质化）
边 = 实验验证的CPI关系（权重 = -log10(活性值/1e9)）
蛋白特征 = 其配体集（已知结合化合物的分子指纹均值）
化合物特征 = ECFP4 分子指纹（2048位）
```

**关键设计：**
- 蛋白用其配体集表征，而非直接用蛋白序列。这使得模型天然支持冷启动：新蛋白只要有配体集（来自数据库背景知识），就能生成有效表征
- 对于核心靶标（冷启动），其配体集来自 TCMSP/HERB 等数据库的计算预测关联（不作为训练标签，仅用于生成蛋白初始表征）

**输出文件：**
- `P4/data/cpi_graph.bin`：DGL 异质图二进制文件
- `P4/data/node_features.npy`：节点特征矩阵
- `P4/data/edge_weights.npy`：边权重

### 步骤 4.3：训练归纳式 GNN 模型

**模型架构：**

```
输入：加权同质图（化合物 + 蛋白节点）
|-- 图卷积层 x 3（GraphSAGE 归纳式聚合）
|   |-- 邻居采样：fanout = [25, 15, 10]
|   |-- 聚合函数：mean pool
|   |-- 激活函数：ReLU
|-- 边预测解码器：节点嵌入点积 -> sigmoid
|-- 损失函数：BCE Loss + 负采样（1:3）
```

**训练参数：**
- batch_size = 512
- epochs = 200（早停 patience=30）
- learning_rate = 0.001（Adam 优化器）
- 负采样比例 = 1:3
- 训练/验证/测试 = 8:1:1

**评估指标：**
- AUROC（ROC曲线下面积）
- AUPRC（精确率-召回率曲线下面积，对不平衡数据更敏感）
- Accuracy（阈值=0.5）

**冷启动验证：**
- 按蛋白划分测试集（确保测试蛋白不在训练集中出现）
- 验证模型对未见蛋白的泛化能力

**输出文件：**
- `P4/models/best_model.pth`：最佳模型权重
- `P4/results/training_log.csv`：训练日志
- `P4/results/evaluation_metrics.json`：评估指标

### 步骤 4.4：核心靶标 x 中药单体穷举预测

**预测流程：**

1. 加载 Phase 3 的中药单体库（SMILES + 分子指纹）
2. 加载 Phase 1 的核心靶标列表（UniProt ID + 蛋白序列）
3. 为每个核心靶标生成蛋白表征（配体集 + 序列特征）
4. 构建预测图：核心靶标节点 + 中药单体节点
5. 使用训练好的 CPI-IGAE 模型推理
6. 输出每个化合物-靶标对的预测得分（0-1）

**预测规模：**
- 核心靶标：5-15 个
- 中药单体：~5,000 个
- 总预测对数：25,000-75,000 对

**输出文件：**
- `P4/results/cpi_predictions.csv`：所有预测结果
- `P4/results/top_predictions.csv`：Top 200 预测结果

### 步骤 4.5：结果分析与 Top-N 候选分子筛选

**分析维度：**

1. **每个核心靶标的 Top 20 预测化合物**
2. **多靶标命中分析**：同时被多个核心靶标预测为阳性的化合物
3. **与已知活性化合物的结构相似性**（Tanimoto 系数）
4. **化合物-靶标-通路网络分析**
5. **类药性 + BBB 穿透性交叉筛选**

**输出文件：**
- `P4/results/per_target_top20.csv`
- `P4/results/multi_target_hits.csv`
- `P4/results/phase4_summary_report.md`

---

## 四、目录结构

```
P4/
|-- data/
|   |-- chembl_cpi_raw.csv
|   |-- chembl_cpi_processed.csv
|   |-- compound_smiles.csv
|   |-- protein_sequences.fasta
|   |-- cpi_graph.bin
|   |-- node_features.npy
|   |-- edge_weights.npy
|-- models/
|   |-- best_model.pth
|-- results/
|   |-- training_log.csv
|   |-- evaluation_metrics.json
|   |-- cpi_predictions.csv
|   |-- top_predictions.csv
|   |-- per_target_top20.csv
|   |-- multi_target_hits.csv
|   |-- phase4_summary_report.md
|-- scripts/
|   |-- download_chembl.py
|   |-- build_cpi_graph.py
|   |-- train_cpi_model.py
|   |-- predict_core_targets.py
|   |-- analyze_results.py
|-- logs/
|   |-- download_chembl.log
|   |-- build_cpi_graph.log
|   |-- train_cpi_model.log
|   |-- predict_core_targets.log
|   |-- analyze_results.log
```

---

## 五、反造假约束（Phase 4 专用）

1. **训练数据必须来自真实API下载**，禁止生成模拟CPI数据
2. **ChEMBL API 调用失败时**，记录日志并重试（最多3次），不可用假数据填充
3. **模型预测结果必须真实**，不可伪造高分
4. **冷启动验证**：必须按蛋白划分测试集，验证泛化性能
5. **所有中间文件**：记录文件大小、行数、MD5校验

---

## 六、后续衔接

- Phase 4 产出的 Top 候选化合物 -> Phase 5 分子对接 + MD 验证
- Phase 4 的蛋白表征（ESM-2 嵌入）-> Phase 5 结合口袋分析
- Phase 4 预测得分 -> Phase 5 多策略加权打分

---

*文档版本: v1.0 | 创建日期: 2026-06-22*
