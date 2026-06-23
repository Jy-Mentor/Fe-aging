# Phase 4 逻辑处理优化方案

> **任务**：针对当前 P4 v4 模型在输入验证、逻辑判断、结果可靠性方面的显著缺陷，参考 CPI-IGAE 文献与 GitHub 开源实现，进行系统性优化。
> **范围**：严格限定在 P4（化合物-蛋白相互作用机器学习筛选），不进入 P5/P6。
> **版本**：v4.5（v4 的逻辑加固与可靠性提升版）

---

## 一、当前状态与问题诊断

### 1.1 已运行结果（v4）

| 指标 | 数值 | 说明 |
|------|------|------|
| TCM 候选化合物 | 1491 | 来自 L3 过滤后的单体库 |
| 目标靶标 | 28 | Phase 1/2 核心基因 |
| 可训练靶标 | 4 | PTGS2、MAPK1、DYRK1A、CTSB |
| 无活性数据靶标 | 11 | EMP1、SAT1 等 |
| 预测记录 | 41,748 | 28 × 1491 |
| EF@1%/5%/10% | 431.55 / 100.53 / 50.24 | 显著偏高，存在逻辑问题 |

### 1.2 已参考的外部资源

| 资源 | 地址 | 参考价值 |
|------|------|----------|
| CPI-IGAE 官方仓库 | https://github.com/wanxiaozhe/CPI-IGAE | 加权同质图、归纳式聚合器、配体集蛋白表征 |
| CPI-IGAE 博士论文 | `参考论文/基于图神经网络的化合物—蛋白质相互作用研究_万晓喆.pdf` | 数据筛选标准、图构建细节、负采样策略 |
| MolTrans | https://github.com/kexinhuang12345/MolTrans | 分子交互 Transformer、DTI 基线 |
| DeepChem | https://github.com/deepchem/deepchem | 分子 ML 工具链、验证框架 |
| DTINet / ConPLex | GitHub 搜索 top 仓库 | 网络整合、蛋白质语言模型增强 |

### 1.3 核心逻辑缺陷清单

#### 缺陷 A：输入数据真实性验证不足

1. **活性数据来源混杂**：ChEMBL 与 BindingDB 的 `canonical_smiles` 列名、记录质量、重复情况未做标准化校验。
2. **SMILES 有效性未在活性数据中强制检查**：`collect_experimental_actives.py` 收集的 64,742 条 SMILES 中有 190 条无法生成有效指纹，但训练时直接丢弃，未做来源追溯。
3. **内置文献活性集 `FERROPTOSIS_ACTIVES` 未经验证**：部分 SMILES 来自粗略构造，未经验证是否真实对应文献活性分子。
4. **蛋白-基因映射未一致性校验**：ChEMBL/BindingDB 返回的基因符号与 L2 蛋白特征表的 `gene_symbol` 未做双向匹配检查。
5. **活性数值/类型未校验**：未剔除 `standard_relation` ≠ '='、活性单位错误、超出合理范围的数据。

#### 缺陷 B：训练标签构造逻辑存在循环

1. **以 Tanimoto 相似度作为软标签**：模型实际学习的是“候选化合物与已知活性化合物的结构相似度”，而非真实的 CPI 关系。
2. **高置信正样本阈值 0.7 过严**：导致 24/28 个靶标无高置信样本，可训练靶标仅 4 个。
3. **用相似度标签评估模型 AUC**：分类任务的真实标签是相似度 > 0.5，本质是相似性分类，不是 CPI 分类，AUC 值虚高。
4. **蛋白质特征在单靶标模型中无判别信息**：同一靶标内所有样本的蛋白特征相同，模型无法学习化合物-蛋白配对特异性。

#### 缺陷 C：预测阶段方法混用且未标注

1. **可训练靶标用集成模型预测，不可训练靶标用相似性预测**，但输出表中统一为 `prediction_score`，未明确区分 `method`。
2. **相似性回退方法在预测阶段重新计算指纹**，与训练阶段预计算缓存不一致，存在重复计算与潜在差异。
3. **无参考靶标输出全 0**，但排序阶段仍纳入统计，扭曲综合得分。

#### 缺陷 D：富集因子（EF）计算不合法

1. **用预测分数 > 0.5 的样本数作为“命中数”**，同时用 `n_high`（相似度 > 0.7 的样本数）作为分母基线，分子分母来自同一相似度逻辑，造成 EF 虚高。
2. 合理的 EF 应以**真实实验活性**或**独立验证集**为 ground truth，而非模型自身的分类阈值。

#### 缺陷 E：候选化合物排序公式权重缺乏依据

1. 综合得分同时包含绝对分数（mean/max）与相对计数（n_hits/n_targets），量纲不一致。
2. `std_score` 被用作“一致性”指标，但 std 与预测置信度无单调关系。
3. 未对 method 为相似性/无参考的靶标预测降权，导致非模型预测结果错误提升排名。

#### 缺陷 F：与 CPI-IGAE 设计路线偏离

原计划采用 CPI-IGAE 的**归纳式图神经网络 + 加权同质图 + 配体集蛋白表征**，但 v4 实现为基于相似性标签的 per-target 随机森林/xgboost，未利用跨靶标共享信息，也未构建化合物-蛋白关系图。

#### 缺陷 G：异常处理与日志不规范

1. 部分 `ImportError` 使用 `pass` 跳过，未记录警告。
2. 输入文件缺失时直接 `return False`，未抛出异常或输出结构化错误报告。
3. 缺少输入文件校验和（MD5/SHA256）与数据 lineage 记录。

---

## 二、优化目标

1. **输入真实性**：建立端到端输入校验，活性数据、SMILES、蛋白特征、化合物特征均须通过真实性/一致性/质量检查。
2. **逻辑判断**：在数据不足、方法回退、结果异常时，自动降级并明确标注，不伪装高置信结果。
3. **评估可靠**：取消循环评估指标，使用真实 CPI 标签（ChEMBL/BindingDB 阳性对）作为 ground truth；对无真实标签的靶标，不计算 EF。
4. **输出可解释**：每条预测记录须标注 `method`（model / similarity / no_reference），排序时按 method 可信度加权。
5. **路线一致性**：在保持 P4 范围内，尽可能引入 CPI-IGAE 的关键思想：配体集蛋白表征、跨靶标负采样、化合物-蛋白配对特征。

---

## 三、技术实现步骤

### 步骤 1：构建强化输入验证模块 `validate_p4_inputs_v2.py`

新增/扩展以下校验：

#### 1.1 文件级校验
- 检查所有必需文件存在性、非空性、可读性。
- 计算每个输入文件的 SHA256 校验和并记录到 `L4/logs/input_checksums.json`。
- 校验文件修改时间是否在合理范围内。

#### 1.2 化合物数据校验
- `tcm_compound_pool_filtered.csv` 与 `rdkit_descriptors.csv`、`ecfp4_fingerprints.npy`、`maccs_fingerprints.npy` 行数严格一致。
- SMILES 列无空值、无重复（按 SMILES_std）。
- 抽检 100% 的 SMILES 可通过 RDKit 解析（当前 100 个抽检不够）。
- 指纹矩阵非全零行、无 NaN/Inf。

#### 1.3 蛋白数据校验
- `target_protein_features.csv`、`protein_descriptors.csv`、`protein_pseaac.csv` 行数一致。
- `gene_symbol` 无空值、无重复。
- 校验 AAC/PseAAC 特征维度：AAC=20，PseAAC=50。
- 核心靶标基因（CORE_GENES + PRIORITY_TARGETS）必须在蛋白表中出现。

#### 1.4 活性数据校验
- ChEMBL/BindingDB CSV 必须存在至少一个。
- 检查必需列：`gene`、`canonical_smiles`、活性值/类型列。
- SMILES 解析率 ≥ 95%，否则记录为 ERROR。
- 基因符号必须在 L2 蛋白表或 `GENE_UNIPROT_MAP` 中可映射。
- 剔除 `standard_relation` 不为 '=' 的记录（若存在该列）。
- 检查重复 (gene, canonical_smiles) 对，去重并记录去重数。
- 对内置 `FERROPTOSIS_ACTIVES` 进行 SMILES 解析检查，无法解析的剔除并告警。

#### 1.5 交叉一致性校验
- 活性数据中的基因集合 vs 蛋白表基因集合：计算匹配率，缺失基因列出清单。
- 活性数据中可通过 RDKit 解析的 SMILES 与 TCM 库 SMILES 的重叠率：若重叠率 < 1%，触发“无正样本”警告。

### 步骤 2：重构训练标签构造逻辑

#### 2.1 区分“真实 CPI 标签”与“相似性辅助信息”
- **真实阳性**：来自 ChEMBL/BindingDB/DrugBank 的实验验证活性化合物。
- **相似性扩展**：仅作为数据增强或冷启动辅助，不得直接作为模型评估的 ground truth。

#### 2.2 改进软标签分层策略

| 层级 | 条件 | 标签 y_cls | 标签 y_reg | 样本权重 | 说明 |
|------|------|-----------|-----------|----------|------|
| 真实阳性 | 在实验活性集中 | 1 | 1.0 | 1.0 | 最高置信 |
| 高相似扩展 | Tanimoto > 0.7 | 1 | max_sim | 0.7 | 结构类似物 |
| 中相似扩展 | 0.5 < Tanimoto ≤ 0.7 | 0/1 | max_sim | 0.4 | 边界样本 |
| 弱相似/背景 | 0.3 < Tanimoto ≤ 0.5 | 0 | max_sim | 0.2 | 软负 |
| 真实阴性 | Tanimoto ≤ 0.3 | 0 | 0.0 | 1.0 | 随机负样本 |

#### 2.3 引入跨靶标真实负样本
- 对每个真实阳性化合物，从其他靶标中随机采样 5-10 个蛋白作为负样本。
- 这引入蛋白特征变化，使模型能学习化合物-蛋白配对特异性。

### 步骤 3：修正模型训练与评估逻辑

#### 3.1 训练入口条件
- 真实阳性数 ≥ 5 且总阳性数（真实+扩展）≥ 10：允许训练。
- 否则标记为 `INSUFFICIENT_DATA`，不训练。

#### 3.2 评估指标
- 使用 **真实阳性 vs 真实阴性** 的 AUROC/AUPRC 作为模型评估指标。
- 不使用相似度扩展样本评估，避免循环验证。
- 记录每个模型的训练/验证曲线（JSON）。

#### 3.3 模型选择
- 保留 RF / XGB / LR / SVM / KNN，但仅保留在真实标签上验证 AUC > 0.6 的模型。
- 对单个靶标，若所有模型 AUC < 0.6，降级为相似性方法。

### 步骤 4：预测阶段方法标注与可信度加权

#### 4.1 每条预测记录强制标注 method
- `model`：该靶标有可训练且验证通过的模型。
- `similarity`：无模型，但有已知活性化合物，使用 Tanimoto 最大相似度。
- `no_reference`：无活性数据，输出 NaN，不参与排序。

#### 4.2 按方法可信度加权
- model 预测：权重 1.0
- similarity 预测：权重 0.5
- no_reference：权重 0.0

### 步骤 5：修正富集因子与候选排序

#### 5.1 富集因子仅对“真实阳性”计算
- 对每个可训练靶标，以该靶标的真实阳性化合物为 ground truth。
- 在 TCM 库中检索这些真实阳性是否出现（按 InChIKey/SMILES）。
- 若真实阳性未出现在 TCM 库中，则无法计算 EF，标记为 `NA`。
- 仅当 TCM 库中包含 ≥ 1 个真实阳性时，计算 EF@1%/5%/10%。

#### 5.2 排序公式改进
- 仅纳入 `method != no_reference` 的预测。
- 对每个化合物计算：
  - `model_score`：method=model 的靶标平均 prediction_score
  - `sim_score`：method=similarity 的靶标平均 prediction_score（加权 0.5）
  - `n_model_targets`：method=model 的命中靶标数
  - `n_sim_targets`：method=similarity 的命中靶标数
  - `priority_bonus`：命中铁衰老核心靶标（ACSL4/GPX4/HMOX1/SLC7A11/TFRC）的加分
- 综合得分：
  ```
  composite = 0.30 * model_score
            + 0.20 * sim_score
            + 0.25 * (n_model_targets / n_model_total)
            + 0.15 * (n_sim_targets / n_sim_total)
            + 0.10 * priority_bonus
  ```

### 步骤 6：引入 CPI-IGAE 关键思想（P4 范围内）

#### 6.1 配体集蛋白表征
- 对每个靶标，用其真实阳性配体的 ECFP4 指纹（按位 1/3 阈值）构建 1024 位蛋白表征。
- 与原 AAC/PseAAC 蛋白特征拼接，作为可选输入。

#### 6.2 化合物-蛋白配对特征
- 不再 per-target 固定蛋白特征，而是为每个 (compound, target) 对构建：
  - 化合物 ECFP4 + MACCS + 描述符
  - 配体集蛋白表征
  - 化合物-蛋白特征交互（如元素级乘积、Tanimoto）

#### 6.3 跨靶标训练（可选）
- 若时间允许，构建一个统一训练集：所有 (compound, target) 对，真实阳性标签=1，负样本标签=0。
- 用单模型（如 XGBClassifier）学习跨靶标 CPI 预测。
- 这更接近 CPI-IGAE 的“共享蛋白/化合物隐空间”思想，但使用表格 ML 而非 GNN。

---

## 四、验证方法

### 4.1 输入验证回归
- 运行 `python L4/scripts/validate_p4_inputs_v2.py`。
- 预期：所有 ERROR 数量 ≤ 0，WARNING 全部记录并可解释。
- 输出：`L4/logs/input_validation_report_v2.json`

### 4.2 代码静态检查
- 运行 `ruff check L4/scripts/phase4_model_pipeline_v4_5.py`
- 预期：无 ERROR，无未处理异常吞掉模式。

### 4.3 单元测试
- 测试 `_vectorized_tanimoto_full`：输出范围 [0,1]，对称性，全 1 输入为 1。
- 测试 `build_soft_label_data`：真实阳性标签=1，真实阴性标签=0，权重非负。
- 测试 `enrichment_analysis`：当 TCM 库无真实阳性时返回空表，不报错。
- 测试 `predict_tcm_pool_v4_5`：method 列仅含 {model, similarity, no_reference}。

### 4.4 运行验证
- 运行完整 pipeline：`python L4/scripts/phase4_model_pipeline_v4_5.py`
- 检查日志：无未捕获异常，所有 WARNING 已记录原因。
- 检查输出文件：
  - `model_performance_v4_5.csv`
  - `tcm_predictions_full_v4_5.csv`
  - `tcm_top_candidates_v4_5.csv`
  - `enrichment_analysis_v4_5.csv`
  - `phase4_report_v4_5.md`

### 4.5 效果评估
- 可训练靶标数量（真实阳性 ≥ 5）≥ 4。
- 每个可训练靶标在真实标签上的 AUROC ≥ 0.6。
- EF 计算严格基于真实阳性，数值合理（通常 < 100）。
- Top 候选化合物中，PTGS2/MAPK1 等富样本靶标占主导，排序公式不偏爱 method=similarity 的靶标。
- 所有预测记录都有 method 标注，no_reference 记录数为 0 或被正确标记。

---

## 五、评估标准

| 评估维度 | 通过标准 | 检查方式 |
|----------|----------|----------|
| 输入真实性 | 无 ERROR，WARNING 全部解释 | 验证报告 |
| 代码质量 | ruff 通过，无 try-except:pass | 静态检查 |
| 逻辑正确性 | 真实标签评估，无循环验证 | 训练指标 |
| 输出可靠性 | method 标注清晰，no_reference 正确降级 | 预测表 |
| 排序合理性 | 模型预测结果优先于相似性结果 | Top 候选表 |
| EF 合法性 | 仅基于真实阳性，数值合理 | EF 表 |
| 可复现性 | 固定随机种子，记录输入 checksum | 日志/JSON |

---

## 六、实施路线图

| 步骤 | 任务 | 输出文件 | 状态 |
|------|------|----------|------|
| 1 | 编写改进方案文档 | `L4/P4_logical_optimization_plan.md` | 进行中 |
| 2 | 实现强化输入验证 | `L4/scripts/validate_p4_inputs_v2.py` | 待开始 |
| 3 | 实现 v4.5 pipeline | `L4/scripts/phase4_model_pipeline_v4_5.py` | 待开始 |
| 4 | 运行验证并调优 | 日志 + 结果文件 | 待开始 |
| 5 | 生成效果评估报告 | `L4/results_v4_5/phase4_report_v4_5.md` | 待开始 |

---

*文档版本: v1.0 | 创建日期: 2026-06-23*
