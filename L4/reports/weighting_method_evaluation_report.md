# P4 机器学习筛选模型加权方法学术合规性评估报告

**评估对象**：`L4/scripts/phase4_model_pipeline_v4.py` 中的训练样本权重与候选化合物综合得分权重。

**评估时间**：2026-06-23

**评估依据**：代码实现、检索到的 5 篇相关领域权威期刊/会议论文、虚拟筛选与多准则决策（MCDM）研究惯例。

---

## 1. 背景与目的

本研究在 Phase 4（P4）中药单体机器学习筛选中使用了两种加权机制：

1. **训练样本权重**：基于化合物与已知活性分子的 ECFP4/Tanimoto 最大相似度，对软标签样本赋予不同权重。
2. **候选化合物排序权重**：基于多模型预测结果的综合得分，对“平均分、最高分、命中率、高置信命中率、预测一致性”五个维度进行加权求和。

本报告旨在从理论依据、透明可复现性、学术规范、文献对比、创新性与合理性等角度，对上述加权方法进行系统性学术合规性评估，并提出改进建议。

---

## 2. 当前加权方法说明

### 2.1 训练样本权重（软标签加权）

代码位置：`L4/scripts/phase4_model_pipeline_v4.py`，第 316–320 行。

设化合物与最近邻已知活性分子的 Tanimoto 相似度为 `s = max_sim`，则样本权重 `w_i` 定义为：

```
if s > 0.7:        w_i = s
elif s > 0.5:      w_i = 0.5 * s
elif s > 0.3:      w_i = 0.2 * s
else:              w_i = 1 - s
```

即：高相似（>0.7）样本权重等于相似度；中相似（0.5–0.7）权重折半；弱相似（0.3–0.5）权重折为 20%；低相似（≤0.3）作为负样本，权重为“不相似度”。

### 2.2 候选化合物综合得分（排序加权）

代码位置：`L4/scripts/phase4_model_pipeline_v4.py`，第 713–719 行。

对每个化合物，先按靶标聚合预测结果：

```python
mean_score = prediction_score.mean()
max_score = prediction_score.max()
std_score = prediction_score.std()
n_hits = sum(prediction_score > 0.5)
n_high = sum(prediction_score > 0.7)
n_targets = count(prediction_score)
```

综合得分 `CS` 为：

```
CS = 0.25 * mean_score
   + 0.25 * max_score
   + 0.20 * n_hits / n_targets
   + 0.15 * n_high / n_targets
   + 0.15 * (1 - clip(std_score, 0, 1))
```

### 2.3 权重来源

上述权重系数为硬编码（hard-coded），在代码中仅通过注释简要说明维度含义，未给出数学推导、参数优化过程或文献依据。

---

## 3. 理论依据评估

### 3.1 训练样本权重

**可取之处**：
- 以化学相似度作为样本置信度的代理变量，符合“结构相似性假设（similar property principle）”的基本思想。
- 对高相似样本赋予更高权重，对低相似样本作为负样本，逻辑上具有直观合理性。

**不足之处**：
- 阈值 0.7、0.5、0.3 以及分段系数 1.0、0.5、0.2 缺乏系统推导。不同指纹、不同靶标家族的最优相似阈值差异显著，未经验证即采用统一阈值。
- 将相似度同时用于标签构造（软标签）和权重分配，存在信息循环风险：模型既学习“相似”，又用“相似”评估模型。
- 负样本权重 `1 - s` 假设不相似度等于负例置信度，该假设未经过概率校准或似然比检验。

### 3.2 综合得分权重

**可取之处**：
- 采用加权求和模型（Weighted Sum Model, WSM），是多准则决策（MCDM）中最常用的形式化方法之一。
- 各分项均经过 `clip` 或比例化处理，数值范围大致落在 [0, 1]，具备初步可合并性。

**不足之处**：
- 权重系数 0.25/0.25/0.20/0.15/0.15 未见专家打分、AHP、熵权法或数据驱动优化过程。
- `mean_score` 与 `max_score` 均来自同一组预测值，二者高度相关，同时以相同权重纳入会造成信息重复。
- `std_score` 表示的是不同靶标预测值的标准差，并不等价于模型不确定性（uncertainty）。低标准差可能仅表示该化合物对所有靶标预测分数都居中，而非高置信。
- 未对权重进行敏感性分析（sensitivity analysis），无法判断 Top 候选化合物对权重变化的稳健性。

---

## 4. 透明性与可复现性评估

| 维度 | 现状 | 评价 |
|---|---|---|
| 公式透明性 | 代码中直接写出公式与系数 | 良好 |
| 参数来源 | 硬编码，无配置文件或文献引用 | 不足 |
| 可复现性 | 在相同代码版本与输入数据下可完全复现 | 良好 |
| 可追溯性 | 未记录权重设计决策与版本变更 | 不足 |
| 敏感性分析 | 未开展 | 不足 |
| 替代方案比较 | 未报告其他权重方案的结果 | 不足 |

结论：当前加权方法在“实现透明”层面合格，但在“科学透明”层面不足，难以让审稿人或读者判断权重选择的科学性。

---

## 5. 学术规范与研究惯例

### 5.1 虚拟筛选中的共识评分（Consensus Scoring）

在分子对接与虚拟筛选领域，共识评分的标准做法包括：

- **排名融合**：取多个打分函数排名的中位数（rank-by-median）或平均值。
- **分数融合**：算术平均、几何平均、调和平均、最小值法等。
- **数据驱动**：基于验证集表现（如 AUC、EF）为各子模型或各打分函数赋予权重。

固定且不等权的加权求和虽然可行，但通常需要明确的理论或经验依据。 arbitrary 权重在主流期刊中较难被接受。

### 5.2 多准则决策（MCDM）规范

MCDM 中权重确定的主流方法包括：

- **主观法**：AHP（层次分析法）、ANP、专家打分。
- **客观法**：熵权法（entropy weighting）、CRITIC、标准离差法。
- **优化法**：基于验证指标（如 AUC、EF、BEDROC）的网格搜索或贝叶斯优化。

无论采用何种方法，均要求：
1. 各准则经过标准化或归一化；
2. 权重之和为 1；
3. 给出权重确定方法及稳健性检验。

当前方法仅满足第 2 条，未满足第 1、3 条。

### 5.3 机器学习集成模型权重

在集成学习中，子模型权重通常依据验证集性能确定，例如：

- 回归任务：按验证集 MSE 的倒数加权。
- 分类任务：按验证集 AUC 或 AUPR 加权；类别不平衡时优先使用 AUPR/PRC。

等权平均（uniform averaging）仅在子模型性能相近且相互独立时被广泛接受。当前 P4 中不同模型（RF、XGB、LR、SVM、KNN）性能差异显著，简单等权或固定加权并非最优惯例。

---

## 6. 文献对比分析

### 6.1 选取文献

| 编号 | 文献 | 期刊/来源 | 与当前研究的相关性 |
|---|---|---|---|
| 1 | Charifson et al., 1999 | *Journal of Medicinal Chemistry* | 共识评分经典文献，提出基于排名的共识策略 |
| 2 | Bajusz et al., 2019 | *Molecules* | 系统比较 7 种数据融合规则 |
| 3 | Klon et al., 2004 | *Journal of Medicinal Chemistry* | 排名中位数共识 + 朴素贝叶斯 |
| 4 | Mamada et al., 2023 | *ACS Omega* | 两种独立方法预测概率平均与一致性共识 |
| 5 | Parker et al., 2025 | arXiv / *Journal of Chemical Information and Modeling* | 异构集成模型按验证性能加权 |

### 6.2 各文献加权处理方式

#### 文献 1：Charifson et al. (1999)

- **方法**：将多个打分函数的排名或分数进行“交集型”共识（intersection-based consensus），优先保留在多个打分函数中均排名靠前的化合物。
- **权重特征**：未使用固定数值权重，而是通过投票/排名阈值实现共识。
- **结论**：共识评分能显著降低假阳性率、提高命中率。

#### 文献 2：Bajusz et al. (2019)

- **方法**：比较 ensemble docking 中 7 种数据融合规则，包括最小值（minimum）、算术平均（arithmetic mean）、几何平均（geometric mean）、调和平均（harmonic mean）等。
- **权重特征**：等权平均或规则型融合；不推荐任意不等权。
- **结论**：几何平均与调和平均在多数案例中优于常用的最小值规则。

#### 文献 3：Klon et al. (2004)

- **方法**：rank-by-median 共识评分与朴素贝叶斯分类器结合。
- **权重特征**：权重来自排名统计与贝叶斯后验概率，而非人工设定。
- **结论**：在对接单独表现不佳的案例中仍能显著提升富集率。

#### 文献 4：Mamada et al. (2023)

- **方法**：将基于分子图像的深度学习（DeepSnap-DL）与基于分子描述符的模型（MD-based）进行集成。
- **权重特征**：
  - 集成模型：两种方法预测概率的简单平均（等权）。
  - 共识模型：仅保留两种方法预测结果一致的化合物。
- **结论**：等权平均已能提升预测性能；一致性共识进一步降低不确定性。

#### 文献 5：Parker et al. (2025)

- **方法**：MetaModel 异构集成框架，组合 RF、XGB、NN、SVM 等多种模型。
- **权重特征**：
  - 回归子模型：按验证集 MSE 的倒数加权。
  - 分类子模型：按验证集 AUC 或 AUPR 加权；类别不平衡时采用 AUPR。
- **结论**：基于验证性能的数据驱动加权优于固定权重或简单平均。

### 6.3 当前方法与文献的异同

| 对比维度 | 当前 P4 方法 | 文献主流做法 |
|---|---|---|
| 权重来源 | 硬编码、经验设定 | 排名融合、等权平均、数据驱动优化 |
| 理论依据 | 仅“直观合理” | 有统计推导或验证指标支撑 |
| 敏感性分析 | 未开展 | 通常进行交叉验证或消融实验 |
| 子模型权重 | 未区分模型性能 | 按验证 AUC/MSE/AUPR 加权 |
| 相似度阈值 | 固定 0.7/0.5/0.3 | 通常经数据集优化（如 SimSpread 中 0.2–0.3） |
| 方法标注 | 未区分 model/similarity/no_reference | 通常区分并降权非模型结果 |

**相同点**：均认可“多信号融合优于单一信号”，并尝试通过加权/共识提升预测可靠性。

**不同点**：当前方法在权重取值与组合规则上依赖主观设定，而文献多采用数据驱动、排名统计或等权融合，并辅以验证指标说明。

---

## 7. 创新性与合理性论证

### 7.1 创新性评估

当前加权方法在形式上并非首创：

- 分段相似度加权与网络推断中的相似度加权思路相似（如 SimSpread）。
- 五维综合得分属于典型的加权求和 MCDM 框架。
- 将“最大得分、命中率、一致性”纳入排序，在虚拟筛选中已有大量先例。

因此，创新性有限。其价值主要体现在**针对小样本、冷启动场景的工程化尝试**，而非方法论创新。

### 7.2 合理性论证

在小样本、跨靶标且阳性数据稀缺的场景下，完全数据驱动的权重优化可能因过拟合而不稳定。采用固定启发式权重可以：

1. 避免在极小样本上进行高方差的参数搜索；
2. 通过多维度（得分、命中率、一致性）降低单一靶标预测的噪声影响；
3. 使排序结果具有可解释性。

但这些合理性应作为**方法选择的前提假设**明确披露，并通过敏感性分析与消融实验加以验证。当前代码尚未完成这一步骤。

---

## 8. 存在的主要问题

1. **权重系数缺乏数学或经验推导**：0.25/0.25/0.20/0.15/0.15 与 1.0/0.5/0.2 未见来源说明。
2. **相似度阈值未经验证**：0.7/0.5/0.3 对 ECFP4 的适用性未在当前数据集上验证。
3. **信息重复与量纲混用**：mean_score 与 max_score 相关；std_score 不等于不确定性。
4. **未区分预测方法来源**：模型预测、相似性推断、无参考结果统一为 prediction_score，导致排序被不同可信度的信号污染。
5. **缺乏敏感性分析与消融实验**：无法证明当前权重优于等权或其他权重方案。
6. **评估指标循环**：训练标签与评估 ground truth 均依赖 Tanimoto 相似度，可能高估模型性能。

---

## 9. 改进建议

### 9.1 短期改进（可在当前 P4 框架内实现）

1. **文档化权重假设**：在代码与报告中明确说明所有权重为“基于小样本的启发式设定”，并给出选择逻辑。
2. **敏感性分析**：对综合得分权重进行网格搜索或蒙特卡洛扰动，报告 Top 50 候选的 Jaccard 稳定性。
3. **消融实验**：分别测试仅使用 mean_score、仅使用 max_score、等权五维、当前权重等方案，比较 AUROC、EF、Top 候选重叠率。
4. **方法分层加权**：
   - model 预测：权重 1.0；
   - similarity 推断：权重 0.5；
   - no_reference：不参与排序。
5. **相似度阈值验证**：在真实阳性集上通过交叉验证确定最优 Tanimoto 阈值，而非固定 0.7/0.5/0.3。

### 9.2 中期改进（建议在后续版本中实现）

1. **数据驱动的子模型权重**：按验证集 AUC 或 AUPR 为 RF/XGB/LR/SVM/KNN 赋权，参考 Parker et al. (2025)。
2. **排名融合替代加权求和**：尝试 Borda 计数、Copeland 法、几何平均等文献验证的共识策略。
3. **MCDM 正规化**：若保留加权求和，采用熵权法或 CRITIC 客观赋权，并报告权重稳健性。
4. **不确定性量化**：使用模型预测方差或集成分歧（ensemble disagreement）替代 std_score 作为一致性指标。

### 9.3 长期改进（若进入 P5/P6）

- 在更大规模实验数据上，通过贝叶斯优化或强化学习自动学习最优权重函数。
- 引入因果推断框架，区分“结构相似性”与“真实 CPI”对排序的贡献。

---

## 10. 结论

当前 P4 模型的加权方法在**工程实现层面透明、可复现**，但在**学术合规性层面存在明显不足**：

- 权重系数缺乏理论推导与经验验证；
- 未遵循虚拟筛选共识评分与 MCDM 的主流规范；
- 与权威文献中的数据驱动、排名融合或等权共识方法相比，当前固定启发式权重的科学依据较弱；
- 创新性有限，合理性需通过敏感性分析与消融实验进一步支撑。

**综合评价**：当前加权方法可作为探索性研究的临时方案，但尚不具备发表在主流药物化学/化学信息学期刊的充分条件。建议按照本报告第 9 章的短期与中期改进方向进行修订，重点补充权重来源说明、敏感性分析、方法分层与数据驱动加权，以提升学术合规性与结果可信度。

---

## 11. 参考文献

1. Charifson P S, Corkery J J, Murcko M A, Walters W P. Consensus Scoring: A Method for Obtaining Improved Hit Rates from Docking Databases of Three-Dimensional Structures into Proteins. *Journal of Medicinal Chemistry*, 1999, 42(25): 5100–5109. DOI: [10.1021/jm990352k](https://doi.org/10.1021/jm990352k)

2. Bajusz D, Rácz A, Héberger K. Comparison of Data Fusion Methods as Consensus Scores for Ensemble Docking. *Molecules*, 2019, 24(15): 2690. DOI: [10.3390/molecules24152690](https://doi.org/10.3390/molecules24152690)

3. Klon A E, Glick M, Davies J W. Combination of a Naive Bayes Classifier with Consensus Scoring Improves Enrichment of High-Throughput Docking Results. *Journal of Medicinal Chemistry*, 2004, 47(18): 4356–4359. DOI: [10.1021/jm049970d](https://doi.org/10.1021/jm049970d)

4. Mamada H, Takahashi M, Ogino M, Nomura Y, Uesawa Y. Predictive Models Based on Molecular Images and Molecular Descriptors for Drug Screening. *ACS Omega*, 2023, 8(40): 37186–37195. DOI: [10.1021/acsomega.3c04073](https://doi.org/10.1021/acsomega.3c04073)

5. Parker M L, Mahmoud S, Montefiore B, Öeren M, Tandon H, Wharrick C, Segall M D. Improving Predictions of Molecular Properties with Graph Featurisation and Heterogeneous Ensemble Models. arXiv:2510.23428 [cs.LG], 2025. URL: [https://arxiv.org/abs/2510.23428](https://arxiv.org/abs/2510.23428)

6. Vigil-Vásquez C, Schüller A. De Novo Prediction of Drug Targets and Candidates by Chemical Similarity-Guided Network-Based Inference. *International Journal of Molecular Sciences*, 2022, 23(17): 9666. DOI: [10.3390/ijms23179666](https://doi.org/10.3390/ijms23179666)
