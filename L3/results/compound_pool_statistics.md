# Phase 3 候选化合物池统计报告

生成时间: 2026-06-23 14:51:57

## 基本信息
- TCMSP 原始化合物: 13,729
- OB/DL 过滤后 (OB≥30%, DL≥0.18): 2,583
- SMILES 获取成功率: 100.0%
- 名称-SMILES 一致性校验后: 2,549（剔除 34 个不确定匹配）
- RDKit 规范化成功率: 2549/2549
- 综合类药性过滤后: 1,480
- SMILES 去重剔除: 907
- **最终候选化合物池: 573**
- **唯一 SMILES 数: 573**

## 类药性过滤统计
| 过滤项 | 通过数 | 通过率 |
|--------|--------|--------|
| Lipinski 五规则 | 2156 | 84.6% |
| BBB 通透性 | 1635 | 64.1% |
| PAINS 毒性 | 2292 | 89.9% |
| **三项全部通过** | 573 | 22.5% |

## BBB 通透性分布
- BBB+/-: 236 (41.2%)
- BBB+: 337 (58.8%)

## 分子量分布
- 均值: 344.3 Da
- 标准差: 67.1 Da
- 最小值: 217.4 Da
- 最大值: 619.2 Da
- 中位数: 334.5 Da
- MW ≤ 500: 555 (96.9%)
- MW > 500: 18 (3.1%)

## OB (口服生物利用度) 分布
- 均值: 53.6%
- 标准差: 18.1%
- 最小值: 30.1%
- 最大值: 135.6%

## 数据质量说明
- 去重前存在 907 行重复 SMILES（可能来自同一化合物在不同草药中的重复收录或名称别名）
- 去重策略：保留每个唯一 SMILES 的首次出现记录
- 名称-SMILES 一致性校验：基于 TCMSP 原始 mw 与 RDKit 计算 MW 比较，偏差阈值 ±5 Da 或 ±5%；剔除 34 个不确定匹配
- 校验原理：当 PubChem/COCONUT 按名称返回的 SMILES 与 TCMSP 记录的分子量显著偏离时，认为该名-构对应关系不可靠，避免错误结构进入下游模型
- 建议：后续可进一步通过 InChIKey 校验名称-结构一致性，并引入 TCMSP 官方或 TCMSID 的结构源进行交叉验证

## 相似性网络
- 节点数: 573
- 边数 (Tanimoto > 0.7): 530
- 网络密度: 0.32%

## TCM 单体覆盖说明
- 原始数据来自 TCMSP 的 13,729 条成分记录，覆盖 TCMSP 502 味中草药
- 经 OB≥30%、DL≥0.18、类药性（Lipinski/BBB/PAINS）、名称-SMILES 一致性校验后，保留 573 个唯一 SMILES
- 候选池以 TCMSP 收录的小分子单体为主（黄酮类、萜类、生物碱、酚酸类等），并不包含复方煎液或粗提物
- 当前数据仅整合 TCMSP + PubChem/COCONUT；未纳入 TCMID、SymMap、HERB、TCMIO 等数据库，后续可扩展以提升结构覆盖度

## 数据来源与参考

### 本流程直接使用的数据源
- TCMSP: Traditional Chinese Medicine Systems Pharmacology Database (Ru et al., 2014, doi:10.1021/ci4005517) — 化合物名称、OB、DL、分子量、草药来源等原始数据
- PubChem (via PubChemPy): 化合物 SMILES 补充 (Kim et al., 2016, doi:10.1093/nar/gkv951; GitHub: https://github.com/mcs07/PubChemPy)
- COCONUT: Collection of Open Natural Products (Sorokina et al., 2021, doi:10.1186/s13321-020-00478-9; https://coconut.naturalproducts.net) — SMILES 补充

### 相关参考数据库/综述（未直接用于本流程，但为 TCM 单体研究常用资源）
- Dryad 数据集: doi:10.5061/dryad.wh70rxwx9
- YaTCM: Li et al., 2018, doi:10.1016/j.csbj.2018.11.002
- TCMSID: Zhang et al., 2022, doi:10.1186/s13321-022-00670-z
- TCMID 2.0: Xue et al., 2013, doi:10.1093/nar/gks1104; Huang et al., 2018, doi:10.1093/nar/gkx1028
- SymMap: Wu et al., 2019, doi:10.1093/nar/gky901
- HERB: Fang et al., 2021, doi:10.1093/nar/gkaa1063
- TCM 数据库综述: Wang et al., 2024, doi:10.3389/fphar.2024.1303693

### 方法与工具参考
- RDKit: Landrum G., open-source cheminformatics toolkit, https://github.com/rdkit/rdkit
- TCMSP-Spider: shujuecn, A Python spider for TCMSP, https://github.com/shujuecn/TCMSP-Spider
- ECFP4: Rogers & Hahn, 2010, doi:10.1021/ci100050t
- MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys
- Lipinski rule of 5: Lipinski et al., 2001, doi:10.1016/S0169-409X(00)00129-0
- PAINS: Baell & Holloway, 2010, doi:10.1021/jm901137j
- QED: Bickerton et al., 2012, doi:10.1038/nchem.1243
- BBB heuristic: Clark, 1999, doi:10.1021/js9803731; Ghose et al., 1999, doi:10.1021/cc9800071
- ETKDGv3: Wang et al., 2020, doi:10.1021/acs.jcim.0c00025 (RDKit ETKDGv3 实现基础)
- MMFF94: Halgren, 1996, doi:10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P
- Murcko scaffold: Bemis & Murcko, 1996, doi:10.1021/jm9602928

## 输出文件清单
| 文件 | 路径 | 大小 |
|------|------|------|
| tcm_compound_pool_filtered.csv | D:\铁衰老 绝不重蹈覆辙\L3\results\tcm_compound_pool_filtered.csv | 142.1 KB |
| ecfp4_fingerprints.npy | D:\铁衰老 绝不重蹈覆辙\L3\results\ecfp4_fingerprints.npy | 1146.1 KB |
| maccs_fingerprints.npy | D:\铁衰老 绝不重蹈覆辙\L3\results\maccs_fingerprints.npy | 93.6 KB |
| rdkit_descriptors.csv | D:\铁衰老 绝不重蹈覆辙\L3\results\rdkit_descriptors.csv | 1059.6 KB |
| compound_similarity_network.csv | D:\铁衰老 绝不重蹈覆辙\L3\results\compound_similarity_network.csv | 34.9 KB |
| compound_pool.pkl.gz | D:\铁衰老 绝不重蹈覆辙\L3\results\compound_pool.pkl.gz | 362.5 KB |