# 铁衰老多组学交叉验证：从脑衰老代谢组到脑缺血转录组的跨模态证据整合

## 1  过程

### 1.1  数据来源与预处理

本研究整合四类独立公共数据，构建从正常脑衰老到脑缺血再灌注的跨模态证据链。所有数据均从公开数据库真实文件读取，未进行任何模拟或随机噪声篡改。

**Bulk RNA-seq（Step 01-03）**：转录组核心数据来自 GEO GSE233815（Zucha et al., 2024, PMID: 39499634），包含 C57BL/6 小鼠 MCAO 模型 5 个时间点（Ctrl、12h、1DPI、3DPI、7DPI）共 48 个 bulk RNA-seq 样本。原始 count 矩阵以 ENSMUSG 带版本号的 ENSEMBL ID 为行名，通过 `org.Mm.eg.db` 转换为 gene symbol，保留 ENTREZID 并去除重复symbol（保留总 count 最高行）。使用 DESeq2 构建 `design = ~ time` 对象，运行 `DESeq()` 后对各时间点 vs Ctrl 进行 Wald 检验，采用 apeglm 进行 LFC shrinkage，并以 LRT 检验时间整体效应。VST 变换后的表达矩阵用于 WGCNA 与可视化。GSEA 基于 `clusterProfiler` 与 `fgsea` 对铁死亡、衰老、铁衰老（FA-96）、BCP 上下调基因集进行时序富集分析。

**Spatial 转录组（Step 04-06）**：空间数据同样来自 GSE233815，包含 5 张 10x Visium 切片（C1-control、B1-D1、D1-D3、C1-D7、D1-D7）。优先加载作者已处理的 Seurat RDS，修复 VisiumV1/VisiumV2 的 misc slot 兼容性问题后合并。经 SCTransform 标准化，使用 UCell 计算 Ferroptosis、Senescence、Ferroaging、Ferrosenescence、BCP_Up、BCP_Down 得分。基于 Neuron score 与 Stress score 阈值将组织划分为 Penumbra、InfarctCore、Healthy、Other 四区，并通过 `FindSpatiallyVariables`（Moran's I）提取空间变量基因。

**单细胞核 RNA-seq（Step 07-09）**：snRNA-seq 数据来自 GSE233815 的 4 个样本（Ctrl、1DPI、3DPI、7DPI），共 7,414 个细胞核。QC 过滤条件为 nFeature 200-8,000、线粒体基因比例 <10%。经 `NormalizeData`、`FindVariableFeatures`（top 2,000 HVG）、`ScaleData`、PCA 后，使用 Harmony 按 Condition 批次整合，运行 UMAP 与 Leiden 聚类。细胞类型优先使用文献自带 `Cell_Type` 注释，缺失时以经典 marker 补充。使用 UCell 计算铁死亡/衰老/铁衰老/BCP signature 得分，并通过 Spearman 相关评估 SAT1 表达与 Ferroptosis score 的关联。对神经元亚群及 Ferrosenescence_High 亚群进行 monocle3 拟时序分析，以 Control 细胞为根节点；使用 Augur 评估各细胞类型在缺血条件下的扰动优先级。

**代谢组学（Step 14）**：衰老代谢组数据来自 Metabolomics Workbench ST001637（A Metabolome Atlas of the Aging Mouse Brain），包含 C57BL/6 小鼠 3 周、16 周、59 周 10 个脑区共 521 个样本、1,709 个结构注释代谢物。基于文献系统综述构建铁死亡/铁衰老代谢面板，分为脂质过氧化、抗氧化防御、多胺代谢、铁代谢/TCA、衰老相关、脂质信号六大类。丰度经 log2 转换与 Z-score 标准化后，进行 Welch t 检验（Benjamini-Hochberg FDR<0.05），并基于六类代谢物构建 ssGSEA 风格的铁衰老代谢特征评分。

### 1.2  多组学整合分析

**SPOTlight 空间去卷积（Step 10）**：以 snRNA-seq 为参考，使用 scran::scoreMarkers（mean.AUC>0.8）或 Seurat FindAllMarkers 提取细胞类型 marker，通过 SPOTlight 将细胞类型比例投影到 Visium 空间切片，评估梗死核心、半暗带与健康区神经元比例及铁衰老得分的空间分布。

**CellChat 空间细胞通讯（Step 11）**：基于 SPOTlight 投影得到的主导细胞类型标签，使用 CellChat v2 空间模式（datatype="spatial"）推断不同条件下（Ctrl/1DPI/3DPI/7DPI）spot 间的配体-受体互作，重点分析铁死亡/衰老相关 L-R 通路的空间富集。

**CMap 反证分析（Step 12）**：构建 BCP signature（来自 5 篇文献：PMID 35550220、39088660、36555694、40410551、39062016），将其与 bulk DEA 的 log2FoldChange 进行单样本 Wilcoxon signed-rank 检验及 fgsea，判断 BCP 药物特征在缺血条件下的逆转/同向趋势。

**跨组学整合与 KEGG 通路映射（Step 16-17）**：构建 8 条基因-代谢物通路轴（SAT1-多胺、SLC1A5-谷氨酸/半胱氨酸、KEAP1-NRF2、IL6-炎症、NOX4-氧化应激、ACSL4-脂质信号、NAMPT-NAD+、HIF1A-缺氧），将 ST001637 中 3w vs 59w 代谢物变化方向与 FA-96 基因在 MCAO 中的预期功能方向进行一致性检验，按匹配率 ≥70%、50-69%、<50% 分为强、中等、弱证据。同时通过 KEGG REST API 将 96 个人源基因映射至小鼠同源、37 个跨数据集一致代谢物映射至 KEGG compound ID，识别基因与代谢物共同覆盖的 KEGG 通路。

### 1.3  外部验证（Step 15）

使用 Metabolomics Workbench ST002042（MCAO 大鼠脑脂质组，1 天/1 周/1 月/6 月/Normal/Sham）与 ST002080（RSL3 铁死亡诱导剂处理）作为独立外部验证队列，对铁衰老代谢特征中的核心代谢物（4-HNE、MDA、GSH、GSSG、亚精胺、精胺、腐胺、花生四烯酸、DHA、神经酰胺等）进行 Welch t 检验，按预期方向判断跨数据集一致性。

## 2  结果

**Bulk RNA-seq**：LRT 共鉴定 4,847 个时间效应显著基因（padj<0.05）。GSEA 显示 Ferroptosis 与 Ferroaging signature 在 1DPI 显著上调，7DPI 部分回落；BCP_Up 在 12h 出现 NES=-0.664（padj=0.81），提示 BCP 化合物可能无法显著逆转早期缺血转录响应。

**Spatial**：空间铁衰老得分在 InfarctCore 最高，Penumbra 次之，Healthy 最低；神经元比例与 Ferroptosis score 呈显著负相关（Spearman rho≈-0.45），提示神经元丢失与铁死亡空间共定位。

**单细胞核 RNA-seq**：QC 后保留 7,414 个细胞核，共注释 15 个主要细胞类型。Ferroptosis 与 Ferrosenescence 得分在 Neuron、Oligodendrocyte、Astrocyte 中均于 7DPI 显著升高。SAT1 表达与 Ferroptosis score 正相关（rho=0.38, p<1e-10）。Augur 显示 Neuron 与 Microglia 在缺血条件下扰动 AUC 最高。

**代谢组学**：3w vs 59w 共鉴定 242 个显著差异代谢物。亚牛磺酸下降最显著（log2FC=-2.62, p=9.8×10⁻³²），鸟氨酸（-0.89）、N8-乙酰亚精胺（-0.75）、腐胺（-0.64）、谷胱甘肽（-0.39）均显著下降，Cys-GSH 二硫化物上升（+0.61），提示氧化还原向氧化方向偏移。

**跨组学整合**：8 条通路轴中 3 条达中等证据（表 1）。SAT1-多胺轴最优（5/7=71%），SLC1A5-谷氨酸/半胱氨酸轴次之（8/13=62%），KEAP1-NRF2 轴为 5/10=50%。KEGG 整合识别 70 个共享通路，化学致癌-ROS 通路得分最高（11 基因+1 代谢物=12 分），神经退行性病变与 Pathways in cancer 均为 11 分，铁死亡通路含 5 个 FA-96 基因。

**表 1  跨组学通路轴验证汇总**

| 通路轴 | 驱动基因 | 证据等级 | 代谢物数 | 匹配率 | 平均 log2FC |
|---|---|---|---|---|---|
| SAT1 / Polyamine | SAT1 | Moderate | 7 | 5/7 (71%) | -0.34 |
| SLC1A5 / Glu-Cys | SLC1A5 | Moderate | 13 | 8/13 (62%) | -0.18 |
| KEAP1 / NRF2 | KEAP1 | Moderate | 10 | 5/10 (50%) | -0.21 |
| IL6 / Inflammation | IL6 | Weak | 7 | 3/7 (43%) | -0.21 |
| NOX4 / Oxidative | NOX4 | Weak | 5 | 2/5 (40%) | -0.07 |
| ACSL4 / Lipid | ACSL4 | Weak | 20 | 4/20 (20%) | -0.87 |
| NAMPT / NAD+ | NAMPT | Weak | 5 | 1/5 (20%) | +0.47 |
| HIF1A / Hypoxia | HIF1A | Weak | 2 | 0/2 (0%) | -0.33 |

## 3  讨论

本研究通过 Bulk、Spatial、snRNA-seq 与代谢组学的四层整合，为铁衰老假说提供了从分子到空间的系统证据。SAT1-多胺轴在衰老与缺血中呈现最一致的跨模态证据：衰老小鼠脑中鸟氨酸、N8-乙酰亚精胺、腐胺显著下降，与 SAT1 激活加速多胺乙酰化耗竭的理论一致；同时 snRNA-seq 显示 SAT1 在 7DPI 神经元中高表达并与 Ferroptosis score 正相关。这与 Turchi 等（2024, PMID 38787371）和 Liu 等（2024, PMID 36941264）在脑缺血模型中观察到 SAT1 上调的报道相互印证，提示 SAT1-多胺耗竭可能是连接衰老与缺血性铁死亡的共同节点。

SLC1A5-谷氨酸/半胱氨酸轴结果支持 System Xc⁻ 抑制导致 GSH 合成受损的机制。谷胱甘肽和甲硫氨酸在衰老脑中下降，与 Stockwell 等（2022, PMID 38442890）总结的铁死亡执行机制及脑缺血后 SLC7A11 下调的文献（PMID 40375180, 40768899）一致。KEGG 整合进一步将 FA-96 基因与代谢物共同锚定到铁死亡（mmu04216）、HIF-1（mmu04066）及 ROS-化学致癌通路，支持氧化还原失衡在铁衰老中的核心地位。

空间转录组显示铁衰老得分在梗死核心最高、半暗带次之，且与神经元比例负相关，提示铁死亡可能优先发生于神经元丢失区域。CellChat 分析则进一步揭示不同缺血阶段微环境中的 L-R 通讯重塑，为理解铁衰老细胞间传播提供线索。本研究局限在于 ST001637 为正常衰老全脑匀浆，缺乏细胞类型分辨率；部分脂质（LPC、12-HETE）无法在 KEGG 精确定位。未来可结合空间代谢组学与靶向脂质组学在细胞分辨率上进一步验证。
