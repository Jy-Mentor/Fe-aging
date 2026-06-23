# CIRI-铁衰老中药单体筛选 —— 详细技术路线（修订版，123个分点）

> **修订说明（v3.0）：** (1) Phase 1：跨平台整合质控，铁衰老基因基准96个，取消与FerrDB取交集；(2) Phase 2：多组学特征严格限定为后验解释用途；(3) **Phase 3：中药单体库构建，产出化合物SMILES+分子指纹供Phase 4预测**；(4) **Phase 4（重写）：基于CPI-IGAE归纳式图神经网络的化合物-蛋白相互作用预测——训练数据来自ChEMBL/BindingDB（>10,000对），模型支持冷启动泛化，对核心靶标×中药单体穷举预测**；(5) Phase 5：基于结构的虚拟筛选（分子对接+MD+ADMET），对Phase 4 Top候选做深度验证；(6) Phase 6：实验验证方案设计。核心原则：**训练数据来自公开数据库（不依赖核心靶标已知配体），CPI-IGAE的归纳式架构天然支持冷启动。**

---

## 第一部分：项目总览与准备工作（第1-10点）

### 1. 项目总体目标
本项目的核心任务是基于多策略计算筛选方法，从海量中药单体中筛选出能够靶向CIRI（脑缺血再灌注损伤）相关铁衰老（Ferro-aging）通路的候选活性分子。项目以刘黎啸2026年发表于Cell Metabolism的铁衰老机制为理论基础，以ACSL4为核心靶标，整合多组学靶标发现、中药单体库构建、基于结构的虚拟筛选、分子动力学模拟验证，最终输出经过计算机模拟严格验证的Top候选中药单体及其完整实验验证方案。

### 2. 核心科学问题
CIRI发生后，脑组织铁过载通过ACSL4-脂质过氧化轴驱动神经细胞铁衰老，导致不可逆的神经功能损伤。科学问题是：是否存在能够直接抑制ACSL4或阻断铁衰老级联反应的中药单体？其作用机制是什么？与传统的"无靶标海筛"不同，本项目的策略是**先锁定靶标，再定向筛选**，以ACSL4为第一优先级靶标。

### 3. 理论基础来源
本路线以5篇核心参考文献为方法论基础：(1) 刘黎啸2026 Cell Metabolism——铁衰老概念、ACSL4靶标验证体系、96个铁衰老基因签名、C11-BODIPY筛选方案；(2) 廖昊森2026——12种机器学习算法集成+WGCNA+中药反向匹配策略；(3) 万晓喆2022博士论文——基于图神经网络的化合物-蛋白质相互作用预测；(4) 朱昊坤2026——异质图动态特征学习的药物重定位；(5) 刘文智2025博士论文——深度学习驱动的疾病-药物-代谢物关联预测。

### 4. 项目工作目录结构
在项目根目录 `d:\铁衰老 绝不重蹈覆辙\` 下创建以下子目录：`L1/`（靶标筛选分析）、`L2/`（多组学特征工程，仅用于后验解释）、`L3/`（中药单体数据库构建）、`L4/`（实验活性数据盘点与筛选策略决策）、`L5/`（主力筛选引擎：分子对接+药效团+MD+ADMET）、`L6/`（实验验证方案设计）、`data/`（统一数据存储）、`results/`（统一结果输出）、`scripts/`（可复用工具脚本）、`logs/`（运行日志）。注意：目录命名需与现有项目结构兼容，避免覆盖已有 `L1 数据集/` 文件夹。

### 5. 环境配置要求
Python 3.9+作为主编程语言，R 4.2+用于统计分析与可视化。核心Python包：pandas、numpy、scipy、scikit-learn、rdkit（化学信息学）、scanpy（单细胞）、biopython（蛋白结构处理）。核心R包：Seurat、WGCNA、limma、clusterProfiler、GSVA。分子对接使用AutoDock Vina（v1.2+），MD模拟使用GROMACS（2023+），药效团使用RDKit或LigandScout。所有依赖通过conda环境管理，环境名称为 `TraeAI-1`。

### 6. 数据管理原则
所有输入数据存放于 `data/` 目录下按来源分子目录；所有中间结果存放于各L层目录的 `results/` 子目录；所有最终输出汇总至 `results/` 目录。严禁将原始数据与中间结果混放。每个分析步骤的输出文件命名格式为 `{步骤编号}_{描述}_{日期}.{扩展名}`，例如 `L1_01_DEG_results_20260622.csv`。

### 7. 代码规范要求
所有Python代码须通过ruff静态检查（ruff check .），所有R代码使用styler格式化。每个分析脚本开头须包含模块文档字符串，说明输入、输出、依赖和运行方式。函数命名使用snake_case，类命名使用PascalCase。**禁止使用try-except: pass吞掉异常**，所有异常必须通过traceback.print_exc()输出完整堆栈。

### 8. 版本控制规范
使用Git进行版本控制，每次完成一个分析步骤后提交一次，提交信息格式为 `[L{n}] {步骤描述}`，例如 `[L1] 完成5个GEO数据集差异表达分析`。不提交数据文件（`data/`与`*.csv`、`*.rds`等加入.gitignore），只提交代码与配置文件。

### 9. 运行日志规范
每个分析脚本运行时自动生成日志文件，存放于 `logs/` 目录。日志文件名与脚本同名，扩展名为 `.log`。日志内容须包含：脚本启动时间、输入文件路径与校验信息（文件大小、行数）、关键参数值、各步骤完成时间、警告信息、错误信息、脚本结束时间。使用Python的logging模块，级别设为INFO。

### 10. 反造假铁律（最高优先级）
所有数据必须来自真实文件读取，禁止生成/模拟任何数据；所有函数名、API、路径必须真实存在，禁止捏造；所有异常必须向上传播，禁止静默吞掉；所有缺失数据必须记录日志警告，禁止静默补零；所有预处理步骤（QC、标准化、批次校正）不可跳过；所有代码报错必须真实报错，禁止伪造成功结果；**训练数据不足时，宁可放弃有监督学习，也不可用计算预测值充当真实标签**。

---

## 第二部分：Phase 1 —— 靶标基因集构建（第11-30点）

### 11. Phase 1 总体目标
构建CIRI背景下铁衰老的核心治疗靶标基因集，作为后续药物筛选的靶标池。目标是通过CIRI差异表达基因、WGCNA加权共表达网络关键模块基因、铁衰老基因集（96个）三者的交集，结合PPI蛋白互作网络拓扑分析，筛选出高置信度核心靶标基因。**注意：不再与FerrDB铁死亡基因取交集，直接使用96个铁衰老基因作为基准。**

### 12. 铁衰老基因集加载与验证（96个基因）
从项目已有文件 `铁衰老基因.txt` 中读取96个铁衰老基因（已更新为96个）。该基因集源自刘黎啸2026 Cell Metabolism论文，包含铁代谢、脂质过氧化、抗氧化系统相关基因。需验证：文件编码为UTF-8，每行一个基因符号，无空行，无重复。使用Python脚本 `L1/load_ferroaging_genes.py` 完成加载，输出基因列表到 `L1/results/ferroaging_genes_96.csv`。**关键地标：** ACSL4、HMOX1、TFRC、GPX4、HIF1A、KEAP1、SOD1、NLRP3、IL6、TLR4、MAPK1等为核心关注基因。

### 13. GEO数据集信息盘点
项目已有5个CIRI相关bulk转录组数据集，均位于 `L1 数据集/bulk/` 目录下：(1) GSE104036——多时序时间点，TC-RNAseq格式；(2) GSE16561——微阵列数据；(3) GSE37587——微阵列数据，非标准化；(4) GSE61616——Affymetrix芯片，7天时间点；(5) GSE97537——Affymetrix芯片，24小时时间点（CIRI急性期关键时间点）。**注意：** 这5个数据集横跨三种不同技术平台（Tag Counting RNA-seq、双通道微阵列、Affymetrix单通道芯片），直接合并存在高风险。

### 14. GSE104036多时序数据集解析
读取 `GSE104036_TC-RNAseq_counts.txt.gz` 计数矩阵文件，解析样本列名，从 `GSE104036_series_matrix.txt.gz` 中提取样本分组与时间点信息。该数据集为TC-RNAseq（Tag Counting RNA-seq）格式，需先进行基因长度校正后再做差异分析。使用Python脚本 `L1/parse_GSE104036.py` 完成解析，输出标准化表达矩阵到 `L1/results/GSE104036_expression_matrix.csv`。

### 15. GSE16561数据集解析
从 `GSE16561_RAW (1).txt.gz` 中提取表达矩阵，从 `GSE16561_series_matrix (1).txt.gz` 中提取表型信息。该数据集为双通道微阵列数据，需进行log2转换和分位数标准化。使用Python脚本 `L1/parse_GSE16561.py` 完成解析。

### 16. GSE37587数据集解析
从 `GSE37587_non-normalized (1).txt.gz` 中提取非标准化表达矩阵，从 `GSE37587_series_matrix (1).txt.gz` 中提取表型信息。注意该数据集为非标准化数据，需进行背景校正和标准化处理。使用Python脚本 `L1/parse_GSE37587.py` 完成解析。

### 17. GSE61616与GSE97537数据集解析（Affymetrix平台）
两个数据集均以RAW tar包形式提供，需先解压获取单个CEL文件（Affymetrix芯片格式），使用R包 `affy` 或 `oligo` 读取CEL文件进行RMA标准化。GSE61616为7天慢性期，GSE97537为24小时急性期，时间点互补，对CIRI时序分析非常重要。从各自的系列矩阵文件中提取表型分组信息。使用R脚本 `L1/parse_affy.R` 完成解析。

### 18. 跨平台整合质控（关键风险防控步骤）
**这是Phase 1最重要的质控环节。** 将5个数据集的标准化表达矩阵按共同基因合并后，运行PCA分析，观察样本是否按技术平台聚团。使用R脚本 `L1/cross_platform_qc.R` 完成。**判断标准：** (1) 若PCA前两个主成分上样本按生物学分组（CIRI vs Sham）分离，而非按平台聚团，说明平台差异可接受，可继续合并分析；(2) 若样本明显按平台聚团（如Affymetrix样本聚在一起、RNA-seq样本聚在一起），说明平台差异远大于生物学差异，**必须改用分平台分析策略**（见第19点）。

### 19. 分平台差异表达分析（备用策略，当跨平台QC未通过时启用）
若第18点PCA显示样本按平台聚团，则废弃ComBat直接合并的方案，改为：对每个数据集独立进行差异表达分析（各自平台使用各自最合适的方法：RNA-seq用DESeq2/edgeR，微阵列用limma），然后使用Robust Rank Aggregation（RRA，R包 `RobustRankAggreg`）对多数据集的差异表达结果进行荟萃分析。RRA方法不要求表达值直接可比，只依赖基因的差异排序，天然规避了跨平台标准化问题。使用R脚本 `L1/per_platform_de_rra.R` 完成。**注意：** 若PCA通过，则仍可用ComBat+limma联合分析（第20点），但须在报告中展示PCA质控图。

### 20. 联合差异表达分析（仅当跨平台QC通过时启用）
对联合表达矩阵使用ComBat（sva R包）校正批次效应，然后使用limma包进行差异表达分析。设计矩阵中包含分组变量（CIRI vs Sham/Control）和数据集来源变量。筛选阈值：|log2 Fold Change| > 0.585 且 adjusted p-value (FDR) < 0.05。注意：ComBat需在log2转换后的表达矩阵上运行，保留疾病/对照分组作为生物学变量不校正。使用R脚本 `L1/differential_expression.R` 完成，输出差异表达基因列表到 `L1/results/DEG_all.csv`。

### 21. 差异表达结果可视化（火山图+诊断图）
使用R包EnhancedVolcano绘制差异表达火山图，标注top 20显著基因。同时绘制以下诊断图：p值分布直方图（检查p值是否均匀分布，排除系统性偏差）、MA图（检查log2FC与表达量的关系）、样本距离热图（检查批次效应/平台效应是否已消除）。使用R脚本 `L1/plot_DEG.R` 完成，输出到 `L1/results/figures/`。

### 22. WGCNA共表达网络构建（基因预过滤）
在联合表达矩阵上运行WGCNA分析。**注意：** 输入矩阵必须预先过滤低表达和低变异基因，建议保留表达量最高的5000-10000个基因（按方差排序），否则WGCNA计算量过大且低信号基因会引入噪声。使用 `varFilter` 或按表达量方差排序取Top N。关键参数：(1) 软阈值β通过 `pickSoftThreshold` 函数自动选择，要求R²>0.85；(2) 使用signed网络类型，minModuleSize=30；(3) 合并相似模块的阈值设为0.25。使用R脚本 `L1/wgcna_analysis.R` 完成。

### 23. WGCNA模块-性状关联分析
将每个WGCNA模块的eigengene（特征基因）与CIRI临床性状（疾病/对照状态、时间点）进行Pearson相关分析。筛选与CIRI状态显著相关的模块（|correlation|>0.5且p<0.05）。使用R包WGCNA的 `moduleTraitCor` 和 `moduleTraitPvalue` 函数。使用R脚本 `L1/wgcna_module_trait.R` 完成，输出模块-性状关联热图到 `L1/results/figures/`。

### 24. 核心基因集汇聚（取交集+预案）
将以下三个基因集取交集：(1) 铁衰老96基因集（来自第12点）；(2) CIRI差异表达基因DEGs（来自第20点或第19点）；(3) WGCNA关键模块基因（来自第23点，取|GS|>0.5且|MM|>0.8的基因）。使用Python脚本 `L1/intersect_gene_sets.py` 完成，输出Venn图到 `L1/results/figures/`。**预案：** 若三交集过小（<5个基因），则放宽条件：(1) WGCNA模块按GS和MM同时排名，取Top 200；(2) 或加入PPI网络一步邻居扩充。**预期交集基因数为5-15个。**

### 25. PPI蛋白互作网络构建
将核心基因集（含预案扩充后的基因）输入STRING数据库（版本12.0），设置物种为Homo sapiens，置信度阈值≥0.4（medium confidence），获取蛋白-蛋白互作网络数据。下载TSV格式文件，保存到 `data/STRING/core_target_ppi.tsv`。使用Python脚本 `L1/fetch_string_ppi.py` 调用STRING API批量获取。

### 26. Cytoscape枢纽基因筛选
在Cytoscape（v3.10+）中导入PPI网络，使用cytoHubba插件进行枢纽基因筛选。选择5种拓扑分析算法：MCC、DMNC、MNC、EPC、BottleNeck。取每种算法排名前10的基因，再取交集作为最终枢纽基因。通过 `cyREST` API自动化操作。注意：ACSL4作为已知铁衰老核心执行者，即使未出现在所有算法交集中，也应手动纳入最终靶标列表。

### 27. GO功能富集分析
使用clusterProfiler R包对核心靶标基因进行GO功能富集分析，覆盖三个本体：BP（生物学过程）、MF（分子功能）、CC（细胞组分）。参数：pvalueCutoff=0.05, qvalueCutoff=0.05。使用R脚本 `L1/go_enrichment.R` 完成，输出气泡图和网络图到 `L1/results/figures/`。

### 28. KEGG通路富集分析
使用clusterProfiler R包对核心靶标基因进行KEGG通路富集分析。重点关注铁死亡通路（hsa04216 Ferroptosis）、HIF-1信号通路（hsa04066）、Nrf2抗氧化通路、炎症通路。使用R脚本 `L1/kegg_enrichment.R` 完成。

### 29. 最终靶标列表确认
综合以下信息确定最终靶标列表（8-15个）：(1) 三交集基因（第24点）；(2) PPI网络hub基因（第26点）；(3) 铁衰老核心基因优先级（ACSL4 > GPX4 = HMOX1 = TFRC > 其他）；(4) 功能富集通路覆盖度（确保靶标覆盖铁死亡、氧化应激、炎症三条关键通路）。最终靶标列表手动确认，记录每个靶标的入选理由，输出到 `L1/results/final_target_list.csv`。

### 30. Phase 1 结果汇总与审查
汇总Phase 1所有产出：最终靶标基因列表（含入选理由、log2FC、PPI度中心性）、DEG结果、WGCNA模块、跨平台QC报告。生成Phase 1总结报告 `L1/results/phase1_summary_report.md`。执行质量检查：所有脚本是否成功运行、日志是否完整、跨平台QC是否通过并记录结果。使用 `ruff check .` 检查Python代码。

---

## 第三部分：Phase 2 —— 多组学特征工程（仅用于后验解释，第31-52点）

### 31. Phase 2 总体目标与使用边界
**关键定位修正：** Phase 2产出的所有多组学特征（单细胞表达、GSVA评分、免疫浸润、PPI拓扑属性等）**仅用于后验解释和文章故事性增强**，**不注入Phase 4/5的药物筛选预测模型**。药物-靶标相互作用（CPI）在分子层面由化合物结构与蛋白口袋的物理化学互补性决定，蛋白在脑缺血中的表达变化与其"能否被小分子结合"没有直接因果关系。将这些特征注入预测模型将引入严重的数据泄露和研究偏倚。

### 32. GSE174574单细胞数据加载
从 `L1 数据集/RNA-seq/GSE174574_10X_organized/` 目录加载6个10X Genomics单细胞样本（Sham_1/2/3, MCAO_1/2/3）。每个样本包含标准10X三件套文件。使用Scanpy的 `sc.read_10x_mtx()` 函数读取，指定 `var_names='gene_symbols'`。使用Python脚本 `L2/sc_load.py` 完成。

### 33. 单细胞数据质量控制
使用Scanpy计算QC指标：每个细胞的基因数、UMI总数、线粒体基因比例。过滤标准：(1) 基因数200-6000；(2) UMI总数1000-40000；(3) 线粒体基因比例<20%（脑组织可放宽至25%）；(4) 核糖体基因比例>5%。使用 `sc.pp.filter_cells` 和 `sc.pp.filter_genes` 函数。使用Python脚本 `L2/sc_qc.py` 完成。

### 34. 单细胞数据标准化与高变基因选择
标准化（normalize_total target_sum=1e4）、log1p转换，选择高变基因（n_top_genes=2000，flavor='seurat_v3'）。使用Python脚本 `L2/sc_preprocess.py` 完成。

### 35. PCA降维与Harmony批次校正
在高变基因上运行PCA（n_comps=50），然后使用Harmony进行批次校正，key='sample'。Harmony在PCA空间上运行，消除不同样本间的技术变异同时保留生物学变异。使用Python脚本 `L2/sc_harmony.py` 完成。

### 36. UMAP可视化与Leiden聚类
在Harmony校正后的PCA空间上计算邻居图（n_neighbors=15, n_pcs=30），然后运行UMAP（min_dist=0.3）。使用Leiden算法进行聚类（resolution=0.8，根据细胞数调整）。使用Python脚本 `L2/sc_cluster.py` 完成。

### 37. 单细胞类型注释
使用CellTypist进行自动细胞类型注释，参考模型根据数据物种选择。对于注释置信度<0.5的细胞，使用已知marker基因手动验证：神经元（Tubb3/Map2）、星形胶质细胞（Gfap/Aqp4）、小胶质细胞（Aif1/Cx3cr1/Tmem119）、少突胶质细胞（Mbp/Mog）、内皮细胞（Cldn5/Pecam1）、周细胞（Pdgfrb）、免疫细胞（Ptprc/Cd3e）。使用Python脚本 `L2/sc_annotation.py` 完成。

### 38. 单细胞铁衰老评分计算
使用scanpy的 `sc.tl.score_genes` 在每个单细胞中计算铁衰老评分，输入基因集为96个铁衰老基因。将评分投影到UMAP图上，比较MCAO组与Sham组在各类细胞类型中的铁衰老评分差异。**此行分析用于后验解释：** 候选化合物靶标富集在哪些细胞类型中表达，而非用于预测模型。使用Python脚本 `L2/sc_ferroaging_score.py` 完成。

### 39. 单细胞伪批量差异表达分析
对每个细胞类型分别进行伪批量差异表达分析（MCAO vs Sham），使用DESeq2。筛选阈值：|log2FC|>1, padj<0.05。**此行分析用于后验解释：** 验证候选靶标在CIRI的单细胞层面是否确实差异表达。使用Python脚本 `L2/sc_pseudobulk_de.py` 完成。

### 40. 单细胞细胞通讯分析（CellChat）
使用CellChat R包分析MCAO vs Sham条件下细胞间配体-受体相互作用的变化。重点关注铁死亡相关配体-受体对（如TFRC-TF铁转运、SLC7A11-谷氨酸转运）。**此行分析用于后验解释：** 为候选化合物靶标提供细胞间调控网络背景。使用R脚本 `L2/cellchat_analysis.R` 完成。

### 41. bulk RNA-seq GSVA通路评分
对Phase 1的bulk表达矩阵，使用GSVA R包计算每个样本的铁死亡通路活性评分。输入基因集：96个铁衰老基因集。输出每个样本的GSVA富集分数，比较CIRI组与Sham组的差异。使用R脚本 `L2/gsva_ferroptosis.R` 完成。

### 42. 免疫浸润分析
使用CIBERSORTx或R包 `immunedeconv` 对bulk表达矩阵进行22种免疫细胞浸润丰度估计。比较CIRI组与对照组免疫细胞组成差异，重点关注巨噬细胞M1/M2极化和中性粒细胞浸润。使用R脚本 `L2/immune_infiltration.R` 完成。

### 43. 靶标蛋白序列特征提取（可用于Phase 4/5的蛋白表征）
从UniProt数据库获取每个核心靶标蛋白的氨基酸序列、结构域注释、翻译后修饰位点、亚细胞定位信息。使用Python脚本 `L2/fetch_uniprot_features.py` 调用UniProt REST API批量获取。输出到 `L2/results/target_protein_features.csv`。**注意：** 序列特征（AAC、PseAAC、ESM-2 embedding）是蛋白的固有属性，与药物结合能力直接相关，可用于Phase 4/5的蛋白表征——这是多组学特征中唯一可以注入预测模型的部分。

### 44. 靶标蛋白结构获取
从PDB数据库或AlphaFold Protein Structure Database获取每个核心靶标蛋白的3D结构。优先选择实验解析结构（X-ray/NMR/Cryo-EM），若无则使用AlphaFold2预测结构。使用Python脚本 `L2/fetch_protein_structures.py` 下载PDB/mmCIF文件到 `data/PDB/` 目录。

### 45. 结合口袋识别
对每个靶标蛋白的3D结构，使用P2Rank或DeepSite工具识别潜在的小分子结合口袋。输出每个口袋的中心坐标（x, y, z）、体积（Å³）、口袋得分。**注意：** 对于ACSL4，参考刘黎啸2026论文中维生素C的结合口袋（T278/S279/T469区域），将该口袋设为第一优先级对接位点。使用Python脚本 `L2/identify_binding_pockets.py` 完成。

### 46. 靶标蛋白分子描述符计算
使用protpy或Protr计算靶标蛋白的分子描述符：氨基酸组成（AAC, 20维）、二肽组成（DC, 400维）、伪氨基酸组成（PseAAC, 50维）、自相关描述符。使用Python脚本 `L2/compute_protein_descriptors.py` 完成。

### 47. 蛋白序列深度学习嵌入（ESM-2）
使用预训练的ESM-2模型（facebook/esm2_t33_650M_UR50D）对每个靶标蛋白的全长序列进行嵌入，取最后一层隐藏状态的均值池化作为蛋白表示向量（1280维），PCA降维至128维。**注意：** ESM-2嵌入是蛋白序列的深度学习表示，直接从氨基酸序列中提取进化信息和结构倾向，与药物结合能力相关，是蛋白特征工程中信息量最大的部分。使用Python脚本 `L2/compute_esm_embeddings.py` 完成。

### 48. 多组学特征整合（仅用于后验报告）
将Phase 2的多组学特征整合为统一的后验解释矩阵：(1) 单细胞表达特征（各细胞类型中铁衰老基因的表达量、MCAO vs Sham伪批量log2FC）；(2) bulk GSVA通路评分；(3) 免疫浸润比例；(4) PPI网络拓扑属性（度中心性、介数中心性）。**这些特征不会进入Phase 4/5的任何预测模型**，仅用于最终报告中候选化合物靶标的后验解读。使用Python脚本 `L2/integrate_explanation_features.py` 完成。

### 49. Phase 2 可视化输出（用于后验解释和文章配图）
生成以下可视化图表：(1) 单细胞UMAP图（按细胞类型着色、按铁衰老评分着色、按分组着色）；(2) 铁衰老基因在各细胞类型中的点图/小提琴图；(3) GSVA铁死亡通路评分箱线图（CIRI vs Sham）；(4) 免疫浸润比例堆叠柱状图；(5) 细胞通讯网络图（MCAO vs Sham差异）。分辨率300 dpi，输出到 `L2/results/figures/`。

### 50. 靶标蛋白特征导出（用于Phase 4/5的蛋白表征）
将蛋白序列/结构特征（第43/46/47点）导出为统一文件，供Phase 4/5的筛选模型使用：(1) 蛋白序列描述符（AAC+DC+PseAAC，约470维）；(2) ESM-2嵌入（128维）；(3) 结合口袋属性（口袋数量、体积、疏水性评分）。使用Python脚本 `L2/export_protein_features.py` 完成，输出到 `L2/results/protein_features_for_screening.csv`。

### 51. Phase 2 结果汇总与审查
汇总Phase 2所有产出：单细胞分析核心发现（铁衰老评分最高的细胞类型、MCAO vs Sham差异细胞类型）、GSVA通路活性、免疫浸润特征、蛋白序列/结构特征。生成Phase 2总结报告 `L2/results/phase2_summary_report.md`。**关键审查：** 确认所有多组学特征（表达量、GSVA、免疫浸润等）未被引入预测模型的特征输入，蛋白序列/结构特征（AAC、PseAAC、ESM-2、口袋属性）是唯一允许进入筛选模型的特征。

### 52. 单细胞铁死亡/铁衰老signature打分（补充分析）
使用AUCell或ssGSEA方法，基于已发表的铁死亡基因signature（如FerrDB Driver/Suppressor基因集）和铁衰老96基因集，在单细胞水平计算通路活性评分。比较不同细胞类型在MCAO vs Sham条件下的通路活性差异。**此行分析用于后验解释：** 帮助理解候选药物靶标的细胞类型特异性。使用R脚本 `L2/sc_aucell.R` 完成。

---

## 第四部分：Phase 3 —— 中药单体数据库构建（第53-72点）

### 53. Phase 3 总体目标
构建一个高质量的中药单体候选库，包含每种单体的化学结构信息（SMILES、分子指纹）、理化性质（分子量、LogP、TPSA、氢键供体/受体数）、已知靶标背景信息（来自数据库的计算预测，仅作参考，不用于模型训练）、来源中药信息，并通过类药性规则和血脑屏障通透性预测进行初步过滤，为Phase 5的基于结构筛选提供候选化合物池。

### 54. TCMSP数据库数据采集
TCMSP（https://tcmsp-e.com/）是本项目中药单体数据的主要来源，包含499味中药、29384种化合物、3311个靶标，提供口服生物利用度（OB）、类药性（DL）、Caco-2渗透性等参数。使用Python脚本 `L3/fetch_tcmsp.py` 下载数据，注意设置合理的请求间隔（≥3秒）避免被封IP。若下载失败，则使用项目已有文件 `SMILES.xlsx` 作为备用。

### 55. HERB数据库数据采集
HERB（http://herb.ac.cn/）包含7263味中药、49258种化合物，提供中药-化合物-靶标-疾病的多维关联。使用Python脚本 `L3/fetch_herb.py` 通过HERB的REST API下载数据。重点获取化合物SMILES、靶标基因名、实验验证类型。

### 56. TCMBank数据库数据采集
TCMBank（https://tcmbank.cn/）提供中药-成分-靶标-疾病的多层关联网络，标注了关联证据类型。使用Python脚本 `L3/fetch_tcmbank.py` 下载化合物-靶标关联数据。

### 57. ETCM 2.0与SymMap数据库补充采集
ETCM 2.0（http://www.tcmip.cn/ETCM2/）提供中药成分的ADMET预测数据。SymMap（http://www.symmap.org/）提供证候-中药-化合物-靶标-疾病的五层映射。使用Python脚本 `L3/fetch_etcm_symmap.py` 下载数据。

### 58. 多数据库整合与去重
将五个数据库的化合物数据整合为一个统一的候选化合物库。使用InChIKey作为去重主键。对于同一化合物在多个数据库中均有记录的情况，保留所有来源信息并标注来源数据库。使用Python脚本 `L3/merge_compound_databases.py` 完成。

### 59. SMILES规范化与标准化
使用RDKit对所有化合物的SMILES字符串进行规范化处理：(1) 去除盐和溶剂（SaltRemover）；(2) 保留最大片段（GetMolFrags）；(3) 中和电荷（Uncharger）；(4) 互变异构体标准化；(5) 生成规范SMILES（CanonSmiles）。对于无法被RDKit解析的SMILES，记录警告日志并排除。使用Python脚本 `L3/standardize_smiles.py` 完成。

### 60. 分子指纹计算
使用RDKit计算三种分子指纹：(1) ECFP4（扩展连接指纹，2048位，半径=2），用于相似性搜索和分子表征；(2) MACCS（166位结构键），用于快速子结构筛选；(3) RDKit 2D描述符（200+维），包括分子量、LogP、TPSA、可旋转键数、氢键供体/受体数、环数、芳香环数等。使用Python脚本 `L3/compute_fingerprints.py` 完成，输出到 `L3/results/compound_fingerprints.csv`。

### 61. Lipinski类药性五规则过滤
对每个化合物进行Lipinski评估：(1) 分子量≤500 Da；(2) LogP≤5；(3) 氢键供体≤5；(4) 氢键受体≤10；(5) 可旋转键≤10。违反规则数≤1的化合物保留，违反≥2的标记为"类药性差"但仍保留。使用RDKit的 `Descriptors` 模块计算各参数。

### 62. 血脑屏障通透性预测
CIRI是脑部疾病，候选单体必须能够穿过血脑屏障（BBB）。使用两种方法预测：(1) RDKit计算TPSA和LogP，TPSA<90 Å²且LogP 1-4的化合物预测为BBB+；(2) 使用预训练的BBB预测模型（如DeepNeuralNet-QSAR的BBB分类器）。BBB预测为阴性的化合物标记为"BBB穿透性差"，在后续筛选中降低权重但不完全排除（因可能存在局部给药或靶向递送策略）。使用Python脚本 `L3/predict_bbb.py` 完成。

### 63. 口服生物利用度评估
口服生物利用度（OB）是中药单体经口服给药的关键参数。TCMSP数据库已提供OB值，对于缺失OB值的化合物，使用SwissADME的预测值或基于分子描述符的OB预测模型补充。OB≥30%的化合物标记为"高生物利用度"。注意：BBB穿透性优先于OB，脑部疾病可能需要局部给药或靶向递送。

### 64. PAINS/毒性风险排除
使用RDKit的PAINS过滤器排除已知的假阳性化合物。同时使用SwissADME的Brenk警报排除含有毒性基团或反应性基团的化合物。PAINS匹配或Brenk警报≥2的化合物排除出候选池。使用Python脚本 `L3/filter_pains_toxicity.py` 完成。

### 65. 已知靶标信息整合（背景知识库，不用于模型训练）
对每个候选化合物，整合五个数据库中的已知靶标信息。构建化合物-靶标二部关联矩阵，值=证据等级。**关键警示：** 这些数据库中的"已知靶标"大多来自TCMSP等系统药理学模型的计算预测，并非实验验证的结合数据。**此矩阵仅作为背景知识库，用于后续结果的参考比对，绝不作为Phase 4/5的模型训练标签。** 使用Python脚本 `L3/integrate_known_targets.py` 完成。

### 66. 化合物-化合物相似性网络构建
基于ECFP4分子指纹，计算所有候选化合物之间的Tanimoto相似系数。构建化合物相似性网络：节点=化合物，边=相似性>0.7的化合物对，边权重=Tanimoto系数。该网络将用于Phase 5的配体相似性搜索和分子聚类。使用Python脚本 `L3/build_compound_similarity_network.py` 完成。

### 67. 候选化合物池统计与质量评估
统计候选化合物池的基本指标：总化合物数、来源数据库分布、类药性合格率、BBB预测阳性比例、OB≥30%比例、PAINS/Brenk排除数、有已知靶标背景信息的化合物比例。生成统计报告 `L3/results/compound_pool_statistics.md`。

### 68. 候选化合物池可视化
绘制以下可视化图表：(1) 化合物分子量分布直方图；(2) LogP vs TPSA散点图（标注BBB+/-区域）；(3) 化合物来源数据库Upset图；(4) 分子指纹PCA降维散点图（按类药性着色）；(5) 化合物相似性网络子图。使用Python脚本输出到 `L3/results/figures/`。

### 69. Phase 3 中间数据导出
将候选化合物池导出为：(1) CSV表格（含化合物ID、名称、SMILES、分子量、LogP、TPSA、BBB预测、OB、类药性评分、来源数据库）；(2) SDF文件（含3D结构，用于分子对接）；(3) Python pickle（含分子指纹矩阵和相似性网络，用于Phase 5）。使用Python脚本 `L3/export_compound_pool.py` 完成。

### 70. 实验验证活性化合物收集（训练数据，严格来源）
**这是Phase 4数据盘点的前置步骤。** 从ChEMBL（v34）、DrugBank（v5.1.10）、BindingDB（2024版）中查询与Phase 1核心靶标基因有**实验验证**活性的化合物：(1) ChEMBL：检索靶标UniProt ID，筛选标准Type='B'（结合测定）、置信度≥8、活性值IC50/Ki/Kd≤10μM；(2) DrugBank：筛选"approved"或"experimental"药物-靶标关系；(3) BindingDB：筛选IC50/Ki/Kd≤10μM的关联。**绝不使用TCMSP/HERB等数据库的计算预测关联作为训练标签。** 使用Python脚本 `L4/collect_experimental_actives.py` 完成，输出到 `L4/results/experimental_actives.csv`。

### 71. Phase 3 结果汇总与审查
汇总Phase 3所有产出：候选化合物池统计表、类药性/BBB/毒性过滤统计、已知靶标背景矩阵（仅供参考）、化合物相似性网络。生成Phase 3总结报告 `L3/results/phase3_summary_report.md`。执行质量检查：确认所有SMILES可被RDKit解析、分子指纹维度正确、无重复化合物。运行 `ruff check .` 检查Python代码。

### 72. 候选化合物3D构象批量生成
使用RDKit的ETKDG方法（v3）为每个候选化合物生成3D构象。参数：numConfs=50（最多生成50个构象），使用MMFF94力场进行能量最小化，选择能量最低的构象作为对接初始构象。输出为SDF文件，保存到 `data/ligands/` 目录。使用Python脚本 `L3/generate_3d_conformers.py` 完成。**注意：** 3D构象生成是分子对接的前置步骤，质量直接影响对接结果。

---

## 第五部分：Phase 4 —— CPI模型训练与中药单体-核心靶标穷举预测（第73-85点）

### 73. Phase 4 总体目标——CPI-IGAE归纳式图神经网络预测
**Phase 4 是本项目的最关键阶段。** 使用ChEMBL/BindingDB/DrugBank等公开数据库的实验验证CPI数据（>10,000对），训练基于CPI-IGAE的归纳式图神经网络模型。该模型通过配体集表征蛋白、加权同质图构建、归纳式图聚合器三大创新，天然支持冷启动——即对训练集中未见过（或极少已知配体）的核心靶标进行预测。训练完成后，对核心靶标（5-15个）× 中药单体（~5,000个）做穷举预测，输出Top候选化合物进入Phase 5结构验证。

**关键澄清：** (1) 训练CPI模型的数据来自ChEMBL/BindingDB中所有人类蛋白的实验验证CPI数据，**不是**来自Phase 1的核心靶标集；(2) 核心靶标（ACSL4、HMOX1等）是预测目标，不是训练数据来源；(3) ChEMBL中IC50/Ki/Kd≤10μM的人类蛋白-化合物对超过10万条，训练样本量完全不是问题；(4) 冷启动是真实挑战——核心靶标在训练集中可能只有极少量已知配体，这正是选择CPI-IGAE的原因。

### 74. 方法基础：CPI-IGAE（万晓喆2022博士论文）
**模型核心创新：**
- **加权同质图：** 将异质的化合物-蛋白关系图转化为同质图。蛋白用其配体集（已知结合化合物的分子指纹均值）表征，而非直接用蛋白序列。这使得新蛋白只要有配体集（来自数据库背景知识），就能生成有效表征。
- **归纳式图神经网络：** 使用GraphSAGE风格的邻居采样和聚合（fanout=[25,15,10]），赋予模型处理冷启动问题的能力——模型推理时不需要目标蛋白在训练集中出现过。
- **端到端学习：** 从加权同质图中学习节点嵌入，通过点积解码预测化合物-蛋白相互作用。训练时使用负采样（1:3），损失函数为BCE Loss。

**仓库地址：** https://github.com/wanxiaozhe/CPI-IGAE
**技术栈：** DGL (Deep Graph Library) + PyTorch + RDKit

### 75. 步骤4.1：从ChEMBL收集CPI训练数据
**数据来源与筛选标准：**
- ChEMBL v34 REST API：筛选 `standard_type` ∈ {IC50, Ki, Kd}，`standard_relation` = '='，`standard_value` ≤ 10,000 nM (10 μM)
- 靶标限定：Homo sapiens，`target_type` = 'SINGLE PROTEIN'，有UniProt ID映射
- 提取字段：化合物ChEMBL ID、SMILES、靶标UniProt ID、活性值（nM）、活性类型、assay ID

**数据量估算：** ChEMBL中IC50/Ki/Kd≤10μM的人类蛋白-化合物对 > 100,000条，去重后预计50,000-80,000对。

**输出文件：** `P4/data/chembl_cpi_raw.csv`（原始）、`P4/data/chembl_cpi_processed.csv`（去重标准化）、`P4/data/compound_smiles.csv`、`P4/data/protein_sequences.fasta`

使用Python脚本 `P4/scripts/download_chembl.py` 完成。

### 76. 步骤4.2：构建加权同质图
**图构建方法：**
- 节点 = 化合物 + 蛋白（所有节点同质化处理）
- 边 = 实验验证的CPI关系，权重 = -log10(活性值_nM / 1e9)
- 化合物特征 = ECFP4分子指纹（2048位，radius=2）
- 蛋白特征 = 其配体集（已知结合化合物的ECFP4指纹均值），若蛋白无已知配体则使用氨基酸组成AAC（20维）+ PseAAC（50维）+ ESM-2嵌入（128维）

**关键设计：** 对于核心靶标（冷启动蛋白），其配体集来自TCMSP/HERB等数据库的计算预测关联（不作为训练标签，仅用于生成蛋白初始表征）。这使得模型在推理时能对核心靶标生成有效的节点嵌入。

**输出文件：** `P4/data/cpi_graph.bin`（DGL图）、`P4/data/node_features.npy`、`P4/data/edge_weights.npy`

使用Python脚本 `P4/scripts/build_cpi_graph.py` 完成。

### 77. 步骤4.3：训练CPI-IGAE模型
**模型架构：**
- 图卷积层 × 3（GraphSAGE归纳式聚合，mean pool，ReLU激活）
- 邻居采样：fanout = [25, 15, 10]
- 边预测解码器：节点嵌入点积 → sigmoid
- 损失函数：BCE Loss + 负采样（1:3）

**训练参数：** batch_size=512, epochs=200（早停patience=30）, lr=0.001（Adam）, 训练/验证/测试=8:1:1

**评估指标：** AUROC、AUPRC（对不平衡数据更敏感）、Accuracy

**冷启动验证：** 按蛋白划分测试集（确保测试蛋白不在训练集中出现），验证模型对未见蛋白的泛化能力。至少保留20%的蛋白作为冷启动测试集。

**输出文件：** `P4/models/best_model.pth`、`P4/results/training_log.csv`、`P4/results/evaluation_metrics.json`

使用Python脚本 `P4/scripts/train_cpi_model.py` 完成。

### 78. 步骤4.4：核心靶标 × 中药单体穷举预测
**预测流程：**
1. 加载Phase 3产出的中药单体库（SMILES + ECFP4分子指纹）
2. 加载Phase 1的核心靶标列表（基因符号 + UniProt ID + 蛋白序列）
3. 为每个核心靶标生成蛋白节点表征（配体集指纹均值 + 序列特征）
4. 构建预测图：将核心靶标节点和中药单体节点加入图中
5. 使用训练好的CPI-IGAE模型做归纳式推理（GraphSAGE不要求目标节点在训练图中）
6. 输出每个化合物-靶标对的预测得分（0-1）

**预测规模：** 核心靶标5-15个 × 中药单体~5,000个 = 25,000-75,000对预测

**输出文件：** `P4/results/cpi_predictions.csv`（全部）、`P4/results/top_predictions.csv`（Top 200）

使用Python脚本 `P4/scripts/predict_core_targets.py` 完成。

### 79. 步骤4.5：结果分析与Top-N候选分子筛选
**分析维度：**
1. 每个核心靶标的Top 20预测化合物（按预测得分降序）
2. 多靶标命中分析：同时被≥3个核心靶标预测为阳性（score>0.7）的化合物
3. 与已知活性化合物的结构相似性（Tanimoto系数，标注已知活性类似物）
4. 基于Phase 1 PPI网络的化合物-靶标网络邻近度分析
5. 类药性（Lipinski五规则）+ BBB穿透性交叉筛选

**输出文件：** `P4/results/per_target_top20.csv`、`P4/results/multi_target_hits.csv`、`P4/results/phase4_summary_report.md`

使用Python脚本 `P4/scripts/analyze_results.py` 完成。

### 80. 配体相似性评分（阳性对照锚定）
从ChEMBL中收集核心靶标的已知活性化合物（即使数量极少），作为阳性对照。将候选化合物与已知活性化合物在ECFP4指纹空间中进行Tanimoto相似性比较。高相似度（Tanimoto>0.7）的候选化合物标注为"已知活性类似物"，给予额外的配体相似性加分。使用Python脚本 `P4/scripts/ligand_similarity.py` 完成。

### 81. 网络邻近度计算
基于Phase 1的PPI网络，计算候选化合物已知靶标（来自TCMSP/HERB等数据库的计算预测关联，仅作背景知识）与CIRI-铁衰老核心靶标之间的网络邻近度（Network Proximity）。邻近度越近，候选化合物通过多靶标效应影响CIRI-铁衰老通路的可能性越大。使用Python脚本 `P4/scripts/network_proximity.py` 完成。

### 82. 多策略打分权重设计
综合Phase 4所有打分因子，设计最终权重：
- CPI-IGAE预测得分（权重50%）——核心指标
- 配体相似性得分（权重15%）——与已知活性的结构相似度
- 网络邻近度得分（权重10%）——多靶标效应的网络拓扑支持
- 类药性+BBB评分（权重15%）——成药性基础
- 多靶标命中加分（权重10%）——被多个核心靶标同时预测为阳性

使用Python脚本 `P4/scripts/design_scoring_weights.py` 完成。

### 83. Top候选化合物短名单确认
使用第82点的权重设计，对中药单体库进行综合排名。对每个核心靶标取Top 20化合物，汇总去重后形成Top 100候选化合物短名单，进入Phase 5的深度验证（分子对接+MD+ADMET）。**优先级规则：** ACSL4靶标的Top化合物优先于其他靶标，多靶标命中化合物优先于单靶标命中。

### 84. Phase 4 结果可视化
生成以下可视化：(1) 训练过程AUROC/AUPRC曲线图；(2) 冷启动 vs 非冷启动蛋白的预测性能对比箱线图；(3) 每个核心靶标的Top 20化合物预测得分柱状图；(4) 化合物-靶标预测得分热图（核心靶标 × Top 50化合物）；(5) 多靶标命中化合物的Upset图。输出到 `P4/results/figures/`，分辨率300 dpi。

### 85. Phase 4 结果汇总与审查
汇总Phase 4所有产出：CPI-IGAE模型训练评估报告、冷启动泛化验证结果、核心靶标×中药单体穷举预测结果、Top 100候选化合物短名单、综合打分排名。生成Phase 4总结报告 `P4/results/phase4_summary_report.md`。**关键记录：** 训练数据量（化合物数、蛋白数、CPI对数）、模型AUROC/AUPRC、冷启动验证AUPRC、各核心靶标的已知配体数量（标注冷启动程度）。执行质量检查：`ruff check P4/`。

---

## 第六部分：Phase 5 —— 主力筛选引擎：基于结构的虚拟筛选+MD验证（第86-110点）

### 86. Phase 5 总体目标
Phase 5是本项目的主力筛选引擎，对Phase 4初筛的Top 100候选化合物（或条件A满足时的更大候选池）进行基于结构的深度虚拟筛选，包括分子对接验证结合模式、药效团建模、分子动力学模拟验证结合稳定性、MM/PBSA结合自由能计算、ADMET成药性预测、网络药理学通路验证，最终输出Top 5-8个经过严格计算机模拟验证的候选中药单体。

### 87. 受体蛋白结构准备
对Phase 1核心靶标蛋白的3D结构进行准备：(1) 使用PyMOL或Biopython去除水分子、配体、离子等非蛋白原子；(2) 使用PDB2PQR或PropKa在pH=7.4条件下分配质子化状态；(3) 使用OpenBabel将格式转换为PDBQT（AutoDock Vina输入格式）；(4) 使用ADFR Suite的 `prepare_receptor` 工具添加Gasteiger电荷。对于ACSL4，参考刘黎啸2026论文中VC结合口袋（T278/S279/T469区域），设置对接盒中心为该口袋的几何中心。使用Python脚本 `L5/prepare_receptor.py` 完成。

### 88. 配体分子结构准备
对Top 100候选化合物和已知活性阳性对照化合物的3D结构进行准备：(1) 使用RDKit从SMILES生成3D构象（ETKDG方法，numConfs=50）；(2) 使用MMFF94力场进行能量最小化，选择能量最低构象；(3) 使用OpenBabel转换为PDBQT格式；(4) 添加Gasteiger电荷，设置可旋转键。使用Python脚本 `L5/prepare_ligands.py` 完成。

### 89. 分子对接（AutoDock Vina，批量运行）
使用AutoDock Vina对每个候选化合物与每个核心靶标进行分子对接。对接参数：(1) exhaustiveness=32；(2) num_modes=20；(3) energy_range=5；(4) 对接盒中心=第45点识别的结合口袋中心，盒子尺寸=25×25×25 Å³。若候选化合物数×靶标数过大（>1000次对接），使用批处理脚本并行运行。使用Python脚本 `L5/run_docking.py` 调用Vina命令行。

### 90. 对接结果分析方法验证
在正式对接前，先进行方法验证：将已知活性化合物（阳性对照）对接到其已知靶标，检查Vina是否能重现已知结合模式（RMSD<2 Å视为重现成功）。若重现率<70%，需调整对接参数（增大exhaustiveness、调整盒子大小）或换用其他对接软件（如Smina、Glide SP）。验证通过后方可进行批量对接。使用Python脚本 `L5/validate_docking.py` 完成。

### 91. 对接结果分析
对每个对接结果进行分析：(1) 结合能（Binding Affinity, kcal/mol），< -7.0为强结合，-7.0至-5.0为中等，> -5.0为弱结合；(2) 结合模式分析：使用PLIP分析氢键、疏水相互作用、π-π堆积、盐桥；(3) 与已知配体/抑制剂的结合模式对比（如ACSL4与VC的结合模式）。使用Python脚本 `L5/analyze_docking.py` 完成。

### 92. 对接结果可视化
使用PyMOL生成以下可视化：(1) 每个化合物-靶标复合物的3D结合模式图（蛋白表面/卡通图+配体球棍模型+氢键虚线）；(2) 配体结合口袋2D相互作用图（使用LigPlot+或PoseView）；(3) 候选化合物与阳性对照（已知抑制剂）的叠合对比图。使用Python脚本 `L5/visualize_docking.py` 调用PyMOL命令行模式。

### 93. 药效团建模（补充筛选维度）
对每个核心靶标，基于其已知活性化合物（阳性对照）构建药效团模型。使用RDKit或LigandScout提取药效团特征（氢键供体/受体、疏水中心、芳香环、正/负电荷中心）。将候选化合物与靶标药效团进行匹配打分，作为分子对接的补充维度。使用Python脚本 `L5/pharmacophore_modeling.py` 完成。

### 94. 对接+药效团共识筛选
综合对接结合能和药效团匹配得分，筛选出Top 10-15候选化合物进入MD模拟验证。筛选标准：对接结合能<-6.0 kcal/mol 且 药效团匹配得分>0.6。使用Python脚本 `L5/consensus_screening.py` 完成。

### 95. 分子动力学模拟准备（GROMACS）
对Top 10-15候选化合物-靶标复合物使用GROMACS进行MD模拟。模拟体系准备：(1) 使用CHARMM36力场描述蛋白和配体（配体参数通过CGenFF生成）；(2) 将复合物置于TIP3P水分子盒子中，盒子边界距蛋白≥12 Å；(3) 添加Na⁺/Cl⁻离子中和体系电荷并达到生理盐浓度（0.15 M NaCl）；(4) 体系总原子数约50000-100000。使用Python脚本 `L5/prepare_md.py` 生成GROMACS输入文件。

### 96. 分子动力学模拟运行
MD模拟流程：(1) 能量最小化（steepest descent, 50000步）；(2) NVT平衡（100 ps, 温度310 K, V-rescale恒温器）；(3) NPT平衡（100 ps, 压力1 bar, Parrinello-Rahman恒压器）；(4) 生产模拟（100 ns, 时间步长2 fs, LINCS约束氢键, 每10 ps保存一帧）。使用GPU加速（CUDA），预计每个模拟6-12小时。使用Python脚本 `L5/run_md.py` 调用GROMACS gmx命令。**注意：** 对ACSL4靶标，参考刘黎啸2026论文的300 ns模拟方案，若有计算资源可延长至200-300 ns以获得更可靠的结合稳定性评估。

### 97. MD轨迹分析
对100 ns MD轨迹进行以下分析：(1) RMSD（蛋白骨架+配体重原子）：RMSD<2 Å且波动<1 Å说明体系稳定；(2) RMSF（蛋白残基均方根波动）：识别结合口袋的柔性/刚性区域；(3) Rg（回转半径）：评估蛋白整体构象紧凑度；(4) 氢键分析：统计配体-蛋白间氢键数量和占有率（occupancy>50%为稳定氢键）；(5) 配体RMSD（相对于对接构象）：<3 Å说明配体在结合口袋中保持稳定。使用Python脚本 `L5/analyze_md.py` 调用GROMACS分析工具和MDTraj库。

### 98. 结合自由能计算（MM/PBSA）
使用gmx_MMPBSA工具对最后20 ns稳定轨迹进行MM/PBSA结合自由能计算。输出：(1) ΔG_bind（总结合自由能，kcal/mol）；(2) 能量分解：ΔG_gas、ΔG_solv、ΔE_ele、ΔE_vdw；(3) 残基能量分解：识别对结合贡献最大的关键残基，与已知结合口袋对比验证。使用Python脚本 `L5/run_mmpbsa.py` 完成。

### 99. ADMET预测
使用SwissADME和ADMETlab 2.0对Top 10-15候选化合物进行ADMET预测。重点关注：(1) 血脑屏障通透性（BBB permeant）；(2) 人体肠道吸收（HIA）；(3) CYP450抑制（CYP2D6, CYP3A4等）；(4) P-gp底物（影响BBB外排）；(5) AMES毒性（致突变性）；(6) hERG抑制（心脏毒性风险）；(7) 肝毒性预测。使用Python脚本 `L5/predict_admet.py` 完成。

### 100. 网络药理学验证
对Top 10候选化合物，进行网络药理学分析：(1) 从ChEMBL/DrugBank/STITCH获取该化合物的所有已知靶标（实验验证+数据库预测）；(2) 对这些靶标进行GO/KEGG富集分析，验证是否富集于铁死亡、氧化应激、炎症等通路；(3) 构建"化合物-靶标-通路-疾病"网络，使用Cytoscape可视化；(4) 验证靶标与CIRI/脑缺血的疾病关联（从DisGeNET/OpenTargets获取）。使用Python/R脚本 `L5/network_pharmacology.py` 完成。

### 101. 多组学后验解释（将Phase 2特征用于故事性增强）
**这是Phase 2多组学特征的正确使用方式。** 对经过分子对接和MD验证的Top 10候选化合物，基于其预测靶标，从Phase 2的多组学数据中提取后验解释：(1) 该靶标在单细胞数据中主要在哪些细胞类型中表达？(2) 该靶标在MCAO vs Sham中是否差异表达？(3) 该靶标所在的细胞类型中铁衰老评分是否升高？(4) 该靶标在PPI网络中是否是hub节点？(5) 该靶标所在的通路在GSVA分析中是否显著激活？这些信息用于增强文章的故事性和生物学合理性，但不参与化合物排名。使用Python脚本 `L5/posthoc_omics_interpretation.py` 完成。

### 102. 候选化合物最终排名（综合打分）
综合Phase 4-5所有验证结果，对Top 10-15候选化合物进行最终排名。排名依据：(1) 对接结合能（权重20%）；(2) MD模拟稳定性（RMSD<2Å，权重15%）；(3) MM/PBSA结合自由能（权重15%）；(4) 药效团匹配得分（权重10%）；(5) 关键残基相互作用一致性（与已知抑制剂对比，权重10%）；(6) ADMET成药性评分（权重10%）；(7) 配体相似性得分（权重5%）；(8) 网络邻近度得分（权重5%）；(9) ML预测得分（权重10%，仅当Phase 4第78点评估通过时启用，否则分配给对接和MD）。使用Python脚本 `L5/final_ranking.py` 计算加权总分并排序。

### 103. 候选化合物综合报告生成
对Top 5-8最终候选化合物，逐条生成综合报告，包含：(1) 化合物基本信息（名称、CAS号、分子式、分子量、SMILES、结构图）；(2) 来源中药信息（中药名称、药用部位、传统功效、现代药理研究）；(3) 预测靶标（Top 5靶标及各自得分）；(4) 对接结果（结合能、结合模式描述、关键相互作用残基）；(5) MD模拟结果（RMSD图、结合自由能、氢键占有率）；(6) ADMET评估（各指标通过/未通过及风险提示）；(7) 网络药理学分析（靶标-通路-疾病网络）；(8) 多组学后验解释（靶标在CIRI单细胞/bulk数据中的表达和通路活性）；(9) 文献支持证据；(10) 总体推荐等级（强烈推荐/推荐/可考虑）。使用Python脚本 `L5/generate_candidate_report.py` 生成Markdown+PDF格式报告。

### 104. 对接结合模式对比分析（关键验证）
对Top 5候选化合物，将其在ACSL4上的对接结合模式与刘黎啸2026论文中维生素C-ACSL4的结合模式进行详细对比。重点比较：(1) 是否与T278/S279/T469口袋结合；(2) 氢键模式是否类似；(3) 疏水接触是否覆盖相似区域。若候选化合物与VC在ACSL4上具有相似的结合模式，则强烈提示其可能具有ACSL4抑制活性。使用Python脚本 `L5/compare_binding_modes.py` 完成。

### 105. 结构新颖性评估
对Top 10候选化合物进行结构新颖性评估：(1) 与已知ACSL4抑制剂/铁死亡抑制剂的Tanimoto相似度（相似度<0.5为结构新颖）；(2) 与TCMSP/HERB数据库中已有靶标注释的化合物的Tanimoto相似度（低相似度意味着未被数据库充分注释，是新发现）。优先推荐结构新颖且结合模式合理的候选化合物。使用Python脚本 `L5/assess_novelty.py` 完成。

### 106. 合成可行性评估
对Top 5候选化合物，使用SYBA（SYnthetic Bayesian Accessibility）或ScScore评估合成可行性。若候选化合物为天然产物且在中药中含量较高，标注"易获取"；若需全合成且合成步数>10步，标注"合成难度高"。该评估影响实验验证的优先级排序。使用Python脚本 `L5/assess_synthesizability.py` 完成。

### 107. 候选化合物-靶标结合自由能分解可视化
对Top 5候选化合物的MM/PBSA残基能量分解结果进行可视化，生成：(1) 每个残基的能量贡献柱状图（正值=不利结合，负值=有利结合）；(2) 关键残基在蛋白3D结构上的着色标注（红色=高贡献，蓝色=低贡献）。使用Python脚本 `L5/visualize_energy_decomposition.py` 完成。

### 108. Phase 5 可视化输出汇总
生成以下可视化图表：(1) 候选化合物对接结合能排名柱状图（按靶标分组）；(2) MD模拟RMSD时间序列图（Top 5候选化合物）；(3) 蛋白-配体氢键占有率热图；(4) MM/PBSA能量分解堆叠柱状图；(5) ADMET雷达图（每个候选化合物一个）；(6) 化合物-靶标-通路-疾病网络图。输出到 `L5/results/figures/`，分辨率300 dpi。

### 109. Phase 5 结果汇总与审查
汇总Phase 5所有产出：对接结果汇总表、MD模拟轨迹文件、MM/PBSA能量分解数据、ADMET预测报告、网络药理学分析结果、多组学后验解释、最终候选化合物排名表、综合报告。生成Phase 5总结报告 `L5/results/phase5_summary_report.md`。执行最终质量检查：确认对接结果合理（结合能<-5 kcal/mol）、MD模拟正常完成（无崩溃）、RMSD在合理范围内。运行 `ruff check .` 检查所有Python代码。

### 110. 对接结果的外部验证（红ocking）
对所有Top 5候选化合物，使用第二种独立对接软件（如Smina或Glide SP）进行交叉验证。若两种对接软件的结合能排名一致（Spearman相关系数>0.7），则对接结果可信；若不一致，需分析原因（如蛋白柔性、水分子介导的相互作用等），并在报告中标注不确定性。使用Python脚本 `L5/cross_validate_docking.py` 完成。

---

## 第七部分：Phase 6 —— 实验验证方案设计（第111-123点）

### 111. Phase 6 总体目标
为Phase 5确定的Top 3-5个候选中药单体设计完整的实验验证方案，包括体外细胞实验和体内动物实验。方案设计参考刘黎啸2026 Cell Metabolism论文的验证体系（特别是ACSL4靶标验证和C11-BODIPY脂质过氧化检测），以及廖昊森2026论文的MCAO大鼠模型验证。该方案作为后续实验执行的蓝图，不在此阶段实际执行实验。

### 112. 体外细胞模型选择与建立
推荐使用两种细胞模型：(1) SH-SY5Y人神经母细胞瘤细胞——最常用的人源神经元模型，适合研究神经保护机制；(2) HT22小鼠海马神经元细胞系——常用于铁死亡研究，对erastin/RSL3诱导的铁死亡敏感。模型建立：使用OGD/R（氧糖剥夺/复氧）模拟CIRI的体外条件。OGD条件：无糖DMEM培养基、95% N₂/5% CO₂、37℃、2-4小时（需预实验优化OGD时间）。复氧条件：正常DMEM完全培养基、95%空气/5% CO₂、37℃、24小时。

### 113. 体外筛选实验设计（初筛，参考刘黎啸2026方案）
(1) 细胞接种于96孔板，5000 cells/孔，3复孔；(2) 候选单体设置5个浓度梯度（1, 5, 10, 25, 50 μM），在OGD前预处理2小时，并在OGD和复氧期间持续给药；(3) 阳性对照：Ferrostatin-1（铁死亡抑制剂，1 μM）或维生素C（ACSL4抑制剂，10 μM）；(4) 阴性对照：DMSO（溶剂对照，浓度<0.1%）；(5) 检测指标：复氧24h后，C11-BODIPY 581/591探针（2 μM, 30 min）检测脂质过氧化水平（流式细胞术），CCK-8检测细胞活力。筛选标准：脂质过氧化降低≥50% 且 细胞活力提升≥30% 的单体进入复筛。

### 114. 体外复筛实验设计（铁死亡特异性验证）
对初筛命中的单体进行铁死亡特异性验证：(1) LDH释放检测（排除坏死/焦亡）；(2) PI/Hoechst双染（区分坏死/凋亡）；(3) Fe²⁺检测（FerroOrange探针）；(4) 脂质过氧化产物：MDA含量（TBARS法）、4-HNE水平（ELISA）；(5) 抗氧化指标：GSH/GSSG比值（DTNB法）、GPX4酶活性（试剂盒）；(6) 铁死亡特异性反向验证：同时加入erastin（10 μM）或RSL3（1 μM）+ 候选单体，观察单体是否能逆转erastin/RSL3诱导的铁死亡。若单体对erastin/RSL3诱导的铁死亡有保护作用，则确认其为铁死亡抑制剂。

### 115. 靶标蛋白Western blot验证
对复筛命中的单体（Top 3-5个），进行靶标蛋白表达水平的Western blot验证。检测蛋白：(1) ACSL4（铁衰老核心执行者）；(2) GPX4（铁死亡关键抑制蛋白）；(3) SLC7A11（xCT，胱氨酸/谷氨酸转运体）；(4) FTH1（铁蛋白重链）；(5) HO-1/HMOX1（血红素加氧酶-1）；(6) Nrf2/p-Nrf2（抗氧化转录因子）；(7) COX2/PTGS2（铁死亡标志物）。内参：β-actin或GAPDH。实验设计：Sham组、OGD/R组、OGD/R+单体（低浓度）、OGD/R+单体（高浓度）、OGD/R+阳性对照，每组3复孔，独立重复3次。

### 116. ACSL4靶标直接结合验证（高优先级候选，参考刘黎啸2026方案）
对于Top 1候选单体，进行靶标直接结合验证：(1) 合成Biotin标记的单体（委托合成公司完成）；(2) Pull-down实验：Biotin-单体与细胞裂解液孵育，Streptavidin磁珠富集结合蛋白；(3) LC-MS/MS鉴定富集蛋白（Orbitrap质谱仪），确认ACSL4是否在富集蛋白列表中；(4) 竞争结合实验：加入过量游离单体竞争，确认结合特异性；(5) 纯化重组蛋白结合实验：表达纯化FLAG-ACSL4蛋白，与Biotin-单体孵育，检测结合信号；(6) 体外酶活实验：使用Acyl-CoA Synthetase Assay Kit，检测单体对ACSL4酶活的剂量依赖性抑制作用。

### 117. 体内动物实验设计（MCAO大鼠模型）
动物模型：SPF级雄性SD大鼠（250-280 g），随机分为6组，每组8-10只：(1) 假手术组（Sham）；(2) MCAO模型组（线栓法，缺血2h/再灌注24h）；(3) MCAO+低剂量单体；(4) MCAO+高剂量单体；(5) MCAO+阳性对照（Fer-1或Edaravone）；(6) MCAO+单体+ACSL4抑制剂（验证ACSL4依赖性）。给药方案：单体在MCAO前30 min腹腔注射（i.p.）或灌胃（i.g.），再灌注后立即再次给药。

### 118. 体内实验检测指标（行为学+组织学）
(1) 神经行为学评分：Bederson评分、平衡木实验、转棒实验、旷场实验，在再灌注24h/48h/72h各评估一次；(2) TTC染色：再灌注24h，测定脑梗死体积百分比；(3) HE染色：神经元形态、空泡化、核固缩；(4) Nissl染色：存活神经元计数；(5) 免疫组化（IHC）/免疫荧光（IF）染色：ACSL4、GPX4、4-HNE、Iba-1（小胶质细胞）、GFAP（星形胶质细胞）。

### 119. 体内实验检测指标（生化+分子）
(6) 脑组织匀浆生化检测：MDA、GSH/GSSG、Fe²⁺、SOD活性；(7) RT-qPCR：所有核心靶标基因mRNA表达水平；(8) Western blot：同第115点的蛋白panel（ACSL4、GPX4、SLC7A11、FTH1、HO-1、Nrf2/p-Nrf2、COX2）。

### 120. 脂质组学分析（可选，高优先级候选，参考刘黎啸2026方案）
对Top 1候选单体处理的MCAO大鼠脑组织，进行脂质组学分析：(1) Bligh-Dyer法提取脂质；(2) LC-MS/MS（QTRAP 6500 PLUS或同等设备）分析PUFA-磷脂（PE/PC）；(3) 检测ACSL4直接产物（20:4-CoA, 22:4-CoA）和下游氧化磷脂。预期结果：候选单体处理组PUFA-磷脂水平降低，证实ACSL4抑制效应。该分析是验证ACSL4靶标抑制的金标准方法。

### 121. 剂量-效应与时间-效应关系分析
在体外实验中，对Top 1-2候选单体进行完整的剂量-效应曲线（7个浓度点，3倍稀释）和时间-效应曲线（0, 3, 6, 12, 24, 48h），计算IC50值和最佳作用时间。在体内实验中，设置3个剂量组（低、中、高），确定最佳有效剂量。使用GraphPad Prism或R包 `drc` 拟合剂量-效应曲线。

### 122. 安全性初步评估
对Top 3候选单体进行安全性初步评估：(1) 体外：正常细胞（非OGD处理）的CCK-8毒性测试，计算SI（选择指数）= IC50(正常细胞)/IC50(OGD模型)；(2) 体内：监测体重变化、肝肾功生化指标（ALT、AST、BUN、Cr）、主要脏器HE染色。SI>5且无明显体内毒性者优先推荐。

### 123. 实验验证方案总结与质控标准
汇总Phase 6实验验证方案，确保以下关键质控节点：(1) 所有实验组至少3次独立重复；(2) 所有定量数据使用mean±SD表示，统计方法使用one-way ANOVA+Tukey post-hoc或Kruskal-Wallis+Dunn's post-hoc；(3) P<0.05为统计学显著，标注具体的P值和统计量；(4) 盲法实验：实验操作者和数据分析者均不知晓分组信息；(5) 阳性对照必须有效（验证模型建立成功）；(6) 所有原始数据（Western blot原始图像、流式数据文件、行为学视频）妥善保存。生成Phase 6实验方案文档 `L6/experimental_protocol.md`。

---

## 附录：项目里程碑与关键决策点

| 阶段 | 里程碑 | 关键决策点 |
|------|--------|------------|
| Phase 1 | 靶标基因集确认 | 跨平台QC是否通过？→ 决定分平台RRA还是联合limma |
| Phase 2 | 多组学特征工程完成 | 确认特征仅用于后验解释，不注入预测模型 |
| Phase 3 | 候选化合物池构建 | 实验活性数据开始收集 |
| Phase 4 | CPI-IGAE模型训练与穷举预测 | **冷启动验证AUPRC>0.5？→ 决定CPI预测可信度** |
| Phase 5 | 主力筛选引擎运行 | 分子对接+MD+MM/PBSA+ADMET完成，Top 5-8候选确认 |
| Phase 6 | 实验方案设计 | 完整实验方案文档交付 |

---

*文档版本: v3.0（修订版）| 修订日期: 2026-06-22 | 修订说明: Phase 4 重写为CPI-IGAE归纳式图神经网络预测方案，训练数据来自ChEMBL/BindingDB（不依赖核心靶标），模型天然支持冷启动*