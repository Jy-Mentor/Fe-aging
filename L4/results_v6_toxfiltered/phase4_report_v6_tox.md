# Phase 4 v6: GAT + HGT 双图神经网络集成 — 训练报告（毒性过滤后重训练）

生成时间: 2026-06-24 00:07:55
总耗时: 4.4 分钟

## 1. 架构
- **GAT 分支**: 同质图（化合物 + 蛋白），化合物和蛋白均为独立 MLP 编码器，边 = 实验 CPI
- **HGT 分支**: 异质图（化合物、蛋白、KEGG 通路），HGTConv 捕捉多靶标协同模式
- **集成**: 加权平均 (GAT 0.5 + HGT 0.5)
- **训练数据**: ChEMBL / BindingDB 实验验证 CPI
- **预测范围**: 铁衰老温靶标
- **TCM 候选池**: 经 L3 毒性过滤（剔除致癌物/致突变物）后的化合物池

## 2. 关键参考
- GAT: Velickovic et al. (2018) ICLR, https://arxiv.org/abs/1710.10903
- HGT: Hu et al. (2020) WWW, https://arxiv.org/abs/2003.01332
  - 官方代码: https://github.com/acbull/pyHGT
  - PyG 实现: torch_geometric.nn.HGTConv
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
- ECFP4: Rogers & Hahn (2010) J. Chem. Inf. Model. 50(5):742-754
- MACCS keys: MDL Information Systems (now BIOVIA)
- RDKit: Landrum G., https://github.com/rdkit/rdkit
- STRING PPI: Szklarczyk et al. (2023) Nucleic Acids Res. 51(D1):D638-D646
- ChEMBL: Mendez et al. (2019) Nucleic Acids Res. 47(D1):D930-D940
- BindingDB: Gilson et al. (2016) Nucleic Acids Res. 44(D1):D1045-D1053

## 3. 数据规模
- TCM 候选池: 532 个化合物（已剔除毒性/致癌物）
- 铁衰老靶标: 23 个基因

## 3.1 管线自检结果
- 总体状态: **PASSED_WITH_WARNINGS**
- 严重性: WARNING
- WARNINGS: 8 条
  - CPI数据含 185 个无效SMILES（将被图构建跳过）
  - CPI数据含 22260 条重复(gene, SMILES)条目
  - 稀疏靶标 (<10条CPI): 10个 — ['VDAC3', 'TFRC', 'VDAC2', 'SP1', 'LCN2', 'SLC3A2', 'ACSL3', 'FTH1', 'LPCAT3', 'SAT1']
  - 低样本靶标 (10-49条): 3个
  - TCM池与训练集有 3 个重叠化合物（可能数据泄漏）

## 4. 模型性能
- GAT best val_auc: 0.9540
- HGT best val_auc: 0.9864

## 5. Top 20 候选化合物

| rank | MOL_ID | molecule_name | composite_score | avg_score | max_score | n_hits | n_high | top_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MOL010023 | senkirkine | 0.8963 | 0.5680 | 0.8897 | 14 | 9 | HIF1A(0.890), NOX4(0.873), PTGS2(0.842), CTSB(0.824), IL1B(0.809) |
| 2 | MOL009411 | protostemotinine | 0.7685 | 0.4755 | 0.9336 | 12 | 7 | HIF1A(0.934), KEAP1(0.802), PTGS2(0.775), NOX4(0.766), IL1B(0.749) |
| 3 | MOL005317 | Deoxyharringtonine | 0.7180 | 0.4591 | 0.9380 | 12 | 5 | HIF1A(0.938), KEAP1(0.773), PDE4B(0.746), CTSB(0.734), PTGS2(0.709) |
| 4 | MOL003633 | Oxynarcotine | 0.7002 | 0.4712 | 0.9127 | 11 | 6 | HIF1A(0.913), NOX4(0.882), ALOX15(0.847), PTGS2(0.823), DYRK1A(0.795) |
| 5 | MOL004759 | napelline | 0.6818 | 0.4463 | 0.9119 | 12 | 6 | HIF1A(0.912), PTGS2(0.830), NOX4(0.813), KEAP1(0.766), DYRK1A(0.753) |
| 6 | MOL009789 | diosbulbin C | 0.6813 | 0.4405 | 0.9242 | 11 | 6 | HIF1A(0.924), ALOX15(0.835), PTGS2(0.832), NOX4(0.799), CTSB(0.739) |
| 7 | MOL000332 | n-coumaroyltyramine | 0.6790 | 0.4568 | 0.9441 | 12 | 3 | CTSB(0.944), PTGS2(0.830), KEAP1(0.726), EPHA4(0.683), IL1B(0.670) |
| 8 | MOL003494 | orbiculin D | 0.6789 | 0.4316 | 0.9577 | 11 | 5 | HIF1A(0.958), PTGS2(0.800), KEAP1(0.779), ALOX15(0.772), NOX4(0.753) |
| 9 | MOL002034 | (5aR,8aS,9R)-9-(3,4,5-trimethoxyphenyl)-5a,6,8a,9-tetrahydro-5H-isobenzofurano[5,6-f][1,3]benzodioxol-8-one | 0.6767 | 0.4566 | 0.9315 | 10 | 5 | HIF1A(0.932), NOX4(0.858), ALOX15(0.850), PTGS2(0.848), KEAP1(0.725) |
| 10 | MOL009586 | isoverticine | 0.6730 | 0.4292 | 0.9397 | 12 | 5 | HIF1A(0.940), DYRK1A(0.793), PTGS2(0.741), NOX4(0.719), CTSB(0.711) |
| 11 | MOL005409 | anisodamine | 0.6653 | 0.4649 | 0.8411 | 12 | 6 | PTGS2(0.841), HIF1A(0.827), CTSB(0.786), KEAP1(0.754), NOX4(0.742) |
| 12 | MOL005248 | gibberellin A29 | 0.6549 | 0.4274 | 0.9238 | 10 | 6 | HIF1A(0.924), PTGS2(0.861), KEAP1(0.821), ALOX15(0.791), NOX4(0.771) |
| 13 | MOL004306 | Zedoalactone B | 0.6489 | 0.4436 | 0.8954 | 10 | 6 | HIF1A(0.895), PTGS2(0.871), ALOX15(0.869), NOX4(0.857), KEAP1(0.721) |
| 14 | MOL004313 | Zedoarolide B | 0.6471 | 0.4391 | 0.8999 | 10 | 6 | HIF1A(0.900), PTGS2(0.874), NOX4(0.841), ALOX15(0.835), KEAP1(0.778) |
| 15 | MOL008475 | Mitraphyllic acid | 0.6462 | 0.4471 | 0.8536 | 12 | 6 | HIF1A(0.854), KEAP1(0.840), PTGS2(0.774), NOX4(0.720), MAPK1(0.714) |
| 16 | MOL008901 | Pseudolaric acid C | 0.6439 | 0.4469 | 0.8811 | 12 | 5 | HIF1A(0.881), PTGS2(0.855), KEAP1(0.821), NOX4(0.802), ALOX15(0.776) |
| 17 | MOL011075 | Shikodonin | 0.6383 | 0.4349 | 0.9035 | 11 | 5 | HIF1A(0.903), PTGS2(0.850), KEAP1(0.807), NOX4(0.793), ALOX15(0.753) |
| 18 | MOL000470 | 8-C-α-L-arabinosylluteolin | 0.6361 | 0.4283 | 0.9062 | 11 | 5 | HIF1A(0.906), PTGS2(0.856), ALOX15(0.815), NOX4(0.794), KEAP1(0.729) |
| 19 | MOL010828 | cynaropicrin | 0.6356 | 0.4287 | 0.8931 | 12 | 5 | HIF1A(0.893), PTGS2(0.874), NOX4(0.818), KEAP1(0.805), ALOX15(0.793) |
| 20 | MOL004305 | Zedoalactone A | 0.6326 | 0.4266 | 0.8931 | 10 | 6 | HIF1A(0.893), PTGS2(0.874), ALOX15(0.825), NOX4(0.821), KEAP1(0.749) |


## 6. 局限
- 训练数据与铁衰老靶标不完全匹配，跨靶标泛化可能有偏差
- 未使用真实 inactive/decoy 负样本，负样本为随机采样
- 通路特征使用 one-hot 编码，信息量有限
- 集成权重 (0.5/0.5) 为简单平均，未优化
- TCM 候选池（经毒性过滤后）包含 532 个唯一 SMILES