# Phase 4 v6: GAT + HGT 双图神经网络集成 — 训练报告

生成时间: 2026-06-23 21:17:35
总耗时: 4.5 分钟

## 1. 架构
- **GAT 分支**: 同质图（化合物 + 蛋白），边 = 实验 CPI，GATConv 捕捉局部结合模式
- **HGT 分支**: 异质图（化合物、蛋白、KEGG 通路），PPI + 通路边捕捉多靶标协同模式
- **集成**: 加权平均 (GAT 0.5 + HGT 0.5)
- **训练数据**: ChEMBL / BindingDB 实验验证 CPI
- **预测范围**: 铁衰老温靶标

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
- TCM 候选池: 574 个化合物
- 铁衰老靶标: 23 个基因

## 4. 模型性能
- GAT best val_auc: 0.9540
- HGT best val_auc: 0.9865

## 5. Top 20 候选化合物

| rank | MOL_ID | molecule_name | composite_score | avg_score | max_score | n_hits | n_high | top_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MOL010023 | senkirkine | 0.8892 | 0.5607 | 0.8783 | 14 | 9 | HIF1A(0.878), NOX4(0.862), PTGS2(0.835), CTSB(0.815), IL1B(0.805) |
| 2 | MOL008146 | Riddelline | 0.8383 | 0.5524 | 0.8609 | 13 | 8 | PTGS2(0.861), NOX4(0.855), HIF1A(0.839), KEAP1(0.838), ALOX15(0.812) |
| 3 | MOL009411 | protostemotinine | 0.7506 | 0.4682 | 0.9295 | 11 | 7 | HIF1A(0.929), KEAP1(0.788), PTGS2(0.768), NOX4(0.757), IL1B(0.746) |
| 4 | MOL003633 | Oxynarcotine | 0.6963 | 0.4661 | 0.9046 | 11 | 6 | HIF1A(0.905), NOX4(0.870), ALOX15(0.835), PTGS2(0.819), DYRK1A(0.789) |
| 5 | MOL009789 | diosbulbin C | 0.6816 | 0.4384 | 0.9191 | 11 | 6 | HIF1A(0.919), ALOX15(0.827), PTGS2(0.824), NOX4(0.789), IL1B(0.732) |
| 6 | MOL004759 | napelline | 0.6813 | 0.4422 | 0.9084 | 12 | 6 | HIF1A(0.908), PTGS2(0.824), NOX4(0.803), KEAP1(0.768), DYRK1A(0.747) |
| 7 | MOL003494 | orbiculin D | 0.6766 | 0.4226 | 0.9538 | 11 | 5 | HIF1A(0.954), PTGS2(0.789), KEAP1(0.779), ALOX15(0.758), NOX4(0.730) |
| 8 | MOL005317 | Deoxyharringtonine | 0.6733 | 0.4514 | 0.9280 | 11 | 4 | HIF1A(0.928), KEAP1(0.758), PDE4B(0.729), CTSB(0.705), PTGS2(0.694) |
| 9 | MOL002034 | (5aR,8aS,9R)-9-(3,4,5-trimethoxyphenyl)-5a,6,8a,9-tetrahydro-5H-isobenzofurano[5,6-f][1,3]benzodioxol-8-one | 0.6708 | 0.4497 | 0.9217 | 10 | 5 | HIF1A(0.922), PTGS2(0.845), ALOX15(0.845), NOX4(0.837), KEAP1(0.715) |
| 10 | MOL005248 | gibberellin A29 | 0.6587 | 0.4273 | 0.9213 | 10 | 6 | HIF1A(0.921), PTGS2(0.854), KEAP1(0.819), ALOX15(0.785), NOX4(0.763) |
| 11 | MOL000332 | n-coumaroyltyramine | 0.6534 | 0.4384 | 0.9351 | 11 | 3 | CTSB(0.935), PTGS2(0.828), KEAP1(0.708), ALOX15(0.644), EPHA4(0.637) |
| 12 | MOL009586 | isoverticine | 0.6473 | 0.4215 | 0.9370 | 12 | 4 | HIF1A(0.937), DYRK1A(0.787), PTGS2(0.729), NOX4(0.710), ALOX15(0.675) |
| 13 | MOL004313 | Zedoarolide B | 0.6466 | 0.4380 | 0.8928 | 10 | 6 | HIF1A(0.893), PTGS2(0.872), NOX4(0.839), ALOX15(0.835), KEAP1(0.760) |
| 14 | MOL004306 | Zedoalactone B | 0.6464 | 0.4415 | 0.8876 | 10 | 6 | HIF1A(0.888), PTGS2(0.870), ALOX15(0.867), NOX4(0.851), IL1B(0.716) |
| 15 | MOL008901 | Pseudolaric acid C | 0.6460 | 0.4466 | 0.8751 | 12 | 5 | HIF1A(0.875), PTGS2(0.852), KEAP1(0.810), NOX4(0.800), ALOX15(0.778) |
| 16 | MOL000470 | 8-C-α-L-arabinosylluteolin | 0.6435 | 0.4330 | 0.9027 | 11 | 5 | HIF1A(0.903), PTGS2(0.851), ALOX15(0.811), NOX4(0.794), KEAP1(0.727) |
| 17 | MOL011075 | Shikodonin | 0.6418 | 0.4357 | 0.8993 | 11 | 5 | HIF1A(0.899), PTGS2(0.847), KEAP1(0.803), NOX4(0.789), ALOX15(0.755) |
| 18 | MOL005409 | anisodamine | 0.6407 | 0.4446 | 0.8353 | 11 | 6 | PTGS2(0.835), HIF1A(0.795), CTSB(0.771), KEAP1(0.725), IL1B(0.707) |
| 19 | MOL008475 | Mitraphyllic acid | 0.6344 | 0.4423 | 0.8450 | 13 | 5 | HIF1A(0.845), KEAP1(0.832), PTGS2(0.765), NOX4(0.712), MAPK1(0.703) |
| 20 | MOL010828 | cynaropicrin | 0.6342 | 0.4263 | 0.8849 | 12 | 5 | HIF1A(0.885), PTGS2(0.870), NOX4(0.807), KEAP1(0.789), ALOX15(0.784) |


## 6. 局限
- 训练数据与铁衰老靶标不完全匹配，跨靶标泛化可能有偏差
- 未使用真实 inactive/decoy 负样本
- 通路特征使用 one-hot 编码，信息量有限
- 集成权重 (0.5/0.5) 为简单平均，未优化
- TCM 候选池（L3 输出）包含 574 个唯一 SMILES，已做去重与一致性校验