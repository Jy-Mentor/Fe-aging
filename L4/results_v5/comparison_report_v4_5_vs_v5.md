# Phase 4 v4.5 vs v5 模型对比报告

生成时间: 2026-06-23 14:57:09

## 1. 叙事逻辑差异

| 维度 | v4.5 | v5 |
| --- | --- | --- |
| 学习范式 | 每个靶标独立二分类 | 多任务阳性-未标记学习（PU） |
| 负样本 | 其他靶标活性分子作为伪阴性 | 未标记样本不强制为阴性 |
| 模型结构 | RF/XGB/LR/SVM/KNN 集成 | 共享编码器 + 38 任务特定头 |
| 验证方式 | 5 折 CV + Murcko 骨架分组 | 单折 Murcko 骨架切分 |
| 特征 | ECFP4 + 蛋白 AAC/PseAAC 交互 | ECFP4 + MACCS + RDKit 2D |

## 2. 总体性能

| 指标 | v4.5 | v5 |
| --- | --- | --- |
| 可训练/可评估靶标数 | 19 | 19 |
| 平均 AUC | 0.9692 | 0.9344 |
| 平均 AUPR | 0.9389 | 0.8460 |
| EF@5% 均值 | 10.07 | 0.00 |
| 模型数量 | 95 | 1 |
| 可训练参数 | - | 2,621,734 |
| 训练耗时 | 6.0 分钟 | 3.0 分钟 |

## 3. 靶标级 AUC/AUPR 对比

| gene | v4.5_ensemble_auc | v4.5_ensemble_aupr | n_real_pos_x | v5_val_auc | v5_val_aupr | n_real_pos_y |
| --- | --- | --- | --- | --- | --- | --- |
| GPX4 | 0.9269 | 0.8718 | 72.0 | 1.0 | 1.0 | 9.0 |
| LGMN | 0.9787 | 0.9243 | 391.0 | 1.0 | 1.0 | 150.0 |
| CXCL10 | 0.9642 | 0.9089 | 203.0 | 1.0 | 1.0 | 32.0 |
| CD74 | 0.958 | 0.8765 | 25.0 | 1.0 | 1.0 | 3.0 |
| BCL6 | 0.994 | 0.9879 | 1571.0 | 0.9995 | 0.9971 | 285.0 |
| IL1B | 0.9859 | 0.9588 | 618.0 | 0.9988 | 0.9895 | 119.0 |
| KDM6B | 0.974 | 0.9354 | 413.0 | 0.9963 | 0.9488 | 47.0 |
| TLR4 | 0.945 | 0.8848 | 191.0 | 0.9891 | 0.9142 | 53.0 |
| PDE4B | 0.9817 | 0.9917 | 6126.0 | 0.9819 | 0.9759 | 1125.0 |
| TP53 | 0.9876 | 0.9688 | 841.0 | 0.9809 | 0.9157 | 92.0 |
| CTSB | 0.9841 | 0.9821 | 2429.0 | 0.9792 | 0.9543 | 529.0 |
| DYRK1A | 0.9787 | 0.989 | 4814.0 | 0.9778 | 0.9616 | 1016.0 |
| NFE2L2 | 0.9877 | 0.9609 | 618.0 | 0.9738 | 0.922 | 97.0 |
| PTGS2 | 0.9738 | 0.9851 | 4086.0 | 0.9635 | 0.9338 | 827.0 |
| MAPK1 | 0.9695 | 0.9927 | 12481.0 | 0.9609 | 0.9737 | 2495.0 |
| STAT3 | 0.9756 | 0.9747 | 1688.0 | 0.9527 | 0.8932 | 599.0 |
| EPHA4 | 0.9744 | 0.9533 | 313.0 | 0.9259 | 0.684 | 57.0 |
| HMOX1 | 0.9299 | 0.8231 | 23.0 | 0.8278 | 0.0096 | 6.0 |
| LCN2 | nan | nan | nan | 0.246 | 0.0007 | 1.0 |
| SLC7A11 | 0.9452 | 0.8701 | 36.0 | nan | nan | nan |

## 4. Top 候选化合物对比

### v4.5 Top 5
| rank | molecule_name | composite_score | avg_score | max_score | top_targets |
| --- | --- | --- | --- | --- | --- |
| 1 | moracin O | 0.3417878523605392 | 0.1655635369274144 | 0.9147188663482666 | CTSB(0.915), EPHA4(0.758), PTGS2(0.359), TLR4(0.334), STAT3(0.295) |
| 2 | 4',5',7-trimethyl-3-methoxyflavone | 0.3277889689273401 | 0.2257549450111885 | 0.7158811986446381 | DYRK1A(0.716), PTGS2(0.638), PDE4B(0.632), EPHA4(0.514), MAPK1(0.478) |
| 3 | isocorynantheic acid | 0.3271917826474588 | 0.2317217258216503 | 0.7663595378398895 | KDM6B(0.766), PTGS2(0.541), STAT3(0.470), DYRK1A(0.439), TLR4(0.433) |
| 4 | Glycoside K_qt | 0.283040988282137 | 0.0975656500745875 | 0.7643277049064636 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 5 | 14-deoxyandrographolide | 0.283040988282137 | 0.0975656500745875 | 0.7643277049064636 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |

### v5 Top 5
| rank | molecule_name | composite_score | avg_score | max_score | top_targets |
| --- | --- | --- | --- | --- | --- |
| 1 | Flazin | 0.39073392795889 | 0.275993138551712 | 0.9988771080970764 | DYRK1A(0.999), MAPK1(0.931), CTSB(0.725), PTGS2(0.544), IRF1(0.500) |
| 2 | Perlolyrine | 0.3903849509201552 | 0.2629144489765167 | 0.9999991655349731 | DYRK1A(1.000), CTSB(0.877), PTGS2(0.791), MAPK1(0.734), IRF1(0.500) |
| 3 | cynaropicrin | 0.381426520410337 | 0.2662492394447326 | 0.997490406036377 | MAPK1(0.997), PTGS2(0.918), TLR4(0.858), IRF1(0.500), LACTB(0.500) |
| 4 | izmirine | 0.3792345715196509 | 0.2571979165077209 | 0.9995055198669434 | PTGS2(1.000), MAPK1(0.984), CTSB(0.715), IRF1(0.500), LACTB(0.500) |
| 5 | Cryptopin | 0.3789180797965903 | 0.2604005038738251 | 0.9998445510864258 | CTSB(1.000), DYRK1A(0.972), MAPK1(0.947), IRF1(0.500), LACTB(0.500) |

### 重叠情况
- v4.5 Top 20: ['(2E,4Z)-5-(1,3-benzodioxol-5-yl)-1-piperidino-penta-2,4-dien-1-one', '14-deoxyandrographolide', '2-Benzo[1,3]dioxol-5-yl-5,7-dimethoxy-chroman', "4',5',7-trimethyl-3-methoxyflavone", 'Aposiopolamine', 'Ariskanin A', 'Dehydrocorybulbine', 'Dehydrocorydalmine', 'Demethoxycapillarisin', 'GA120', 'Glycoside K_qt', 'florilenalin isobutyrate', 'isocorynantheic acid', 'isogosferol', 'kadsurin B', 'moracin O', 'pachypodol', 'pinoresinol-4-O-beta-D-apiosyl-beta-D-glucopyranoside', 'poriol', 'protostemotinine']
- v5 Top 20: ['(4aR,5R,8R,8aR)-5,8-dihydroxy-3,5,8a-trimethyl-6,7,8,9-tetrahydro-4aH-benzo[f]benzofuran-4-one', 'Cryptopin', 'Demethoxycapillarisin', 'Flazin', 'Isocorypalmine', 'Isolicoflavonol', 'Licoagroisoflavone', 'Moracin D', 'Norartocarpetin', 'Perlolyrine', 'Prostaglandin B1', 'Rubrofusarin', 'Yinyanghuo C', '[(1S)-3-[(E)-but-2-enyl]-2-methyl-4-oxo-1-cyclopent-2-enyl] (1R,3R)-3-[(E)-3-methoxy-2-methyl-3-oxoprop-1-enyl]-2,2-dimethylcyclopropane-1-carboxylate', 'capillarisin', 'cynaropicrin', 'inulicin', 'izmirine', 'melianoninol', 'quindoline']
- 共同候选: ['Demethoxycapillarisin']

## 5. 关键发现

1. **v5 平均 AUC 低于 v4.5**：v4.5 ensemble AUC 0.9692，v5 val AUC 0.9374。
   这符合预期：v4.5 的负样本来自其他靶标活性分子，任务更简单；
   v5 的 PU 评估在 held-out 阳性与大量未标记样本之间进行，挑战性更高。

2. **v5 存在过置信现象**：部分 TCM 化合物在 DYRK1A、PDE4B 等任务上预测概率为 1.000，
   可能由 sigmoid 输出在强阳性特征上的饱和导致，建议后续加入标签平滑或温度缩放。

3. **无数据靶标输出默认 0.5**：v5 对 13 个没有已知阳性数据的靶标（如 EMP1、ACSL4 等）
   输出接近 0.5 的概率，这些分数不应被解释为真实活性预测。

4. **v5 富集因子下降**：v5 的 EF@5% 显著低于 v4.5，主要原因是 v5 对 TCM 中真实阳性
   （PTGS2、STAT3 各 1 个）的排序不够靠前。

5. **v5 效率更高**：单模型、3 分钟完成训练，参数量 2.62M，
   在部署和维护成本上优于 v4.5 的 95 模型集成。

## 6. 建议

- 对 v5 实施温度缩放（Temperature Scaling）或标签平滑，缓解过置信。
- 为无数据靶标显式标记 'NO_DATA'，避免 0.5 默认分被误读。
- 引入真实 inactive/decoy 负样本，进一步校准 PU 损失。
- 尝试多折 Murcko 切分或时间切分，获得更稳健的 v5 性能估计。
- 若追求最高 CV AUC，可保留 v4.5；若追求统一化学空间与部署效率，v5 更具扩展性。

## 7. 关键参考
### 分子表示与化学信息学工具
- Rogers & Hahn (2010) Extended-Connectivity Fingerprints. J. Chem. Inf. Model. 50(5):742-754. doi:10.1021/ci100050t
- MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys
- RDKit: Landrum G., open-source cheminformatics toolkit, https://github.com/rdkit/rdkit
- Murcko scaffold: Bemis & Murcko (1996) The Properties of Known Drugs. 1. Molecular Frameworks. J. Med. Chem. doi:10.1021/jm9602928

### 阳性-未标记学习（PU Learning）
- Kiryo et al. (2017) Positive-Unlabeled Learning with Non-Negative Risk Estimator. NeurIPS. arXiv:1703.00593
- nnPU 官方实现: https://github.com/kiryor/nnPUlearning
- Hao et al. (2024) PU-Learning-Based Data Augmentation for Multitarget Drug Discovery. Int. J. Mol. Sci. doi:10.3390/ijms25158239

### 药物-靶标相互作用深度学习方法
- DeepPurpose: Huang et al. (2020) DeepPurpose: a deep learning library for drug-target interaction prediction. Bioinformatics. doi:10.1093/bioinformatics/btaa1005; https://github.com/kexinhuang12345/DeepPurpose
- MolTrans: Huang et al. (2021) MolTrans: Molecular Interaction Transformer for drug-target interaction prediction. Bioinformatics. doi:10.1093/bioinformatics/btaa880; https://github.com/kexinhuang12345/MolTrans

### TCM 数据库
- TCMSP: Ru et al. (2014) TCMSP: A Database of Systems Pharmacology for Drug Discovery from Herbal Medicines. J. Chem. Inf. Model. doi:10.1021/ci4005517
- Wang et al. (2024) A critical assessment of Traditional Chinese Medicine databases. Front. Pharmacol. doi:10.3389/fphar.2024.1303693
