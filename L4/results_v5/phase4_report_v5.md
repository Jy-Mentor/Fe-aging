# Phase 4 v5: MT-PUNN 训练报告

生成时间: 2026-06-23 18:48:04
总耗时: 3.5 分钟

## 1. 架构与叙事
- 模型: Multi-Task Positive-Unlabeled Neural Network (MT-PUNN)
- 输入: 化合物 ECFP4 + MACCS + RDKit 2D 描述符
- 共享编码器: 3 层 MLP（BatchNorm + Dropout）
- 输出头: 96 个靶标独立 sigmoid（铁衰老差异表达基因集）
- 损失: nnPU（非负风险估计），未标记样本不再强制为阴性
- 验证: Murcko 骨架切分

## 2. 模型规模
- 可训练参数: 2,629,216
- 最终训练损失: 0.0001
- 最终验证损失: 0.0241

## 3. 靶标级 CV 性能
| gene | val_auc | val_aupr | n_pos | prior |
| --- | --- | --- | --- | --- |
| CD74 | 1.0 | 1.0 | 1 | 0.000699999975040555 |
| CXCL10 | 1.0 | 1.0 | 34 | 0.00559999980032444 |
| LGMN | 0.9998 | 0.99 | 28 | 0.01080000028014183 |
| IL1B | 0.9994 | 0.9938 | 148 | 0.017100000753998756 |
| KEAP1 | 0.9924 | 0.9747 | 282 | 0.03310000151395798 |
| BCL6 | 0.9911 | 0.9834 | 288 | 0.043299999088048935 |
| KDM6B | 0.9822 | 0.9629 | 59 | 0.01140000019222498 |
| TLR4 | 0.9764 | 0.7941 | 27 | 0.0052999998442828655 |
| ALOX15 | 0.9706 | 0.7966 | 149 | 0.020400000736117363 |
| PDE4B | 0.9705 | 0.9633 | 1124 | 0.16899999976158142 |
| HMOX1 | 0.9689 | 0.5819 | 9 | 0.0006000000284984708 |
| SAT1 | 0.9665 | 0.0147 | 1 | 9.999999747378752e-05 |
| CTSB | 0.9638 | 0.9148 | 411 | 0.06700000166893005 |
| DYRK1A | 0.9588 | 0.9268 | 986 | 0.13279999792575836 |
| NOX4 | 0.9495 | 0.8453 | 52 | 0.008899999782443047 |
| EPHA4 | 0.9429 | 0.7491 | 70 | 0.00860000029206276 |
| MAPK1 | 0.9387 | 0.9328 | 2348 | 0.34439998865127563 |
| PTGS2 | 0.9348 | 0.8788 | 897 | 0.11270000040531158 |
| HIF1A | 0.928 | 0.7883 | 120 | 0.019500000402331352 |
| LCN2 | 0.7262 | 0.0027 | 2 | 9.999999747378752e-05 |
| SP1 | 0.5325 | 0.0011 | 1 | 9.999999747378752e-05 |

## 4. 富集因子
| gene | top_percent | n_top | n_hits | n_pos_tcm | baseline_rate | enrichment_factor |
| --- | --- | --- | --- | --- | --- | --- |
| ALOX15 | 1 | 6 | 0 | 1 | 0.00174 | 0.0 |
| ALOX15 | 5 | 29 | 1 | 1 | 0.00174 | 19.79 |
| ALOX15 | 10 | 57 | 1 | 1 | 0.00174 | 10.07 |
| NOX4 | 1 | 6 | 1 | 1 | 0.00174 | 95.67 |
| NOX4 | 5 | 29 | 1 | 1 | 0.00174 | 19.79 |
| NOX4 | 10 | 57 | 1 | 1 | 0.00174 | 10.07 |
| PTGS2 | 1 | 6 | 0 | 1 | 0.00174 | 0.0 |
| PTGS2 | 5 | 29 | 0 | 1 | 0.00174 | 0.0 |
| PTGS2 | 10 | 57 | 0 | 1 | 0.00174 | 0.0 |

## 5. Top 候选化合物
| rank | MOL_ID | molecule_name | SMILES | composite_score | avg_score | max_score | n_hits | n_high | n_targets | consistency | top_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | MOL012719 | moracin O | CC(C)(O)[C@@H]1Cc2cc3cc(-c4cc(O)cc(O)c4)oc3cc2O1 | 0.41519526640574134 | 0.41591084003448486 | 0.9995051622390747 | 3 | 2 | 96 | 0.8010431528091431 | HIF1A(1.000), DYRK1A(0.815), PTGS2(0.613), ZEB1(0.500), HMGB1(0.500) |
| 2 | MOL001781 | Indigo | O=C1C(c2[nH]c3ccccc3c2O)=Nc2ccccc21 | 0.414709347486496 | 0.4125545024871826 | 0.9976536631584167 | 4 | 2 | 96 | 0.7891226410865784 | DYRK1A(0.998), MAPK1(0.782), ALOX15(0.633), PTGS2(0.590), ZEB1(0.500) |
| 3 | MOL003859 | Moracin E | CC1(C)C=Cc2c(cc(O)cc2-c2cc3ccc(O)cc3o2)O1 | 0.4146528840065002 | 0.41307365894317627 | 0.9999974966049194 | 3 | 3 | 96 | 0.7823128700256348 | HIF1A(1.000), PTGS2(0.920), DYRK1A(0.899), ZEB1(0.500), HMGB1(0.500) |
| 4 | MOL004884 | Licoisoflavone B | CC1(C)C=Cc2c(ccc(-c3coc4cc(O)cc(O)c4c3=O)c2O)O1 | 0.41375449597835534 | 0.40935131907463074 | 0.9999285936355591 | 3 | 3 | 96 | 0.7846338152885437 | HIF1A(1.000), PTGS2(0.847), DYRK1A(0.748), ZEB1(0.500), HMGB1(0.500) |
| 5 | MOL003170 | Gentisein | O=c1c2cc(O)ccc2oc2cc(O)cc(O)c12 | 0.41305039227008816 | 0.4081645905971527 | 0.9962508082389832 | 3 | 3 | 96 | 0.7885085344314575 | DYRK1A(0.996), PTGS2(0.735), MAPK1(0.702), ZEB1(0.500), ICA1(0.500) |
| 6 | MOL008046 | Demethoxycapillarisin | O=c1cc(Oc2ccc(O)cc2)oc2cc(O)cc(O)c12 | 0.4127426753441492 | 0.4146316945552826 | 0.9927487969398499 | 3 | 2 | 96 | 0.7938674092292786 | DYRK1A(0.993), PTGS2(0.905), MAPK1(0.609), ZEB1(0.500), ICA1(0.500) |
| 7 | MOL003858 | Moracin D | CC1(C)C=Cc2c(O)cc(-c3cc4ccc(O)cc4o3)cc2O1 | 0.41233233312765755 | 0.4101809561252594 | 0.9999994039535522 | 3 | 2 | 96 | 0.7886149883270264 | HIF1A(1.000), PTGS2(0.907), DYRK1A(0.581), ZEB1(0.500), HMGB1(0.500) |
| 8 | MOL001004 | pelargonidin | Oc1ccc(-c2[o+]c3cc(O)cc(O)c3cc2O)cc1 | 0.41186294853687283 | 0.412665456533432 | 0.9836081862449646 | 3 | 3 | 96 | 0.7884167432785034 | DYRK1A(0.984), PTGS2(0.894), MAPK1(0.741), ZEB1(0.500), ICA1(0.500) |
| 9 | MOL004912 | Glabrone | CC1(C)C=Cc2c(ccc(-c3coc4cc(O)ccc4c3=O)c2O)O1 | 0.4116917828718821 | 0.407883882522583 | 0.9997206330299377 | 3 | 2 | 96 | 0.78965824842453 | HIF1A(1.000), DYRK1A(0.727), PTGS2(0.633), ZEB1(0.500), HMGB1(0.500) |
| 10 | MOL004891 | shinpterocarpin | CC1(C)C=Cc2c(ccc3c2OC[C@H]2c4ccc(O)cc4O[C@@H]32)O1 | 0.41141847173372903 | 0.4073154926300049 | 0.999795138835907 | 3 | 2 | 96 | 0.7884812951087952 | HIF1A(1.000), DYRK1A(0.736), PTGS2(0.646), ZEB1(0.500), HMGB1(0.500) |
| 11 | MOL012976 | coumestrol | O=c1oc2cc(O)ccc2c2oc3cc(O)ccc3c12 | 0.4111668507258097 | 0.40745866298675537 | 0.9983981251716614 | 3 | 2 | 96 | 0.7883296012878418 | DYRK1A(0.998), MAPK1(0.754), PTGS2(0.646), ZEB1(0.500), ICA1(0.500) |
| 12 | MOL004911 | Glabrene | CC1(C)C=Cc2c(O)ccc(C3=Cc4ccc(O)cc4OC3)c2O1 | 0.4111261814832687 | 0.4119199812412262 | 0.9833479523658752 | 3 | 3 | 96 | 0.7838059663772583 | HIF1A(0.983), DYRK1A(0.977), PTGS2(0.730), ZEB1(0.500), HMGB1(0.500) |
| 13 | MOL001756 | quindoline | c1ccc2nc3c(cc2c1)[nH]c1ccccc13 | 0.4110435674587885 | 0.4057474434375763 | 0.9999829530715942 | 3 | 2 | 96 | 0.7890607714653015 | DYRK1A(1.000), MAPK1(0.726), PTGS2(0.534), ZEB1(0.500), ICA1(0.500) |
| 14 | MOL009135 | ellipticine | Cc1c2ccncc2c(C)c2c1[nH]c1ccccc12 | 0.410538179675738 | 0.40512219071388245 | 0.999997615814209 | 3 | 2 | 96 | 0.7858533263206482 | DYRK1A(1.000), MAPK1(0.797), PTGS2(0.522), ZEB1(0.500), ICA1(0.500) |
| 15 | MOL004908 | Glabridin | CC1(C)C=Cc2c(ccc3c2OC[C@@H](c2ccc(O)cc2O)C3)O1 | 0.40910473962624866 | 0.4055478274822235 | 0.9999293088912964 | 2 | 2 | 96 | 0.7912119626998901 | HIF1A(1.000), PTGS2(0.714), ZEB1(0.500), EPHA2(0.500), FBXO31(0.500) |
| 16 | MOL004384 | Yinyanghuo C | CC1(C)C=Cc2cc(-c3cc(=O)c4c(O)cc(O)cc4o3)ccc2O1 | 0.4090652306874593 | 0.40720248222351074 | 0.9999791383743286 | 2 | 2 | 96 | 0.7857532501220703 | HIF1A(1.000), PTGS2(0.973), ZEB1(0.500), EPHA2(0.500), FBXO31(0.500) |
| 17 | MOL004444 | Ziebeimine | C[C@@H]1CC[C@H]2[C@H](C)C3=C(CN2C1)[C@@H]1C[C@H]2[C@@H](C[C@@H](O)[C@H]4C[C@H](O)CC[C@@]42C)[C@@H]1CC3 | 0.40877267022927605 | 0.40800634026527405 | 0.9958095550537109 | 2 | 2 | 96 | 0.7887552380561829 | PDE4B(0.996), HIF1A(0.978), ZEB1(0.500), EPHA2(0.500), FBXO31(0.500) |
| 18 | MOL006596 | Glyceollin | CC1(C)C=Cc2c(ccc3c2OC[C@@]2(O)c4ccc(O)cc4O[C@@H]32)O1 | 0.408665065964063 | 0.4079560935497284 | 0.9967542290687561 | 2 | 2 | 96 | 0.7859405875205994 | HIF1A(0.997), DYRK1A(0.990), ZEB1(0.500), HMGB1(0.500), ERN1(0.500) |
| 19 | MOL005419 | (6E,8E,10Z,12Z,14E,16E,18E,20Z,22Z,24E,26E)-2,6,10,14,19,23,27,31-octamethyldotriaconta-2,6,8,10,12,14,16,18,20,22,24,26,30-tridecaene | CC(=O)O[C@@H](CCC(C)C)[C@@H](C)[C@H]1CC[C@H]2[C@@H]3C[C@@H](OC(C)=O)[C@@]4(O)C[C@@H](O)CC[C@]4(CO)[C@H]3CC[C@]12C | 0.4081132253011067 | 0.4055328369140625 | 0.9986377358436584 | 2 | 2 | 96 | 0.78392493724823 | HIF1A(0.999), PDE4B(0.998), ZEB1(0.500), EPHA2(0.500), FBXO31(0.500) |
| 20 | MOL004386 | Yinyanghuo E | CC1(C)C=Cc2cc(-c3cc(=O)c4c(O)cc(O)cc4o3)cc(O)c2O1 | 0.40800970693429306 | 0.40396079421043396 | 0.9995143413543701 | 2 | 2 | 96 | 0.7858526706695557 | HIF1A(1.000), PTGS2(0.893), ZEB1(0.500), EPHA2(0.500), FBXO31(0.500) |

## 6. 与 v4.5 的关键差异
- v4.5: 每个靶标独立训练浅层 ML，以其他靶标活性为伪阴性；
- v5: 所有靶标共享深度编码器，采用 PU 学习，不将未标记样本强制标记为阴性；
- v5 的 AUC/AUPR 使用未标记子集计算，通常低于 v4.5，但更贴近真实筛选场景。

## 7. 局限
- 类别先验 π_p 为保守估计，可能影响 PU 风险校准；
- 未使用真实 inactive 或 decoy，EF 仍为乐观估计；
- 深层模型在小样本靶标上可能不如 v4.5 稳定；
- TCM 候选池（L3 输出）包含 574 个唯一 SMILES，已做 SMILES 去重与名称-SMILES 一致性校验（剔除 MW 偏差过大的条目）。

## 8. 关键参考
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