# Phase 4 v4.5: CIRI铁衰老中药单体ML筛选 - 模型构建报告

生成时间: 2026-06-23 12:04:16
总耗时: 6.0 分钟

## 1. 数据概览
- TCM化合物总数: 1491
- 蛋白靶标总数: 38
- 有真实活性数据的靶标数: 25
- 真实正样本总数（去重后）: 36954

## 2. 靶标分层
- 集成模型可训练靶标: 19 个
- 回退到相似性方法: 6 个
- 无参考数据: 13 个
- 可训练靶标列表: BCL6, CD74, CTSB, CXCL10, DYRK1A, EPHA4, GPX4, HMOX1, IL1B, KDM6B, LGMN, MAPK1, NFE2L2, PDE4B, PTGS2, SLC7A11, STAT3, TLR4, TP53

## 3. 模型性能汇总（CV 真实标签）
- 保留模型数: 95
- RF: 平均 AUC=0.9932±0.0056, 平均 AUPR=0.9886, 覆盖靶标=19
- XGB: 平均 AUC=0.9885±0.0110, 平均 AUPR=0.9809, 覆盖靶标=19
- LR: 平均 AUC=0.9896±0.0091, 平均 AUPR=0.9838, 覆盖靶标=19
- SVM: 平均 AUC=0.9845±0.0132, 平均 AUPR=0.9776, 覆盖靶标=19
- KNN: 平均 AUC=0.8902±0.0310, 平均 AUPR=0.7639, 覆盖靶标=19
- ensemble 平均 AUC=0.9692, 平均 AUPR=0.9389
- ⚠️ 有 64 个模型 CV AUC > 0.98，建议检查负样本设计、骨架泄漏及结果可解释性。

## 4. 富集因子分析
- EF@1%: mean=0.00, max=0.00
- EF@5%: mean=10.07, max=20.15
- EF@10%: mean=5.00, max=10.01

## 5. Top 20 候选化合物
| 排名 | 化合物 | 综合得分 | 平均得分 | 高置信命中 | Top 靶标 |
|------|--------|----------|----------|------------|----------|
| 1 | moracin O | 0.3418 | 0.1656 | 2 | CTSB(0.915), EPHA4(0.758), PTGS2(0.359), TLR4(0.334), STAT3(0.295) |
| 2 | 4',5',7-trimethyl-3-methoxyfla | 0.3278 | 0.2258 | 1 | DYRK1A(0.716), PTGS2(0.638), PDE4B(0.632), EPHA4(0.514), MAPK1(0.478) |
| 3 | isocorynantheic acid | 0.3272 | 0.2317 | 1 | KDM6B(0.766), PTGS2(0.541), STAT3(0.470), DYRK1A(0.439), TLR4(0.433) |
| 4 | Glycoside K_qt | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 5 | 14-deoxyandrographolide | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 6 | pachypodol | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 7 | Aposiopolamine | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 8 | Demethoxycapillarisin | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 9 | pinoresinol-4-O-beta-D-apiosyl | 0.2830 | 0.0976 | 1 | EPHA4(0.764), CTSB(0.195), PTGS2(0.183), TLR4(0.173), GPX4(0.136) |
| 10 | GA120 | 0.2738 | 0.0744 | 1 | FTH1(0.751), SP1(0.120), TLR4(0.115), PTGS2(0.107), KDM6B(0.099) |
| 11 | Dehydrocorybulbine | 0.2735 | 0.1662 | 0 | LGMN(0.660), STAT3(0.415), PDE4B(0.355), DYRK1A(0.354), CTSB(0.327) |
| 12 | 2-Benzo[1,3]dioxol-5-yl-5,7-di | 0.2650 | 0.1343 | 0 | EPHA4(0.650), PTGS2(0.350), DYRK1A(0.228), STAT3(0.202), TLR4(0.200) |
| 13 | Ariskanin A | 0.2596 | 0.1068 | 0 | DYRK1A(0.665), PTGS2(0.353), TLR4(0.160), SLC7A11(0.141), STAT3(0.133) |
| 14 | poriol | 0.2530 | 0.1963 | 0 | PTGS2(0.512), DYRK1A(0.500), PDE4B(0.400), EPHA4(0.382), STAT3(0.354) |
| 15 | Dehydrocorydalmine | 0.2529 | 0.1405 | 0 | PTGS2(0.582), DYRK1A(0.321), STAT3(0.303), SLC7A11(0.287), PDE4B(0.271) |
| 16 | kadsurin B | 0.2481 | 0.0672 | 0 | PTGS2(0.663), MAPK1(0.114), GPX4(0.097), CD74(0.093), LGMN(0.072) |
| 17 | (2E,4Z)-5-(1,3-benzodioxol-5-y | 0.2324 | 0.1097 | 0 | CTSB(0.518), PTGS2(0.289), TLR4(0.264), EPHA4(0.234), STAT3(0.201) |
| 18 | isogosferol | 0.2324 | 0.1097 | 0 | CTSB(0.518), PTGS2(0.289), TLR4(0.264), EPHA4(0.234), STAT3(0.201) |
| 19 | florilenalin isobutyrate | 0.2300 | 0.1536 | 0 | DYRK1A(0.493), EPHA4(0.428), PTGS2(0.320), PDE4B(0.317), TLR4(0.315) |
| 20 | protostemotinine | 0.2285 | 0.0941 | 0 | TP53(0.511), PTGS2(0.219), MAPK1(0.146), GPX4(0.140), SLC7A11(0.129) |

## 6. 候选排序权重（显式记录）
composite_score = 0.30 * avg_score + 0.20 * max_score + 0.20 * (n_hits / n_targets) + 0.20 * (n_high / n_targets) + 0.10 * consistency
- avg_score: 该化合物在所有可预测靶标上的平均预测分
- max_score: 最大预测分
- n_hits: prediction_score > 0.5 的靶标数
- n_high: prediction_score > 0.7 的靶标数
- consistency: 1 - std(prediction_score)，衡量跨靶标预测一致性

## 7. v4.5 改进总结
- 真实活性 vs 相似性扩展标签分离，CV 评估仅使用真实标签。
- CV 采用 Murcko 骨架分组的 StratifiedGroupKFold，防止相同骨架同时进入训练/验证集。
- 多模型集成 + 5 折 CV，AUC>0.6 才保留，并记录 AUC/AUPR 标准差。
- Borda 排序融合、概率几何平均、AUC 加权（小样本收缩）。
- 跨靶标 CPI 模型共享负样本/蛋白信息。
- per-target 与 cross-target 方法感知加权融合。
- 仅对 TCM 中真实出现的活性样本计算 EF。
- 新增 AUC  Sanity Check：对 AUC>0.98 或 <0.6 的模型发出警告。

## 8. 已知局限性与使用建议
- 当前真实负样本主要来自其他靶标活性分子，任务可能过简，导致 CV AUC 偏高。
- 富集因子（EF）使用与训练相同的真实正样本计算，属于乐观估计，不可视为独立验证。
- 如用于论文发表，建议补充：1) 外部独立测试集；2) 真实 inactive/ decoy 负样本；3) 时间切分或分子骨架切分验证。