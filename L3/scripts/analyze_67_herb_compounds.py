"""
分析67个与9味中药重叠的v6候选化合物的训练结果
"""
import pandas as pd
import numpy as np
from pathlib import Path

base = Path("d:/铁衰老 绝不重蹈覆辙")

# 读取数据
herbs_df = pd.read_excel(base / "L3/results/9herbs_tcmsp_ingredients.xlsx")
pred_df = pd.read_csv(base / "L4/results_v6/tcm_predictions_full_v6.csv")
top_df = pd.read_csv(base / "L4/results_v6/tcm_top_candidates_v6.csv")

# 获取9味药的MOL_ID
herb_mol_ids = set(herbs_df['MOL_ID'].dropna().astype(str))

# 筛选67个化合物
pred_67 = pred_df[pred_df['MOL_ID'].astype(str).isin(herb_mol_ids)].copy()
top_67 = top_df[top_df['MOL_ID'].astype(str).isin(herb_mol_ids)].copy()

print("=" * 70)
print("67个中药来源候选化合物的 v6 训练结果分析")
print("=" * 70)

# 关联所属中药
herb_map = {}
for _, row in herbs_df.iterrows():
    mid = str(row['MOL_ID'])
    if mid not in herb_map:
        herb_map[mid] = set()
    herb_map[mid].add(row['herb_cn_name'])

pred_67['source_herbs'] = pred_67['MOL_ID'].astype(str).apply(lambda x: ', '.join(sorted(herb_map.get(x, set()))))
top_67['source_herbs'] = top_67['MOL_ID'].astype(str).apply(lambda x: ', '.join(sorted(herb_map.get(x, set()))))

print(f"\n1. 总体概况")
print(f"   9味中药成分进入 v6 预测池: {len(pred_67)} 个")
print(f"   其中进入 Top 500: {len(top_67)} 个")

# 关键指标统计
print(f"\n2. 预测分数分布 (67个化合物)")
for col in ['avg_score', 'max_score', 'composite_score', 'n_targets', 'n_hits']:
    if col in pred_67.columns:
        print(f"   {col}: 均值={pred_67[col].mean():.4f}, 中位数={pred_67[col].median():.4f}, 最小={pred_67[col].min():.4f}, 最大={pred_67[col].max():.4f}")

# 与全部预测池对比
print(f"\n3. 与全部预测池 (N={len(pred_df)}) 的对比")
for col in ['avg_score', 'max_score', 'composite_score', 'n_targets', 'n_hits']:
    if col in pred_67.columns and col in pred_df.columns:
        h_mean = pred_67[col].mean()
        a_mean = pred_df[col].mean()
        h_med = pred_67[col].median()
        a_med = pred_df[col].median()
        print(f"   {col}: 67个均值={h_mean:.4f} vs 全部均值={a_mean:.4f} (差异: {h_mean-a_mean:+.4f})")
        print(f"          67个中位={h_med:.4f} vs 全部中位={a_med:.4f}")

# 命中靶点分析
target_cols = [c for c in pred_67.columns if c not in ['MOL_ID', 'molecule_name', 'SMILES', 'avg_score', 'max_score', 'n_hits', 'n_high', 'consistency', 'composite_score', 'n_targets', 'top_targets', 'rank', 'source_herbs']]
print(f"\n4. 靶点活性分析 (67个化合物)")
print(f"   可分析靶点列数: {len(target_cols)}")

# 计算每个靶点的平均预测概率
herb_target_avg = pred_67[target_cols].mean().sort_values(ascending=False)
all_target_avg = pred_df[target_cols].mean().sort_values(ascending=False)

print(f"\n   67个化合物中平均预测概率最高的10个靶点:")
for target, score in herb_target_avg.head(10).items():
    all_score = all_target_avg[target]
    print(f"      {target}: {score:.4f} (全部均值: {all_score:.4f}, 差异: {score-all_score:+.4f})")

# 高活性计数 (概率 > 0.5)
high_act = (pred_67[target_cols] > 0.5).sum(axis=0).sort_values(ascending=False)
print(f"\n   67个化合物中预测概率 >0.5 的化合物数量 (Top 10 靶点):")
for target, count in high_act.head(10).items():
    print(f"      {target}: {count} / {len(pred_67)} ({count/len(pred_67)*100:.1f}%)")

# 排名分布
if 'rank' in top_67.columns:
    print(f"\n5. Top 500 排名分布 (67个中的 {len(top_67)} 个)")
    print(f"   最佳排名: {top_67['rank'].min()}")
    print(f"   最差排名: {top_67['rank'].max()}")
    print(f"   平均排名: {top_67['rank'].mean():.1f}")
    print(f"   中位排名: {top_67['rank'].median():.1f}")
    
    bins = [1, 50, 100, 200, 300, 400, 500]
    top_67['rank_bin'] = pd.cut(top_67['rank'], bins=bins, right=False, labels=['1-49', '50-99', '100-199', '200-299', '300-399', '400-499'])
    print(f"\n   排名区间分布:")
    for b, c in top_67['rank_bin'].value_counts().sort_index().items():
        print(f"      {b}: {c} 个")

# 来源中药分布
if 'source_herbs' in top_67.columns:
    print(f"\n6. 来源中药分布 (Top 67)")
    herb_counts = {}
    for herbs_str in top_67['source_herbs']:
        for herb in herbs_str.split(', '):
            herb_counts[herb] = herb_counts.get(herb, 0) + 1
    for herb, count in sorted(herb_counts.items(), key=lambda x: -x[1]):
        print(f"   {herb}: {count} 个")

# 输出详细列表
print(f"\n7. 67个化合物详细列表 (按 composite_score 排序)")
top_67_sorted = top_67.sort_values('composite_score', ascending=False)
for _, row in top_67_sorted.iterrows():
    print(f"   Rank {int(row['rank'])} | {row['MOL_ID']} | {row['molecule_name']} | {row['source_herbs']} | score={row['composite_score']:.4f} | targets={int(row['n_targets'])}")

# 保存结果
out_file = base / "L3/results/67_herb_compounds_v6_analysis.csv"
top_67_sorted.to_csv(out_file, index=False)
print(f"\n8. 结果已保存: {out_file}")

print("\n" + "=" * 70)
print("分析完成")
print("=" * 70)
