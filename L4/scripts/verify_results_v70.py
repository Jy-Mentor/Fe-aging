"""验证v70预测结果完整性 - 仅用内置模块"""
import csv
import os
import statistics

RESULTS_DIR = r"d:\铁衰老 绝不重蹈覆辙\L4\results_v10_minibatch"

# 1. 检查全量预测文件
print("=" * 60)
print("1. 全量预测文件 (tcm_predictions_full_v70.csv)")
print("=" * 60)

full_path = os.path.join(RESULTS_DIR, "tcm_predictions_full_v70.csv")
file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
print(f"  文件大小: {file_size_mb:.1f} MB")

with open(full_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    cols = reader.fieldnames
    print(f"  列数: {len(cols)}")
    print(f"  列名: {cols[:10]}...")
    
    # 确定名称列 - 用 molecule_name
    name_col = 'molecule_name'
    
    beta_records = []
    all_scores = []
    all_ranks = []
    row_count = 0
    
    for row in reader:
        row_count += 1
        name = row.get(name_col, '')
        if '石竹烯' in name or 'caryophyllene' in name.lower():
            beta_records.append({
                'name': name,
                'score': row.get('composite_score', 'N/A'),
                'rank': row.get('rank', 'N/A')
            })
        try:
            score = float(row.get('composite_score', 0))
            all_scores.append(score)
        except (ValueError, TypeError):
            pass

print(f"  总行数: {row_count}")

# 2. beta-石竹烯 搜索结果
print("\n" + "=" * 60)
print("2. beta-石竹烯 (Caryophyllene) 搜索结果")
print("=" * 60)
print(f"  找到 {len(beta_records)} 条匹配记录:")
for rec in beta_records:
    print(f"    - {rec['name']} | composite_score={rec['score']} | rank={rec['rank']}")

# 3. 检查Top 500
print("\n" + "=" * 60)
print("3. Top 500 候选 (tcm_top_candidates_v70.csv)")
print("=" * 60)

top_path = os.path.join(RESULTS_DIR, "tcm_top_candidates_v70.csv")
top_size_mb = os.path.getsize(top_path) / (1024 * 1024)
print(f"  文件大小: {top_size_mb:.2f} MB")

with open(top_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    top_rows = list(reader)
    print(f"  行数: {len(top_rows)}")
    
    print("\n  Top 30 化合物:")
    for i, row in enumerate(top_rows[:30]):
        name = row.get('molecule_name', row.get('Ingredient_name', 'N/A'))
        mol_id = row.get('MOL_ID', '?')
        score = row.get('composite_score', 'N/A')
        rank = row.get('rank', i+1)
        print(f"    Rank {rank}: {name} (MOL_ID={mol_id}) | composite_score={score}")
    
    # beta-石竹烯 in top 500
    beta_top = [r for r in top_rows if '石竹烯' in r.get('molecule_name', '') or 'caryophyllene' in r.get('molecule_name', '').lower()]
    print(f"\n  beta-石竹烯在Top500中: {len(beta_top)} 条")
    for rec in beta_top:
        print(f"    - Rank {rec.get('rank','?')}: {rec.get('molecule_name','?')} | composite_score={rec.get('composite_score','?')}")

# 4. 统计分数分布
print("\n" + "=" * 60)
print("4. composite_score 分布统计")
print("=" * 60)
if all_scores:
    all_scores.sort()
    n = len(all_scores)
    print(f"  有效分数数: {n}")
    print(f"  min: {min(all_scores):.4f}")
    print(f"  max: {max(all_scores):.4f}")
    print(f"  mean: {statistics.mean(all_scores):.4f}")
    print(f"  median: {statistics.median(all_scores):.4f}")
    if n > 1:
        print(f"  std: {statistics.stdev(all_scores):.4f}")
    
    for q_pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        idx = int(n * q_pct / 100)
        idx = min(idx, n-1)
        print(f"  q{q_pct:02d}: {all_scores[idx]:.4f}")

# 5. 版本比对 (v67 vs v70)
print("\n" + "=" * 60)
print("5. v67 vs v70 版本比对")
print("=" * 60)
v67_path = os.path.join(RESULTS_DIR, "tcm_predictions_full_v67.csv")
if os.path.exists(v67_path):
    with open(v67_path, 'r', encoding='utf-8') as f:
        v67_reader = csv.DictReader(f)
        v67_rows = list(v67_reader)
        print(f"  v67 行数: {len(v67_rows)}")
        # Check v67 for beta-caryophyllene
        v67_beta = [r for r in v67_rows if '石竹烯' in r.get('molecule_name','') or 'caryophyllene' in r.get('molecule_name','').lower()]
        print(f"  v67 beta-石竹烯: {len(v67_beta)} 条")
        for rec in v67_beta[:5]:
            print(f"    - {rec.get('molecule_name','?')} | composite_score={rec.get('composite_score','?')}")
    
    # Compare top 10
    if 'v67_rows' in dir():
        v67_scores = []
        for r in v67_rows:
            try:
                v67_scores.append(float(r.get('composite_score', 0)))
            except:
                pass
        v67_scores.sort()
        if v67_scores:
            print(f"  v67 max: {max(v67_scores):.4f}, mean: {statistics.mean(v67_scores):.4f}")
        if all_scores:
            print(f"  v70 max: {max(all_scores):.4f}, mean: {statistics.mean(all_scores):.4f}")

# 6. 候选池beta-石竹烯验证
print("\n" + "=" * 60)
print("6. 壮药候选池beta-石竹烯验证")
print("=" * 60)
pool_path = r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_herb_augmented_pool.csv"
with open(pool_path, 'r', encoding='utf-8') as f:
    pool_reader = csv.DictReader(f)
    pool_cols = pool_reader.fieldnames
    print(f"  候选池列名: {pool_cols[:12]}...")
    pool_rows = list(pool_reader)
    
    # 确定名称列
    if 'molecule_name' in pool_cols:
        pool_name_col = 'molecule_name'
    elif 'Ingredient_name' in pool_cols:
        pool_name_col = 'Ingredient_name'
    else:
        pool_name_col = pool_cols[1]  # fallback
    
    pool_beta = [r for r in pool_rows if '石竹烯' in r.get(pool_name_col, '') or 'caryophyllene' in r.get(pool_name_col, '').lower()]
    print(f"  候选池中beta-石竹烯: {len(pool_beta)} 条")
    for rec in pool_beta[:10]:
        name = rec.get(pool_name_col, '?')
        smiles = rec.get('SMILES_std', rec.get('SMILES', '?'))
        print(f"    - {name} | SMILES={smiles[:60]}...")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)
