#!/usr/bin/env python3
"""HERB 2.0 壮药覆盖分析"""
import pandas as pd

failed = pd.read_csv(r"d:\铁衰老 绝不重蹈覆辙\L3\results\zhuangyao_scrape_failed.csv")
print(f"失败壮药总数: {len(failed)}")

# 30味已通过TCMSP模糊匹配
fuzzy_set = {
    "大叶骨碎补","广山药","广西海风藤","广金钱草","小槐花","无患子果","毛两面针",
    "毛鸡骨草","当归藤","余甘子汁","苦玄参","岩黄连","南板蓝根","蛇床子油",
    "蓝花柴胡","丁茄根","三七叶","三七姜","大半边莲","大浮萍","广山楂叶",
    "广钩藤","毛郁金","水半夏","白木香","光石韦","红杜仲","秃叶黄柏","草豆蔻","紫苏叶",
}

all_failed = failed["cn_name_clean"].tolist()
remaining = [n for n in all_failed if n not in fuzzy_set]
print(f"TCMSP模糊匹配: {len(fuzzy_set)}")
print(f"待HERB补充: {len(remaining)}")

# 排除非植物药
animals_minerals = ["蛤蚧","蟾蜍","鳖","乌鸡","鸡","鸭","龟","蛇","蚕","蜂","蜈","蚣","蛛",
    "蝎","蝉","蛤","蛎","蟅","虫","蚁","蜕","滑石","石膏","朱砂","雄黄","芒硝","明矾",
    "炉甘石","硫黄","赤石脂","龙骨","牡蛎","珍珠","玛瑙","钟乳","磁石","礞石","琥珀",
    "硼砂","白矾","甲鱼","鳖甲"]

remaining_plant = []
remaining_nonplant = []
for n in remaining:
    is_nonplant = any(a in n for a in animals_minerals)
    if is_nonplant:
        remaining_nonplant.append(n)
    else:
        remaining_plant.append(n)
print(f"植物药: {len(remaining_plant)}, 非植物药: {len(remaining_nonplant)}")

# 加载HERB
herb = pd.read_csv(r"D:\下载\HERB_herb_info_v2.txt", sep="\t", low_memory=False)
print(f"\nHERB药材总数: {len(herb)}")
tcmsp_count = herb["TCMSP_id"].notna().sum()
print(f"有TCMSP_id: {tcmsp_count}")

herb_names = set(herb["Herb_cn_name"].dropna().unique())

# 匹配
exact = {}
contains = {}
for z in remaining_plant:
    if z in herb_names:
        exact[z] = z
        continue
    for h in herb_names:
        if z in h:
            contains[z] = h
            break

print(f"\nHERB精确匹配: {len(exact)}")
print(f"HERB包含匹配: {len(contains)}")

no_match = [z for z in remaining_plant if z not in exact and z not in contains]
print(f"\n完全无法匹配(植物药): {len(no_match)}")
print("样本:", no_match[:40])
print("\n非植物药(合理排除):", len(remaining_nonplant))
for n in remaining_nonplant[:20]:
    print(f"  {n}")
