import requests
import re
import json
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup as bs

root_url = "https://www.tcmsp-e.com/tcmspsearch.php"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}

session = requests.Session()
session.headers.update(headers)

# 1. 获取 token
html = session.get(root_url, timeout=30).text
soup = bs(html, "html.parser")
token_input = soup.find("input", {"name": "token"})
token = token_input["value"] if token_input else None
print(f"Token: {token}")

target_herbs = ["柴胡", "桂枝", "黄芩", "人参", "甘草", "半夏", "白芍", "大枣", "生姜"]

all_results = []

for herb_name in target_herbs:
    print(f"\n========== 正在查询: {herb_name} ==========")
    
    # 2. 搜索药名
    search_url = f"{root_url}?qs=herb_all_name&q={herb_name}&token={token}"
    html = session.get(search_url, timeout=30).text
    
    match = re.search(r'data:\s*(\[.*?\]),', html)
    if not match:
        print(f"  未找到搜索结果")
        continue
    
    herbs = json.loads(match.group(1))
    # 选择精确匹配的药名
    target_herb = None
    for h in herbs:
        if h['herb_cn_name'] == herb_name:
            target_herb = h
            break
    if not target_herb and herbs:
        target_herb = herbs[0]  #  fallback to first result
    
    cn_name = target_herb['herb_cn_name']
    en_name = target_herb['herb_en_name']
    pinyin = target_herb['herb_pinyin']
    print(f"  匹配到: {cn_name} / {en_name} / {pinyin}")
    
    # 3. 获取成分页面
    en_name_encoded = en_name.replace(" ", "%20")
    herb_url = f"{root_url}?qr={en_name_encoded}&qsr=herb_en_name&token={token}"
    html = session.get(herb_url, timeout=30).text
    
    # 查找成分数据
    soup = bs(html, "html.parser")
    scripts = soup.find_all("script")
    
    ingredients_data = None
    for script in scripts:
        text = script.string if script.string else ""
        if "kendoGrid" in text and "data:" in text and "molecule_ID" in text:
            m = re.search(r'data:\s*(\[.*?\]),\s*pageSize', text, re.DOTALL)
            if m:
                data_str = m.group(1)
                try:
                    ingredients_data = json.loads(data_str)
                    print(f"  成功提取 {len(ingredients_data)} 个成分")
                except json.JSONDecodeError as e:
                    print(f"  JSON解析失败: {e}")
                break
    
    if not ingredients_data:
        print(f"  未找到成分数据")
        continue
    
    # 4. 记录结果
    for ing in ingredients_data:
        all_results.append({
            "herb_cn_name": cn_name,
            "herb_en_name": en_name,
            "herb_pinyin": pinyin,
            "molecule_ID": ing.get("molecule_ID"),
            "MOL_ID": ing.get("MOL_ID"),
            "molecule_name": ing.get("molecule_name"),
            "ob": ing.get("ob"),
            "dl": ing.get("dl"),
            "mw": ing.get("mw"),
            "alogp": ing.get("alogp"),
            "bbb": ing.get("bbb"),
            "caco2": ing.get("caco2"),
            "halflife": ing.get("halflife"),
            "hdon": ing.get("hdon"),
            "hacc": ing.get("hacc"),
            "FASA": ing.get("FASA"),
        })

# 5. 保存结果
if all_results:
    df = pd.DataFrame(all_results)
    save_dir = Path("d:/铁衰老 绝不重蹈覆辙/L3/results")
    save_dir.mkdir(exist_ok=True)
    
    output_file = save_dir / "9herbs_tcmsp_ingredients.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ 结果已保存: {output_file}")
    print(f"总计: {len(df)} 条成分记录")
    
    # 统计每味药的成分数
    stats = df.groupby('herb_cn_name').size().sort_values(ascending=False)
    print("\n每味药成分数量:")
    for herb, count in stats.items():
        print(f"  {herb}: {count} 个")
else:
    print("\n未获取到任何数据")
