"""
批量爬取TCMSP中药成分数据，建立草药-成分映射关系
目标：大柴胡汤 + 桂枝茯苓丸 + 艾草 + 其他相关药味
"""
import requests
import re
import json
import time
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup as bs

ROOT_URL = "https://www.tcmsp-e.com/tcmspsearch.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}
SAVE_DIR = Path("L3/results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 大柴胡汤：柴胡、黄芩、半夏、生姜、大枣、枳实、大黄、白芍
# 桂枝茯苓丸：桂枝、茯苓、牡丹皮、桃仁、白芍
# 艾草（艾叶）
# 还有常见中药：甘草、白术、陈皮等
TARGET_HERBS = [
    # 大柴胡汤
    "柴胡", "黄芩", "半夏", "生姜", "大枣", "枳实", "大黄", "白芍",
    # 桂枝茯苓丸
    "桂枝", "茯苓", "牡丹皮", "桃仁",
    # 艾草
    "艾叶",
    # 其他常见
    "甘草", "白术", "陈皮", "人参", "黄芪", "当归", "川芎",
    "丹参", "三七", "黄连", "黄柏", "栀子", "连翘", "金银花",
    "薄荷", "紫苏叶", "藿香", "苍术", "厚朴", "砂仁",
    "麻黄", "杏仁", "石膏", "知母", "麦冬", "五味子",
    "熟地", "山药", "山茱萸", "泽泻", "猪苓",
    "益母草", "红花", "赤芍", "延胡索", "木香", "香附",
    "枳壳", "桔梗", "瓜蒌", "薤白", "葛根", "升麻",
    "柴胡", "桑叶", "菊花", "蔓荆子", "蝉蜕", "牛蒡子",
]
# 去重保序
seen = set()
target_herbs = []
for h in TARGET_HERBS:
    if h not in seen:
        seen.add(h)
        target_herbs.append(h)
print(f"目标药味: {len(target_herbs)} 味")


def get_token(session):
    html = session.get(ROOT_URL, timeout=30).text
    soup = bs(html, "html.parser")
    token_input = soup.find("input", {"name": "token"})
    return token_input["value"] if token_input else None


def search_herb(session, token, herb_name):
    """搜索草药，返回匹配的草药信息字典"""
    search_url = f"{ROOT_URL}?qs=herb_all_name&q={herb_name}&token={token}"
    html = session.get(search_url, timeout=30).text
    
    match = re.search(r'data:\s*(\[.*?\]),', html)
    if not match:
        return None
    
    try:
        herbs = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    
    # 优先精确匹配
    for h in herbs:
        if h.get('herb_cn_name') == herb_name:
            return h
    # 次选：包含
    for h in herbs:
        if herb_name in h.get('herb_cn_name', ''):
            return h
    return herbs[0] if herbs else None


def get_herb_ingredients(session, token, herb_info):
    """获取某味药的所有成分"""
    en_name = herb_info.get('herb_en_name', '')
    en_name_encoded = en_name.replace(" ", "%20")
    herb_url = f"{ROOT_URL}?qr={en_name_encoded}&qsr=herb_en_name&token={token}"
    
    try:
        html = session.get(herb_url, timeout=30).text
    except Exception as e:
        print(f"  请求失败: {e}")
        return []
    
    soup = bs(html, "html.parser")
    scripts = soup.find_all("script")
    
    for script in scripts:
        text = script.string if script.string else ""
        if "kendoGrid" in text and "data:" in text and "molecule_ID" in text:
            m = re.search(r'data:\s*(\[.*?\]),\s*pageSize', text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
    return []


def main():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    token = get_token(session)
    print(f"Token: {token}")
    if not token:
        print("获取token失败")
        return
    
    all_results = []
    failed_herbs = []
    
    for i, herb_name in enumerate(target_herbs):
        print(f"\n[{i+1}/{len(target_herbs)}] 查询: {herb_name}")
        
        herb_info = search_herb(session, token, herb_name)
        if not herb_info:
            print(f"  未找到")
            failed_herbs.append(herb_name)
            continue
        
        cn_name = herb_info.get('herb_cn_name', '')
        en_name = herb_info.get('herb_en_name', '')
        pinyin = herb_info.get('herb_pinyin', '')
        print(f"  匹配: {cn_name} / {en_name} / {pinyin}")
        
        ingredients = get_herb_ingredients(session, token, herb_info)
        print(f"  成分数: {len(ingredients)}")
        
        for ing in ingredients:
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
        
        time.sleep(1.0)  # 礼貌爬取
    
    # 保存
    if all_results:
        df = pd.DataFrame(all_results)
        output = SAVE_DIR / "herb_ingredient_mapping.xlsx"
        df.to_excel(output, index=False)
        print(f"\n✅ 总记录: {len(df)} 条")
        print(f"✅ 保存: {output}")
        
        # 统计
        stats = df.groupby('herb_cn_name').size().sort_values(ascending=False)
        print(f"\n=== 每味药成分数 ===")
        for herb, count in stats.items():
            print(f"  {herb}: {count}")
        
        # 统计唯一成分数
        unique_mol = df['MOL_ID'].nunique()
        print(f"\n唯一成分(MOL_ID): {unique_mol}")
    
    if failed_herbs:
        print(f"\n❌ 失败的药味: {failed_herbs}")


if __name__ == "__main__":
    main()
