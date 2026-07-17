#!/usr/bin/env python3
"""爬取30味模糊匹配壮药在TCMSP中的成分"""
import json, logging, re, sys, time
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"
L3_LOGS.mkdir(parents=True, exist_ok=True)

ROOT_URL = "https://www.tcmsp-e.com/tcmspsearch.php"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(L3_LOGS / "zhuangyao_round2_scrape.log", encoding="utf-8", mode="w"), logging.StreamHandler(sys.stdout)], force=True)
logger = logging.getLogger(__name__)

# 壮药->TCMSP映射
ZHUANG_TO_TCMSP = {
    "大叶骨碎补": "骨碎补", "广山药": "山药", "广西海风藤": "海风藤", "广金钱草": "金钱草",
    "小槐花": "槐花", "无患子果": "无患子", "毛两面针": "两面针", "毛鸡骨草": "鸡骨草",
    "当归藤": "当归", "余甘子汁": "余甘子", "苦玄参": "玄参", "岩黄连": "黄连",
    "南板蓝根": "板蓝根", "蛇床子油": "蛇床子", "蓝花柴胡": "柴胡", "丁茄根": "茄根",
    "三七叶": "三七", "三七姜": "三七", "大半边莲": "半边莲", "大浮萍": "浮萍",
    "广山楂叶": "山楂叶", "广钩藤": "钩藤", "毛郁金": "郁金", "水半夏": "半夏",
    "白木香": "木香", "光石韦": "石韦", "红杜仲": "杜仲", "秃叶黄柏": "黄柏",
    "草豆蔻": "豆蔻", "紫苏叶": "紫苏"
}

# 已在现有scrape中的TCMSP药材
EXISTING_SCRAPED = {"三七", "山药", "黄柏", "木香", "当归", "黄连", "柴胡", "半夏"}

NEW_TARGETS = sorted(set(ZHUANG_TO_TCMSP.values()) - EXISTING_SCRAPED)
logger.info(f"需要新爬取: {len(NEW_TARGETS)} 味 -> {NEW_TARGETS}")


def get_token(session):
    html = session.get(ROOT_URL, timeout=30).text
    soup = bs(html, "html.parser")
    token_input = soup.find("input", {"name": "token"})
    return token_input["value"] if token_input else None


def search_herb(session, token, herb_name):
    search_url = f"{ROOT_URL}?qs=herb_all_name&q={herb_name}&token={token}"
    try:
        html = session.get(search_url, timeout=30).text
    except Exception as e:
        logger.warning(f"  搜索失败 [{herb_name}]: {e}")
        return None
    match = re.search(r'data:\s*(\[.*?\]),', html)
    if not match:
        return None
    try:
        herbs = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not herbs:
        return None
    for h in herbs:
        if h.get("herb_cn_name") == herb_name:
            return h
    for h in herbs:
        if herb_name in h.get("herb_cn_name", ""):
            return h
    return herbs[0]


def get_herb_ingredients(session, token, herb_info):
    en_name = herb_info.get("herb_en_name", "")
    herb_url = f"{ROOT_URL}?qr={en_name.replace(' ', '%20')}&qsr=herb_en_name&token={token}"
    try:
        html = session.get(herb_url, timeout=30).text
    except Exception as e:
        logger.warning(f"  成分请求失败 [{en_name}]: {e}")
        return []
    soup = bs(html, "html.parser")
    for script in soup.find_all("script"):
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
    all_results = []
    failed = []

    session = requests.Session()
    session.headers.update(HEADERS)
    token = get_token(session)
    logger.info(f"Token: {token}")
    if not token:
        logger.error("获取token失败")
        sys.exit(1)

    for i, herb_name in enumerate(NEW_TARGETS):
        logger.info(f"[{i+1}/{len(NEW_TARGETS)}] 查询: {herb_name}")
        herb_info = search_herb(session, token, herb_name)
        if not herb_info:
            logger.warning(f"  未找到: {herb_name}")
            failed.append(herb_name)
            time.sleep(0.5)
            continue

        cn = herb_info.get("herb_cn_name", "")
        en = herb_info.get("herb_en_name", "")
        py = herb_info.get("herb_pinyin", "")
        logger.info(f"  匹配: {cn} / {en} / {py}")

        ingredients = get_herb_ingredients(session, token, herb_info)
        logger.info(f"  成分数: {len(ingredients)}")

        for ing in ingredients:
            all_results.append({
                "herb_cn_name": cn, "herb_en_name": en, "herb_pinyin": py,
                "molecule_ID": ing.get("molecule_ID"),
                "MOL_ID": str(ing.get("MOL_ID", "")).strip(),
                "molecule_name": ing.get("molecule_name"),
                "ob": ing.get("ob"), "dl": ing.get("dl"),
                "mw": ing.get("mw"), "alogp": ing.get("alogp"),
                "bbb": ing.get("bbb"), "caco2": ing.get("caco2"),
                "halflife": ing.get("halflife"), "hdon": ing.get("hdon"),
                "hacc": ing.get("hacc"), "FASA": ing.get("FASA"),
            })
        time.sleep(1.0)

    if all_results:
        df = pd.DataFrame(all_results)
        out = L3_RESULTS / "zhuangyao_round2_ingredients.xlsx"
        df.to_excel(out, index=False)
        logger.info(f"保存: {out} ({len(df)} 条记录, {df['herb_cn_name'].nunique()} 味药, {df['MOL_ID'].nunique()} 唯一MOL_ID)")

    if failed:
        logger.warning(f"失败: {failed}")
        pd.DataFrame({"herb_name": failed}).to_csv(L3_RESULTS / "zhuangyao_round2_failed.csv", index=False)

    logger.info("完成")


if __name__ == "__main__":
    main()