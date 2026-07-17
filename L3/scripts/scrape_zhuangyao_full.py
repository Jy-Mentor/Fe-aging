#!/usr/bin/env python3
"""
壮药全量化合物爬取脚本 — 基于广西壮药名录 (375味)
===================================================
数据来源: zhuangyao_data/guangxi_zhuangyao_list.csv (375条)
爬取目标: TCMSP (https://www.tcmsp-e.com) 中每味壮药的化合物成分
输出: L3/results/zhuangyao_ingredient_mapping_full.xlsx

数据真实性原则: 所有数据从真实网站爬取，不捏造任何数据。
爬取失败的药味记录在日志中，不伪造数据。
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ZHUANGYAO_CSV = PROJECT_ROOT / "zhuangyao_data" / "guangxi_zhuangyao_list.csv"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"
L3_LOGS.mkdir(parents=True, exist_ok=True)
L3_RESULTS.mkdir(parents=True, exist_ok=True)

OUTPUT_XLSX = L3_RESULTS / "zhuangyao_ingredient_mapping_full.xlsx"
PROGRESS_CSV = L3_RESULTS / "zhuangyao_scrape_progress.csv"
FAILED_CSV = L3_RESULTS / "zhuangyao_scrape_failed.csv"

ROOT_URL = "https://www.tcmsp-e.com/tcmspsearch.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L3_LOGS / "zhuangyao_scrape.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def load_zhuangyao_herbs() -> list[dict]:
    """加载壮药名录，清洗名称"""
    df = pd.read_csv(ZHUANGYAO_CSV)
    logger.info(f"壮药名录: {len(df)} 条 (卷1/2008: {(df['volume']==1).sum()}, 卷2/2011: {(df['volume']==2).sum()})")
    herbs = []
    for _, row in df.iterrows():
        cn_name = str(row["cn_name"]).strip()
        cn_name_clean = cn_name.split("（")[0].split("(")[0].strip()
        herbs.append({
            "idx": int(row["idx"]),
            "cn_name": cn_name,
            "cn_name_clean": cn_name_clean,
            "zhuang_name": str(row["zhuang_name"]).strip(),
            "volume": int(row["volume"]),
            "year": int(row["year"]),
        })
    return herbs


def get_token(session: requests.Session) -> str | None:
    """获取 TCMSP 搜索 token"""
    try:
        html = session.get(ROOT_URL, timeout=30).text
        soup = bs(html, "html.parser")
        token_input = soup.find("input", {"name": "token"})
        return token_input["value"] if token_input else None
    except Exception as e:
        logger.error(f"获取 token 失败: {e}")
        return None


def search_herb(session: requests.Session, token: str, herb_name: str) -> dict | None:
    """搜索草药，返回匹配的草药信息字典"""
    search_url = f"{ROOT_URL}?qs=herb_all_name&q={herb_name}&token={token}"
    try:
        html = session.get(search_url, timeout=30).text
    except Exception as e:
        logger.warning(f"  搜索请求失败 [{herb_name}]: {e}")
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


def get_herb_ingredients(session: requests.Session, token: str, herb_info: dict) -> list[dict]:
    """获取某味药的所有成分"""
    en_name = herb_info.get("herb_en_name", "")
    en_name_encoded = en_name.replace(" ", "%20")
    herb_url = f"{ROOT_URL}?qr={en_name_encoded}&qsr=herb_en_name&token={token}"

    try:
        html = session.get(herb_url, timeout=30).text
    except Exception as e:
        logger.warning(f"  成分请求失败 [{en_name}]: {e}")
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


def load_progress() -> tuple[set[str], list[dict]]:
    """加载已完成的爬取进度"""
    if not PROGRESS_CSV.exists():
        return set(), []
    df = pd.read_csv(PROGRESS_CSV)
    done = set(df["cn_name_clean"].tolist())
    existing = df.to_dict("records")
    logger.info(f"加载进度: {len(done)} 味药已完成")
    return done, existing


def save_progress(all_results: list[dict]):
    """保存爬取进度到 CSV"""
    if not all_results:
        return
    df = pd.DataFrame(all_results)
    df.to_csv(PROGRESS_CSV, index=False, encoding="utf-8")
    df.to_excel(OUTPUT_XLSX, index=False)
    logger.info(f"进度已保存: {len(df)} 条记录, 完成 {df['herb_cn_name'].nunique()} 味药")


def save_failed(failed: list[dict]):
    """保存失败的药味列表"""
    if not failed:
        return
    df = pd.DataFrame(failed)
    df.to_csv(FAILED_CSV, index=False, encoding="utf-8")
    logger.warning(f"失败列表已保存: {len(failed)} 味药")


def main():
    herbs = load_zhuangyao_herbs()
    done_set, existing_results = load_progress()
    all_results = existing_results.copy()
    failed_herbs = []

    session = requests.Session()
    session.headers.update(HEADERS)

    token = get_token(session)
    logger.info(f"Token: {token}")
    if not token:
        logger.error("获取 token 失败，无法继续")
        sys.exit(1)

    n_total = len(herbs)
    n_done = 0
    n_failed = 0
    n_ingredients = 0

    for i, herb in enumerate(herbs):
        cn_name_clean = herb["cn_name_clean"]

        if cn_name_clean in done_set:
            n_done += 1
            continue

        logger.info(f"[{i+1}/{n_total}] 查询: {herb['cn_name']} (卷{herb['volume']}/{herb['year']})")

        herb_info = search_herb(session, token, cn_name_clean)
        if not herb_info:
            logger.warning(f"  TCMSP 未找到: {cn_name_clean}")
            failed_herbs.append({
                "cn_name": herb["cn_name"],
                "cn_name_clean": cn_name_clean,
                "zhuang_name": herb["zhuang_name"],
                "volume": herb["volume"],
                "year": herb["year"],
                "reason": "TCMSP search returned no match",
            })
            n_failed += 1
            done_set.add(cn_name_clean)
            time.sleep(0.5)
            continue

        cn_name_matched = herb_info.get("herb_cn_name", "")
        en_name = herb_info.get("herb_en_name", "")
        pinyin = herb_info.get("herb_pinyin", "")
        logger.info(f"  匹配: {cn_name_matched} / {en_name} / {pinyin}")

        ingredients = get_herb_ingredients(session, token, herb_info)
        n_ing = len(ingredients)
        logger.info(f"  成分数: {n_ing}")
        n_ingredients += n_ing

        for ing in ingredients:
            all_results.append({
                "herb_cn_name": cn_name_matched,
                "herb_en_name": en_name,
                "herb_pinyin": pinyin,
                "zhuang_name": herb["zhuang_name"],
                "zhuang_volume": herb["volume"],
                "zhuang_year": herb["year"],
                "molecule_ID": ing.get("molecule_ID"),
                "MOL_ID": str(ing.get("MOL_ID", "")).strip(),
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

        done_set.add(cn_name_clean)
        n_done += 1

        if n_done % 10 == 0:
            save_progress(all_results)
            save_failed(failed_herbs)
            logger.info(f"  进度: {n_done}/{n_total} 完成, {n_failed} 失败, {n_ingredients} 成分")

        time.sleep(1.0)

    save_progress(all_results)
    save_failed(failed_herbs)

    logger.info("=" * 70)
    logger.info("爬取完成")
    logger.info(f"  总壮药: {n_total}")
    logger.info(f"  成功爬取: {n_done - n_failed}")
    logger.info(f"  失败: {n_failed}")
    logger.info(f"  总成分记录: {n_ingredients}")
    logger.info(f"  唯一MOL_ID: {pd.DataFrame(all_results)['MOL_ID'].nunique() if all_results else 0}")
    logger.info(f"  输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()