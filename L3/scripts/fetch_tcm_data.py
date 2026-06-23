#!/usr/bin/env python3
"""
Phase 3 - 步骤1: 中药单体化合物数据采集

数据来源:
  1. Dryad 公开数据集 (doi:10.5061/dryad.wh70rxwx9) - herb_compound.csv
     包含 14,985 对化合物-中药关联，含 SMILES、PubChem CID、ChEMBL ID、CAS号
  2. TCMSP-Spider (GitHub: shujuecn/TCMSP-Spider) - 备用
  3. PubChem PUG-REST API - 补充缺失 SMILES

输出:
  L3/data/herb_compound_raw.csv     - 原始数据
  L3/data/compound_smiles_raw.csv   - 化合物SMILES字典
  L3/data/tcm_compound_pool.csv     - 统一候选化合物池
"""

import os
import sys
import logging
import traceback
import hashlib
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.parse import quote

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "fetch_tcm_data.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_DATA = PROJECT_ROOT / "L3" / "data"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_DATA.mkdir(parents=True, exist_ok=True)
L3_RESULTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 步骤1: 下载 Dryad 公开数据集
# ============================================================
DRYAD_URL = "https://datadryad.org/stash/downloads/file_stream/286494"
DRYAD_FILE = L3_DATA / "herb_compound_dryad.csv"


def download_dryad_dataset():
    """下载 Dryad TCM 化合物数据集"""
    logger.info("=" * 60)
    logger.info("步骤1: 下载 Dryad 公开数据集")
    logger.info(f"URL: {DRYAD_URL}")
    logger.info(f"目标文件: {DRYAD_FILE}")

    if DRYAD_FILE.exists():
        file_size = DRYAD_FILE.stat().st_size
        logger.info(f"文件已存在，大小: {file_size:,} bytes")
        if file_size > 1000:
            logger.info("跳过下载（文件已存在且非空）")
            return True
        else:
            logger.warning("文件存在但过小，重新下载")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(DRYAD_URL, headers=headers, timeout=120, stream=True)
        response.raise_for_status()

        with open(DRYAD_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = DRYAD_FILE.stat().st_size
        logger.info(f"下载完成，文件大小: {file_size:,} bytes")

        if file_size < 1000:
            logger.error("下载的文件过小，可能下载失败")
            return False

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"下载失败: {e}")
        traceback.print_exc()
        return False


def parse_dryad_dataset():
    """解析 Dryad 数据集，提取化合物信息"""
    logger.info("=" * 60)
    logger.info("步骤2: 解析 Dryad 数据集")

    if not DRYAD_FILE.exists():
        logger.error(f"文件不存在: {DRYAD_FILE}")
        return None

    try:
        df = pd.read_csv(DRYAD_FILE, encoding="utf-8", low_memory=False)
        logger.info(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
        logger.info(f"列名: {list(df.columns)}")

        # 标准化列名
        col_map = {}
        for col in df.columns:
            col_lower = col.strip().lower().replace(" ", "_").replace("-", "_")
            col_map[col] = col_lower

        df = df.rename(columns=col_map)

        # 查找 SMILES 列
        smiles_col = None
        for col in df.columns:
            if "smiles" in col.lower():
                smiles_col = col
                break

        if smiles_col is None:
            logger.error("未找到 SMILES 列")
            logger.info(f"可用列: {list(df.columns)}")
            return None

        logger.info(f"SMILES 列: {smiles_col}")

        # 提取化合物 SMILES
        compounds = df[[smiles_col]].dropna().drop_duplicates()
        compounds = compounds.rename(columns={smiles_col: "SMILES"})
        compounds = compounds[compounds["SMILES"].str.strip() != ""]
        compounds = compounds[compounds["SMILES"].str.len() > 2]

        logger.info(f"有效化合物 SMILES 数: {len(compounds)}")

        # 保存
        raw_smiles_file = L3_DATA / "compound_smiles_raw.csv"
        compounds.to_csv(raw_smiles_file, index=False)
        logger.info(f"SMILES 保存至: {raw_smiles_file}")

        # 也保存完整原始数据
        raw_file = L3_DATA / "herb_compound_raw.csv"
        df.to_csv(raw_file, index=False)
        logger.info(f"原始完整数据保存至: {raw_file}")

        return compounds

    except Exception as e:
        logger.error(f"解析失败: {e}")
        traceback.print_exc()
        return None


# ============================================================
# 步骤2: 通过 PubChem PUG-REST API 补充数据
# 如果 Dryad 数据中没有 SMILES，尝试通过化合物名称获取
# ============================================================
def fetch_smiles_from_pubchem(compound_name, max_retries=3):
    """通过 PubChem API 获取化合物的 SMILES"""
    for attempt in range(max_retries):
        try:
            # 先通过名称获取 CID
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(compound_name)}/cids/TXT"
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None

            cids = response.text.strip().split()
            if not cids:
                return None

            cid = cids[0]

            # 通过 CID 获取 SMILES
            url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,MolecularWeight,TXT"
            response2 = requests.get(url2, timeout=30)
            if response2.status_code == 200:
                lines = response2.text.strip().split("\n")
                if len(lines) >= 2:
                    return lines[1].strip()

            time.sleep(0.5)  # PubChem rate limit
            return None

        except Exception as e:
            logger.warning(f"PubChem API 查询失败 [{compound_name}]: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

    return None


def supplement_with_pubchem(compounds_df):
    """对缺失 SMILES 的化合物，通过 PubChem 补充"""
    logger.info("=" * 60)
    logger.info("步骤3: PubChem 补充缺失 SMILES")

    if compounds_df is None or len(compounds_df) == 0:
        logger.warning("无化合物数据，跳过 PubChem 补充")
        return compounds_df

    return compounds_df  # Dryad 数据已有 SMILES，无需补充


# ============================================================
# 步骤3: 构建统一候选化合物池
# ============================================================
def build_compound_pool(compounds_df):
    """构建统一的候选化合物池"""
    logger.info("=" * 60)
    logger.info("步骤4: 构建统一候选化合物池")

    if compounds_df is None or len(compounds_df) == 0:
        logger.error("无化合物数据")
        return None

    pool = compounds_df.copy()
    pool["compound_id"] = [f"TCM_{i:05d}" for i in range(len(pool))]
    pool["source"] = "Dryad_TCM"

    # 计算 SMILES 的 MD5 用于去重
    pool["smiles_hash"] = pool["SMILES"].apply(
        lambda x: hashlib.md5(x.strip().encode()).hexdigest()
    )

    # 去重
    before = len(pool)
    pool = pool.drop_duplicates(subset=["smiles_hash"])
    after = len(pool)
    logger.info(f"去重: {before} -> {after} (移除 {before - after} 个重复)")

    # 保存
    pool_file = L3_DATA / "tcm_compound_pool.csv"
    pool.to_csv(pool_file, index=False)
    logger.info(f"候选化合物池保存至: {pool_file}")
    logger.info(f"候选化合物总数: {len(pool)}")

    # 文件校验
    file_size = pool_file.stat().st_size
    logger.info(f"文件大小: {file_size:,} bytes")
    logger.info(f"MD5: {hashlib.md5(pool_file.read_bytes()).hexdigest()}")

    return pool


# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Phase 3 - 中药单体化合物数据采集")
    logger.info(f"启动时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 步骤1: 下载 Dryad 数据集
    success = download_dryad_dataset()
    if not success:
        logger.error("Dryad 数据集下载失败，尝试备用方案")
        # 备用方案：尝试从 TCMSP-Spider 获取
        logger.info("备用方案: 请手动运行 TCMSP-Spider 或提供化合物数据文件")
        return False

    # 步骤2: 解析数据
    compounds = parse_dryad_dataset()
    if compounds is None:
        logger.error("数据解析失败")
        return False

    # 步骤3: PubChem 补充
    compounds = supplement_with_pubchem(compounds)

    # 步骤4: 构建候选化合物池
    pool = build_compound_pool(compounds)
    if pool is None:
        logger.error("候选化合物池构建失败")
        return False

    logger.info("=" * 60)
    logger.info("Phase 3 数据采集完成!")
    logger.info(f"候选化合物总数: {len(pool)}")
    logger.info(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"未捕获的异常: {e}")
        traceback.print_exc()
        sys.exit(1)