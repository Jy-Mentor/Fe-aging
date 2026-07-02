"""
修复 SMILES 获取：使用 PubChem REST API 直接搜索 + 名称模糊匹配验证
替代 phase3_pipeline.py 中不可靠的 COCONUT/PubChem 名称搜索
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import requests
from rdkit import Chem

# 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_ROOT = PROJECT_ROOT / "L3"
L3_DATA = L3_ROOT / "data"
L3_RESULTS = L3_ROOT / "results"
L3_LOGS = L3_ROOT / "logs"
TCMSP_DIR = L3_ROOT / "TCMSP-Spider" / "data" / "sample_data"

for d in [L3_DATA, L3_RESULTS, L3_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

NEW_SMILES_CACHE = L3_DATA / "pubchem_smiles_cache_v2.json"
TCMSP_INGREDIENTS = TCMSP_DIR / "ingredients_data.xlsx"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L3_LOGS / "fix_smiles.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def name_similarity(a, b):
    """计算两个名称的相似度 (0-1)"""
    a = a.lower().strip()
    b = b.lower().strip()
    return SequenceMatcher(None, a, b).ratio()


def search_pubchem_by_name(name, timeout=10):
    """
    通过名称搜索 PubChem，返回 CID 列表。
    使用 PubChem REST API 的 name 端点。
    """
    try:
        url = f"{PUBCHEM_BASE}/compound/name/{name}/cids/JSON"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if 'IdentifierList' in data:
                return data['IdentifierList']['CID']
    except Exception as e:
        logger.debug(f"  PubChem搜索失败 [{name}]: {e}")
    return []


def get_compound_info(cid, timeout=10):
    """
    通过 CID 获取化合物的标题、SMILES 和分子量。
    """
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/property/Title,CanonicalSMILES,MolecularWeight,InChIKey/JSON"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if 'PropertyTable' in data:
                return data['PropertyTable']['Properties'][0]
    except Exception as e:
        logger.debug(f"  PubChem属性获取失败 [CID={cid}]: {e}")
    return None


def get_compound_synonyms(cid, timeout=10):
    """获取化合物的同义词列表"""
    try:
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            if 'InformationList' in data:
                return data['InformationList']['Information'][0].get('Synonym', [])
    except Exception:
        pass
    return []


def find_best_match(name, cids, mw_tcmsp, max_mw_diff=5.0):
    """
    从 CID 列表中找出最佳匹配（仅用 Title 对比，不查同义词以提速）：
    1. 检查 PubChem 标题是否与 TCMSP 名称匹配
    2. 检查 MW 是否在允许范围内
    3. 返回最佳匹配的 SMILES
    """
    best_match = None
    best_score = 0
    best_info = None
    best_reason = ""

    for cid in cids[:3]:  # 只检查前3个候选
        info = get_compound_info(cid)
        if info is None:
            continue

        pub_title = info.get('Title', '')
        pub_mw = float(info.get('MolecularWeight', 0))
        pub_smiles = info.get('CanonicalSMILES', '')
        pub_inchikey = info.get('InChIKey', '')

        if not pub_smiles:
            continue

        # 检查MW
        mw_diff = abs(pub_mw - mw_tcmsp)
        mw_ok = mw_diff <= max_mw_diff or (mw_tcmsp > 0 and mw_diff / mw_tcmsp <= 0.05)

        # 检查名称匹配（仅Title，不查同义词加速）
        title_sim = name_similarity(name, pub_title)

        # 评分：名称匹配 + MW匹配
        score = title_sim * 0.6 + (1.0 - min(mw_diff / max(mw_tcmsp, 1), 1.0)) * 0.4

        if mw_ok:
            score += 0.2  # MW匹配加分

        if score > best_score:
            best_score = score
            best_match = pub_smiles
            best_info = {
                'cid': cid,
                'pub_title': pub_title,
                'pub_mw': pub_mw,
                'name_sim': title_sim,
                'mw_diff': mw_diff,
                'inchikey': pub_inchikey,
            }
            if title_sim >= 0.8 and mw_ok:
                best_reason = "NAME_MATCH"
            elif mw_ok:
                best_reason = "MW_MATCH"
            elif title_sim >= 0.5:
                best_reason = "PARTIAL_NAME"

    return best_match, best_info, best_score, best_reason


def fix_smiles():
    """主函数：重新获取所有化合物的SMILES"""
    logger.info("=" * 70)
    logger.info("SMILES 修复：使用 PubChem REST API + 名称匹配验证")
    logger.info("=" * 70)

    # 加载 TCMSP 原始数据
    df = pd.read_excel(TCMSP_INGREDIENTS)
    logger.info(f"TCMSP 化合物总数: {len(df):,}")

    # 应用 OB/DL 过滤（与 phase3 一致）
    active = df[(df["ob"] >= 30.0) & (df["dl"] >= 0.18)].copy()
    active = active.reset_index(drop=True)
    logger.info(f"OB/DL 过滤后: {len(active):,}")

    # 加载已有缓存
    cache = {}
    if NEW_SMILES_CACHE.exists():
        with open(NEW_SMILES_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"已有缓存: {len(cache)} 条")

    # 重新获取 SMILES
    names = active["molecule_name"].dropna().unique().tolist()
    logger.info(f"待查询: {len(names)} 个唯一名称")

    smiles_map = {}
    match_reasons = {"NAME_MATCH": 0, "MW_MATCH": 0, "PARTIAL_NAME": 0}
    failed = []
    suspicious = []

    for i, name in enumerate(names):
        if name in cache:
            entry = cache[name]
            if isinstance(entry, dict):
                smiles_map[name] = entry['smiles']
            else:
                smiles_map[name] = entry
            continue

        row = active[active["molecule_name"] == name].iloc[0]
        mw = float(row["mw"])

        # 搜索 PubChem
        cids = search_pubchem_by_name(name)
        if not cids:
            # 尝试简化名称（去掉空格、特殊字符）
            simplified = name.replace(" ", "").replace("-", " ").replace("_", " ")
            if simplified != name:
                cids = search_pubchem_by_name(simplified)

        if cids:
            smiles, info, score, reason = find_best_match(name, cids, mw)
            if smiles and score >= 0.5:
                smiles_map[name] = smiles
                cache[name] = {
                    'smiles': smiles,
                    'cid': info['cid'],
                    'pub_title': info['pub_title'],
                    'name_sim': info['name_sim'],
                    'mw_diff': info['mw_diff'],
                    'reason': reason,
                    'inchikey': info['inchikey'],
                }
                match_reasons[reason] = match_reasons.get(reason, 0) + 1
                if score < 0.7:
                    suspicious.append((name, score, reason))
            else:
                failed.append((name, "LOW_SCORE", f"score={score:.2f}"))
        else:
            failed.append((name, "NOT_FOUND", "no PubChem CID"))

        # 进度报告
        if (i + 1) % 100 == 0:
            logger.info(f"  进度: {i+1}/{len(names)} | "
                       f"MATCH: {len(smiles_map)} | FAILED: {len(failed)} | "
                       f"NAME: {match_reasons['NAME_MATCH']} | "
                       f"MW: {match_reasons['MW_MATCH']} | "
                       f"PARTIAL: {match_reasons['PARTIAL_NAME']}")

        time.sleep(0.25)  # 礼貌爬取

    # 保存缓存
    with open(NEW_SMILES_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # 统计
    logger.info(f"\n=== SMILES 获取结果 ===")
    logger.info(f"  成功: {len(smiles_map)}/{len(names)} ({len(smiles_map)/len(names)*100:.1f}%)")
    logger.info(f"  失败: {len(failed)}")
    logger.info(f"  匹配方式: NAME={match_reasons['NAME_MATCH']}, "
               f"MW={match_reasons['MW_MATCH']}, PARTIAL={match_reasons['PARTIAL_NAME']}")

    # 验证 SMILES 质量：随机抽查几个关键化合物
    logger.info(f"\n=== 关键化合物验证 ===")
    key_compounds = [
        ("MOL000001", "anthocyanidin"),
        ("MOL000422", "kaempferol"),
        ("MOL000173", "wogonin"),
        ("MOL000392", "formononetin"),
        ("MOL000098", "quercetin"),
        ("MOL002288", "Emodin-1-O-beta-D-glucopyranoside"),
        ("MOL001001", "quercetin-3-O-beta-D-glucuronide"),
        ("MOL004328", "naringenin"),
        ("MOL000006", "luteolin"),
        ("MOL007424", "artemisinin"),
    ]
    for mol_id, name in key_compounds:
        row = active[active["MOL_ID"] == mol_id]
        if len(row) == 0:
            continue
        mw = float(row["mw"].values[0])
        if name in smiles_map:
            smiles = smiles_map[name]
            # 验证MW
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                from rdkit.Chem import Descriptors
                rdkit_mw = Descriptors.MolWt(mol)
                diff = abs(rdkit_mw - mw)
                status = "OK" if diff < 5 else f"MISMATCH (diff={diff:.1f})"
                logger.info(f"  {mol_id} {name}: SMILES={smiles[:60]}... MW_diff={diff:.1f} [{status}]")
            else:
                logger.warning(f"  {mol_id} {name}: INVALID SMILES")
        else:
            logger.warning(f"  {mol_id} {name}: NOT FOUND")

    # 保存失败列表
    if failed:
        failed_df = pd.DataFrame(failed, columns=["name", "reason", "detail"])
        failed_path = L3_RESULTS / "smiles_fix_failed.csv"
        failed_df.to_csv(failed_path, index=False)
        logger.info(f"\n失败列表已保存: {failed_path} ({len(failed)} 个)")

    # 保存可疑列表
    if suspicious:
        susp_df = pd.DataFrame(suspicious, columns=["name", "score", "reason"])
        susp_path = L3_RESULTS / "smiles_fix_suspicious.csv"
        susp_df.to_csv(susp_path, index=False)
        logger.info(f"可疑列表已保存: {susp_path} ({len(suspicious)} 个)")

    # 保存 SMILES 映射到 CSV（供 phase3 使用）
    mapping_rows = []
    for name, smiles in smiles_map.items():
        entry = cache.get(name, {})
        if isinstance(entry, dict):
            mapping_rows.append({
                "molecule_name": name,
                "SMILES": smiles,
                "PubChem_CID": entry.get("cid", ""),
                "PubChem_Title": entry.get("pub_title", ""),
                "Name_Similarity": entry.get("name_sim", 0),
                "MW_Diff": entry.get("mw_diff", 0),
                "Match_Reason": entry.get("reason", ""),
                "InChIKey": entry.get("inchikey", ""),
            })
        else:
            mapping_rows.append({
                "molecule_name": name,
                "SMILES": smiles,
                "PubChem_CID": "",
                "PubChem_Title": "",
                "Name_Similarity": 0,
                "MW_Diff": 0,
                "Match_Reason": "CACHED",
                "InChIKey": "",
            })

    mapping_df = pd.DataFrame(mapping_rows)
    mapping_path = L3_RESULTS / "tcmsp_smiles_fixed.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"SMILES 映射表已保存: {mapping_path} ({len(mapping_df)} 条)")

    logger.info("=" * 70)
    logger.info("SMILES 修复完成!")
    logger.info("=" * 70)

    return smiles_map, mapping_df


if __name__ == "__main__":
    fix_smiles()