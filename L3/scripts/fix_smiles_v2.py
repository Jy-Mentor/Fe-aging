"""
修复 SMILES 获取 v2：使用 pubchempy 批量获取 + 多候选匹配验证
优于 REST API 的原因：pubchempy 单次调用返回多个候选，无需逐 CID 查询
"""
import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import pubchempy as pcp
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


def name_similarity(a, b):
    """计算两个名称的相似度 (0-1)，忽略大小写和空格"""
    a = a.lower().strip().replace("_", " ").replace("-", " ")
    b = b.lower().strip().replace("_", " ").replace("-", " ")
    return SequenceMatcher(None, a, b).ratio()


def get_smiles_from_pubchem_v2(name, mw_tcmsp, max_mw_diff=5.0):
    """
    通过 pubchempy 获取化合物的 SMILES，使用多候选匹配：
    1. 搜索名称，获取最多 5 个候选（带超时）
    2. 对每个候选检查 Title 匹配度和 MW 差异
    3. 返回最佳匹配的 SMILES
    
    返回: (smiles, info_dict) 或 (None, None)
    """
    result = [None, None]
    
    def _query():
        try:
            compounds = pcp.get_compounds(name, 'name', listkey_count=5)
            if not compounds:
                # 尝试简化名称
                simplified = name.split("(")[0].strip()
                if simplified != name and len(simplified) > 3:
                    compounds = pcp.get_compounds(simplified, 'name', listkey_count=5)
            
            if not compounds:
                return
            
            best_score = 0
            best_smiles = None
            best_info = None
            
            for cpd in compounds[:5]:
                if not cpd.canonical_smiles:
                    continue
                
                pub_title = cpd.synonyms[0] if cpd.synonyms else cpd.iupac_name or ""
                pub_mw = float(cpd.molecular_weight) if cpd.molecular_weight else 0
                pub_smiles = cpd.canonical_smiles
                
                title_sim = name_similarity(name, pub_title)
                
                if pub_mw > 0:
                    mw_diff = abs(pub_mw - mw_tcmsp)
                else:
                    mol = Chem.MolFromSmiles(pub_smiles)
                    if mol:
                        from rdkit.Chem import Descriptors
                        pub_mw = Descriptors.MolWt(mol)
                        mw_diff = abs(pub_mw - mw_tcmsp)
                    else:
                        mw_diff = 999
                
                mw_ok = (mw_diff <= max_mw_diff) or (mw_tcmsp > 0 and mw_diff / mw_tcmsp <= 0.05)
                
                score = title_sim * 0.5 + (1.0 - min(mw_diff / max(mw_tcmsp, 1), 1.0)) * 0.5
                if mw_ok:
                    score += 0.3
                
                if score > best_score:
                    best_score = score
                    best_smiles = pub_smiles
                    best_info = {
                        'cid': cpd.cid,
                        'pub_title': pub_title,
                        'pub_mw': pub_mw,
                        'name_sim': round(title_sim, 3),
                        'mw_diff': round(mw_diff, 1),
                        'score': round(score, 3),
                    }
            
            result[0] = best_smiles
            result[1] = best_info
        except Exception as e:
            logger.debug(f"  pubchempy 查询异常 [{name}]: {e}")
    
    t = threading.Thread(target=_query, daemon=True)
    t.start()
    t.join(timeout=10)  # 10秒超时
    if t.is_alive():
        logger.debug(f"  pubchempy 查询超时 [{name}]")
        return None, None
    
    return result[0], result[1]


def fix_smiles():
    logger.info("=" * 70)
    logger.info("SMILES 修复 v2：使用 pubchempy 批量获取 + 多候选匹配")
    logger.info("=" * 70)
    
    # 加载 TCMSP 原始数据
    df = pd.read_excel(TCMSP_INGREDIENTS)
    logger.info(f"TCMSP 化合物总数: {len(df):,}")
    
    # OB/DL 过滤
    active = df[(df["ob"] >= 30.0) & (df["dl"] >= 0.18)].copy().reset_index(drop=True)
    logger.info(f"OB/DL 过滤后: {len(active):,}")
    
    # 去重名称
    name_groups = active.groupby("molecule_name")
    names = list(name_groups.groups.keys())
    logger.info(f"唯一名称: {len(names)}")
    
    # 加载缓存
    cache = {}
    if NEW_SMILES_CACHE.exists():
        with open(NEW_SMILES_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"已有缓存: {len(cache)} 条")
    
    # 批量获取
    smiles_map = {}
    stats = {"NAME_MATCH": 0, "MW_MATCH": 0, "PARTIAL": 0, "FAILED": 0}
    failed = []
    
    t0 = time.time()
    for i, name in enumerate(names):
        # 从缓存取
        if name in cache:
            entry = cache[name]
            if isinstance(entry, dict):
                smiles_map[name] = entry['smiles']
            else:
                smiles_map[name] = entry
            continue
        
        # 获取 TCMSP MW
        mw = float(name_groups.get_group(name).iloc[0]["mw"])
        
        # 查询 PubChem
        smiles, info = get_smiles_from_pubchem_v2(name, mw)
        
        if smiles and info:
            smiles_map[name] = smiles
            cache[name] = {
                'smiles': smiles,
                'cid': info['cid'],
                'pub_title': info['pub_title'],
                'name_sim': info['name_sim'],
                'mw_diff': info['mw_diff'],
                'score': info['score'],
            }
            
            if info['name_sim'] >= 0.8 and info['mw_diff'] <= 5.0:
                stats["NAME_MATCH"] += 1
            elif info['mw_diff'] <= 5.0:
                stats["MW_MATCH"] += 1
            else:
                stats["PARTIAL"] += 1
        else:
            stats["FAILED"] += 1
            failed.append(name)
        
        # 进度 & 缓存
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(names) - i - 1) / rate / 60
            logger.info(f"  [{i+1}/{len(names)}] ({elapsed:.0f}s, {rate:.1f}/s, ETA={eta:.1f}min) | "
                       f"OK={len(smiles_map)} | FAIL={len(failed)} | "
                       f"NAME={stats['NAME_MATCH']} MW={stats['MW_MATCH']} PART={stats['PARTIAL']}")
            with open(NEW_SMILES_CACHE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        
        time.sleep(0.15)  # 礼貌
    
    # 最终保存
    with open(NEW_SMILES_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    # 统计
    elapsed = time.time() - t0
    logger.info(f"\n=== 完成 ===")
    logger.info(f"  总耗时: {elapsed/60:.1f}min")
    logger.info(f"  成功: {len(smiles_map)}/{len(names)} ({len(smiles_map)/len(names)*100:.1f}%)")
    logger.info(f"  失败: {len(failed)}")
    logger.info(f"  匹配: NAME={stats['NAME_MATCH']} MW={stats['MW_MATCH']} PART={stats['PARTIAL']}")
    
    # 验证关键化合物
    logger.info(f"\n=== 关键化合物验证 ===")
    key_mols = [
        ("MOL000001", "anthocyanidin"),
        ("MOL000422", "kaempferol"),
        ("MOL000173", "wogonin"),
        ("MOL000098", "quercetin"),
        ("MOL002288", "Emodin-1-O-beta-D-glucopyranoside"),
        ("MOL004328", "naringenin"),
        ("MOL000006", "luteolin"),
        ("MOL007424", "artemisinin"),
        ("MOL000392", "formononetin"),
        ("MOL001454", "berberine"),
    ]
    for mol_id, name in key_mols:
        row = active[active["MOL_ID"] == mol_id]
        if len(row) == 0:
            continue
        mw = float(row["mw"].values[0])
        if name in smiles_map:
            smi = smiles_map[name]
            mol = Chem.MolFromSmiles(smi)
            if mol:
                from rdkit.Chem import Descriptors
                rdkit_mw = Descriptors.MolWt(mol)
                diff = abs(rdkit_mw - mw)
                status = "OK" if diff < 5 else f"MISMATCH(diff={diff:.1f})"
                logger.info(f"  {mol_id} {name}: {smi[:60]}... MW_diff={diff:.1f} [{status}]")
            else:
                logger.warning(f"  {mol_id} {name}: INVALID SMILES")
        else:
            logger.warning(f"  {mol_id} {name}: NOT FOUND")
    
    # 保存 SMILES 映射
    rows = []
    for name, smiles in smiles_map.items():
        entry = cache.get(name, {})
        if isinstance(entry, dict):
            rows.append({
                "molecule_name": name,
                "SMILES": smiles,
                "PubChem_CID": entry.get("cid", ""),
                "PubChem_Title": entry.get("pub_title", ""),
                "Name_Similarity": entry.get("name_sim", 0),
                "MW_Diff": entry.get("mw_diff", 0),
                "Score": entry.get("score", 0),
            })
    
    mapping_df = pd.DataFrame(rows)
    mapping_path = L3_RESULTS / "tcmsp_smiles_fixed.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"\nSMILES 映射表: {mapping_path} ({len(mapping_df)} 条)")
    
    # 失败列表
    if failed:
        pd.DataFrame({"molecule_name": failed}).to_csv(
            L3_RESULTS / "smiles_fix_failed.csv", index=False
        )
        logger.info(f"失败列表: {L3_RESULTS / 'smiles_fix_failed.csv'} ({len(failed)} 个)")
    
    logger.info("=" * 70)
    return smiles_map, mapping_df


if __name__ == "__main__":
    fix_smiles()