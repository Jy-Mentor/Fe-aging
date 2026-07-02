"""
修复 SMILES 获取 v3：PubChem REST API + 多进程并行
每化合物仅2次API调用（CID搜索 + SMILES获取），4进程并行
"""
import os, sys, json, time, logging
from pathlib import Path
from multiprocessing import Pool
import traceback

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

# 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_ROOT = PROJECT_ROOT / "L3"
L3_DATA = L3_ROOT / "data"
L3_RESULTS = L3_ROOT / "results"
L3_LOGS = L3_ROOT / "logs"
TCMSP_DIR = L3_ROOT / "TCMSP-Spider" / "data" / "sample_data"
TCMSP_INGREDIENTS = TCMSP_DIR / "ingredients_data.xlsx"
NEW_CACHE = L3_DATA / "pubchem_smiles_cache_v3.json"

for d in [L3_DATA, L3_RESULTS, L3_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L3_LOGS / "fix_smiles_v3.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def query_one(name_mw):
    """查询单个化合物：返回 (name, smiles, cid, pub_mw, mw_diff) 或 (name, None, ...)"""
    name, mw = name_mw
    try:
        # Step 1: 搜索 CID
        r = requests.get(f"{PUBCHEM}/compound/name/{name}/cids/JSON", timeout=8)
        if r.status_code != 200 or 'IdentifierList' not in r.json():
            return (name, None, 0, 0, 0)
        
        cid = r.json()['IdentifierList']['CID'][0]
        
        # Step 2: 获取 SMILES + MW
        r2 = requests.get(
            f"{PUBCHEM}/compound/cid/{cid}/property/CanonicalSMILES,MolecularWeight/JSON",
            timeout=8
        )
        if r2.status_code != 200:
            return (name, None, cid, 0, 0)
        
        props = r2.json()['PropertyTable']['Properties'][0]
        smiles = props.get('CanonicalSMILES', '')
        pub_mw = float(props.get('MolecularWeight', 0))
        
        if not smiles:
            return (name, None, cid, pub_mw, 0)
        
        # Step 3: 验证 MW
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                rdkit_mw = Descriptors.MolWt(mol)
                mw_diff = abs(rdkit_mw - mw)
            else:
                mw_diff = 999
        except:
            mw_diff = abs(pub_mw - mw) if pub_mw > 0 else 999
        
        return (name, smiles, cid, pub_mw, mw_diff)
    
    except Exception as e:
        return (name, None, 0, 0, 0)


def main():
    logger.info("=" * 70)
    logger.info("SMILES 修复 v3：PubChem REST API + 多进程并行")
    logger.info("=" * 70)
    
    # 加载数据
    df = pd.read_excel(TCMSP_INGREDIENTS)
    logger.info(f"TCMSP 化合物总数: {len(df):,}")
    
    active = df[(df["ob"] >= 30.0) & (df["dl"] >= 0.18)].copy().reset_index(drop=True)
    logger.info(f"OB/DL 过滤后: {len(active):,}")
    
    # 构建查询列表：(name, mw)
    name_mw_list = list(zip(active["molecule_name"], active["mw"]))
    
    # 去重（同名称可能对应多个 MOL_ID，但 SMILES 相同）
    seen = set()
    unique_queries = []
    for nm in name_mw_list:
        if nm[0] not in seen:
            seen.add(nm[0])
            unique_queries.append(nm)
    
    logger.info(f"唯一名称: {len(unique_queries)}")
    
    # 加载缓存
    cache = {}
    if NEW_CACHE.exists():
        with open(NEW_CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"已有缓存: {len(cache)} 条")
    
    # 过滤已缓存
    to_query = []
    for nm in unique_queries:
        if nm[0] not in cache:
            to_query.append(nm)
    
    logger.info(f"待查询: {len(to_query)} (已缓存: {len(unique_queries) - len(to_query)})")
    
    if len(to_query) == 0:
        logger.info("全部已缓存，跳过查询")
    else:
        t0 = time.time()
        n_workers = 4
        
        with Pool(n_workers) as pool:
            results = pool.map(query_one, to_query)
        
        elapsed = time.time() - t0
        logger.info(f"查询完成: {len(results)} 个化合物, {elapsed:.0f}s ({len(results)/elapsed:.1f}/s)")
        
        # 更新缓存
        ok_count = 0
        fail_count = 0
        for name, smiles, cid, pub_mw, mw_diff in results:
            if smiles:
                cache[name] = {
                    'smiles': smiles,
                    'cid': cid,
                    'pub_mw': pub_mw,
                    'mw_diff': round(mw_diff, 1),
                }
                ok_count += 1
            else:
                cache[name] = None
                fail_count += 1
        
        logger.info(f"成功: {ok_count}, 失败: {fail_count} ({ok_count/len(results)*100:.1f}%)")
        
        # 保存缓存
        with open(NEW_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    
    # 构建 SMILES map
    smiles_map = {}
    for name, entry in cache.items():
        if entry and isinstance(entry, dict) and 'smiles' in entry:
            # 验证 MW
            row = active[active["molecule_name"] == name]
            if len(row) > 0:
                mw = float(row["mw"].values[0])
                mw_diff = entry.get('mw_diff', 0)
                if mw_diff <= 5 or (mw > 0 and mw_diff / mw <= 0.05):
                    smiles_map[name] = entry['smiles']
    
    logger.info(f"SMILES map (MW验证通过): {len(smiles_map)}")
    
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
                rdkit_mw = Descriptors.MolWt(mol)
                diff = abs(rdkit_mw - mw)
                status = "OK" if diff < 5 else f"MISMATCH(diff={diff:.1f})"
                logger.info(f"  {mol_id} {name}: {smi[:60]}... MW_diff={diff:.1f} [{status}]")
            else:
                logger.warning(f"  {mol_id} {name}: INVALID SMILES")
        else:
            logger.warning(f"  {mol_id} {name}: NOT FOUND")
    
    # 保存映射表
    rows = []
    for name, smi in smiles_map.items():
        entry = cache.get(name, {})
        rows.append({
            "molecule_name": name,
            "SMILES": smi,
            "PubChem_CID": entry.get("cid", "") if isinstance(entry, dict) else "",
            "PubChem_MW": entry.get("pub_mw", 0) if isinstance(entry, dict) else 0,
            "MW_Diff": entry.get("mw_diff", 0) if isinstance(entry, dict) else 0,
        })
    
    mapping_df = pd.DataFrame(rows)
    mapping_path = L3_RESULTS / "tcmsp_smiles_fixed_v3.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"\nSMILES 映射表: {mapping_path} ({len(mapping_df)} 条)")
    
    logger.info("=" * 70)
    logger.info("SMILES 修复完成!")
    return smiles_map, mapping_df


if __name__ == "__main__":
    main()