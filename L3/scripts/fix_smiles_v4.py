"""
修复 SMILES v4：COCONUT本地 + 旧PubChem缓存验证 + 增量补充查询
策略：先用本地数据库，仅对缺失的化合物增量查询PubChem
"""
import os, sys, json, time, logging, hashlib
from pathlib import Path
from difflib import SequenceMatcher
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
COCONUT_CSV = L3_DATA / "coconut_csv" / "coconut_csv_lite-05-2026.csv"
OLD_CACHE = L3_DATA / "pubchem_smiles_cache.json"
NEW_CACHE = L3_DATA / "pubchem_smiles_cache_v4.json"

for d in [L3_DATA, L3_RESULTS, L3_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L3_LOGS / "fix_smiles_v4.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def load_coconut():
    """加载 COCONUT 数据库，建立名称→SMILES映射"""
    if not COCONUT_CSV.exists():
        logger.warning("COCONUT数据库未找到")
        return {}
    
    df = pd.read_csv(COCONUT_CSV, usecols=["canonical_smiles", "name", "molecular_weight"], low_memory=False)
    df = df.dropna(subset=["canonical_smiles"])
    df["name_lower"] = df["name"].str.lower().str.strip()
    
    coconut_map = {}
    for _, row in df.iterrows():
        name = row["name_lower"]
        # 只保留第一个匹配（通常是最佳匹配）
        if name not in coconut_map:
            coconut_map[name] = {
                "smiles": row["canonical_smiles"],
                "mw": float(row["molecular_weight"]) if pd.notna(row["molecular_weight"]) else 0,
            }
    
    logger.info(f"COCONUT: {len(coconut_map):,} 条唯一名称映射")
    return coconut_map


def load_old_cache():
    """加载旧 PubChem 缓存"""
    if not OLD_CACHE.exists():
        return {}
    with open(OLD_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    logger.info(f"旧PubChem缓存: {len(cache)} 条")
    return cache


def verify_smiles_mw(smiles, mw_tcmsp, max_diff=5.0):
    """用RDKit验证SMILES的MW是否与TCMSP一致"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, 999
        rdkit_mw = Descriptors.MolWt(mol)
        diff = abs(rdkit_mw - mw_tcmsp)
        rel_diff = diff / max(mw_tcmsp, 1)
        return (diff <= max_diff or rel_diff <= 0.05), diff
    except:
        return False, 999


def query_pubchem(name):
    """查询单个化合物的PubChem SMILES"""
    try:
        r = requests.get(f"{PUBCHEM}/compound/name/{name}/cids/JSON", timeout=8)
        if r.status_code != 200 or 'IdentifierList' not in r.json():
            return None, None, 0
        
        cid = r.json()['IdentifierList']['CID'][0]
        r2 = requests.get(
            f"{PUBCHEM}/compound/cid/{cid}/property/CanonicalSMILES,MolecularWeight/JSON",
            timeout=8
        )
        if r2.status_code != 200:
            return None, cid, 0
        
        props = r2.json()['PropertyTable']['Properties'][0]
        smiles = props.get('CanonicalSMILES', '')
        pub_mw = float(props.get('MolecularWeight', 0))
        
        return smiles, cid, pub_mw
    except:
        return None, None, 0


def main():
    logger.info("=" * 70)
    logger.info("SMILES 修复 v4：COCONUT本地 + 缓存验证 + 增量PubChem")
    logger.info("=" * 70)
    
    # 加载数据
    tcmsp = pd.read_excel(TCMSP_INGREDIENTS)
    active = tcmsp[(tcmsp["ob"] >= 30.0) & (tcmsp["dl"] >= 0.18)].copy()
    logger.info(f"TCMSP 活性化合物: {len(active)}")
    
    # 构建名称→MW映射
    name_mw = {}
    for _, row in active.iterrows():
        name = row["molecule_name"]
        if pd.notna(name):
            name_mw[name] = float(row["mw"])
    
    unique_names = list(name_mw.keys())
    logger.info(f"唯一名称: {len(unique_names)}")
    
    # Step 1: 加载本地数据库
    coconut = load_coconut()
    old_cache = load_old_cache()
    
    # Step 2: 三级匹配策略
    smiles_map = {}
    sources = {"COCONUT": 0, "OLD_CACHE": 0, "PUBCHEM": 0}
    failed = []
    to_query = []
    
    for name in unique_names:
        mw = name_mw[name]
        name_lower = name.lower().strip()
        
        # Level 1: COCONUT 精确名称匹配
        if name_lower in coconut:
            c_entry = coconut[name_lower]
            ok, diff = verify_smiles_mw(c_entry["smiles"], mw)
            if ok:
                smiles_map[name] = c_entry["smiles"]
                sources["COCONUT"] += 1
                continue
        
        # Level 2: 旧 PubChem 缓存
        if name in old_cache:
            cached = old_cache[name]
            if isinstance(cached, str):
                ok, diff = verify_smiles_mw(cached, mw)
                if ok:
                    smiles_map[name] = cached
                    sources["OLD_CACHE"] += 1
                    continue
        
        # Level 3: 需要查询 PubChem
        to_query.append(name)
    
    logger.info(f"COCONUT: {sources['COCONUT']}, OLD_CACHE: {sources['OLD_CACHE']}, 待查询: {len(to_query)}")
    
    # Step 3: 增量查询 PubChem
    new_cache = {}
    if NEW_CACHE.exists():
        with open(NEW_CACHE, "r", encoding="utf-8") as f:
            new_cache = json.load(f)
        logger.info(f"新缓存已有: {len(new_cache)} 条")
        
        # 从新缓存中匹配
        for name in to_query[:]:
            if name in new_cache:
                entry = new_cache[name]
                if entry and isinstance(entry, dict) and 'smiles' in entry:
                    ok, diff = verify_smiles_mw(entry['smiles'], name_mw[name])
                    if ok:
                        smiles_map[name] = entry['smiles']
                        sources["PUBCHEM"] += 1
                        to_query.remove(name)
        
        logger.info(f"从新缓存匹配: {sources['PUBCHEM']}, 仍需查询: {len(to_query)}")
    
    if to_query:
        t0 = time.time()
        logger.info(f"开始查询 PubChem: {len(to_query)} 个化合物...")
        
        for i, name in enumerate(to_query):
            mw = name_mw[name]
            smiles, cid, pub_mw = query_pubchem(name)
            
            if smiles:
                ok, diff = verify_smiles_mw(smiles, mw)
                if ok:
                    smiles_map[name] = smiles
                    new_cache[name] = {
                        "smiles": smiles,
                        "cid": cid,
                        "pub_mw": pub_mw,
                        "mw_diff": round(diff, 1),
                    }
                    sources["PUBCHEM"] += 1
                else:
                    new_cache[name] = None
                    failed.append((name, "MW_MISMATCH", f"diff={diff:.1f}"))
            else:
                new_cache[name] = None
                failed.append((name, "NOT_FOUND", ""))
            
            # 进度 & 缓存
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(to_query) - i - 1) / rate / 60
                logger.info(f"  [{i+1}/{len(to_query)}] ({elapsed:.1f}s, {rate:.1f}/s, ETA={eta:.1f}min) | "
                           f"OK={sources['PUBCHEM']} | FAIL={len(failed)}")
                with open(NEW_CACHE, "w", encoding="utf-8") as f:
                    json.dump(new_cache, f, ensure_ascii=False, indent=2)
            
            time.sleep(0.1)
        
        # 最终保存
        with open(NEW_CACHE, "w", encoding="utf-8") as f:
            json.dump(new_cache, f, ensure_ascii=False, indent=2)
        
        elapsed = time.time() - t0
        logger.info(f"PubChem查询完成: {elapsed:.0f}s ({len(to_query)/elapsed:.1f}/s)")
    
    # 统计
    logger.info(f"\n=== 最终结果 ===")
    logger.info(f"总成功: {len(smiles_map)}/{len(unique_names)} ({len(smiles_map)/len(unique_names)*100:.1f}%)")
    logger.info(f"  COCONUT: {sources['COCONUT']}")
    logger.info(f"  OLD_CACHE: {sources['OLD_CACHE']}")
    logger.info(f"  PUBCHEM: {sources['PUBCHEM']}")
    logger.info(f"  失败: {len(failed)}")
    
    # 验证关键化合物
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
        ("MOL001001", "quercetin-3-O-beta-D-glucuronide"),
        ("MOL000470", "8-C-alpha-L-arabinosylluteolin"),
    ]
    logger.info(f"\n=== 关键化合物验证 ===")
    for mol_id, name in key_mols:
        r = active[active["MOL_ID"] == mol_id]
        if len(r) == 0:
            continue
        mw = float(r["mw"].values[0])
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
    
    # 保存完整映射表
    rows = []
    for name, smi in smiles_map.items():
        mw = name_mw.get(name, 0)
        mol = Chem.MolFromSmiles(smi)
        rdkit_mw = Descriptors.MolWt(mol) if mol else 0
        rows.append({
            "molecule_name": name,
            "SMILES": smi,
            "MW_TCMSP": mw,
            "MW_RDKit": round(rdkit_mw, 1),
            "MW_Diff": round(abs(rdkit_mw - mw), 1),
        })
    
    mapping_df = pd.DataFrame(rows)
    mapping_path = L3_RESULTS / "tcmsp_smiles_fixed_v4.csv"
    mapping_df.to_csv(mapping_path, index=False)
    logger.info(f"\nSMILES 映射表: {mapping_path} ({len(mapping_df)} 条)")
    
    # 失败列表
    if failed:
        pd.DataFrame(failed, columns=["name", "reason", "detail"]).to_csv(
            L3_RESULTS / "smiles_fix_failed_v4.csv", index=False
        )
    
    logger.info("=" * 70)
    logger.info("SMILES 修复完成!")
    return smiles_map, mapping_df


if __name__ == "__main__":
    main()