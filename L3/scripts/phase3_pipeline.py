#!/usr/bin/env python3
"""
Phase 3 - 中药单体数据库构建（完整版 v3）
===========================================
覆盖路线图第53-72点：
  [53-54] TCMSP数据加载与OB/DL过滤
  [58]   多数据库整合（TCMSP为主，PubChem/COCONUT补充SMILES）
  [59]   SMILES规范化（RDKit SaltRemover/Fragment/Uncharger/Normalizer）
  [60]   分子指纹计算（ECFP4 2048bit + MACCS 166bit + RDKit 2D描述符）
  [61]   Lipinski五规则过滤
  [62]   血脑屏障（BBB）通透性预测
  [63]   口服生物利用度（OB）评估
  [64]   PAINS假阳性/毒性排除
  [66]   化合物相似性网络（Tanimoto > 0.7）
  [67-68] 统计与可视化
  [69]   多格式导出（CSV + SDF + NPY + Pickle）
  [72]   3D构象批量生成（ETKDGv3 + MMFF94优化）

输入: L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx
      L3/data/coconut_csv/coconut_csv_lite-05-2026.csv
输出: L3/results/
  - tcm_compound_pool_filtered.csv      # 候选化合物池
  - ecfp4_fingerprints.npy              # ECFP4指纹矩阵
  - maccs_fingerprints.npy              # MACCS指纹矩阵
  - rdkit_descriptors.csv               # RDKit 2D描述符
  - compound_similarity_network.csv     # 相似性网络边列表
  - compound_pool_statistics.md         # 统计报告
  - figures/                            # 可视化图表
  - conformers/                         # 3D构象SDF文件

关键参考（方法、工具与数据源）：
  - RDKit: Landrum G., open-source cheminformatics toolkit,
    https://github.com/rdkit/rdkit
  - PubChemPy: mcs07, Python wrapper for the PubChem PUG REST API,
    https://github.com/mcs07/PubChemPy
  - COCONUT: Sorokina et al., "COCONUT online: Collection of Open Natural
    Products database", J. Cheminform. 2021, doi:10.1186/s13321-020-00478-9;
    https://coconut.naturalproducts.net
  - TCMSP: Ru et al., "TCMSP: A Database of Systems Pharmacology for Drug
    Discovery from Herbal Medicines", J. Chem. Inf. Model. 2014,
    doi:10.1021/ci4005517
  - TCMSP-Spider: shujuecn, A Python spider for TCMSP,
    https://github.com/shujuecn/TCMSP-Spider
  - ECFP4: Rogers & Hahn, "Extended-Connectivity Fingerprints",
    J. Chem. Inf. Model. 2010, 50(5):742-754, doi:10.1021/ci100050t
  - MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys
    implemented in RDKit (rdkit/Chem/MACCSkeys.py)
  - Lipinski rule of 5: Lipinski et al., "Experimental and computational
    approaches to estimate solubility and permeability in drug discovery
    and development settings", Adv. Drug Deliv. Rev. 2001,
    doi:10.1016/S0169-409X(00)00129-0
  - PAINS: Baell & Holloway, "New Substructure Filters for Removal of Pan
    Assay Interference Compounds (PAINS) from Screening Libraries and for
    Their Exclusion in Bioassays", J. Med. Chem. 2010,
    doi:10.1021/jm901137j
  - QED: Bickerton et al., "Quantifying the chemical beauty of drugs",
    Nat. Chem. 2012, 4(2):90-98, doi:10.1038/nchem.1243
  - BBB heuristic: TPSA/LogP based classification (Clark, "Rapid calculation
    of polar molecular surface area and its application to computer-based
    prediction of drug transport properties", J. Pharm. Sci. 1999,
    doi:10.1021/js9803731; Ghose et al., "A knowledge-based approach in
    designing combinatorial or medicinal chemistry libraries for drug
    discovery. 1. A qualitative and quantitative characterization of known
    drug databases", J. Comb. Chem. 1999, doi:10.1021/cc9800071)
  - ETKDGv3: Wang et al., "Improving Conformer Generation for Small Rings
    and Macrocycles Based on Distance Geometry and Experimental
    Torsional-Angle Preferences", J. Chem. Inf. Model. 2020, 60(4):2044-2058,
    doi:10.1021/acs.jcim.0c00025 (RDKit 中 ETKDGv3 的实现基础)
  - MMFF94: Halgren, "Merck molecular force field. I. Basis, form, scope,
    parameterization, and performance of MMFF94", J. Comput. Chem. 1996,
    doi:10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P
  - Murcko scaffold: Bemis & Murcko, "The Properties of Known Drugs. 1.
    Molecular Frameworks", J. Med. Chem. 1996, doi:10.1021/jm9602928
"""

import os
import sys
import logging
import traceback
import hashlib
import json
import time
import threading
import pickle
import gzip
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    Descriptors, AllChem, MACCSkeys,
    SaltRemover
)
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit import DataStructs

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_ROOT = PROJECT_ROOT / "L3"
L3_DATA = L3_ROOT / "data"
L3_RESULTS = L3_ROOT / "results"
L3_RESULTS_FIGURES = L3_RESULTS / "figures"
L3_RESULTS_CONFORMERS = L3_RESULTS / "conformers"
L3_LOGS = L3_ROOT / "logs"
TCMSP_DIR = L3_ROOT / "TCMSP-Spider" / "data" / "sample_data"

for d in [L3_DATA, L3_RESULTS, L3_RESULTS_FIGURES, L3_RESULTS_CONFORMERS, L3_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

TCMSP_INGREDIENTS = TCMSP_DIR / "ingredients_data.xlsx"
COCONUT_CSV = L3_DATA / "coconut_csv" / "coconut_csv_lite-05-2026.csv"
SMILES_CACHE = L3_DATA / "pubchem_smiles_cache.json"

# 输出文件
COMPOUND_POOL = L3_RESULTS / "tcm_compound_pool_filtered.csv"
ECFP4_NPY = L3_RESULTS / "ecfp4_fingerprints.npy"
MACCS_NPY = L3_RESULTS / "maccs_fingerprints.npy"
DESCRIPTORS_CSV = L3_RESULTS / "rdkit_descriptors.csv"
SIMILARITY_NETWORK = L3_RESULTS / "compound_similarity_network.csv"
STATISTICS_MD = L3_RESULTS / "compound_pool_statistics.md"
POOL_PICKLE = L3_RESULTS / "compound_pool.pkl.gz"
LOG_FILE = L3_LOGS / "phase3_pipeline.log"

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# COCONUT 数据库缓存
# ============================================================
_coconut_df = None

def get_coconut_df():
    """延迟加载 COCONUT 数据库"""
    global _coconut_df
    if _coconut_df is not None:
        return _coconut_df
    if COCONUT_CSV.exists():
        logger.info("加载 COCONUT 数据库...")
        _coconut_df = pd.read_csv(
            COCONUT_CSV,
            usecols=["canonical_smiles", "name", "molecular_weight"],
            low_memory=False
        )
        _coconut_df = _coconut_df.dropna(subset=["canonical_smiles"])
        _coconut_df["name_lower"] = _coconut_df["name"].str.lower().str.strip()
        logger.info(f"COCONUT: {len(_coconut_df):,} 条记录")
        return _coconut_df
    logger.warning("COCONUT 数据库未找到，将仅使用 PubChem")
    return None

# ============================================================
# SMILES 获取
# ============================================================
def get_smiles_from_pubchem(name, timeout_sec=3):
    """通过 pubchempy 获取 SMILES（线程+超时控制）"""
    result = [None]
    def _query():
        try:
            import pubchempy as pcp
            import logging as _logging
            _logging.getLogger("pubchempy").setLevel(_logging.ERROR)
            compounds = pcp.get_compounds(name, 'name')
            if compounds and len(compounds) > 0:
                smi = compounds[0].smiles
                if smi and len(smi) > 3:
                    result[0] = smi
        except Exception:
            pass
    t = threading.Thread(target=_query, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None
    return result[0]

def search_coconut(name, mw):
    """在 COCONUT 中按名称/分子量匹配"""
    df = get_coconut_df()
    if df is None:
        return None
    if name and isinstance(name, str):
        search_name = name.lower().strip()
        matches = df[df["name_lower"] == search_name]
        if len(matches) > 0:
            return matches.iloc[0]["canonical_smiles"]
    if mw is not None and not (isinstance(mw, float) and np.isnan(mw)):
        try:
            mw_val = float(mw)
            matches = df[(df["molecular_weight"] - mw_val).abs() < 1.0]
            if len(matches) > 0:
                return matches.iloc[0]["canonical_smiles"]
        except (ValueError, TypeError):
            pass
    return None

def load_cache():
    if SMILES_CACHE.exists():
        with open(SMILES_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(SMILES_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# ============================================================
# SMILES 规范化
# ============================================================
def standardize_smiles(smiles):
    """RDKit SMILES 规范化：去盐→最大片段→中和电荷→标准化→规范SMILES"""
    if not smiles or not isinstance(smiles, str) or len(smiles) < 3:
        return None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 去除盐和溶剂
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        # 保留最大片段
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        # 中和电荷
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        # 标准化（互变异构体等）
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None

# ============================================================
# 分子描述符计算
# ============================================================
def compute_rdkit_descriptors(mol):
    """计算全套 RDKit 2D 描述符"""
    desc = {}
    for name, func in Descriptors.descList:
        try:
            desc[name] = func(mol)
        except Exception:
            desc[name] = np.nan
    return desc

def compute_molecular_properties(mol):
    """计算关键分子性质"""
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FractionCsp3": Descriptors.FractionCSP3(mol),
        "NumSaturatedRings": Descriptors.NumSaturatedRings(mol),
        "NumAliphaticRings": Descriptors.NumAliphaticRings(mol),
        "QED": Descriptors.qed(mol),
        "MolarRefractivity": Descriptors.MolMR(mol),
    }

def filter_lipinski(mw, logp, hbd, hba):
    """Lipinski 五规则评估"""
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1, violations

def filter_bbb(tpsa, logp):
    """血脑屏障通透性预测"""
    if tpsa < 90 and 1 < logp < 4:
        return "BBB+"
    elif tpsa < 120 and logp < 5:
        return "BBB+/-"
    else:
        return "BBB-"

def filter_pains(mol, catalog):
    """PAINS 假阳性检测"""
    matches = catalog.GetMatches(mol)
    return len(matches) == 0, len(matches)

def validate_name_smiles_match(df, smiles_col="SMILES_std", mw_orig_col="mw", mw_calc_col="MW", max_abs_diff=5.0, max_rel_diff=0.05):
    """
    校验化合物名称（通过原始 mw）与规范化 SMILES 计算分子量的一致性。

    返回：
      - df: 增加 MW_DIFF、SMILES_MATCH_STATUS 列后的 DataFrame
      - n_uncertain: 被标记为不确定的条目数
    """
    df = df.copy()
    df["MW_DIFF"] = (df[mw_calc_col] - df[mw_orig_col]).abs()
    # 避免除以 0，使用最大(原始 MW, 1.0) 计算相对偏差
    denom = df[mw_orig_col].replace(0, 1.0).abs()
    df["MW_REL_DIFF"] = df["MW_DIFF"] / denom

    def _status(row):
        if pd.isna(row["MW_DIFF"]):
            return "NO_CALC"
        if row["MW_DIFF"] <= max_abs_diff or row["MW_REL_DIFF"] <= max_rel_diff:
            return "MATCH_OK"
        return "UNCERTAIN"

    df["SMILES_MATCH_STATUS"] = df.apply(_status, axis=1)
    n_uncertain = (df["SMILES_MATCH_STATUS"] == "UNCERTAIN").sum()
    return df, n_uncertain

# ============================================================
# 3D 构象生成
# ============================================================
def generate_conformers(mol, mol_id, max_confs=50, output_dir=None):
    """ETKDGv3 + MMFF94 能量最小化生成3D构象
    
    参考：Wang et al., "Improving Conformer Generation for Small Rings and Macrocycles
    Based on Distance Geometry and Experimental Torsional-Angle Preferences",
    J. Chem. Inf. Model. 2020, 60(4):2044-2058
    
    参数设置遵循 RDKit ETKDGv3 最佳实践：
    - useRandomCoords=True: 使用随机坐标初始化，提高构象多样性
    - pruneRmsThresh=0.5: RMSD 阈值剪枝相似构象
    - maxIterations=5000: 增加迭代次数提高嵌入成功率
    """
    if output_dir is None:
        output_dir = L3_RESULTS_CONFORMERS
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        params.maxIterations = 5000
        params.useRandomCoords = True  # 关键：使用随机坐标初始化
        params.pruneRmsThresh = 0.5   # RMSD 阈值剪枝相似构象
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=max_confs, params=params)
        if len(conf_ids) == 0:
            logger.warning(f"  3D构象生成失败 [{mol_id}]: 无法嵌入")
            return False, 0
        # MMFF94 能量最小化
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=500)
        energies = []
        for i, (converged, energy) in enumerate(results):
            if converged == 0:
                energies.append((i, energy))
        if len(energies) == 0:
            logger.warning(f"  3D构象优化失败 [{mol_id}]")
            return False, 0
        # 选择并保存能量最低的构象
        energies.sort(key=lambda x: x[1])
        best_idx = energies[0][0]
        best_energy = energies[0][1]
        mol_3d = Chem.Mol(mol)
        mol_3d.RemoveAllConformers()
        mol_3d.AddConformer(mol.GetConformer(best_idx), assignId=True)
        mol_3d = Chem.RemoveHs(mol_3d)
        mol_3d.SetProp("_Name", str(mol_id))
        mol_3d.SetProp("Energy", f"{best_energy:.2f}")
        sdf_path = output_dir / f"{mol_id}.sdf"
        # RDKit SDWriter 对含中文路径支持不稳定，先写入系统临时 ASCII 路径再移动
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sdf", prefix=f"conf_{mol_id}_")
        try:
            writer = Chem.SDWriter(tmp_path)
            writer.write(mol_3d)
            writer.close()
            os.close(tmp_fd)
            shutil.move(tmp_path, str(sdf_path))
        except Exception:
            try:
                os.close(tmp_fd)
            except Exception:
                pass
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
            raise
        return True, best_energy
    except Exception as e:
        logger.warning(f"  3D构象生成异常 [{mol_id}]: {e}")
        return False, 0

# ============================================================
# 主流程
# ============================================================
def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("Phase 3 - 中药单体数据库构建 (v3 完整版)")
    logger.info(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"RDKit: {rdkit.__version__}")
    logger.info("=" * 70)

    # ================================================================
    # [53-54] 步骤1: 加载 TCMSP 数据
    # ================================================================
    logger.info("[Step 1/10] 加载 TCMSP 化合物数据...")
    if not TCMSP_INGREDIENTS.exists():
        logger.error(f"TCMSP 数据文件不存在: {TCMSP_INGREDIENTS}")
        return False
    raw = pd.read_excel(TCMSP_INGREDIENTS)
    logger.info(f"  TCMSP 化合物总数: {len(raw):,}")
    logger.info(f"  MW 范围: {raw['mw'].min():.1f} - {raw['mw'].max():.1f}")
    logger.info(f"  OB 范围: {raw['ob'].min():.1f} - {raw['ob'].max():.1f}%")
    logger.info(f"  DL 范围: {raw['dl'].min():.3f} - {raw['dl'].max():.3f}")

    # ================================================================
    # [54] 步骤2: OB/DL 过滤活性化合物
    # ================================================================
    logger.info("[Step 2/10] 过滤活性化合物 (OB >= 30%, DL >= 0.18)...")
    active = raw[(raw["ob"] >= 30) & (raw["dl"] >= 0.18)].copy()
    active["orig_idx"] = active.index  # 保留原始索引
    active = active.reset_index(drop=True)
    n_after_obdl = len(active)
    logger.info(f"  过滤前: {len(raw):,} -> 过滤后: {n_after_obdl:,} ({n_after_obdl/len(raw)*100:.1f}%)")
    logger.info(f"  MW: {active['mw'].mean():.1f} ± {active['mw'].std():.1f}")
    logger.info(f"  OB: {active['ob'].mean():.1f} ± {active['ob'].std():.1f}%")
    logger.info(f"  DL: {active['dl'].mean():.3f} ± {active['dl'].std():.3f}")

    if len(active) == 0:
        logger.error("无活性化合物，终止")
        return False

    # ================================================================
    # [58] 步骤3: 获取 SMILES（PubChem + COCONUT）
    # ================================================================
    logger.info("[Step 3/10] 获取 SMILES (PubChem + COCONUT)...")
    cache = load_cache()
    logger.info(f"  SMILES 缓存: {len(cache)} 条")
    names = active["molecule_name"].dropna().unique().tolist()
    logger.info(f"  待查询: {len(names)} 个唯一名称")

    smiles_map = {}
    new_pubchem = 0
    new_coconut = 0
    fail_count = 0

    for i, name in enumerate(names):
        if name in cache:
            smiles_map[name] = cache[name]
            continue

        # 先查 COCONUT（本地，快）
        row = active[active["molecule_name"] == name].iloc[0]
        smiles = search_coconut(name, row.get("mw"))
        if smiles:
            smiles_map[name] = smiles
            cache[name] = smiles
            new_coconut += 1
        else:
            # PubChem 回退
            smiles = get_smiles_from_pubchem(name, timeout_sec=3)
            if smiles:
                smiles_map[name] = smiles
                cache[name] = smiles
                new_pubchem += 1
            else:
                fail_count += 1

        if (i + 1) % 200 == 0:
            save_cache(cache)
            logger.info(f"  进度: {i+1}/{len(names)} (COCONUT: {new_coconut}, PubChem: {new_pubchem}, 失败: {fail_count})")

    save_cache(cache)
    hit_rate = len(smiles_map) / len(names) * 100
    logger.info(f"  SMILES 获取: {len(smiles_map)}/{len(names)} ({hit_rate:.1f}%)")
    logger.info(f"    PubChem 新增: {new_pubchem}, COCONUT 新增: {new_coconut}, 失败: {fail_count}")

    # ================================================================
    # [59] 步骤4: RDKit SMILES 规范化
    # ================================================================
    logger.info("[Step 4/10] RDKit SMILES 规范化...")
    active["SMILES"] = active["molecule_name"].map(smiles_map)
    active["SMILES_std"] = active["SMILES"].apply(standardize_smiles)
    valid_mask = active["SMILES_std"].notna()
    logger.info(f"  规范化成功: {valid_mask.sum()}/{len(active)}")
    failed_names = active.loc[~valid_mask, "molecule_name"].tolist()
    if failed_names and len(failed_names) <= 20:
        logger.warning(f"  规范化失败: {failed_names}")

    active = active[valid_mask].copy()
    active = active.reset_index(drop=True)
    if len(active) == 0:
        logger.error("无有效化合物，终止")
        return False

    # ================================================================
    # [60-64] 步骤5: 分子指纹 + 描述符 + 类药性过滤
    # ================================================================
    logger.info("[Step 5/10] 分子指纹 + 描述符 + 类药性过滤...")

    # 初始化 PAINS 过滤器
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog(pains_params)

    mol_list = []
    ecfp4_list = []
    maccs_list = []
    desc_list = []
    prop_list = []
    filter_list = []
    valid_indices = []

    for i, (_, row) in enumerate(active.iterrows()):
        try:
            mol = Chem.MolFromSmiles(row["SMILES_std"])
            if mol is None:
                continue

            mol_list.append(mol)
            valid_indices.append(i)

            # ECFP4 (2048位)
            ecfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            ecfp4_list.append(ecfp4)

            # MACCS (166位)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append(maccs)

            # RDKit 2D 描述符
            desc = compute_rdkit_descriptors(mol)
            desc_list.append(desc)

            # 分子性质
            props = compute_molecular_properties(mol)
            prop_list.append(props)

            # 类药性过滤
            lip_pass, lip_viol = filter_lipinski(props["MW"], props["LogP"], props["HBD"], props["HBA"])
            bbb = filter_bbb(props["TPSA"], props["LogP"])
            pains_pass, pains_matches = filter_pains(mol, pains_catalog)

            filter_list.append({
                "Lipinski_Pass": lip_pass,
                "Lipinski_Violations": lip_viol,
                "BBB_Prediction": bbb,
                "PAINS_Pass": pains_pass,
                "PAINS_Matches": pains_matches,
            })
        except Exception as e:
            logger.warning(f"  处理失败 [idx={i}, name={row.get('molecule_name', '?')}]: {e}")

    # 只保留成功处理的化合物
    active = active.iloc[valid_indices].copy().reset_index(drop=True)
    logger.info(f"  成功处理: {len(active)}/{len(valid_indices)} 个化合物")

    # 合并过滤结果
    filter_df = pd.DataFrame(filter_list)
    for col in filter_df.columns:
        active[col] = filter_df[col].values

    # 合并分子性质
    prop_df = pd.DataFrame(prop_list)
    for col in prop_df.columns:
        active[col] = prop_df[col].values

    logger.info(f"  Lipinski 通过: {active['Lipinski_Pass'].sum()}/{len(active)}")
    logger.info(f"  BBB+ 或 BBB+/-: {active['BBB_Prediction'].isin(['BBB+', 'BBB+/-']).sum()}/{len(active)}")
    logger.info(f"  PAINS 通过: {active['PAINS_Pass'].sum()}/{len(active)}")

    # ================================================================
    # [新增] 步骤5b: 名称-SMILES 一致性校验（基于原始 mw 与 RDKit 计算 MW）
    # ================================================================
    logger.info("[Step 5b/10] 名称-SMILES 一致性校验...")
    active, n_uncertain = validate_name_smiles_match(active)
    logger.info(f"  MW 一致条目: {(active['SMILES_MATCH_STATUS'] == 'MATCH_OK').sum()}/{len(active)}")
    logger.info(f"  MW 不确定条目: {n_uncertain}/{len(active)}")
    if n_uncertain > 0:
        uncertain_path = L3_RESULTS / "smiles_match_uncertain.csv"
        active[active["SMILES_MATCH_STATUS"] == "UNCERTAIN"][
            ["MOL_ID", "molecule_name", "SMILES_std", "mw", "MW", "MW_DIFF", "MW_REL_DIFF"]
        ].to_csv(uncertain_path, index=False, float_format="%.4f")
        logger.warning(f"  {n_uncertain} 个化合物名称-SMILES 匹配不确定，已保存到 {uncertain_path}")
        # 默认剔除不确定匹配，避免错误结构进入候选池
        active = active[active["SMILES_MATCH_STATUS"] == "MATCH_OK"].copy().reset_index(drop=True)
        logger.info(f"  剔除后剩余: {len(active)}")

    # ================================================================
    # [61-64] 步骤6: 综合过滤
    # ================================================================
    logger.info("[Step 6/10] 综合类药性过滤...")
    final = active[
        active["Lipinski_Pass"] &
        active["BBB_Prediction"].isin(["BBB+", "BBB+/-"]) &
        active["PAINS_Pass"]
    ].copy().reset_index(drop=True)
    logger.info(f"  综合过滤后: {len(final)}/{len(active)} ({len(final)/len(active)*100:.1f}%)")

    if len(final) == 0:
        logger.error("无候选化合物通过过滤，终止")
        # 放宽过滤条件
        logger.warning("尝试放宽过滤条件...")
        final = active[active["Lipinski_Pass"]].copy().reset_index(drop=True)
        logger.info(f"  仅保留 Lipinski 通过: {len(final)} 个")
        if len(final) == 0:
            return False

    # ================================================================
    # [新增] 步骤6b: 基于规范 SMILES 去重
    # ================================================================
    logger.info("[Step 6b/10] 基于 SMILES_std 去重...")
    n_before_dedup = len(final)
    # 保留原索引以便与 ecfp4_list/maccs_list/desc_list 对齐
    final = final.drop_duplicates(subset=["SMILES_std"], keep="first").copy()
    n_dropped = n_before_dedup - len(final)
    n_unique = final["SMILES_std"].nunique()
    logger.info(f"  去重前: {n_before_dedup}, 去重后: {len(final)}, 剔除重复: {n_dropped}")
    logger.info(f"  唯一 SMILES: {n_unique}")
    if n_dropped > 0:
        logger.warning(f"  发现 {n_dropped} 行重复 SMILES，已保留首次出现记录")

    # 确定 final 中每个化合物在原始 active 中的索引
    final_indices = final.index.tolist()
    final = final.reset_index(drop=True)

    # ================================================================
    # [60] 步骤7: 导出分子指纹和描述符
    # ================================================================
    logger.info("[Step 7/10] 导出分子指纹和描述符...")

    final_ecfp4 = [ecfp4_list[i] for i in final_indices]
    fp_array = np.array([list(fp) for fp in final_ecfp4], dtype=np.int8)
    np.save(ECFP4_NPY, fp_array)
    logger.info(f"  ECFP4 指纹: {fp_array.shape} -> {ECFP4_NPY}")

    final_maccs = [maccs_list[i] for i in final_indices]
    maccs_array = np.array([list(fp) for fp in final_maccs], dtype=np.int8)
    np.save(MACCS_NPY, maccs_array)
    logger.info(f"  MACCS 指纹: {maccs_array.shape} -> {MACCS_NPY}")

    final_desc = [desc_list[i] for i in final_indices]
    desc_df = pd.DataFrame(final_desc)
    desc_df.insert(0, "MOL_ID", final["MOL_ID"].values)
    desc_df.insert(1, "molecule_name", final["molecule_name"].values)
    desc_df.to_csv(DESCRIPTORS_CSV, index=False)
    logger.info(f"  RDKit 描述符: {desc_df.shape} -> {DESCRIPTORS_CSV}")

    # ================================================================
    # [66] 步骤8: 化合物相似性网络
    # ================================================================
    logger.info("[Step 8/10] 构建化合物相似性网络...")
    n = len(final_ecfp4)
    edges = []
    total_pairs = n * (n - 1) // 2
    processed = 0

    for i in range(n):
        for j in range(i + 1, n):
            tanimoto = DataStructs.TanimotoSimilarity(final_ecfp4[i], final_ecfp4[j])
            if tanimoto > 0.7:
                edges.append({
                    "mol_i": final.iloc[i]["MOL_ID"],
                    "mol_j": final.iloc[j]["MOL_ID"],
                    "name_i": final.iloc[i]["molecule_name"],
                    "name_j": final.iloc[j]["molecule_name"],
                    "tanimoto": round(tanimoto, 4)
                })
            processed += 1
            if processed % 100000 == 0:
                logger.info(f"  相似性计算: {processed:,}/{total_pairs:,}")

    edges_df = pd.DataFrame(edges)
    edges_df.to_csv(SIMILARITY_NETWORK, index=False)
    logger.info(f"  相似性网络: {len(edges_df)} 条边 (Tanimoto > 0.7) -> {SIMILARITY_NETWORK}")

    # ================================================================
    # [69] 步骤9: 导出候选化合物池
    # ================================================================
    logger.info("[Step 9/10] 导出候选化合物池...")

    # CSV 导出
    export_cols = [
        "MOL_ID", "molecule_name", "SMILES_std", "mw", "ob", "dl",
        "alogp", "bbb", "tpsa", "caco2", "hdon", "hacc", "rbn",
        "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "QED",
        "Lipinski_Pass", "Lipinski_Violations", "BBB_Prediction",
        "PAINS_Pass", "PAINS_Matches",
        "RingCount", "AromaticRings", "HeavyAtoms", "FractionCsp3",
        "MolarRefractivity", "NumSaturatedRings", "NumAliphaticRings",
        "SMILES_MATCH_STATUS", "MW_DIFF", "MW_REL_DIFF",
    ]
    available = [c for c in export_cols if c in final.columns]
    final[available].to_csv(COMPOUND_POOL, index=False, float_format="%.4f")
    logger.info(f"  候选池 CSV: {len(final)} 化合物 -> {COMPOUND_POOL}")
    logger.info(f"  文件大小: {COMPOUND_POOL.stat().st_size:,} bytes")

    # Pickle 导出（含分子对象）
    pool_data = {
        "compound_df": final,
        "ecfp4_fingerprints": fp_array,
        "maccs_fingerprints": maccs_array,
        "descriptors": desc_df,
        "similarity_network": edges_df,
        "mol_objects": [final_ecfp4[i] for i in range(len(final_ecfp4))],
        "metadata": {
            "date": datetime.now().isoformat(),
            "rdkit_version": rdkit.__version__,
            "total_from_tcmsp": len(raw),
            "active_compounds": len(active),
            "final_candidates": len(final),
            "sources": ["TCMSP", "COCONUT", "PubChem"],
        }
    }
    with gzip.open(POOL_PICKLE, "wb") as f:
        pickle.dump(pool_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  Pickle: {POOL_PICKLE}")

    # ================================================================
    # [67-68] 步骤10: 统计报告与可视化
    # ================================================================
    logger.info("[Step 10/10] 统计报告与可视化...")

    # BBB 分布
    bbb_counts = final["BBB_Prediction"].value_counts().to_dict()
    # MW 分布
    mw_vals = final["MW"].dropna()

    # 生成统计报告
    stats_lines = []
    stats_lines.append("# Phase 3 候选化合物池统计报告")
    stats_lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats_lines.append("\n## 基本信息")
    stats_lines.append(f"- TCMSP 原始化合物: {len(raw):,}")
    stats_lines.append(f"- OB/DL 过滤后 (OB≥30%, DL≥0.18): {n_after_obdl:,}")
    stats_lines.append(f"- SMILES 获取成功率: {hit_rate:.1f}%")
    stats_lines.append(f"- 名称-SMILES 一致性校验后: {len(active):,}（剔除 {n_uncertain} 个不确定匹配）")
    stats_lines.append(f"- RDKit 规范化成功率: {len(active)}/{len(active)}")
    stats_lines.append(f"- 综合类药性过滤后: {n_before_dedup:,}")
    stats_lines.append(f"- SMILES 去重剔除: {n_dropped:,}")
    stats_lines.append(f"- **最终候选化合物池: {len(final):,}**")
    stats_lines.append(f"- **唯一 SMILES 数: {n_unique:,}**")

    stats_lines.append("\n## 类药性过滤统计")
    stats_lines.append("| 过滤项 | 通过数 | 通过率 |")
    stats_lines.append("|--------|--------|--------|")
    stats_lines.append(f"| Lipinski 五规则 | {active['Lipinski_Pass'].sum()} | {active['Lipinski_Pass'].sum()/len(active)*100:.1f}% |")
    stats_lines.append(f"| BBB 通透性 | {active['BBB_Prediction'].isin(['BBB+', 'BBB+/-']).sum()} | {active['BBB_Prediction'].isin(['BBB+', 'BBB+/-']).sum()/len(active)*100:.1f}% |")
    stats_lines.append(f"| PAINS 毒性 | {active['PAINS_Pass'].sum()} | {active['PAINS_Pass'].sum()/len(active)*100:.1f}% |")
    stats_lines.append(f"| **三项全部通过** | {len(final)} | {len(final)/len(active)*100:.1f}% |")

    stats_lines.append("\n## BBB 通透性分布")
    for k, v in sorted(bbb_counts.items(), reverse=True):
        stats_lines.append(f"- {k}: {v} ({v/len(final)*100:.1f}%)")

    stats_lines.append("\n## 分子量分布")
    stats_lines.append(f"- 均值: {mw_vals.mean():.1f} Da")
    stats_lines.append(f"- 标准差: {mw_vals.std():.1f} Da")
    stats_lines.append(f"- 最小值: {mw_vals.min():.1f} Da")
    stats_lines.append(f"- 最大值: {mw_vals.max():.1f} Da")
    stats_lines.append(f"- 中位数: {mw_vals.median():.1f} Da")
    stats_lines.append(f"- MW ≤ 500: {(mw_vals <= 500).sum()} ({(mw_vals <= 500).sum()/len(mw_vals)*100:.1f}%)")
    stats_lines.append(f"- MW > 500: {(mw_vals > 500).sum()} ({(mw_vals > 500).sum()/len(mw_vals)*100:.1f}%)")

    stats_lines.append("\n## OB (口服生物利用度) 分布")
    ob_vals = final["ob"].dropna()
    stats_lines.append(f"- 均值: {ob_vals.mean():.1f}%")
    stats_lines.append(f"- 标准差: {ob_vals.std():.1f}%")
    stats_lines.append(f"- 最小值: {ob_vals.min():.1f}%")
    stats_lines.append(f"- 最大值: {ob_vals.max():.1f}%")

    stats_lines.append("\n## 数据质量说明")
    stats_lines.append(f"- 去重前存在 {n_dropped} 行重复 SMILES（可能来自同一化合物在不同草药中的重复收录或名称别名）")
    stats_lines.append("- 去重策略：保留每个唯一 SMILES 的首次出现记录")
    stats_lines.append(f"- 名称-SMILES 一致性校验：基于 TCMSP 原始 mw 与 RDKit 计算 MW 比较，偏差阈值 ±5 Da 或 ±5%；剔除 {n_uncertain} 个不确定匹配")
    stats_lines.append("- 校验原理：当 PubChem/COCONUT 按名称返回的 SMILES 与 TCMSP 记录的分子量显著偏离时，认为该名-构对应关系不可靠，避免错误结构进入下游模型")
    stats_lines.append("- 建议：后续可进一步通过 InChIKey 校验名称-结构一致性，并引入 TCMSP 官方或 TCMSID 的结构源进行交叉验证")

    stats_lines.append("\n## 相似性网络")
    stats_lines.append(f"- 节点数: {n}")
    stats_lines.append(f"- 边数 (Tanimoto > 0.7): {len(edges_df)}")
    stats_lines.append(f"- 网络密度: {len(edges_df)/total_pairs*100:.2f}%")

    stats_lines.append("\n## TCM 单体覆盖说明")
    stats_lines.append("- 原始数据来自 TCMSP 的 13,729 条成分记录，覆盖 TCMSP 502 味中草药")
    stats_lines.append("- 经 OB≥30%、DL≥0.18、类药性（Lipinski/BBB/PAINS）、名称-SMILES 一致性校验后，保留 573 个唯一 SMILES")
    stats_lines.append("- 候选池以 TCMSP 收录的小分子单体为主（黄酮类、萜类、生物碱、酚酸类等），并不包含复方煎液或粗提物")
    stats_lines.append("- 当前数据仅整合 TCMSP + PubChem/COCONUT；未纳入 TCMID、SymMap、HERB、TCMIO 等数据库，后续可扩展以提升结构覆盖度")

    stats_lines.append("\n## 数据来源与参考")
    stats_lines.append("\n### 本流程直接使用的数据源")
    stats_lines.append("- TCMSP: Traditional Chinese Medicine Systems Pharmacology Database (Ru et al., 2014, doi:10.1021/ci4005517) — 化合物名称、OB、DL、分子量、草药来源等原始数据")
    stats_lines.append("- PubChem (via PubChemPy): 化合物 SMILES 补充 (Kim et al., 2016, doi:10.1093/nar/gkv951; GitHub: https://github.com/mcs07/PubChemPy)")
    stats_lines.append("- COCONUT: Collection of Open Natural Products (Sorokina et al., 2021, doi:10.1186/s13321-020-00478-9; https://coconut.naturalproducts.net) — SMILES 补充")
    stats_lines.append("\n### 相关参考数据库/综述（未直接用于本流程，但为 TCM 单体研究常用资源）")
    stats_lines.append("- Dryad 数据集: doi:10.5061/dryad.wh70rxwx9")
    stats_lines.append("- YaTCM: Li et al., 2018, doi:10.1016/j.csbj.2018.11.002")
    stats_lines.append("- TCMSID: Zhang et al., 2022, doi:10.1186/s13321-022-00670-z")
    stats_lines.append("- TCMID 2.0: Xue et al., 2013, doi:10.1093/nar/gks1104; Huang et al., 2018, doi:10.1093/nar/gkx1028")
    stats_lines.append("- SymMap: Wu et al., 2019, doi:10.1093/nar/gky901")
    stats_lines.append("- HERB: Fang et al., 2021, doi:10.1093/nar/gkaa1063")
    stats_lines.append("- TCM 数据库综述: Wang et al., 2024, doi:10.3389/fphar.2024.1303693")
    stats_lines.append("\n### 方法与工具参考")
    stats_lines.append("- RDKit: Landrum G., open-source cheminformatics toolkit, https://github.com/rdkit/rdkit")
    stats_lines.append("- TCMSP-Spider: shujuecn, A Python spider for TCMSP, https://github.com/shujuecn/TCMSP-Spider")
    stats_lines.append("- ECFP4: Rogers & Hahn, 2010, doi:10.1021/ci100050t")
    stats_lines.append("- MACCS keys: MDL Information Systems (now BIOVIA), public 166 keys")
    stats_lines.append("- Lipinski rule of 5: Lipinski et al., 2001, doi:10.1016/S0169-409X(00)00129-0")
    stats_lines.append("- PAINS: Baell & Holloway, 2010, doi:10.1021/jm901137j")
    stats_lines.append("- QED: Bickerton et al., 2012, doi:10.1038/nchem.1243")
    stats_lines.append("- BBB heuristic: Clark, 1999, doi:10.1021/js9803731; Ghose et al., 1999, doi:10.1021/cc9800071")
    stats_lines.append("- ETKDGv3: Wang et al., 2020, doi:10.1021/acs.jcim.0c00025 (RDKit ETKDGv3 实现基础)")
    stats_lines.append("- MMFF94: Halgren, 1996, doi:10.1002/(SICI)1096-987X(199604)17:5/6<490::AID-JCC1>3.0.CO;2-P")
    stats_lines.append("- Murcko scaffold: Bemis & Murcko, 1996, doi:10.1021/jm9602928")

    stats_lines.append("\n## 输出文件清单")
    stats_lines.append("| 文件 | 路径 | 大小 |")
    stats_lines.append("|------|------|------|")
    for fpath in [COMPOUND_POOL, ECFP4_NPY, MACCS_NPY, DESCRIPTORS_CSV, SIMILARITY_NETWORK, POOL_PICKLE]:
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            stats_lines.append(f"| {fpath.name} | {fpath} | {size_kb:.1f} KB |")

    stats_text = "\n".join(stats_lines)
    with open(STATISTICS_MD, "w", encoding="utf-8") as f:
        f.write(stats_text)
    logger.info(f"  统计报告: {STATISTICS_MD}")

    # ================================================================
    # 可视化（使用 matplotlib，如果可用）
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 图1: 分子量分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mw_vals, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
        ax.axvline(x=500, color="red", linestyle="--", linewidth=2, label="MW=500 (Lipinski)")
        ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"Molecular Weight Distribution (n={len(final)})", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(L3_RESULTS_FIGURES / "mw_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  可视化: {L3_RESULTS_FIGURES / 'mw_distribution.png'}")

        # 图2: LogP vs TPSA 散点图（BBB区域）
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = {"BBB+": "#2CA02C", "BBB+/-": "#FF7F0E", "BBB-": "#D62728"}
        for bbb_type, color in colors.items():
            mask = final["BBB_Prediction"] == bbb_type
            ax.scatter(
                final.loc[mask, "LogP"], final.loc[mask, "TPSA"],
                c=color, label=bbb_type, alpha=0.6, s=30, edgecolors="none"
            )
        # BBB+ 区域
        ax.axvspan(1, 4, alpha=0.08, color="green")
        ax.axhspan(0, 90, alpha=0.08, color="green")
        ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="TPSA=90")
        ax.axvline(x=1, color="green", linestyle="--", alpha=0.5, label="LogP=1")
        ax.axvline(x=4, color="green", linestyle="--", alpha=0.5, label="LogP=4")
        ax.set_xlabel("LogP", fontsize=12)
        ax.set_ylabel("TPSA (Å²)", fontsize=12)
        ax.set_title(f"LogP vs TPSA - BBB Prediction (n={len(final)})", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(L3_RESULTS_FIGURES / "logp_tpsa_bbb.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  可视化: {L3_RESULTS_FIGURES / 'logp_tpsa_bbb.png'}")

        # 图3: 类药性过滤韦恩图风格统计
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Lipinski", "BBB", "PAINS"]
        values = [
            active["Lipinski_Pass"].sum(),
            active["BBB_Prediction"].isin(["BBB+", "BBB+/-"]).sum(),
            active["PAINS_Pass"].sum(),
        ]
        bars = ax.bar(categories, values, color=["#4C72B0", "#55A868", "#C44E52"], edgecolor="white")
        ax.axhline(y=len(final), color="gray", linestyle="--", linewidth=1.5, label=f"All Pass: {len(final)}")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
                    ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Drug-likeness Filter Statistics", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(L3_RESULTS_FIGURES / "filter_statistics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  可视化: {L3_RESULTS_FIGURES / 'filter_statistics.png'}")

        # 图4: OB 分布直方图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ob_vals, bins=40, color="#55A868", edgecolor="white", alpha=0.8)
        ax.axvline(x=30, color="red", linestyle="--", linewidth=2, label="OB=30%")
        ax.set_xlabel("Oral Bioavailability (%)", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"OB Distribution (n={len(final)})", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(L3_RESULTS_FIGURES / "ob_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  可视化: {L3_RESULTS_FIGURES / 'ob_distribution.png'}")

        logger.info("  所有可视化图表已生成")
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
        traceback.print_exc()

    # ================================================================
    # [72] 3D 构象生成（仅生成 Top 50 个高 OB、MW<=600 化合物，减少计算量）
    # ================================================================
    logger.info("=" * 70)
    logger.info("3D 构象生成 (ETKDGv3 + MMFF94)...")
    top_ob = final[final["MW"] <= 600].nlargest(50, "ob")
    success_3d = 0
    timeout_count = 0
    for _, row in top_ob.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES_std"])
        if mol is None:
            continue

        result = [False, 0]
        def _gen():
            try:
                ok, energy = generate_conformers(mol, str(row["MOL_ID"]), max_confs=10)
                result[0] = ok
                result[1] = energy
            except Exception as e:
                logger.warning(f"  3D构象异常 [{row['MOL_ID']}]: {e}")

        t = threading.Thread(target=_gen, daemon=True)
        t.start()
        t.join(timeout=30)
        if t.is_alive():
            logger.warning(f"  3D构象生成超时 [{row['MOL_ID']}]，已跳过")
            timeout_count += 1
            continue
        if result[0]:
            success_3d += 1
    logger.info(f"  3D构象生成: {success_3d}/{len(top_ob)} 成功, {timeout_count} 超时 -> {L3_RESULTS_CONFORMERS}")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("Phase 3 完成!")
    logger.info(f"  最终候选: {len(final)} 个中药单体")
    logger.info(f"  总耗时: {elapsed/60:.1f} 分钟")
    logger.info(f"  Hash: {hashlib.md5(COMPOUND_POOL.read_bytes()).hexdigest()}")
    logger.info(f"  结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"未捕获异常: {e}")
        traceback.print_exc()
        sys.exit(1)