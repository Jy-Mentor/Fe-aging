#!/usr/bin/env python3
"""
毒性过滤器模块 — 基于多数据库联合的致癌物/毒性化合物剔除
===========================================================
多层过滤策略：
  Layer 1 — 致癌性特异结构警报 (20 种 SMARTS，覆盖基因毒性/非基因毒性/致突变性)
  Layer 2 — 已知致癌物 InChIKey 精确匹配 (IARC Group 1/2A/2B + CPDB + NTP)
  Layer 3 — 已知致癌物 SMILES 标准化后精确匹配

匹配方式：InChIKey (优先) → canonical SMILES (次选) → CAS 号 (最后)
剔除策略：任一阳性即剔除（保守策略）

关键参考文献：
  - Brenk R. et al. (2008) "Lessons Learnt from Assembling Screening Libraries
    for Drug Discovery for Neglected Diseases", ChemMedChem 3:435-444.
    https://doi.org/10.1002/cmdc.200700139
  - RDKit FilterCatalog: https://github.com/rdkit/rdkit
  - IARC Monographs: https://monographs.iarc.who.int/list-of-classifications/
  - CPDB: Gold L.S. et al. Environmental Health Perspectives 79:259-272 (1989)
    https://ftp.nlm.nih.gov/projects/SISFTP/CPDB/

输出：
  L3/data/toxicity_exclusion_log.csv     — 剔除日志
  L3/data/toxicity_filter_stats.json     — 过滤统计
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_DATA = PROJECT_ROOT / "L3" / "data"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L3_LOGS = PROJECT_ROOT / "L3" / "logs"

for d in [L3_DATA, L3_RESULTS, L3_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            L3_LOGS / "toxicity_filter.log", encoding="utf-8", mode="w"
        ),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. 致癌物数据库 — 已知天然产物致癌物 (手动整理)
# ============================================================
# 数据来源：
#   - IARC Monographs Volumes 1-136
#   - CPDB (Carcinogenic Potency Database)
#   - 文献报道的天然产物致癌物
#
# IARC 分类：
#   Group 1: 对人类致癌 (116 agents)
#   Group 2A: 很可能对人类致癌 (71 agents)
#
# 以下仅列出有明确化学结构且可能在植物中存在的化合物

KNOWN_CARCINOGENS_SMILES: Dict[str, Dict[str, str]] = {
    # === IARC Group 1 (对人类致癌) ===
    "aristolochic_acid_I": {
        "smiles": "COC1=C(C=C2C(=C1)C(=O)OC3=C2C(=CC(=C3)[N+](=O)[O-])C(=O)O)OC",
        "name": "Aristolochic acid I",
        "cas": "313-67-7",
        "source": "IARC Group 1",
        "reason": "IARC Group 1 carcinogen — aristolochic acid",
    },
    "safrole": {
        "smiles": "C=CCC1=CC2=C(C=C1)OCO2",
        "name": "Safrole",
        "cas": "94-59-7",
        "source": "IARC Group 2B",
        "reason": "IARC Group 2B — hepatocarcinogen in rodents",
    },
    "estragole": {
        "smiles": "COC1=CC=C(C=C1)CC=C",
        "name": "Estragole",
        "cas": "140-67-0",
        "source": "IARC Group 2B",
        "reason": "IARC Group 2B — hepatocarcinogen (alkenylbenzene)",
    },
    "methyleugenol": {
        "smiles": "COC1=C(C=C(C=C1)CC=C)OC",
        "name": "Methyleugenol",
        "cas": "93-15-2",
        "source": "IARC Group 2B",
        "reason": "IARC Group 2B — hepatocarcinogen (alkenylbenzene)",
    },
    "beta_asarone": {
        "smiles": "COC1=CC(=C(C=C1C=CC)OC)OC",
        "name": "beta-Asarone",
        "cas": "5273-86-9",
        "source": "IARC Group 2B",
        "reason": "IARC (possibly carcinogenic) — hepatocarcinogen",
    },
    "bracken_fern_ptaquiloside": {
        "smiles": "CC1(CC2=C(C1=O)CC3C(C2)(C3(C)C)O)C",
        "name": "Ptaquiloside",
        "cas": "87625-62-5",
        "source": "IARC Group 2B",
        "reason": "IARC Group 2B — bracken fern carcinogen",
    },
    # === IARC Group 2A (很可能对人类致癌) ===
    "acrylamide": {
        "smiles": "C=CC(=O)N",
        "name": "Acrylamide",
        "cas": "79-06-1",
        "source": "IARC Group 2A",
        "reason": "IARC Group 2A — probably carcinogenic to humans",
    },
    # === CPDB 阳性 (啮齿类动物致癌物) ===
    "coumarin": {
        "smiles": "C1=CC=C2C(=C1)C=CC(=O)O2",
        "name": "Coumarin",
        "cas": "91-64-5",
        "source": "CPDB / NTP",
        "reason": "NTP: clear evidence of carcinogenicity in rodent bioassays",
    },
    "pulegone": {
        "smiles": "CC1CCC(=C(C)C)C(=O)C1",
        "name": "Pulegone",
        "cas": "89-82-7",
        "source": "CPDB / NTP",
        "reason": "NTP: hepatocarcinogen in rodent bioassays",
    },
    "aflatoxin_B1": {
        "smiles": "COC1=C2C(=C(C(=C1)OC)OC)C(=O)C3=C(C4=C(C=C3O2)OCO4)CC(=O)C",
        "name": "Aflatoxin B1",
        "cas": "1162-65-8",
        "source": "IARC Group 1",
        "reason": "IARC Group 1 — potent hepatocarcinogen",
    },
    "psoralen": {
        "smiles": "O=C1C=CC2=CC3=C(C=C2O1)OC=C3",
        "name": "Psoralen",
        "cas": "66-97-7",
        "source": "IARC Group 1 (with UV)",
        "reason": "IARC Group 1 with UV — photomutagenic furocoumarin",
    },
    "bergapten": {
        "smiles": "COC1=C2C=CC(=O)OC2=CC3=C1C=CO3",
        "name": "Bergapten (5-methoxypsoralen)",
        "cas": "484-20-8",
        "source": "IARC Group 1 (with UV)",
        "reason": "IARC Group 1 with UV — photomutagenic furocoumarin",
    },
    "cycasin": {
        "smiles": "C[N+](=N[O-])CO[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
        "name": "Cycasin (methylazoxymethanol glucoside)",
        "cas": "14901-08-7",
        "source": "IARC Group 2B",
        "reason": "IARC Group 2B — azoxy glycoside, hepatocarcinogen / neurocarcinogen",
    },
    "riddelliine": {
        "smiles": "CC=C1CC(=C)C(C(=O)OCC2=CCN3C2C(CC3)OC1=O)(CO)O",
        "name": "Riddelliine",
        "cas": "23246-96-0",
        "source": "NTP / IARC Group 2B",
        "reason": "NTP: clear evidence — pyrrolizidine alkaloid carcinogen",
    },

    # === 吡咯里西啶生物碱类 (Pyrrolizidine Alkaloids) — 已知肝致癌物 ===
    # 注意：仅列出 1,2-不饱和 PA 母体。N-氧化物（如 indicine N-oxide）本身无毒，
    # 但可在体内还原为毒性母核，属前致癌物。当前保守策略下，N-氧化物通过 Layer 1
    # 的 pyrrolizidine_unsat SMARTS 仍会被匹配（因为母核含不饱和双键），无需单独列出。
    "senkirkine": {
        "smiles": "CC1C(C(=O)OC2CCN3C2C(=CC3)COC(=O)C(=CC)C)COC1=O",
        "name": "Senkirkine",
        "cas": "2318-18-5",
        "source": "Literature / IARC",
        "reason": "Pyrrolizidine alkaloid — hepatocarcinogen (veno-occlusive disease)",
    },
    "senecionine": {
        "smiles": "CC=C1CC(C(C(=O)OCC2=CCN3C2C(CC3)OC1=O)(C)O)C",
        "name": "Senecionine",
        "cas": "130-01-8",
        "source": "Literature / IARC",
        "reason": "Pyrrolizidine alkaloid — hepatocarcinogen",
    },
    "monocrotaline": {
        "smiles": "CC1C(=O)OC2CCN3C2C(=CC3)COC(=O)C(C1(C)O)(C)O",
        "name": "Monocrotaline",
        "cas": "315-22-0",
        "source": "Literature / IARC",
        "reason": "Pyrrolizidine alkaloid — hepatocarcinogen / pulmonary toxicity",
    },
    "lasiocarpine": {
        "smiles": "CC(C)(O)C(=O)OC1CCN2C1C(=CC2)COC(=O)C(=CC)CO",
        "name": "Lasiocarpine",
        "cas": "303-34-4",
        "source": "Literature / IARC",
        "reason": "Pyrrolizidine alkaloid — hepatocarcinogen",
    },

    # === 硝基化合物类 (Nitro compounds) — 已知致突变/致癌 ===
    "nitroso_compounds": {
        "smiles": "",  # 通用标记，通过 Brenk 硝基警报匹配
        "name": "N-Nitroso compounds (general)",
        "cas": "",
        "source": "IARC Group 2A/2B",
        "reason": "N-Nitroso compounds — potent carcinogens (structural alert)",
    },
}


# ============================================================
# 2. 致癌物 InChIKey 集合 (用于快速匹配)
# ============================================================
def _compute_inchikey(smiles: str) -> Optional[str]:
    """计算 SMILES 的 InChIKey"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


# 预计算已知致癌物的 InChIKey
KNOWN_CARCINOGEN_INCHIKEYS: Dict[str, Dict] = {}
for _key, _info in KNOWN_CARCINOGENS_SMILES.items():
    smi = _info.get("smiles", "")
    if smi:
        ik = _compute_inchikey(smi)
        if ik:
            KNOWN_CARCINOGEN_INCHIKEYS[ik] = _info


# ============================================================
# 3. 致癌性/致突变性特异结构警报 (Benigni/Bossa + Ashby/Tennant)
# ============================================================
# 参考：
#   Benigni & Bossa, "Mechanisms of chemical carcinogenicity and
#     mutagenicity: a review with implications for predictive
#     toxicology", Chem. Rev. 2011, 111(4):2507-2536
#   Ashby & Tennant, "Chemical structure, Salmonella mutagenicity
#     and extent of carcinogenicity as indicators of genotoxic
#     carcinogenesis", Mutat. Res. 1988, 204(1):17-115
#   RDKit PAINS filter: Baell & Holloway, J. Med. Chem. 2010
#
# 注意：不使用 Brenk/NIH/ZINC 通用过滤器（它们针对 HTS 假阳性/反应性
# 而非致癌性，会误杀大量安全天然产物）。仅保留致癌性/致突变性证据明确
# 的结构警报。

CARCINOGENICITY_SMARTS: Dict[str, Dict[str, str]] = {
    # === 基因毒性致癌物 (Genotoxic Carcinogens) ===
    "nitro_group": {
        "smarts": "[N+](=O)[O-]",
        "desc": "Nitro group",
        "reason": "Genotoxic carcinogen — nitroreduction to reactive nitrenium ion",
    },
    "n_nitroso": {
        "smarts": "[N;!R]=O",
        "desc": "N-Nitroso group",
        "reason": "Potent genotoxic carcinogen — α-hydroxylation → DNA alkylation",
    },
    "aromatic_amine": {
        "smarts": "[cR][NH2]",
        "desc": "Aromatic amine",
        "reason": "Genotoxic carcinogen — N-hydroxylation to reactive nitrenium",
    },
    "epoxide": {
        "smarts": "C1OC1",
        "desc": "Epoxide",
        "reason": "DNA alkylating agent — direct-acting mutagen/carcinogen",
    },
    "hydrazine": {
        "smarts": "[NH2]-[NH2,N]",
        "desc": "Hydrazine group",
        "reason": "Genotoxic carcinogen — metabolic activation → DNA methylation",
    },
    "aziridine": {
        "smarts": "C1NC1",
        "desc": "Aziridine",
        "reason": "DNA alkylating agent — direct-acting mutagen/carcinogen",
    },
    "nitrosamine": {
        "smarts": "N-[N;X3]=O",
        "desc": "N-Nitrosamine",
        "reason": "Potent genotoxic carcinogen (IARC Group 2A/2B)",
    },
    # === 非基因毒性致癌物 (Non-genotoxic Carcinogens) ===
    "pah_3ring": {
        "smarts": "c1ccc2c(c1)ccc3ccccc23",
        "desc": "Polycyclic aromatic hydrocarbon (>=3 fused rings)",
        "reason": "PAH — metabolic epoxidation → DNA adducts (IARC Group 1/2A/2B)",
    },
    "pah_bay_region": {
        "smarts": "c1cc2c(cc1)-c1c(cccc1)-c1ccccc21",
        "desc": "PAH with bay region (>=4 fused rings)",
        "reason": "PAH bay region — potent genotoxic carcinogen (IARC Group 1)",
    },
    "aflatoxin_like": {
        "smarts": "O=c1c2cocc2c3occc3c1",
        "desc": "Aflatoxin-like furocoumarin",
        "reason": "Aflatoxin analog — potent hepatocarcinogen (IARC Group 1)",
    },
    "alkyl_nitrosourea": {
        "smarts": "N=C(O)N(N=O)",
        "desc": "Alkyl nitrosourea",
        "reason": "DNA alkylating carcinogen — spontaneous decomposition",
    },
    # === 致突变性结构警报 (Mutagenicity Alerts) ===
    "alkyl_halide_primary": {
        "smarts": "[Cl,Br,I]-[CH2]-[#6]",
        "desc": "Primary alkyl halide (aliphatic)",
        "reason": "Potential alkylating agent — direct mutagenicity risk",
    },
    "alkyl_halide_secondary": {
        "smarts": "[Cl,Br,I]-[CH1](-[#6])-[#6]",
        "desc": "Secondary alkyl halide (aliphatic)",
        "reason": "Potential alkylating agent — mutagenicity risk",
    },
    "sulfonate_ester": {
        "smarts": "S(=O)(=O)O[CH3,CH2]",
        "desc": "Sulfonate ester",
        "reason": "Alkylating agent — genotoxic impurity (ICH M7 Class 1)",
    },
    "mustard": {
        "smarts": "[Cl,Br]-[CH2]-[CH2]-[N,S]",
        "desc": "Nitrogen/sulfur mustard",
        "reason": "DNA cross-linking alkylating agent — potent carcinogen",
    },
    "quinone": {
        "smarts": "O=C1C=CC(=O)C=C1",
        "desc": "Quinone",
        "reason": "Redox cycling + DNA adducts — potential carcinogen",
    },
    # === 吡咯里西啶生物碱 (Pyrrolizidine Alkaloids) — 仅1,2-不饱和型 ===
    # SMARTS `C1CCN2C1C=CC2` 要求双键(C=C)存在于含N的五元环中，
    # 因此仅匹配1,2-不饱和PA（肝毒性），不匹配饱和PA（如platyphylline）。
    # 已验证：Riddelline/Senkirkine → 匹配；饱和PA → 不匹配。
    "pyrrolizidine_unsat": {
        "smarts": "C1CCN2C1C=CC2",
        "desc": "1,2-unsaturated pyrrolizidine core",
        "reason": "Pyrrolizidine alkaloid — hepatocarcinogen (IARC Group 2B)",
    },
    # === 烯基苯类 (Alkenylbenzenes) — 肝致癌物，通用模式 ===
    # 修正：覆盖safrole(亚甲二氧基)、estragole(对甲氧基)、myristicin(3-甲氧基-4,5-亚甲二氧基)
    "alkenylbenzene": {
        "smarts": "c1ccccc1[CH2][CH]=[CH2]",
        "desc": "Alkenylbenzene core (allylbenzene)",
        "reason": "Alkenylbenzene — hepatocarcinogen via 1'-hydroxylation (IARC Group 2B)",
    },
    # === 补骨脂素类/呋喃香豆素 (Psoralens/Furocoumarins) — 光致癌性 ===
    "psoralen_core": {
        "smarts": "O=c1cc2ccoc2c2occc12",
        "desc": "Psoralen/furocoumarin core",
        "reason": "Furocoumarin — photomutagenic / photocarcinogenic (IARC Group 1 with UV)",
    },
}

# 预编译 SMARTS 模式
_carcinogen_smarts_patterns: Dict[str, Tuple[Chem.Mol, str, str]] = {}


def _get_carcinogen_smarts_patterns() -> Dict[str, Tuple[Chem.Mol, str, str]]:
    """获取预编译的致癌性 SMARTS 模式 (lazy init)"""
    global _carcinogen_smarts_patterns
    if _carcinogen_smarts_patterns:
        return _carcinogen_smarts_patterns
    for key, info in CARCINOGENICITY_SMARTS.items():
        mol = Chem.MolFromSmarts(info["smarts"])
        if mol is not None:
            _carcinogen_smarts_patterns[key] = (mol, info["desc"], info["reason"])
        else:
            logger.warning(f"无效 SMARTS [{key}]: {info['smarts']}")
    logger.info(
        f"致癌性特异结构警报: {len(_carcinogen_smarts_patterns)} 条 "
        f"(基因毒性/非基因毒性/致突变性)"
    )
    return _carcinogen_smarts_patterns


def _check_carcinogen_smarts(mol: Chem.Mol) -> List[str]:
    """使用致癌性 SMARTS 模式检查分子"""
    alerts = []
    patterns = _get_carcinogen_smarts_patterns()
    for key, (pattern, desc, reason) in patterns.items():
        if mol.HasSubstructMatch(pattern):
            alerts.append(f"Carcinogen alert: {desc} — {reason}")
    return alerts


# ============================================================
# 4. 化合物标准化
# ============================================================
def normalize_smiles(smiles: str) -> Optional[str]:
    """标准化 SMILES: 去盐 → 最大片段 → 中和电荷 → CanonSmiles

    注意: SaltRemover.StripMol() 会移除常见盐离子 (Na⁺, Cl⁻ 等)。
    对于羧酸盐类化合物，剥离去盐后变为游离酸形式，InChIKey 可能改变。
    这是已知的局限性，建议未来改用中性分子标准化流程。
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        from rdkit.Chem.SaltRemover import SaltRemover

        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)
    except Exception:
        pass
    if mol is None:
        return None
    # 取最大片段
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    # 中和电荷
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, canonical=True)


# ============================================================
# 5. 核心毒性过滤器
# ============================================================
def check_toxicity(
    smiles: str,
) -> Tuple[bool, List[str]]:
    """
    检查单个化合物是否应被剔除。

    过滤层级：
      Layer 1: 致癌性特异结构警报 (20种基因毒性/非基因毒性/致突变警报，含PAH、吡咯里西啶、烯基苯、补骨脂素)
      Layer 2: 已知致癌物 InChIKey 精确匹配 (IARC/CPDB/NTP)
      Layer 3: 已知致癌物 SMILES 标准化后精确匹配

    Args:
        smiles: 化合物 SMILES

    Returns:
        (is_toxic, reasons): 是否应剔除 + 原因列表
    """
    reasons = []
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return True, ["Invalid SMILES"]

    # Layer 1: 致癌性特异结构警报 (SMARTS 直接匹配)
    carcinogen_alerts = _check_carcinogen_smarts(mol)
    reasons.extend(carcinogen_alerts)

    # Layer 2: 已知致癌物 InChIKey 匹配
    inchikey = Chem.MolToInchiKey(mol)
    if inchikey in KNOWN_CARCINOGEN_INCHIKEYS:
        info = KNOWN_CARCINOGEN_INCHIKEYS[inchikey]
        reasons.append(
            f"Known carcinogen: {info['name']} [{info['source']}] — "
            f"{info['reason']}"
        )

    # Layer 3: 致癌物 SMILES 精确匹配 (标准化后)
    norm_smi = normalize_smiles(smiles)
    if norm_smi:
        for key, info in KNOWN_CARCINOGENS_SMILES.items():
            if info.get("smiles") and normalize_smiles(info["smiles"]) == norm_smi:
                reasons.append(
                    f"Known carcinogen (SMILES match): {info['name']} "
                    f"[{info['source']}] — {info['reason']}"
                )
                break

    return len(reasons) > 0, reasons


# ============================================================
# 6. 批量过滤
# ============================================================
def filter_compounds(
    df: pd.DataFrame,
    smiles_col: str = "SMILES_std",
    name_col: str = "molecule_name",
    mol_id_col: str = "MOL_ID",
    strict_mode: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    批量过滤化合物，剔除毒性/致癌物。

    Args:
        df: 输入化合物表
        smiles_col: SMILES 列名
        name_col: 化合物名称列名
        mol_id_col: 化合物 ID 列名
        strict_mode: True=任一阳性即剔除; False=至少2个警报才剔除

    Returns:
        (clean_df, excluded_df, stats): 过滤后数据集 + 剔除列表 + 统计信息
    """
    logger.info("=" * 60)
    logger.info(f"毒性过滤器: 输入 {len(df)} 个化合物")
    logger.info(f"剔除策略: {'严格模式 (任一阳性即剔除)' if strict_mode else '宽松模式 (至少2个警报)'}")

    # 预加载致癌性 SMARTS 模式
    _get_carcinogen_smarts_patterns()

    excluded = []
    clean_indices = []
    total_alerts = 0
    alert_types: Dict[str, int] = {}

    for idx, row in df.iterrows():
        smi = str(row.get(smiles_col, ""))
        name = str(row.get(name_col, ""))
        mol_id = str(row.get(mol_id_col, idx))

        is_toxic, reasons = check_toxicity(smi)

        if is_toxic:
            if strict_mode or len(reasons) >= 2:
                excluded.append(
                    {
                        "MOL_ID": mol_id,
                        "molecule_name": name,
                        "SMILES": smi,
                        "reasons": " | ".join(reasons),
                        "alert_count": len(reasons),
                    }
                )
                total_alerts += len(reasons)
                for r in reasons:
                    # 简化分类
                    cat = r.split(":")[0].strip() if ":" in r else r
                    alert_types[cat] = alert_types.get(cat, 0) + 1
                continue

        clean_indices.append(idx)

    clean_df = df.iloc[clean_indices].reset_index(drop=True)
    excluded_df = pd.DataFrame(excluded)

    stats = {
        "input_count": len(df),
        "excluded_count": len(excluded_df),
        "clean_count": len(clean_df),
        "exclusion_rate": round(len(excluded_df) / len(df) * 100, 2) if len(df) > 0 else 0,
        "total_alerts": total_alerts,
        "alert_types": alert_types,
        "strict_mode": strict_mode,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"剔除: {len(excluded_df)} 个 ({stats['exclusion_rate']}%)")
    logger.info(f"保留: {len(clean_df)} 个")
    logger.info(f"警报类型分布: {json.dumps(alert_types, ensure_ascii=False)}")

    return clean_df, excluded_df, stats


# ============================================================
# 7. 输出
# ============================================================
def save_filter_results(
    clean_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    stats: Dict,
    output_prefix: str = "tcm_compound_pool",
) -> Dict[str, str]:
    """保存过滤结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 剔除日志
    excl_path = L3_DATA / f"toxicity_exclusion_log_{timestamp}.csv"
    excluded_df.to_csv(excl_path, index=False, encoding="utf-8-sig")
    logger.info(f"剔除日志: {excl_path}")

    # 过滤后数据集
    clean_path = L3_RESULTS / f"{output_prefix}_tox_filtered.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")
    logger.info(f"过滤后化合物池: {clean_path}")

    # 统计信息
    stats_path = L3_DATA / f"toxicity_filter_stats_{timestamp}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"过滤统计: {stats_path}")

    return {
        "exclusion_log": str(excl_path),
        "clean_pool": str(clean_path),
        "stats": str(stats_path),
    }


# ============================================================
# 8. 代码自检
# ============================================================
def self_check(
    clean_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    original_df: pd.DataFrame,
    sample_size: int = 30,
) -> Dict:
    """
    代码自检：验证过滤结果的正确性。

    检查项:
    1. 匹配准确性 — 随机抽样人工验证
    2. 剔除比例 — 检查是否合理
    3. 数据完整性 — 确保字段未丢失
    4. 日志可追溯性
    """
    logger.info("=" * 60)
    logger.info("开始代码自检...")

    results = {}

    # 检查 1: 数据完整性
    cols_original = set(original_df.columns)
    cols_clean = set(clean_df.columns)
    missing_cols = cols_original - cols_clean
    extra_cols = cols_clean - cols_original

    results["column_integrity"] = {
        "original_cols": len(cols_original),
        "clean_cols": len(cols_clean),
        "missing_cols": list(missing_cols),
        "extra_cols": list(extra_cols),
        "passed": len(missing_cols) == 0,
    }
    if missing_cols:
        logger.warning(f"列缺失: {missing_cols}")
    else:
        logger.info("列完整性检查通过")

    # 检查 2: 数量一致性
    total_after = len(clean_df) + len(excluded_df)
    results["count_consistency"] = {
        "original": len(original_df),
        "clean + excluded": total_after,
        "matched": total_after == len(original_df),
        "passed": total_after == len(original_df),
    }
    if total_after != len(original_df):
        logger.error(f"数量不一致! 原始 {len(original_df)} ≠ 干净 {len(clean_df)} + 剔除 {len(excluded_df)} = {total_after}")
    else:
        logger.info("数量一致性检查通过")

    # 检查 3: 随机抽样验证
    sample_n = min(sample_size, len(excluded_df))
    if sample_n > 0:
        sample = excluded_df.sample(n=sample_n, random_state=42)
        results["sample_validation"] = {
            "sample_size": sample_n,
            "samples": [],
        }
        for _, row in sample.iterrows():
            results["sample_validation"]["samples"].append(
                {
                    "MOL_ID": row["MOL_ID"],
                    "name": row["molecule_name"],
                    "reasons": row["reasons"],
                    "alert_count": row["alert_count"],
                }
            )
        logger.info(f"随机抽样 {sample_n} 个剔除化合物供人工验证")
    else:
        results["sample_validation"] = {"sample_size": 0, "samples": []}

    # 检查 4: 剔除比例（仅记录，不做硬性判断）
    excl_rate = len(excluded_df) / len(original_df) * 100 if len(original_df) > 0 else 0
    results["exclusion_rate_check"] = {
        "rate": round(excl_rate, 2),
        "total_excluded": len(excluded_df),
        "total_original": len(original_df),
        "note": (
            "天然产物中低毒性化合物比例高，剔除率 <5% 属正常；"
            "若含大量PA/烯基苯类成分，剔除率可达30%以上"
        ),
    }
    logger.info(
        f"剔除比例: {excl_rate:.2f}% ({len(excluded_df)}/{len(original_df)})"
    )

    # 检查 5: NaN 和空值 (排除已知可空字段)
    known_nullable = {"ob", "dl", "bbb", "caco2", "PAINS_Matches", "MW_DIFF", "MW_REL_DIFF"}
    nan_cols = [
        c for c in clean_df.columns
        if clean_df[c].isna().any() and c not in known_nullable
    ]
    results["nan_check"] = {
        "columns_with_nan": nan_cols,
        "known_nullable_ignored": sorted(known_nullable),
        "passed": len(nan_cols) == 0,
    }
    if nan_cols:
        logger.warning(f"以下列含 NaN: {nan_cols}")
    else:
        logger.info("NaN 检查通过")

    # 检查 6: SMILES 有效性
    invalid_smiles = 0
    smiles_col_clean = "SMILES_std" if "SMILES_std" in clean_df.columns else "SMILES"
    if smiles_col_clean in clean_df.columns:
        for smi in clean_df[smiles_col_clean]:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                invalid_smiles += 1
    results["smiles_validity"] = {
        "invalid_count": invalid_smiles,
        "passed": invalid_smiles == 0,
    }
    if invalid_smiles > 0:
        logger.warning(f"{invalid_smiles} 个无效 SMILES")
    else:
        logger.info("SMILES 有效性检查通过")

    # 总体结果
    all_passed = all(
        results[k].get("passed", True)
        for k in results
        if isinstance(results[k], dict) and "passed" in results[k]
    )
    results["overall"] = "PASSED" if all_passed else "FAILED"

    logger.info(f"代码自检结果: {results['overall']}")
    if not all_passed:
        logger.error("自检未通过! 请检查上述警告。")

    return results


# ============================================================
# 9. 主函数
# ============================================================
def main():
    """主入口：加载 TCM 候选池 → 毒性过滤 → 输出 + 自检"""
    tcm_path = L3_RESULTS / "tcm_compound_pool_filtered.csv"
    if not tcm_path.exists():
        logger.error(f"TCM 候选池文件不存在: {tcm_path}")
        sys.exit(1)

    logger.info(f"加载 TCM 候选池: {tcm_path}")
    df = pd.read_csv(tcm_path)
    logger.info(f"原始化合物数: {len(df)}")

    # 执行过滤
    clean_df, excluded_df, stats = filter_compounds(
        df, strict_mode=True  # 保守策略：任一阳性即剔除
    )

    # 保存结果
    output_paths = save_filter_results(clean_df, excluded_df, stats)

    # 代码自检
    check_results = self_check(clean_df, excluded_df, df)

    # 保存自检报告
    report = {
        "filter_stats": stats,
        "self_check": check_results,
        "output_paths": output_paths,
        "known_carcinogen_count": len(KNOWN_CARCINOGEN_INCHIKEYS),
        "carcinogen_filter_count": len(_get_carcinogen_smarts_patterns()),
        "timestamp": datetime.now().isoformat(),
    }

    report_path = L3_DATA / f"toxicity_self_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"自检报告: {report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("毒性过滤器 — 执行摘要")
    print("=" * 60)
    print(f"  致癌性结构警报: {stats.get('alert_types', {}).get('Carcinogen alert', 0)} 条")
    print(f"  已知致癌物匹配: {stats.get('alert_types', {}).get('Known carcinogen', 0)} 条")
    print(f"  输入化合物: {stats['input_count']}")
    print(f"  剔除: {stats['excluded_count']} ({stats['exclusion_rate']}%)")
    print(f"  保留: {stats['clean_count']}")
    print(f"  自检结果: {check_results['overall']}")
    print("=" * 60)

    return clean_df, excluded_df, stats, check_results


if __name__ == "__main__":
    clean_df, excluded_df, stats, check_results = main()