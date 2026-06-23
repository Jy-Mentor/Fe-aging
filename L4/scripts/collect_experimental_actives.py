#!/usr/bin/env python3
"""
Phase 4 - Step 70: 实验验证活性化合物收集
============================================
数据源：
  ChEMBL v34  - REST API (IC50/Ki/Kd ≤ 10μM, confidence_score ≥ 8)
  BindingDB   - REST API getLigandsByUniprots (IC50/Ki/Kd ≤ 10μM)
  DrugBank    - UniProt REST API cross-reference (已知配体)

输出：
  L4/results/experimental_actives_summary.csv     - 按靶标统计阳性样本数
  L4/results/experimental_actives_detail.csv      - 所有活性化合物详情
  L4/results/experimental_actives_report.md       - 汇总报告
"""

import sys
import logging
import traceback
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# 抑制第三方库冗余日志
logging.getLogger("chembl_webresource_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
L4_ROOT = PROJECT_ROOT / "L4"
L4_DATA = L4_ROOT / "data"
L4_RESULTS = L4_ROOT / "results"
L4_LOGS = L4_ROOT / "logs"

for d in [L4_DATA, L4_RESULTS, L4_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = L4_LOGS / "collect_experimental_actives.log"

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
# 基因 -> UniProt 映射
# ============================================================
# Phase 1 核心靶标 (29个基因) + 铁衰老关键靶标
GENE_UNIPROT_MAP = {
    # Phase 1 RRA 核心靶标
    "EMP1":   "P54849",
    "SAT1":   "P21673",
    "TLR4":   "O00206",
    "LCN2":   "P80188",
    "EPHA4":  "P54764",
    "CXCL10": "P02778",
    "KLF6":   "Q99612",
    "SP1":    "P08047",
    "CD74":   "P04233",
    "PTGS2":  "P35354",
    "IRF1":   "P10914",
    "FBXO31": "Q5XUX0",
    "LGMN":   "Q99538",
    "IGFBP7": "Q16270",
    "IL1B":   "P01584",
    "MAPK1":  "P28482",
    "KDM6B":  "O15054",
    "PDE4B":  "Q07343",
    "RUNX3":  "Q13761",
    "CTSB":   "P07858",
    "LACTB":  "P83111",
    "LPCAT3": "Q6P1A2",
    "EGR1":   "P18146",
    "BCL6":   "P41182",
    "GMFB":   "P60983",
    "HBP1":   "O60381",
    "SOD1":   "P00441",
    "DYRK1A": "Q13627",
    # 铁衰老关键靶标 (用户指定重点关注)
    "ACSL4":  "O60488",
    "GPX4":   "P36969",
    "HMOX1":  "P09601",
    "FTH1":   "P02794",
    "FTL":    "P02792",
    "SLC7A11":"Q9UPY5",
    "TFRC":   "P02786",
    "SLC3A2": "P08195",
    "NFE2L2": "Q16236",
    "KEAP1":  "Q14145",
    "ALOX15": "P16050",
    "ALOX5":  "P09917",
    "NOX4":   "Q9NPH5",
    "VDAC2":  "P45880",
    "VDAC3":  "Q9Y277",
    "CISD1":  "Q9NZ45",
    "ACSL3":  "O95573",
    "TP53":   "P04637",
    "NFKB1":  "P19838",
    "RELA":   "Q04206",
    "STAT3":  "P40763",
    "HIF1A":  "Q16665",
    "MTOR":   "P42345",
    "ATG5":   "Q9H1Y0",
    "ATG7":   "O95352",
    "BECN1":  "Q14457",
    "SQSTM1": "Q13501",
    "MAP1LC3B":"Q9GZQ8",
    # 铁衰老基因集补充（Phase 1 原始差异表达基因）
    "ABCC1":  "P33527",
    "ACVR1B": "P36896",
    "ATF3":   "P18847",
    "ATG3":   "Q9NT62",
    "BAP1":   "Q92560",
    "BRD7":   "Q9NPI1",
    "CAVIN1": "Q6NZI2",
    "CD82":   "P27701",
    "CDO1":   "Q16878",
    "COX7A1": "P24310",
    "DPEP1":  "P16444",
    "DPP4":   "P27487",
    "DUOX1":  "Q9NRD9",
    "E2F1":   "Q01094",
    "E2F3":   "O00716",
    "EBF3":   "Q9H4W6",
    "EDN1":   "P05305",
    "EPHA2":  "P29317",
    "ERN1":   "O75460",
    "FOSL1":  "P15407",
    "HERPUD1":"Q15011",
    "HMGB1":  "P09429",
    "ICA1":   "Q92629",
    "IFNG":   "P01579",
    "IL6":    "P05231",
    "IRF7":   "Q92985",
    "IRF9":   "Q00978",
    "LIFR":   "P42702",
    "LOX":    "P28300",
    "MAP3K14":"Q99558",
    "MAPK14": "Q16539",
    "MCU":    "Q8NE86",
    "MEN1":   "O00255",
    "MPO":    "P05164",
    "NLRP3":  "Q96P20",
    "NR1D1":  "Q14995",
    "NR2F2":  "P24468",
    "NUAK2":  "Q9H093",
    "PADI4":  "Q9UM07",
    "PPP2R2B":"Q00005",
    "PRKD1":  "Q15139",
    "PTBP1":  "P26599",
    "RBM3":   "P98179",
    "S100A8": "P05109",
    "SETD7":  "Q8WTS6",
    "SLAMF8": "Q9P0V8",
    "SLC1A5": "Q15758",
    "SMARCB1":"Q12824",
    "SMURF2": "Q9HAU4",
    "SNCA":   "P37840",
    "SOCS1":  "O15524",
    "SOCS2":  "O14508",
    "SPATA2": "Q9UM82",
    "TBX2":   "Q13207",
    "TNFAIP1":"Q13829",
    "TNFAIP3":"P21580",
    "TXNIP":  "Q9H3M7",
    "WNT5A":  "P41221",
    "WWTR1":  "Q9GZV5",
    "YAP1":   "P46937",
    "ZEB1":   "P37275",
}

# 反向映射
UNIPROT_GENE_MAP = {v: k for k, v in GENE_UNIPROT_MAP.items()}

# ============================================================
# 1. ChEMBL v34 查询 (requests 直连, 更稳定)
# ============================================================
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def chembl_get_with_timeout(url, timeout=20, max_retries=3):
    """带超时和重试的 ChEMBL API GET 请求"""
    import requests
    headers = {"Accept": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning(f"  ChEMBL 429, 等待 {attempt + 1}s...")
                time.sleep(attempt + 1)
            else:
                logger.warning(f"  ChEMBL HTTP {resp.status_code}: {url[:120]}")
                return None
        except Exception as e:
            logger.warning(f"  ChEMBL 请求失败 (重试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(2)
    return None


def collect_chembl_data(gene_uniprot_map):
    """从 ChEMBL v34 收集所有靶标的活性数据"""
    logger.info("=" * 60)
    logger.info("[1/3] ChEMBL v34 数据收集")
    logger.info("=" * 60)

    all_records = []
    target_stats = {}

    for idx, (gene, uniprot_id) in enumerate(gene_uniprot_map.items(), 1):
        logger.info(f"  [{idx}/{len(gene_uniprot_map)}] 查询: {gene} ({uniprot_id})")

        # 1. 查询 target_chembl_id
        target_url = f"{CHEMBL_API_BASE}/target.json?target_components__accession={uniprot_id}&limit=20"
        target_data = chembl_get_with_timeout(target_url, timeout=10)

        if not target_data or not target_data.get("targets"):
            logger.warning("    -> ChEMBL 中未找到靶标")
            target_stats[gene] = {"uniprot": uniprot_id, "chembl_id": None, "count": 0, "status": "NOT_FOUND"}
            continue

        # 优先选 human
        selected = None
        for t in target_data["targets"]:
            if t.get("organism") == "Homo sapiens":
                selected = t
                break
        if not selected:
            selected = target_data["targets"][0]

        target_chembl_id = selected.get("target_chembl_id")
        pref_name = selected.get("pref_name", "")
        logger.info(f"    -> ChEMBL ID: {target_chembl_id}, Name: {pref_name}")

        # 2. 查询活性数据 (IC50/Ki/Kd ≤ 10μM, confidence_score ≥ 8)
        # ChEMBL API 的 __in 对 standard_type 支持不佳，分别查询三种类型再合并
        activities = []
        for standard_type in ["IC50", "Ki", "Kd"]:
            activity_url = (
                f"{CHEMBL_API_BASE}/activity.json"
                f"?target_chembl_id__exact={target_chembl_id}"
                f"&standard_type__exact={standard_type}"
                f"&standard_value__lte=10000"
                f"&confidence_score__gte=8"
                f"&limit=1000"
            )

            page_url = activity_url
            page_count = 0
            while page_url and page_count < 100:  # 最多100页 x 1000 = 10万条
                data = chembl_get_with_timeout(page_url, timeout=20)
                if not data:
                    break
                acts = data.get("activities", [])
                if not acts:
                    break
                activities.extend(acts)
                page_count += 1
                page_url = data.get("page_meta", {}).get("next")
                if page_url:
                    # ChEMBL 有时返回相对 URL, 需要补全
                    if page_url.startswith("/"):
                        page_url = f"https://www.ebi.ac.uk{page_url}"
                    elif page_url.startswith("http://"):
                        page_url = page_url.replace("http://", "https://")

        unique_compounds = set()
        for act in activities:
            mol_id = act.get("molecule_chembl_id")
            if mol_id:
                unique_compounds.add(mol_id)
            all_records.append({
                "source": "ChEMBL",
                "gene": gene,
                "uniprot_id": uniprot_id,
                "target_chembl_id": target_chembl_id,
                "target_pref_name": pref_name,
                "molecule_chembl_id": mol_id,
                "molecule_pref_name": act.get("molecule_pref_name"),
                "canonical_smiles": act.get("canonical_smiles"),
                "standard_type": act.get("standard_type"),
                "standard_value_nM": act.get("standard_value"),
                "pchembl_value": act.get("pchembl_value"),
                "confidence_score": act.get("confidence_score"),
                "assay_description": (act.get("assay_description") or "")[:200],
            })

        unique_count = len(unique_compounds)
        target_stats[gene] = {
            "uniprot": uniprot_id,
            "chembl_id": target_chembl_id,
            "count": unique_count,
            "total_measurements": len(activities),
            "status": "OK" if unique_count > 0 else "ZERO_ACTIVES"
        }
        logger.info(f"    -> {len(activities)} 条测量记录, {unique_count} 个唯一化合物")

    return all_records, target_stats


# ============================================================
# 2. BindingDB 数据收集 (REST API)
# ============================================================
BINDINGDB_API_BASE = "https://bindingdb.org/rest"


def query_bindingdb_by_uniprots(uniprot_ids, cutoff_nM=10000, batch_size=1):
    """
    使用 BindingDB REST API 批量查询 UniProt ID 的活性数据。
    cutoff_nM: 活性阈值, 默认 10 μM
    batch_size: 每批查询的 UniProt ID 数量。默认 1，避免 API 返回记录无法精确归属时造成误分配。
    """
    logger.info("=" * 60)
    logger.info("[2/3] BindingDB 数据收集 (REST API)")
    logger.info("=" * 60)

    all_records = []
    target_stats = {}

    import requests

    total = len(uniprot_ids)
    for start in range(0, total, batch_size):
        batch = uniprot_ids[start:start + batch_size]
        batch_str = ",".join(batch)
        url = f"{BINDINGDB_API_BASE}/getLigandsByUniprots?uniprot={batch_str}&cutoff={cutoff_nM}&response=application/json"

        logger.info(f"  批次 {start+1}-{min(start+batch_size, total)}/{total}: {batch_str}")

        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                logger.warning(f"    BindingDB HTTP {resp.status_code}: {url[:150]}")
                continue

            data = resp.json()
            affinities = data.get("getLindsByUniprotsResponse", {}).get("affinities", [])
            if not affinities:
                logger.info("    -> 0 条记录")
                continue

            logger.info(f"    -> {len(affinities)} 条测量记录")

            for aff in affinities:
                uniprot = aff.get("uniprot") or aff.get("query", "").split("_")[-1]
                # BindingDB 返回的 query 是 target name, 无法直接拿到 UniProt ID
                # 通过 smiles/monomerid 推断; batch_size=1 时可精确归属
                # 更精确的做法: 如果 aff 中有 uniprot 字段则使用
                if not uniprot or uniprot not in UNIPROT_GENE_MAP:
                    uniprot = batch[0]  # 回退

                gene = UNIPROT_GENE_MAP.get(uniprot, uniprot)
                target_stats.setdefault(gene, {"count": 0, "total_measurements": 0})

                monomer_id = aff.get("monomerid")
                smiles = aff.get("smile") or aff.get("smiles")
                aff_type = aff.get("affinity_type")
                aff_val = aff.get("affinity")

                all_records.append({
                    "source": "BindingDB",
                    "gene": gene,
                    "uniprot_id": uniprot,
                    "molecule_name": aff.get("compoundname") or aff.get("chemicalname"),
                    "bindingdb_monomer_id": monomer_id,
                    "canonical_smiles": smiles,
                    "standard_type": aff_type,
                    "standard_value_nM": aff_val,
                    "target_name": aff.get("query"),
                    "pmid": aff.get("pmid"),
                    "doi": aff.get("doi"),
                })

                target_stats[gene]["total_measurements"] += 1
                if monomer_id:
                    target_stats[gene]["count"] += 1

        except Exception as e:
            logger.warning(f"    BindingDB 查询失败: {e}")
            continue

    # 去重: 按 (gene, monomer_id) 统计唯一化合物数
    for gene in target_stats:
        seen = set()
        for r in all_records:
            if r["gene"] == gene and r.get("bindingdb_monomer_id"):
                seen.add(r["bindingdb_monomer_id"])
        target_stats[gene]["count"] = len(seen)

    logger.info(f"  BindingDB 完成: {len(all_records)} 条活性记录, {len(target_stats)} 个靶标")
    return all_records, target_stats


# ============================================================
# 3. DrugBank 数据收集 (UniProt cross-reference)
# ============================================================
UNIPROT_API_BASE = "https://rest.uniprot.org/uniprotkb"


def collect_drugbank_data(gene_uniprot_map):
    """
    通过 UniProt REST API 获取每个 UniProt ID 的 DrugBank 交叉引用。
    返回 DrugBank ID 数量作为已知配体估算。
    """
    logger.info("=" * 60)
    logger.info("[3/3] DrugBank 数据收集 (UniProt cross-reference)")
    logger.info("=" * 60)

    records = []
    target_stats = {}

    import requests

    for idx, (gene, uniprot_id) in enumerate(gene_uniprot_map.items(), 1):
        logger.info(f"  [{idx}/{len(gene_uniprot_map)}] 查询: {gene} ({uniprot_id})")

        url = f"{UNIPROT_API_BASE}/{uniprot_id}.json"
        try:
            resp = requests.get(url, timeout=20, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                logger.warning(f"    UniProt HTTP {resp.status_code}")
                target_stats[gene] = {"count": 0, "status": f"HTTP_{resp.status_code}"}
                continue

            data = resp.json()
            refs = data.get("uniProtKBCrossReferences", [])
            drugbank_refs = [r for r in refs if r.get("database") == "DrugBank"]

            for ref in drugbank_refs:
                db_id = ref.get("id", "")
                generic_name = ""
                for prop in ref.get("properties", []):
                    if prop.get("key") == "GenericName":
                        generic_name = prop.get("value", "")
                        break

                records.append({
                    "source": "DrugBank",
                    "gene": gene,
                    "uniprot_id": uniprot_id,
                    "drugbank_id": db_id,
                    "drug_name": generic_name,
                    "note": "UniProt cross-reference (known ligand)"
                })

            target_stats[gene] = {
                "count": len(drugbank_refs),
                "status": "OK" if len(drugbank_refs) > 0 else "NO_DATA"
            }
            logger.info(f"    -> {len(drugbank_refs)} 个 DrugBank 配体")
            time.sleep(0.3)  # 礼貌请求

        except Exception as e:
            logger.warning(f"    DrugBank 查询失败 [{gene}]: {e}")
            target_stats[gene] = {"count": 0, "status": "QUERY_FAILED"}

    logger.info(f"  DrugBank 完成: {len(records)} 条记录")
    return records, target_stats


# ============================================================
# 4. 汇总与报告
# ============================================================
def _count_unique_by_gene(records, gene_col="gene", smiles_col="canonical_smiles"):
    """按 gene 统计唯一 SMILES 数量。DrugBank 无 SMILES 时按 drugbank_id 统计。"""
    counts = {}
    for r in records:
        gene = r.get(gene_col)
        if not gene:
            continue
        smi = r.get(smiles_col)
        if not smi or pd.isna(smi):
            smi = r.get("drugbank_id")
        if not smi or pd.isna(smi):
            continue
        counts.setdefault(gene, set()).add(str(smi).strip())
    return {g: len(s) for g, s in counts.items()}


def generate_summary(chembl_records, bindingdb_records, drugbank_records,
                     chembl_stats, bindingdb_stats, drugbank_stats, gene_uniprot_map):
    """生成三源汇总统计，按 (gene, canonical_smiles) 去重统计唯一化合物数。"""
    logger.info("=" * 60)
    logger.info("汇总统计")
    logger.info("=" * 60)

    chembl_unique = _count_unique_by_gene(chembl_records)
    bindingdb_unique = _count_unique_by_gene(bindingdb_records)
    drugbank_unique = _count_unique_by_gene(drugbank_records)

    summary_rows = []
    for gene, uniprot_id in gene_uniprot_map.items():
        c = chembl_stats.get(gene, {})
        b = bindingdb_stats.get(gene, {})

        chembl_count = chembl_unique.get(gene, 0)
        bindingdb_count = bindingdb_unique.get(gene, 0)
        drugbank_count = drugbank_unique.get(gene, 0)

        # 跨源去重：合并同一基因在三源中的唯一 SMILES/drugbank_id
        merged = set()
        for rec in chembl_records + bindingdb_records + drugbank_records:
            if rec.get("gene") != gene:
                continue
            smi = rec.get("canonical_smiles")
            if not smi or pd.isna(smi):
                smi = rec.get("drugbank_id")
            if smi and not pd.isna(smi):
                merged.add(str(smi).strip())
        total = len(merged)

        summary_rows.append({
            "Gene": gene,
            "UniProt": uniprot_id,
            "Priority": "",
            "ChEMBL_Actives": chembl_count,
            "BindingDB_Actives": bindingdb_count,
            "DrugBank_Actives": drugbank_count,
            "Total_Estimate": total,
            "ChEMBL_Status": c.get("status", "?") if isinstance(c, dict) else "?",
            "BindingDB_Status": "OK" if bindingdb_count > 0 else "ZERO" if isinstance(b, dict) else "?",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["Priority", "Total_Estimate"],
                                         ascending=[False, False])

    summary_path = L4_RESULTS / "experimental_actives_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  汇总文件: {summary_path}")

    return summary_df


def generate_report(summary_df, chembl_records, bindingdb_records, drugbank_records):
    """生成 Markdown 报告"""
    lines = []
    lines.append("# Phase 4 Step 70: 实验验证活性化合物盘点报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## 数据源")
    lines.append("- **ChEMBL v34**: REST API, IC50/Ki/Kd ≤ 10μM, confidence_score ≥ 8")
    lines.append("- **BindingDB**: REST API (`/getLigandsByUniprots`), IC50/Ki/Kd ≤ 10μM")
    lines.append("- **DrugBank**: UniProt REST API cross-reference (DrugBank ID / GenericName)")

    lines.append("\n## 总体统计")
    lines.append(f"- ChEMBL 活性记录: {len(chembl_records)} 条")
    lines.append(f"- BindingDB 活性记录: {len(bindingdb_records)} 条")
    lines.append(f"- DrugBank 活性记录: {len(drugbank_records)} 条")

    # 重点关注靶标
    lines.append("\n## 重点关注靶标 (ACSL4, GPX4, HMOX1 等)")
    priority_df = summary_df[summary_df["Priority"] == "YES"]
    lines.append("| 基因 | UniProt | ChEMBL | BindingDB | DrugBank | 合计 |")
    lines.append("|------|---------|--------|-----------|----------|------|")
    for _, row in priority_df.iterrows():
        lines.append(f"| {row['Gene']} | {row['UniProt']} | {row['ChEMBL_Actives']} | {row['BindingDB_Actives']} | {row['DrugBank_Actives']} | {row['Total_Estimate']} |")

    # 所有靶标
    lines.append("\n## 全部靶标统计")
    lines.append("| 基因 | UniProt | ChEMBL | BindingDB | DrugBank | 合计 | 状态 |")
    lines.append("|------|---------|--------|-----------|----------|------|------|")
    for _, row in summary_df.iterrows():
        priority_mark = " **" if row["Priority"] == "YES" else ""
        priority_mark_end = "**" if row["Priority"] == "YES" else ""
        lines.append(f"| {priority_mark}{row['Gene']}{priority_mark_end} | {row['UniProt']} | {row['ChEMBL_Actives']} | {row['BindingDB_Actives']} | {row['DrugBank_Actives']} | {row['Total_Estimate']} | {row['ChEMBL_Status']} |")

    # 门槛分析
    lines.append("\n## 门槛分析")
    lines.append("\n### 阳性样本数 ≥ 20 的靶标 (可独立建模)")
    rich = summary_df[summary_df["Total_Estimate"] >= 20]
    if len(rich) > 0:
        for _, row in rich.iterrows():
            lines.append(f"- **{row['Gene']}** ({row['UniProt']}): {row['Total_Estimate']} 个活性化合物")
    else:
        lines.append("- 无")

    lines.append("\n### 阳性样本数 1-19 的靶标 (可做少样本学习)")
    medium = summary_df[(summary_df["Total_Estimate"] >= 1) & (summary_df["Total_Estimate"] < 20)]
    if len(medium) > 0:
        for _, row in medium.iterrows():
            lines.append(f"- {row['Gene']} ({row['UniProt']}): {row['Total_Estimate']} 个活性化合物")
    else:
        lines.append("- 无")

    lines.append("\n### 阳性样本数 = 0 的靶标 (冷启动, 依赖归纳式GNN)")
    zero = summary_df[summary_df["Total_Estimate"] == 0]
    if len(zero) > 0:
        for _, row in zero.iterrows():
            lines.append(f"- {row['Gene']} ({row['UniProt']}): 0 个活性化合物 -> 依赖 CPI-IGAE 归纳式推理")
    else:
        lines.append("- 无")

    # 策略建议
    lines.append("\n## 策略建议 (基于上述盘点)")
    total_positive = summary_df["Total_Estimate"].sum()
    positive_targets = (summary_df["Total_Estimate"] > 0).sum()
    lines.append(f"\n- 总阳性样本数: {total_positive} (去重前)")
    lines.append(f"- 有阳性样本的靶标数: {positive_targets}/{len(summary_df)}")
    lines.append(f"- 冷启动靶标数: {(summary_df['Total_Estimate'] == 0).sum()}/{len(summary_df)}")

    if total_positive >= 10000:
        lines.append("\n**结论: 阳性样本充足 (≥10,000), 可直接训练 CPI-IGAE 模型。**")
    elif total_positive >= 1000:
        lines.append(f"\n**结论: 阳性样本偏少 ({total_positive}), 建议: 1) 放宽活性阈值至 30μM; 2) 纳入更多 UniProt ID; 3) 使用 BindingDB 扩展。**")
    else:
        lines.append(f"\n**结论: 阳性样本不足 ({total_positive}), 建议转入 Phase 5 基于结构的虚拟筛选。**")

    report_path = L4_RESULTS / "experimental_actives_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  报告文件: {report_path}")

    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 4 Step 70: 实验验证活性化合物收集")
    logger.info(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"靶标数: {len(GENE_UNIPROT_MAP)} 个（铁衰老基因集）")
    logger.info("=" * 60)

    uniprot_ids = list(GENE_UNIPROT_MAP.values())

    # 1. ChEMBL
    chembl_records, chembl_stats = collect_chembl_data(GENE_UNIPROT_MAP)

    # 保存 ChEMBL 原始数据
    if chembl_records:
        chembl_df = pd.DataFrame(chembl_records)
        chembl_df.to_csv(L4_RESULTS / "chembl_active_compounds.csv", index=False)
        logger.info(f"  ChEMBL 数据保存: {len(chembl_df)} 条")

    # 2. BindingDB
    bindingdb_records, bindingdb_stats = query_bindingdb_by_uniprots(uniprot_ids, cutoff_nM=10000)

    if bindingdb_records:
        bind_df = pd.DataFrame(bindingdb_records)
        # 确保 standard_value_nM 为数值类型，避免后续模型训练时类型不一致
        bind_df["standard_value_nM"] = pd.to_numeric(
            bind_df["standard_value_nM"], errors="coerce"
        )
        bind_df.to_csv(L4_RESULTS / "bindingdb_active_compounds.csv", index=False)
        logger.info(f"  BindingDB 数据保存: {len(bind_df)} 条")

    # 3. DrugBank
    drugbank_records, drugbank_stats = collect_drugbank_data(GENE_UNIPROT_MAP)

    if drugbank_records:
        db_df = pd.DataFrame(drugbank_records)
        db_df.to_csv(L4_RESULTS / "drugbank_active_compounds.csv", index=False)
        logger.info(f"  DrugBank 数据保存: {len(db_df)} 条")

    # 4. 汇总
    summary_df = generate_summary(
        chembl_records, bindingdb_records, drugbank_records,
        chembl_stats, bindingdb_stats, drugbank_stats, GENE_UNIPROT_MAP
    )

    # 5. 报告
    report = generate_report(summary_df, chembl_records, bindingdb_records, drugbank_records)

    # 6. 保存详细数据
    all_details = []
    for r in chembl_records:
        all_details.append(r)
    for r in bindingdb_records:
        all_details.append(r)
    for r in drugbank_records:
        all_details.append(r)

    if all_details:
        detail_df = pd.DataFrame(all_details)
        detail_df.to_csv(L4_RESULTS / "experimental_actives_detail.csv", index=False)
        logger.info(f"  详细数据: {len(detail_df)} 条 -> {L4_RESULTS / 'experimental_actives_detail.csv'}")

    # 打印报告
    print("\n" + report)

    elapsed = time.time() - t_start
    logger.info(f"\n总耗时: {elapsed/60:.1f} 分钟")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"未捕获异常: {e}")
        traceback.print_exc()
        sys.exit(1)