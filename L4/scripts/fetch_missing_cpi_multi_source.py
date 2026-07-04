#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多源CPI数据补充脚本：从ChEMBL和PubChem为45个缺失铁衰老基因挖掘CPI数据。
顺序查询 + time.sleep 控制速率，不使用 multiprocessing。
ChEMBL间隔0.3秒，PubChem间隔0.5秒。
输出: cpi_supplement_v30.csv

PubChem API 策略:
  1. 按 proteinname 搜索 assay: assay/target/proteinname/{gene}/aids/JSON
  2. 获取 assay concise 数据: assay/aid/{aid}/concise/JSON
  3. 批量查询 CID -> SMILES: compound/cid/{cids}/property/CanonicalSMILES/JSON
"""

import os
import sys
import time
import traceback
import requests
import pandas as pd

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
OUTPUT_DIR = os.path.join(BASE_DIR, "L4", "results_v10_minibatch")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cpi_supplement_v30.csv")

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

REQUEST_TIMEOUT = 30
CHEMBL_SLEEP = 0.3
PUBCHEM_SLEEP = 0.5

# 45个缺失基因的UniProt映射
MISSING_UNIPROT = {
    "ACSL4": "O60488", "ATF3": "P18847", "ATG3": "Q9NT62", "CAVIN1": "Q6NZI2",
    "CD82": "P27701", "CDO1": "Q16878", "COX7A1": "P24310", "E2F1": "Q01094",
    "E2F3": "O00716", "EBF3": "Q9H4W6", "EDN1": "P05305", "EGR1": "P18146",
    "EMP1": "P54849", "FBXO31": "Q5XUX0", "FOSL1": "P15407", "GMFB": "P60983",
    "HBP1": "O60381", "HERPUD1": "Q15011", "HMGB1": "P09429", "ICA1": "Q05084",
    "IFNG": "P01579", "IGFBP7": "Q16270", "IRF1": "P10914", "IRF7": "Q92985",
    "IRF9": "Q00978", "KLF6": "Q99612", "LACTB": "P83111", "MCU": "Q8NE86",
    "PPP2R2B": "Q00005", "PTBP1": "P26599", "RBM3": "P98179", "RUNX3": "Q13761",
    "SLAMF8": "Q9P0V8", "SMARCB1": "Q12824", "SOCS1": "O15524", "SOCS2": "O14508",
    "SOD1": "P00441", "SPATA2": "Q9UM82", "TBX2": "Q13207", "TNFAIP1": "Q13829",
    "TNFAIP3": "P21580", "TXNIP": "Q9H3M7", "WNT5A": "P41221", "WWTR1": "Q9GZV5",
    "ZEB1": "P37275",
}

CSV_COLUMNS = ["gene", "uniprot_id", "canonical_smiles", "activity_value",
               "activity_unit", "activity_type", "pubmed_id", "source"]


# ============================================================
# ChEMBL API 查询
# ============================================================
def get_chembl_target_id(uniprot_id):
    """通过UniProt ID获取ChEMBL target ID"""
    try:
        url = f"{CHEMBL_BASE}/target.json"
        params = {"target_components__accession": uniprot_id, "limit": 1}
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            targets = data.get("targets", [])
            if targets:
                return targets[0]["target_chembl_id"]
        else:
            print(f"    [ChEMBL Target] HTTP {resp.status_code} for {uniprot_id}")
    except requests.exceptions.Timeout:
        print(f"    [ChEMBL Target] 请求超时: {uniprot_id}")
    except requests.exceptions.ConnectionError as e:
        print(f"    [ChEMBL Target] 连接错误: {uniprot_id} - {e}")
    except Exception as e:
        print(f"    [ChEMBL Target] 异常: {uniprot_id} - {e}")
        traceback.print_exc()
    return None


def get_chembl_activities(target_chembl_id):
    """通过ChEMBL target ID获取IC50 <= 10000nM的活性数据"""
    records = []
    try:
        url = f"{CHEMBL_BASE}/activity.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "standard_value__lte": 10000,
            "limit": 500,
        }
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            for act in data.get("activities", []):
                smiles = act.get("canonical_smiles", "")
                value = act.get("standard_value")
                if smiles and value is not None:
                    try:
                        val = float(value)
                        if val <= 10000:
                            records.append({
                                "canonical_smiles": smiles,
                                "activity_value": val,
                                "activity_unit": "nM",
                                "activity_type": "IC50",
                                "pubmed_id": str(act.get("document_chembl_id", "")),
                                "source": "ChEMBL_API"
                            })
                    except (ValueError, TypeError):
                        pass
        else:
            print(f"    [ChEMBL Activity] HTTP {resp.status_code} for {target_chembl_id}")
    except requests.exceptions.Timeout:
        print(f"    [ChEMBL Activity] 请求超时: {target_chembl_id}")
    except requests.exceptions.ConnectionError as e:
        print(f"    [ChEMBL Activity] 连接错误: {target_chembl_id} - {e}")
    except Exception as e:
        print(f"    [ChEMBL Activity] 异常: {target_chembl_id} - {e}")
        traceback.print_exc()
    return records


# ============================================================
# PubChem API 查询
# ============================================================
def _safe_pubchem_request(url, label="", timeout=None):
    """安全的PubChem API请求，带超时和错误处理"""
    if timeout is None:
        timeout = REQUEST_TIMEOUT
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None  # 未找到，正常情况
        else:
            print(f"    [PubChem {label}] HTTP {resp.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"    [PubChem {label}] 请求超时")
    except requests.exceptions.ConnectionError as e:
        print(f"    [PubChem {label}] 连接错误: {e}")
    except Exception as e:
        print(f"    [PubChem {label}] 异常: {e}")
        traceback.print_exc()
    return None


def pubchem_search_assays(gene_name):
    """按 proteinname 搜索PubChem BioAssay，返回AID列表"""
    url = f"{PUBCHEM_BASE}/assay/target/proteinname/{gene_name}/aids/JSON"
    data = _safe_pubchem_request(url, label=f"search:{gene_name}")
    if data is None:
        return []
    try:
        aids = data.get("IdentifierList", {}).get("AID", [])
        return [int(a) for a in aids]
    except Exception as e:
        print(f"    [PubChem search:{gene_name}] 解析响应异常: {e}")
        return []


def pubchem_get_assay_concise(aid):
    """获取单个PubChem assay的concise数据，返回IC50相关行（CID, 活性值, PubMed ID）"""
    url = f"{PUBCHEM_BASE}/assay/aid/{aid}/concise/JSON"
    data = _safe_pubchem_request(url, label=f"concise:{aid}")
    if data is None:
        return []

    results = []
    try:
        table = data.get("Table", {})
        columns = table.get("Columns", {}).get("Column", [])
        rows = table.get("Row", [])

        if not columns or not rows:
            return results

        # 找到关键列索引
        col_idx = {c: i for i, c in enumerate(columns)}
        cid_idx = col_idx.get("CID")
        outcome_idx = col_idx.get("Activity Outcome")
        value_idx = col_idx.get("Activity Value [uM]")
        name_idx = col_idx.get("Activity Name")
        assay_name_idx = col_idx.get("Assay Name")
        pmid_idx = col_idx.get("PubMed ID")

        if cid_idx is None:
            return results

        # 先判断整个assay是否为IC50相关（检查Assay Name）
        assay_is_ic50 = False
        if assay_name_idx is not None and rows:
            first_cells = rows[0].get("Cell", [])
            if assay_name_idx < len(first_cells):
                assay_is_ic50 = "IC50" in str(first_cells[assay_name_idx]).upper()

        for row in rows:
            cells = row.get("Cell", [])
            if len(cells) <= cid_idx:
                continue

            cid = cells[cid_idx]
            if not cid or cid == "0":
                continue

            # 检查Activity Name是否包含IC50
            activity_name = ""
            if name_idx is not None and name_idx < len(cells):
                activity_name = str(cells[name_idx]).upper()

            # 检查Activity Name或Assay Name是否包含IC50
            is_ic50_related = "IC50" in activity_name or assay_is_ic50
            if not is_ic50_related:
                continue

            # 检查Activity Outcome
            outcome = ""
            if outcome_idx is not None and outcome_idx < len(cells):
                outcome = str(cells[outcome_idx])

            if outcome != "Active":
                continue

            # 获取活性值 (uM)
            value_str = ""
            if value_idx is not None and value_idx < len(cells):
                value_str = cells[value_idx]

            if not value_str:
                continue

            try:
                value_um = float(value_str)
                value_nm = value_um * 1000.0  # uM -> nM
                if value_nm > 10000:
                    continue
            except (ValueError, TypeError):
                continue

            # 获取PubMed ID
            pubmed_id = ""
            if pmid_idx is not None and pmid_idx < len(cells):
                pubmed_id = str(cells[pmid_idx])

            results.append({
                "cid": cid,
                "activity_value_nm": value_nm,
                "pubmed_id": pubmed_id if pubmed_id and pubmed_id != "0" else f"PubChem_AID_{aid}",
            })

    except Exception as e:
        print(f"    [PubChem concise:{aid}] 解析异常: {e}")
        traceback.print_exc()

    return results


def pubchem_batch_smiles(cids):
    """批量查询CID -> SMILES，返回 {cid: smiles} 映射"""
    if not cids:
        return {}

    cid_map = {}
    # 分批处理，每批最多100个CID
    batch_size = 100
    for i in range(0, len(cids), batch_size):
        batch = cids[i:i + batch_size]
        cid_str = ",".join(batch)
        url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/CanonicalSMILES,IsomericSMILES/JSON"
        data = _safe_pubchem_request(url, label=f"SMILES_batch:{len(batch)}")
        if data is None:
            continue

        try:
            props = data.get("PropertyTable", {}).get("Properties", [])
            for prop in props:
                cid_val = str(prop.get("CID", ""))
                # PubChem返回的SMILES键名: 优先SMILES, 回退CanonicalSMILES/ConnectivitySMILES
                smiles = prop.get("SMILES", "") or prop.get("CanonicalSMILES", "") or prop.get("ConnectivitySMILES", "")
                if smiles and cid_val:
                    cid_map[cid_val] = smiles
        except Exception as e:
            print(f"    [PubChem SMILES batch] 解析异常: {e}")

        time.sleep(PUBCHEM_SLEEP)

    return cid_map


def process_gene_pubchem(gene_name):
    """PubChem查询：按proteinname搜索assay，获取IC50 CPI数据"""
    all_records = []
    aids = pubchem_search_assays(gene_name)

    if not aids:
        print(f"  [{gene_name}] PubChem: 无assay")
        return all_records

    print(f"  [{gene_name}] PubChem: {len(aids)} 个assay")

    # 限制最多20个assay
    max_assays = 20
    if len(aids) > max_assays:
        print(f"  [{gene_name}] PubChem: 限制到前{max_assays}个assay")
        aids = sorted(aids)[:max_assays]

    # 收集所有assay的IC50数据
    all_assay_results = []  # [(cid, activity_value_nm, pubmed_id), ...]
    for aid in aids:
        concise_results = pubchem_get_assay_concise(aid)
        if concise_results:
            all_assay_results.extend(concise_results)
        time.sleep(PUBCHEM_SLEEP)

    if not all_assay_results:
        print(f"  [{gene_name}] PubChem: 有assay但无IC50 Active数据")
        return all_records

    # 收集所有CID，批量查询SMILES
    unique_cids = list(set(r["cid"] for r in all_assay_results))
    print(f"  [{gene_name}] PubChem: {len(all_assay_results)} 条IC50, {len(unique_cids)} 个唯一CID, 查询SMILES...")

    cid_smiles = pubchem_batch_smiles(unique_cids)

    # 组装最终记录
    for r in all_assay_results:
        smiles = cid_smiles.get(r["cid"], "")
        if smiles:
            all_records.append({
                "gene": gene_name,
                "uniprot_id": MISSING_UNIPROT.get(gene_name, ""),
                "canonical_smiles": smiles,
                "activity_value": r["activity_value_nm"],
                "activity_unit": "nM",
                "activity_type": "IC50",
                "pubmed_id": r["pubmed_id"],
                "source": "PubChem_API"
            })

    pubchem_cnt = len(all_records)
    if pubchem_cnt > 0:
        print(f"  [{gene_name}] PubChem: {pubchem_cnt} 条CPI (含SMILES)")
    else:
        print(f"  [{gene_name}] PubChem: 0 条CPI (SMILES查询失败)")
    return all_records


# ============================================================
# 基因处理
# ============================================================
def process_gene_chembl(gene_name, uniprot_id):
    """ChEMBL查询：获取CPI数据"""
    records = []
    target_id = get_chembl_target_id(uniprot_id)
    if not target_id:
        print(f"  [{gene_name}] {uniprot_id} 无ChEMBL target")
        return records

    records = get_chembl_activities(target_id)
    for r in records:
        r["gene"] = gene_name
        r["uniprot_id"] = uniprot_id

    cnt = len(records)
    if cnt > 0:
        print(f"  [{gene_name}] {uniprot_id} -> {target_id} (ChEMBL): {cnt} 条CPI")
    else:
        print(f"  [{gene_name}] {uniprot_id} -> {target_id} (ChEMBL): 0 条")
    return records


def process_gene(gene_name):
    """处理单个基因：ChEMBL + PubChem"""
    uniprot_id = MISSING_UNIPROT.get(gene_name, "")
    if not uniprot_id:
        print(f"  [{gene_name}] 无UniProt映射，跳过")
        return gene_name, 0, 0, []

    all_records = []

    # ChEMBL查询
    chembl_records = process_gene_chembl(gene_name, uniprot_id)
    all_records.extend(chembl_records)
    time.sleep(CHEMBL_SLEEP)

    # PubChem查询
    pubchem_records = process_gene_pubchem(gene_name)
    all_records.extend(pubchem_records)

    # 去重（按SMILES去重，保留首次出现的source）
    seen_smiles = set()
    deduped = []
    for r in all_records:
        smi = r["canonical_smiles"]
        if smi not in seen_smiles:
            seen_smiles.add(smi)
            deduped.append(r)

    chembl_cnt = len([r for r in deduped if r["source"] == "ChEMBL_API"])
    pubchem_cnt = len([r for r in deduped if r["source"] == "PubChem_API"])

    print(f"  [{gene_name}] 总计: ChEMBL={chembl_cnt}, PubChem={pubchem_cnt}, 去重后={len(deduped)}")
    return gene_name, chembl_cnt, pubchem_cnt, deduped


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 70)
    print("铁衰老项目 - 多源CPI数据补充 (ChEMBL + PubChem)")
    print("=" * 70)
    print(f"  缺失基因数: {len(MISSING_UNIPROT)}")
    print(f"  ChEMBL间隔: {CHEMBL_SLEEP}s | PubChem间隔: {PUBCHEM_SLEEP}s")
    print(f"  输出文件: {OUTPUT_FILE}")
    print()

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 顺序处理所有基因
    all_records = []
    gene_stats = {}  # gene_name -> {"chembl": int, "pubchem": int, "total": int}

    genes = sorted(MISSING_UNIPROT.keys())
    total = len(genes)
    start_time = time.time()

    for idx, gene_name in enumerate(genes):
        print(f"\n[{idx+1}/{total}] 处理基因: {gene_name}")
        print("-" * 50)

        try:
            gene, chembl_cnt, pubchem_cnt, records = process_gene(gene_name)
            all_records.extend(records)
            gene_stats[gene] = {
                "chembl": chembl_cnt,
                "pubchem": pubchem_cnt,
                "total": len(records)
            }
        except Exception as e:
            print(f"  [{gene_name}] 处理失败: {e}")
            traceback.print_exc()
            gene_stats[gene_name] = {"chembl": 0, "pubchem": 0, "total": 0}

    elapsed = time.time() - start_time

    # ============================================================
    # 保存结果
    # ============================================================
    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    if all_records:
        df_result = pd.DataFrame(all_records)
        # 确保列顺序
        df_result = df_result[CSV_COLUMNS]
        # 全局去重
        before = len(df_result)
        df_result = df_result.drop_duplicates(subset=["gene", "canonical_smiles"])
        after = len(df_result)
        print(f"  全局去重: {before} -> {after}")

        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  保存到: {OUTPUT_FILE}")
        print(f"  总记录数: {len(df_result)}")
    else:
        print("  [警告] 未获取到任何CPI数据，生成空文件")
        df_result = pd.DataFrame(columns=CSV_COLUMNS)
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # ============================================================
    # 统计报告
    # ============================================================
    print("\n" + "=" * 70)
    print("搜索统计报告")
    print("=" * 70)
    print(f"  总基因数: {total}")
    print(f"  总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")

    genes_with_chembl = sum(1 for s in gene_stats.values() if s["chembl"] > 0)
    genes_with_pubchem = sum(1 for s in gene_stats.values() if s["pubchem"] > 0)
    genes_with_any = sum(1 for s in gene_stats.values() if s["total"] > 0)
    total_chembl = sum(s["chembl"] for s in gene_stats.values())
    total_pubchem = sum(s["pubchem"] for s in gene_stats.values())
    total_all = sum(s["total"] for s in gene_stats.values())

    print(f"  ChEMBL有数据: {genes_with_chembl} 个基因, {total_chembl} 条CPI")
    print(f"  PubChem有数据: {genes_with_pubchem} 个基因, {total_pubchem} 条CPI")
    print(f"  总计有CPI: {genes_with_any} 个基因, {total_all} 条CPI")

    print("\n  每个基因详细统计:")
    print(f"  {'基因':<10} {'ChEMBL':>8} {'PubChem':>8} {'合计':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6}")
    for gene in sorted(gene_stats.keys()):
        s = gene_stats[gene]
        print(f"  {gene:<10} {s['chembl']:>8} {s['pubchem']:>8} {s['total']:>6}")

    # 无数据基因
    no_data = [g for g in sorted(gene_stats.keys()) if gene_stats[g]["total"] == 0]
    if no_data:
        print(f"\n  无CPI数据的基因 ({len(no_data)}个):")
        for g in no_data:
            print(f"    - {g}")

    print(f"\n  输出文件: {OUTPUT_FILE}")
    print("\n完成!")


if __name__ == "__main__":
    main()