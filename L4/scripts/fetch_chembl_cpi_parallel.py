#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行从ChEMBL API补充铁衰老基因CPI数据
ChEMBL API比BindingDB更可靠，5进程并行查询
"""
import pandas as pd
import requests
import time
import os
from multiprocessing import Pool

BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_96_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
EXP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "L4", "results_v10_minibatch")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cpi_supplement_v29.csv")

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# 70个缺失基因的UniProt映射（用于ChEMBL target查询）
UNIPROT_MAP = {
    "ABCC1":"P33527","ACVR1B":"P36896","ATF3":"P18847","ATG3":"Q9NT62",
    "BAP1":"Q92560","BRD7":"Q9NPI1","CAVIN1":"Q6NZI2","CD82":"P27701",
    "CDO1":"Q16878","COX7A1":"P24310","DPEP1":"P16444","DPP4":"P27487",
    "DUOX1":"Q9NRD9","E2F1":"Q01094","E2F3":"O00716","EBF3":"Q9H4W6",
    "EDN1":"P05305","EGR1":"P18146","EMP1":"P54849","EPHA2":"P29317",
    "ERN1":"O75460","FBXO31":"Q5XUX0","FOSL1":"P15407","GMFB":"P60983",
    "HBP1":"O60381","HERPUD1":"Q15011","HMGB1":"P09429","ICA1":"Q05084",
    "IFNG":"P01579","IGFBP7":"Q16270","IL6":"P05231","IRF1":"P10914",
    "IRF7":"Q92985","IRF9":"Q00978","KLF6":"Q99612","LACTB":"P83111",
    "LIFR":"P42702","LOX":"P28300","MAP3K14":"Q99558","MAPK14":"Q16539",
    "MCU":"Q8NE86","MEN1":"O00255","MPO":"P05164","NLRP3":"Q96P20",
    "NR1D1":"P20393","NR2F2":"P24468","NUAK2":"Q9H093","PADI4":"Q9UM07",
    "PPP2R2B":"Q00005","PRKD1":"Q15139","PTBP1":"P26599","RBM3":"P98179",
    "RUNX3":"Q13761","S100A8":"P05109","SETD7":"Q8WTS6","SLAMF8":"Q9P0V8",
    "SLC1A5":"Q15758","SMARCB1":"Q12824","SMURF2":"Q9HAU4","SNCA":"P37840",
    "SOCS1":"O15524","SOCS2":"O14508","SPATA2":"Q9UM82","TBX2":"Q13207",
    "TNFAIP1":"Q13829","TNFAIP3":"P21580","TXNIP":"Q9H3M7","WNT5A":"P41221",
    "WWTR1":"Q9GZV5","YAP1":"P46937","ZEB1":"P37275",
}


def get_missing_genes():
    """确定缺失基因"""
    df_96 = pd.read_csv(GENES_96_FILE)
    genes_96 = set(df_96['gene_symbol'].unique())
    covered = set()
    if os.path.exists(EXP_CPI_FILE):
        try:
            covered |= set(pd.read_csv(EXP_CPI_FILE, low_memory=False)['gene'].dropna().unique())
        except:
            pass
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("cpi_supplement_v") and f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
                    if 'gene' in df.columns:
                        covered |= set(df['gene'].dropna().unique())
                except:
                    pass
    return sorted(genes_96 - covered)


def get_chembl_target_id(uniprot_id, timeout=20):
    """通过UniProt ID获取ChEMBL target ID"""
    try:
        url = f"{CHEMBL_BASE}/target.json"
        params = {"target_components__accession": uniprot_id, "limit": 1}
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            targets = data.get("targets", [])
            if targets:
                return targets[0]["target_chembl_id"]
    except:
        pass
    return None


def get_chembl_activities(target_chembl_id, timeout=30):
    """通过ChEMBL target ID获取活性数据"""
    records = []
    try:
        # 获取IC50/Ki/Kd数据，ic50<=10000nM
        url = f"{CHEMBL_BASE}/activity.json"
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": "IC50",
            "standard_relation": "=",
            "standard_units": "nM",
            "standard_value__lte": 10000,
            "limit": 500,
        }
        resp = requests.get(url, params=params, timeout=timeout)
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
                                "gene": "",
                                "uniprot_id": "",
                                "canonical_smiles": smiles,
                                "activity_value": val,
                                "activity_unit": "nM",
                                "activity_type": "IC50",
                                "pubmed_id": str(act.get("document_chembl_id", "")),
                                "source": "ChEMBL_API"
                            })
                    except (ValueError, TypeError):
                        pass
    except:
        pass
    return records


def process_gene(gene_name):
    """处理单个基因"""
    records = []
    uniprot_id = UNIPROT_MAP.get(gene_name, "")
    if not uniprot_id:
        print(f"  [{gene_name}] 无UniProt映射，跳过")
        return gene_name, 0, records
    
    target_id = get_chembl_target_id(uniprot_id)
    if not target_id:
        print(f"  [{gene_name}] {uniprot_id} 无ChEMBL target")
        return gene_name, 0, records
    
    records = get_chembl_activities(target_id)
    for r in records:
        r["gene"] = gene_name
        r["uniprot_id"] = uniprot_id
    
    cnt = len(records)
    if cnt > 0:
        print(f"  [{gene_name}] {uniprot_id} -> {target_id}: {cnt} 条CPI")
    else:
        print(f"  [{gene_name}] {uniprot_id} -> {target_id}: 0 条")
    return gene_name, cnt, records


def process_gene_group(gene_group):
    """处理一组基因"""
    results = []
    stats = {}
    for gene_name in gene_group:
        _, cnt, records = process_gene(gene_name)
        results.extend(records)
        stats[gene_name] = cnt
        time.sleep(0.3)  # 避免API限流
    return results, stats


def main():
    print("=" * 60)
    print("铁衰老项目 - ChEMBL CPI并行补充 (5进程)")
    print("=" * 60)
    
    print("\n[步骤1] 确定缺失基因...")
    missing_genes = get_missing_genes()
    print(f"  {len(missing_genes)} 个基因缺失CPI数据")
    
    query_genes = [g for g in missing_genes if g in UNIPROT_MAP]
    print(f"  有UniProt映射的基因: {len(query_genes)}")
    
    # 分成5组
    group_size = max(1, len(query_genes) // 5)
    groups = [query_genes[i:i+group_size] for i in range(0, len(query_genes), group_size)]
    groups = groups[:5]  # 最多5组
    print(f"  分成 {len(groups)} 组并行处理")
    for i, g in enumerate(groups):
        print(f"    组{i+1}: {len(g)} 个基因: {g[:3]}...")
    
    print(f"\n[步骤2] 并行查询ChEMBL API ({len(groups)} 进程)...")
    all_records = []
    all_stats = {}
    
    with Pool(processes=min(len(groups), 5)) as pool:
        group_results = pool.map(process_gene_group, groups)
    
    for records, stats in group_results:
        all_records.extend(records)
        all_stats.update(stats)
    
    print(f"\n[步骤3] 保存结果...")
    if all_records:
        df_result = pd.DataFrame(all_records)
        before = len(df_result)
        df_result = df_result.drop_duplicates(subset=["gene", "canonical_smiles"])
        after = len(df_result)
        print(f"  去重: {before} -> {after}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  保存到: {OUTPUT_FILE}")
        print(f"  总记录数: {len(df_result)}")
    else:
        print("  [警告] ChEMBL API未返回数据")
        pd.DataFrame(columns=["gene","uniprot_id","canonical_smiles","activity_value",
                              "activity_unit","activity_type","pubmed_id","source"]
                    ).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    # 统计
    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)
    genes_with_data = sum(1 for v in all_stats.values() if v > 0)
    total_records = sum(all_stats.values())
    print(f"  查询基因数: {len(query_genes)}")
    print(f"  获取到CPI的基因数: {genes_with_data}")
    print(f"  总CPI记录数: {total_records}")
    
    if genes_with_data > 0:
        print("\n  每个基因新增CPI记录数:")
        for gene in sorted(all_stats.keys()):
            cnt = all_stats[gene]
            if cnt > 0:
                print(f"    {gene}: {cnt} 条")
    
    no_data = [g for g in query_genes if all_stats.get(g, 0) == 0]
    if no_data:
        print(f"\n  无CPI数据的基因 ({len(no_data)}个):")
        for g in no_data:
            print(f"    - {g}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()