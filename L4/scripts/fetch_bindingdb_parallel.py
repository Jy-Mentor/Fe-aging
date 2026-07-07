#!/usr/bin/env python
import logging
logger = logging.getLogger(__name__)

"""
并行从BindingDB REST API + 本地TSV分片补充铁衰老基因CPI数据
策略：70个基因分5组，5个进程并行查询BindingDB API
"""
import pandas as pd
import requests
import time
import sys
import os
import csv
import json
from multiprocessing import Pool, Manager
from functools import partial

BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_96_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
EXP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "L4", "results_v10_minibatch")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cpi_supplement_v29.csv")

# 70个缺失基因的UniProt映射
KNOWN_UNIPROT_MAP = {
    "TLR4": "O00206", "HMOX1": "P09601", "MAPK1": "P28482",
    "NFE2L2": "Q16236", "TP53": "P04637", "STAT3": "P40763",
    "MTOR": "P42345", "NFKB1": "P19838", "RELA": "Q04206",
    "ALOX5": "P09917", "ATG7": "O95352", "MAP1LC3B": "Q9GZQ8",
    "ACSL4": "O60488", "IGFBP7": "Q16270", "SOD1": "P00441",
    "GPX4": "P36969", "SLC7A11": "Q9UPY5", "FTH1": "P02794",
    "BCL6": "P41182", "DYRK1A": "Q13627", "EPHA4": "P54764",
    "HIF1A": "Q16665", "PDE4B": "Q07343", "CD74": "P04233",
    "LGMN": "Q99538", "NOX4": "Q9NPH5", "PTGS2": "P35354",
    "KDM6B": "O15054", "LCN2": "P80188", "SAT1": "P21673",
    "TFRC": "P02786", "KEAP1": "Q14145", "IL1B": "P01584",
    "CXCL10": "P02778", "CTSB": "P07858", "S100A8": "P05109",
    "LPCAT3": "Q6P1A2", "ACVR1B": "P36896", "EPHA2": "P29317",
    "ERN1": "O75460", "DPP4": "P27487", "MAPK14": "Q16539",
    "NLRP3": "Q96P20", "MPO": "P05164", "IL6": "P05231",
    "IFNG": "P01579", "HMGB1": "P09429", "TXNIP": "Q9H3M7",
    "EGR1": "P18146", "IRF1": "P10914", "SOCS1": "O15524",
    "SP1": "P08047", "WNT5A": "P41221", "ZEB1": "P37275",
    "ATF3": "P18847", "ATG3": "Q9NT62", "BAP1": "Q92560",
    "BRD7": "Q9NPI1", "CAVIN1": "Q6NZI2", "CD82": "P27701",
    "CDO1": "Q16878", "COX7A1": "P24310", "DPEP1": "P16444",
    "DUOX1": "Q9NRD9", "E2F1": "Q01094", "E2F3": "O00716",
    "EBF3": "Q9H4W6", "EDN1": "P05305", "EMP1": "P54849",
    "FBXO31": "Q5XUX0", "FOSL1": "P15407", "GMFB": "P60983",
    "HBP1": "O60381", "HERPUD1": "Q15011", "ICA1": "Q05084",
    "IRF7": "Q92985", "IRF9": "Q00978", "KLF6": "Q99612",
    "LACTB": "P83111", "LIFR": "P42702", "LOX": "P28300",
    "MAP3K14": "Q99558", "MCU": "Q8NE86", "MEN1": "O00255",
    "NR1D1": "P20393", "NR2F2": "P24468", "NUAK2": "Q9H093",
    "PADI4": "Q9UM07", "PPP2R2B": "Q00005", "PRKD1": "Q15139",
    "PTBP1": "P26599", "RBM3": "P98179", "RUNX3": "Q13761",
    "SETD7": "Q8WTS6", "SLAMF8": "Q9P0V8", "SLC1A5": "Q15758",
    "SMARCB1": "Q12824", "SMURF2": "Q9HAU4", "SNCA": "P37840",
    "SOCS2": "O14508", "SPATA2": "Q9UM82", "TBX2": "Q13207",
    "TNFAIP1": "Q13829", "TNFAIP3": "P21580", "WWTR1": "Q9GZV5",
    "YAP1": "P46937", "ABCC1": "P33527",
}

BINDINGDB_API = "https://www.bindingdb.org/axis2/services/BDBService"


def get_missing_genes():
    """确定缺失基因"""
    df_96 = pd.read_csv(GENES_96_FILE)
    genes_96 = set(df_96['gene_symbol'].unique())
    covered = set()
    if os.path.exists(EXP_CPI_FILE):
        covered |= set(pd.read_csv(EXP_CPI_FILE, low_memory=False)['gene'].dropna().unique())
    # 也检查已有的补充数据
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith("cpi_supplement_v") and f.endswith(".csv"):
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
            if 'gene' in df.columns:
                covered |= set(df['gene'].dropna().unique())
    missing = sorted(genes_96 - covered)
    return missing


def query_bindingdb_api(uniprot_id, gene_name, timeout=30):
    """通过BindingDB REST API查询单个基因的CPI数据"""
    records = []
    try:
        # 查询IC50数据
        url = f"{BINDINGDB_API}/getLigandsByUniprotID"
        params = {
            "uniprot": uniprot_id,
            "response": "json",
            "affinity_type": "IC50",
            "affinity_cutoff": 10000,
            "max_results": 1000,
        }
        resp = requests.get(url, params=params, timeout=timeout, headers={"Accept": "application/json"})
        if resp.status_code != 200:
            return records
        
        try:
            data = resp.json()
        except Exception:
            logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")
            return records

        # 解析返回数据
        if isinstance(data, dict) and "getLigandsByUniprotIDResponse" in data:
            ligands = data["getLigandsByUniprotIDResponse"].get("affinities", [])
            if not isinstance(ligands, list):
                ligands = [ligands] if ligands else []
            
            for lig in ligands:
                smiles = lig.get("smiles", "")
                ic50 = lig.get("ic50", None)
                pubmed = lig.get("pmid", "")
                
                if smiles and ic50 is not None:
                    try:
                        ic50_val = float(ic50)
                        if ic50_val <= 10000:
                            records.append({
                                "gene": gene_name,
                                "uniprot_id": uniprot_id,
                                "canonical_smiles": smiles,
                                "activity_value": ic50_val,
                                "activity_unit": "nM",
                                "activity_type": "IC50",
                                "pubmed_id": str(pubmed) if pubmed else "",
                                "source": "BindingDB_API"
                            })
                    except (ValueError, TypeError):
                        pass
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
        pass

    return records


def query_uniprot_api(gene_name, timeout=30):
    """通过UniProt API获取蛋白的UniProt ID"""
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene:{gene_name}+AND+organism_id:9606+AND+reviewed:true",
            "format": "json",
            "size": 1,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0].get("primaryAccession", "")
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
        pass

    return ""


def process_gene_group(gene_group):
    """处理一组基因，查询BindingDB API"""
    results = []
    stats = {}
    for gene_name in gene_group:
        uniprot_id = KNOWN_UNIPROT_MAP.get(gene_name, "")
        if not uniprot_id:
            # 尝试从UniProt API获取
            uniprot_id = query_uniprot_api(gene_name)
            if not uniprot_id:
                print(f"  [{gene_name}] 无UniProt ID，跳过")
                stats[gene_name] = 0
                continue
        
        print(f"  [{gene_name}] {uniprot_id} 查询中...")
        records = query_bindingdb_api(uniprot_id, gene_name)
        results.extend(records)
        stats[gene_name] = len(records)
        print(f"  [{gene_name}] 获取 {len(records)} 条CPI记录")
        time.sleep(0.5)  # 避免API限流
    return results, stats


def main():
    print("=" * 60)
    print("铁衰老项目 - BindingDB CPI并行补充 (5进程)")
    print("=" * 60)
    
    # 确定缺失基因
    print("\n[步骤1] 确定缺失基因...")
    missing_genes = get_missing_genes()
    print(f"  {len(missing_genes)} 个基因缺失CPI数据")
    
    # 过滤出有UniProt映射的基因
    query_genes = [g for g in missing_genes if g in KNOWN_UNIPROT_MAP]
    print(f"  有UniProt映射的基因: {len(query_genes)}")
    
    # 分成5组
    group_size = max(1, len(query_genes) // 5)
    groups = [query_genes[i:i+group_size] for i in range(0, len(query_genes), group_size)]
    print(f"  分成 {len(groups)} 组并行处理")
    for i, g in enumerate(groups):
        print(f"    组{i+1}: {len(g)} 个基因: {g[:3]}...")
    
    # 并行处理
    print(f"\n[步骤2] 并行查询BindingDB API ({len(groups)} 进程)...")
    all_records = []
    all_stats = {}
    
    with Pool(processes=min(len(groups), 5)) as pool:
        group_results = pool.map(process_gene_group, groups)
    
    for records, stats in group_results:
        all_records.extend(records)
        all_stats.update(stats)
    
    # 保存结果
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
        print("  [警告] API未返回任何数据，尝试TSV方式...")
        # 回退到TSV方式
        print("  请运行 fetch_bindingdb_cpi_v2.py 进行TSV下载")
        # 创建空文件
        pd.DataFrame(columns=["gene","uniprot_id","canonical_smiles","activity_value",
                              "activity_unit","activity_type","pubmed_id","source"]
                    ).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    # 统计
    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)
    genes_with_data = sum(1 for v in all_stats.values() if v > 0)
    print(f"  查询基因数: {len(query_genes)}")
    print(f"  获取到CPI的基因数: {genes_with_data}")
    print(f"  总CPI记录数: {len(all_records)}")
    
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