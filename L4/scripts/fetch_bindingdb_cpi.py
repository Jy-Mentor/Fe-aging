#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从BindingDB API补充铁衰老基因的CPI数据
"""

import pandas as pd
import requests
import time
import sys
import os
import json
import xml.etree.ElementTree as ET

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_96_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
EXP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
SUPP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v28.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v29.csv")

# BindingDB API
BINDINGDB_BASE = "https://bindingdb.org/axis2/services/BDBService"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# 已知的基因-UniProt映射（从已有数据中提取）
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
    "CXCL10": "P02778", "CD74": "P04233", "CTSB": "P07858",
    "S100A8": "P05109", "LPCAT3": "Q6P1A2", "ACVR1B": "P36896",
    "EPHA2": "P29317", "ERN1": "O75460", "DPP4": "P27487",
    "MAPK14": "Q16539", "NLRP3": "Q96P20", "MPO": "P05164",
    "IL6": "P05231", "IFNG": "P01579", "HMGB1": "P09429",
    "TXNIP": "Q9H3M7", "EGR1": "P18146", "IRF1": "P10914",
    "SOCS1": "O15524", "SP1": "P08047", "WNT5A": "P41221",
    "ZEB1": "P37275", "ATF3": "P18847", "ATG3": "Q9NT62",
    "BAP1": "Q92560", "BRD7": "Q9NPI1", "CAVIN1": "Q6NZI2",
    "CD82": "P27701", "CDO1": "Q16878", "COX7A1": "P24310",
    "DPEP1": "P16444", "DUOX1": "Q9NRD9", "E2F1": "Q01094",
    "E2F3": "O00716", "EBF3": "Q9H4W6", "EDN1": "P05305",
    "EMP1": "P54849", "FBXO31": "Q5XUX0", "FOSL1": "P15407",
    "GMFB": "P60983", "HBP1": "O60381", "HERPUD1": "Q15011",
    "ICA1": "Q05084", "IRF7": "Q92985", "IRF9": "Q00978",
    "KLF6": "Q99612", "LACTB": "P83111", "LIFR": "P42702",
    "LOX": "P28300", "MAP3K14": "Q99558", "MCU": "Q8NE86",
    "MEN1": "O00255", "NR1D1": "P20393", "NR2F2": "P24468",
    "NUAK2": "Q9H093", "PADI4": "Q9UM07", "PPP2R2B": "Q00005",
    "PRKD1": "Q15139", "PTBP1": "P26599", "RBM3": "P98179",
    "RUNX3": "Q13761", "SETD7": "Q8WTS6", "SLAMF8": "Q9P0V8",
    "SLC1A5": "Q15758", "SMARCB1": "Q12824", "SMURF2": "Q9HAU4",
    "SNCA": "P37840", "SOCS2": "O14508", "SPATA2": "Q9UM82",
    "TBX2": "Q13207", "TNFAIP1": "Q13829", "TNFAIP3": "P21580",
    "WWTR1": "Q9GZV5", "YAP1": "P46937", "ABCC1": "P33527",
}

def get_missing_genes():
    """确定哪些基因没有CPI数据"""
    df_96 = pd.read_csv(GENES_96_FILE)
    genes_96 = set(df_96['gene_symbol'].unique())
    
    # 从已有CPI数据中获取已覆盖的基因
    covered_genes = set()
    if os.path.exists(EXP_CPI_FILE):
        df_exp = pd.read_csv(EXP_CPI_FILE, low_memory=False)
        covered_genes |= set(df_exp['gene'].dropna().unique())
    if os.path.exists(SUPP_CPI_FILE):
        df_supp = pd.read_csv(SUPP_CPI_FILE)
        if 'gene' in df_supp.columns:
            covered_genes |= set(df_supp['gene'].dropna().unique())
    
    covered = genes_96 & covered_genes
    missing = sorted(genes_96 - covered_genes)
    
    print(f"96个基因中已覆盖: {len(covered)}, 缺失: {len(missing)}")
    return missing


def query_uniprot_for_gene(gene_symbol):
    """通过UniProt REST API查询基因的UniProt ID"""
    if gene_symbol in KNOWN_UNIPROT_MAP:
        return KNOWN_UNIPROT_MAP[gene_symbol]
    
    params = {
        "query": f"gene:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "format": "json",
        "size": 5
    }
    try:
        resp = requests.get(UNIPROT_API, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            if results:
                # 取第一个结果（reviewed Swiss-Prot条目）
                uniprot_id = results[0].get("primaryAccession", "")
                return uniprot_id
    except Exception as e:
        print(f"  [警告] UniProt查询 {gene_symbol} 失败: {e}")
    return None


def query_bindingdb_by_uniprot(uniprot_id, gene_symbol):
    """通过BindingDB API查询CPI数据"""
    url = f"{BINDINGDB_BASE}/getLigandsByUniprotID"
    params = {
        "uniprot": uniprot_id,
        "affinity_type": "IC50",
        "cutoff": 10000,
        "response": "json"
    }
    
    records = []
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            print(f"  [警告] BindingDB API返回状态码 {resp.status_code} for {gene_symbol} ({uniprot_id})")
            return records
        
        raw_text = resp.text
        
        # 尝试解析JSON
        try:
            data = resp.json()
        except json.JSONDecodeError:
            # 有些返回可能是XML格式
            try:
                root = ET.fromstring(raw_text)
                print(f"  [警告] {gene_symbol} 返回XML格式，尝试解析...")
                # 简单的XML解析
                return records
            except:
                print(f"  [警告] 无法解析BindingDB响应 for {gene_symbol}")
                return records
        
        # 处理JSON响应
        if isinstance(data, dict):
            # 检查是否有getLigandsByUniprotIDResponse
            if "getLigandsByUniprotIDResponse" in data:
                inner = data["getLigandsByUniprotIDResponse"]
                if "return" in inner:
                    data = inner["return"]
            
            if "affinities" in data:
                affinities = data["affinities"]
                if isinstance(affinities, list):
                    for aff in affinities:
                        record = parse_bindingdb_affinity(aff, gene_symbol, uniprot_id)
                        if record:
                            records.append(record)
                elif isinstance(affinities, dict):
                    record = parse_bindingdb_affinity(affinities, gene_symbol, uniprot_id)
                    if record:
                        records.append(record)
        
        elif isinstance(data, list):
            for item in data:
                record = parse_bindingdb_affinity(item, gene_symbol, uniprot_id)
                if record:
                    records.append(record)
        
        # 限制每个基因最多500条
        if len(records) > 500:
            records = records[:500]
            
    except requests.exceptions.Timeout:
        print(f"  [超时] BindingDB查询超时 for {gene_symbol} ({uniprot_id})")
    except requests.exceptions.ConnectionError:
        print(f"  [连接错误] BindingDB连接失败 for {gene_symbol} ({uniprot_id})")
    except Exception as e:
        print(f"  [错误] BindingDB查询失败 for {gene_symbol}: {e}")
    
    return records


def parse_bindingdb_affinity(aff_data, gene_symbol, uniprot_id):
    """解析BindingDB返回的亲和力数据"""
    try:
        if isinstance(aff_data, str):
            # 可能是JSON字符串
            try:
                aff_data = json.loads(aff_data)
            except:
                return None
        
        smiles = None
        ic50_value = None
        ic50_unit = "nM"
        pubmed_id = None
        
        if isinstance(aff_data, dict):
            smiles = aff_data.get("canonicalSmiles") or aff_data.get("smiles") or aff_data.get("ligandSmiles")
            ic50_value = aff_data.get("ic50") or aff_data.get("ki") or aff_data.get("kd") or aff_data.get("ec50")
            ic50_unit = aff_data.get("ic50Unit") or aff_data.get("unit") or "nM"
            pubmed_id = aff_data.get("pmid") or aff_data.get("pubmedId") or aff_data.get("pubmed_id")
        
        if smiles and smiles.strip():
            return {
                "gene": gene_symbol,
                "uniprot_id": uniprot_id,
                "canonical_smiles": str(smiles).strip(),
                "activity_value": ic50_value,
                "activity_unit": ic50_unit,
                "activity_type": "IC50",
                "pubmed_id": pubmed_id,
                "source": "BindingDB"
            }
    except Exception as e:
        pass
    return None


def main():
    print("=" * 60)
    print("铁衰老项目 - BindingDB CPI数据补充")
    print("=" * 60)
    
    # 步骤1: 确定缺失基因
    print("\n[步骤1] 确定缺失基因...")
    missing_genes = get_missing_genes()
    print(f"  {len(missing_genes)} 个基因缺失CPI数据:")
    for i, g in enumerate(missing_genes):
        print(f"    {i+1}. {g}")
    
    # 步骤2: 查询UniProt ID
    print("\n[步骤2] 查询UniProt ID...")
    gene_uniprot_map = {}
    for gene in missing_genes:
        uniprot = query_uniprot_for_gene(gene)
        if uniprot:
            gene_uniprot_map[gene] = uniprot
            print(f"  {gene} -> {uniprot}")
        else:
            print(f"  {gene} -> 未找到UniProt ID")
        time.sleep(0.3)  # 避免API限流
    
    print(f"\n  成功获取UniProt ID: {len(gene_uniprot_map)}/{len(missing_genes)}")
    
    # 步骤3: 从BindingDB获取CPI数据
    print("\n[步骤3] 从BindingDB API获取CPI数据...")
    all_records = []
    gene_stats = {}
    
    for idx, (gene, uniprot) in enumerate(gene_uniprot_map.items()):
        print(f"\n  [{idx+1}/{len(gene_uniprot_map)}] 查询 {gene} ({uniprot})...")
        records = query_bindingdb_by_uniprot(uniprot, gene)
        all_records.extend(records)
        gene_stats[gene] = len(records)
        print(f"    获取到 {len(records)} 条记录")
        time.sleep(0.5)  # 避免API限流
    
    # 步骤4: 保存结果
    print("\n[步骤4] 保存结果...")
    if all_records:
        df_result = pd.DataFrame(all_records)
        # 去重（基于gene + canonical_smiles）
        df_result = df_result.drop_duplicates(subset=["gene", "canonical_smiles"])
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  保存到: {OUTPUT_FILE}")
        print(f"  总记录数: {len(df_result)}")
    else:
        # 创建空文件保持格式
        df_result = pd.DataFrame(columns=[
            "gene", "uniprot_id", "canonical_smiles", 
            "activity_value", "activity_unit", "activity_type",
            "pubmed_id", "source"
        ])
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  未获取到任何数据，保存空文件到: {OUTPUT_FILE}")
    
    # 步骤5: 统计报告
    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)
    print(f"  查询了基因数: {len(missing_genes)}")
    print(f"  成功获取UniProt ID: {len(gene_uniprot_map)}")
    print(f"  成功获取CPI数据的基因数: {sum(1 for v in gene_stats.values() if v > 0)}")
    print(f"  总CPI记录数: {len(all_records)}")
    print(f"  去重后记录数: {len(df_result)}")
    
    print("\n  每个基因新增CPI记录数:")
    for gene in sorted(gene_stats.keys()):
        cnt = gene_stats[gene]
        if cnt > 0:
            print(f"    {gene}: {cnt} 条")
        else:
            print(f"    {gene}: 0 条 (无数据)")
    
    # 仍然没有数据的基因
    no_data_genes = [g for g in missing_genes if g not in gene_uniprot_map or gene_stats.get(g, 0) == 0]
    if no_data_genes:
        print(f"\n  仍然没有CPI数据的基因 ({len(no_data_genes)}个):")
        for g in no_data_genes:
            print(f"    - {g}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()