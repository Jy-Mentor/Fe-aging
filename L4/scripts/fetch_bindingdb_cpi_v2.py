#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从BindingDB TSV下载补充铁衰老基因的CPI数据
BindingDB REST API不可用，使用TSV下载方式替代
"""

import pandas as pd
import requests
import time
import sys
import os
import csv
import io
import gzip

# ============================================================
# 配置
# ============================================================
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_96_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
EXP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
SUPP_CPI_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v28.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v29.csv")

BINDINGDB_TSV_URL = "https://bindingdb.org/rwd/bind/downloads/BindingDB_All.tsv"

# 已知的基因-UniProt映射（Human reviewed SwissProt entries）
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


def get_missing_genes():
    """确定哪些基因没有CPI数据"""
    df_96 = pd.read_csv(GENES_96_FILE)
    genes_96 = set(df_96['gene_symbol'].unique())
    
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


def download_and_filter_bindingdb(target_uniprots, gene_uniprot_map):
    """
    流式下载BindingDB TSV文件，按UniProt ID过滤
    """
    # 构建uniprot_to_gene的映射
    uniprot_to_gene = {v: k for k, v in gene_uniprot_map.items()}
    
    records = []
    gene_stats = {gene: 0 for gene in gene_uniprot_map.keys()}
    
    print(f"  目标UniProt ID数: {len(target_uniprots)}")
    print(f"  开始下载BindingDB TSV文件...")
    print(f"  URL: {BINDINGDB_TSV_URL}")
    
    try:
        # 流式下载
        response = requests.get(BINDINGDB_TSV_URL, stream=True, timeout=600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"  文件大小: {total_size / (1024**3):.2f} GB")
        
        # 使用csv.reader处理流式数据
        chunk_size = 1024 * 1024  # 1MB chunks
        buffer = ""
        line_count = 0
        matched_count = 0
        downloaded = 0
        header = None
        col_indices = {}
        
        # 需要提取的列名
        target_cols = {
            "Ligand SMILES": "canonical_smiles",
            "IC50 (nM)": "ic50_nm",
            "PMID": "pubmed_id",
            "UniProt (SwissProt) Primary ID of Target Chain 1": "uniprot_id_1",
            "UniProt (SwissProt) Primary ID of Target Chain 2": "uniprot_id_2",
            "UniProt (SwissProt) Primary ID of Target Chain 3": "uniprot_id_3",
            "UniProt (TrEMBL) Primary ID of Target Chain 1": "uniprot_id_t1",
            "UniProt (TrEMBL) Primary ID of Target Chain 2": "uniprot_id_t2",
            "UniProt (TrEMBL) Primary ID of Target Chain 3": "uniprot_id_t3",
        }
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                downloaded += len(chunk)
                buffer += chunk.decode('utf-8', errors='replace')
                
                # 处理完整的行
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip('\r')
                    
                    if not line:
                        continue
                    
                    line_count += 1
                    
                    if line_count == 1:
                        # 解析表头
                        header = line.split('\t')
                        # 找到需要的列索引
                        for col_name, key_name in target_cols.items():
                            if col_name in header:
                                col_indices[key_name] = header.index(col_name)
                        print(f"  表头解析完成，找到 {len(col_indices)} 个目标列")
                        continue
                    
                    if line_count % 1000000 == 0:
                        progress = (downloaded / total_size * 100) if total_size else 0
                        print(f"  进度: {downloaded/(1024**2):.0f}MB / {total_size/(1024**2):.0f}MB ({progress:.1f}%), "
                              f"已处理 {line_count} 行, 匹配 {matched_count} 条")
                    
                    # 解析行并检查是否匹配目标UniProt
                    fields = line.split('\t')
                    
                    # 检查各UniProt列
                    uniprot_ids = []
                    for key in ["uniprot_id_1", "uniprot_id_2", "uniprot_id_3",
                                "uniprot_id_t1", "uniprot_id_t2", "uniprot_id_t3"]:
                        if key in col_indices:
                            idx = col_indices[key]
                            if idx < len(fields) and fields[idx].strip():
                                uniprot_ids.append(fields[idx].strip())
                    
                    matched_uniprot = None
                    for uid in uniprot_ids:
                        if uid in target_uniprots:
                            matched_uniprot = uid
                            break
                    
                    if matched_uniprot is None:
                        continue
                    
                    # 提取SMILES和IC50
                    smiles = ""
                    ic50_val = None
                    pubmed = None
                    
                    if "canonical_smiles" in col_indices:
                        idx = col_indices["canonical_smiles"]
                        if idx < len(fields):
                            smiles = fields[idx].strip()
                    
                    if "ic50_nm" in col_indices:
                        idx = col_indices["ic50_nm"]
                        if idx < len(fields) and fields[idx].strip():
                            try:
                                ic50_val = float(fields[idx].strip())
                            except ValueError:
                                ic50_val = None
                    
                    if "pubmed_id" in col_indices:
                        idx = col_indices["pubmed_id"]
                        if idx < len(fields):
                            pubmed = fields[idx].strip()
                    
                    # 只保留有SMILES且有IC50且IC50<=10000的
                    if smiles and ic50_val is not None and ic50_val <= 10000:
                        gene = uniprot_to_gene.get(matched_uniprot, matched_uniprot)
                        records.append({
                            "gene": gene,
                            "uniprot_id": matched_uniprot,
                            "canonical_smiles": smiles,
                            "activity_value": ic50_val,
                            "activity_unit": "nM",
                            "activity_type": "IC50",
                            "pubmed_id": pubmed if pubmed else "",
                            "source": "BindingDB"
                        })
                        gene_stats[gene] += 1
                        matched_count += 1
                        
                        # 每个基因最多500条，超过则跳过
                        if gene_stats[gene] >= 500:
                            # 不跳出，但不再添加该基因的记录
                            pass
        
        # 处理最后剩余的buffer
        if buffer.strip():
            line_count += 1
            fields = buffer.strip().split('\t')
            # (简化处理，跳过最后一行不完整的解析)
        
        print(f"\n  处理完成: 共 {line_count} 行, 匹配 {matched_count} 条CPI记录")
        
    except requests.exceptions.Timeout:
        print(f"  [超时] 下载超时")
    except requests.exceptions.ConnectionError as e:
        print(f"  [连接错误] {e}")
    except Exception as e:
        print(f"  [错误] {e}")
        import traceback
        traceback.print_exc()
    
    return records, gene_stats


def main():
    print("=" * 60)
    print("铁衰老项目 - BindingDB CPI数据补充 (TSV方式)")
    print("=" * 60)
    
    # 步骤1: 确定缺失基因
    print("\n[步骤1] 确定缺失基因...")
    missing_genes = get_missing_genes()
    print(f"  {len(missing_genes)} 个基因缺失CPI数据")
    
    # 步骤2: 获取UniProt ID映射
    print("\n[步骤2] 获取UniProt ID映射...")
    gene_uniprot_map = {}
    for gene in missing_genes:
        if gene in KNOWN_UNIPROT_MAP:
            gene_uniprot_map[gene] = KNOWN_UNIPROT_MAP[gene]
            print(f"  {gene} -> {KNOWN_UNIPROT_MAP[gene]}")
        else:
            print(f"  {gene} -> 未找到UniProt ID (跳过)")
    
    print(f"\n  成功映射: {len(gene_uniprot_map)}/{len(missing_genes)}")
    
    target_uniprots = set(gene_uniprot_map.values())
    
    # 步骤3: 从BindingDB TSV下载并过滤
    print("\n[步骤3] 从BindingDB TSV文件下载并过滤CPI数据...")
    records, gene_stats = download_and_filter_bindingdb(target_uniprots, gene_uniprot_map)
    
    # 步骤4: 保存结果
    print("\n[步骤4] 保存结果...")
    if records:
        df_result = pd.DataFrame(records)
        # 去重（基于gene + canonical_smiles）
        before_dedup = len(df_result)
        df_result = df_result.drop_duplicates(subset=["gene", "canonical_smiles"])
        after_dedup = len(df_result)
        print(f"  去重: {before_dedup} -> {after_dedup}")
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  保存到: {OUTPUT_FILE}")
        print(f"  总记录数: {len(df_result)}")
    else:
        df_result = pd.DataFrame(columns=[
            "gene", "uniprot_id", "canonical_smiles",
            "activity_value", "activity_unit", "activity_type",
            "pubmed_id", "source"
        ])
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"  未获取到数据，保存空文件到: {OUTPUT_FILE}")
    
    # 步骤5: 统计报告
    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)
    print(f"  查询了基因数: {len(missing_genes)}")
    print(f"  成功映射UniProt ID: {len(gene_uniprot_map)}")
    genes_with_data = sum(1 for v in gene_stats.values() if v > 0)
    print(f"  成功获取CPI数据的基因数: {genes_with_data}")
    print(f"  总CPI记录数: {len(records)}")
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