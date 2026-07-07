#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

"""
CPI数据补充脚本 v27 - 聚焦版
只针对 drugbank_supplemental.csv 中的 ACSL4, SOD1, IGFBP7 三个基因
从ChEMBL和PubChem获取SMILES
"""
import pandas as pd
import os
import json
import urllib.request
import urllib.parse
import time
from rdkit import Chem

BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
MAIN_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
CHEMBL_FILE = os.path.join(BASE_DIR, "L4", "results", "chembl_active_compounds.csv")
DRUGBANK_SUPP_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "drugbank_supplemental.csv")
PREV_SUPP_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v25_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27.csv")
REPORT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27_report.txt")

def validate_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if smiles in ('nan', 'None', ''):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")
        return None

def get_smiles_from_pubchem(drug_name):
    time.sleep(0.3)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(drug_name)}/property/CanonicalSMILES/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties']
                if props:
                    # PubChem returns 'CanonicalSMILES' or 'ConnectivitySMILES'
                    smi = props[0].get('CanonicalSMILES') or props[0].get('ConnectivitySMILES')
                    if smi:
                        return smi
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")

    return None

print("=" * 70)
print("CPI数据补充脚本 v27 - 聚焦版 (ACSL4, SOD1, IGFBP7)")
print("=" * 70)

# 1. 读取96基因
genes_df = pd.read_csv(GENES_FILE)
genes_96 = set(genes_df['gene_symbol'].str.strip().str.upper())

# 2. 读取主CPI
main_cpi = pd.read_csv(MAIN_CPI_FILE, low_memory=False)
main_cpi['gene'] = main_cpi['gene'].str.strip().str.upper()
main_genes_with_smiles = set()
main_smiles_by_gene = {}
for _, row in main_cpi.iterrows():
    gene = row['gene']
    smiles = str(row.get('canonical_smiles', '')).strip()
    if smiles and smiles != 'nan' and smiles != 'None':
        main_genes_with_smiles.add(gene)
        if gene not in main_smiles_by_gene:
            main_smiles_by_gene[gene] = set()
        main_smiles_by_gene[gene].add(smiles)

# 3. 读取已有补充
prev_supp = pd.read_csv(PREV_SUPP_FILE, low_memory=False)
prev_supp['gene'] = prev_supp['gene'].str.strip().str.upper()
prev_smiles_by_gene = {}
for _, row in prev_supp.iterrows():
    gene = row['gene']
    smiles = str(row.get('smiles', '')).strip()
    if smiles and smiles != 'nan' and smiles != 'None':
        if gene not in prev_smiles_by_gene:
            prev_smiles_by_gene[gene] = set()
        prev_smiles_by_gene[gene].add(smiles)

# 4. 读取DrugBank补充
db_supp = pd.read_csv(DRUGBANK_SUPP_FILE, low_memory=False)
db_supp['gene'] = db_supp['gene'].str.strip().str.upper()
print(f"\nDrugBank补充数据: {len(db_supp)} 条记录, 基因: {sorted(db_supp['gene'].unique())}")

# 5. 读取ChEMBL数据建立SMILES索引
chembl = pd.read_csv(CHEMBL_FILE, low_memory=False)
chembl_smiles = {}
for _, row in chembl.iterrows():
    name = str(row.get('molecule_pref_name', '')).strip().upper()
    smiles = str(row.get('canonical_smiles', '')).strip()
    if name and name != 'NAN' and smiles and smiles != 'NAN':
        if name not in chembl_smiles:
            chembl_smiles[name] = smiles
print(f"ChEMBL化合物名称索引: {len(chembl_smiles)} 个")

# 6. 合并已有SMILES
existing_smiles = {}
for gene, sset in main_smiles_by_gene.items():
    existing_smiles[gene] = sset.copy()
for gene, sset in prev_smiles_by_gene.items():
    if gene not in existing_smiles:
        existing_smiles[gene] = set()
    existing_smiles[gene].update(sset)

# 7. 为每个DrugBank药物获取SMILES
print("\n获取SMILES...")
supplement_records = []
chembl_found = 0
pubchem_found = 0
not_found = 0

for _, row in db_supp.iterrows():
    gene = row['gene']
    db_id = str(row.get('drugbank_id', '')).strip()
    drug_name = str(row.get('drug_name', '')).strip()
    uniprot = str(row.get('uniprot_id', '')).strip()
    note = str(row.get('note', '')).strip()
    
    smiles = None
    source_detail = ''
    
    # 尝试1: ChEMBL
    drug_name_upper = drug_name.upper()
    if drug_name_upper in chembl_smiles:
        smiles = chembl_smiles[drug_name_upper]
        source_detail = 'ChEMBL'
        chembl_found += 1
    
    # 尝试2: PubChem
    if smiles is None:
        smiles = get_smiles_from_pubchem(drug_name)
        if smiles:
            source_detail = 'PubChem'
            pubchem_found += 1
        else:
            not_found += 1
    
    # 验证SMILES
    validated = validate_smiles(smiles) if smiles else None
    if validated is None:
        status = "SKIP(无SMILES)" if smiles is None else "SKIP(验证失败)"
        print(f"  [{status}] {gene}/{db_id}: {drug_name}")
        continue
    
    # 去重
    if gene in existing_smiles and validated in existing_smiles[gene]:
        print(f"  [DUP] {gene}/{db_id}: {drug_name}")
        continue
    
    if gene not in existing_smiles:
        existing_smiles[gene] = set()
    existing_smiles[gene].add(validated)
    
    record = {
        'gene': gene,
        'smiles': validated,
        'uniprot': uniprot,
        'source': 'DrugBank',
        'activity_value_nm': None,
        'activity_type': 'DrugBank_reference',
        'compound_name': drug_name,
        'target_name': '',
        'pmid': '',
        'doi': '',
        'drugbank_id': db_id,
        'note': note,
        'smiles_source': source_detail,
    }
    supplement_records.append(record)
    print(f"  [{source_detail}] {gene}/{db_id}: {drug_name}")

print(f"\nSMILES来源: ChEMBL={chembl_found}, PubChem={pubchem_found}, 未找到={not_found}")

# 8. 生成输出
print("\n生成cpi_supplement_v27.csv...")
if supplement_records:
    supp_df = pd.DataFrame(supplement_records)
    supp_df = supp_df.drop_duplicates(subset=['gene', 'smiles'])
    columns = ['gene', 'smiles', 'uniprot', 'source', 'activity_value_nm',
               'activity_type', 'compound_name', 'target_name', 'pmid', 'doi',
               'drugbank_id', 'note']
    columns = [c for c in columns if c in supp_df.columns]
    supp_df = supp_df[columns]
    supp_df.to_csv(OUTPUT_FILE, index=False)
    print(f"输出: {OUTPUT_FILE}")
    print(f"记录数: {len(supp_df)}, 基因数: {len(supp_df['gene'].unique())}")
else:
    supp_df = pd.DataFrame(columns=['gene','smiles','uniprot','source','activity_value_nm','activity_type','compound_name'])
    supp_df.to_csv(OUTPUT_FILE, index=False)
    print("警告: 无补充记录!")

# 9. 生成报告
report = []
report.append("=" * 70)
report.append("CPI数据补充报告 v27")
report.append("=" * 70)
report.append(f"生成时间: {pd.Timestamp.now()}")
report.append("")
report.append(f"主CPI中有SMILES的基因: {len(main_genes_with_smiles)}")
report.append(f"已有补充(v25)基因: {len(prev_smiles_by_gene)}")
report.append("")
report.append("--- 补充来源 ---")
report.append(f"DrugBank补充文件基因: ACSL4, SOD1, IGFBP7")
report.append(f"SMILES来源: ChEMBL交叉引用={chembl_found}, PubChem={pubchem_found}, 未找到={not_found}")
report.append("")

if supplement_records:
    report.append("--- 各基因补充情况 ---")
    gene_counts = supp_df.groupby('gene').size()
    for g, c in gene_counts.items():
        report.append(f"  {g}: {c} 条")
    report.append(f"\n总补充: {len(supp_df)} 条, 涉及 {len(gene_counts)} 个基因")
else:
    report.append("无补充记录")

report.append("")
report.append("--- 关键基因覆盖 ---")
for gene in ['ACSL4', 'MAPK14', 'MPO', 'NLRP3', 'DPP4', 'SOD1', 'IGFBP7']:
    in_main = gene in main_genes_with_smiles
    supp_cnt = len(supp_df[supp_df['gene'] == gene]) if supplement_records else 0
    status = '✓' if (in_main or supp_cnt > 0) else '✗'
    report.append(f"  {gene}: 主CPI={'✓' if in_main else '✗'} | 补充={supp_cnt}条 | {status}")

report.append("")
report.append("--- SMILES验证 ---")
if supplement_records:
    valid = sum(1 for _, r in supp_df.iterrows() if validate_smiles(r['smiles']))
    report.append(f"  有效: {valid}/{len(supp_df)}")

report.append("")
report.append("--- 仍缺失基因 ---")
supp_genes = set(supp_df['gene'].unique()) if supplement_records else set()
covered = main_genes_with_smiles | supp_genes
missing = genes_96 - covered
report.append(f"  缺失: {len(missing)} 个")
if missing:
    report.append(f"  列表: {sorted(missing)}")

report_text = '\n'.join(report)
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f"\n报告: {REPORT_FILE}")
print("完成!")