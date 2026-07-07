#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

"""
CPI数据补充脚本 v27 - 最终版
1. 从ChEMBL数据中交叉引用DrugBank药物的SMILES
2. 从PubChem REST API获取剩余DrugBank药物的SMILES
3. 所有SMILES通过RDKit验证
4. 去重并生成cpi_supplement_v27.csv
"""
import pandas as pd
import os
import sys
import json
import urllib.request
import urllib.error
import urllib.parse
import time
from rdkit import Chem

# ========== 配置 ==========
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
MAIN_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
BINDINGDB_FILE = os.path.join(BASE_DIR, "L4", "results", "bindingdb_active_compounds_cleaned.csv")
CHEMBL_FILE = os.path.join(BASE_DIR, "L4", "results", "chembl_active_compounds.csv")
DRUGBANK_SUPP_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "drugbank_supplemental.csv")
PREV_SUPPLEMENT_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v25_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27.csv")
REPORT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27_report.txt")

print("=" * 70)
print("CPI数据补充脚本 v27 - 最终版")
print("=" * 70)

# ========== 1. 读取铁衰老96基因 ==========
print("\n[1] 读取铁衰老96基因列表...")
genes_df = pd.read_csv(GENES_FILE)
genes_96 = set(genes_df['gene_symbol'].str.strip().str.upper())
print(f"    铁衰老96基因: {len(genes_96)}")

# ========== 2. 读取主CPI数据 ==========
print("\n[2] 读取主CPI数据...")
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
print(f"    主CPI中有SMILES的基因: {len(main_genes_with_smiles)}")

# ========== 3. 读取已有补充数据 ==========
print("\n[3] 读取已有CPI补充数据...")
prev_supp = pd.read_csv(PREV_SUPPLEMENT_FILE, low_memory=False)
prev_supp['gene'] = prev_supp['gene'].str.strip().str.upper()
prev_smiles_by_gene = {}
for _, row in prev_supp.iterrows():
    gene = row['gene']
    smiles = str(row.get('smiles', '')).strip()
    if smiles and smiles != 'nan' and smiles != 'None':
        if gene not in prev_smiles_by_gene:
            prev_smiles_by_gene[gene] = set()
        prev_smiles_by_gene[gene].add(smiles)
print(f"    已有补充记录数: {len(prev_supp)}")

# ========== 4. 读取DrugBank补充数据 ==========
print("\n[4] 读取DrugBank补充数据...")
db_supp = pd.read_csv(DRUGBANK_SUPP_FILE, low_memory=False)
db_supp['gene'] = db_supp['gene'].str.strip().str.upper()
print(f"    DrugBank补充记录数: {len(db_supp)}")
print(f"    DrugBank补充基因: {sorted(db_supp['gene'].unique())}")

# ========== 5. 读取ChEMBL数据用于交叉引用 ==========
print("\n[5] 读取ChEMBL数据用于交叉引用SMILES...")
chembl = pd.read_csv(CHEMBL_FILE, low_memory=False)
print(f"    ChEMBL记录数: {len(chembl)}")

# 构建ChEMBL中所有drug_name到SMILES的映射（从full main CPI更准确）
main_cpi_full = pd.read_csv(r"d:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail.csv", low_memory=False)
# 从main_cpi_full中提取所有DrugBank记录的drug_name和对应的ChEMBL SMILES
drugbank_smiles_map = {}
# 首先从ChEMBL数据中获取所有SMILES（按drug_name索引）
# ChEMBL数据中的molecule_pref_name可能包含drug名称
print(f"    构建ChEMBL SMILES索引...")

# 从ChEMBL获取所有唯一化合物SMILES（按molecule_pref_name）
chembl_smiles = {}
for _, row in chembl.iterrows():
    name = str(row.get('molecule_pref_name', '')).strip().upper()
    smiles = str(row.get('canonical_smiles', '')).strip()
    if name and name != 'NAN' and smiles and smiles != 'NAN':
        if name not in chembl_smiles:
            chembl_smiles[name] = smiles

print(f"    ChEMBL中唯一化合物名称: {len(chembl_smiles)}")

# 从完整主CPI中获取DrugBank记录的drug_name
for _, row in main_cpi_full.iterrows():
    source = str(row.get('source', '')).strip()
    if source == 'DrugBank':
        drug_name = str(row.get('drug_name', '')).strip()
        db_id = str(row.get('drugbank_id', '')).strip()
        if drug_name and drug_name != 'nan':
            if db_id and db_id != 'nan':
                drugbank_smiles_map[db_id] = {'name': drug_name, 'smiles': None}

# ========== 6. SMILES验证函数 ==========
def validate_smiles(smiles):
    """验证SMILES是否有效，返回规范化SMILES"""
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
        logger.exception("捕获到异常并继续执行（原 except 'Exception' 静默吞掉）")
        return None

# ========== 7. 从PubChem获取SMILES ==========
def get_smiles_from_pubchem(drug_name):
    """通过PubChem REST API获取化合物的规范SMILES"""
    time.sleep(0.3)  # Rate limiting
    try:
        # 搜索化合物
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(drug_name)}/property/CanonicalSMILES/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties']
                if props and 'CanonicalSMILES' in props[0]:
                    return props[0]['CanonicalSMILES']
    except Exception as e:
        logger.exception("捕获到异常并继续执行（原 except 'Exception as e' 静默吞掉）")
        pass

    return None

# ========== 8. 尝试从ChEMBL和PubChem获取SMILES ==========
print("\n[6] 为DrugBank药物获取SMILES...")

# 首先尝试从ChEMBL查找
for db_id, info in drugbank_smiles_map.items():
    drug_name_upper = info['name'].upper()
    if drug_name_upper in chembl_smiles:
        info['smiles'] = chembl_smiles[drug_name_upper]
        print(f"    [ChEMBL] {db_id} ({info['name']}): SMILES已找到")

# 统计ChEMBL找到的
chembl_found = sum(1 for v in drugbank_smiles_map.values() if v['smiles'])

# 对ChEMBL未找到的，尝试PubChem
print(f"\n    ChEMBL找到: {chembl_found}/{len(drugbank_smiles_map)}")
print(f"    尝试PubChem查找剩余药物...")

for db_id, info in drugbank_smiles_map.items():
    if info['smiles'] is None:
        smiles = get_smiles_from_pubchem(info['name'])
        if smiles:
            validated = validate_smiles(smiles)
            if validated:
                info['smiles'] = validated
                print(f"    [PubChem] {db_id} ({info['name']}): SMILES已找到")
            else:
                print(f"    [PubChem] {db_id} ({info['name']}): SMILES无效")
        else:
            print(f"    [FAIL] {db_id} ({info['name']}): 未找到SMILES")

pubchem_found = sum(1 for v in drugbank_smiles_map.values() if v['smiles']) - chembl_found
total_found = sum(1 for v in drugbank_smiles_map.values() if v['smiles'])
print(f"\n    PubChem找到: {pubchem_found}")
print(f"    总计找到SMILES: {total_found}/{len(drugbank_smiles_map)}")

# ========== 9. 构建补充记录 ==========
print("\n[7] 构建补充CPI记录...")

# 合并已有SMILES集合（去重用）
existing_smiles = {}
for gene, smiles_set in main_smiles_by_gene.items():
    existing_smiles[gene] = smiles_set.copy()
for gene, smiles_set in prev_smiles_by_gene.items():
    if gene not in existing_smiles:
        existing_smiles[gene] = set()
    existing_smiles[gene].update(smiles_set)

supplement_records = []

# 为每个DrugBank补充基因构建记录
for _, row in db_supp.iterrows():
    gene = row['gene']
    db_id = str(row.get('drugbank_id', '')).strip()
    drug_name = str(row.get('drug_name', '')).strip()
    uniprot = str(row.get('uniprot_id', '')).strip()
    note = str(row.get('note', '')).strip()
    
    if db_id not in drugbank_smiles_map:
        continue
    
    smiles = drugbank_smiles_map[db_id]['smiles']
    if smiles is None:
        continue
    
    validated_smiles = validate_smiles(smiles)
    if validated_smiles is None:
        print(f"    [SKIP] {gene} / {db_id}: SMILES验证失败")
        continue
    
    # 去重检查
    if gene in existing_smiles and validated_smiles in existing_smiles[gene]:
        continue
    
    if gene not in existing_smiles:
        existing_smiles[gene] = set()
    existing_smiles[gene].add(validated_smiles)
    
    record = {
        'gene': gene,
        'smiles': validated_smiles,
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
    }
    supplement_records.append(record)

# ========== 10. 生成输出 ==========
print("\n[8] 生成cpi_supplement_v27.csv...")

if supplement_records:
    supp_df = pd.DataFrame(supplement_records)
    # 确保列顺序
    columns = ['gene', 'smiles', 'uniprot', 'source', 'activity_value_nm',
               'activity_type', 'compound_name', 'target_name', 'pmid', 'doi',
               'drugbank_id', 'note']
    columns = [c for c in columns if c in supp_df.columns]
    supp_df = supp_df[columns]
    
    # 最终去重：gene + smiles
    supp_df = supp_df.drop_duplicates(subset=['gene', 'smiles'])
    
    supp_df.to_csv(OUTPUT_FILE, index=False)
    print(f"    输出文件: {OUTPUT_FILE}")
    print(f"    总补充记录数: {len(supp_df)}")
    print(f"    涉及基因数: {len(supp_df['gene'].unique())}")
else:
    print("    警告: 没有找到任何补充记录!")
    supp_df = pd.DataFrame(columns=['gene', 'smiles', 'uniprot', 'source', 'activity_value_nm',
                                     'activity_type', 'compound_name', 'target_name', 'pmid', 'doi'])
    supp_df.to_csv(OUTPUT_FILE, index=False)

# ========== 11. 生成补充报告 ==========
print("\n[9] 生成补充报告...")

report_lines = []
report_lines.append("=" * 70)
report_lines.append("CPI数据补充报告 v27")
report_lines.append("=" * 70)
report_lines.append(f"生成时间: {pd.Timestamp.now()}")
report_lines.append("")

report_lines.append(f"铁衰老96基因总数: {len(genes_96)}")
report_lines.append(f"主CPI中有SMILES的基因: {len(main_genes_with_smiles)}")
report_lines.append(f"BindingDB中96基因覆盖: 20个(全在已有CPI中)")
report_lines.append(f"DrugBank中96基因覆盖: 21个")
report_lines.append(f"DrugBank补充文件中基因: {sorted(db_supp['gene'].unique())}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("SMILES来源统计")
report_lines.append("=" * 70)
report_lines.append(f"  从ChEMBL交叉引用: {chembl_found} 个药物")
report_lines.append(f"  从PubChem获取: {pubchem_found} 个药物")
report_lines.append(f"  未找到SMILES: {len(drugbank_smiles_map) - total_found} 个药物")
report_lines.append("")

# 列出未找到SMILES的药物
not_found = [(db_id, info['name']) for db_id, info in drugbank_smiles_map.items() if info['smiles'] is None]
if not_found:
    report_lines.append("未找到SMILES的药物:")
    for db_id, name in not_found:
        report_lines.append(f"  - {db_id}: {name}")
    report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("各基因补充情况")
report_lines.append("=" * 70)

if supplement_records:
    gene_counts = supp_df.groupby('gene').size().sort_values(ascending=False)
    report_lines.append(f"{'基因':<12} {'补充记录数':<12} {'数据来源'}")
    report_lines.append("-" * 50)
    for gene, count in gene_counts.items():
        sources = supp_df[supp_df['gene'] == gene]['source'].unique()
        source_str = ', '.join(sources)
        report_lines.append(f"{gene:<12} {count:<12} {source_str}")
    
    report_lines.append("")
    report_lines.append(f"总补充记录数: {len(supp_df)}")
    report_lines.append(f"涉及基因数: {len(gene_counts)}")
else:
    report_lines.append("无补充记录")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("关键铁衰老基因覆盖情况")
report_lines.append("=" * 70)
key_genes = ['ACSL4', 'MAPK14', 'MPO', 'NLRP3', 'DPP4', 'SOD1', 'IGFBP7']
for gene in key_genes:
    in_main = gene in main_genes_with_smiles
    supp_count = 0
    if supplement_records:
        supp_count = len(supp_df[supp_df['gene'] == gene])
    in_supp = supp_count > 0
    report_lines.append(f"  {gene}: 主CPI={'✓' if in_main else '✗'} | 本次补充={supp_count}条 | 状态={'✓ 有CPI数据' if (in_main or in_supp) else '✗ 仍缺失'}")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("SMILES验证统计")
report_lines.append("=" * 70)
if supplement_records:
    valid_count = 0
    invalid_count = 0
    for _, row in supp_df.iterrows():
        if validate_smiles(row['smiles']):
            valid_count += 1
        else:
            invalid_count += 1
    report_lines.append(f"  有效SMILES: {valid_count}")
    report_lines.append(f"  无效SMILES: {invalid_count}")
    if len(supp_df) > 0:
        report_lines.append(f"  SMILES验证通过率: {valid_count/len(supp_df)*100:.1f}%")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("仍缺失CPI数据的基因 (96 - 已有CPI - 本次补充)")
report_lines.append("=" * 70)
supplemented_genes = set(supp_df['gene'].unique()) if supplement_records else set()
all_covered = main_genes_with_smiles | supplemented_genes
still_missing = genes_96 - all_covered
report_lines.append(f"  仍缺失: {len(still_missing)} 个基因")
if still_missing:
    report_lines.append(f"  列表: {sorted(still_missing)}")
else:
    report_lines.append("  所有基因已覆盖!")

report_text = '\n'.join(report_lines)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f"\n报告已保存到: {REPORT_FILE}")
print("\n完成任务!")