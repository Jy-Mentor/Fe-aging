#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

"""
CPI数据补充脚本 v27
从BindingDB和DrugBank数据中补充铁衰老96基因中缺失的CPI记录
"""
import pandas as pd
import os
from rdkit import Chem

# ========== 配置 ==========
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
GENES_FILE = os.path.join(BASE_DIR, "L1", "results", "ferroaging_genes_96.csv")
MAIN_CPI_FILE = os.path.join(BASE_DIR, "L4", "results", "experimental_actives_detail_cleaned.csv")
BINDINGDB_FILE = os.path.join(BASE_DIR, "L4", "results", "bindingdb_active_compounds_cleaned.csv")
DRUGBANK_FILE = os.path.join(BASE_DIR, "L4", "results", "drugbank_active_compounds.csv")
PREV_SUPPLEMENT_FILE = os.path.join(BASE_DIR, "L4", "results_v10_minibatch", "cpi_supplement_v25_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27.csv")
REPORT_FILE = os.path.join(BASE_DIR, "L4", "results", "cpi_supplement_v27_report.txt")

print("=" * 70)
print("CPI数据补充脚本 v27 - 从BindingDB和DrugBank补充缺失CPI数据")
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
print(f"    主CPI记录数: {len(main_cpi)}")
print(f"    主CPI列名: {list(main_cpi.columns)}")

# 主CPI中已有SMILES的基因
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
print(f"    主CPI中有SMILES的基因列表: {sorted(main_genes_with_smiles)}")

# 主CPI中所有出现的基因（含DrugBank来源无SMILES的）
main_genes_all = set(main_cpi['gene'].unique())
print(f"    主CPI中所有出现的基因: {len(main_genes_all)}")

# ========== 3. 识别缺失基因 ==========
print("\n[3] 识别缺失CPI的基因...")
# 完全不在主CPI中的基因
genes_missing_from_main = genes_96 - main_genes_all
print(f"    完全不在主CPI中的基因: {len(genes_missing_from_main)}")
print(f"    列表: {sorted(genes_missing_from_main)}")

# 在主CPI中但无SMILES的基因（仅DrugBank记录无化合物）
genes_no_smiles_in_main = main_genes_all - main_genes_with_smiles
print(f"    在主CPI中但无SMILES的基因: {len(genes_no_smiles_in_main)}")
print(f"    列表: {sorted(genes_no_smiles_in_main)}")

# 需要补充的基因 = 完全缺失 + 无SMILES
genes_need_supplement = genes_missing_from_main | genes_no_smiles_in_main
print(f"    需要补充的基因总数: {len(genes_need_supplement)}")

# ========== 4. 读取已有补充数据 ==========
print("\n[4] 读取已有CPI补充数据...")
prev_supp = pd.read_csv(PREV_SUPPLEMENT_FILE, low_memory=False)
prev_supp['gene'] = prev_supp['gene'].str.strip().str.upper()
print(f"    已有补充记录数: {len(prev_supp)}")
print(f"    已有补充列名: {list(prev_supp.columns)}")

# 已有补充中的SMILES
prev_smiles_by_gene = {}
for _, row in prev_supp.iterrows():
    gene = row['gene']
    smiles = str(row.get('smiles', '')).strip()
    if smiles and smiles != 'nan' and smiles != 'None':
        if gene not in prev_smiles_by_gene:
            prev_smiles_by_gene[gene] = set()
        prev_smiles_by_gene[gene].add(smiles)

prev_genes = set(prev_supp['gene'].unique())
print(f"    已有补充的基因: {len(prev_genes)}")
print(f"    列表: {sorted(prev_genes)}")

# 更新：已有补充覆盖的基因中，哪些实际上已有SMILES
supplemented_genes = set()
for gene in genes_need_supplement:
    main_smiles = main_smiles_by_gene.get(gene, set())
    prev_smiles = prev_smiles_by_gene.get(gene, set())
    if main_smiles or prev_smiles:
        supplemented_genes.add(gene)

# 仍然完全无SMILES的基因
genes_still_need = genes_need_supplement - supplemented_genes
print(f"    已有补充后仍完全无SMILES的基因: {len(genes_still_need)}")
print(f"    列表: {sorted(genes_still_need)}")

# 重新定义：所有需要补充的基因（已有补充的不再重复补充）
# 但我们需要检查BindingDB和DrugBank中是否有新数据
genes_to_query = genes_need_supplement  # 所有需要补充的基因
print(f"    本次查询的基因: {len(genes_to_query)}")

# ========== 5. 读取BindingDB数据 ==========
print("\n[5] 读取BindingDB清洗后数据...")
bdb = pd.read_csv(BINDINGDB_FILE, low_memory=False)
bdb['gene'] = bdb['gene'].str.strip().str.upper()
print(f"    BindingDB记录数: {len(bdb)}")
print(f"    BindingDB列名: {list(bdb.columns)}")
bdb_genes = set(bdb['gene'].unique())
print(f"    BindingDB中的基因数: {len(bdb_genes)}")

# 96基因中在BindingDB中的
bdb_in_96 = bdb_genes & genes_96
print(f"    96基因在BindingDB中的: {len(bdb_in_96)}")
print(f"    列表: {sorted(bdb_in_96)}")

# 需要补充且在BindingDB中的
bdb_to_use = bdb_in_96 & genes_to_query
print(f"    需要补充且在BindingDB中的: {len(bdb_to_use)}")
print(f"    列表: {sorted(bdb_to_use)}")

# 需要补充但不在BindingDB中的
bdb_missing = genes_to_query - bdb_in_96
print(f"    需要补充但不在BindingDB中的: {len(bdb_missing)}")
print(f"    列表: {sorted(bdb_missing)}")

# ========== 6. 读取DrugBank数据 ==========
print("\n[6] 读取DrugBank数据...")
db = pd.read_csv(DRUGBANK_FILE, low_memory=False)
db['gene'] = db['gene'].str.strip().str.upper()
print(f"    DrugBank记录数: {len(db)}")
print(f"    DrugBank列名: {list(db.columns)}")
db_genes = set(db['gene'].unique())
print(f"    DrugBank中的基因数: {len(db_genes)}")

# 96基因中在DrugBank中的
db_in_96 = db_genes & genes_96
print(f"    96基因在DrugBank中的: {len(db_in_96)}")
print(f"    列表: {sorted(db_in_96)}")

# 需要补充且在DrugBank中的
db_to_use = db_in_96 & genes_to_query
print(f"    需要补充且在DrugBank中的: {len(db_to_use)}")
print(f"    列表: {sorted(db_to_use)}")

# 需要补充但完全不在任何数据源中的
in_any_source = bdb_in_96 | db_in_96 | main_genes_with_smiles
genes_no_data = genes_96 - in_any_source
print(f"\n    96基因中完全无任何CPI数据源的: {len(genes_no_data)}")
print(f"    列表: {sorted(genes_no_data)}")

# ========== 7. 构建SMILES验证函数 ==========
def validate_smiles(smiles):
    """验证SMILES是否有效"""
    if not smiles or not isinstance(smiles, str):
        return None
    smiles = smiles.strip()
    if smiles in ('nan', 'None', ''):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 返回规范化的SMILES
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        logger.exception("捕获到异常并继续执行（原 except '' 静默吞掉）")
        return None

# ========== 8. 从BindingDB提取补充数据 ==========
print("\n[7] 从BindingDB提取补充数据...")

def get_all_smiles_set(main_by_gene, prev_by_gene):
    """获取所有已存在的SMILES（去重用）"""
    all_smiles = {}
    for gene, smiles_set in main_by_gene.items():
        if gene not in all_smiles:
            all_smiles[gene] = set()
        all_smiles[gene].update(smiles_set)
    for gene, smiles_set in prev_by_gene.items():
        if gene not in all_smiles:
            all_smiles[gene] = set()
        all_smiles[gene].update(smiles_set)
    return all_smiles

existing_smiles = get_all_smiles_set(main_smiles_by_gene, prev_smiles_by_gene)

supplement_records = []

# 从BindingDB提取
for gene in sorted(bdb_to_use):
    gene_data = bdb[bdb['gene'] == gene]
    existing = existing_smiles.get(gene, set())
    gene_count = 0
    
    for _, row in gene_data.iterrows():
        smiles = str(row.get('canonical_smiles', '')).strip()
        validated_smiles = validate_smiles(smiles)
        if validated_smiles is None:
            continue
        
        # 去重检查
        if validated_smiles in existing:
            continue
        
        existing.add(validated_smiles)
        
        # 获取活性值
        standard_value = row.get('standard_value_nM', None)
        try:
            activity_value_nm = float(standard_value) if pd.notna(standard_value) else None
        except (ValueError, TypeError):
            activity_value_nm = None
        
        record = {
            'gene': gene,
            'smiles': validated_smiles,
            'uniprot': str(row.get('uniprot_id', '')).strip() if pd.notna(row.get('uniprot_id')) else '',
            'source': 'BindingDB',
            'activity_value_nm': activity_value_nm,
            'activity_type': str(row.get('standard_type', '')).strip() if pd.notna(row.get('standard_type')) else '',
            'compound_name': str(row.get('molecule_name', '')).strip() if pd.notna(row.get('molecule_name')) else '',
            'target_name': str(row.get('target_name', '')).strip() if pd.notna(row.get('target_name')) else '',
            'pmid': str(row.get('pmid', '')).strip() if pd.notna(row.get('pmid')) else '',
            'doi': str(row.get('doi', '')).strip() if pd.notna(row.get('doi')) else '',
        }
        supplement_records.append(record)
        gene_count += 1
    
    print(f"    {gene}: BindingDB补充 {gene_count} 条")

# ========== 9. 从DrugBank提取补充数据 ==========
print("\n[8] 从DrugBank提取补充数据...")

# DrugBank数据没有SMILES！需要从DrugBank中获取SMILES
# 先检查主CPI数据中是否有DrugBank来源的SMILES可以参考
# 如果有drugbank_id，可以从ChEMBL/DrugBank交叉引用中获取SMILES

# 策略：检查DrugBank记录，看主CPI中是否有对应drugbank_id的SMILES
main_drugbank_smiles = {}
for _, row in main_cpi.iterrows():
    db_id = str(row.get('drugbank_id', '')).strip()
    smiles = str(row.get('canonical_smiles', '')).strip()
    if db_id and db_id != 'nan' and smiles and smiles != 'nan':
        main_drugbank_smiles[db_id] = validate_smiles(smiles)

print(f"    主CPI中DrugBank来源的SMILES数: {len(main_drugbank_smiles)}")

# 对于DrugBank数据，需要检查是否有SMILES
# DrugBank文件中没有SMILES列，所以只能记录药物名称但无法提供SMILES
# 除非我们能从其他来源获取SMILES

# 检查DrugBank文件中是否有smiles列
drugbank_has_smiles = 'canonical_smiles' in db.columns or 'smiles' in db.columns
print(f"    DrugBank文件有SMILES列: {drugbank_has_smiles}")

if drugbank_has_smiles:
    smiles_col = 'canonical_smiles' if 'canonical_smiles' in db.columns else 'smiles'
else:
    print("    DrugBank文件无SMILES列，仅记录药物名称信息")
    smiles_col = None

for gene in sorted(db_to_use):
    gene_data = db[db['gene'] == gene]
    existing = existing_smiles.get(gene, set())
    gene_count = 0
    
    for _, row in gene_data.iterrows():
        db_id = str(row.get('drugbank_id', '')).strip() if pd.notna(row.get('drugbank_id')) else ''
        drug_name = str(row.get('drug_name', '')).strip() if pd.notna(row.get('drug_name')) else ''
        
        # 尝试获取SMILES
        smiles = None
        if smiles_col:
            smiles = str(row.get(smiles_col, '')).strip()
        
        # 如果DrugBank文件没有SMILES，尝试从主CPI中查找
        if not smiles or smiles == 'nan':
            if db_id in main_drugbank_smiles:
                smiles = main_drugbank_smiles[db_id]
        
        validated_smiles = validate_smiles(smiles) if smiles else None
        
        if validated_smiles is None:
            # 没有SMILES，记录药物名称但跳过
            continue
        
        # 去重检查
        if validated_smiles in existing:
            continue
        
        existing.add(validated_smiles)
        
        record = {
            'gene': gene,
            'smiles': validated_smiles,
            'uniprot': str(row.get('uniprot_id', '')).strip() if pd.notna(row.get('uniprot_id')) else '',
            'source': 'DrugBank',
            'activity_value_nm': None,  # DrugBank通常没有活性值
            'activity_type': '',
            'compound_name': drug_name,
            'target_name': '',
            'pmid': '',
            'doi': '',
            'drugbank_id': db_id,
            'note': str(row.get('note', '')).strip() if pd.notna(row.get('note')) else '',
        }
        supplement_records.append(record)
        gene_count += 1
    
    if gene_count > 0:
        print(f"    {gene}: DrugBank补充 {gene_count} 条")

# ========== 10. 生成输出 ==========
print("\n[9] 生成补充CPI数据文件...")

if supplement_records:
    supp_df = pd.DataFrame(supplement_records)
    # 确保列顺序
    columns = ['gene', 'smiles', 'uniprot', 'source', 'activity_value_nm',
               'activity_type', 'compound_name', 'target_name', 'pmid', 'doi']
    # 只保留存在的列
    columns = [c for c in columns if c in supp_df.columns]
    supp_df = supp_df[columns]
    
    supp_df.to_csv(OUTPUT_FILE, index=False)
    print(f"    输出文件: {OUTPUT_FILE}")
    print(f"    总补充记录数: {len(supp_df)}")
else:
    print("    警告: 没有找到任何补充记录!")
    supp_df = pd.DataFrame(columns=['gene', 'smiles', 'uniprot', 'source', 'activity_value_nm',
                                     'activity_type', 'compound_name', 'target_name', 'pmid', 'doi'])
    supp_df.to_csv(OUTPUT_FILE, index=False)

# ========== 11. 生成补充报告 ==========
print("\n[10] 生成补充报告...")

report_lines = []
report_lines.append("=" * 70)
report_lines.append("CPI数据补充报告 v27")
report_lines.append("=" * 70)
report_lines.append(f"生成时间: {pd.Timestamp.now()}")
report_lines.append("")

report_lines.append(f"铁衰老96基因总数: {len(genes_96)}")
report_lines.append(f"主CPI中有SMILES的基因: {len(main_genes_with_smiles)}")
report_lines.append(f"主CPI中无SMILES的基因: {len(genes_no_smiles_in_main)}")
report_lines.append(f"完全不在主CPI中的基因: {len(genes_missing_from_main)}")
report_lines.append(f"已有补充数据(v25)的基因: {len(prev_genes)}")
report_lines.append("")

report_lines.append("=" * 70)
report_lines.append("各基因补充情况")
report_lines.append("=" * 70)

if supplement_records:
    # 按基因统计
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
report_lines.append("数据来源统计")
report_lines.append("=" * 70)
if supplement_records:
    source_counts = supp_df['source'].value_counts()
    for source, count in source_counts.items():
        report_lines.append(f"  {source}: {count} 条")
    
    # BindingDB来源统计
    bdb_supp = supp_df[supp_df['source'] == 'BindingDB']
    if len(bdb_supp) > 0:
        report_lines.append(f"\n  BindingDB补充基因: {len(bdb_supp['gene'].unique())}")
        report_lines.append(f"  BindingDB补充记录: {len(bdb_supp)}")
        for gene in sorted(bdb_supp['gene'].unique()):
            cnt = len(bdb_supp[bdb_supp['gene'] == gene])
            report_lines.append(f"    {gene}: {cnt} 条")
    
    # DrugBank来源统计
    db_supp = supp_df[supp_df['source'] == 'DrugBank']
    if len(db_supp) > 0:
        report_lines.append(f"\n  DrugBank补充基因: {len(db_supp['gene'].unique())}")
        report_lines.append(f"  DrugBank补充记录: {len(db_supp)}")
        for gene in sorted(db_supp['gene'].unique()):
            cnt = len(db_supp[db_supp['gene'] == gene])
            report_lines.append(f"    {gene}: {cnt} 条")
else:
    report_lines.append("  无补充数据")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("关键铁衰老基因覆盖情况")
report_lines.append("=" * 70)
key_genes = ['ACSL4', 'MAPK14', 'MPO', 'NLRP3', 'DPP4', 'SOD1', 'IGFBP7']
for gene in key_genes:
    in_main = gene in main_genes_with_smiles
    in_bdb = gene in bdb_in_96
    in_db = gene in db_in_96
    in_supp = False
    supp_count = 0
    if supplement_records:
        supp_count = len(supp_df[supp_df['gene'] == gene])
        in_supp = supp_count > 0
    report_lines.append(f"  {gene}: 主CPI={'✓' if in_main else '✗'} | BindingDB={'✓' if in_bdb else '✗'} | DrugBank={'✓' if in_db else '✗'} | 本次补充={supp_count}条")

report_lines.append("")
report_lines.append("=" * 70)
report_lines.append("完全无任何CPI数据源的基因")
report_lines.append("=" * 70)
if genes_no_data:
    report_lines.append(f"  {len(genes_no_data)} 个基因:")
    for gene in sorted(genes_no_data):
        report_lines.append(f"    - {gene}")
else:
    report_lines.append("  (无) 所有基因至少有一个数据源")

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
    report_lines.append(f"  SMILES验证通过率: {valid_count/len(supp_df)*100:.1f}%")

report_text = '\n'.join(report_lines)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print(f"\n报告已保存到: {REPORT_FILE}")
print("\n完成任务!")