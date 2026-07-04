"""
交叉引用 dhimmel/bindingdb 数据与铁衰老缺失基因
"""
import gzip
import pandas as pd

# 读取 binding.tsv.gz
with gzip.open(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\dhimmel_bindingdb\data\binding.tsv.gz', 'rt') as f:
    df = pd.read_csv(f, sep='\t', low_memory=False)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# 缺失基因的UniProt映射
gene_uniprot = {
    'ATF3': 'P18847', 'ATG3': 'Q9NT62', 'CAVIN1': 'Q6NZI2', 'CD82': 'P27701',
    'CDO1': 'Q16878', 'COX7A1': 'P24310', 'E2F1': 'Q01094', 'E2F3': 'O00716',
    'EBF3': 'Q9H4W6', 'EDN1': 'P05305', 'EGR1': 'P18146', 'EMP1': 'P54849',
    'FBXO31': 'Q5XUX0', 'FOSL1': 'P15407', 'GMFB': 'P60983', 'HBP1': 'O60381',
    'HERPUD1': 'Q15011', 'ICA1': 'Q05084', 'IGFBP7': 'Q16270', 'IRF1': 'P10914',
    'IRF7': 'Q92985', 'IRF9': 'Q00978', 'KLF6': 'Q99612', 'LACTB': 'P83111',
    'PPP2R2B': 'Q00005', 'RUNX3': 'Q13761', 'SLAMF8': 'Q9P0V8', 'SOCS1': 'O15524',
    'SOCS2': 'O14508', 'SPATA2': 'Q9UM82', 'TBX2': 'Q13207', 'TNFAIP1': 'Q13829',
    'TNFAIP3': 'P21580', 'TXNIP': 'Q9H3M7', 'WNT5A': 'P41221', 'WWTR1': 'Q9GZV5',
    'ZEB1': 'P37275'
}

all_uniprots = set(df['uniprot'].dropna().str.upper())
missing_uniprots = set(gene_uniprot.values())
hits = all_uniprots & missing_uniprots

print(f"\nTotal unique UniProt IDs in binding.tsv.gz: {len(all_uniprots)}")
print(f"Missing gene UniProt hits: {len(hits)}")

# 反向查找并显示详细信息
for uniprot in sorted(hits):
    for gene, up in gene_uniprot.items():
        if up == uniprot:
            subset = df[df['uniprot'].str.upper() == uniprot]
            print(f"\n{gene} ({uniprot}): {len(subset)} CPI records")
            measures = subset['measure'].value_counts().to_dict()
            print(f"  Measures: {measures}")
            print(f"  Affinity range: {subset['affinity_nM'].min():.1f} - {subset['affinity_nM'].max():.1f} nM")
            print(f"  Sources: {subset['source'].value_counts().to_dict()}")
            break

if not hits:
    print("\nNo hits found in binding.tsv.gz - checking the DrugBank-collapsed file instead...")

# 也检查 DrugBank-collapsed 文件
df2 = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\dhimmel_bindingdb\data\bindings-drugbank-gene.tsv', sep='\t')
all_genes2 = set(df2['gene_symbol'].dropna().str.upper())
missing_upper = set(g.upper() for g in gene_uniprot.keys())
hits2 = all_genes2 & missing_upper
print(f"\nDrugBank-collapsed hits: {hits2}")

# 也检查 bindings-drugbank-collapsed.tsv
df3 = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\dhimmel_bindingdb\data\bindings-drugbank-collapsed.tsv', sep='\t')
print(f"\nCollapsed file columns: {list(df3.columns)}")
if 'gene_symbol' in df3.columns:
    all_genes3 = set(df3['gene_symbol'].dropna().str.upper())
    hits3 = all_genes3 & missing_upper
    print(f"Collapsed hits: {hits3}")