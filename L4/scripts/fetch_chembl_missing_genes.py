import logging
logger = logging.getLogger(__name__)

"""
通过 ChEMBL API 查询铁衰老缺失基因的化合物活性数据
"""
import pandas as pd
import time
from pathlib import Path
from chembl_webresource_client.new_client import new_client

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 铁衰老全部核心基因
CORE_GENES = [
    "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3",
    "BAP1", "BCL6", "BRD7",
    "CAVIN1", "CD74", "CD82", "CDO1", "COX7A1", "CTSB", "CXCL10",
    "DPEP1", "DPP4", "DUOX1", "DYRK1A",
    "E2F1", "E2F3", "EBF3", "EDN1", "EGR1", "EMP1", "EPHA2", "EPHA4", "ERN1",
    "FBXO31", "FOSL1",
    "GMFB",
    "HBP1", "HERPUD1", "HIF1A", "HMGB1", "HMOX1",
    "ICA1", "IFNG", "IGFBP7", "IL1B", "IL6", "IRF1", "IRF7", "IRF9",
    "KDM6B", "KEAP1", "KLF6",
    "LACTB", "LCN2", "LGMN", "LIFR", "LOX", "LPCAT3",
    "MAP3K14", "MAPK1", "MAPK14", "MCU", "MEN1", "MPO",
    "NLRP3", "NOX4", "NR1D1", "NR2F2", "NUAK2",
    "PADI4", "PDE4B", "PPP2R2B", "PRKD1", "PTBP1", "PTGS2",
    "RBM3", "RUNX3",
    "S100A8", "SAT1", "SETD7", "SLAMF8", "SLC1A5", "SMARCB1", "SMURF2", "SNCA",
    "SOCS1", "SOCS2", "SOD1", "SP1", "SPATA2",
    "TBX2", "TFRC", "TLR4", "TNFAIP1", "TNFAIP3", "TXNIP",
    "WNT5A", "WWTR1",
    "YAP1",
    "ZEB1",
]

# 读取已有 combined CPI 数据，获取已覆盖基因
combined_path = PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned_combined.csv"
if combined_path.exists():
    combined = pd.read_csv(combined_path)
    gene_col = [c for c in combined.columns if 'gene' in c.lower()][0] if any('gene' in c.lower() for c in combined.columns) else None
    existing_genes = set(combined[gene_col].dropna().str.upper()) if gene_col else set()
else:
    existing_genes = set()

missing_genes = [g for g in CORE_GENES if g.upper() not in existing_genes]
print(f"总基因数: {len(CORE_GENES)}")
print(f"已有数据: {len(existing_genes)}")
print(f"需查询: {len(missing_genes)}")
print(f"缺失基因: {missing_genes}")

# ChEMBL API 客户端
target_client = new_client.target
activity_client = new_client.activity
molecule_client = new_client.molecule

# Gene symbol to UniProt mapping (fallback)
GENE_UNIPROT = {
    'ATF3': 'P18847', 'ATG3': 'Q9NT62', 'CAVIN1': 'Q6NZI2', 'CD82': 'P27701',
    'CDO1': 'Q16878', 'COX7A1': 'P24310', 'E2F1': 'Q01094', 'E2F3': 'O00716',
    'EBF3': 'Q9H4W6', 'EDN1': 'P05305', 'EGR1': 'P18146', 'EMP1': 'P54849',
    'FBXO31': 'Q5XUX0', 'FOSL1': 'P15407', 'GMFB': 'P60983', 'HBP1': 'O60381',
    'HERPUD1': 'Q15011', 'ICA1': 'Q05084', 'IGFBP7': 'Q16270', 'IRF1': 'P10914',
    'IRF7': 'Q92985', 'IRF9': 'Q00978', 'KLF6': 'Q99612', 'LACTB': 'P83111',
    'PPP2R2B': 'Q00005', 'RUNX3': 'Q13761', 'SLAMF8': 'Q9P0V8', 'SOCS1': 'O15524',
    'SOCS2': 'O14508', 'SPATA2': 'Q9UM82', 'TBX2': 'Q13207', 'TNFAIP1': 'Q13829',
    'TNFAIP3': 'P21580', 'TXNIP': 'Q9H3M7', 'WNT5A': 'P41221', 'WWTR1': 'Q9GZV5',
    'ZEB1': 'P37275',
    'ABCC1': 'P33527', 'ACVR1B': 'P36896', 'ACSL4': 'O60488', 'ALOX15': 'P16050',
    'BAP1': 'Q92560', 'BCL6': 'P41182', 'BRD7': 'Q9NPI1',
    'CD74': 'P04233', 'CTSB': 'P07858', 'CXCL10': 'P02778',
    'DPEP1': 'P16444', 'DPP4': 'P27487', 'DUOX1': 'Q9NRD8', 'DYRK1A': 'Q13627',
    'EPHA2': 'P29317', 'EPHA4': 'P54764', 'ERN1': 'O75460',
    'HIF1A': 'Q16665', 'HMGB1': 'P09429', 'HMOX1': 'P09601',
    'IFNG': 'P01579', 'IL1B': 'P01584', 'IL6': 'P05231',
    'KDM6B': 'O15054', 'KEAP1': 'Q14145',
    'LCN2': 'P80188', 'LGMN': 'Q99538', 'LIFR': 'P42702', 'LOX': 'P28300', 'LPCAT3': 'Q6P1A2',
    'MAP3K14': 'Q99558', 'MAPK1': 'P28482', 'MAPK14': 'Q16539', 'MCU': 'Q8NE86', 'MEN1': 'O00255', 'MPO': 'P05164',
    'NLRP3': 'Q96P20', 'NOX4': 'Q9NPH5', 'NR1D1': 'P20393', 'NR2F2': 'P24468', 'NUAK2': 'Q9H093',
    'PADI4': 'Q9UM07', 'PDE4B': 'Q07343', 'PRKD1': 'Q15139', 'PTBP1': 'P26599', 'PTGS2': 'P35354',
    'RBM3': 'P98179', 'S100A8': 'P05109', 'SAT1': 'P21673', 'SETD7': 'Q8WTS6', 'SLC1A5': 'Q15758',
    'SMARCB1': 'Q12824', 'SMURF2': 'Q9HAU4', 'SNCA': 'P37840', 'SOD1': 'P00441', 'SP1': 'P08047',
    'TFRC': 'P02786', 'TLR4': 'O00206',
    'YAP1': 'P46937',
}

results = []

for gene in missing_genes:
    print(f"\n查询 {gene}...")
    try:
        # 通过 gene symbol 搜索 ChEMBL target
        targets = target_client.filter(pref_name__iexact=gene).only(['target_chembl_id', 'pref_name', 'organism'])
        target_list = list(targets)
        
        if len(target_list) == 0:
            # 尝试通过 UniProt 搜索
            uniprot = GENE_UNIPROT.get(gene)
            if uniprot:
                targets = target_client.filter(target_components__accession=uniprot).only(['target_chembl_id', 'pref_name', 'organism'])
                target_list = list(targets)
        
        if len(target_list) == 0:
            print(f"  {gene}: 未找到ChEMBL靶标")
            continue
        
        # 筛选人类靶标
        human_targets = [t for t in target_list if t.get('organism') == 'Homo sapiens']
        if not human_targets:
            human_targets = target_list[:3]  # 取前3个
        
        for target in human_targets[:3]:  # 最多查3个靶标
            chembl_id = target['target_chembl_id']
            print(f"  靶标: {target['pref_name']} ({chembl_id})")
            
            # 查询活性数据
            activities = activity_client.filter(
                target_chembl_id=chembl_id,
                standard_type__in=['IC50', 'Ki', 'Kd'],
                standard_relation__in=['=', '<', '<='],
                standard_units='nM'
            ).only([
                'molecule_chembl_id', 'standard_type', 'standard_value',
                'standard_relation', 'canonical_smiles', 'assay_chembl_id',
                'document_chembl_id'
            ])
            
            count = 0
            for act in activities:
                if act.get('canonical_smiles') and act.get('standard_value'):
                    results.append({
                        'gene': gene,
                        'target_chembl_id': chembl_id,
                        'target_name': target.get('pref_name', ''),
                        'molecule_chembl_id': act.get('molecule_chembl_id', ''),
                        'canonical_smiles': act.get('canonical_smiles', ''),
                        'standard_type': act.get('standard_type', ''),
                        'standard_value_nM': act.get('standard_value', ''),
                        'standard_relation': act.get('standard_relation', ''),
                        'source': 'ChEMBL_API',
                    })
                    count += 1
            print(f"    活性记录: {count}")
            time.sleep(0.5)  # API rate limiting
        
        time.sleep(1)
        
    except Exception as e:
        print(f"  {gene}: 查询失败 - {e}")
        time.sleep(2)

# 保存结果
if results:
    df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "L4" / "data" / "github_sources" / "chembl_missing_genes.csv"
    df.to_csv(out_path, index=False)
    print("\n=== 查询完成 ===")
    print(f"总记录数: {len(df)}")
    print(f"覆盖基因: {df['gene'].nunique()}")
    print(f"基因列表: {sorted(df['gene'].unique())}")
    print(f"已保存到: {out_path}")
else:
    print("\n未找到任何活性数据")