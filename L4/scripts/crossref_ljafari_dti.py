"""交叉引用 ljafari DTI dataset 与铁衰老缺失基因"""
import pandas as pd

dti = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\ljafari_dti\dataset\DTI.csv')
dpi = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\ljafari_dti\dataset\DPI.csv')

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

# Check DTI.csv
all_uniprots = set(dti['UniprotID'].dropna().str.upper())
missing_uniprots = set(gene_uniprot.values())
hits = all_uniprots & missing_uniprots

print(f"DTI.csv unique UniProt IDs: {len(all_uniprots)}")
print(f"Missing gene hits in DTI.csv: {hits}")
for up in sorted(hits):
    for gene, u in gene_uniprot.items():
        if u == up:
            subset = dti[dti['UniprotID'].str.upper() == up]
            print(f"  {gene} ({up}): {len(subset)} records")
            atypes = subset['ActionType'].value_counts().to_dict()
            print(f"    ActionTypes: {atypes}")
            print(f"    DrugIDs: {subset['DrugID'].tolist()}")
            break

# Check DPI.csv
all_proteins = set(dpi['ProteinID'].dropna().str.upper())
hits2 = all_proteins & missing_uniprots
print(f"\nDPI.csv unique Protein IDs: {len(all_proteins)}")
print(f"Missing gene hits in DPI.csv: {hits2}")

# Check drug_ids.csv for SMILES
drug_ids = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L4\data\github_sources\ljafari_dti\dataset\drug_ids.csv')
print(f"\nDrug IDs columns: {list(drug_ids.columns)}")
print(drug_ids.head().to_string())

# Check if drug_ids has SMILES
if 'SMILES' in drug_ids.columns or 'smiles' in drug_ids.columns:
    smi_col = 'SMILES' if 'SMILES' in drug_ids.columns else 'smiles'
    print(f"\nDrug IDs has SMILES column: {smi_col}")
    print(drug_ids[[smi_col]].head())