"""检查化合物池中SMILES质量"""
import pandas as pd
import requests
import time

df = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L3\results\tcm_compound_pool_filtered.csv')
print(f"Total compounds: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print()

# 检查SMILES_MATCH_STATUS分布
print("=== SMILES匹配状态 ===")
print(df['SMILES_MATCH_STATUS'].value_counts())
print()

# 检查MW_DIFF异常值
print("=== MW差异 > 5的化合物 ===")
bad_mw = df[df['MW_DIFF'] > 5]
print(f"共 {len(bad_mw)} 个化合物MW差异 > 5\n")
for _, row in bad_mw.head(10).iterrows():
    print(f"  {row['MOL_ID']}: {row['molecule_name'][:50]} | MW_orig={row['mw']} | MW_rdkit={row['MW']:.1f} | diff={row['MW_DIFF']:.1f}")
print()

# 检查几个关键化合物
key_mols = {
    'MOL000001': 'anthocyanidin',
    'MOL002288': 'Emodin-1-O-beta-D-glucopyranoside',
    'MOL000422': 'kaempferol',
    'MOL000173': 'wogonin',
    'MOL001001': 'quercetin-3-O-beta-D-glucuronide',
    'MOL000098': 'quercetin',
}

print("=== 关键化合物SMILES验证 ===")
for mol_id, name in key_mols.items():
    row = df[df['MOL_ID'] == mol_id]
    if len(row) == 0:
        print(f"{mol_id} ({name}): NOT IN POOL")
        continue
    
    smiles = row['SMILES_std'].values[0]
    source = row['SMILES_SOURCE'].values[0]
    mw_tcmsp = row['mw'].values[0]
    mw_rdkit = row['MW'].values[0]
    mw_diff = row['MW_DIFF'].values[0]
    
    # 从PubChem获取正确SMILES
    try:
        r = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON')
        if 'IdentifierList' in r.json():
            cid = r.json()['IdentifierList']['CID'][0]
            r2 = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/ConnectivitySMILES,MolecularWeight/JSON')
            props = r2.json()['PropertyTable']['Properties'][0]
            pub_smiles = props.get('ConnectivitySMILES', 'N/A')
            pub_mw = float(props.get('MolecularWeight', 0))
            
            # 检查当前SMILES是否与PubChem一致
            match = "MATCH" if smiles == pub_smiles else "MISMATCH"
            print(f"{mol_id} ({name}):")
            print(f"  Source: {source}")
            print(f"  Current SMILES: {smiles[:70]}...")
            print(f"  PubChem SMILES: {pub_smiles[:70]}...")
            print(f"  TCMSP MW={mw_tcmsp}, RDKit MW={mw_rdkit}, PubChem MW={pub_mw}")
            print(f"  Status: {match}")
            print()
        time.sleep(0.3)
    except Exception as e:
        print(f"{mol_id} ({name}): ERROR - {e}")
        print()

# 统计MW差异分布
print("=== MW差异分布 ===")
print(df['MW_DIFF'].describe())
print(f"\nMW_DIFF > 5: {len(df[df['MW_DIFF'] > 5])}")
print(f"MW_DIFF > 10: {len(df[df['MW_DIFF'] > 10])}")