"""审计化合物池SMILES质量"""
import pandas as pd
import requests
import time
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv(r'd:\铁衰老 绝不重蹈覆辙\L3\results\tcm_compound_pool_filtered.csv')
print(f"Total compounds: {len(df)}")

# 审计结果
mismatch_count = 0
match_count = 0
not_found_count = 0
results = []

for idx, row in df.iterrows():
    mol_id = row['MOL_ID']
    name = row['molecule_name']
    current_smiles = row['SMILES_std']
    mw_tcmsp = row['mw']
    mw_rdkit = row['MW']
    
    try:
        # 从PubChem获取
        r = requests.get(
            f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON',
            timeout=10
        )
        if 'IdentifierList' not in r.json():
            not_found_count += 1
            results.append((mol_id, name, 'NOT_FOUND', '', '', 0))
            continue
        
        cid = r.json()['IdentifierList']['CID'][0]
        r2 = requests.get(
            f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Title,ConnectivitySMILES,MolecularWeight/JSON',
            timeout=10
        )
        props = r2.json()['PropertyTable']['Properties'][0]
        pub_smiles = props.get('ConnectivitySMILES', '')
        pub_title = props.get('Title', '')
        pub_mw = float(props.get('MolecularWeight', 0))
        
        # 比较SMILES
        if current_smiles == pub_smiles:
            match_count += 1
            results.append((mol_id, name, 'MATCH', pub_smiles, pub_title, pub_mw))
        else:
            mismatch_count += 1
            results.append((mol_id, name, 'MISMATCH', pub_smiles, pub_title, pub_mw))
        
        if (idx + 1) % 50 == 0:
            print(f"  Progress: {idx+1}/{len(df)} (match={match_count}, mismatch={mismatch_count}, not_found={not_found_count})")
        
        time.sleep(0.2)
    
    except Exception as e:
        not_found_count += 1
        results.append((mol_id, name, 'ERROR', '', str(e)[:50], 0))

print(f"\n=== 审计结果 ===")
print(f"MATCH: {match_count}")
print(f"MISMATCH: {mismatch_count}")
print(f"NOT_FOUND/ERROR: {not_found_count}")

# 保存结果
result_df = pd.DataFrame(results, columns=['MOL_ID', 'molecule_name', 'Status', 'PubChem_SMILES', 'PubChem_Title', 'PubChem_MW'])
result_df.to_csv(r'd:\铁衰老 绝不重蹈覆辙\L3\results\smiles_audit.csv', index=False)
print(f"\n审计结果已保存至 L3/results/smiles_audit.csv")

# 显示一些MISMATCH的例子
print("\n=== MISMATCH示例 ===")
mismatches = result_df[result_df['Status'] == 'MISMATCH']
for _, row in mismatches.head(10).iterrows():
    orig = df[df['MOL_ID'] == row['MOL_ID']]
    if len(orig) > 0:
        print(f"\n{row['MOL_ID']}: {row['molecule_name'][:60]}")
        print(f"  Current:  {orig['SMILES_std'].values[0][:80]}")
        print(f"  PubChem:  {row['PubChem_SMILES'][:80]}")
        print(f"  MW(TCMSP)={orig['mw'].values[0]}, MW(PubChem)={row['PubChem_MW']}")