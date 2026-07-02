"""
为白名单化合物补充SMILES（使用PubChem API正确接口）
"""
import pandas as pd
import requests
import time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

L3 = Path('L3')
RESULTS = L3 / 'results'

raw = pd.read_excel(L3 / 'TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
smiles_fixed = pd.read_csv(RESULTS / 'tcmsp_smiles_fixed_v4.csv')

whitelist_names = [
    'beta-caryophyllene', 'caryophyllene', 'caryophyllene oxide',
    'saikosaponin a', 'saikosaponin d',
    'baicalein', 'baicalin', 'wogonin', 'wogonoside', 'oroxylin a',
    'emodin', 'rhein', 'aloe-emodin', 'chrysophanol', 'physcion', 'sennoside a',
    'paeoniflorin', 'albiflorin', 'paeonol',
    'cinnamaldehyde', 'cinnamic acid', 'coumarin',
    'pachymic acid', 'poricoic acid a',
    'amygdalin', 'prunasin',
    '6-gingerol', '8-gingerol', '10-gingerol', '6-shogaol',
    'naringin', 'hesperidin', 'nobiletin', 'tangeretin', 'synephrine',
    'quercetin', 'kaempferol', 'luteolin', 'apigenin',
    'berberine', 'curcumin', 'resveratrol',
    'liquiritin', 'glycyrrhizin', 'glycyrrhetinic acid', 'astragaloside iv',
]

existing_names = set(smiles_fixed['molecule_name'].str.lower())
missing = [n for n in whitelist_names if n.lower() not in existing_names]
print(f'已有SMILES: {len(whitelist_names) - len(missing)}/{len(whitelist_names)}')
print(f'缺失: {len(missing)} 个')

# PubChem PUG REST API
new_smiles = []
failed = []

for name in missing:
    try:
        # 用 cid 查询更可靠，先搜名字拿cid
        name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/cids/JSON"
        resp = requests.get(name_url, timeout=15)
        if resp.status_code != 200:
            failed.append((name, f'name_search_{resp.status_code}'))
            print(f'  ✗ {name}: name search {resp.status_code}')
            time.sleep(0.3)
            continue

        cid = resp.json()['IdentifierList']['CID'][0]
        # 用cid查SMILES
        prop_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES,MolecularWeight/JSON"
        resp2 = requests.get(prop_url, timeout=15)

        if resp2.status_code == 200:
            props = resp2.json()['PropertyTable']['Properties'][0]
            smi = props.get('CanonicalSMILES', '') or props.get('IsomericSMILES', '') or props.get('ConnectivitySMILES', '')
            mw = float(props.get('MolecularWeight', 0))
            if smi:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    new_smiles.append({
                        'molecule_name': name,
                        'SMILES': smi,
                        'MW_PubChem': mw,
                        'source': 'PubChem_API',
                    })
                    print(f'  ✓ {name}: CID={cid}, MW={mw:.1f}')
                else:
                    failed.append((name, 'RDKit_invalid'))
                    print(f'  ✗ {name}: RDKit无法解析')
            else:
                failed.append((name, 'no_SMILES'))
                print(f'  ✗ {name}: 无SMILES')
        else:
            failed.append((name, f'prop_{resp2.status_code}'))
            print(f'  ✗ {name}: prop {resp2.status_code}')
    except Exception as e:
        failed.append((name, str(e)[:50]))
        print(f'  ✗ {name}: {e}')
    time.sleep(0.3)

print(f'\n成功获取: {len(new_smiles)}/{len(missing)}')
if failed:
    print(f'失败: {len(failed)} 个')
    for n, r in failed:
        print(f'  {n}: {r}')

# 合并
if new_smiles:
    new_df = pd.DataFrame(new_smiles)
    existing = pd.read_csv(RESULTS / 'tcmsp_smiles_fixed_v4.csv')
    print(f'\n合并到修复文件: 原{len(existing)} + 新{len(new_df)}')
    combined = pd.concat([existing, new_df[['molecule_name', 'SMILES']]], ignore_index=True)
    combined = combined.drop_duplicates(subset=['molecule_name'], keep='first')
    out_path = RESULTS / 'tcmsp_smiles_fixed_v4_1.csv'
    combined.to_csv(out_path, index=False)
    print(f'  保存: {out_path} ({len(combined)} 条)')

print('\n✅ 完成！')
