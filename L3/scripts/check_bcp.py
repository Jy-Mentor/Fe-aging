import pandas as pd

smiles = pd.read_csv('L3/results/tcmsp_smiles_fixed_v4.csv')
raw = pd.read_excel('L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx')

for name in ['beta-caryophyllene', 'caryophyllene']:
    match = raw[raw['molecule_name'].str.lower().str.contains(name.lower(), na=False)]
    print(f'\n=== {name} in TCMSP ===')
    if len(match) > 0:
        for _, r in match.head(10).iterrows():
            print(f'  {r["molecule_name"]}: OB={r["ob"]:.1f}%, DL={r["dl"]:.3f}, MOL_ID={r["MOL_ID"]}')
            s = smiles[smiles['molecule_name'] == r['molecule_name']]
            print(f'    SMILES映射: {len(s)} 条')
            if len(s) > 0:
                print(f'    SMILES: {s.iloc[0]["SMILES"][:80]}...')
    else:
        print('  未找到')

# 查修复文件中有没有
print('\n=== 在SMILES修复文件中搜索 ===')
for name in ['caryophyllene', 'Caryophyllene', 'beta-caryophyllene']:
    s = smiles[smiles['molecule_name'].str.lower().str.contains(name.lower(), na=False)]
    print(f'  {name}: {len(s)} 条')
    for _, r in s.head(5).iterrows():
        print(f'    {r["molecule_name"]}')
