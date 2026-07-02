import pandas as pd

# 检查黄芩素、汉黄芩素等关键化合物在各阶段的状态
raw = pd.read_excel('L3/TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
pool = pd.read_csv('L3/results/tcm_compound_pool_filtered.csv')

key_compounds = [
    'baicalein', 'wogonin', 'baicalin', 'scutellarin',  # 黄芩
    'aloe-emodin', 'emodin', 'rhein', 'chrysophanol', 'physcion',  # 大黄
    'paeoniflorin', 'paeonol',  # 白芍/牡丹皮
    'cinnamaldehyde', 'cinnamic acid',  # 桂枝
    'pachymic acid', 'poricoic acid',  # 茯苓
    'liquiritin', 'glycyrrhizic acid', 'glycyrrhetinic acid',  # 甘草
    'berberine', 'palmatine', 'coptisine',  # 黄连
    'saikosaponin', 'saikoside',  # 柴胡
    'gingerol', 'shogaol',  # 生姜
    'nobiletin', 'hesperidin', 'naringin',  # 枳实/陈皮
    'amygdalin', 'prunasin',  # 桃仁
]

print(f'{"化合物":<25} {"OB":>6} {"DL":>6} {"OB/DL":>7} {"Lipinski":>9} {"BBB":>6} {"PAINS":>7} {"在池中":>7}')
print('-' * 90)

pool_names = set(pool['molecule_name'].str.lower())

for name in key_compounds:
    matches = raw[raw['molecule_name'].str.lower().str.contains(name.lower(), na=False)]
    if len(matches) == 0:
        print(f'{name:<25}  NOT_IN_TCMSP')
        continue
    
    # 取最匹配的
    r = matches.iloc[0]
    ob = r['ob']
    dl = r['dl']
    obdl_pass = '✓' if (ob >= 30 and dl >= 0.18) else '✗'
    in_pool = '✓' if r['molecule_name'].lower() in pool_names else '✗'
    
    print(f'{r["molecule_name"][:24]:<25} {ob:>6.1f} {dl:>6.3f} {obdl_pass:>7} {"?":>9} {"?":>6} {"?":>7} {in_pool:>7}')
