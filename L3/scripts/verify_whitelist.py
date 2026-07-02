import pandas as pd

pool = pd.read_csv('L3/results/tcm_compound_pool_comprehensive.csv')

key_names = ['beta-caryophyllene', 'baicalein', 'emodin', 'paeonol', 'cinnamaldehyde',
             '6-gingerol', '6-shogaol', 'curcumin', 'resveratrol', 'amygdalin',
             'saikosaponin a', 'naringin', 'hesperidin', 'apigenin', 'tangeretin',
             'glycyrrhizin', 'glycyrrhetinic acid', 'astragaloside iv',
             'caryophyllene oxide', 'wogonoside', 'sennoside a', 'coumarin',
             'cinnamic acid']

print('关键化合物在综合池中的情况:')
header = f"{'化合物':<25s} {'分数':>6s} {'等级':<14s} {'OB%':>6s} {'DL':>6s} {'BBB':<8s} {'白名单':<6s} {'中药来源':<20s}"
print(header)
print('-' * len(header))

for name in key_names:
    match = pool[pool['molecule_name'].str.lower() == name.lower()]
    if len(match) == 0:
        match = pool[pool['molecule_name'].str.lower().str.contains(name.lower()[:8], na=False)]
    if len(match) > 0:
        r = match.iloc[0]
        wl = '★' if r['is_whitelist'] else ' '
        herbs = str(r.get('herb_origins', ''))[:20]
        print(f"{r['molecule_name'][:25]:<25s} {r['comprehensive_score']:>6.1f} {r['tier']:<14s} {r['ob']:>6.1f} {r['dl']:>6.3f} {r['BBB_Prediction']:<8s} {wl} {herbs}")
    else:
        print(f"{name:<25s}   不在池中")

print(f'\n总白名单数: {pool["is_whitelist"].sum()}')
print(f'综合评分版总数: {len(pool)}')
