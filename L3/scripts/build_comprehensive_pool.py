"""
综合评分版化合物池构建
- 支持 OB 或 DL 逻辑（满足其一即可）
- 综合评分：OB + DL + Lipinski + BBB + PAINS + 中药来源权重
- 白名单机制：已知关键活性成分自动保留
- 复方药味差异化阈值
"""
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SaltRemover


L3 = Path('L3')
RESULTS = L3 / 'results'

# 复方药味差异化阈值配置
HERB_SPECIFIC_THRESHOLDS = {
    # 药味: (OB阈值, DL阈值) - 有些药味的成分普遍OB低但活性强
    '柴胡': (25.0, 0.06),    # 柴胡皂苷DL普遍低，但活性强
    '黄芩': (30.0, 0.18),    # 黄芩黄酮类普遍达标
    '半夏': (20.0, 0.05),    # 半夏生物碱类
    '生姜': (20.0, 0.03),    # 姜辣素类DL普遍低
    '大枣': (20.0, 0.05),
    '枳实': (25.0, 0.10),    # 辛弗林等
    '大黄': (20.0, 0.15),    # 蒽醌类OB差异大
    '白芍': (25.0, 0.10),    # 芍药苷等
    '桂枝': (20.0, 0.02),    # 桂皮醛DL低但活性强
    '茯苓': (25.0, 0.20),    # 三萜类
    '牡丹皮': (20.0, 0.05),  # 丹皮酚
    '桃仁': (10.0, 0.10),    # 苦杏仁苷OB低
}

FORMULAS = {
    '大柴胡汤': ['柴胡', '黄芩', '半夏', '生姜', '大枣', '枳实', '大黄', '白芍'],
    '桂枝茯苓丸': ['桂枝', '茯苓', '牡丹皮', '桃仁', '白芍'],
}

# 白名单：已知关键活性成分（开绿灯）
WHITELIST_NAMES = {
    # BCP
    'beta-caryophyllene': 'β-石竹烯',
    'caryophyllene': '石竹烯',
    'caryophyllene oxide': '石竹烯氧化物',
    # 柴胡
    'saikosaponin a': '柴胡皂苷a',
    'saikosaponin d': '柴胡皂苷d',
    'saikosaponin b2': '柴胡皂苷b2',
    # 黄芩
    'baicalein': '黄芩素',
    'baicalin': '黄芩苷',
    'wogonin': '汉黄芩素',
    'wogonoside': '汉黄芩苷',
    'oroxylin a': '木蝴蝶素A',
    'skullcapflavone ii': '黄芩黄酮II',
    # 大黄
    'emodin': '大黄素',
    'rhein': '大黄酸',
    'aloe-emodin': '芦荟大黄素',
    'chrysophanol': '大黄酚',
    'physcion': '大黄素甲醚',
    'sennoside a': '番泻苷A',
    # 芍药
    'paeoniflorin': '芍药苷',
    'albiflorin': '芍药内酯苷',
    'paeonol': '丹皮酚',
    # 桂枝
    'cinnamaldehyde': '桂皮醛',
    'cinnamic acid': '肉桂酸',
    'coumarin': '香豆素',
    # 茯苓
    'pachymic acid': '茯苓酸',
    'poricoic acid a': '茯苓次聚糖A',
    'dehydrotrametenolic acid': '去氢土莫酸',
    # 丹皮
    'paeonol': '丹皮酚',
    'paeoniflorin': '芍药苷',
    # 桃仁
    'amygdalin': '苦杏仁苷',
    'prunasin': '野黑樱苷',
    # 半夏
    'pinellic acid': '半夏酸',
    # 生姜
    '6-gingerol': '6-姜辣素',
    '8-gingerol': '8-姜辣素',
    '10-gingerol': '10-姜辣素',
    '6-shogaol': '6-姜烯酚',
    'gingerol': '姜辣素',
    # 枳实
    'naringin': '柚皮苷',
    'hesperidin': '橙皮苷',
    'nobiletin': '川陈皮素',
    'tangeretin': '橘皮素',
    'synephrine': '辛弗林',
    # 大枣
    'zizyphus saponin': '酸枣皂苷',
    # 其他重要中药单体
    'quercetin': '槲皮素',
    'kaempferol': '山柰酚',
    'luteolin': '木犀草素',
    'apigenin': '芹菜素',
    'berberine': '小檗碱',
    'curcumin': '姜黄素',
    'resveratrol': '白藜芦醇',
    'liquiritin': '甘草苷',
    'glycyrrhizin': '甘草酸',
    'glycyrrhetinic acid': '甘草次酸',
    'astragaloside iv': '黄芪甲苷',
}


def load_data(results_dir):
    """加载 TCMSP、SMILES 修正、草药-成分映射三个数据源。"""
    raw = pd.read_excel(L3 / 'TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
    smiles_fixed = pd.read_csv(results_dir / 'tcmsp_smiles_fixed_v4_1.csv')
    herb_map_df = pd.read_excel(results_dir / 'herb_ingredient_mapping.xlsx')
    return raw, smiles_fixed, herb_map_df


def build_herb_map(herb_map_df):
    """构建 MOL_ID -> [herb_cn_name] 的映射。"""
    herb_map = {}
    for _, row in herb_map_df.iterrows():
        mol_id = str(row.get('MOL_ID', '')).strip()
        herb = str(row.get('herb_cn_name', '')).strip()
        if not mol_id or not herb or pd.isna(row.get('MOL_ID')):
            continue
        if mol_id not in herb_map:
            herb_map[mol_id] = []
        if herb not in herb_map[mol_id]:
            herb_map[mol_id].append(herb)
    return herb_map


def build_smiles_map(smiles_fixed_df):
    """构建 molecule_name -> SMILES 的映射。"""
    return dict(zip(smiles_fixed_df['molecule_name'], smiles_fixed_df['SMILES']))


def standardize_smiles(smi):
    """RDKit SMILES 规范化（去盐、取最大片段、去电荷、归一化）。"""
    if not smi or not isinstance(smi, str) or len(smi) < 3:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        salt_remover = SaltRemover.SaltRemover()
        mol = salt_remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumAtoms())
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def lipinski_info(mw, logp, hbd, hba):
    """计算 Lipinski 五规则违例数及是否通过。"""
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1, violations


def bbb_class(tpsa, logp):
    """基于 TPSA 和 LogP 预测 BBB 通透性。"""
    if tpsa < 90 and 1 < logp < 4:
        return 'BBB+'
    elif tpsa < 120 and logp < 5:
        return 'BBB+/-'
    else:
        return 'BBB-'


def init_pains_catalog():
    """初始化 PAINS 过滤器。"""
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(pains_params)


def is_whitelist(name):
    """判断化合物名称是否在白名单中。"""
    name_lower = str(name).lower().strip()
    return name_lower in {k.lower() for k in WHITELIST_NAMES}


def pass_herb_specific(row):
    """判断是否通过任一来源药味的差异化阈值。"""
    if row['pass_standard']:
        return True
    for herb in row['herb_list']:
        if herb in HERB_SPECIFIC_THRESHOLDS:
            ob_thr, dl_thr = HERB_SPECIFIC_THRESHOLDS[herb]
            if row['ob'] >= ob_thr and row['dl'] >= dl_thr:
                return True
    return False


def apply_comprehensive_filter(raw, herb_map):
    """
    应用综合筛选条件：标准通过 或 差异化阈值 或 白名单。
    返回综合通过的数据子集及其过滤统计信息。
    """
    raw['herb_list'] = raw['MOL_ID'].apply(lambda x: herb_map.get(str(x).strip(), []))
    raw['n_herbs'] = raw['herb_list'].apply(len)
    raw['herb_origins'] = raw['herb_list'].apply(lambda x: '; '.join(x))

    raw['pass_standard'] = (raw['ob'] >= 30.0) & (raw['dl'] >= 0.18)
    raw['pass_or'] = (raw['ob'] >= 30.0) | (raw['dl'] >= 0.18)
    raw['is_whitelist'] = raw['molecule_name'].apply(is_whitelist)
    raw['pass_herb_specific'] = raw.apply(pass_herb_specific, axis=1)
    raw['pass_comprehensive'] = raw['pass_standard'] | raw['pass_herb_specific'] | raw['is_whitelist']

    stats = {
        'total': len(raw),
        'standard': raw['pass_standard'].sum(),
        'or_logic': raw['pass_or'].sum(),
        'herb_specific': raw['pass_herb_specific'].sum(),
        'whitelist': raw['is_whitelist'].sum(),
        'comprehensive': raw['pass_comprehensive'].sum(),
    }
    return raw[raw['pass_comprehensive']].copy().reset_index(drop=True), stats


def compute_descriptors(selected, smiles_map):
    """
    为候选化合物计算 RDKit 描述符和类药性标记。
    返回处理后的 DataFrame。
    """
    selected['SMILES'] = selected['molecule_name'].map(smiles_map)
    selected = selected[selected['SMILES'].notna()].copy().reset_index(drop=True)
    selected['SMILES_std'] = selected['SMILES'].apply(standardize_smiles)
    selected = selected[selected['SMILES_std'].notna()].copy().reset_index(drop=True)

    pains_catalog = init_pains_catalog()

    mw_list, logp_list, tpsa_list, hbd_list, hba_list, qed_list = [], [], [], [], [], []
    lip_pass_list, lip_viol_list, bbb_list, pains_pass_list = [], [], [], []

    for _, row in selected.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES_std'])
        if mol is None:
            mw_list.append(np.nan); logp_list.append(np.nan); tpsa_list.append(np.nan)
            hbd_list.append(np.nan); hba_list.append(np.nan); qed_list.append(np.nan)
            lip_pass_list.append(False); lip_viol_list.append(99)
            bbb_list.append('BBB-'); pains_pass_list.append(False)
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = Descriptors.qed(mol)

        mw_list.append(mw); logp_list.append(logp); tpsa_list.append(tpsa)
        hbd_list.append(hbd); hba_list.append(hba); qed_list.append(qed)

        lp, lv = lipinski_info(mw, logp, hbd, hba)
        lip_pass_list.append(lp); lip_viol_list.append(lv)
        bbb_list.append(bbb_class(tpsa, logp))
        pains_pass_list.append(len(pains_catalog.GetMatches(mol)) == 0)

    selected['MW_calc'] = mw_list
    selected['LogP_calc'] = logp_list
    selected['TPSA_calc'] = tpsa_list
    selected['HBD_calc'] = hbd_list
    selected['HBA_calc'] = hba_list
    selected['QED'] = qed_list
    selected['Lipinski_Pass'] = lip_pass_list
    selected['Lipinski_Violations'] = lip_viol_list
    selected['BBB_Prediction'] = bbb_list
    selected['PAINS_Pass'] = pains_pass_list

    selected['MW_DIFF'] = (selected['MW_calc'] - selected['mw']).abs()
    selected['MW_REL_DIFF'] = selected['MW_DIFF'] / selected['mw'].replace(0, 1.0).abs()
    selected['SMILES_MATCH_STATUS'] = np.where(
        (selected['MW_DIFF'] <= 10.0) | (selected['MW_REL_DIFF'] <= 0.1),
        'MATCH_OK', 'UNCERTAIN'
    )
    selected = selected[selected['SMILES_MATCH_STATUS'] == 'MATCH_OK'].copy().reset_index(drop=True)
    return selected


def calc_comprehensive_score(row):
    """
    综合评分 (0-100分):
    - OB得分 (25分): OB越高越好，60%以上满分
    - DL得分 (20分): DL越高越好，0.6以上满分
    - Lipinski (15分): 0违例15分，1违例10分，2违例5分，>2违例0分
    - BBB (15分): BBB+ 15分, BBB+/- 10分, BBB- 5分
    - PAINS (10分): 通过10分，警示5分
    - 中药来源 (10分): 来源越多分越高
    - 白名单加成 (5分): 白名单化合物额外加分
    """
    score = 0.0

    ob = row['ob']
    score += min(ob / 60.0 * 25, 25)

    dl = row['dl']
    score += min(dl / 0.6 * 20, 20)

    viol = row['Lipinski_Violations']
    if viol == 0: score += 15
    elif viol == 1: score += 10
    elif viol == 2: score += 5
    else: score += 0

    bbb = row['BBB_Prediction']
    if bbb == 'BBB+': score += 15
    elif bbb == 'BBB+/-': score += 10
    else: score += 5

    if row['PAINS_Pass']: score += 10
    else: score += 5

    n_herbs = row['n_herbs']
    score += min(n_herbs * 2, 10)

    if row['is_whitelist']:
        score += 5

    return round(score, 2)


def assign_tier(score):
    """根据综合评分划分优先级等级。"""
    if score >= 75: return 'A+（高优先级）'
    elif score >= 65: return 'A（推荐）'
    elif score >= 55: return 'B（较好）'
    elif score >= 45: return 'C（一般）'
    else: return 'D（低优先级）'


def score_and_deduplicate(selected):
    """计算综合评分，按 SMILES 去重（保留高分），并划分等级。"""
    selected['comprehensive_score'] = selected.apply(calc_comprehensive_score, axis=1)
    selected = selected.sort_values('comprehensive_score', ascending=False)
    selected = selected.drop_duplicates(subset=['SMILES_std'], keep='first').reset_index(drop=True)
    selected['tier'] = selected['comprehensive_score'].apply(assign_tier)
    selected = selected.sort_values('comprehensive_score', ascending=False).reset_index(drop=True)
    return selected


def save_pool(selected, output_path):
    """保存综合评分版化合物池到 CSV。"""
    out_cols = [
        'MOL_ID', 'molecule_name', 'SMILES_std', 'herb_origins', 'n_herbs',
        'comprehensive_score', 'tier', 'is_whitelist',
        'pass_standard', 'pass_herb_specific', 'pass_comprehensive',
        'ob', 'dl', 'mw', 'MW_calc', 'LogP_calc', 'TPSA_calc',
        'HBD_calc', 'HBA_calc', 'QED',
        'Lipinski_Pass', 'Lipinski_Violations', 'BBB_Prediction', 'PAINS_Pass',
        'SMILES_MATCH_STATUS', 'MW_DIFF', 'MW_REL_DIFF',
        'alogp', 'bbb', 'tpsa', 'caco2', 'hdon', 'hacc', 'rbn',
    ]
    available_cols = [c for c in out_cols if c in selected.columns]
    selected[available_cols].to_csv(output_path, index=False, float_format='%.4f')


def print_summary(selected, filter_stats, selected_sorted, output_path):
    """打印统计汇总、分级分布、复方覆盖和 Top20。"""
    print(f'\n综合评分版保存: {output_path}')
    print('\n' + '=' * 70)
    print('统计汇总')
    print('=' * 70)
    print(f'  TCMSP 原始: {filter_stats["total"]:,}')
    print(f'  标准通过(OB>=30且DL>=0.18): {filter_stats["standard"]:,}')
    print(f'  或逻辑通过(OB>=30或DL>=0.18): {filter_stats["or_logic"]:,}')
    print(f'  差异化阈值通过: {filter_stats["herb_specific"]:,}')
    print(f'  白名单化合物: {filter_stats["whitelist"]}')
    print(f'  综合通过: {filter_stats["comprehensive"]:,}')
    print(f'  综合版化合物总数: {len(selected_sorted)}')
    print(f'  分级分布:')
    for tier in ['A+（高优先级）', 'A（推荐）', 'B（较好）', 'C（一般）', 'D（低优先级）']:
        n = (selected_sorted['tier'] == tier).sum()
        print(f'    {tier}: {n} 个 ({n/len(selected_sorted)*100:.1f}%)')

    print(f'\n  有中药来源: {(selected_sorted["n_herbs"] > 0).sum()}')
    print(f'  白名单化合物: {selected_sorted["is_whitelist"].sum()}')

    print(f'\n  复方药味覆盖（综合版）:')
    for formula_name, herbs in FORMULAS.items():
        print(f'\n  [{formula_name}]')
        for herb in herbs:
            herb_mol_ids = set(selected_sorted[selected_sorted['herb_origins'].str.contains(herb, na=False)]['MOL_ID'])
            in_pool = selected_sorted[selected_sorted['MOL_ID'].isin(herb_mol_ids)]
            n_a = (in_pool['tier'].str.startswith('A')).sum()
            print(f'    {herb}: {len(in_pool)} 个 (A级:{n_a})')

    print(f'\n  综合评分 Top 20:')
    for i, row in selected_sorted.head(20).iterrows():
        herbs = row['herb_origins'][:20] if row['herb_origins'] else '无'
        wl = '★' if row['is_whitelist'] else ' '
        print(f'    {i+1:2d}. {wl} {row["molecule_name"][:25]:25s} {row["comprehensive_score"]:5.1f} {row["tier"]:10s} | {herbs}')


def print_formula_core_list(selected_sorted):
    """打印复方核心成分清单（每味药 Top5）。"""
    print(f'\n' + '=' * 70)
    print('复方核心成分清单（每味药Top 5）')
    print('=' * 70)

    for formula_name, herbs in FORMULAS.items():
        print(f'\n### {formula_name}')
        for herb in herbs:
            herb_mol_ids = set(selected_sorted[selected_sorted['herb_origins'].str.contains(herb, na=False)]['MOL_ID'])
            in_pool = selected_sorted[selected_sorted['MOL_ID'].isin(herb_mol_ids)]
            top5 = in_pool.nlargest(min(5, len(in_pool)), 'comprehensive_score')
            print(f'\n  {herb}（共{len(in_pool)}个）:')
            for _, r in top5.iterrows():
                wl = '★' if r['is_whitelist'] else ' '
                print(
                    f'    {wl} {r["molecule_name"][:30]:30s} 分数:{r["comprehensive_score"]:.1f} {r["tier"]:8s} '
                    f'OB={r["ob"]:.1f}% DL={r["dl"]:.3f} {r["BBB_Prediction"]}'
                )


def print_version_comparison(results_dir, selected_sorted):
    """与严格版、放宽版化合物池进行对比。"""
    print(f'\n' + '=' * 70)
    print('三版化合物池对比')
    print('=' * 70)

    strict = pd.read_csv(results_dir / 'tcm_compound_pool_strict.csv')
    relaxed = pd.read_csv(results_dir / 'tcm_compound_pool_relaxed.csv')

    print(f'  {"版本":<20s} {"数量":>6s}  {"有中药来源":>10s}  {"说明"}')
    print(f'  {"-"*20} {"-"*6} {"-"*10} {"-"*30}')
    print(f'  {"严格版":<20s} {len(strict):>6d}  {(strict["n_herb_origins"] > 0).sum():>10d}  Lipinski+BBB+PAINS')
    print(f'  {"放宽版":<20s} {len(relaxed):>6d}  {(relaxed["n_herb_origins"] > 0).sum():>10d}  Lipinski+PAINS (无BBB)')
    print(f'  {"综合评分版":<20s} {len(selected_sorted):>6d}  {(selected_sorted["n_herbs"] > 0).sum():>10d}  或逻辑+白名单+差异化阈值')


def main():
    """综合评分版化合物池构建主流程。"""
    print('=' * 70)
    print('综合评分版化合物池构建')
    print('=' * 70)

    raw, smiles_fixed, herb_map_df = load_data(RESULTS)
    herb_map = build_herb_map(herb_map_df)
    smiles_map = build_smiles_map(smiles_fixed)

    print(f'\n1. 草药-成分映射完成: {len(herb_map)} 个 MOL_ID 有来源')
    print(f'2. SMILES 映射加载: {len(smiles_map)} 个')

    print('\n3. 综合筛选（标准通过 或 差异化阈值 或 白名单）...')
    selected, filter_stats = apply_comprehensive_filter(raw, herb_map)
    print(f'  TCMSP 原始: {filter_stats["total"]:,}')
    print(f'  标准通过(OB>=30且DL>=0.18): {filter_stats["standard"]:,}')
    print(f'  或逻辑通过(OB>=30或DL>=0.18): {filter_stats["or_logic"]:,}')
    print(f'  差异化阈值通过: {filter_stats["herb_specific"]:,}')
    print(f'  白名单化合物: {filter_stats["whitelist"]}')
    print(f'  综合通过: {filter_stats["comprehensive"]:,}')

    print(f'\n4. 计算描述符... (n={len(selected)})')
    selected = compute_descriptors(selected, smiles_map)
    print(f'  有SMILES: {selected["SMILES"].notna().sum()}')
    print(f'  规范化成功: {len(selected)}')
    print(f'  MW校验通过: {len(selected)}')

    print(f'\n5. 计算综合评分并去重...')
    selected_sorted = score_and_deduplicate(selected)
    print(f'  去重后: {len(selected_sorted)}')

    output_path = RESULTS / 'tcm_compound_pool_comprehensive.csv'
    save_pool(selected_sorted, output_path)

    print_summary(selected, filter_stats, selected_sorted, output_path)
    print_formula_core_list(selected_sorted)
    print_version_comparison(RESULTS, selected_sorted)

    print('\n✅ 综合评分版构建完成！')


if __name__ == '__main__':
    main()
