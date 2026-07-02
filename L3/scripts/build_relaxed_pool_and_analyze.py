"""
生成放宽版化合物池 + 复方药味分析 + 关键化合物诊断

输出：
1. 严格版（原）：Lipinski + BBB + PAINS → 571个
2. 放宽版：Lipinski + PAINS（去掉BBB要求） → ~900+个
3. 复方各药味主要单体清单（含被过滤原因）
4. BCP等关键化合物的过滤阶段诊断
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import SaltRemover


L3 = Path('L3')
RESULTS = L3 / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

FORMULAS = {
    '大柴胡汤': ['柴胡', '黄芩', '半夏', '生姜', '大枣', '枳实', '大黄', '白芍'],
    '桂枝茯苓丸': ['桂枝', '茯苓', '牡丹皮', '桃仁', '白芍'],
}

KEY_COMPOUNDS = {
    'beta-caryophyllene': 'β-石竹烯 (BCP)',
    'baicalein': '黄芩素',
    'wogonin': '汉黄芩素',
    'baicalin': '黄芩苷',
    'aloe-emodin': '芦荟大黄素',
    'emodin': '大黄素',
    'rhein': '大黄酸',
    'paeoniflorin': '芍药苷',
    'paeonol': '丹皮酚',
    'cinnamaldehyde': '桂皮醛',
    'cinnamic acid': '肉桂酸',
    'pachymic acid': '茯苓酸',
    'liquiritin': '甘草苷',
    'glycyrrhetinic acid': '甘草次酸',
    '6-gingerol': '6-姜辣素',
    '6-shogaol': '6-姜烯酚',
    'nobiletin': '川陈皮素',
    'hesperidin': '橙皮苷',
    'amygdalin': '苦杏仁苷',
    'saikosaponin a': '柴胡皂苷a',
    'saikosaponin d': '柴胡皂苷d',
    'berberine': '小檗碱',
    'quercetin': '槲皮素',
    'kaempferol': '山柰酚',
    'luteolin': '木犀草素',
    'curcumin': '姜黄素',
    'resveratrol': '白藜芦醇',
}


def load_data(results_dir):
    """加载 TCMSP 成分、SMILES 修正和草药-成分映射。"""
    raw = pd.read_excel(L3 / 'TCMSP-Spider/data/sample_data/ingredients_data.xlsx')
    smiles_fixed = pd.read_csv(results_dir / 'tcmsp_smiles_fixed_v4.csv')
    herb_map_df = pd.read_excel(results_dir / 'herb_ingredient_mapping.xlsx')
    return raw, smiles_fixed, herb_map_df


def build_herb_map(herb_map_df):
    """构建 MOL_ID -> [herb_cn_name] 映射。"""
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
    """构建 molecule_name -> SMILES 映射。"""
    return dict(zip(smiles_fixed_df['molecule_name'], smiles_fixed_df['SMILES']))


def standardize_smiles(smi):
    """RDKit SMILES 规范化。"""
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


def filter_lipinski(mw, logp, hbd, hba):
    """Lipinski 五规则过滤。"""
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return violations <= 1, violations


def filter_bbb(tpsa, logp):
    """BBB 通透性预测。"""
    if tpsa < 90 and 1 < logp < 4:
        return 'BBB+'
    elif tpsa < 120 and logp < 5:
        return 'BBB+/-'
    else:
        return 'BBB-'


def filter_pains(mol, catalog):
    """PAINS 过滤。"""
    return len(catalog.GetMatches(mol)) == 0


def init_pains_catalog():
    """初始化 PAINS 过滤器。"""
    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(pains_params)


def obdl_filter(raw):
    """OB/DL 基础过滤。"""
    return raw[(raw['ob'] >= 30.0) & (raw['dl'] >= 0.18)].copy().reset_index(drop=True)


def map_and_standardize_smiles(active, smiles_map):
    """为活性化合物映射 SMILES 并规范化。"""
    active['SMILES'] = active['molecule_name'].map(smiles_map)
    active = active[active['SMILES'].notna()].copy().reset_index(drop=True)
    active['SMILES_std'] = active['SMILES'].apply(standardize_smiles)
    active = active[active['SMILES_std'].notna()].copy().reset_index(drop=True)
    return active


def compute_descriptors(active):
    """计算 RDKit 描述符、Lipinski/BBB/PAINS 过滤结果。"""
    pains_catalog = init_pains_catalog()

    lip_pass_list = []
    lip_viol_list = []
    bbb_list = []
    pains_pass_list = []
    mw_calc_list = []
    logp_calc_list = []
    tpsa_calc_list = []
    hbd_calc_list = []
    hba_calc_list = []
    qed_list = []

    print('  计算描述符...', end='', flush=True)
    for _, row in active.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES_std'])
        if mol is None:
            lip_pass_list.append(False); lip_viol_list.append(99)
            bbb_list.append('BBB-'); pains_pass_list.append(False)
            mw_calc_list.append(np.nan); logp_calc_list.append(np.nan)
            tpsa_calc_list.append(np.nan); hbd_calc_list.append(np.nan)
            hba_calc_list.append(np.nan); qed_list.append(np.nan)
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        qed = Descriptors.qed(mol)

        mw_calc_list.append(mw); logp_calc_list.append(logp)
        tpsa_calc_list.append(tpsa); hbd_calc_list.append(hbd)
        hba_calc_list.append(hba); qed_list.append(qed)

        lip_pass, lip_viol = filter_lipinski(mw, logp, hbd, hba)
        lip_pass_list.append(lip_pass); lip_viol_list.append(lip_viol)
        bbb_list.append(filter_bbb(tpsa, logp))
        pains_pass_list.append(filter_pains(mol, pains_catalog))

    active['MW_calc'] = mw_calc_list
    active['LogP_calc'] = logp_calc_list
    active['TPSA_calc'] = tpsa_calc_list
    active['HBD_calc'] = hbd_calc_list
    active['HBA_calc'] = hba_calc_list
    active['QED'] = qed_list
    active['Lipinski_Pass'] = lip_pass_list
    active['Lipinski_Violations'] = lip_viol_list
    active['BBB_Prediction'] = bbb_list
    active['PAINS_Pass'] = pains_pass_list
    print(' done')
    return active


def validate_mw_consistency(active):
    """基于 RDKit 计算 MW 和 TCMSP 原始 mw 进行一致性校验。"""
    active['MW_DIFF'] = (active['MW_calc'] - active['mw']).abs()
    active['MW_REL_DIFF'] = active['MW_DIFF'] / active['mw'].replace(0, 1.0).abs()
    active['SMILES_MATCH_STATUS'] = np.where(
        (active['MW_DIFF'] <= 5.0) | (active['MW_REL_DIFF'] <= 0.05),
        'MATCH_OK', 'UNCERTAIN'
    )
    n_uncertain = (active['SMILES_MATCH_STATUS'] == 'UNCERTAIN').sum()
    print(f'  MW 一致: {(active["SMILES_MATCH_STATUS"] == "MATCH_OK").sum()}, 不确定: {n_uncertain}')
    active = active[active['SMILES_MATCH_STATUS'] == 'MATCH_OK'].copy().reset_index(drop=True)
    return active


def add_herb_info(df, herb_map):
    """添加草药来源列。"""
    df = df.copy()
    origins = []
    n_origins = []
    for _, row in df.iterrows():
        mol_id = str(row.get('MOL_ID', '')).strip()
        herbs = herb_map.get(mol_id, [])
        origins.append('; '.join(herbs))
        n_origins.append(len(herbs))
    df['herb_origins'] = origins
    df['n_herb_origins'] = n_origins
    return df


def build_pools(active):
    """生成严格版和放宽版化合物池。"""
    strict = active[
        active['Lipinski_Pass'] &
        active['BBB_Prediction'].isin(['BBB+', 'BBB+/-']) &
        active['PAINS_Pass']
    ].copy().reset_index(drop=True)
    strict = strict.drop_duplicates(subset=['SMILES_std'], keep='first').reset_index(drop=True)

    relaxed = active[
        active['Lipinski_Pass'] &
        active['PAINS_Pass']
    ].copy().reset_index(drop=True)
    relaxed = relaxed.drop_duplicates(subset=['SMILES_std'], keep='first').reset_index(drop=True)

    return strict, relaxed


def save_pools(strict, relaxed, results_dir):
    """保存严格版和放宽版化合物池到 CSV。"""
    out_cols = ['MOL_ID', 'molecule_name', 'SMILES_std', 'herb_origins', 'n_herb_origins',
                'mw', 'ob', 'dl', 'MW_calc', 'LogP_calc', 'TPSA_calc',
                'HBD_calc', 'HBA_calc', 'QED',
                'Lipinski_Pass', 'Lipinski_Violations', 'BBB_Prediction', 'PAINS_Pass',
                'SMILES_MATCH_STATUS', 'MW_DIFF', 'MW_REL_DIFF',
                'alogp', 'bbb', 'tpsa', 'caco2', 'hdon', 'hacc', 'rbn']

    strict.to_csv(results_dir / 'tcm_compound_pool_strict.csv', index=False, float_format='%.4f')
    relaxed.to_csv(results_dir / 'tcm_compound_pool_relaxed.csv', index=False, float_format='%.4f')


def generate_formula_report(active, formulas, output_path):
    """生成复方各药味主要单体分析报告。"""
    report_lines = []
    report_lines.append('# 复方药味主要单体分析报告')
    report_lines.append(f'\n生成时间: {datetime.now().strftime("%Y-%m-%d")}')
    report_lines.append('\n## 说明')
    report_lines.append('- 以下为各药味中通过OB>=30%且DL>=0.18的前10个高OB成分')
    report_lines.append('- [池内✓]表示在放宽版化合物池中，[池内✗]表示被进一步过滤（Lipinski/PAINS）')
    report_lines.append('- BBB列：BBB+ / BBB+/- 可通过血脑屏障，BBB- 不能')

    for formula_name, herbs in formulas.items():
        report_lines.append(f'\n## {formula_name}')
        report_lines.append(f'\n药味: {"、".join(herbs)}')

        for herb in herbs:
            herb_mols = herb_map_df_global[herb_map_df_global['herb_cn_name'] == herb]
            herb_active = active[active['MOL_ID'].isin(herb_mols['MOL_ID'])]
            herb_active_obdl = herb_active[(herb_active['ob'] >= 30) & (herb_active['dl'] >= 0.18)]
            top10 = herb_active_obdl.nlargest(10, 'ob')

            report_lines.append(f'\n### {herb}')
            report_lines.append(f'- TCMSP收录成分: {len(herb_mols)} 个')
            report_lines.append(f'- 通过OB/DL: {len(herb_active_obdl)} 个')
            report_lines.append(f'- 在放宽版池中: {herb_active[herb_active["Lipinski_Pass"] & herb_active["PAINS_Pass"]]["SMILES_std"].nunique()} 个')
            report_lines.append(f'- 在严格版池中: {herb_active[herb_active["Lipinski_Pass"] & herb_active["BBB_Prediction"].isin(["BBB+","BBB+/-"]) & herb_active["PAINS_Pass"]]["SMILES_std"].nunique()} 个')
            report_lines.append('')
            report_lines.append('| 化合物 | OB% | DL | MW | BBB | Lipinski | PAINS | 在放宽池 | 在严格池 |')
            report_lines.append('|--------|-----|----|-----|-----|----------|-------|----------|----------|')

            for _, r in top10.iterrows():
                in_relaxed = '✓' if (r['Lipinski_Pass'] and r['PAINS_Pass']) else '✗'
                in_strict = '✓' if (r['Lipinski_Pass'] and r['BBB_Prediction'] in ['BBB+', 'BBB+/-'] and r['PAINS_Pass']) else '✗'
                lip = '✓' if r['Lipinski_Pass'] else '✗'
                pains = '✓' if r['PAINS_Pass'] else '✗'
                report_lines.append(
                    f"| {r['molecule_name'][:30]} | {r['ob']:.1f} | {r['dl']:.3f} | "
                    f"{r['MW_calc']:.1f} | {r['BBB_Prediction']} | {lip} | {pains} | {in_relaxed} | {in_strict} |"
                )

    report_text = '\n'.join(report_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)


def generate_key_compounds_diagnosis(raw, active, output_path):
    """生成关键化合物过滤阶段诊断报告。"""
    diagnosis_lines = []
    diagnosis_lines.append('# 关键化合物过滤阶段诊断')
    diagnosis_lines.append(f'\n生成时间: {datetime.now().strftime("%Y-%m-%d")}')
    diagnosis_lines.append('\n## 说明')
    diagnosis_lines.append('- 跟踪每个关键化合物在各过滤阶段的状态')
    diagnosis_lines.append('- 过滤阶段：OB/DL → SMILES → MW校验 → Lipinski → BBB → PAINS')
    diagnosis_lines.append('\n## 诊断结果')
    diagnosis_lines.append('')
    diagnosis_lines.append('| 化合物 | 中文名 | OB% | DL | OB/DL | SMILES | MW校验 | Lipinski | BBB | PAINS | 放宽池 | 严格池 | 备注 |')
    diagnosis_lines.append('|--------|--------|-----|----|-------|--------|--------|----------|-----|-------|--------|--------|------|')

    for eng_name, cn_name in KEY_COMPOUNDS.items():
        matches = raw[raw['molecule_name'].str.lower() == eng_name.lower()]
        if len(matches) == 0:
            matches = raw[raw['molecule_name'].str.lower().str.contains(eng_name.lower(), na=False)]
        if len(matches) == 0:
            diagnosis_lines.append(f'| {eng_name} | {cn_name} | - | - | ✗ | - | - | - | - | - | ✗ | ✗ | 不在TCMSP中 |')
            continue

        r = matches.iloc[0]
        ob = r['ob']
        dl = r['dl']
        obdl_pass = '✓' if (ob >= 30 and dl >= 0.18) else '✗'

        act_row = active[active['MOL_ID'] == r['MOL_ID']]
        if len(act_row) == 0:
            diagnosis_lines.append(
                f"| {r['molecule_name'][:25]} | {cn_name} | {ob:.1f} | {dl:.3f} | {obdl_pass} | "
                f"✗ | - | - | - | - | ✗ | ✗ | SMILES获取/规范化失败 |"
            )
            continue

        ar = act_row.iloc[0]
        smi_ok = '✓'
        mw_ok = '✓' if ar['SMILES_MATCH_STATUS'] == 'MATCH_OK' else '✗'
        lip_ok = '✓' if ar['Lipinski_Pass'] else '✗'
        bbb_ok = '✓' if ar['BBB_Prediction'] in ['BBB+', 'BBB+/-'] else '✗'
        pains_ok = '✓' if ar['PAINS_Pass'] else '✗'
        relaxed_ok = '✓' if (ar['Lipinski_Pass'] and ar['PAINS_Pass']) else '✗'
        strict_ok = '✓' if (ar['Lipinski_Pass'] and ar['BBB_Prediction'] in ['BBB+', 'BBB+/-'] and ar['PAINS_Pass']) else '✗'

        notes = []
        if ob < 30: notes.append(f'OB={ob:.1f}%<30%')
        if dl < 0.18: notes.append(f'DL={dl:.3f}<0.18')
        if not ar['Lipinski_Pass']: notes.append(f'Lipinski违例={ar["Lipinski_Violations"]}')
        if ar['BBB_Prediction'] == 'BBB-':
            notes.append(f'BBB- (TPSA={ar["TPSA_calc"]:.1f}, LogP={ar["LogP_calc"]:.2f})')
        if not ar['PAINS_Pass']: notes.append('PAINS警示')
        note_str = '; '.join(notes) if notes else '全部通过'

        diagnosis_lines.append(
            f"| {r['molecule_name'][:25]} | {cn_name} | {ob:.1f} | {dl:.3f} | {obdl_pass} | "
            f"{smi_ok} | {mw_ok} | {lip_ok} | {bbb_ok} | {pains_ok} | {relaxed_ok} | {strict_ok} | {note_str} |"
        )

    diagnosis_text = '\n'.join(diagnosis_lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(diagnosis_text)


def print_final_summary(raw, active, strict, relaxed, formulas, herb_map_df):
    """打印最终统计摘要和复方药味覆盖。"""
    print('\n' + '=' * 70)
    print('7. 最终统计摘要')
    print('=' * 70)
    print(f'  TCMSP原始: {len(raw):,}')
    print(f'  OB/DL通过: {len(active):,}')
    print(f'  严格版 (Lipinski+BBB+PAINS): {len(strict)}')
    print(f'  放宽版 (Lipinski+PAINS): {len(relaxed)}')
    print(f'  有中药来源的(放宽版): {(relaxed["n_herb_origins"] > 0).sum()}')

    all_herbs = set()
    for v in formulas.values():
        all_herbs.update(v)

    print(f'\n  复方药味在放宽池中的覆盖:')
    for herb in sorted(all_herbs):
        herb_mol_ids = set(herb_map_df[herb_map_df['herb_cn_name'] == herb]['MOL_ID'])
        in_relaxed = relaxed[relaxed['MOL_ID'].isin(herb_mol_ids)]
        print(f'    {herb}: {len(in_relaxed)} 个')

    print('\n✅ 全部完成！')


def main():
    """放宽版化合物池构建 + 复方分析 + 关键化合物诊断主流程。"""
    global herb_map_df_global

    print('=' * 70)
    print('1. 构建草药-成分映射')
    print('=' * 70)
    raw, smiles_fixed, herb_map_df = load_data(RESULTS)
    herb_map_df_global = herb_map_df
    herb_map = build_herb_map(herb_map_df)
    smiles_map = build_smiles_map(smiles_fixed)
    print(f'  有草药来源的 MOL_ID: {len(herb_map)}')

    print('\n' + '=' * 70)
    print('2. 加载修正后的 SMILES')
    print('=' * 70)
    print(f'  SMILES 映射数: {len(smiles_map)}')

    print('\n' + '=' * 70)
    print('3. OB/DL 过滤 + RDKit 处理')
    print('=' * 70)
    active = obdl_filter(raw)
    print(f'  OB/DL 过滤后: {len(active)}')

    active = map_and_standardize_smiles(active, smiles_map)
    print(f'  有 SMILES: {active["SMILES"].notna().sum()}')
    print(f'  规范化成功: {len(active)}')

    active = compute_descriptors(active)
    active = validate_mw_consistency(active)
    active = add_herb_info(active, herb_map)

    print('\n' + '=' * 70)
    print('4. 生成两版化合物池')
    print('=' * 70)
    strict, relaxed = build_pools(active)
    print(f'  严格版 (Lipinski+BBB+PAINS): {len(strict)} 个')
    print(f'  放宽版 (Lipinski+PAINS, 无BBB): {len(relaxed)} 个')
    print(f'  放宽版多出: {len(relaxed) - len(strict)} 个 ({(len(relaxed)-len(strict))/len(strict)*100:.1f}%)')
    save_pools(strict, relaxed, RESULTS)
    print(f'  严格版保存: {RESULTS / "tcm_compound_pool_strict.csv"}')
    print(f'  放宽版保存: {RESULTS / "tcm_compound_pool_relaxed.csv"}')

    print('\n' + '=' * 70)
    print('5. 大柴胡汤 + 桂枝茯苓丸 各药味主要单体')
    print('=' * 70)
    report_path = RESULTS / 'formula_herb_ingredients_report.md'
    generate_formula_report(active, FORMULAS, report_path)
    print(f'  复方报告保存: {report_path}')

    print('\n' + '=' * 70)
    print('6. 关键化合物过滤阶段诊断')
    print('=' * 70)
    diag_path = RESULTS / 'key_compounds_diagnosis.md'
    generate_key_compounds_diagnosis(raw, active, diag_path)
    print(f'  诊断报告保存: {diag_path}')

    print_final_summary(raw, active, strict, relaxed, FORMULAS, herb_map_df)


if __name__ == '__main__':
    main()
