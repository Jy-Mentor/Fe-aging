# 模块: L1/core_gene_intersection.py
# 功能: 核心基因集汇聚
import pandas as pd
import numpy as np
import os, sys, logging, json

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, 'core_gene_intersection.log'), encoding='utf-8'),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(PROJECT_ROOT, 'L1', 'results')

logger.info('='*60)
logger.info('Step 1: 加载铁衰老96基因')
ferroaging_df = pd.read_csv(os.path.join(RESULT_DIR, 'ferroaging_genes_96.csv'))
ferroaging_genes = set(ferroaging_df['gene_symbol'].str.upper().str.strip())
logger.info('铁衰老基因数: %d', len(ferroaging_genes))

logger.info('='*60)
logger.info('Step 2: 定义RRA显著DEGs')
rra_df = pd.read_csv(os.path.join(RESULT_DIR, 'ferroaging_genes_RRA_intersection.csv'))
rra_df['abs_rank'] = rra_df['MedianRank'].abs()
sig1 = rra_df[(rra_df['abs_rank'] >= 1.0) & (rra_df['N_Datasets'] >= 3)]
logger.info('策略1 (|MedianRank|>=1, N>=3): %d 个基因', len(sig1))
sig2 = rra_df[(rra_df['abs_rank'] >= 0.5) & (rra_df['N_Datasets'] >= 4)]
logger.info('策略2 (|MedianRank|>=0.5, N>=4): %d 个基因', len(sig2))
sig3 = rra_df[(rra_df['Up_Count'] >= rra_df['Down_Count']) & (rra_df['N_Datasets'] >= 3)]
logger.info('策略3 (上调为主, N>=3): %d 个基因', len(sig3))
rra_sig_genes = set(sig1['GeneSymbol'].str.upper().str.strip())
logger.info('RRA显著DEGs: %d', len(rra_sig_genes))

logger.info('='*60)
logger.info('Step 3: 加载WGCNA模块基因')
ilmn_df = pd.read_csv(os.path.join(RESULT_DIR, 'ILMN_probe_to_gene.csv'))
probe_to_gene = dict(zip(ilmn_df['Probe'], ilmn_df['GeneSymbol']))
logger.info('ILMN探针映射: %d', len(probe_to_gene))

wgcna_df = pd.read_csv(os.path.join(RESULT_DIR, 'wgcna_GSE16561', 'gene_module_assignment.csv'))
def map_p2g(pid):
    if pid in probe_to_gene and pd.notna(probe_to_gene[pid]) and probe_to_gene[pid].strip():
        return probe_to_gene[pid].strip().upper()
    return None
wgcna_df['GeneSymbol'] = wgcna_df['Gene'].apply(map_p2g)
wgcna_mapped = wgcna_df[wgcna_df['GeneSymbol'].notna()].copy()
gene_module = wgcna_mapped.groupby('GeneSymbol')['Module'].agg(lambda x: x.value_counts().index[0]).to_dict()
logger.info('基因-模块映射: %d 个唯一基因', len(gene_module))

logger.info('='*60)
logger.info('Step 4: 定义WGCNA显著模块')
mt_cor = pd.read_csv(os.path.join(RESULT_DIR, 'wgcna_GSE16561', 'module_trait_correlation.csv'))
mt_pval = pd.read_csv(os.path.join(RESULT_DIR, 'wgcna_GSE16561', 'module_trait_pvalue.csv'))
cor_col = [c for c in mt_cor.columns if c != 'Module'][0]
pval_col = [c for c in mt_pval.columns if c != 'Module'][0]
mt_cor[cor_col] = pd.to_numeric(mt_cor[cor_col], errors='coerce')
mt_pval[pval_col] = pd.to_numeric(mt_pval[pval_col], errors='coerce')
mt_cor['abs_cor'] = mt_cor[cor_col].abs()
mt_cor['pval'] = mt_pval[pval_col]
sig_modules_df = mt_cor[(mt_cor['abs_cor'] > 0.3) & (mt_cor['pval'] < 0.05)]
logger.info('显著模块: %s', sig_modules_df[['Module', cor_col, 'pval']].to_string())

sig_module_colors = set()
for _, row in sig_modules_df.iterrows():
    mn = row['Module']
    if mn.startswith('ME'):
        sig_module_colors.add(mn[2:])
logger.info('显著模块颜色: %s', sorted(sig_module_colors))

wgcna_module_genes = set(g for g, m in gene_module.items() if m in sig_module_colors)
logger.info('WGCNA显著模块基因数: %d', len(wgcna_module_genes))
for mod in sorted(sig_module_colors):
    logger.info('  模块 %s: %d 个基因', mod, sum(1 for g,m in gene_module.items() if m==mod))

logger.info('='*60)
logger.info('Step 5: 核心基因集汇聚')
fa_rra = ferroaging_genes & rra_sig_genes
fa_wgcna = ferroaging_genes & wgcna_module_genes
core_genes = fa_rra & fa_wgcna
logger.info('铁衰老 \u2229 RRA: %d', len(fa_rra))
logger.info('铁衰老 \u2229 WGCNA: %d', len(fa_wgcna))
logger.info('核心基因集: %d 个基因', len(core_genes))
logger.info('基因: %s', sorted(core_genes))

if len(core_genes) < 5:
    logger.warning('核心基因交集过小 (n=%d), 启用扩大策略', len(core_genes))
    core_genes = fa_rra | fa_wgcna
    logger.info('扩大后: %d 个基因', len(core_genes))

logger.info('='*60)
logger.info('Step 6: 保存结果')
core_details = rra_df[rra_df['GeneSymbol'].str.upper().isin(core_genes)].copy()
core_details['WGCNA_Module'] = core_details['GeneSymbol'].str.upper().map(gene_module)
core_details['FerroAging'] = 'Yes'
core_details = core_details.sort_values('abs_rank', ascending=False)
core_details.to_csv(os.path.join(RESULT_DIR, 'core_genes_final.csv'), index=False, encoding='utf-8-sig')
logger.info('最终核心靶标列表 (%d个):', len(core_details))
for _, row in core_details.iterrows():
    logger.info('  %-12s | RRA=%.2f | N=%d | Module=%s | Dir=%s',
        row['GeneSymbol'], row['MedianRank'], row['N_Datasets'],
        row.get('WGCNA_Module','N/A'), row['Direction'])

# Venn数据
venn_data = {
    'ferroaging_only': sorted(ferroaging_genes - rra_sig_genes - wgcna_module_genes),
    'rra_only': sorted(rra_sig_genes - ferroaging_genes - wgcna_module_genes),
    'wgcna_only': sorted(wgcna_module_genes - ferroaging_genes - rra_sig_genes),
    'fa_rra': sorted(fa_rra - wgcna_module_genes),
    'fa_wgcna': sorted(fa_wgcna - rra_sig_genes),
    'rra_wgcna': sorted((rra_sig_genes & wgcna_module_genes) - ferroaging_genes),
    'core': sorted(core_genes)
}
with open(os.path.join(RESULT_DIR, 'venn_data.json'), 'w', encoding='utf-8') as f:
    json.dump(venn_data, f, ensure_ascii=False, indent=2)
logger.info('Venn: FA=%d RRA=%d WGCNA=%d Core=%d',
    len(venn_data['ferroaging_only']), len(venn_data['rra_only']),
    len(venn_data['wgcna_only']), len(venn_data['core']))
logger.info('Phase 1 核心基因集汇聚完成!')
