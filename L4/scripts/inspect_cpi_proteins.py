import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from pathlib import Path

root = Path('L4/results_v10_minibatch')

cpi_df = pd.read_csv('L4/results/experimental_actives_detail_cleaned.csv', low_memory=False)
cpi_genes = sorted(cpi_df['gene'].unique())
print(f'CPI genes: {len(cpi_genes)}')
print(f'sample: {cpi_genes[:10]}')

# global
global_d = np.load(root/'esm2_protein_embeddings.npz', allow_pickle=True)
global_genes = set(global_d.keys())
print(f'\nGlobal embeddings: {len(global_genes)}')
print(f'CPI genes in global: {len([g for g in cpi_genes if g in global_genes])}/{len(cpi_genes)}')

# residue pooled
pool_d = np.load(root/'esm2_residue_pooled_embeddings.npz', allow_pickle=True)
pool_genes = set(pool_d.keys())
print(f'\nResidue pooled embeddings: {len(pool_genes)}')
print(f'CPI genes in residue pooled: {len([g for g in cpi_genes if g in pool_genes])}/{len(cpi_genes)}')

# raw residue
res_d = torch.load(root/'esm2_150M_residue_features.pt', map_location='cpu')
res_genes = set(res_d['genes'])
print(f'\nRaw residue embeddings: {len(res_genes)}')
print(f'CPI genes in raw residue: {len([g for g in cpi_genes if g in res_genes])}/{len(cpi_genes)}')

# missing
missing_global = [g for g in cpi_genes if g not in global_genes]
missing_pool = [g for g in cpi_genes if g not in pool_genes]
print(f'\nMissing in global: {missing_global[:20]}')
print(f'Missing in residue pooled: {missing_pool[:20]}')
