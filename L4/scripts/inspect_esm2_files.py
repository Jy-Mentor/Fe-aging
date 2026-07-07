import logging
logger = logging.getLogger(__name__)

import numpy as np
import torch
from pathlib import Path

p = Path('L4/results_v10_minibatch')

print('=== esm2_protein_embeddings.npz (global CLS) ===')
d = np.load(p/'esm2_protein_embeddings.npz', allow_pickle=True)
keys = list(d.keys())
print(f'keys count: {len(keys)}')
print(f'sample keys: {keys[:5]}')
for k in keys[:3]:
    arr = d[k]
    print(f'{k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}')

print('\n=== esm2_residue_pooled_embeddings.npz ===')
d = np.load(p/'esm2_residue_pooled_embeddings.npz', allow_pickle=True)
keys = list(d.keys())
print(f'keys count: {len(keys)}')
print(f'sample keys: {keys[:5]}')
for k in keys[:3]:
    arr = d[k]
    print(f'{k}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}')

print('\n=== esm2_150M_residue_features.pt (per-residue) ===')
d = torch.load(p/'esm2_150M_residue_features.pt', map_location='cpu')
print('type:', type(d))
print('keys:', list(d.keys()))
if isinstance(d, dict):
    n = len(d['genes']) if 'genes' in d else 'N/A'
    print(f'genes count: {n}')
    for k in list(d.keys()):
        v = d[k]
        if hasattr(v, 'shape'):
            print(f'{k}: shape={v.shape}, dtype={v.dtype}')
        elif hasattr(v, '__len__') and len(v) > 0:
            sample = v[0]
            if hasattr(sample, 'shape'):
                print(f'{k}: list len={len(v)}, sample shape={sample.shape}, dtype={sample.dtype}')
            else:
                print(f'{k}: list len={len(v)}, sample type={type(sample)}, value={sample}')
        else:
            print(f'{k}: type={type(v)}')
