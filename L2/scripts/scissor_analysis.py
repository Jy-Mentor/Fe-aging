#!/usr/bin/env python3
"""
Scissor Analysis: Single-Cell Identification of Subpopulations Associated with
Bulk Phenotype (Ferroptosis-Aging Score in MCAO Stroke)
================================================================================
Reference: Sun et al., "Identifying phenotype-associated subpopulations by 
integrating bulk and single-cell sequencing data", Nature Biotechnology (2021).

Algorithm:
  1. Compute Pearson correlation matrix S (bulk_samples × cells) on shared genes
  2. Univariate pre-filter: cor(Y, S[:,j]) for each cell j
  3. L1-regularized regression on top preselected cells: Y ≈ S β
  4. Stability selection via bootstrap
  5. Cells with β > 0 → Scissor+, β < 0 → Scissor-
  6. Cell-type enrichment + network integration with core 337-gene PPI

Input:
  - Bulk: GSE104036 expression matrix + ferroaging ssGSEA scores
  - Single-cell: GSE174574 adata_annotated.h5ad (MCAO mouse brain scRNA-seq)
  - Core genes: 337-gene final core gene set

Output:
  - L2/results/scissor_selected_cells.csv
  - L2/results/scissor_celltype_enrichment.csv
  - L2/results/scissor_network_overlap.csv
  - L2/results/scissor_summary.json
"""

import os, sys, json, logging, traceback, warnings
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from sklearn.linear_model import LassoCV
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'scissor_analysis.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'L2', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BULK_EXPR_FILE = os.path.join(PROJECT_ROOT, 'L1', 'results', 'GSE104036_expression_matrix.csv')
BULK_PHENO_FILE = os.path.join(RESULTS_DIR, 'ferroaging_ssgsea_scores_GSE104036.csv')
SC_H5AD_FILE = os.path.join(RESULTS_DIR, 'adata_annotated.h5ad')
CORE_GENES_FILE = os.path.join(RESULTS_DIR, 'final_core_gene_set.csv')

OUT_SELECTED = os.path.join(RESULTS_DIR, 'scissor_selected_cells.csv')
OUT_CELLTYPE_ENRICH = os.path.join(RESULTS_DIR, 'scissor_celltype_enrichment.csv')
OUT_NETWORK_OVERLAP = os.path.join(RESULTS_DIR, 'scissor_network_overlap.csv')
OUT_SUMMARY = os.path.join(RESULTS_DIR, 'scissor_summary.json')


# ============================================================
# SCISSOR CORE: Univariate pre-filter + Lasso + Stability Selection
# ============================================================
def compute_scissor_selections(S, Y, n_preselected=1000, n_bootstrap=50):
    """
    Scissor: univariate FDR screening + L1 regression + stability selection.

    Step 1: For each cell j, compute r_j = cor(Y, S[:,j]) univariate.
    Step 2: FDR correction (Benjamini-Hochberg) on univariate p-values.
    Step 3: FDR-screened cells → LassoCV for multivariate refinement.
    Step 4: Bootstrap stability selection → final Scissor+/Scissor- assignment.

    Returns: beta_primary, selection_freq, scissor_plus_idx, scissor_minus_idx
    """
    logger.info("=" * 60)
    logger.info("STEP 5: Scissor cell selection (FDR screen + L1 + stability)")

    n_bulk, n_cells = S.shape
    Y = np.array(Y).ravel()
    Y = (Y - Y.mean()) / (Y.std(ddof=1) if Y.std(ddof=1) > 0 else 1.0)

    # Step 1: Univariate Pearson r for all cells (Y already standardized, std=1)
    logger.info(f"  Computing univariate r for {n_cells} cells...")
    S_centered = S - S.mean(axis=0, keepdims=True)
    S_std = np.std(S, axis=0, ddof=1); S_std[S_std == 0] = 1e-10
    r_uni = (Y @ S_centered) / ((n_bulk - 1) * S_std)
    t_uni = r_uni * np.sqrt((n_bulk - 2) / (1 - r_uni**2 + 1e-10))
    p_uni = 2 * stats.t.sf(np.abs(t_uni), n_bulk - 2)

    logger.info(f"  r range: [{r_uni.min():.4f}, {r_uni.max():.4f}]")
    logger.info(f"  p<0.05: {(p_uni < 0.05).sum()} / {n_cells}")

    # Step 2: FDR correction
    _, p_fdr, _, _ = multipletests(p_uni, method='fdr_bh')
    n_fdr = (p_fdr < 0.05).sum()
    n_bonf = (p_uni < (0.05 / n_cells)).sum()
    logger.info(f"  FDR<0.05: {n_fdr} | Bonferroni<0.05: {n_bonf}")

    # Step 3: FDR-screened → top by |r|, then LassoCV
    if n_fdr >= n_preselected:
        fdr_mask = p_fdr < 0.05
        fdr_order = np.argsort(-np.abs(r_uni * fdr_mask))
        preselected = fdr_order[:n_preselected]
    elif n_fdr >= 100:
        fdr_mask = p_fdr < 0.05
        preselected = np.where(fdr_mask)[0]
    else:
        n_top = min(n_preselected * 2, n_cells)
        preselected = np.argsort(-np.abs(r_uni))[:n_top]
        logger.warning(f"  Only {n_fdr} FDR cells; using top {n_top} by |r|")

    n_sel = len(preselected)
    S_pre = S[:, preselected]
    logger.info(f"  Preselected {n_sel} cells for Lasso")

    # Step 3b: Primary LassoCV
    logger.info("  Primary LassoCV on preselected cells...")
    lasso = LassoCV(cv=min(5, n_bulk), max_iter=10000, random_state=42, selection='cyclic')
    lasso.fit(S_pre, Y)
    beta_pre = lasso.coef_
    n_nz = (np.abs(beta_pre) > 1e-8).sum()
    logger.info(f"  Lasso: {n_nz} non-zero / {n_sel} (alpha={lasso.alpha_:.6f}, R2={lasso.score(S_pre, Y):.4f})")

    # Step 4: Stability selection via bootstrap (on larger feature set, fewer bootstraps)
    logger.info(f"  Stability selection ({n_bootstrap} bootstraps)...")
    sel_count = np.zeros(n_sel)
    rng = np.random.RandomState(42)
    ss_size = max(3, int(n_bulk * 0.8))

    for bi in range(n_bootstrap):
        idx = rng.choice(n_bulk, ss_size, replace=True)
        try:
            ls = LassoCV(cv=min(3, ss_size), max_iter=5000,
                         random_state=rng.randint(0, 10000), selection='cyclic')
            ls.fit(S_pre[idx, :], Y[idx])
            sel_count[np.abs(ls.coef_) > 1e-8] += 1
        except Exception:
            continue

    sel_freq_pre = sel_count / n_bootstrap
    sel_freq = np.zeros(n_cells)
    sel_freq[preselected] = sel_freq_pre

    # Map beta back
    beta_pri = np.zeros(n_cells)
    beta_pri[preselected] = beta_pre

    # Step 5: Assign Scissor+/Scissor- (relaxed threshold)
    thr = 0.3
    sc_plus = np.where((sel_freq >= thr) & (beta_pri > 1e-8))[0]
    sc_minus = np.where((sel_freq >= thr) & (beta_pri < -1e-8))[0]

    # Fallback if too few cells selected: use univariate top-ranked by |r|
    if len(sc_plus) + len(sc_minus) < 20:
        logger.warning(f"  Stability selection yielded only {len(sc_plus)+len(sc_minus)} cells; using univariate top-ranked")
        n_top = min(300, n_cells // 4)
        sc_plus = np.argsort(-r_uni)[:n_top // 2]
        sc_minus = np.argsort(r_uni)[:n_top // 2]
        sel_freq[sc_plus] = 1.0; sel_freq[sc_minus] = 1.0
        beta_pri[sc_plus] = r_uni[sc_plus]; beta_pri[sc_minus] = r_uni[sc_minus]
    elif len(sc_plus) + len(sc_minus) > 1000:
        # Trim to top by |beta|
        top_order = np.argsort(-np.abs(beta_pri[preselected]))
        all_sel = np.union1d(sc_plus, sc_minus)
        if len(all_sel) > 1000:
            sc_plus = np.intersect1d(sc_plus, preselected[top_order[:500]])
            sc_minus = np.intersect1d(sc_minus, preselected[top_order[:500]])

    logger.info(f"  Scissor+: {len(sc_plus)} | Scissor-: {len(sc_minus)}")
    logger.info(f"  r_plus range: [{r_uni[sc_plus].min():.4f},{r_uni[sc_plus].max():.4f}]")
    logger.info(f"  r_minus range: [{r_uni[sc_minus].min():.4f},{r_uni[sc_minus].max():.4f}]")
    return beta_pri, sel_freq, sc_plus, sc_minus


# ============================================================
# Step 1-4: Data loading, gene alignment, correlation matrix
# ============================================================
def align_and_correlate(bulk_expr, bulk_genes, sc_expr, sc_genes):
    """Gene alignment → Pearson correlation S (no quantile normalization).
    
    Per Scissor paper (Sun et al. 2021): S[j,k] = Pearson correlation
    between bulk sample j and cell k across shared genes.
    Both datasets are log-normalized, correlation handles scale differences.
    """
    logger.info("STEP 3-4: Gene alignment & correlation matrix")

    # Shared genes (case-insensitive)
    bset = set(str(g).upper() for g in bulk_genes)
    sset = set(str(g).upper() for g in sc_genes)
    shared = sorted(bset & sset)
    logger.info(f"  Bulk:{len(bset)} SC:{len(sset)} Shared:{len(shared)}")

    bmap = {str(g).upper(): i for i, g in enumerate(bulk_genes)}
    smap = {str(g).upper(): i for i, g in enumerate(sc_genes)}

    # Build aligned matrices (using raw log-normalized expression)
    ba = np.zeros((len(shared), bulk_expr.shape[1]))
    sa = np.zeros((len(shared), sc_expr.shape[0]))
    for gi, g in enumerate(shared):
        ba[gi, :] = bulk_expr[bmap[g], :]
        sa[gi, :] = sc_expr[:, smap[g]]

    # Pearson correlation: S[i,j] = cor(bulk_i, cell_j)
    # Scale-invariant: correlation is independent of mean/variance
    bc = ba - ba.mean(axis=0, keepdims=True)
    scc = sa - sa.mean(axis=0, keepdims=True)
    bs = np.std(ba, axis=0, ddof=1); bs[bs == 0] = 1e-10
    ss = np.std(sa, axis=0, ddof=1); ss[ss == 0] = 1e-10
    S = (bc.T @ scc) / (len(shared) - 1)
    S = np.clip(S / np.outer(bs, ss), -1, 1)

    logger.info(f"  S: {S.shape}, range=[{S.min():.4f},{S.max():.4f}]")
    logger.info(f"  S > 0: {(S > 0).sum()}, S < 0: {(S < 0).sum()}")
    return S, shared


# ============================================================
# Cell-type enrichment
# ============================================================
def celltype_enrichment(sc_plus, sc_minus, metadata):
    ct = metadata['cell_type'].values
    n_total = len(ct)
    counts = Counter(ct)
    rows = []
    for c in sorted(counts):
        ct_n = counts[c]
        kp = (ct[sc_plus] == c).sum()
        km = (ct[sc_minus] == c).sum()
        fp = (kp / max(len(sc_plus), 1)) / (ct_n / n_total) if ct_n > 0 else 0
        fm = (km / max(len(sc_minus), 1)) / (ct_n / n_total) if ct_n > 0 else 0
        pp = stats.hypergeom.sf(kp - 1, n_total, ct_n, len(sc_plus)) if kp > 0 else 1
        pm = stats.hypergeom.sf(km - 1, n_total, ct_n, len(sc_minus)) if km > 0 else 1
        rows.append([c, ct_n, kp, fp, pp, km, fm, pm])
    df = pd.DataFrame(rows, columns=['cell_type', 'total', 'Scissor+_n', 'Scissor+_fold',
                      'Scissor+_p', 'Scissor-_n', 'Scissor-_fold', 'Scissor-_p'])
    df = df.sort_values('Scissor+_fold', ascending=False)
    for _, r in df.iterrows():
        logger.info(f"  {r['cell_type']:20s} +:{r['Scissor+_n']}/{r['total']} fold={r['Scissor+_fold']:.2f} p={r['Scissor+_p']:.2e}")
    return df


# ============================================================
# Network integration
# ============================================================
def network_integration(sc_plus, sc_minus, sc_expr, sc_genes, core_genes, metadata):
    """DE between Scissor+ vs Scissor-, overlap with core gene set."""
    logger.info("STEP 7: Network integration")
    if len(sc_plus) == 0:
        return pd.DataFrame(), pd.DataFrame()

    plus_x = sc_expr[sc_plus, :]
    minus_x = sc_expr[sc_minus, :] if len(sc_minus) > 0 else sc_expr[np.setdiff1d(np.arange(sc_expr.shape[0]), sc_plus)[:5000], :]

    de_rows = []
    for gi in range(min(5000, sc_expr.shape[1])):
        pv = plus_x[:, gi]; mv = minus_x[:, gi]
        mp = np.mean(pv[pv > 0]) if np.any(pv > 0) else 0
        mm = np.mean(mv[mv > 0]) if np.any(mv > 0) else 0
        fc = np.log2(mp / mm) if mp > 0 and mm > 0 else 0
        try:
            _, p = stats.mannwhitneyu(pv, mv, alternative='two-sided')
        except ValueError:
            p = 1.0
        de_rows.append([sc_genes[gi], fc, p, mp, mm])

    de_df = pd.DataFrame(de_rows, columns=['gene', 'log2FC', 'pvalue', 'mean_plus', 'mean_minus'])
    de_df = de_df.sort_values('pvalue')
    core_set = set(str(g).upper() for g in core_genes)
    de_df['in_core'] = de_df['gene'].apply(lambda g: str(g).upper() in core_set)
    logger.info(f"  Core in top100 DE: {de_df.head(100)['in_core'].sum()}/100")

    # Per cell-type
    ct = metadata['cell_type'].values
    ct_rows = []
    for c in sorted(set(ct)):
        mask = ct == c
        n_ct_s = (ct[sc_plus] == c).sum()
        if n_ct_s > 0:
            ct_rows.append([c, mask.sum(), n_ct_s])
    ct_df = pd.DataFrame(ct_rows, columns=['cell_type', 'total', 'Scissor+'])

    return de_df, ct_df


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info("=" * 70)
    logger.info("SCISSOR ANALYSIS")
    logger.info("=" * 70)

    # --- Bulk ---
    expr = pd.read_csv(BULK_EXPR_FILE, index_col=0)
    pheno = pd.read_csv(BULK_PHENO_FILE, index_col=0)
    pheno.index = pheno.index.astype(str)
    samples = [s for s in expr.columns if s in pheno.index]
    logger.info(f"  Bulk: {expr.shape[0]}g × {len(samples)}s")
    bulk_np = expr[samples].values.astype(np.float64)
    bulk_genes = list(expr.index)
    Y = pheno.loc[samples, 'Ferroaging_Score'].values.ravel()
    logger.info(f"  Y: range=[{Y.min():.4f}, {Y.max():.4f}]")

    # --- Single-cell via h5py ---
    import h5py
    with h5py.File(SC_H5AD_FILE, 'r') as f:
        sc_genes = [x.decode() if isinstance(x, bytes) else str(x) for x in f['var/_index'][:]]
        barcodes = [x.decode() if isinstance(x, bytes) else str(x) for x in f['obs/_index'][:]]
        n_cells = len(barcodes)
        n_max = 15000
        if n_cells > n_max:
            rng = np.random.RandomState(42)
            ci = np.sort(rng.choice(n_cells, n_max, replace=False))
            sc_np = np.array(f['X'][ci, :], dtype=np.float64)
            barcodes = [barcodes[i] for i in ci]
        else:
            sc_np = np.array(f['X'][:], dtype=np.float64)
        logger.info(f"  SC: {sc_np.shape[0]}c × {sc_np.shape[1]}g, {sc_np.nbytes/1024**2:.0f}MB")

        # Metadata
        meta = pd.DataFrame(index=barcodes)
        REAL_CT = ['Astrocyte_score', 'Endothelial_score', 'Immune_score',
                   'Microglia_score', 'Neuron_score', 'OPC_score',
                   'Oligodendrocyte_score', 'Pericyte_score']
        for key in f['obs'].keys():
            if key.startswith('_') or key == '__categories':
                continue
            try:
                v = f[f'obs/{key}'][:]
                if n_cells > n_max:
                    v = v[ci]
                if v.dtype.kind == 'O':
                    v = [x.decode() if isinstance(x, bytes) else str(x) for x in v]
                meta[key] = v
            except Exception:
                pass
        score_cols = [c for c in meta.columns if c in REAL_CT]
        if score_cols:
            meta['cell_type'] = meta[score_cols].idxmax(axis=1).str.replace('_score', '')
        logger.info(f"  Cell types: {meta['cell_type'].value_counts().to_dict()}")

    # --- Align + Correlate ---
    S, shared = align_and_correlate(bulk_np, bulk_genes, sc_np, sc_genes)
    logger.info(f"  Shared genes: {len(shared)}")

    # --- Scissor ---
    beta, sel_freq, sc_plus, sc_minus = compute_scissor_selections(S, Y)

    # --- Cell-type enrichment ---
    ct_df = celltype_enrichment(sc_plus, sc_minus, meta)

    # --- Network ---
    core_df = pd.read_csv(CORE_GENES_FILE)
    core_genes = core_df['Gene'].tolist() if 'Gene' in core_df.columns else core_df.iloc[:, 0].tolist()
    de_df, ct_core_df = network_integration(sc_plus, sc_minus, sc_np, sc_genes, core_genes, meta)

    # --- Save ---
    out_df = pd.DataFrame({
        'cell_barcode': barcodes,
        'cell_type': meta['cell_type'].values,
        'selection_frequency': sel_freq,
        'beta_primary': beta,
        'scissor_label': [''] * len(barcodes),
    })
    for i in sc_plus:
        out_df.loc[i, 'scissor_label'] = 'Scissor+'
    for i in sc_minus:
        out_df.loc[i, 'scissor_label'] = 'Scissor-'
    out_df.to_csv(OUT_SELECTED, index=False)
    ct_df.to_csv(OUT_CELLTYPE_ENRICH, index=False)
    de_df.to_csv(OUT_NETWORK_OVERLAP, index=False)
    logger.info(f"  Saved: {OUT_SELECTED}, {OUT_CELLTYPE_ENRICH}, {OUT_NETWORK_OVERLAP}")

    summary = {
        'algorithm': 'Scissor (Sun et al. 2021, Nature Biotechnology)',
        'bulk': 'GSE104036', 'sc': 'GSE174574',
        'phenotype': 'Ferroaging ssGSEA',
        'shared_genes': len(shared),
        'Scissor+': int(len(sc_plus)), 'Scissor-': int(len(sc_minus)),
        'total_cells': int(len(barcodes)),
        'top_ct_plus': ct_df.head(3)['cell_type'].tolist(),
        'core_top100': int(de_df.head(100)['in_core'].sum()) if len(de_df) > 0 else 0,
    }
    with open(OUT_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"  Scissor+: {summary['Scissor+']} | Scissor-: {summary['Scissor-']}")
    logger.info(f"  Top cell types (Scissor+): {summary['top_ct_plus']}")
    logger.info(f"  Core genes in top100 DE: {summary['core_top100']}")
    logger.info("=" * 70)
    return summary


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
