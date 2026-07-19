#!/usr/bin/env python3
"""
Phase 2 - Single-cell RNA-seq Analysis Pipeline
================================================
GSE174574: Mouse MCAO brain scRNA-seq (24h post-stroke)
3 Sham + 3 MCAO samples, 10X Genomics platform

Steps:
  1. Load 10X data from 6 samples
  2. QC filtering + doublet detection (scrublet)
  3. Normalization, log1p, HVG selection
  4. PCA + Harmony batch correction
  5. UMAP + Leiden clustering
  6. Cell type annotation (CellTypist + manual markers)
  7. Ferroaging score calculation (96 genes)
  8. Pseudobulk DE analysis (MCAO vs Sham per cell type)
  9. Visualization

Input:
  - L1 数据集/RNA-seq/GSE174574_10X_organized/{MCAO,Sham}_*/ (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)
  - 铁衰老基因.txt (96 ferroaging genes)

Output:
  - L2/results/adata_qc.h5ad
  - L2/results/adata_processed.h5ad
  - L2/results/adata_annotated.h5ad
  - L2/results/sc_ferroaging_scores.csv
  - L2/results/sc_pseudobulk_de.csv
  - L2/results/figures/ (UMAP, violin, dot plots)
  - L2/results/sc_metadata.csv

Usage:
  python L2/sc_pipeline.py
"""

import os
import sys
import logging
import traceback
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Logging Setup
# ============================================================
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'sc_pipeline.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=True)

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'L1 数据集', 'RNA-seq', 'GSE174574_10X_organized')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'L2', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
FERROAGING_GENES_FILE = os.path.join(PROJECT_ROOT, '铁衰老基因.txt')
CORE_GENES_FILE = os.path.join(PROJECT_ROOT, 'L1', 'results', 'core_genes_final.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Sample definitions
SAMPLES = {
    'Sham_1': 'Sham',
    'Sham_2': 'Sham',
    'Sham_3': 'Sham',
    'MCAO_1': 'MCAO',
    'MCAO_2': 'MCAO',
    'MCAO_3': 'MCAO',
}

# Known mouse brain cell type markers
MARKER_GENES = {
    'Neuron': ['Tubb3', 'Map2', 'Rbfox3', 'Syp', 'Snap25'],
    'Astrocyte': ['Gfap', 'Aqp4', 'Aldh1l1', 'Slc1a3', 'S100b'],
    'Microglia': ['Aif1', 'Cx3cr1', 'Tmem119', 'P2ry12', 'Csf1r'],
    'Oligodendrocyte': ['Mbp', 'Mog', 'Plp1', 'Olig1', 'Olig2'],
    'OPC': ['Pdgfra', 'Cspg4', 'Sox10', 'Vcan'],
    'Endothelial': ['Cldn5', 'Pecam1', 'Tek', 'Cdh5', 'Flt1'],
    'Pericyte': ['Pdgfrb', 'Rgs5', 'Cspg4', 'Anpep'],
    'Immune': ['Ptprc', 'Cd3e', 'Cd4', 'Cd8a', 'Cd19'],
}


def load_ferroaging_genes():
    """Load 96 ferroaging genes from file."""
    logger.info("Loading ferroaging genes from %s", FERROAGING_GENES_FILE)
    with open(FERROAGING_GENES_FILE, 'r', encoding='utf-8') as f:
        genes = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d ferroaging genes", len(genes))
    return genes


def load_core_genes():
    """Load 28 core target genes from P1 results."""
    logger.info("Loading core target genes from %s", CORE_GENES_FILE)
    df = pd.read_csv(CORE_GENES_FILE)
    genes = df['GeneSymbol'].tolist()
    logger.info("Loaded %d core target genes", len(genes))
    return genes


def human_to_mouse_gene(human_genes):
    """Convert human gene symbols to mouse gene symbols.
    
    Most human gene symbols are uppercase and mouse are title-case (e.g., ACSL4 -> Acsl4).
    Also handles genes with different names between species.
    
    Returns:
        dict: {human_gene: mouse_gene}
    """
    # Known human-mouse gene symbol differences (comprehensive mapping)
    KNOWN_MAP = {
        # Ferroptosis / Iron metabolism
        'ACSL4': 'Acsl4', 'HMOX1': 'Hmox1', 'TFRC': 'Tfrc', 'GPX4': 'Gpx4',
        'HIF1A': 'Hif1a', 'KEAP1': 'Keap1', 'SOD1': 'Sod1', 'NLRP3': 'Nlrp3',
        'IL6': 'Il6', 'TLR4': 'Tlr4', 'MAPK1': 'Mapk1', 'PTGS2': 'Ptgs2',
        'CXCL10': 'Cxcl10', 'LCN2': 'Lcn2', 'IL1B': 'Il1b', 'CD74': 'Cd74',
        'IRF1': 'Irf1', 'SP1': 'Sp1', 'KLF6': 'Klf6', 'EGR1': 'Egr1',
        'BCL6': 'Bcl6', 'CTSB': 'Ctsb', 'SAT1': 'Sat1', 'KDM6B': 'Kdm6b',
        'LGMN': 'Lgmn', 'IGFBP7': 'Igfbp7', 'PDE4B': 'Pde4b', 'EMP1': 'Emp1',
        'EPHA4': 'Epha4', 'RUNX3': 'Runx3', 'FBXO31': 'Fbxo31',
        'LPCAT3': 'Lpcat3', 'DYRK1A': 'Dyrk1a', 'LACTB': 'Lactb',
        'GMFB': 'Gmfb', 'HBP1': 'Hbp1', 'MAPK14': 'Mapk14',
        # Other ferroaging genes
        'ABCC1': 'Abcc1', 'ACVR1B': 'Acvr1b', 'ALOX15': 'Alox15',
        'ATF3': 'Atf3', 'ATG3': 'Atg3', 'BAP1': 'Bap1', 'BRD7': 'Brd7',
        'CAVIN1': 'Cavin1', 'CD82': 'Cd82', 'CDO1': 'Cdo1',
        'COX7A1': 'Cox7a1', 'DPEP1': 'Dpep1', 'DPP4': 'Dpp4',
        'DUOX1': 'Duox1', 'E2F1': 'E2f1', 'E2F3': 'E2f3', 'EBF3': 'Ebf3',
        'EDN1': 'Edn1', 'EPHA2': 'Epha2', 'ERN1': 'Ern1',
        'FOSL1': 'Fosl1', 'HERPUD1': 'Herpud1', 'HMGB1': 'Hmgb1',
        'ICA1': 'Ica1', 'IFNG': 'Ifng', 'IRF7': 'Irf7', 'IRF9': 'Irf9',
        'LIFR': 'Lifr', 'LOX': 'Lox', 'MAP3K14': 'Map3k14',
        'MCU': 'Mcu', 'MEN1': 'Men1', 'MPO': 'Mpo', 'NOX4': 'Nox4',
        'NR1D1': 'Nr1d1', 'NR2F2': 'Nr2f2', 'NUAK2': 'Nuak2',
        'PADI4': 'Padi4', 'PPP2R2B': 'Ppp2r2b', 'PRKD1': 'Prkd1',
        'PTBP1': 'Ptbp1', 'RBM3': 'Rbm3', 'S100A8': 'S100a8',
        'SETD7': 'Setd7', 'SLAMF8': 'Slamf8', 'SLC1A5': 'Slc1a5',
        'SMARCB1': 'Smarcb1', 'SMURF2': 'Smurf2', 'SNCA': 'Snca',
        'SOCS1': 'Socs1', 'SOCS2': 'Socs2', 'SPATA2': 'Spata2',
        'TBX2': 'Tbx2', 'TNFAIP1': 'Tnfaip1', 'TNFAIP3': 'Tnfaip3',
        'TXNIP': 'Txnip', 'WNT5A': 'Wnt5a', 'WWTR1': 'Wwtr1', 'YAP1': 'Yap1',
        'ZEB1': 'Zeb1',
    }
    
    mouse_genes = {}
    for hg in human_genes:
        if hg in KNOWN_MAP:
            mouse_genes[hg] = KNOWN_MAP[hg]
        else:
            # Default: capitalize first letter, lowercase rest
            mouse_genes[hg] = hg.capitalize()
    
    n_mapped = len([g for g in human_genes if g in mouse_genes])
    logger.info("Human->Mouse gene conversion: %d/%d genes mapped", n_mapped, len(human_genes))
    return mouse_genes


def step1_load_data():
    """Step 1: Load 10X Genomics data for all 6 samples."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading 10X data for %d samples", len(SAMPLES))
    logger.info("=" * 60)

    adatas = {}
    for sample_name, group in SAMPLES.items():
        sample_dir = os.path.join(DATA_DIR, sample_name)
        logger.info("Loading %s from %s", sample_name, sample_dir)

        if not os.path.exists(sample_dir):
            logger.error("Sample directory not found: %s", sample_dir)
            raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

        try:
            adata = sc.read_10x_mtx(
                sample_dir,
                var_names='gene_symbols',
                cache=True,
                gex_only=True
            )
            adata.var_names_make_unique()
            adata.obs['sample'] = sample_name
            adata.obs['group'] = group
            adatas[sample_name] = adata
            logger.info("  %s: %d cells, %d genes", sample_name, adata.n_obs, adata.n_vars)
        except Exception as e:
            logger.error("Failed to load %s: %s", sample_name, e)
            traceback.print_exc()
            raise

    # Concatenate all samples
    logger.info("Concatenating all samples...")
    adata = adatas[list(adatas.keys())[0]].concatenate(
        [adatas[k] for k in list(adatas.keys())[1:]],
        batch_key='batch',
        batch_categories=list(SAMPLES.keys()),
        join='outer'
    )
    logger.info("Combined: %d cells, %d genes", adata.n_obs, adata.n_vars)

    return adata


def step2_qc(adata):
    """Step 2: Quality control filtering."""
    logger.info("=" * 60)
    logger.info("STEP 2: Quality Control")
    logger.info("=" * 60)

    # Calculate QC metrics
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    adata.var['ribo'] = adata.var_names.str.startswith(('Rps', 'Rpl', 'Mrps', 'Mrpl'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)

    # Log pre-filtering stats
    logger.info("Pre-filtering stats:")
    logger.info("  Total cells: %d", adata.n_obs)
    logger.info("  Median genes: %d", np.median(adata.obs['n_genes_by_counts']))
    logger.info("  Median UMIs: %d", np.median(adata.obs['total_counts']))
    logger.info("  Median MT%%: %.2f", np.median(adata.obs['pct_counts_mt']))

    # Filtering criteria (mouse brain tissue)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=6000)
    adata = adata[adata.obs['total_counts'] < 40000, :]
    adata = adata[adata.obs['pct_counts_mt'] < 25, :].copy()
    sc.pp.filter_genes(adata, min_cells=3)

    logger.info("Post-filtering stats:")
    logger.info("  Total cells: %d", adata.n_obs)
    logger.info("  Total genes: %d", adata.n_vars)

    # QC violin plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sc.pl.violin(adata, 'n_genes_by_counts', ax=axes[0], show=False)
    sc.pl.violin(adata, 'total_counts', ax=axes[1], show=False)
    sc.pl.violin(adata, 'pct_counts_mt', ax=axes[2], show=False)
    fig.suptitle('QC Metrics (Post-filtering)', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'qc_violin.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # QC by sample
    qc_stats = adata.obs.groupby('sample').agg(
        n_cells=('n_genes_by_counts', 'count'),
        median_genes=('n_genes_by_counts', 'median'),
        median_UMIs=('total_counts', 'median'),
        median_MT_pct=('pct_counts_mt', 'median')
    ).reset_index()
    qc_stats.to_csv(os.path.join(RESULTS_DIR, 'qc_stats_by_sample.csv'), index=False)
    logger.info("QC stats saved to qc_stats_by_sample.csv")

    return adata


def step3_doublet_detection(adata):
    """Step 3: Doublet detection using Scrublet."""
    logger.info("=" * 60)
    logger.info("STEP 3: Doublet Detection (Scrublet)")
    logger.info("=" * 60)

    try:
        sc.external.pp.scrublet(adata, batch_key='sample')
        n_doublets = adata.obs['predicted_doublet'].sum()
        logger.info("Detected %d doublets (%.2f%%)", n_doublets, 100 * n_doublets / adata.n_obs)
        adata.obs['doublet_score'] = adata.obs['doublet_score'].astype(float)
    except Exception as e:
        logger.warning("Scrublet failed, skipping doublet detection: %s", e)
        logger.warning("Traceback: %s", traceback.format_exc())
        adata.obs['predicted_doublet'] = False
        adata.obs['doublet_score'] = 0.0

    return adata


def step4_preprocess(adata):
    """Step 4: Normalization, HVG selection, log1p."""
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocessing (Normalization, HVG, Log1p)")
    logger.info("=" * 60)

    # Save raw counts
    adata.raw = adata.copy()

    # Normalize to 10,000 counts per cell
    sc.pp.normalize_total(adata, target_sum=1e4)
    logger.info("Normalized to 10,000 counts per cell")

    # Log1p transform
    sc.pp.log1p(adata)
    logger.info("Log1p transformed")

    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3', batch_key='sample')
    n_hvg = adata.var['highly_variable'].sum()
    logger.info("Selected %d highly variable genes", n_hvg)

    return adata


def step5_pca_harmony(adata):
    """Step 5: PCA + Harmony batch correction."""
    logger.info("=" * 60)
    logger.info("STEP 5: PCA + Harmony Batch Correction")
    logger.info("=" * 60)

    # Scale data
    sc.pp.scale(adata, max_value=10)
    logger.info("Data scaled")

    # PCA
    sc.tl.pca(adata, n_comps=50, svd_solver='arpack', use_highly_variable=True)
    logger.info("PCA completed: %d components", adata.obsm['X_pca'].shape[1])

    # PCA variance ratio plot
    fig, ax = plt.subplots(figsize=(8, 4))
    sc.pl.pca_variance_ratio(adata, n_pcs=50, show=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'pca_variance_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Harmony batch correction
    try:
        sc.external.pp.harmony_integrate(adata, key='sample', basis='X_pca', adjusted_basis='X_pca_harmony')
        logger.info("Harmony integration completed")
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony'].copy()
    except Exception as e:
        logger.warning("Harmony integration failed, using uncorrected PCA: %s", e)
        logger.warning("Traceback: %s", traceback.format_exc())

    return adata


def step6_umap_clustering(adata):
    """Step 6: UMAP visualization + Leiden clustering."""
    logger.info("=" * 60)
    logger.info("STEP 6: UMAP + Leiden Clustering")
    logger.info("=" * 60)

    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    logger.info("Neighbors computed")

    # UMAP
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    logger.info("UMAP computed")

    # Leiden clustering at multiple resolutions
    for res in [0.4, 0.8, 1.2]:
        sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')
        n_clusters = adata.obs[f'leiden_r{res}'].nunique()
        logger.info("Leiden (r=%.1f): %d clusters", res, n_clusters)

    # Use resolution 0.8 as default
    adata.obs['cluster'] = adata.obs['leiden_r0.8']

    # UMAP plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    sc.pl.umap(adata, color='sample', ax=axes[0, 0], show=False, legend_loc='right margin',
               title='Sample')
    sc.pl.umap(adata, color='group', ax=axes[0, 1], show=False, legend_loc='right margin',
               title='Group (Sham vs MCAO)')
    sc.pl.umap(adata, color='cluster', ax=axes[0, 2], show=False, legend_loc='right margin',
               title='Leiden Clusters (r=0.8)')
    sc.pl.umap(adata, color='n_genes_by_counts', ax=axes[1, 0], show=False,
               title='n_genes')
    sc.pl.umap(adata, color='total_counts', ax=axes[1, 1], show=False,
               title='Total UMIs')
    sc.pl.umap(adata, color='pct_counts_mt', ax=axes[1, 2], show=False,
               title='MT%')

    fig.suptitle('GSE174574 scRNA-seq: UMAP Overview', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'umap_overview.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    return adata


def step7_cell_annotation(adata):
    """Step 7: Cell type annotation using CellTypist + manual markers."""
    logger.info("=" * 60)
    logger.info("STEP 7: Cell Type Annotation")
    logger.info("=" * 60)

    try:
        import celltypist
        from celltypist import models

        # Download mouse brain model if needed
        logger.info("Setting up CellTypist...")
        model_path = models.download_models(force_update=False, model='Mouse_Adult_Brain.pkl')
        logger.info("CellTypist model: %s", model_path)

        # Predict cell types
        predictions = celltypist.annotate(
            adata,
            model='Mouse_Adult_Brain.pkl',
            majority_voting=True,
            over_clustering='leiden_r0.8'
        )

        adata.obs['celltypist_cell_type'] = predictions.predicted_labels['majority_voting']
        adata.obs['celltypist_conf_score'] = predictions.predicted_labels['conf_score'].values

        logger.info("CellTypist annotation completed")
        logger.info("Cell types found: %s", adata.obs['celltypist_cell_type'].unique().tolist())

    except Exception as e:
        logger.warning("CellTypist annotation failed, using manual annotation: %s", e)
        logger.warning("Traceback: %s", traceback.format_exc())
        adata.obs['celltypist_cell_type'] = 'Unknown'
        adata.obs['celltypist_conf_score'] = 0.0

    # Manual marker gene validation
    logger.info("Computing marker gene scores for manual validation...")

    # Score each cell type
    for ct_name, ct_markers in MARKER_GENES.items():
        # Find markers present in the data
        present_markers = [g for g in ct_markers if g in adata.var_names]
        if len(present_markers) == 0:
            logger.warning("  No markers found for %s in data", ct_name)
            continue

        try:
            sc.tl.score_genes(adata, gene_list=present_markers, score_name=f'{ct_name}_score')
        except Exception as e:
            logger.warning("  Failed to score %s: %s", ct_name, e)

    # Determine cell type based on marker scores (for cells with low CellTypist confidence)
    score_cols = [f'{ct}_score' for ct in MARKER_GENES if f'{ct}_score' in adata.obs.columns]
    if score_cols:
        best_type = adata.obs[score_cols].idxmax(axis=1).str.replace('_score', '')
        # Use CellTypist where confident, fall back to manual where not
        low_conf = adata.obs['celltypist_conf_score'] < 0.5
        adata.obs['cell_type'] = adata.obs['celltypist_cell_type'].copy()
        adata.obs.loc[low_conf, 'cell_type'] = best_type.loc[low_conf]
    else:
        adata.obs['cell_type'] = adata.obs['celltypist_cell_type']

    # Log cell type distribution
    ct_counts = adata.obs['cell_type'].value_counts()
    logger.info("Cell type distribution:")
    for ct, count in ct_counts.items():
        logger.info("  %s: %d cells (%.1f%%)", ct, count, 100 * count / adata.n_obs)

    # UMAP by cell type
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color='cell_type', ax=axes[0], show=False, legend_loc='right margin',
               title='Cell Types')
    sc.pl.umap(adata, color='celltypist_conf_score', ax=axes[1], show=False,
               title='CellTypist Confidence', cmap='viridis')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'umap_cell_types.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Marker gene dotplot
    all_markers = []
    for markers in MARKER_GENES.values():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))
    present_markers = [g for g in all_markers if g in adata.var_names]

    if present_markers:
        fig, ax = plt.subplots(figsize=(14, 6))
        sc.pl.dotplot(adata, present_markers, groupby='cell_type', ax=ax, show=False,
                      title='Marker Gene Expression by Cell Type')
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, 'marker_dotplot.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    return adata


def step8_ferroaging_score(adata, ferroaging_genes):
    """Step 8: Calculate ferroaging score per cell."""
    logger.info("=" * 60)
    logger.info("STEP 8: Ferroaging Score Calculation")
    logger.info("=" * 60)

    # Find ferroaging genes present in the dataset
    present_genes = [g for g in ferroaging_genes if g in adata.var_names]
    missing_genes = [g for g in ferroaging_genes if g not in adata.var_names]
    logger.info("Ferroaging genes present: %d / %d", len(present_genes), len(ferroaging_genes))
    if missing_genes:
        logger.warning("Missing ferroaging genes: %s", missing_genes)

    if len(present_genes) == 0:
        logger.error("No ferroaging genes found in the dataset!")
        return adata

    # Use raw counts for scoring
    sc.tl.score_genes(adata, gene_list=present_genes, score_name='ferroaging_score',
                      use_raw=True)
    logger.info("Ferroaging score computed")

    # Compare MCAO vs Sham
    from scipy.stats import mannwhitneyu
    sham_scores = adata[adata.obs['group'] == 'Sham'].obs['ferroaging_score']
    mcao_scores = adata[adata.obs['group'] == 'MCAO'].obs['ferroaging_score']
    stat, pval = mannwhitneyu(mcao_scores, sham_scores, alternative='two-sided')

    logger.info("Ferroaging score - Sham mean: %.4f, MCAO mean: %.4f", sham_scores.mean(), mcao_scores.mean())
    logger.info("Mann-Whitney U test: stat=%.2f, p=%.4e", stat, pval)

    # Per cell-type ferroaging scores
    ct_scores = adata.obs.groupby('cell_type').agg(
        mean_score=('ferroaging_score', 'mean'),
        n_cells=('ferroaging_score', 'count'),
        Sham_mean=('ferroaging_score', lambda x: x[adata.obs.loc[x.index, 'group'] == 'Sham'].mean()),
        MCAO_mean=('ferroaging_score', lambda x: x[adata.obs.loc[x.index, 'group'] == 'MCAO'].mean()),
    ).reset_index()
    ct_scores['delta'] = ct_scores['MCAO_mean'] - ct_scores['Sham_mean']
    ct_scores.to_csv(os.path.join(RESULTS_DIR, 'sc_ferroaging_scores.csv'), index=False)
    logger.info("Cell-type ferroaging scores saved to sc_ferroaging_scores.csv")

    # UMAP colored by ferroaging score
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sc.pl.umap(adata, color='ferroaging_score', ax=axes[0], show=False,
               title='Ferroaging Score', cmap='RdYlBu_r')
    sc.pl.umap(adata, color='ferroaging_score', ax=axes[1], show=False,
               title='Ferroaging Score (split by group)', groups=['Sham', 'MCAO'],
               na_in_legend=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'umap_ferroaging_score.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Violin plot by cell type
    fig, ax = plt.subplots(figsize=(14, 6))
    sc.pl.violin(adata, 'ferroaging_score', groupby='cell_type', rotation=45,
                 ax=ax, show=False)
    ax.set_title('Ferroaging Score by Cell Type', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ferroaging_score_by_celltype.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Boxplot: MCAO vs Sham per cell type
    fig, axes = plt.subplots(1, 1, figsize=(14, 6))
    plot_data = adata.obs[['cell_type', 'group', 'ferroaging_score']].copy()
    sns.boxplot(data=plot_data, x='cell_type', y='ferroaging_score', hue='group',
                palette={'Sham': '#4ECDC4', 'MCAO': '#FF6B6B'}, ax=axes)
    axes.set_title('Ferroaging Score: MCAO vs Sham by Cell Type', fontsize=14)
    axes.tick_params(axis='x', rotation=45)
    axes.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ferroaging_score_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    return adata


def step9_pseudobulk_de(adata):
    """Step 9: Pseudobulk differential expression analysis (MCAO vs Sham per cell type)."""
    logger.info("=" * 60)
    logger.info("STEP 9: Pseudobulk DE Analysis")
    logger.info("=" * 60)

    from scipy.stats import mannwhitneyu

    # Ensure we have raw counts
    if adata.raw is None:
        logger.warning("No raw counts available, skipping pseudobulk DE")
        return adata

    cell_types = adata.obs['cell_type'].unique()
    all_results = []

    for ct in cell_types:
        ct_mask = adata.obs['cell_type'] == ct
        if ct_mask.sum() < 50:
            logger.info("  %s: too few cells (%d), skipping", ct, ct_mask.sum())
            continue

        ct_adata = adata[ct_mask].copy()
        # Use raw counts from the original adata.raw, not the subset's .raw
        if adata.raw is not None:
            ct_raw = adata.raw[ct_mask]
            ct_adata = ct_raw.to_adata()
        # else: use the log-normalized data (already in ct_adata)

        # Pseudobulk per sample
        samples = ct_adata.obs['sample'].unique()
        pseudobulk_list = []
        sample_groups = []

        for s in samples:
            s_mask = ct_adata.obs['sample'] == s
            if s_mask.sum() < 5:
                continue
            pseudobulk = ct_adata[s_mask].X.sum(axis=0)
            if hasattr(pseudobulk, 'A1'):
                pseudobulk = pseudobulk.A1
            else:
                pseudobulk = np.array(pseudobulk).flatten()
            pseudobulk_list.append(pseudobulk)
            sample_groups.append(ct_adata.obs['group'].iloc[0])

        if len(pseudobulk_list) < 4:
            continue

        pseudobulk_matrix = np.vstack(pseudobulk_list)
        sample_groups = np.array(sample_groups)

        sham_idx = sample_groups == 'Sham'
        mcao_idx = sample_groups == 'MCAO'

        if sham_idx.sum() < 2 or mcao_idx.sum() < 2:
            continue

        # Per-gene Mann-Whitney U test
        genes = ct_adata.var_names
        for i, gene in enumerate(genes):
            sham_vals = pseudobulk_matrix[sham_idx, i]
            mcao_vals = pseudobulk_matrix[mcao_idx, i]
            sham_mean = sham_vals.mean()
            mcao_mean = mcao_vals.mean()

            if sham_mean == 0 and mcao_mean == 0:
                continue

            try:
                stat, pval = mannwhitneyu(mcao_vals, sham_vals, alternative='two-sided')
                log2fc = np.log2((mcao_mean + 1) / (sham_mean + 1))
                all_results.append({
                    'cell_type': ct,
                    'gene': gene,
                    'log2FC': log2fc,
                    'sham_mean': sham_mean,
                    'mcao_mean': mcao_mean,
                    'pvalue': pval
                })
            except Exception:
                continue

        logger.info("  %s: %d cells, %d genes tested", ct, ct_mask.sum(), len(genes))

    if all_results:
        de_df = pd.DataFrame(all_results)
        # Multiple testing correction
        try:
            from scipy.stats import false_discovery_control
            de_df['padj'] = false_discovery_control(de_df['pvalue'])
        except ImportError:
            from statsmodels.stats.multitest import multipletests
            _, de_df['padj'], _, _ = multipletests(de_df['pvalue'], method='fdr_bh')

        # Save all results
        de_df.to_csv(os.path.join(RESULTS_DIR, 'sc_pseudobulk_de.csv'), index=False)
        logger.info("Pseudobulk DE results saved: %d gene-cell_type pairs", len(de_df))

        # Significant results
        sig_de = de_df[(de_df['padj'] < 0.05) & (abs(de_df['log2FC']) > 0.585)]
        logger.info("Significant DE: %d pairs (padj<0.05, |log2FC|>0.585)", len(sig_de))
        sig_de.to_csv(os.path.join(RESULTS_DIR, 'sc_pseudobulk_de_significant.csv'), index=False)

        # Check core target genes
        core_genes = load_core_genes()
        core_de = de_df[de_df['gene'].isin(core_genes)]
        if len(core_de) > 0:
            core_de_sig = core_de[(core_de['padj'] < 0.05) & (abs(core_de['log2FC']) > 0.585)]
            core_de_sig.to_csv(os.path.join(RESULTS_DIR, 'sc_pseudobulk_de_core_genes.csv'), index=False)
            logger.info("Core target genes with significant DE: %d", len(core_de_sig))

    return adata


def step10_visualize_core_genes(adata, core_genes):
    """Step 10: Visualize core target genes on UMAP."""
    logger.info("=" * 60)
    logger.info("STEP 10: Core Target Gene Visualization")
    logger.info("=" * 60)

    # Find core genes present in data
    present_genes = [g for g in core_genes if g in adata.var_names]
    logger.info("Core genes present in scRNA data: %d / %d", len(present_genes), len(core_genes))

    if len(present_genes) == 0:
        logger.warning("No core target genes found in scRNA-seq data")
        return

    # Dotplot of core genes by cell type
    fig, ax = plt.subplots(figsize=(16, 6))
    sc.pl.dotplot(adata, present_genes, groupby='cell_type', ax=ax, show=False,
                  title='Core Target Genes Expression by Cell Type')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'core_genes_dotplot.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # UMAP for top core genes
    top_genes = present_genes[:12]
    n_cols = 4
    n_rows = (len(top_genes) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, gene in enumerate(top_genes):
        sc.pl.umap(adata, color=gene, ax=axes[i], show=False, title=gene,
                   cmap='viridis', vmax='p99')

    for j in range(len(top_genes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Core Target Genes on UMAP', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'core_genes_umap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Violin plots for top core genes by group
    top_plot_genes = present_genes[:8]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, gene in enumerate(top_plot_genes):
        sc.pl.violin(adata, gene, groupby='group', ax=axes[i], show=False, rotation=0)
        axes[i].set_title(gene, fontsize=12)

    for j in range(len(top_plot_genes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Core Target Genes: Sham vs MCAO', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'core_genes_violin.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def step11_export_metadata(adata):
    """Step 11: Export cell metadata for downstream use."""
    logger.info("=" * 60)
    logger.info("STEP 11: Export Cell Metadata")
    logger.info("=" * 60)

    meta_cols = ['sample', 'group', 'cell_type', 'celltypist_cell_type',
                 'celltypist_conf_score', 'ferroaging_score',
                 'n_genes_by_counts', 'total_counts', 'pct_counts_mt',
                 'predicted_doublet', 'doublet_score']

    available_cols = [c for c in meta_cols if c in adata.obs.columns]
    meta_df = adata.obs[available_cols].copy()
    meta_df.to_csv(os.path.join(RESULTS_DIR, 'sc_metadata.csv'))
    logger.info("Cell metadata saved: %d cells, %d columns", len(meta_df), len(available_cols))

    # Summary statistics
    summary = {
        'total_cells': adata.n_obs,
        'total_genes': adata.n_vars,
        'n_cell_types': adata.obs['cell_type'].nunique(),
        'n_sham_cells': (adata.obs['group'] == 'Sham').sum(),
        'n_mcao_cells': (adata.obs['group'] == 'MCAO').sum(),
        'ferroaging_score_sham_mean': adata[adata.obs['group'] == 'Sham'].obs['ferroaging_score'].mean(),
        'ferroaging_score_mcao_mean': adata[adata.obs['group'] == 'MCAO'].obs['ferroaging_score'].mean(),
    }
    pd.Series(summary).to_csv(os.path.join(RESULTS_DIR, 'sc_summary.csv'), header=['value'])
    logger.info("Summary statistics saved to sc_summary.csv")

    return adata


def main():
    """Main pipeline execution."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Single-cell RNA-seq Analysis Pipeline")
    logger.info("Dataset: GSE174574 (Mouse MCAO brain, 24h)")
    logger.info("=" * 60)

    try:
        # Load reference gene sets (human symbols)
        ferroaging_genes_human = load_ferroaging_genes()
        core_genes_human = load_core_genes()

        # Convert to mouse gene symbols for this mouse dataset
        ferroaging_map = human_to_mouse_gene(ferroaging_genes_human)
        ferroaging_genes = list(ferroaging_map.values())
        core_map = human_to_mouse_gene(core_genes_human)
        core_genes = list(core_map.values())
        logger.info("Converted %d ferroaging genes and %d core genes to mouse symbols",
                    len(ferroaging_genes), len(core_genes))

        # Step 1: Load data
        adata = step1_load_data()

        # Step 2: QC
        adata = step2_qc(adata)

        # Step 3: Doublet detection
        adata = step3_doublet_detection(adata)

        # Step 4: Preprocessing
        adata = step4_preprocess(adata)

        # Step 5: PCA + Harmony
        adata = step5_pca_harmony(adata)

        # Step 6: UMAP + Clustering
        adata = step6_umap_clustering(adata)

        # Step 7: Cell type annotation
        adata = step7_cell_annotation(adata)

        # Step 8: Ferroaging score
        adata = step8_ferroaging_score(adata, ferroaging_genes)

        # Step 9: Pseudobulk DE
        adata = step9_pseudobulk_de(adata)

        # Step 10: Core gene visualization
        step10_visualize_core_genes(adata, core_genes)

        # Step 11: Export metadata
        adata = step11_export_metadata(adata)

        # Save final AnnData
        adata.write(os.path.join(RESULTS_DIR, 'adata_annotated.h5ad'), compression='gzip')
        logger.info("Final AnnData saved to adata_annotated.h5ad")

        logger.info("=" * 60)
        logger.info("PHASE 2 SINGLE-CELL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()