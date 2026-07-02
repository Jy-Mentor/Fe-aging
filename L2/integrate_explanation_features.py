#!/usr/bin/env python3
# Phase 2 - Multi-omics Integration for Posterior Explanation
import os, sys, logging, traceback, warnings, json
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'integrate_explanation.log')
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'L2', 'results')
L1_RESULTS = os.path.join(PROJECT_ROOT, 'L1', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def merge_ppi_topology(explanation):
    """Merge PPI network topology features."""
    ppi_file = os.path.join(L1_RESULTS, 'ppi_network_nodes.csv')
    if not os.path.exists(ppi_file):
        logger.warning("PPI nodes file not found, skipping")
        return explanation

    ppi = pd.read_csv(ppi_file)
    # Map column names
    col_map = {}
    for c in ppi.columns:
        if c.lower() == 'gene':
            col_map[c] = 'gene_symbol'
    ppi = ppi.rename(columns=col_map)

    if 'gene_symbol' not in ppi.columns:
        logger.warning("PPI file missing gene column, skipping")
        return explanation

    ppi = ppi.set_index('gene_symbol')

    for col in ['Degree', 'Betweenness_Centrality', 'Closeness_Centrality', 'Hub_Rank']:
        if col in ppi.columns:
            new_col = 'ppi_' + col.lower()
            explanation[new_col] = ppi[col].reindex(explanation.index)

    logger.info("PPI topology merged: %d features", sum(1 for c in explanation.columns if c.startswith('ppi_')))
    return explanation


def merge_rra_results(explanation):
    """Merge RRA differential expression results."""
    rra_file = os.path.join(L1_RESULTS, 'RRA_gene_level_integrated.csv')
    if not os.path.exists(rra_file):
        logger.warning("RRA file not found, skipping")
        return explanation

    rra = pd.read_csv(rra_file)
    if 'GeneSymbol' in rra.columns:
        rra = rra.set_index('GeneSymbol')

    for col in ['MedianRank', 'N_Datasets', 'Up_Count', 'Down_Count', 'Direction']:
        if col in rra.columns:
            new_col = 'rra_' + col.lower()
            explanation[new_col] = rra[col].reindex(explanation.index)

    logger.info("RRA results merged: %d features", sum(1 for c in explanation.columns if c.startswith('rra_')))
    return explanation


def merge_wgcna_module(explanation):
    """Merge WGCNA module assignment from core_genes_final.csv."""
    core_file = os.path.join(L1_RESULTS, 'core_genes_final.csv')
    if not os.path.exists(core_file):
        logger.warning("Core genes file not found, skipping")
        return explanation

    core = pd.read_csv(core_file)
    if 'GeneSymbol' not in core.columns:
        return explanation

    core = core.set_index('GeneSymbol')

    for col in ['WGCNA_Module', 'WGCNA_ModuleLabel', 'FerroAging']:
        if col in core.columns:
            new_col = col.lower()
            explanation[new_col] = core[col].reindex(explanation.index)

    logger.info("WGCNA module merged: %d features", sum(1 for c in explanation.columns if c.startswith('wgcna') or c == 'ferroaging'))
    return explanation


def merge_sc_ferroaging(explanation):
    """Merge single-cell ferroaging score summary."""
    sc_file = os.path.join(RESULTS_DIR, 'sc_ferroaging_scores.csv')
    sc_summary_file = os.path.join(RESULTS_DIR, 'sc_summary.csv')

    if not os.path.exists(sc_file):
        logger.warning("Single-cell ferroaging scores not found, skipping")
        return explanation

    sc_scores = pd.read_csv(sc_file)
    logger.info("Single-cell ferroaging scores: %d cell types", len(sc_scores))

    # Add summary stats
    if os.path.exists(sc_summary_file):
        sc_summary = pd.read_csv(sc_summary_file)
        sc_summary = sc_summary.set_index(sc_summary.columns[0])
        for idx, row in sc_summary.iterrows():
            col_name = 'sc_' + idx.strip().replace(' ', '_').lower()
            explanation[col_name] = row['value']
        logger.info("SC summary stats merged")

    return explanation


def merge_gsva_summary(explanation):
    """Merge GSVA ferroaging pathway scores summary."""
    gsva_file = os.path.join(RESULTS_DIR, 'gsva_ferroaging_scores.csv')
    if not os.path.exists(gsva_file):
        logger.warning("GSVA scores not found, skipping")
        return explanation

    gsva = pd.read_csv(gsva_file)
    logger.info("GSVA scores: %d samples across %d datasets", len(gsva), gsva['dataset'].nunique())

    # Per-dataset summary
    gsva_summary_rows = []
    for ds in gsva['dataset'].unique():
        ds_data = gsva[gsva['dataset'] == ds]
        gsva_summary_rows.append({
            'metric': f'gsva_{ds}_mean',
            'value': ds_data['gsva_score'].mean()
        })
        gsva_summary_rows.append({
            'metric': f'gsva_{ds}_std',
            'value': ds_data['gsva_score'].std()
        })
        gsva_summary_rows.append({
            'metric': f'gsva_{ds}_n',
            'value': len(ds_data)
        })

    # Separate stroke vs control stats for GSE16561
    if 'GSE16561' in gsva['dataset'].values:
        gse16561 = gsva[gsva['dataset'] == 'GSE16561']
        # Infer group from sample name
        stroke_samples = gse16561[gse16561['sample'].str.contains('Stroke', case=False)]
        ctrl_samples = gse16561[gse16561['sample'].str.contains('Control', case=False)]
        if len(stroke_samples) > 0:
            gsva_summary_rows.append({
                'metric': 'gsva_GSE16561_Stroke_mean',
                'value': stroke_samples['gsva_score'].mean()
            })
        if len(ctrl_samples) > 0:
            gsva_summary_rows.append({
                'metric': 'gsva_GSE16561_Control_mean',
                'value': ctrl_samples['gsva_score'].mean()
            })

    # For GSE104036, separate Sham vs I/R groups
    if 'GSE104036' in gsva['dataset'].values:
        gse104036 = gsva[gsva['dataset'] == 'GSE104036']
        sham_samples = gse104036[gse104036['sample'].str.startswith('S')]
        ir_samples = gse104036[gse104036['sample'].str.startswith('I')]
        ctrl_samples = gse104036[gse104036['sample'].str.startswith('C')]
        if len(sham_samples) > 0:
            gsva_summary_rows.append({
                'metric': 'gsva_GSE104036_Sham_mean',
                'value': sham_samples['gsva_score'].mean()
            })
        if len(ir_samples) > 0:
            gsva_summary_rows.append({
                'metric': 'gsva_GSE104036_IR_mean',
                'value': ir_samples['gsva_score'].mean()
            })
        if len(ctrl_samples) > 0:
            gsva_summary_rows.append({
                'metric': 'gsva_GSE104036_Control_mean',
                'value': ctrl_samples['gsva_score'].mean()
            })

    gsva_summary = pd.DataFrame(gsva_summary_rows)
    gsva_summary.to_csv(os.path.join(RESULTS_DIR, 'gsva_summary_stats.csv'), index=False)
    logger.info("GSVA summary stats saved: %d metrics", len(gsva_summary))

    # Merge GSVA summary stats into explanation (global features, same value for all genes)
    for _, row in gsva_summary.iterrows():
        explanation[row['metric']] = row['value']

    return explanation


def merge_protein_features(explanation):
    """Merge protein features from UniProt and descriptors."""
    prot_file = os.path.join(RESULTS_DIR, 'target_protein_features.csv')
    desc_file = os.path.join(RESULTS_DIR, 'protein_descriptors.csv')

    if not os.path.exists(prot_file):
        logger.warning("Protein features not found, skipping")
        return explanation

    prot = pd.read_csv(prot_file)
    if 'gene_symbol' in prot.columns:
        prot = prot.set_index('gene_symbol')

    # Add basic protein properties
    for col in ['sequence_length', 'mass', 'n_domains', 'n_ptms', 'n_transmembrane',
                'subcellular_main', 'has_signal_peptide', 'has_transmembrane', 'reviewed']:
        if col in prot.columns:
            new_col = 'protein_' + col
            explanation[new_col] = prot[col].reindex(explanation.index)

    logger.info("Protein features merged: %d features", sum(1 for c in explanation.columns if c.startswith('protein_')))

    # Add descriptor summary (AAC hydrophobicity, charge etc.)
    if os.path.exists(desc_file):
        desc = pd.read_csv(desc_file)
        if desc.columns[0] == 'gene_symbol' or 'gene_symbol' in desc.columns:
            gene_col = 'gene_symbol' if 'gene_symbol' in desc.columns else desc.columns[0]
            desc = desc.set_index(gene_col)

            # AAC summary: mean hydrophobicity (A, V, L, I, F, W, M = hydrophobic)
            hydrophobic_aa = ['AAC_A', 'AAC_V', 'AAC_L', 'AAC_I', 'AAC_F', 'AAC_W', 'AAC_M']
            hydro_cols = [c for c in hydrophobic_aa if c in desc.columns]
            if hydro_cols:
                explanation['protein_aac_hydrophobicity'] = desc[hydro_cols].sum(axis=1).reindex(explanation.index)

            # AAC summary: mean charge (R, K = positive; D, E = negative)
            if all(c in desc.columns for c in ['AAC_R', 'AAC_K', 'AAC_D', 'AAC_E']):
                explanation['protein_aac_positive_charge'] = (desc['AAC_R'] + desc['AAC_K']).reindex(explanation.index)
                explanation['protein_aac_negative_charge'] = (desc['AAC_D'] + desc['AAC_E']).reindex(explanation.index)

            logger.info("Protein descriptor summary merged")

    return explanation


def merge_immune_summary(explanation):
    """Check immune infiltration status."""
    immune_file = os.path.join(RESULTS_DIR, 'immune_infiltration.csv')
    if os.path.exists(immune_file):
        immune = pd.read_csv(immune_file)
        if 'message' in immune.columns:
            logger.warning("Immune infiltration: %s", immune['message'].iloc[0])
        else:
            logger.info("Immune infiltration loaded: %d rows", len(immune))
    return explanation


def generate_integration_report(explanation):
    """Generate a comprehensive integration report."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("Phase 2: Multi-omics Integration Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Core genes: {len(explanation)}")
    report_lines.append(f"Total features: {len(explanation.columns)}")
    report_lines.append("")

    # Feature categories
    report_lines.append("Feature Categories:")
    cat_map = {
        'ppi_': 'PPI Network Topology',
        'rra_': 'RRA Differential Expression',
        'wgcna_': 'WGCNA Module Assignment',
        'ferroaging': 'FerroAging Gene Status',
        'sc_': 'Single-cell RNA-seq',
        'gsva_': 'GSVA Pathway Scores',
        'protein_': 'Protein Features',
    }
    for prefix, label in cat_map.items():
        cols = [c for c in explanation.columns if c.startswith(prefix)]
        if cols:
            report_lines.append(f"  [{label}]: {len(cols)} features - {', '.join(cols[:5])}{'...' if len(cols) > 5 else ''}")

    report_lines.append("")
    report_lines.append("Data Completeness:")
    for col in explanation.columns:
        if col == 'gene_symbol':
            continue
        non_null = explanation[col].notna().sum()
        completeness = non_null / len(explanation) * 100
        report_lines.append(f"  {col}: {non_null}/{len(explanation)} ({completeness:.1f}%)")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("NOTE: All features are for POSTERIOR EXPLANATION only.")
    report_lines.append("They must NOT be used as input to prediction models.")
    report_lines.append("=" * 70)

    report_path = os.path.join(RESULTS_DIR, 'integration_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    logger.info("Integration report saved to: %s", report_path)

    # Print report
    for line in report_lines:
        logger.info(line)


def main():
    logger.info("=" * 60)
    logger.info("Phase 2: Multi-omics Integration for Posterior Explanation")
    logger.info("=" * 60)

    # Load core genes
    core_genes = pd.read_csv(os.path.join(L1_RESULTS, 'core_genes_final.csv'))
    gene_list = core_genes['GeneSymbol'].tolist()
    logger.info("Core target genes: %d", len(gene_list))

    # Build explanation matrix
    explanation = pd.DataFrame({'gene_symbol': gene_list})
    explanation = explanation.set_index('gene_symbol')

    # Merge all feature layers
    logger.info("--- Merging feature layers ---")

    explanation = merge_ppi_topology(explanation)
    explanation = merge_rra_results(explanation)
    explanation = merge_wgcna_module(explanation)
    explanation = merge_sc_ferroaging(explanation)
    explanation = merge_gsva_summary(explanation)
    explanation = merge_protein_features(explanation)
    explanation = merge_immune_summary(explanation)

    # Save explanation matrix
    explanation = explanation.reset_index()
    output_path = os.path.join(RESULTS_DIR, 'explanation_features.csv')
    explanation.to_csv(output_path, index=False)
    logger.info("Explanation features saved: %d genes x %d columns -> %s",
                len(explanation), len(explanation.columns), output_path)

    # Generate integration report
    generate_integration_report(explanation)

    logger.info("=" * 60)
    logger.info("Phase 2 Integration COMPLETED")
    logger.info("=" * 60)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        sys.exit(1)
