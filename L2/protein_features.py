#!/usr/bin/env python3
"""
Phase 2 - Protein Feature Extraction for Core Targets
======================================================
Extracts protein sequence, structural, and embedding features for the 28 core target genes.
These features CAN be used in Phase 4/5 prediction models.

Steps:
  1. Fetch UniProt features (sequence, domains, PTMs, subcellular location)
  2. Fetch protein structures (AlphaFold/PDB)
  3. Compute protein descriptors (AAC, DC, PseAAC)
  4. Compute ESM-2 embeddings
  5. Identify binding pockets (P2Rank)
  6. Export unified protein features for screening

Input:
  - L1/results/core_genes_final.csv (28 core target genes)
  - UniProt REST API (uniprot.org)
  - AlphaFold/PDB (ebi.ac.uk)

Output:
  - L2/results/target_protein_features.csv (UniProt features)
  - L2/results/protein_descriptors.csv (AAC, DC, PseAAC)
  - L2/results/esm_embeddings.npy (ESM-2 embeddings)
  - L2/results/protein_features_for_screening.csv (unified export)
  - L2/data/PDB/ (downloaded structures)

Usage:
  python L2/protein_features.py
"""

import os
import sys
import logging
import traceback
import warnings
import time
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

# ============================================================
# Logging Setup
# ============================================================
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'protein_features.log')
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

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'L2', 'results')
PDB_DIR = os.path.join(PROJECT_ROOT, 'L2', 'data', 'PDB')
DATA_DIR = os.path.join(PROJECT_ROOT, 'L2', 'data')
CORE_GENES_FILE = os.path.join(PROJECT_ROOT, 'L1', 'results', 'core_genes_final.csv')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Human gene symbol to UniProt ID mapping (manually curated for 28 core targets)
# This mapping is based on UniProt and is essential for accurate API queries
GENE_TO_UNIPROT = {
    'TLR4': 'O00206',
    'CD74': 'P04233',
    'IL1B': 'P01584',
    'PTGS2': 'P35354',
    'CXCL10': 'P02778',
    'IRF1': 'P10914',
    'LCN2': 'P80188',
    'SP1': 'P08047',
    'MAPK1': 'P28482',
    'KLF6': 'Q99612',
    'EGR1': 'P18146',
    'BCL6': 'P41182',
    'CTSB': 'P07858',
    'SAT1': 'P21673',
    'KDM6B': 'O15054',
    'LGMN': 'Q99538',
    'IGFBP7': 'Q16270',
    'PDE4B': 'Q07343',
    'EMP1': 'P54849',
    'SOD1': 'P00441',
    'EPHA4': 'P54764',
    'RUNX3': 'Q13761',
    'FBXO31': 'Q5XUX0',
    'LPCAT3': 'Q6P1A2',
    'DYRK1A': 'Q13627',
    'LACTB': 'P83111',
    'GMFB': 'P60983',
    'HBP1': 'O60381',
}


def load_core_genes():
    """Load 28 core target genes."""
    df = pd.read_csv(CORE_GENES_FILE)
    genes = df['GeneSymbol'].tolist()
    logger.info("Loaded %d core target genes", len(genes))
    return genes


def fetch_uniprot_sequence(uniprot_id):
    """Fetch protein sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            sequence = ''.join(lines[1:])
            return sequence
        else:
            logger.warning("  UniProt API returned %d for %s", resp.status_code, uniprot_id)
            return None
    except Exception as e:
        logger.warning("  Failed to fetch sequence for %s: %s", uniprot_id, e)
        return None


def fetch_uniprot_annotations(uniprot_id):
    """Fetch UniProt annotations (domains, PTMs, subcellular location) via REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            logger.warning("  UniProt annotations API returned %d for %s", resp.status_code, uniprot_id)
            return None
    except Exception as e:
        logger.warning("  Failed to fetch annotations for %s: %s", uniprot_id, e)
        return None


def parse_uniprot_features(uniprot_id, data):
    """Parse UniProt JSON to extract key features.

    注意：UniProt REST API 返回的 feature type 是 Title Case
    （如 Domain, Signal, Transmembrane, Modified residue），
    因此比较时使用 .lower() 进行不区分大小写的匹配。
    """
    features = {
        'uniprot_id': uniprot_id,
        'protein_name': '',
        'gene_name': '',
        'length': 0,
        'mass': 0,
        'n_domains': 0,
        'n_ptms': 0,
        'n_phospho': 0,
        'n_ubiquitination': 0,
        'n_acetylation': 0,
        'subcellular_main': '',
        'has_signal_peptide': False,
        'has_transmembrane': False,
        'n_transmembrane': 0,
        'reviewed': False,
    }

    if data is None:
        return features

    try:
        features['protein_name'] = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
        features['gene_name'] = data.get('genes', [{}])[0].get('geneName', {}).get('value', '')
        features['length'] = data.get('sequence', {}).get('length', 0)
        features['mass'] = data.get('sequence', {}).get('molWeight', 0)
        features['reviewed'] = data.get('entryType', '') == 'UniProtKB reviewed (Swiss-Prot)'

        # Comments
        comments = data.get('comments', [])
        for comment in comments:
            if comment.get('commentType', '').upper() == 'SUBCELLULAR LOCATION':
                locations = comment.get('subcellularLocations', [])
                if locations:
                    features['subcellular_main'] = locations[0].get('location', {}).get('value', '')

        # Features (domains, PTMs, signal peptide, transmembrane)
        feat_list = data.get('features', [])
        for feat in feat_list:
            ftype = feat.get('type', '').lower()
            if ftype in ('domain', 'zinc finger', 'repeat'):
                features['n_domains'] += 1
            elif ftype in ('mod_res', 'crosslnk', 'modified residue', 'cross-link',
                           'glycosylation', 'disulfide bond', 'lipidation',
                           'propeptide', 'initiator methionine'):
                features['n_ptms'] += 1
                desc = feat.get('description', '')
                if 'phospho' in desc.lower():
                    features['n_phospho'] += 1
                elif 'ubiquitin' in desc.lower():
                    features['n_ubiquitination'] += 1
                elif 'acetyl' in desc.lower():
                    features['n_acetylation'] += 1
            elif ftype in ('signal', 'signal peptide'):
                features['has_signal_peptide'] = True
            elif ftype in ('transmem', 'transmembrane'):
                features['has_transmembrane'] = True
                features['n_transmembrane'] += 1

    except Exception as e:
        logger.warning("  Error parsing UniProt features for %s: %s", uniprot_id, e)

    return features


def step1_uniprot_features(genes):
    """Step 1: Fetch UniProt features for all core target genes."""
    logger.info("=" * 60)
    logger.info("STEP 1: UniProt Feature Extraction")
    logger.info("=" * 60)

    all_features = []

    for gene in genes:
        uniprot_id = GENE_TO_UNIPROT.get(gene)
        if uniprot_id is None:
            logger.warning("No UniProt ID mapping for %s, skipping", gene)
            continue

        logger.info("Processing %s (UniProt: %s)...", gene, uniprot_id)

        # Fetch sequence
        sequence = fetch_uniprot_sequence(uniprot_id)
        if sequence:
            logger.info("  Sequence length: %d", len(sequence))
        else:
            logger.warning("  No sequence retrieved for %s", gene)

        # Fetch annotations
        annotations = fetch_uniprot_annotations(uniprot_id)
        features = parse_uniprot_features(uniprot_id, annotations)
        features['gene_symbol'] = gene
        features['sequence'] = sequence if sequence else ''
        features['sequence_length'] = len(sequence) if sequence else 0
        all_features.append(features)

        # Rate limiting
        time.sleep(0.5)

    df = pd.DataFrame(all_features)
    df.to_csv(os.path.join(RESULTS_DIR, 'target_protein_features.csv'), index=False)
    logger.info("UniProt features saved: %d proteins", len(df))
    logger.info("  Reviewed (Swiss-Prot): %d", df['reviewed'].sum())
    logger.info("  Mean length: %.1f", df['sequence_length'].mean())
    logger.info("  Mean mass: %.1f Da", df['mass'].mean())

    return df


def fetch_alphafold_structure(uniprot_id):
    """Download AlphaFold predicted structure in PDB format."""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    pdb_file = os.path.join(PDB_DIR, f"AF-{uniprot_id}-F1-model_v4.pdb")

    if os.path.exists(pdb_file):
        logger.info("  AlphaFold structure already exists: %s", pdb_file)
        return pdb_file

    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with open(pdb_file, 'w') as f:
                f.write(resp.text)
            logger.info("  Downloaded AlphaFold structure: %s", pdb_file)
            return pdb_file
        else:
            logger.warning("  AlphaFold API returned %d for %s", resp.status_code, uniprot_id)
            return None
    except Exception as e:
        logger.warning("  Failed to download AlphaFold for %s: %s", uniprot_id, e)
        return None


def step2_protein_structures(protein_features):
    """Step 2: Download AlphaFold structures for core target proteins."""
    logger.info("=" * 60)
    logger.info("STEP 2: Protein Structure Download (AlphaFold)")
    logger.info("=" * 60)

    results = []

    for _, row in protein_features.iterrows():
        uniprot_id = row['uniprot_id']
        gene = row['gene_symbol']

        if not uniprot_id or pd.isna(uniprot_id):
            continue

        logger.info("Downloading structure for %s (%s)...", gene, uniprot_id)
        pdb_path = fetch_alphafold_structure(uniprot_id)
        results.append({
            'uniprot_id': uniprot_id,
            'gene_symbol': gene,
            'pdb_file': pdb_path if pdb_path else '',
            'has_structure': pdb_path is not None
        })
        time.sleep(0.3)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'protein_structures.csv'), index=False)
    logger.info("Structures: %d / %d downloaded", df['has_structure'].sum(), len(df))
    return df


def compute_aac(sequence):
    """Compute Amino Acid Composition (20 features)."""
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aac = np.zeros(20)
    if len(sequence) == 0:
        return aac
    for aa in sequence:
        idx = aa_order.find(aa)
        if idx >= 0:
            aac[idx] += 1
    return aac / len(sequence)


def compute_dc(sequence):
    """Compute Dipeptide Composition (400 features)."""
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    dc = np.zeros(400)
    if len(sequence) < 2:
        return dc
    for i in range(len(sequence) - 1):
        aa1 = aa_order.find(sequence[i])
        aa2 = aa_order.find(sequence[i + 1])
        if aa1 >= 0 and aa2 >= 0:
            dc[aa1 * 20 + aa2] += 1
    total = dc.sum()
    if total > 0:
        dc = dc / total
    return dc


def compute_pseaac(sequence, lambda_val=30, w=0.05):
    """Compute Pseudo Amino Acid Composition (50 features: 20 AAC + 30 sequence-order)."""
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'

    # Hydrophobicity, hydrophilicity, mass
    hydrophobicity = {
        'A': 0.62, 'C': 0.29, 'D': -0.90, 'E': -0.74, 'F': 1.19,
        'G': 0.48, 'H': -0.40, 'I': 1.38, 'K': -1.50, 'L': 1.06,
        'M': 0.64, 'N': -0.78, 'P': 0.12, 'Q': -0.85, 'R': -2.53,
        'S': -0.18, 'T': -0.05, 'V': 1.08, 'W': 0.81, 'Y': 0.26
    }
    hydrophilicity = {
        'A': -0.5, 'C': -1.0, 'D': 3.0, 'E': 3.0, 'F': -2.5,
        'G': 0.0, 'H': -0.5, 'I': -1.8, 'K': 3.0, 'L': -1.8,
        'M': -1.3, 'N': 0.2, 'P': 0.0, 'Q': 0.2, 'R': 3.0,
        'S': 0.3, 'T': -0.4, 'V': -1.5, 'W': -3.4, 'Y': -2.3
    }
    mass = {
        'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
        'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
        'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
        'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
    }

    if len(sequence) == 0:
        return np.zeros(20 + lambda_val)

    # AAC
    aac = np.zeros(20)
    for aa in sequence:
        idx = aa_order.find(aa)
        if idx >= 0:
            aac[idx] += 1
    aac = aac / len(sequence)

    # Sequence-order correlation factors
    # Convert to property vectors
    h1 = np.array([hydrophobicity.get(aa, 0.0) for aa in sequence])
    h2 = np.array([hydrophilicity.get(aa, 0.0) for aa in sequence])
    h3 = np.array([mass.get(aa, 0.0) for aa in sequence])

    # Normalize
    h1 = (h1 - h1.mean()) / (h1.std() + 1e-8)
    h2 = (h2 - h2.mean()) / (h2.std() + 1e-8)
    h3 = (h3 - h3.mean()) / (h3.std() + 1e-8)

    # Tier correlation
    theta = np.zeros(lambda_val)
    for k in range(1, lambda_val + 1):
        if len(sequence) <= k:
            break
        sum_val = 0
        for i in range(len(sequence) - k):
            sum_val += (h1[i] - h1[i + k])**2 + (h2[i] - h2[i + k])**2 + (h3[i] - h3[i + k])**2
        theta[k - 1] = sum_val / (3 * (len(sequence) - k))

    denominator = aac.sum() + w * theta.sum()
    if denominator == 0:
        return np.concatenate([aac, np.zeros(lambda_val)])

    pseaac = np.concatenate([
        aac / (1 + w * theta.sum()) if (1 + w * theta.sum()) != 0 else aac,
        (w * theta) / (1 + w * theta.sum()) if (1 + w * theta.sum()) != 0 else theta
    ])

    # Ensure exactly 50 dimensions
    if len(pseaac) > 50:
        return pseaac[:50]
    elif len(pseaac) < 50:
        return np.pad(pseaac, (0, 50 - len(pseaac)))

    return pseaac


def step3_protein_descriptors(protein_features):
    """Step 3: Compute protein molecular descriptors."""
    logger.info("=" * 60)
    logger.info("STEP 3: Protein Descriptors (AAC, DC, PseAAC)")
    logger.info("=" * 60)

    all_aac = []
    all_dc = []
    all_pseaac = []
    gene_list = []

    for _, row in protein_features.iterrows():
        sequence = row.get('sequence', '')
        gene = row['gene_symbol']

        if not sequence or pd.isna(sequence) or len(sequence) == 0:
            logger.warning("  No sequence for %s, skipping descriptor computation", gene)
            continue

        logger.info("  Computing descriptors for %s (len=%d)", gene, len(sequence))

        aac = compute_aac(sequence)
        dc = compute_dc(sequence)
        pseaac = compute_pseaac(sequence)

        all_aac.append(aac)
        all_dc.append(dc)
        all_pseaac.append(pseaac)
        gene_list.append(gene)

    # Build DataFrames
    aa_order = 'ACDEFGHIKLMNPQRSTVWY'
    aac_df = pd.DataFrame(np.array(all_aac), index=gene_list, columns=[f'AAC_{aa}' for aa in aa_order])
    aac_df.index.name = 'gene_symbol'
    aac_df.to_csv(os.path.join(RESULTS_DIR, 'protein_descriptors.csv'))
    logger.info("AAC descriptors saved: %d proteins x 20 features", len(aac_df))

    pseaac_df = pd.DataFrame(np.array(all_pseaac), index=gene_list,
                             columns=[f'PseAAC_{i}' for i in range(50)])
    pseaac_df.to_csv(os.path.join(RESULTS_DIR, 'protein_pseaac.csv'))
    logger.info("PseAAC descriptors saved: %d proteins x 50 features", len(pseaac_df))

    # Combine into full descriptor matrix
    dc_df = pd.DataFrame(np.array(all_dc), index=gene_list,
                         columns=[f'DC_{i}' for i in range(400)])
    full_descriptors = pd.concat([aac_df, pseaac_df, dc_df], axis=1)
    full_descriptors.to_csv(os.path.join(RESULTS_DIR, 'protein_descriptors_full.csv'))
    logger.info("Full descriptors saved: %d proteins x %d features",
                len(full_descriptors), full_descriptors.shape[1])

    return full_descriptors


def step4_esm_embeddings(protein_features):
    """Step 4: Compute ESM-2 protein embeddings."""
    logger.info("=" * 60)
    logger.info("STEP 4: ESM-2 Protein Embeddings")
    logger.info("=" * 60)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel

        model_name = 'facebook/esm2_t33_650M_UR50D'
        logger.info("Loading ESM-2 model: %s", model_name)
        logger.info("(This may take a few minutes and requires significant memory)")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: %s", device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        embeddings = {}
        gene_list = []

        for _, row in protein_features.iterrows():
            sequence = row.get('sequence', '')
            gene = row['gene_symbol']

            if not sequence or pd.isna(sequence) or len(sequence) == 0:
                logger.warning("  No sequence for %s, skipping ESM-2", gene)
                continue

            logger.info("  Computing ESM-2 embedding for %s (len=%d)...", gene, len(sequence))

            try:
                # Truncate long sequences
                max_len = 1024
                if len(sequence) > max_len:
                    sequence = sequence[:max_len]

                inputs = tokenizer(sequence, return_tensors='pt', padding=True, truncation=True, max_length=max_len)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling over sequence length
                    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

                embeddings[gene] = emb
                gene_list.append(gene)
                logger.info("    Embedding shape: %s", emb.shape)

            except Exception as e:
                logger.warning("  ESM-2 failed for %s: %s", gene, e)
                logger.warning("  Traceback: %s", traceback.format_exc())
                continue

        if embeddings:
            # Create embedding matrix
            emb_dim = embeddings[gene_list[0]].shape[0]
            emb_matrix = np.zeros((len(gene_list), emb_dim))
            for i, gene in enumerate(gene_list):
                emb_matrix[i] = embeddings[gene]

            np.save(os.path.join(RESULTS_DIR, 'esm_embeddings.npy'), emb_matrix)
            pd.DataFrame({'gene_symbol': gene_list}).to_csv(
                os.path.join(RESULTS_DIR, 'esm_gene_order.csv'), index=False
            )
            logger.info("ESM-2 embeddings saved: %d proteins x %d dimensions", len(gene_list), emb_dim)

            # PCA reduction to 128 dimensions
            from sklearn.decomposition import PCA
            n_components = min(128, emb_matrix.shape[0] - 1, emb_matrix.shape[1])
            pca = PCA(n_components=n_components)
            emb_pca = pca.fit_transform(emb_matrix)
            np.save(os.path.join(RESULTS_DIR, 'esm_embeddings_pca128.npy'), emb_pca)
            logger.info("ESM-2 PCA-reduced (128d) saved: %d proteins", len(gene_list))
            logger.info("PCA explained variance: %.2f", pca.explained_variance_ratio_.sum())

        else:
            logger.warning("No ESM-2 embeddings computed")

    except ImportError as e:
        logger.warning("PyTorch/Transformers not available, skipping ESM-2 embeddings: %s", e)
    except Exception as e:
        logger.error("ESM-2 embedding computation failed: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())


def step5_export_unified_features(protein_features, descriptors_df):
    """Step 5: Export unified protein features for Phase 4/5 screening."""
    logger.info("=" * 60)
    logger.info("STEP 5: Export Unified Protein Features for Screening")
    logger.info("=" * 60)

    # Build unified feature table
    gene_order = protein_features['gene_symbol'].tolist()

    unified = protein_features[['gene_symbol', 'uniprot_id', 'length', 'mass',
                                 'n_domains', 'n_ptms', 'n_phospho', 'n_ubiquitination',
                                 'n_acetylation', 'has_signal_peptide', 'has_transmembrane',
                                 'n_transmembrane', 'reviewed', 'subcellular_main']].copy()

    # Merge with descriptors
    if descriptors_df is not None and len(descriptors_df) > 0:
        unified = unified.merge(descriptors_df, left_on='gene_symbol', right_index=True, how='left')

    # Check for ESM-2 embeddings
    esm_order_file = os.path.join(RESULTS_DIR, 'esm_gene_order.csv')
    esm_emb_file = os.path.join(RESULTS_DIR, 'esm_embeddings_pca128.npy')
    if os.path.exists(esm_order_file) and os.path.exists(esm_emb_file):
        esm_order = pd.read_csv(esm_order_file)['gene_symbol'].tolist()
        esm_emb = np.load(esm_emb_file)
        esm_df = pd.DataFrame(esm_emb, index=esm_order,
                              columns=[f'ESM_{i}' for i in range(esm_emb.shape[1])])
        unified = unified.merge(esm_df, left_on='gene_symbol', right_index=True, how='left')
        logger.info("ESM-2 PCA features merged")

    # Check for structures
    struct_file = os.path.join(RESULTS_DIR, 'protein_structures.csv')
    if os.path.exists(struct_file):
        struct_df = pd.read_csv(struct_file)
        unified = unified.merge(struct_df[['gene_symbol', 'has_structure', 'pdb_file']],
                                on='gene_symbol', how='left')
        logger.info("Structure info merged")

    unified.to_csv(os.path.join(RESULTS_DIR, 'protein_features_for_screening.csv'), index=False)
    logger.info("Unified protein features saved: %d proteins x %d features",
                len(unified), unified.shape[1])

    # Summary
    feature_cols = [c for c in unified.columns if c.startswith(('AAC_', 'PseAAC_', 'DC_', 'ESM_'))]
    logger.info("Protein-intrinsic features: %d columns", len(feature_cols))
    logger.info("These features are allowed for Phase 4/5 prediction models")

    return unified


def main():
    """Main extraction pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Protein Feature Extraction")
    logger.info("=" * 60)

    try:
        # Load core genes
        genes = load_core_genes()

        # Step 1: UniProt features
        protein_features = step1_uniprot_features(genes)

        # Step 2: Protein structures
        step2_protein_structures(protein_features)

        # Step 3: Protein descriptors
        descriptors_df = step3_protein_descriptors(protein_features)

        # Step 4: ESM-2 embeddings
        step4_esm_embeddings(protein_features)

        # Step 5: Export unified features
        step5_export_unified_features(protein_features, descriptors_df)

        logger.info("=" * 60)
        logger.info("PROTEIN FEATURE EXTRACTION COMPLETED")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()