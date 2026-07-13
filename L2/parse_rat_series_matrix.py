#!/usr/bin/env python3
"""
Parse GSE97537 and GSE61616 GEO series matrix files, build expression matrices
and sample metadata with probe-to-gene mapping via mygene.
"""
import gzip, os, time, csv
import numpy as np
import pandas as pd
import mygene

# ========== Config ==========
FILES = {
    'GSE97537': {
        'path': r'C:/Users/Jy-Mentor-7/Downloads/GSE97537_series_matrix.txt.gz',
        'encoding': 'ascii'
    },
    'GSE61616': {
        'path': r'C:/Users/Jy-Mentor-7/Downloads/GSE61616_series_matrix.txt.gz',
        'encoding': 'utf-8'
    }
}

OUT_DIR = r'D:\铁衰老 绝不重蹈覆辙\L1\results'
os.makedirs(OUT_DIR, exist_ok=True)

# ========== Step 1: Parse series matrices ==========
print("=" * 60)
print("Parsing GEO series matrix files...")
print("=" * 60)

all_expr = {}
all_meta = {}

for gse_id, info in FILES.items():
    print(f"\n--- {gse_id} ---")
    
    with gzip.open(info['path'], 'rt', encoding=info['encoding'], errors='replace') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = None
    metadata = {}
    sample_ids = []
    group_labels = []
    
    for i, line in enumerate(lines):
        if line.startswith('!series_matrix_table_begin'):
            data_start = i
            break
        
        # Parse metadata lines  
        if line.startswith('!Sample_title'):
            # Parse tab-separated quoted fields
            parts = line.split('\t')
            titles = [p.strip().strip('"') for p in parts[1:]]
        
        if line.startswith('!Sample_geo_accession'):
            parts = line.split('\t')
            sample_ids = [p.strip().strip('"') for p in parts[1:]]
        
        if line.startswith('!Sample_characteristics') and 'stress' in line.lower():
            # Group assignment from stress/treatment
            parts = line.strip().split('\t')
            for p in parts[1:]:
                val = p.strip().strip('"').lower()
                if 'sham' in val:
                    group_labels.append('Sham')
                elif 'middle cerebral artery occlusion' in val or 'mcao' in val:
                    group_labels.append('MCAO')
                elif 'model' in val:
                    group_labels.append('MCAO')
                elif 'xst' in val or 'xuesaitong' in val:
                    group_labels.append('XST')
                else:
                    group_labels.append('Unknown')
    
    # For GSE61616, fix groups based on sample titles
    if gse_id == 'GSE61616':
        group_labels = []
        for sid in sample_ids:
            for i_l, l in enumerate(lines[:data_start]):
                if l.startswith('!Sample_title') and sid in l:
                    parts = l.split('\t')
                    for p in parts[1:]:
                        val = p.strip().strip('"').lower()
                        if sid in l and p == parts[1 + sample_ids.index(sid)] if sid in sample_ids else False:
                            pass
                    break
            
        # Re-parse for GSE61616
        for i_l, l in enumerate(lines[:data_start]):
            if l.startswith('!Sample_title'):
                parts = l.strip().split('\t')
                titles_list = [p.strip().strip('"') for p in parts[1:]]
                if len(titles_list) == len(sample_ids):
                    for t in titles_list:
                        t_lower = t.lower()
                        if 'sham' in t_lower:
                            group_labels.append('Sham')
                        elif 'model' in t_lower:
                            group_labels.append('MCAO')
                        elif 'xst' in t_lower:
                            group_labels.append('XST')
                        else:
                            group_labels.append('Unknown')
                break
    
    # Parse expression data
    print(f"  Data starts at line {data_start+1}")
    print(f"  Samples: {len(sample_ids)} -> {sample_ids}")
    print(f"  Groups: {group_labels}")
    
    # Parse expression values
    header = lines[data_start + 1].strip().split('\t')
    header = [h.strip().strip('"') for h in header]
    print(f"  Header: {header[:3]}...")
    
    probes = []
    expr_data = []
    for line in lines[data_start + 2:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        probe_id = parts[0].strip().strip('"')
        vals = [float(v.strip().strip('"')) for v in parts[1:]]
        probes.append(probe_id)
        expr_data.append(vals)
    
    expr_df = pd.DataFrame(expr_data, index=probes, columns=sample_ids)
    print(f"  Expression matrix: {expr_df.shape}")
    
    # Build metadata
    meta_df = pd.DataFrame({
        'sample': sample_ids,
        'group': group_labels,
        'dataset': gse_id,
        'species': 'Rat',
        'platform': 'GPL1355'
    })
    
    all_expr[gse_id] = expr_df
    all_meta[gse_id] = meta_df

# ========== Step 2: Probe-to-Gene Mapping via mygene ==========
print("\n" + "=" * 60)
print("Querying mygene for probe-to-gene mapping...")
print("=" * 60)

mg = mygene.MyGeneInfo()

# Combine unique probes from both datasets
all_probes = sorted(set().union(*[set(df.index) for df in all_expr.values()]))
print(f"Total unique probes: {len(all_probes)}")

# Batch query (1000 per call)
batch_size = 1000
probe_to_gene = {}

for batch_start in range(0, len(all_probes), batch_size):
    batch = all_probes[batch_start:batch_start + batch_size]
    try:
        results = mg.querymany(batch, scopes='reporter', species='rat',
                              fields='symbol', as_dataframe=True)
    except Exception as e:
        print(f"  Error at batch {batch_start}: {e}, retrying...")
        time.sleep(5)
        results = mg.querymany(batch, scopes='reporter', species='rat',
                              fields='symbol', as_dataframe=True)
    
    for probe_id, row in results.iterrows():
        if 'symbol' in row and pd.notna(row['symbol']):
            probe_to_gene[probe_id] = str(row['symbol'])
    
    if (batch_start // batch_size) % 5 == 0:
        print(f"  Processed {batch_start + len(batch)}/{len(all_probes)} probes, "
              f"mapped {len(probe_to_gene)} so far")

print(f"\nTotal mapped probes: {len(probe_to_gene)}/{len(all_probes)} "
      f"({len(probe_to_gene)/len(all_probes)*100:.1f}%)")

# Save mapping
map_df = pd.DataFrame([
    {'Probe': k, 'GeneSymbol': v} for k, v in probe_to_gene.items()
])
map_df.to_csv(os.path.join(OUT_DIR, 'GPL1355_probe_to_gene.csv'), index=False)
print(f"Saved GPL1355_probe_to_gene.csv: {len(map_df)} entries")

# ========== Step 3: Collapse probes to genes (max probe) ==========
print("\n" + "=" * 60)
print("Collapsing probes to gene-level expression (max probe)...")
print("=" * 60)

for gse_id in all_expr:
    print(f"\n--- {gse_id} ---")
    expr = all_expr[gse_id]
    
    # Map probes to genes
    gene_expr = {}
    missing = 0
    for probe in expr.index:
        gene = probe_to_gene.get(probe, None)
        if gene is None:
            missing += 1
            continue
        vals = expr.loc[probe].values
        if gene in gene_expr:
            # Max probe per gene
            gene_expr[gene] = np.maximum(gene_expr[gene], vals)
        else:
            gene_expr[gene] = vals
    
    gene_df = pd.DataFrame(gene_expr, index=expr.columns).T
    gene_df.index.name = 'GeneSymbol'
    print(f"  Genes: {len(gene_df)}, Probes unmapped: {missing}")
    
    # Save
    out_path = os.path.join(OUT_DIR, f'{gse_id}_expression_matrix.csv')
    gene_df.to_csv(out_path)
    print(f"  Saved: {out_path}")
    
    # Save metadata
    meta_path = os.path.join(OUT_DIR, f'{gse_id}_sample_meta.csv')
    all_meta[gse_id].to_csv(meta_path, index=False)
    print(f"  Saved: {meta_path}")

# ========== Step 4: Quick validation ==========
print("\n" + "=" * 60)
print("Quick Validation")
print("=" * 60)

# Load ferroaging genes for coverage check
fer_file = r'C:\Users\Jy-Mentor-7\Desktop\申请书\铁衰老数据集.txt'
with open(fer_file) as f:
    fer_genes = set(l.strip() for l in f if l.strip())

# Rat orthologs (human→rat, uppercase first letter)
human_to_rat_simple = {
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
    'ZEB1': 'Zeb1'
}

for gse_id in all_expr:
    gene_df = pd.read_csv(os.path.join(OUT_DIR, f'{gse_id}_expression_matrix.csv'), index_col=0)
    rat_genes = [human_to_rat_simple.get(g, g) for g in fer_genes]
    common = [g for g in rat_genes if g in gene_df.index]
    print(f"  {gse_id} coverage: {len(common)}/{len(fer_genes)} ({len(common)/len(fer_genes)*100:.1f}%)")

print("\nDone!")
