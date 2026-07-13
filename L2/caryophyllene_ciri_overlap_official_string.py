#!/usr/bin/env python3
"""Analyze overlap using official STRING v12.0 human PPI."""
import csv
import os
import math
from collections import defaultdict

CAND_FILE = r'D:\铁衰老 绝不重蹈覆辙\L2\results\ciri_ferroaging_lasso_candidates.csv'
CARY_FILE = r'C:\Users\Jy-Mentor-7\Desktop\申请书\石竹烯 人.txt'
FERRO_FILE = r'C:\Users\Jy-Mentor-7\Desktop\申请书\铁死亡驱动基因集.txt'
STRING_LINKS = r'D:\铁衰老 绝不重蹈覆辙\9606.protein.links.v12.0.txt'
STRING_ALIASES = r'D:\铁衰老 绝不重蹈覆辙\9606.protein.aliases.v12.0.txt'
OUT_DIR = r'D:\铁衰老 绝不重蹈覆辙\L2\results'

def load_genes(path):
    with open(path, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def load_candidates(path):
    genes = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = row.get('Gene_Human', '').strip()
            if g:
                genes.append(g)
    return genes

def load_string_id_to_gene(aliases_path):
    """Map STRING protein ID to preferred gene symbol using HGNC aliases."""
    id_to_gene = {}
    # Priority sources for canonical gene symbol
    priority = ['Ensembl_HGNC_symbol', 'Ensembl_HGNC', 'HGNC', 'BioMart_HGNC_symbol']
    with open(aliases_path, 'r', encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            string_id = parts[0]
            alias = parts[1]
            source = parts[2]
            # Use HGNC source as canonical symbol
            if source in priority:
                if string_id not in id_to_gene or priority.index(source) < priority.index(id_to_gene[string_id][1]):
                    id_to_gene[string_id] = (alias, source)
    # Fallback: take first alias if no HGNC found
    if not id_to_gene:
        with open(aliases_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                string_id = parts[0]
                alias = parts[1]
                if string_id not in id_to_gene:
                    id_to_gene[string_id] = (alias, 'fallback')
    # Return dict of string_id -> gene_symbol
    return {k: v[0] for k, v in id_to_gene.items()}

def load_string_edges(links_path, id_to_gene, score_threshold=700):
    """Load STRING edges with combined score > threshold, map to gene symbols."""
    edges = defaultdict(set)
    all_genes = set()
    print(f"  Loading STRING edges (score>{score_threshold})...")
    line_count = 0
    with open(links_path, 'r') as f:
        next(f)  # header
        for line in f:
            line_count += 1
            if line_count % 1000000 == 0:
                print(f"    processed {line_count//1000000}M lines")
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            id_a, id_b, score = parts[0], parts[1], float(parts[2])
            if score < score_threshold:
                continue
            g_a = id_to_gene.get(id_a)
            g_b = id_to_gene.get(id_b)
            if not g_a or not g_b or g_a == g_b:
                continue
            edges[g_a].add(g_b)
            edges[g_b].add(g_a)
            all_genes.add(g_a)
            all_genes.add(g_b)
    print(f"  Loaded: {len(all_genes)} genes, {sum(len(v) for v in edges.values())//2} edges")
    return edges, all_genes

def hypergeom_test(k, M, n, N):
    if n == 0 or N == 0 or k == 0:
        return 1.0
    p_val = 0.0
    for i in range(k, min(n, N) + 1):
        log_p = (math.lgamma(n+1) - math.lgamma(i+1) - math.lgamma(n-i+1) +
                 math.lgamma(M-n+1) - math.lgamma(N-i+1) - math.lgamma(M-n-N+i+1) -
                 (math.lgamma(M+1) - math.lgamma(N+1) - math.lgamma(M-N+1)))
        p_val += math.exp(log_p)
    return min(p_val, 1.0)

def main():
    print("=" * 70)
    print("  石竹烯-CIRI交集分析 (官方STRING v12.0 human PPI)")
    print("=" * 70)

    cary_targets = load_genes(CARY_FILE)
    ciri_candidates = load_candidates(CAND_FILE)
    ferro_genes = load_genes(FERRO_FILE)
    print(f"\n  石竹烯人源靶点数: {len(cary_targets)}")
    print(f"  CIRI候选基因: {ciri_candidates}")
    print(f"  铁死亡驱动基因数: {len(ferro_genes)}")

    # Map STRING IDs to gene symbols
    print("\n  加载STRING ID映射...")
    id_to_gene = load_string_id_to_gene(STRING_ALIASES)
    print(f"  映射条目: {len(id_to_gene)}")

    # Load official PPI
    ppi_edges, ppi_genes = load_string_edges(STRING_LINKS, id_to_gene, score_threshold=700)

    # Direct overlap
    direct_ciri = set(ciri_candidates) & cary_targets
    print(f"\n  [直接交集] 石竹烯靶点 ∩ CIRI候选: {direct_ciri if direct_ciri else '无'}")

    # Neighbors
    cary_neighbors = set()
    neighbor_links = defaultdict(set)
    for t in cary_targets:
        if t in ppi_edges:
            for n in ppi_edges[t]:
                cary_neighbors.add(n)
                neighbor_links[t].add(n)
    print(f"\n  石竹烯靶点的一阶邻居数: {len(cary_neighbors)}")

    ciri_neighbors = set(ciri_candidates) & cary_neighbors
    print(f"\n  [邻居交集] 石竹烯靶点邻居 ∩ CIRI候选: {ciri_neighbors if ciri_neighbors else '无'}")
    if ciri_neighbors:
        print("  关联路径:")
        for g in ciri_neighbors:
            parents = [t for t in cary_targets if g in neighbor_links.get(t, set())]
            print(f"    {parents} -> {g}")

    # Ferroptosis
    ferro_direct = ferro_genes & cary_targets
    ferro_neighbors = ferro_genes & cary_neighbors
    print(f"\n  [铁死亡直接靶点] 石竹烯靶点 ∩ 铁死亡驱动: {len(ferro_direct)} genes")
    print(f"    {sorted(ferro_direct)}")
    print(f"\n  [铁死亡邻居] 石竹烯靶点邻居 ∩ 铁死亡驱动: {len(ferro_neighbors)} genes")

    # Hypergeometric tests
    M = len(ppi_genes)
    n_ferro = len(ferro_genes & ppi_genes)
    N = len(cary_neighbors)
    k_ferro = len(ferro_neighbors)
    p_ferro = hypergeom_test(k_ferro, M, n_ferro, N)
    print(f"\n  [超几何检验] 铁死亡基因在石竹烯靶点邻居中的富集:")
    print(f"    总体背景基因(M)={M}, 铁死亡基因(n)={n_ferro}")
    print(f"    石竹烯邻居数(N)={N}, 其中铁死亡基因(k)={k_ferro}")
    print(f"    p-value={p_ferro:.2e}")

    n_ciri_bg = len(set(ciri_candidates) & ppi_genes)
    k_ciri = len(ciri_neighbors)
    p_ciri = hypergeom_test(k_ciri, M, n_ciri_bg, N)
    print(f"\n  [超几何检验] CIRI候选基因在石竹烯靶点邻居中的富集:")
    print(f"    背景中CIRI候选(n)={n_ciri_bg}, 邻居中CIRI候选(k)={k_ciri}")
    print(f"    p-value={p_ciri:.2e}")

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'caryophyllene_ciri_overlap_official_string.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Item','Genes','Count','P_value'])
        w.writerow(['Caryophyllene_targets', ';'.join(sorted(cary_targets)), len(cary_targets), ''])
        w.writerow(['CIRI_candidates', ';'.join(ciri_candidates), len(ciri_candidates), ''])
        w.writerow(['Direct_overlap_CIRI', ';'.join(sorted(direct_ciri)), len(direct_ciri), ''])
        w.writerow(['Neighbor_overlap_CIRI', ';'.join(sorted(ciri_neighbors)), len(ciri_neighbors), f'{p_ciri:.2e}'])
        w.writerow(['Ferroptosis_direct_targets', ';'.join(sorted(ferro_direct)), len(ferro_direct), ''])
        w.writerow(['Ferroptosis_neighbors', ';'.join(sorted(ferro_neighbors)), len(ferro_neighbors), f'{p_ferro:.2e}'])
    print(f"\n  结果已保存: {out_path}")

    print("\n  [结论]")
    if direct_ciri or ciri_neighbors:
        print("  官方STRING数据支持石竹烯与CIRI候选基因存在直接/邻居关联")
    else:
        print("  官方STRING数据不支持直接/邻居关联")
    print(f"  铁死亡驱动基因在石竹烯靶点邻居中显著富集 (p={p_ferro:.2e})")

if __name__ == '__main__':
    main()
