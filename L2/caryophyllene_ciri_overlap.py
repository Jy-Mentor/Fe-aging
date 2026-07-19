#!/usr/bin/env python3
"""Analyze overlap between beta-caryophyllene targets and CIRI-ferroaging candidates."""
import csv
import os
import math
from collections import defaultdict

# File paths
CAND_FILE = r'D:\铁衰老 绝不重蹈覆辙\L2\results\ciri_ferroaging_lasso_candidates.csv'
CARY_FILE = r'C:\Users\Jy-Mentor-7\Desktop\申请书\石竹烯 人.txt'
FERRO_FILE = r'C:\Users\Jy-Mentor-7\Desktop\申请书\铁死亡驱动基因集.txt'
PPI_EDGE_FILES = [
    r'D:\铁衰老 绝不重蹈覆辙\L1\results\ppi_network_edges.csv',
    r'D:\铁衰老 绝不重蹈覆辙\L1\results\ppi_network_extended_edges.csv'
]
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

def load_ppi_edges(paths, score_threshold=0.7):
    """Load PPI edges with combined score > threshold."""
    edges = defaultdict(set)
    all_nodes = set()
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] PPI file not found: {p}")
            continue
        with open(p, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try column names
                a = row.get('source', row.get('gene_a', '')).strip()
                b = row.get('target', row.get('gene_b', '')).strip()
                if not a or not b or a == b:
                    continue
                s_raw = row.get('weight', row.get('combined_score', '0'))
                try:
                    s = float(s_raw)
                except ValueError:
                    continue
                # Normalize combined_score if >1 (STRING uses 0-1000)
                if s > 1:
                    s = s / 1000.0
                if s >= score_threshold:
                    edges[a].add(b)
                    edges[b].add(a)
                    all_nodes.add(a)
                    all_nodes.add(b)
    return edges, all_nodes

def hypergeom_test(k, M, n, N):
    """Right-tail hypergeometric p-value.
    k: successes in draw
    M: population size
    n: successes in population
    N: draw size
    """
    if n == 0 or N == 0 or k == 0:
        return 1.0
    p_val = 0.0
    for i in range(k, min(n, N) + 1):
        # C(n,i)*C(M-n,N-i)/C(M,N)
        log_p = (math.lgamma(n+1) - math.lgamma(i+1) - math.lgamma(n-i+1) +
                 math.lgamma(M-n+1) - math.lgamma(N-i+1) - math.lgamma(M-n-N+i+1) -
                 (math.lgamma(M+1) - math.lgamma(N+1) - math.lgamma(M-N+1)))
        p_val += math.exp(log_p)
    return min(p_val, 1.0)

def main():
    print("=" * 70)
    print("  石竹烯靶点与CIRI-铁衰老候选基因交集分析")
    print("=" * 70)

    # Load gene sets
    cary_targets = load_genes(CARY_FILE)
    ciri_candidates = load_candidates(CAND_FILE)
    ferro_genes = load_genes(FERRO_FILE)

    print(f"\n  石竹烯人源靶点数: {len(cary_targets)}")
    print(f"  CIRI候选基因: {ciri_candidates}")
    print(f"  铁死亡驱动基因数: {len(ferro_genes)}")

    # Direct overlap
    direct_ciri = set(ciri_candidates) & cary_targets
    print(f"\n  [直接交集] 石竹烯靶点 ∩ CIRI候选: {direct_ciri if direct_ciri else '无'}")

    # PPI neighbors
    ppi_edges, ppi_nodes = load_ppi_edges(PPI_EDGE_FILES, score_threshold=0.7)
    print(f"\n  PPI网络: {len(ppi_nodes)} nodes, {sum(len(v) for v in ppi_edges.values())//2} edges (score>0.7)")

    # First-order neighbors of caryophyllene targets
    cary_neighbors = set()
    neighbor_links = defaultdict(set)
    for t in cary_targets:
        if t in ppi_edges:
            for n in ppi_edges[t]:
                cary_neighbors.add(n)
                neighbor_links[t].add(n)
    print(f"  石竹烯靶点的一阶邻居数: {len(cary_neighbors)}")

    # CIRI candidates as neighbors
    ciri_neighbors = set(ciri_candidates) & cary_neighbors
    print(f"\n  [邻居交集] 石竹烯靶点邻居 ∩ CIRI候选: {ciri_neighbors if ciri_neighbors else '无'}")
    if ciri_neighbors:
        print("  关联路径:")
        for g in ciri_neighbors:
            parents = [t for t in cary_targets if g in neighbor_links.get(t, set())]
            print(f"    {parents} -> {g}")

    # Ferroptosis drivers as direct targets and neighbors
    ferro_direct = ferro_genes & cary_targets
    ferro_neighbors = ferro_genes & cary_neighbors
    print(f"\n  [铁死亡直接靶点] 石竹烯靶点 ∩ 铁死亡驱动: {len(ferro_direct)} genes")
    print(f"    {sorted(ferro_direct)}")
    print(f"\n  [铁死亡邻居] 石竹烯靶点邻居 ∩ 铁死亡驱动: {len(ferro_neighbors)} genes")
    print(f"    {sorted(ferro_neighbors)}")

    # Hypergeometric test: ferroptosis genes among caryophyllene neighbors
    M = len(ppi_nodes)
    n = len(ferro_genes & ppi_nodes)
    N = len(cary_neighbors)
    k = len(ferro_neighbors)
    p_ferro = hypergeom_test(k, M, n, N)
    print("\n  [超几何检验] 铁死亡基因在石竹烯靶点邻居中的富集:")
    print(f"    总体背景基因(M)={M}, 铁死亡基因(n)={n}")
    print(f"    石竹烯邻居数(N)={N}, 其中铁死亡基因(k)={k}")
    print(f"    p-value={p_ferro:.2e}")

    # Hypergeometric test: CIRI candidates among neighbors
    n_ciri_bg = len(set(ciri_candidates) & ppi_nodes)
    k_ciri = len(ciri_neighbors)
    p_ciri = hypergeom_test(k_ciri, M, n_ciri_bg, N)
    print("\n  [超几何检验] CIRI候选基因在石竹烯靶点邻居中的富集:")
    print(f"    背景中CIRI候选(n)={n_ciri_bg}, 邻居中CIRI候选(k)={k_ciri}")
    print(f"    p-value={p_ciri:.2e}")

    # Save results
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'caryophyllene_ciri_overlap.csv')
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

    # Decision guidance
    print("\n  [论文逻辑建议]")
    if direct_ciri or ciri_neighbors:
        print("  情况A: 存在直接靶点或紧密邻居。可构建核心网络:")
        print("  石竹烯靶点 -> 中间蛋白 -> CIRI候选基因/铁死亡驱动基因")
    else:
        print("  情况B: 直接交集有限。建议采用两层论证:")
        print("  (1) CIRI候选基因(SAT1/KLF6等)是铁衰老高活性状态的转录标志物")
        print("  (2) 石竹烯靶点显著富集铁死亡驱动基因，可能通过上游调控间接影响铁衰老")

if __name__ == '__main__':
    main()
