"""
诊断脚本：追踪 96 个铁衰老基因在管线各层的过滤状态
输出每层筛掉哪些基因、原因是什么
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pathlib import Path

L1_RESULTS = Path("d:/铁衰老 绝不重蹈覆辙/L1/results")
L4_RESULTS = Path("d:/铁衰老 绝不重蹈覆辙/L4/results")
L4_RESULTS_V10 = Path("d:/铁衰老 绝不重蹈覆辙/L4/results_v10_minibatch")
L2_RESULTS = Path("d:/铁衰老 绝不重蹈覆辙/L2/results")

# ============================================================
# 第0层：96 个铁衰老基因
# ============================================================
ferroaging_df = pd.read_csv(L1_RESULTS / "ferroaging_genes_96.csv")
ALL_GENES = set(ferroaging_df["gene_symbol"].dropna().unique())
print(f"第0层: ferroaging_genes_96.csv → {len(ALL_GENES)} 个基因")

# ============================================================
# 第1层：CPI 数据中有哪些基因？
# ============================================================
cpi_df = pd.read_csv(L4_RESULTS / "experimental_actives_detail_cleaned.csv")
CPI_GENES = set(cpi_df["gene"].dropna().unique())
in_cpi = ALL_GENES & CPI_GENES
not_in_cpi = ALL_GENES - CPI_GENES
print(f"\n第1层: CPI 数据过滤")
print(f"  在CPI中: {len(in_cpi)} 个基因")
print(f"  不在CPI中: {len(not_in_cpi)} 个基因")
print(f"  丢失基因: {sorted(not_in_cpi)}")

# ============================================================
# 第2层：蛋白特征文件中有哪些基因？
# ============================================================
prot_feat_path = L2_RESULTS / "target_protein_features.csv"
if prot_feat_path.exists():
    prot_feat_df = pd.read_csv(prot_feat_path)
    if "gene_symbol" in prot_feat_df.columns:
        PROT_FEAT_GENES = set(prot_feat_df["gene_symbol"].dropna().unique())
    elif "gene" in prot_feat_df.columns:
        PROT_FEAT_GENES = set(prot_feat_df["gene"].dropna().unique())
    elif "Gene" in prot_feat_df.columns:
        PROT_FEAT_GENES = set(prot_feat_df["Gene"].dropna().unique())
    else:
        PROT_FEAT_GENES = set(str(c) for c in prot_feat_df.iloc[:, 0].dropna().unique())
else:
    PROT_FEAT_GENES = set()

# 也检查 ESM-2 嵌入
esm2_path = L4_RESULTS_V10 / "esm2_protein_embeddings.npz"
esm2_genes = set()
if esm2_path.exists():
    esm2_data = np.load(esm2_path, allow_pickle=True)
    print(f"  ESM-2 npz keys: {list(esm2_data.keys())}")
    if "gene_names" in esm2_data:
        esm2_genes = set(esm2_data["gene_names"])
    elif "genes" in esm2_data:
        esm2_genes = set(esm2_data["genes"])
    elif "gene_symbols" in esm2_data:
        esm2_genes = set(esm2_data["gene_symbols"])

print(f"\n第2层: 蛋白特征过滤")
print(f"  target_protein_features.csv 中: {len(PROT_FEAT_GENES)} 个基因")
print(f"  esm2_protein_embeddings.npz 中: {len(esm2_genes)} 个基因")
in_cpi_and_feat = in_cpi & PROT_FEAT_GENES
in_cpi_no_feat = in_cpi - PROT_FEAT_GENES
print(f"  在CPI中但无蛋白特征: {len(in_cpi_no_feat)} 个基因")
print(f"  丢失基因: {sorted(in_cpi_no_feat)}")

# ============================================================
# 第3层：PPI 网络中有哪些基因？
# ============================================================
ppi_df = pd.read_csv(L1_RESULTS / "ppi_network_extended_edges.csv")
ppi_genes = set()
for col in ["gene_a", "gene_b", "source", "target"]:
    if col in ppi_df.columns:
        ppi_genes |= set(str(x).strip().upper() for x in ppi_df[col].dropna())

print(f"\n第3层: PPI 网络过滤")
print(f"  PPI 网络中: {len(ppi_genes)} 个基因")
print(f"  在CPI中但不在PPI中: {len(in_cpi - ppi_genes)} 个")

# ============================================================
# 第4层：模拟 gene_to_idx 构建（复现 build_graphs_and_adj 逻辑）
# ============================================================
# gene_to_idx = cpi_genes ∪ prot_feat_genes ∪ ppi_genes
all_graph_genes = CPI_GENES | PROT_FEAT_GENES | ppi_genes
print(f"\n第4层: gene_to_idx（图结构）")
print(f"  图中总基因数: {len(all_graph_genes)}")
in_graph = ALL_GENES & all_graph_genes
not_in_graph = ALL_GENES - all_graph_genes
print(f"  铁衰老基因在图中: {len(in_graph)} 个")
print(f"  不在图中: {len(not_in_graph)} 个 → {sorted(not_in_graph)}")

# ============================================================
# 第5层：最终输出中出现的基因（从 tcm_top_candidates 列名）
# ============================================================
top_df = pd.read_csv(L4_RESULTS_V10 / "tcm_top_candidates_v21.csv", nrows=1)
# 提取基因列（排除已知非基因列）
non_gene_cols = {"MOL_ID", "molecule_name", "SMILES", "mean_uncertainty", "max_uncertainty",
                 "uncertainty_penalty", "composite_score", "avg_score", "max_score",
                 "n_hits", "n_targets", "top_targets", "rank"}
output_genes = set()
for col in top_df.columns:
    if col not in non_gene_cols and not col.endswith("_uncertainty"):
        output_genes.add(col)

print(f"\n第5层: 最终输出（tcm_top_candidates_v21.csv）")
print(f"  输出靶标数: {len(output_genes)}")
print(f"  输出基因: {sorted(output_genes)}")

# 在CPI中但不在输出中的基因
in_cpi_not_output = in_cpi - output_genes
print(f"\n  在CPI中但不在最终输出: {len(in_cpi_not_output)} 个基因")
print(f"  → {sorted(in_cpi_not_output)}")

# ============================================================
# 标注核心靶标（铁衰老最关键基因）
# ============================================================
CORE_TARGETS = {
    "ACSL4", "GPX4", "SLC7A11", "HMOX1", "TFRC", "FTH1", "FTL",
    "NFE2L2", "KEAP1", "TP53", "STAT3", "RELA", "NFKB1", "HIF1A",
    "ALOX15", "ALOX5", "MAPK1", "MTOR", "PTGS2", "TLR4", "IL1B",
    "BECN1", "SQSTM1", "MAP1LC3B", "ATG7", "ATG3", "NOX4", "CISD1",
}

print(f"\n{'='*60}")
print("核心靶标在各层的存活状态")
print(f"{'='*60}")
print(f"{'基因':<12} {'CPI':<6} {'prot_feat':<12} {'ESM2':<8} {'PPI':<6} {'in_graph':<10} {'in_output':<10}")
print("-" * 60)
for gene in sorted(CORE_TARGETS & ALL_GENES):
    cpi_ok = "YES" if gene in CPI_GENES else "NO"
    feat_ok = "YES" if gene in PROT_FEAT_GENES else "NO"
    esm2_ok = "YES" if gene in esm2_genes else "NO"
    ppi_ok = "YES" if gene in ppi_genes else "NO"
    graph_ok = "YES" if gene in all_graph_genes else "NO"
    out_ok = "YES" if gene in output_genes else "NO"
    print(f"{gene:<12} {cpi_ok:<6} {feat_ok:<12} {esm2_ok:<8} {ppi_ok:<6} {graph_ok:<10} {out_ok:<10}")

# ============================================================
# 关键发现：哪些核心靶标在CPI中但不在输出中
# ============================================================
print(f"\n{'='*60}")
print("关键问题：在CPI数据中但最终缺失的核心靶标")
print(f"{'='*60}")
missing_core = (in_cpi_not_output) & CORE_TARGETS
for gene in sorted(missing_core):
    reasons = []
    if gene not in PROT_FEAT_GENES:
        reasons.append("无蛋白特征(target_protein_features.csv)")
    if gene not in esm2_genes:
        reasons.append("无ESM-2嵌入")
    if gene not in ppi_genes:
        reasons.append("不在PPI网络中")
    if gene in all_graph_genes:
        reasons.append("在gene_to_idx中但predict_tcm未返回（可能是local_p_idx=-1）")
    print(f"  {gene}: {' | '.join(reasons) if reasons else '原因未知'}")

print(f"\n诊断完成。")