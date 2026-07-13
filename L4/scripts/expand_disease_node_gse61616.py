#!/usr/bin/env python3
import logging
logger = logging.getLogger(__name__)

r"""
扩展 GSE61616 疾病节点 (disease_gene_edges.csv)。

策略：
1. 从 GSE61616_DE_gene_level.csv 读取完整差异表达结果（rat Affymetrix 已映射到基因符号）。
2. 将 rat/mouse 风格的基因符号映射到人类同源基因符号（复用 L1 的 curated ortholog map）。
3. 选取 GSE61616 中显著差异表达的基因：adj.P.Val < 0.05 且 |logFC| > 1.0。
   阈值依据：Nie et al. (2022) J Inflamm Res 与 Yang et al. (2025) Front Immunol
   对 GSE61616 MCAO/Sham 的分析均采用 |logFC|>1 & adj.P.Val<0.05。
4. 同时保留 GSEA Ferroaging leading edge 中的核心驱动基因。
5. 最终疾病基因集 = 显著 DEGs ∪ GSEA leading edge，保证生物学依据并扩展节点规模。
6. 输出 d:\铁衰老 绝不重蹈覆辙\L4\results\disease_gene_edges.csv。

输入：
- d:\铁衰老 绝不重蹈覆辙\L1\results\GSE61616_DE_gene_level.csv
- d:\铁衰老 绝不重蹈覆辙\L1\results\ferroaging_genes_96.csv
- d:\铁衰老 绝不重蹈覆辙\L2\results\gsea_ferroaging_vs_ferroptosis.csv
- d:\铁衰老 绝不重蹈覆辙\L4\results\experimental_actives_detail_cleaned.csv (CPI 蛋白列表)

输出：
- d:\铁衰老 绝不重蹈覆辙\L4\results\disease_gene_edges.csv
"""

import sys
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

# 允许从 L1 导入 ortholog 映射
PROJECT_ROOT = Path(r"d:\铁衰老 绝不重蹈覆辙")
L1_DIR = PROJECT_ROOT / "L1"
if str(L1_DIR) not in sys.path:
    sys.path.insert(0, str(L1_DIR))

from map_probes_to_genes import get_species_ortholog_map, map_to_human_ortholog

# ============================================================
# 路径
# ============================================================
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v10_minibatch"

DE_FILE = L1_RESULTS / "GSE61616_DE_gene_level.csv"
FA_FILE = L1_RESULTS / "ferroaging_genes_96.csv"
GSEA_FILE = L2_RESULTS / "gsea_ferroaging_vs_ferroptosis.csv"
CPI_FILE = PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned.csv"
OUTPUT_FILE = L4_RESULTS / "disease_gene_edges.csv"

# 选择阈值
PVAL_THRESHOLD = 0.05
LOGFC_THRESHOLD = 1.0  # v32: 参照 GSE61616 缺血再灌注文献标准（|logFC|>1, adj.P.Val<0.05）


def load_de_gene_level(path: Path) -> pd.DataFrame:
    """读取 GSE61616 基因级 DE 结果并映射到人类同源基因符号。"""
    df = pd.read_csv(path)
    required = {"GeneSymbol", "logFC", "adj.P.Val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DE 文件缺少列: {missing}")

    ortholog_map = get_species_ortholog_map()
    species = df["Species"].iloc[0] if "Species" in df.columns else "Rat"

    rows = []
    for _, row in df.iterrows():
        raw_symbols = str(row["GeneSymbol"]).strip()
        if not raw_symbols or raw_symbols.lower() == "nan":
            continue
        # 处理多基因注释，例如 "A1i3 /// Cpamd8 /// ..."
        symbols = [s.strip() for s in raw_symbols.split("///")]
        for sym in symbols:
            if not sym or sym.startswith("LOC"):
                continue
            human_sym = map_to_human_ortholog(sym, species, ortholog_map)
            if not human_sym:
                continue
            rows.append({
                "OriginalID": row.get("OriginalID", "")
                if isinstance(row.get("OriginalID"), str)
                else row.get("Probe", ""),
                "GeneSymbol": human_sym,
                "logFC": float(row["logFC"]),
                "adj.P.Val": float(row["adj.P.Val"]),
                "P.Value": float(row["P.Value"]),
                "Dataset": row.get("Dataset", "GSE61616"),
                "Species": species,
            })

    mapped = pd.DataFrame(rows)
    # 对同一个 human gene symbol，保留 adj.P.Val 最小的记录
    mapped = mapped.sort_values("adj.P.Val").drop_duplicates(subset=["GeneSymbol"], keep="first")
    return mapped


def load_ferroaging_genes(path: Path) -> set:
    df = pd.read_csv(path)
    return set(df["gene_symbol"].dropna().astype(str).str.strip().str.upper())


def load_gsea_leading_edge(path: Path, dataset: str = "GSE61616", pathway: str = "Ferroaging") -> dict:
    """
    读取 GSEA 结果，返回指定数据集/通路的 leading edge 基因信息。
    返回 dict: gene_symbol -> {"padj": float, "nes": float}
    """
    df = pd.read_csv(path)
    sub = df[(df["dataset"] == dataset) & (df["pathway"] == pathway)]
    if sub.empty:
        raise ValueError(f"GSEA 文件中未找到 {dataset}/{pathway}")

    row = sub.iloc[0]
    leading_edge = [g.strip().upper() for g in str(row["leadingEdge"]).split(";") if g.strip()]
    info = {
        "padj": float(row["padj"]),
        "nes": float(row["NES"]),
        "pval": float(row["pval"]),
        "size": int(row["size"]),
        "leading_edge": leading_edge,
    }
    return info


def load_cpi_proteins(path: Path) -> set:
    if not path.exists():
        print(f"  [警告] CPI 文件不存在: {path}，跳过 CPI 重叠统计")
        return set()
    df = pd.read_csv(path, low_memory=False)
    return set(df["gene"].dropna().astype(str).str.strip().str.upper())


def main():
    print("=" * 60)
    print("扩展 GSE61616 疾病节点")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1/5] 读取 DE 结果: {DE_FILE}")
    de = load_de_gene_level(DE_FILE)
    print(f"       映射到 {len(de)} 个唯一人类基因符号")

    print(f"\n[2/5] 读取 Ferroaging 96 基因集: {FA_FILE}")
    fa_genes = load_ferroaging_genes(FA_FILE)
    print(f"       Ferroaging 基因数: {len(fa_genes)}")

    print(f"\n[3/5] 读取 GSEA 结果: {GSEA_FILE}")
    gsea_info = load_gsea_leading_edge(GSEA_FILE, dataset="GSE61616", pathway="Ferroaging")
    leading_edge = set(gsea_info["leading_edge"])
    print(f"       GSEA padj={gsea_info['padj']:.2e}, NES={gsea_info['nes']:.3f}, leading edge={len(leading_edge)}")

    print(f"\n[4/5] 读取 CPI 蛋白列表: {CPI_FILE}")
    cpi_genes = load_cpi_proteins(CPI_FILE)
    print(f"       CPI 蛋白数: {len(cpi_genes)}")

    # 2. 选择策略
    # Tier 1: 全部 GSE61616 显著差异表达基因（不再限制为 Ferroaging 96 子集）
    sig_deg = de[
        (de["adj.P.Val"] < PVAL_THRESHOLD) &
        (de["logFC"].abs() > LOGFC_THRESHOLD)
    ].copy()
    sig_deg_genes = set(sig_deg["GeneSymbol"])

    # Tier 2: GSEA leading edge 基因（uppercase 后）
    leading_edge_in_fa = leading_edge & fa_genes
    print("\n[5/5] 选择策略统计:")
    print(f"       - GSE61616 全部显著 DEGs (adj.P.Val<{PVAL_THRESHOLD}, |logFC|>{LOGFC_THRESHOLD}): {len(sig_deg_genes)}")
    print(f"       - 其中属于 Ferroaging 96 基因集: {len(sig_deg_genes & fa_genes)}")
    print(f"       - GSEA Ferroaging leading edge 基因: {len(leading_edge)}")
    print(f"       - Leading edge 中属于 Ferroaging 96 的基因: {len(leading_edge_in_fa)}")

    # 最终集合：显著 DEGs ∪ leading edge
    final_genes = sig_deg_genes | leading_edge
    print(f"       - 合并后候选疾病基因数: {len(final_genes)}")

    # 3. 与参考集合的重叠
    overlap_fa = final_genes & fa_genes
    overlap_cpi = final_genes & cpi_genes
    overlap_leading = final_genes & leading_edge
    print("\n[统计]")
    print(f"       - 候选基因中属于 Ferroaging 96 基因集: {len(overlap_fa)}/{len(final_genes)}")
    print(f"       - 候选基因中属于 CPI 蛋白: {len(overlap_cpi)}/{len(final_genes)}")
    print(f"       - 候选基因中属于 GSEA leading edge: {len(overlap_leading)}/{len(final_genes)}")

    # 4. 构建输出表
    records = []
    for gene in sorted(final_genes):
        in_sig = gene in sig_deg_genes
        in_le = gene in leading_edge

        if in_sig and in_le:
            source = "GSE61616_significant_DEG+GSEA_leadingEdge"
        elif in_sig:
            source = "GSE61616_significant_DEG"
        else:
            source = "GSEA_Ferroaging_leadingEdge"

        # padj/nes：leading edge 基因用 GSEA 层面的值；仅显著 DEG 用基因层面 adj.P.Val
        if in_le:
            padj = gsea_info["padj"]
            nes = gsea_info["nes"]
        else:
            row = sig_deg[sig_deg["GeneSymbol"] == gene].iloc[0]
            padj = float(row["adj.P.Val"])
            nes = np.nan

        records.append({
            "disease_name": "Ferroaging",
            "disease_type": "neurodegenerative",
            "gene_symbol": gene,
            "evidence": "GSE61616_MCAO_vs_Sham_DEG",
            "source": source,
            "padj": padj,
            "nes": nes,
        })

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["source", "gene_symbol"])
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[输出] 已保存至: {OUTPUT_FILE}")
    print(f"       行数: {len(out_df)}")

    # 5. 详细分类打印
    print("\n[分类明细]")
    for source, sub in out_df.groupby("source"):
        print(f"       - {source}: {len(sub)} 个")

    print("\n与 CPI 蛋白重叠的基因:", sorted(overlap_cpi))
    print("\n完成!")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print(traceback.format_exc())
        sys.exit(1)
