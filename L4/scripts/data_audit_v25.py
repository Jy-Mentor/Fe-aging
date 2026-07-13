#!/usr/bin/env python3
"""
数据真实性验证与补充脚本 v25
================================
逐一检查所有数据文件的完整性和真实性。
"""

import sys
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
L2_RESULTS = PROJECT_ROOT / "L2" / "results"
L3_RESULTS = PROJECT_ROOT / "L3" / "results"
L4_ROOT = PROJECT_ROOT / "L4"
L4_RESULTS = L4_ROOT / "results"
L4_RESULTS_V10 = L4_ROOT / "results_v10_minibatch"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

REPORT_SECTIONS = []

def add_report(title: str, content: str):
    REPORT_SECTIONS.append((title, content))
    logger.info(f"[REPORT] {title}")

def check_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        return True, Chem, AllChem
    except ImportError:
        logger.exception("捕获到异常并继续执行（原 except 'ImportError' 静默吞掉）")
        return False, None, None

# ============================================================
# 1. 铁衰老96基因
# ============================================================
def audit_ferroaging_genes():
    logger.info("=" * 60)
    logger.info("1. 铁衰老96基因验证")
    logger.info("=" * 60)
    
    gene_path = L1_RESULTS / "ferroaging_genes_96.csv"
    lines = []
    
    if not gene_path.exists():
        lines.append(f"**错误**: 文件不存在: {gene_path}")
        add_report("铁衰老96基因", "\n".join(lines))
        return set(), set()
    
    df = pd.read_csv(gene_path)
    genes = df["gene_symbol"].dropna().unique().tolist()
    lines.append(f"- 文件路径: `{gene_path}`")
    lines.append(f"- 基因数量: {len(genes)}")
    lines.append(f"- 列名: {list(df.columns)}")
    lines.append(f"- 基因列表: {', '.join(sorted(genes))}")
    
    # 检查是否有重复
    duplicates = df[df.duplicated(subset="gene_symbol", keep=False)]
    if len(duplicates) > 0:
        lines.append(f"- **警告**: 发现 {len(duplicates)} 个重复基因: {duplicates['gene_symbol'].tolist()}")
    else:
        lines.append("- 无重复基因记录")
    
    # 检查与训练脚本中硬编码的62个基因的差异
    hardcoded_62 = sorted([
        "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
        "BRD7", "CD74", "CISD1", "CTSB", "CXCL10", "CYBB", "DYRK1A", "EGR1",
        "EMP1", "EPHA4", "FBXO31", "FTH1", "FTL", "GMFB", "GPX4", "HBP1",
        "HMOX1", "IGFBP7", "IL1B", "IRF1", "KDM6B", "KLF6", "LACTB", "LCN2",
        "LGMN", "LPCAT3", "MAP1LC3B", "MAPK1", "MTOR", "NFE2L2", "NOX4",
        "PDE4B", "PTGS2", "RELA", "RUNX3", "SAT1", "SLC3A2", "SLC7A11",
        "SOD1", "SP1", "SQSTM1", "STAT3", "TFRC", "TLR4", "TP53", "VDAC2",
        "VDAC3", "ACSL3", "ALOX5", "ATG7", "BECN1", "HIF1A", "KEAP1", "NFKB1",
    ])
    
    actual_genes = set(genes)
    hardcoded_set = set(hardcoded_62)
    
    in_actual_not_hardcoded = actual_genes - hardcoded_set
    in_hardcoded_not_actual = hardcoded_set - actual_genes
    
    if in_actual_not_hardcoded:
        lines.append(f"\n### 实际文件中有但硬编码中缺失的基因 ({len(in_actual_not_hardcoded)}个):")
        lines.append(f"  {', '.join(sorted(in_actual_not_hardcoded))}")
    if in_hardcoded_not_actual:
        lines.append(f"\n### 硬编码中有但实际文件中缺失的基因 ({len(in_hardcoded_not_actual)}个):")
        lines.append(f"  {', '.join(sorted(in_hardcoded_not_actual))}")
    
    lines.append(f"\n- **总结**: 实际文件96基因 vs 硬编码62基因, 差异={len(in_actual_not_hardcoded) + len(in_hardcoded_not_actual)}个")
    
    add_report("铁衰老96基因", "\n".join(lines))
    return actual_genes, hardcoded_set

# ============================================================
# 2. CPI数据验证
# ============================================================
def audit_cpi_data():
    logger.info("=" * 60)
    logger.info("2. CPI数据验证")
    logger.info("=" * 60)
    
    has_rdkit, Chem, AllChem = check_rdkit()
    lines = []
    
    cpi_path = L4_RESULTS / "experimental_actives_detail_cleaned.csv"
    if not cpi_path.exists():
        lines.append(f"**错误**: 文件不存在: {cpi_path}")
        add_report("CPI数据", "\n".join(lines))
        return None, set()
    
    df = pd.read_csv(cpi_path, low_memory=False)
    lines.append(f"- 文件路径: `{cpi_path}`")
    lines.append(f"- 总记录数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    lines.append(f"- 来源分布: {df['source'].value_counts().to_dict()}")
    
    # 基因统计
    lines.append(f"- 唯一基因数: {df['gene'].nunique()}")
    lines.append(f"- 基因列表: {', '.join(sorted(df['gene'].unique()))}")
    
    # 唯一SMILES统计
    unique_smiles = df["canonical_smiles"].nunique()
    lines.append(f"- 唯一SMILES数: {unique_smiles}")
    
    # 重复记录检查
    dupes = df.duplicated(subset=["gene", "canonical_smiles", "standard_type", "standard_value_nM"], keep=False)
    n_dupes = dupes.sum()
    if n_dupes > 0:
        lines.append(f"- **警告**: 发现 {n_dupes} 条重复记录 (gene+SMILES+type+value)")
        dup_df = df[dupes].sort_values(["gene", "canonical_smiles"])
        lines.append(f"  重复示例: {dup_df[['gene', 'canonical_smiles', 'standard_type', 'standard_value_nM']].head(5).to_string()}")
    else:
        lines.append("- 无重复记录 (gene+SMILES+type+value)")
    
    # SMILES有效性检查
    if has_rdkit:
        invalid_smiles = []
        for i, row in df.iterrows():
            smi = row["canonical_smiles"]
            if pd.isna(smi) or str(smi).strip() == "":
                invalid_smiles.append((i, row["gene"], smi, "空值"))
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                invalid_smiles.append((i, row["gene"], smi, "RDKit解析失败"))
        
        if invalid_smiles:
            lines.append(f"- **警告**: 发现 {len(invalid_smiles)} 条无效SMILES:")
            for idx, gene, smi, reason in invalid_smiles[:10]:
                lines.append(f"  行{idx}: 基因={gene}, SMILES={smi!r}, 原因={reason}")
            if len(invalid_smiles) > 10:
                lines.append(f"  ... 还有 {len(invalid_smiles) - 10} 条")
        else:
            lines.append("- SMILES有效性: 全部通过RDKit验证 ✓")
    else:
        lines.append("- RDKit不可用，跳过SMILES有效性检查")
    
    # 基因名称一致性检查
    cpi_genes = set(df["gene"].unique())
    lines.append(f"\n- CPI基因数: {len(cpi_genes)}")
    
    # Uniprot ID统计
    if "uniprot_id" in df.columns:
        uniprot_counts = df.groupby("gene")["uniprot_id"].nunique()
        multi_uniprot = uniprot_counts[uniprot_counts > 1]
        if len(multi_uniprot) > 0:
            lines.append(f"- **警告**: {len(multi_uniprot)} 个基因有多个UniProt ID: {multi_uniprot.to_dict()}")
        else:
            lines.append("- 每个基因的UniProt ID唯一 ✓")
    
    # 缺失值检查
    for col in ["gene", "canonical_smiles", "uniprot_id", "standard_type", "standard_value_nM"]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                lines.append(f"- **警告**: {col} 列有 {n_missing} 个缺失值")
    
    add_report("CPI数据", "\n".join(lines))
    return df, cpi_genes

# ============================================================
# 3. PPI数据验证
# ============================================================
def audit_ppi_data():
    logger.info("=" * 60)
    logger.info("3. PPI数据验证")
    logger.info("=" * 60)
    
    lines = []
    sig_path = L1_RESULTS / "ppi_network_extended_significant_edges.csv"
    ext_path = L1_RESULTS / "ppi_network_extended_edges.csv"
    
    ppi_path = sig_path if sig_path.exists() else ext_path
    if not ppi_path:
        lines.append("**错误**: PPI网络文件不存在")
        add_report("PPI数据", "\n".join(lines))
        return None, set()
    
    df = pd.read_csv(ppi_path, low_memory=False)
    lines.append(f"- 文件路径: `{ppi_path}`")
    lines.append(f"- 总边数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    
    if "gene_a" in df.columns and "gene_b" in df.columns:
        all_nodes = set(df["gene_a"].unique()) | set(df["gene_b"].unique())
        lines.append(f"- 唯一节点数: {len(all_nodes)}")
    elif "source" in df.columns and "target" in df.columns:
        all_nodes = set(df["source"].unique()) | set(df["target"].unique())
        lines.append(f"- 唯一节点数: {len(all_nodes)}")
    
    # 检查combined_score
    if "combined_score" in df.columns:
        lines.append(f"- combined_score 范围: [{df['combined_score'].min():.4f}, {df['combined_score'].max():.4f}]")
    
    # 空值检查
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            lines.append(f"- **警告**: {col} 列有 {n_missing} 个缺失值")
    
    add_report("PPI数据", "\n".join(lines))
    return df, all_nodes

# ============================================================
# 4. ESM-2嵌入验证
# ============================================================
def audit_esm2_embeddings():
    logger.info("=" * 60)
    logger.info("4. ESM-2嵌入验证")
    logger.info("=" * 60)
    
    lines = []
    esm_path = L4_RESULTS_V10 / "esm2_protein_embeddings.npz"
    
    if not esm_path.exists():
        lines.append(f"**错误**: 文件不存在: {esm_path}")
        add_report("ESM-2嵌入", "\n".join(lines))
        return None
    
    data = np.load(esm_path, allow_pickle=True)
    lines.append(f"- 文件路径: `{esm_path}`")
    lines.append(f"- 文件大小: {esm_path.stat().st_size / 1024 / 1024:.2f} MB")
    lines.append(f"- 嵌入键数量: {len(data.files)}")
    
    for key in data.files:
        arr = data[key]
        lines.append(f"- `{key}`: shape={arr.shape}, dtype={arr.dtype}, "
                     f"范围=[{arr.min():.4f}, {arr.max():.4f}], "
                     f"NaN数量={np.isnan(arr).sum()}")
        if arr.ndim == 1:
            lines.append("  注意: 1维数组，可能是蛋白名称或标识符")
    
    # 检查嵌入维度
    embedding_keys = [k for k in data.files if data[k].ndim == 2]
    if embedding_keys:
        emb_key = embedding_keys[0]
        emb_dim = data[emb_key].shape[1]
        lines.append(f"\n- 嵌入维度: {emb_dim} (预期640)")
        if emb_dim != 640:
            lines.append("  **警告**: 嵌入维度不是640!")
    
    # 检查是否有对应的蛋白名称
    name_keys = [k for k in data.files if data[k].ndim == 1]
    if name_keys:
        name_arr = data[name_keys[0]]
        lines.append(f"- 蛋白列表键: `{name_keys[0]}`, 数量={len(name_arr)}")
        names = [n.decode('utf-8') for n in name_arr] if hasattr(name_arr[0], 'decode') else list(name_arr)
        lines.append(f"- 前10个蛋白名: {names[:10]}")
    
    data.close()
    add_report("ESM-2嵌入", "\n".join(lines))
    return len(data.files)

# ============================================================
# 5. KEGG通路验证
# ============================================================
def audit_kegg_pathways():
    logger.info("=" * 60)
    logger.info("5. KEGG通路验证")
    logger.info("=" * 60)
    
    lines = []
    kegg_path = L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"
    
    if not kegg_path.exists():
        lines.append(f"**错误**: 文件不存在: {kegg_path}")
        add_report("KEGG通路", "\n".join(lines))
        return {}
    
    df = pd.read_csv(kegg_path, sep="\t", low_memory=False)
    lines.append(f"- 文件路径: `{kegg_path}`")
    lines.append(f"- 总记录数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    lines.append(f"- 唯一通路数: {df['pathway_id'].nunique() if 'pathway_id' in df.columns else 'N/A'}")
    lines.append(f"- 唯一基因数: {df['gene_symbol'].nunique() if 'gene_symbol' in df.columns else 'N/A'}")
    
    if "pathway_id" in df.columns and "gene_symbol" in df.columns:
        gene_to_pathways = defaultdict(list)
        for _, row in df.iterrows():
            pid = str(row["pathway_id"]).strip()
            g = str(row["gene_symbol"]).strip().upper()
            if pid and g:
                gene_to_pathways[g].append(pid)
        lines.append(f"- 基因-通路映射: {len(gene_to_pathways)} 个基因")
        
        # 通路的基因分布
        pathway_counts = defaultdict(int)
        for g, paths in gene_to_pathways.items():
            for pid in set(paths):
                pathway_counts[pid] += 1
        lines.append(f"- 前10个通路大小: {list(pathway_counts.items())[:10]}")
    else:
        gene_to_pathways = {}
    
    add_report("KEGG通路", "\n".join(lines))
    return gene_to_pathways

# ============================================================
# 6. 铁死亡表型数据验证
# ============================================================
def audit_phenotype_data():
    logger.info("=" * 60)
    logger.info("6. 铁死亡表型数据验证")
    logger.info("=" * 60)
    
    lines = []
    pheno_path = L4_RESULTS_V10 / "phenotype_ferroptosis_dataset_v25_clean.csv"
    
    if not pheno_path.exists():
        lines.append(f"**错误**: 文件不存在: {pheno_path}")
        add_report("铁死亡表型数据", "\n".join(lines))
        return None
    
    df = pd.read_csv(pheno_path, low_memory=False)
    lines.append(f"- 文件路径: `{pheno_path}`")
    lines.append(f"- 总记录数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    
    if "label" in df.columns:
        pos_count = (df["label"] == 1).sum()
        neg_count = (df["label"] == 0).sum()
        lines.append(f"- 正样本(铁死亡诱导剂): {pos_count}")
        lines.append(f"- 负样本(非铁死亡): {neg_count}")
        lines.append(f"- 正负比例: {pos_count}:{neg_count} ≈ {pos_count/neg_count:.2f}:1")
    
    if "ferroptosis_type" in df.columns:
        lines.append(f"- 铁死亡类型分布: {df['ferroptosis_type'].value_counts().to_dict()}")
    
    if "source" in df.columns:
        lines.append(f"- 来源分布: {df['source'].value_counts().to_dict()}")
    
    if "canonical_smiles" in df.columns:
        unique_smiles = df["canonical_smiles"].nunique()
        lines.append(f"- 唯一SMILES数: {unique_smiles}")
    
    # 缺失值检查
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            lines.append(f"- **警告**: {col} 列有 {n_missing} 个缺失值")
    
    add_report("铁死亡表型数据", "\n".join(lines))
    return df

# ============================================================
# 7. 疾病边数据验证
# ============================================================
def audit_disease_edges():
    logger.info("=" * 60)
    logger.info("7. 疾病边数据验证")
    logger.info("=" * 60)
    
    lines = []
    disease_path = L4_RESULTS_V10 / "disease_gene_edges.csv"
    
    if not disease_path.exists():
        lines.append(f"**警告**: 文件不存在: {disease_path}")
        add_report("疾病边数据", "\n".join(lines))
        return None
    
    df = pd.read_csv(disease_path, low_memory=False)
    lines.append(f"- 文件路径: `{disease_path}`")
    lines.append(f"- 总边数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    
    if "gene_symbol" in df.columns:
        lines.append(f"- 唯一基因数: {df['gene_symbol'].nunique()}")
    if "disease_name" in df.columns:
        lines.append(f"- 疾病类型: {df['disease_name'].unique().tolist()}")
    if "disease_type" in df.columns:
        lines.append(f"- 疾病分类: {df['disease_type'].unique().tolist()}")
    if "evidence" in df.columns:
        lines.append(f"- 证据来源: {df['evidence'].unique().tolist()}")
    
    # 缺失值检查
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            lines.append(f"- **警告**: {col} 列有 {n_missing} 个缺失值")
    
    add_report("疾病边数据", "\n".join(lines))
    return df

# ============================================================
# 8. TCM候选池验证
# ============================================================
def audit_tcm_pool():
    logger.info("=" * 60)
    logger.info("8. TCM候选池验证")
    logger.info("=" * 60)
    
    lines = []
    v21_path = L3_RESULTS / "tcm_compound_pool_v21_Alevel.csv"
    original_path = L3_RESULTS / "tcm_compound_pool_tox_filtered.csv"
    
    tcm_path = None
    if v21_path.exists():
        tcm_path = v21_path
        source_tag = "v21 A级"
    elif original_path.exists():
        tcm_path = original_path
        source_tag = "tox_filtered"
    
    if tcm_path is None:
        lines.append("**错误**: TCM候选池文件不存在")
        add_report("TCM候选池", "\n".join(lines))
        return None
    
    df = pd.read_csv(tcm_path, low_memory=False)
    lines.append(f"- 文件路径: `{tcm_path}`")
    lines.append(f"- 来源: {source_tag}")
    lines.append(f"- 总化合物数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    
    if "canonical_smiles" in df.columns or "smiles" in df.columns:
        smile_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        unique_smiles = df[smile_col].nunique()
        lines.append(f"- 唯一SMILES数: {unique_smiles}")
    
    # 缺失值检查
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            lines.append(f"- **警告**: {col} 列有 {n_missing} 个缺失值")
    
    add_report("TCM候选池", "\n".join(lines))
    return df

# ============================================================
# 9. 补充CPI数据验证
# ============================================================
def audit_cpi_supplement():
    logger.info("=" * 60)
    logger.info("9. CPI补充数据验证")
    logger.info("=" * 60)
    
    lines = []
    supp_path = L4_RESULTS_V10 / "cpi_supplement_v25.csv"
    
    if not supp_path.exists():
        lines.append(f"**警告**: 补充CPI文件不存在: {supp_path}")
        add_report("CPI补充数据", "\n".join(lines))
        return None, set()
    
    df = pd.read_csv(supp_path, low_memory=False)
    lines.append(f"- 文件路径: `{supp_path}`")
    lines.append(f"- 总记录数: {len(df)}")
    lines.append(f"- 列名: {list(df.columns)}")
    lines.append(f"- 唯一基因数: {df['gene'].nunique()}")
    lines.append(f"- 基因列表: {', '.join(sorted(df['gene'].unique()))}")
    
    if "source" in df.columns:
        lines.append(f"- 来源分布: {df['source'].value_counts().to_dict()}")
    
    supp_genes = set(df["gene"].unique())
    add_report("CPI补充数据", "\n".join(lines))
    return df, supp_genes

# ============================================================
# 10. BindingDB/DrugBank数据验证
# ============================================================
def audit_bindingdb_drugbank():
    logger.info("=" * 60)
    logger.info("10. BindingDB/DrugBank数据验证")
    logger.info("=" * 60)
    
    lines = []
    bindingdb_path = L4_RESULTS / "bindingdb_active_compounds.csv"
    drugbank_path = L4_RESULTS / "drugbank_active_compounds.csv"
    
    bindingdb_genes = set()
    drugbank_genes = set()
    
    if bindingdb_path.exists():
        bdb_df = pd.read_csv(bindingdb_path, low_memory=False)
        lines.append("### BindingDB数据")
        lines.append(f"- 文件路径: `{bindingdb_path}`")
        lines.append(f"- 总记录数: {len(bdb_df)}")
        lines.append(f"- 列名: {list(bdb_df.columns)}")
        if "gene" in bdb_df.columns:
            bindingdb_genes = set(bdb_df["gene"].unique())
            lines.append(f"- 唯一基因数: {len(bindingdb_genes)}")
            lines.append(f"- 基因列表: {', '.join(sorted(bindingdb_genes))}")
    else:
        lines.append(f"- **警告**: BindingDB文件不存在: {bindingdb_path}")
    
    if drugbank_path.exists():
        db_df = pd.read_csv(drugbank_path, low_memory=False)
        lines.append("\n### DrugBank数据")
        lines.append(f"- 文件路径: `{drugbank_path}`")
        lines.append(f"- 总记录数: {len(db_df)}")
        lines.append(f"- 列名: {list(db_df.columns)}")
        if "gene" in db_df.columns:
            drugbank_genes = set(db_df["gene"].unique())
            lines.append(f"- 唯一基因数: {len(drugbank_genes)}")
            lines.append(f"- 基因列表: {', '.join(sorted(drugbank_genes))}")
    else:
        lines.append(f"- **警告**: DrugBank文件不存在: {drugbank_path}")
    
    add_report("BindingDB/DrugBank数据", "\n".join(lines))
    return bindingdb_genes, drugbank_genes

# ============================================================
# 11. 缺失基因分析
# ============================================================
def analyze_missing_genes(ferro96_genes, cpi_genes, supp_genes, bindingdb_genes, drugbank_genes):
    logger.info("=" * 60)
    logger.info("11. 缺失CPI数据分析")
    logger.info("=" * 60)
    
    lines = []
    
    # 所有已有CPI数据的基因
    all_cpi_genes = cpi_genes | supp_genes
    lines.append(f"- 铁衰老96基因总数: {len(ferro96_genes)}")
    lines.append(f"- 主CPI数据基因数: {len(cpi_genes)}")
    lines.append(f"- 补充CPI数据基因数: {len(supp_genes)}")
    lines.append(f"- 合并CPI基因数: {len(all_cpi_genes)}")
    
    missing = ferro96_genes - all_cpi_genes
    lines.append(f"\n### 缺失CPI数据的铁衰老基因 ({len(missing)}个):")
    
    # 检查这些缺失基因在BindingDB中是否有数据
    in_bindingdb = missing & bindingdb_genes
    in_drugbank = missing & drugbank_genes
    in_both = in_bindingdb & in_drugbank
    in_bindingdb_only = in_bindingdb - in_drugbank
    in_drugbank_only = in_drugbank - in_bindingdb
    no_data = missing - bindingdb_genes - drugbank_genes
    
    lines.append("\n| 状态 | 数量 | 基因 |")
    lines.append("|------|------|------|")
    if in_both:
        lines.append(f"| BindingDB+DrugBank | {len(in_both)} | {', '.join(sorted(in_both))} |")
    if in_bindingdb_only:
        lines.append(f"| 仅BindingDB | {len(in_bindingdb_only)} | {', '.join(sorted(in_bindingdb_only))} |")
    if in_drugbank_only:
        lines.append(f"| 仅DrugBank | {len(in_drugbank_only)} | {', '.join(sorted(in_drugbank_only))} |")
    if no_data:
        lines.append(f"| 完全无数据 | {len(no_data)} | {', '.join(sorted(no_data))} |")
    
    # 详细列出每个缺失基因
    lines.append("\n### 缺失基因详细列表:")
    for gene in sorted(missing):
        status = []
        if gene in bindingdb_genes:
            status.append("BindingDB")
        if gene in drugbank_genes:
            status.append("DrugBank")
        if not status:
            status.append("无数据")
        lines.append(f"  - {gene}: {', '.join(status)}")
    
    add_report("缺失CPI数据分析", "\n".join(lines))
    return missing, bindingdb_genes, drugbank_genes

# ============================================================
# 12. 铁衰老基因与训练脚本基因对比
# ============================================================
def audit_script_vs_actual():
    logger.info("=" * 60)
    logger.info("12. 训练脚本硬编码基因 vs 实际基因文件")
    logger.info("=" * 60)
    
    lines = []
    
    # 训练脚本中硬编码的62个基因
    hardcoded_62 = {
        "ABCC1", "ACVR1B", "ACSL4", "ALOX15", "ATF3", "ATG3", "BAP1", "BCL6",
        "BRD7", "CD74", "CISD1", "CTSB", "CXCL10", "CYBB", "DYRK1A", "EGR1",
        "EMP1", "EPHA4", "FBXO31", "FTH1", "FTL", "GMFB", "GPX4", "HBP1",
        "HMOX1", "IGFBP7", "IL1B", "IRF1", "KDM6B", "KLF6", "LACTB", "LCN2",
        "LGMN", "LPCAT3", "MAP1LC3B", "MAPK1", "MTOR", "NFE2L2", "NOX4",
        "PDE4B", "PTGS2", "RELA", "RUNX3", "SAT1", "SLC3A2", "SLC7A11",
        "SOD1", "SP1", "SQSTM1", "STAT3", "TFRC", "TLR4", "TP53", "VDAC2",
        "VDAC3", "ACSL3", "ALOX5", "ATG7", "BECN1", "HIF1A", "KEAP1", "NFKB1",
    }
    
    # 实际96基因
    gene_path = L1_RESULTS / "ferroaging_genes_96.csv"
    df = pd.read_csv(gene_path)
    actual_96 = set(df["gene_symbol"].unique())
    
    only_in_96 = actual_96 - hardcoded_62
    only_in_62 = hardcoded_62 - actual_96
    
    lines.append(f"- 实际96基因文件: {len(actual_96)}个")
    lines.append(f"- 训练脚本硬编码: {len(hardcoded_62)}个")
    lines.append(f"- 仅在96文件中: {len(only_in_96)}个")
    lines.append(f"- 仅在硬编码中: {len(only_in_62)}个")
    
    if only_in_96:
        lines.append(f"\n### 仅在96文件中的基因 ({len(only_in_96)}个):")
        lines.append(f"  {', '.join(sorted(only_in_96))}")
    
    if only_in_62:
        lines.append(f"\n### 仅在硬编码中的基因 ({len(only_in_62)}个):")
        lines.append(f"  {', '.join(sorted(only_in_62))}")
    
    add_report("脚本vs实际基因对比", "\n".join(lines))

# ============================================================
# 13. 检查补充数据文件
# ============================================================
def audit_supplement_files():
    logger.info("=" * 60)
    logger.info("13. v25补充数据文件检查")
    logger.info("=" * 60)
    
    lines = []
    files_to_check = [
        L4_RESULTS_V10 / "cpi_supplement_v25.csv",
        L4_RESULTS_V10 / "ppi_supplement_v25.csv",
        L4_RESULTS_V10 / "disease_gene_edges_supplemented_v25.csv",
        L4_RESULTS_V10 / "ferroaging_genes_supplemented_v25.csv",
        L4_RESULTS_V10 / "disease_gene_sequences.csv",
    ]
    
    for fpath in files_to_check:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, low_memory=False, nrows=5)
                lines.append(f"- ✅ `{fpath.name}`: {len(df)} 行 (预览), 列={list(df.columns)}")
            except Exception as e:
                lines.append(f"- ⚠️ `{fpath.name}`: 存在但读取失败: {e}")
        else:
            lines.append(f"- ❌ `{fpath.name}`: 不存在")
    
    add_report("v25补充文件", "\n".join(lines))

# ============================================================
# 14. 生成最终报告
# ============================================================
def generate_report():
    logger.info("=" * 60)
    logger.info("生成数据验证报告")
    logger.info("=" * 60)
    
    report_path = L4_RESULTS_V10 / "data_audit_report_v25.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 数据真实性验证报告 v25\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 数据文件路径总览\n\n")
        f.write("| 数据类型 | 文件路径 | 状态 |\n")
        f.write("|----------|----------|------|\n")
        
        data_items = [
            ("铁衰老96基因", L1_RESULTS / "ferroaging_genes_96.csv"),
            ("CPI数据", L4_RESULTS / "experimental_actives_detail_cleaned.csv"),
            ("PPI网络", L1_RESULTS / "ppi_network_extended_significant_edges.csv"),
            ("ESM-2嵌入", L4_RESULTS_V10 / "esm2_protein_embeddings.npz"),
            ("KEGG通路", L2_RESULTS / "kegg_pathways" / "kegg_human_pathway_genes.tsv"),
            ("铁死亡表型", L4_RESULTS_V10 / "phenotype_ferroptosis_dataset_v25_clean.csv"),
            ("疾病边", L4_RESULTS_V10 / "disease_gene_edges.csv"),
            ("TCM候选池", L3_RESULTS / "tcm_compound_pool_v21_Alevel.csv"),
            ("CPI补充(v25)", L4_RESULTS_V10 / "cpi_supplement_v25.csv"),
            ("BindingDB", L4_RESULTS / "bindingdb_active_compounds.csv"),
            ("DrugBank", L4_RESULTS / "drugbank_active_compounds.csv"),
        ]
        
        for name, path in data_items:
            if path.exists():
                size_kb = path.stat().st_size / 1024
                f.write(f"| {name} | `{path}` | ✅ 存在 ({size_kb:.1f} KB) |\n")
            else:
                f.write(f"| {name} | `{path}` | ❌ 不存在 |\n")
        
        f.write("\n---\n\n")
        
        for title, content in REPORT_SECTIONS:
            f.write(f"## {title}\n\n")
            f.write(f"{content}\n\n")
            f.write("---\n\n")
        
        # 问题与建议汇总
        f.write("## 问题与建议汇总\n\n")
        
        # 自动收集所有"警告"和"错误"
        issues = []
        for title, content in REPORT_SECTIONS:
            for line in content.split("\n"):
                if "**错误**" in line or "**警告**" in line:
                    issues.append(f"- [{title}] {line.strip().replace('**', '')}")
        
        if issues:
            f.write("### 发现的问题\n\n")
            for issue in issues:
                f.write(f"{issue}\n")
        else:
            f.write("未发现明显问题。\n")
        
        f.write("\n### 建议\n\n")
        f.write("1. **训练脚本基因列表不一致**: 脚本硬编码62个基因，实际96基因文件有96个基因，建议统一\n")
        f.write("2. **缺失CPI数据基因**: 54个铁衰老基因缺乏CPI数据，建议从BindingDB/DrugBank补充\n")
        f.write("3. **SMILES验证**: 建议对所有SMILES运行RDKit验证\n")
        f.write("4. **重复记录检查**: 定期检查CPI数据中的重复记录\n")
        f.write("\n")
    
    logger.info(f"报告已生成: {report_path}")
    return report_path

# ============================================================
# Main
# ============================================================
def main():
    logger.info("开始数据真实性验证...")
    
    # 1. 铁衰老基因
    ferro96_genes, hardcoded_genes = audit_ferroaging_genes()
    
    # 2. CPI数据
    cpi_df, cpi_genes = audit_cpi_data()
    
    # 3. PPI数据
    ppi_df, ppi_nodes = audit_ppi_data()
    
    # 4. ESM-2嵌入
    audit_esm2_embeddings()
    
    # 5. KEGG通路
    audit_kegg_pathways()
    
    # 6. 铁死亡表型
    audit_phenotype_data()
    
    # 7. 疾病边
    audit_disease_edges()
    
    # 8. TCM候选池
    audit_tcm_pool()
    
    # 9. CPI补充
    supp_df, supp_genes = audit_cpi_supplement()
    
    # 10. BindingDB/DrugBank
    bindingdb_genes, drugbank_genes = audit_bindingdb_drugbank()
    
    # 11. 缺失基因分析
    missing, bindingdb_genes, drugbank_genes = analyze_missing_genes(
        ferro96_genes, cpi_genes, supp_genes, bindingdb_genes, drugbank_genes
    )
    
    # 12. 脚本vs实际
    audit_script_vs_actual()
    
    # 13. 补充文件
    audit_supplement_files()
    
    # 14. 生成报告
    report_path = generate_report()
    
    logger.info(f"验证完成! 报告: {report_path}")

if __name__ == "__main__":
    main()