"""
铁衰老项目数据管线完整性验证脚本
验证从数据加载到训练准备的完整流程
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
import random

# 设置随机种子保证可复现
random.seed(42)

# 文件路径配置
BASE_DIR = r"d:\铁衰老 绝不重蹈覆辙"
FILES_TO_CHECK = {
    "ferroaging_genes_96.csv":              r"L1\results\ferroaging_genes_96.csv",
    "experimental_actives_detail_cleaned":  r"L4\results\experimental_actives_detail_cleaned.csv",
    "ppi_dedup":                            r"L1\results\ppi_network_extended_significant_edges_dedup.csv",
    "ppi_original":                         r"L1\results\ppi_network_extended_significant_edges.csv",
    "esm2_embeddings":                      r"L4\results_v10_minibatch\esm2_protein_embeddings.npz",
    "kegg_pathway_genes":                   r"L2\results\kegg_pathways\kegg_human_pathway_genes.tsv",
    "phenotype_v25_clean":                  r"L4\results_v10_minibatch\phenotype_ferroptosis_dataset_v25_clean.csv",
    "disease_gene_edges":                   r"L4\results_v10_minibatch\disease_gene_edges.csv",
    "tcm_compound_pool":                    r"L3\results\tcm_compound_pool_v21_Alevel.csv",
    "cpi_supplement_v25_cleaned":           r"L4\results_v10_minibatch\cpi_supplement_v25_cleaned.csv",
    "cpi_supplement_v26_fixed":             r"L4\results_v10_minibatch\cpi_supplement_v26_fixed.csv",
    "cpi_supplement_v27":                   r"L4\results_v10_minibatch\cpi_supplement_v27.csv",
    "cpi_supplement_v28":                   r"L4\results_v10_minibatch\cpi_supplement_v28.csv",
}

def check_file_exists(file_path):
    """检查文件是否存在并返回大小信息"""
    full_path = os.path.join(BASE_DIR, file_path)
    if os.path.exists(full_path):
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        return True, f"{size_mb:.2f} MB"
    else:
        return False, "文件不存在"

def validate_smiles(smiles_list):
    """验证SMILES是否有效，返回(无效数, 无效列表)"""
    invalid_count = 0
    invalid_smiles = []
    for i, smi in enumerate(smiles_list):
        if pd.isna(smi) or str(smi).strip() == '':
            invalid_count += 1
            invalid_smiles.append((i, str(smi)))
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                invalid_count += 1
                invalid_smiles.append((i, str(smi)))
        except Exception:
            invalid_count += 1
            invalid_smiles.append((i, str(smi)))
    return invalid_count, invalid_smiles

def main():
    print("=" * 80)
    print("铁衰老项目数据管线完整性验证")
    print("=" * 80)
    print()

    # ========== 1. 文件存在性检查 ==========
    print("=" * 80)
    print("1. 数据文件存在性检查")
    print("=" * 80)

    existing_files = []
    missing_files = []

    for name, rel_path in FILES_TO_CHECK.items():
        exists, size_info = check_file_exists(rel_path)
        if exists:
            existing_files.append((name, rel_path, size_info))
            print(f"✅ {name:<40} {size_info:>10}")
        else:
            missing_files.append((name, rel_path))
            print(f"❌ {name:<40} {'文件不存在':>10}")
            print(f"   预期路径: {os.path.join(BASE_DIR, rel_path)}")

    print()
    print(f"总计: {len(existing_files)}/{len(FILES_TO_CHECK)} 文件存在")
    if missing_files:
        print(f"缺失文件: {len(missing_files)} 个")
        for name, rel_path in missing_files:
            print(f"  - {name}: {rel_path}")
    print()

    if len(missing_files) == len(FILES_TO_CHECK):
        print("所有文件都缺失，无法进行后续验证")
        return

    # ========== 2. 数据一致性验证 ==========
    print("=" * 80)
    print("2. 数据一致性验证")
    print("=" * 80)
    print()

    # 2.1 加载ferroaging基因列表
    ferroaging_path = os.path.join(BASE_DIR, r"L1\results\ferroaging_genes_96.csv")
    if os.path.exists(ferroaging_path):
        ferro_df = pd.read_csv(ferroaging_path)
        gene_col = 'gene_symbol' if 'gene_symbol' in ferro_df.columns else 'gene'
        ferro_genes = set(ferro_df[gene_col].str.upper().unique())
        print(f"✅ 加载铁衰老基因列表 (列名: {gene_col}): {len(ferro_genes)} 个基因")
        print(f"   前10个基因: {sorted(list(ferro_genes))[:10]}")
    else:
        print("❌ 无法加载铁衰老基因列表，跳过相关验证")
        ferro_genes = set()
    print()

    # 2.2 CPI数据中的基因与ferroaging基因比较
    cpi_path = os.path.join(BASE_DIR, r"L4\results\experimental_actives_detail_cleaned.csv")
    if os.path.exists(cpi_path):
        cpi_df = pd.read_csv(cpi_path)
        cpi_genes = set(cpi_df['gene'].str.upper().unique())
        print(f"✅ 加载CPI实验数据: {len(cpi_df)} 条记录")
        print(f"   独特基因数: {len(cpi_genes)}")
        print(f"   独特化合物数: {cpi_df['canonical_smiles'].nunique()}")
        print(f"   列名: {list(cpi_df.columns)}")

        if len(ferro_genes) > 0:
            genes_not_in_ferro = cpi_genes - ferro_genes
            print(f"   不在ferroaging_96中的基因: {len(genes_not_in_ferro)} 个")
            if len(genes_not_in_ferro) > 0:
                print(f"   这些基因: {sorted(list(genes_not_in_ferro))}")
                print("   说明: 这些可能是额外补充的文献实验数据，不在初始WGCNA筛选的96个基因中")
            genes_missing_in_cpi = ferro_genes - cpi_genes
            print(f"   ferroaging_96中没有CPI数据的基因: {len(genes_missing_in_cpi)} 个")
            if len(genes_missing_in_cpi) > 0:
                if len(genes_missing_in_cpi) <= 10:
                    print(f"   列表: {sorted(list(genes_missing_in_cpi))}")
                else:
                    print(f"   前10个: {sorted(list(genes_missing_in_cpi))[:10]}...")
    else:
        print("❌ 无法加载CPI数据")
        cpi_genes = set()
    print()

    # 2.3 验证ESM-2嵌入覆盖所有CPI基因
    esm_path = os.path.join(BASE_DIR, r"L4\results_v10_minibatch\esm2_protein_embeddings.npz")
    if os.path.exists(esm_path):
        esm_data = np.load(esm_path, allow_pickle=True)
        esm_genes = set([g.upper() for g in esm_data.keys()])
        print(f"✅ 加载ESM-2蛋白嵌入: {len(esm_genes)} 个基因")
        print(f"   文件大小: {os.path.getsize(esm_path) / (1024*1024):.2f} MB")

        if len(cpi_genes) > 0:
            missing_emb = cpi_genes - esm_genes
            print(f"   CPI基因缺失ESM嵌入: {len(missing_emb)} 个")
            if len(missing_emb) > 0:
                print(f"   缺失: {sorted(list(missing_emb))}")

            extra_emb = esm_genes - cpi_genes
            print(f"   ESM有嵌入但不在CPI中的基因: {len(extra_emb)} 个")

        # 与ferroaging基因比较
        if len(ferro_genes) > 0:
            ferro_missing_emb = ferro_genes - esm_genes
            print(f"   ferroaging_96中缺失ESM嵌入: {len(ferro_missing_emb)} 个")
            if len(ferro_missing_emb) > 0:
                print(f"   缺失: {sorted(list(ferro_missing_emb))}")
    else:
        print("❌ 无法加载ESM-2嵌入")
    print()

    # 2.4 验证PPI网络节点数
    ppi_dedup_path = os.path.join(BASE_DIR, r"L1\results\ppi_network_extended_significant_edges_dedup.csv")
    if os.path.exists(ppi_dedup_path):
        ppi_df = pd.read_csv(ppi_dedup_path)
        # PPI columns: gene_a, gene_b, combined_score
        all_nodes = set(ppi_df['gene_a'].str.upper()) | set(ppi_df['gene_b'].str.upper())
        print(f"✅ 加载PPI网络 (去重后): {len(ppi_df)} 条边")
        print(f"   独特节点数: {len(all_nodes)}")
        if len(ferro_genes) > 0:
            overlap = all_nodes & ferro_genes
            print(f"   包含ferroaging基因: {len(overlap)}/{len(ferro_genes)}")
            if len(overlap) < len(ferro_genes):
                missing = ferro_genes - all_nodes
                print(f"   不在PPI中的ferroaging基因: {len(missing)} 个")
                if len(missing) <= 10:
                    print(f"   列表: {sorted(list(missing))}")

    ppi_original_path = os.path.join(BASE_DIR, r"L1\results\ppi_network_extended_significant_edges.csv")
    if os.path.exists(ppi_original_path):
        ppi_original_df = pd.read_csv(ppi_original_path)
        print(f"✅ 原始PPI网络: {len(ppi_original_df)} 条边")
    print()

    # 2.5 验证KEGG通路数
    kegg_path = os.path.join(BASE_DIR, r"L2\results\kegg_pathways\kegg_human_pathway_genes.tsv")
    if os.path.exists(kegg_path):
        kegg_df = pd.read_csv(kegg_path, sep='\t')
        print(f"✅ 加载KEGG通路注释: {kegg_df['pathway_id'].nunique()} 个通路")
        print(f"   独特基因数: {kegg_df['gene_symbol'].nunique()}")
        print(f"   总基因-通路关联: {len(kegg_df)}")
        print(f"   前5个通路:")
        pathway_sample = kegg_df[['pathway_id', 'pathway_name']].drop_duplicates().head()
        for _, row in pathway_sample.iterrows():
            print(f"     - {row['pathway_id']}: {row['pathway_name']}")
    else:
        print("❌ 无法加载KEGG通路数据")
    print()

    # ========== 3. 补充数据完整性验证 ==========
    print("=" * 80)
    print("3. 补充数据完整性验证")
    print("=" * 80)
    print()

    # v25_cleaned
    v25_path = os.path.join(BASE_DIR, r"L4\results_v10_minibatch\cpi_supplement_v25_cleaned.csv")
    if os.path.exists(v25_path):
        v25_df = pd.read_csv(v25_path)
        print(f"cpi_supplement_v25_cleaned: {len(v25_df)} 条记录")
        print(f"   列名: {list(v25_df.columns)}")
        if 'smiles' in v25_df.columns:
            invalid, invalid_smis = validate_smiles(v25_df['smiles'])
            print(f"   无效SMILES: {invalid} / {len(v25_df)}")
            if invalid > 0 and invalid <= 10:
                print(f"   无效列表: {invalid_smis}")
            elif invalid > 10:
                print(f"   前10个无效: {invalid_smis[:10]}")
    else:
        print("❌ cpi_supplement_v25_cleaned 不存在")
    print()

    # v26_fixed
    v26_path = os.path.join(BASE_DIR, r"L4\results_v10_minibatch\cpi_supplement_v26_fixed.csv")
    if os.path.exists(v26_path):
        v26_df = pd.read_csv(v26_path)
        print(f"cpi_supplement_v26_fixed: {len(v26_df)} 条记录")
        print(f"   列名: {list(v26_df.columns)}")
        required_cols = ['smiles', 'uniprot', 'gene']
        missing_cols = [col for col in required_cols if col not in v26_df.columns]
        if not missing_cols:
            print("   ✅ 所有必需列(smiles, uniprot, gene)都存在")
            invalid, invalid_smis = validate_smiles(v26_df['smiles'])
            print(f"   无效SMILES: {invalid} / {len(v26_df)}")
            # v26 has DrugBank entries with empty SMILES - count them
            empty_smiles = v26_df['smiles'].isna().sum() + (v26_df['smiles'] == '').sum()
            print(f"   空SMILES (DrugBank参考条目): {empty_smiles}")
            if invalid > 0 and invalid <= 10:
                print(f"   无效列表: {invalid_smis}")
        else:
            print(f"   ❌ 缺失列: {missing_cols}")
    else:
        print("❌ cpi_supplement_v26_fixed 不存在")
    print()

    # v27
    v27_path = os.path.join(BASE_DIR, r"L4\results_v10_minibatch\cpi_supplement_v27.csv")
    if os.path.exists(v27_path):
        v27_df = pd.read_csv(v27_path)
        print(f"cpi_supplement_v27: {len(v27_df)} 条记录")
        print(f"   列名: {list(v27_df.columns)}")
        if 'smiles' in v27_df.columns:
            invalid, invalid_smis = validate_smiles(v27_df['smiles'])
            print(f"   无效SMILES: {invalid} / {len(v27_df)}")
            if invalid > 0 and invalid <= 10:
                print(f"   无效列表: {invalid_smis}")
    else:
        print("❌ cpi_supplement_v27 不存在")
    print()

    # v28
    v28_path = os.path.join(BASE_DIR, r"L4\results_v10_minibatch\cpi_supplement_v28.csv")
    if os.path.exists(v28_path):
        v28_df = pd.read_csv(v28_path)
        print(f"cpi_supplement_v28: {len(v28_df)} 条记录")
        print(f"   列名: {list(v28_df.columns)}")
        if 'smiles' in v28_df.columns:
            invalid, invalid_smis = validate_smiles(v28_df['smiles'])
            print(f"   无效SMILES: {invalid} / {len(v28_df)}")
            if invalid > 0 and invalid <= 10:
                print(f"   无效列表: {invalid_smis}")
    else:
        print("❌ cpi_supplement_v28 不存在")
    print()

    # ========== 4. 数据拆分验证 ==========
    print("=" * 80)
    print("4. 数据拆分验证")
    print("=" * 80)
    print()

    if os.path.exists(cpi_path):
        cpi_df = pd.read_csv(cpi_path)
        print(f"CPI基因: {cpi_df['gene'].nunique()}")
        print(f"CPI化合物: {cpi_df['canonical_smiles'].nunique()}")

        # 模拟拆分
        compounds = list(cpi_df['canonical_smiles'].unique())
        random.shuffle(compounds)
        n_train = int(len(compounds) * 0.85)
        print(f"训练化合物: {n_train}, 验证化合物: {len(compounds) - n_train}")
        print(f"训练比例: {n_train/len(compounds)*100:.1f}%")

        # 检查基因-化合物分布
        print()
        print(f"每个基因平均化合物数: {len(cpi_df) / cpi_df['gene'].nunique():.2f}")
        gene_compound_counts = cpi_df.groupby('gene')['canonical_smiles'].nunique().sort_values(ascending=False)
        print(f"化合物最多的前5个基因:")
        for gene, count in gene_compound_counts.head().items():
            print(f"  - {gene}: {count} 个化合物")
        print(f"化合物最少的后5个基因:")
        for gene, count in gene_compound_counts.tail().items():
            print(f"  - {gene}: {count} 个化合物")
    else:
        print("❌ 无法加载CPI数据进行拆分验证")

    print()
    print("=" * 80)
    print("验证完成")
    print("=" * 80)

if __name__ == "__main__":
    main()