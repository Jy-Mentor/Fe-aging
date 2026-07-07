#!/usr/bin/env python3
"""
CPI数据补充脚本 v28 - 从BindingDB和DrugBank补充铁衰老96基因的CPI数据
目标: 为zero-shot靶标（无CPI数据）补充化合物-蛋白质互作数据
"""
import pandas as pd
import numpy as np
import sys
import os
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / 'results_v10_minibatch' / 'cpi_supplement_v28_report.txt', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# 路径定义
BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"
RESULTS_V10 = BASE / "results_v10_minibatch"

# ============================================================
# 步骤1: 分析当前覆盖情况
# ============================================================
logger.info("=" * 60)
logger.info("步骤1: 分析当前CPI覆盖情况")
logger.info("=" * 60)

# 加载铁衰老96基因
genes_df = pd.read_csv(RESULTS_V10 / "ferroaging_genes_supplemented_v25.csv")
ferro_genes = set(genes_df["gene_symbol"].str.strip().str.upper())
logger.info(f"铁衰老96基因: {len(ferro_genes)} 个")
logger.info(f"基因列表: {sorted(ferro_genes)}")

# 加载主CPI数据
main_cpi = pd.read_csv(RESULTS / "experimental_actives_detail_cleaned.csv", low_memory=False)
main_cpi_genes = set(main_cpi["gene"].str.strip().str.upper())
logger.info(f"主CPI数据: {len(main_cpi)} 条记录, {len(main_cpi_genes)} 个唯一基因")

# 加载v25补充数据
v25 = pd.read_csv(RESULTS_V10 / "cpi_supplement_v25_cleaned.csv", low_memory=False)
v25_genes = set(v25["gene"].str.strip().str.upper())
logger.info(f"v25补充数据: {len(v25)} 条记录, {len(v25_genes)} 个唯一基因")

# 加载v26补充数据
v26 = pd.read_csv(RESULTS_V10 / "cpi_supplement_v26.csv", low_memory=False)
logger.info(f"v26补充数据: {len(v26)} 条记录, 列名: {v26.columns.tolist()}")

# 加载v27补充数据
v27 = pd.read_csv(RESULTS_V10 / "cpi_supplement_v27.csv", low_memory=False)
v27_genes = set(v27["gene"].str.strip().str.upper())
logger.info(f"v27补充数据: {len(v27)} 条记录, {len(v27_genes)} 个唯一基因")

# 合并所有已覆盖基因
all_covered_genes = main_cpi_genes | v25_genes | v27_genes
# 处理v26的列名差异
if "gene" in v26.columns:
    v26_genes = set(v26["gene"].str.strip().str.upper())
    all_covered_genes = all_covered_genes | v26_genes
    logger.info(f"v26数据: {len(v26)} 条记录, {len(v26_genes)} 个唯一基因")

# 统计覆盖情况
covered_genes = ferro_genes & all_covered_genes
missing_genes = ferro_genes - all_covered_genes
logger.info(f"\n覆盖统计:")
logger.info(f"  有CPI数据的基因: {len(covered_genes)} 个")
logger.info(f"  无CPI数据的基因 (zero-shot): {len(missing_genes)} 个")
logger.info(f"  Zero-shot基因列表: {sorted(missing_genes)}")

# SMILES验证函数（提前定义，供所有步骤使用）
def validate_smiles(smi):
    if pd.isna(smi) or not isinstance(smi, str) or len(smi.strip()) == 0:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    return mol.GetNumHeavyAtoms() != 0

# 统计每个基因的CPI记录数
gene_counts = {}
for gene in covered_genes:
    cnt = 0
    cnt += len(main_cpi[main_cpi["gene"].str.strip().str.upper() == gene])
    cnt += len(v25[v25["gene"].str.strip().str.upper() == gene])
    if "gene" in v27.columns:
        cnt += len(v27[v27["gene"].str.strip().str.upper() == gene])
    if "gene" in v26.columns:
        cnt += len(v26[v26["gene"].str.strip().str.upper() == gene])
    gene_counts[gene] = cnt

low_count_genes = [(g, c) for g, c in gene_counts.items() if c < 10]
low_count_genes.sort(key=lambda x: x[1])
logger.info(f"\nCPI记录数 < 10 的基因: {len(low_count_genes)} 个")
for g, c in low_count_genes:
    logger.info(f"  {g}: {c} 条")

# ============================================================
# 步骤2: 检查并修复cpi_supplement_v26.csv
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("步骤2: 检查cpi_supplement_v26.csv列名")
logger.info("=" * 60)

logger.info(f"v26 列名: {v26.columns.tolist()}")
# 检查是否缺少smiles和uniprot列
needs_fix = False
if "smiles" not in v26.columns and "canonical_smiles" in v26.columns:
    logger.info("v26缺少'smiles'列，但有'canonical_smiles'列，可以重命名")
    v26 = v26.rename(columns={"canonical_smiles": "smiles"})
    needs_fix = True
if "uniprot" not in v26.columns and "uniprot_id" in v26.columns:
    logger.info("v26缺少'uniprot'列，但有'uniprot_id'列，可以重命名")
    v26 = v26.rename(columns={"uniprot_id": "uniprot"})
    needs_fix = True

if needs_fix:
    # 同时检查standard_type -> activity_type, standard_value_nM -> activity_value_nm
    if "standard_type" in v26.columns and "activity_type" not in v26.columns:
        v26 = v26.rename(columns={"standard_type": "activity_type"})
    if "standard_value_nM" in v26.columns and "activity_value_nm" not in v26.columns:
        v26 = v26.rename(columns={"standard_value_nM": "activity_value_nm"})
    logger.info(f"修复后v26列名: {v26.columns.tolist()}")
    # 保存修复后的v26
    fixed_path = RESULTS_V10 / "cpi_supplement_v26_fixed.csv"
    v26.to_csv(fixed_path, index=False)
    logger.info(f"修复后的v26已保存到: {fixed_path}")
else:
    logger.info("v26列名正常，无需修复")

# 更新v26基因集
if "gene" in v26.columns:
    v26_genes = set(v26["gene"].str.strip().str.upper())
    all_covered_genes = main_cpi_genes | v25_genes | v27_genes | v26_genes

# ============================================================
# 步骤3: 从BindingDB补充数据
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("步骤3: 从BindingDB补充CPI数据")
logger.info("=" * 60)

bindingdb = pd.read_csv(RESULTS / "bindingdb_active_compounds.csv", low_memory=False)
logger.info(f"BindingDB数据: {len(bindingdb)} 条记录")
logger.info(f"BindingDB列名: {bindingdb.columns.tolist()}")

# 标准化基因名
bindingdb["gene_upper"] = bindingdb["gene"].str.strip().str.upper()

# 过滤铁衰老96基因中缺乏CPI数据的基因
bindingdb_missing = bindingdb[bindingdb["gene_upper"].isin(missing_genes)]
logger.info(f"BindingDB中匹配到zero-shot基因的数据: {len(bindingdb_missing)} 条")

if len(bindingdb_missing) > 0:
    bd_missing_genes = bindingdb_missing["gene_upper"].unique()
    logger.info(f"可补充的基因: {sorted(bd_missing_genes)}")
    
    # 只保留高质量数据: IC50/Ki/Kd < 10μM (10000 nM)
    valid_types = ["IC50", "Ki", "Kd", "EC50", "IC50", "MIC"]
    bindingdb_missing["standard_type_upper"] = bindingdb_missing["standard_type"].str.strip().str.upper()
    
    # 过滤活动类型
    bindingdb_filtered = bindingdb_missing[
        bindingdb_missing["standard_type_upper"].isin([t.upper() for t in valid_types])
    ].copy()
    logger.info(f"  有效活动类型过滤后: {len(bindingdb_filtered)} 条")
    
    # 过滤活性值 < 10000 nM (10μM)
    bindingdb_filtered = bindingdb_filtered[
        bindingdb_filtered["standard_value_nM"].notna() &
        (bindingdb_filtered["standard_value_nM"] < 10000)
    ].copy()
    logger.info(f"  活性值 < 10μM 过滤后: {len(bindingdb_filtered)} 条")
    
    # 验证SMILES
    bindingdb_filtered["smiles_valid"] = bindingdb_filtered["canonical_smiles"].apply(validate_smiles)
    n_invalid_smiles = (~bindingdb_filtered["smiles_valid"]).sum()
    if n_invalid_smiles > 0:
        logger.warning(f"  {n_invalid_smiles} 条无效SMILES被过滤")
        invalid_genes = bindingdb_filtered[~bindingdb_filtered["smiles_valid"]]["gene_upper"].unique()
        logger.warning(f"  涉及基因: {sorted(invalid_genes)}")
    
    bindingdb_filtered = bindingdb_filtered[bindingdb_filtered["smiles_valid"]].copy()
    
    # 构建输出DataFrame
    bd_supplement = pd.DataFrame({
        "gene": bindingdb_filtered["gene"],
        "uniprot": bindingdb_filtered["uniprot_id"],
        "smiles": bindingdb_filtered["canonical_smiles"],
        "activity_type": bindingdb_filtered["standard_type"],
        "activity_value_nm": bindingdb_filtered["standard_value_nM"],
        "source": bindingdb_filtered["source"],
        "compound_name": bindingdb_filtered["molecule_name"],
        "target_name": bindingdb_filtered["target_name"],
        "pmid": bindingdb_filtered["pmid"],
        "doi": bindingdb_filtered["doi"],
        "note": "BindingDB补充 (CPI v28)"
    })
    
    bd_supplement = bd_supplement.drop_duplicates(subset=["gene", "smiles"], keep="first")
    logger.info(f"  BindingDB补充最终: {len(bd_supplement)} 条记录, {bd_supplement['gene'].nunique()} 个基因")
    
    # 保存
    bd_out_path = RESULTS_V10 / "cpi_supplement_v28_bindingdb.csv"
    bd_supplement.to_csv(bd_out_path, index=False)
    logger.info(f"  已保存到: {bd_out_path}")
else:
    logger.info("BindingDB中没有匹配到zero-shot基因的数据")
    bd_supplement = pd.DataFrame()

# ============================================================
# 步骤4: 从DrugBank补充数据
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("步骤4: 从DrugBank补充CPI数据")
logger.info("=" * 60)

# DrugBank数据需要从drugbank_active_compounds.csv加载，但该文件没有SMILES
# 需要从drugbank_supplemental.csv获取SMILES
drugbank_ref = pd.read_csv(RESULTS / "drugbank_active_compounds.csv", low_memory=False)
logger.info(f"DrugBank参考数据: {len(drugbank_ref)} 条记录")
logger.info(f"DrugBank参考列名: {drugbank_ref.columns.tolist()}")

# 检查是否有supplemental文件
drugbank_supp = RESULTS_V10 / "drugbank_supplemental.csv"
if drugbank_supp.exists():
    drugbank_supp_df = pd.read_csv(drugbank_supp, low_memory=False)
    logger.info(f"DrugBank补充数据: {len(drugbank_supp_df)} 条记录")
    logger.info(f"DrugBank补充列名: {drugbank_supp_df.columns.tolist()}")
else:
    logger.warning("drugbank_supplemental.csv 不存在")
    drugbank_supp_df = pd.DataFrame()

# 标准化基因名
if "gene" in drugbank_ref.columns:
    drugbank_ref["gene_upper"] = drugbank_ref["gene"].str.strip().str.upper()
    # 过滤铁衰老96基因中缺乏CPI数据的基因
    db_missing = drugbank_ref[drugbank_ref["gene_upper"].isin(missing_genes)]
    logger.info(f"DrugBank中匹配到zero-shot基因的数据: {len(db_missing)} 条")
    
    if len(db_missing) > 0:
        db_missing_genes = db_missing["gene_upper"].unique()
        logger.info(f"可补充的基因: {sorted(db_missing_genes)}")
        
        # DrugBank参考数据没有SMILES，需要从supplemental获取
        if len(drugbank_supp_df) > 0 and "drugbank_id" in drugbank_supp_df.columns:
            # 合并获取SMILES
            if "canonical_smiles" in drugbank_supp_df.columns or "smiles" in drugbank_supp_df.columns:
                smi_col = "canonical_smiles" if "canonical_smiles" in drugbank_supp_df.columns else "smiles"
                
                db_with_smiles = db_missing.merge(
                    drugbank_supp_df[["drugbank_id", smi_col]].drop_duplicates(subset=["drugbank_id"]),
                    on="drugbank_id", how="left"
                )
                logger.info(f"  合并SMILES后: {len(db_with_smiles)} 条")
                
                # 过滤无SMILES的记录
                db_no_smiles = db_with_smiles[db_with_smiles[smi_col].isna() | (db_with_smiles[smi_col] == "")]
                if len(db_no_smiles) > 0:
                    logger.warning(f"  {len(db_no_smiles)} 条记录无SMILES (DrugBank参考数据可能不包含结构信息)")
                
                db_with_smiles = db_with_smiles[db_with_smiles[smi_col].notna() & (db_with_smiles[smi_col] != "")].copy()
                
                # 验证SMILES
                db_with_smiles["smiles_valid"] = db_with_smiles[smi_col].apply(validate_smiles)
                n_invalid = (~db_with_smiles["smiles_valid"]).sum()
                if n_invalid > 0:
                    logger.warning(f"  {n_invalid} 条无效SMILES被过滤")
                
                db_with_smiles = db_with_smiles[db_with_smiles["smiles_valid"]].copy()
                
                # 构建输出
                db_supplement = pd.DataFrame({
                    "gene": db_with_smiles["gene"],
                    "uniprot": db_with_smiles["uniprot_id"],
                    "smiles": db_with_smiles[smi_col],
                    "activity_type": "DrugBank_reference",
                    "activity_value_nm": np.nan,
                    "source": "DrugBank",
                    "compound_name": db_with_smiles.get("drug_name", np.nan),
                    "target_name": "",
                    "pmid": "",
                    "doi": "",
                    "drugbank_id": db_with_smiles["drugbank_id"],
                    "note": "DrugBank补充 (CPI v28) - UniProt交叉引用已知配体"
                })
                
                db_supplement = db_supplement.drop_duplicates(subset=["gene", "smiles"], keep="first")
                logger.info(f"  DrugBank补充最终: {len(db_supplement)} 条记录, {db_supplement['gene'].nunique()} 个基因")
                
                db_out_path = RESULTS_V10 / "cpi_supplement_v28_drugbank.csv"
                db_supplement.to_csv(db_out_path, index=False)
                logger.info(f"  已保存到: {db_out_path}")
            else:
                logger.warning("DrugBank supplemental数据无SMILES列，无法补充")
                db_supplement = pd.DataFrame()
        else:
            logger.warning("无法获取DrugBank SMILES数据")
            db_supplement = pd.DataFrame()
    else:
        logger.info("DrugBank中没有匹配到zero-shot基因的数据")
        db_supplement = pd.DataFrame()
else:
    logger.warning("DrugBank参考数据缺少gene列")
    db_supplement = pd.DataFrame()

# ============================================================
# 步骤5: 合并所有补充数据
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("步骤5: 合并所有补充数据")
logger.info("=" * 60)

# 收集所有补充数据源
all_supplements = []

# 添加v25补充数据
v25_for_merge = v25[["gene", "uniprot", "smiles"]].copy() if "smiles" in v25.columns else pd.DataFrame()
if len(v25_for_merge) > 0:
    all_supplements.append(v25_for_merge)

# 添加v26补充数据
if "gene" in v26.columns and "smiles" in v26.columns and "uniprot" in v26.columns:
    v26_for_merge = v26[["gene", "uniprot", "smiles"]].copy()
    all_supplements.append(v26_for_merge)

# 添加v27补充数据
if "gene" in v27.columns and "smiles" in v27.columns and "uniprot" in v27.columns:
    v27_for_merge = v27[["gene", "uniprot", "smiles"]].copy()
    all_supplements.append(v27_for_merge)

# 添加BindingDB补充
if len(bd_supplement) > 0:
    bd_for_merge = bd_supplement[["gene", "uniprot", "smiles"]].copy()
    all_supplements.append(bd_for_merge)

# 添加DrugBank补充
if len(db_supplement) > 0:
    db_for_merge = db_supplement[["gene", "uniprot", "smiles"]].copy()
    all_supplements.append(db_for_merge)

# 合并
if all_supplements:
    merged = pd.concat(all_supplements, ignore_index=True)
    merged["gene"] = merged["gene"].str.strip().str.upper()
    logger.info(f"合并前总计: {len(merged)} 条记录")
    
    # 去重 (gene + SMILES)
    merged = merged.drop_duplicates(subset=["gene", "smiles"], keep="first")
    logger.info(f"去重后: {len(merged)} 条记录")
    
    # 验证SMILES
    merged["smiles_valid"] = merged["smiles"].apply(validate_smiles)
    n_invalid = (~merged["smiles_valid"]).sum()
    if n_invalid > 0:
        logger.warning(f"合并后存在 {n_invalid} 条无效SMILES，将被过滤")
        invalid_genes = merged[~merged["smiles_valid"]]["gene"].unique()
        logger.warning(f"涉及基因: {sorted(invalid_genes)}")
    merged = merged[merged["smiles_valid"]].copy()
    merged = merged.drop(columns=["smiles_valid"])
    
    # 去除与主CPI数据重复的 gene+SMILES
    main_cpi["gene_upper"] = main_cpi["gene"].str.strip().str.upper()
    main_cpi_pairs = set(zip(main_cpi["gene_upper"], main_cpi["canonical_smiles"], strict=False))
    
    merged["pair"] = list(zip(merged["gene"], merged["smiles"], strict=False))
    n_before = len(merged)
    merged = merged[~merged["pair"].isin(main_cpi_pairs)].copy()
    n_removed = n_before - len(merged)
    if n_removed > 0:
        logger.info(f"与主CPI数据去重移除: {n_removed} 条")
    merged = merged.drop(columns=["pair"])
    
    # 统计补充了多少基因
    new_genes = set(merged["gene"]) & missing_genes
    logger.info(f"\n补充统计:")
    logger.info(f"  最终补充记录: {len(merged)} 条")
    logger.info(f"  覆盖基因: {merged['gene'].nunique()} 个")
    logger.info(f"  其中新补充的zero-shot基因: {len(new_genes)} 个")
    logger.info(f"  新补充基因列表: {sorted(new_genes)}")
    
    # 仍缺失的基因
    still_missing = missing_genes - new_genes
    if still_missing:
        logger.info(f"  仍无CPI数据的基因: {len(still_missing)} 个")
        logger.info(f"  仍缺失列表: {sorted(still_missing)}")
    
    # 保存最终补充文件
    out_path = RESULTS_V10 / "cpi_supplement_v28.csv"
    # 保持与主加载脚本兼容的列名: gene, smiles, uniprot
    merged.to_csv(out_path, index=False)
    logger.info(f"\n最终补充文件已保存到: {out_path}")
    logger.info(f"文件大小: {out_path.stat().st_size / 1024:.1f} KB")
else:
    logger.warning("没有可合并的补充数据！")
    merged = pd.DataFrame()

# ============================================================
# 步骤6: 检查主脚本更新需求
# ============================================================
logger.info("\n" + "=" * 60)
logger.info("步骤6: 主脚本加载逻辑检查")
logger.info("=" * 60)

# 检查主脚本中的supplement_paths
main_script = BASE / "scripts" / "phase4_v10_minibatch.py"
if main_script.exists():
    logger.info(f"主脚本路径: {main_script}")
    # 检查是否已包含v28路径
    with open(main_script, 'r', encoding='utf-8') as f:
        content = f.read()
    if "cpi_supplement_v28" in content:
        logger.info("主脚本已包含cpi_supplement_v28.csv路径")
    else:
        logger.info("主脚本尚未包含cpi_supplement_v28.csv，需要手动添加")
        logger.info("需要在supplement_paths列表中添加: L4_RESULTS / 'cpi_supplement_v28.csv'")
else:
    logger.warning(f"主脚本不存在: {main_script}")

logger.info("\n" + "=" * 60)
logger.info("CPI数据补充脚本 v28 执行完成")
logger.info("=" * 60)