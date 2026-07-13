#!/usr/bin/env python3
"""
铁衰老项目 - 数据补充脚本 v25
根据数据真实性验证报告，执行以下补充操作:
  1. 铁死亡表型数据去重修复 (7对跨标签高相似SMILES)
  2. CPI训练集泄漏标记 (12个TCM化合物与CPI训练集重叠)
  3. KEGG通路缺失基因补充 (尝试Reactome/WikiPathways)
  4. 蛋白特征缺失基因补充 (尝试UniProt API)
  5. 生成补充报告
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback

import pandas as pd
import requests
from rdkit import Chem

# ============================================================
# 配置
# ============================================================
BASE = r"d:\铁衰老 绝不重蹈覆辙"
PHENO_PATH = os.path.join(BASE, "L4", "results_v10_minibatch", "phenotype_ferroptosis_dataset.csv")
PHENO_CLEAN_PATH = os.path.join(BASE, "L4", "results_v10_minibatch", "phenotype_ferroptosis_dataset_v25_clean.csv")
CPI_PATH = os.path.join(BASE, "L4", "results", "experimental_actives_detail_cleaned.csv")
TCM_POOL_PATH = os.path.join(BASE, "L3", "results", "tcm_compound_pool_v21_Alevel.csv")
KEGG_PATHWAY_PATH = os.path.join(BASE, "L2", "results", "kegg_pathways", "kegg_human_pathway_genes.tsv")
GENES_PATH = os.path.join(BASE, "L1", "results", "ferroaging_genes_96.csv")
PROTEIN_FEAT_PATH = os.path.join(BASE, "L2", "results", "target_protein_features.csv")
LOGS_DIR = os.path.join(BASE, "L4", "logs")
CPI_LEAK_PATH = os.path.join(LOGS_DIR, "cpi_leakage_v25.txt")
REPORT_PATH = os.path.join(LOGS_DIR, "data_supplement_report_v25.txt")

os.makedirs(LOGS_DIR, exist_ok=True)

# ── 日志配置 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "data_supplement_v25.log"), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

report_lines = []
supplement_results = {}

def log(msg):
    """同时输出到日志和报告缓冲区"""
    logger.info(msg)
    report_lines.append(msg)

def log_section(title):
    log("\n" + "=" * 70)
    log(f"  {title}")
    log("=" * 70)

# ============================================================
# 工具函数
# ============================================================

def get_canonical_nostereo(smi):
    """生成非立体化学感知的规范SMILES"""
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def is_valid_smiles(smi):
    """检查SMILES是否有效"""
    if pd.isna(smi) or not isinstance(smi, str) or smi.strip() == "":
        return False
    mol = Chem.MolFromSmiles(smi.strip())
    return mol is not None

# ============================================================
# 任务1: 铁死亡表型数据去重修复
# ============================================================

def task1_deduplicate_phenotype():
    log_section("任务1: 铁死亡表型数据去重修复")
    
    try:
        pheno_df = pd.read_csv(PHENO_PATH)
        log(f"  读取文件: {PHENO_PATH}")
        log(f"  原始行数: {len(pheno_df)}")
        log(f"  列名: {list(pheno_df.columns)}")
        log(f"  标签分布: {pheno_df['label'].value_counts().to_dict()}")
        
        # 生成非立体化学感知的SMILES
        pheno_df["smiles_nostereo"] = pheno_df["canonical_smiles"].apply(get_canonical_nostereo)
        
        # 找到跨标签的重复SMILES
        dupe_groups = pheno_df.groupby("smiles_nostereo")["label"].nunique()
        cross_label_dupes = dupe_groups[dupe_groups > 1].index.tolist()
        
        log(f"\n  跨标签重复SMILES: {len(cross_label_dupes)} 个")
        
        for smi in cross_label_dupes:
            subset = pheno_df[pheno_df["smiles_nostereo"] == smi]
            names = subset.get("compound_name", "N/A").tolist()
            labels = subset["label"].tolist()
            sources = subset.get("source", "N/A").tolist()
            log(f"    SMILES: {smi}")
            log(f"      labels: {labels}, names: {names}, sources: {sources}")
        
        # 修复策略：保留第一个出现的标签，删除重复
        pheno_df_clean = pheno_df.drop_duplicates(subset=["smiles_nostereo"], keep="first")
        pheno_df_clean = pheno_df_clean.drop(columns=["smiles_nostereo"])
        
        log(f"\n  清洗前: {len(pheno_df)}, 清洗后: {len(pheno_df_clean)}")
        log(f"  删除行数: {len(pheno_df) - len(pheno_df_clean)}")
        
        # 同时输出同标签重复（仅为信息）
        same_label_dupes = dupe_groups[dupe_groups == 1].index.tolist()
        same_label_count = 0
        for smi in same_label_dupes:
            subset = pheno_df[pheno_df["smiles_nostereo"] == smi]
            if len(subset) > 1:
                same_label_count += 1
        
        log(f"  同标签重复SMILES(去重前): {same_label_count} 组")
        
        pheno_df_clean.to_csv(PHENO_CLEAN_PATH, index=False)
        log(f"\n  ✅ 清洗后数据已保存: {PHENO_CLEAN_PATH}")
        
        supplement_results["task1"] = {
            "status": "SUCCESS",
            "original_rows": len(pheno_df),
            "cleaned_rows": len(pheno_df_clean),
            "removed_rows": len(pheno_df) - len(pheno_df_clean),
            "cross_label_dupes": len(cross_label_dupes),
            "cross_label_details": {smi: pheno_df[pheno_df["smiles_nostereo"] == smi]["label"].tolist() for smi in cross_label_dupes}
        }
        
    except Exception as e:
        log(f"  ❌ 任务1失败: {e}")
        log(traceback.format_exc())
        supplement_results["task1"] = {"status": "FAILED", "error": str(e)}

# ============================================================
# 任务2: CPI训练集泄漏标记
# ============================================================

def task2_cpi_leakage():
    log_section("任务2: CPI训练集泄漏标记")
    
    try:
        # 读取CPI训练集
        cpi_df = pd.read_csv(CPI_PATH)
        log(f"  读取CPI训练集: {CPI_PATH}")
        log(f"  行数: {len(cpi_df)}, 唯一SMILES: {cpi_df['canonical_smiles'].nunique()}")
        
        # 读取TCM候选池
        tcm_df = pd.read_csv(TCM_POOL_PATH)
        log(f"  读取TCM候选池: {TCM_POOL_PATH}")
        log(f"  行数: {len(tcm_df)}, 列名: {list(tcm_df.columns)}")
        
        # 确定SMILES列名
        smiles_col = None
        for col in tcm_df.columns:
            if 'smiles' in col.lower():
                smiles_col = col
                break
        if smiles_col is None:
            log("  ❌ 未找到SMILES列")
            supplement_results["task2"] = {"status": "FAILED", "error": "未找到SMILES列"}
            return
        
        log(f"  使用SMILES列: {smiles_col}")
        
        # 生成非立体化学感知的SMILES进行匹配
        cpi_smiles_set = set()
        cpi_smiles_detail = {}
        for _, row in cpi_df.iterrows():
            smi = str(row["canonical_smiles"])
            smi_nostereo = get_canonical_nostereo(smi)
            cpi_smiles_set.add(smi_nostereo)
            if smi_nostereo not in cpi_smiles_detail:
                cpi_smiles_detail[smi_nostereo] = []
            cpi_smiles_detail[smi_nostereo].append({
                "smiles": smi,
                "gene": row.get("gene", "N/A"),
                "source": row.get("source", "N/A")
            })
        
        log(f"  CPI训练集唯一SMILES(非立体): {len(cpi_smiles_set)}")
        
        # 找出重叠的SMILES
        tcm_smiles = tcm_df[smiles_col].dropna().astype(str).tolist()
        tcm_smiles_nostereo = [get_canonical_nostereo(s) for s in tcm_smiles]
        
        overlap_smiles = []
        overlap_mol_ids = []
        for i, (smi, smi_ns) in enumerate(zip(tcm_smiles, tcm_smiles_nostereo, strict=False)):
            if smi_ns in cpi_smiles_set:
                overlap_smiles.append(smi)
                overlap_mol_ids.append(tcm_df.iloc[i].get("MOL_ID", f"row_{i}"))
        
        log(f"\n  重叠SMILES数: {len(overlap_smiles)}")
        
        # 输出详细信息
        leakage_lines = []
        leakage_lines.append("=" * 70)
        leakage_lines.append("CPI训练集泄漏标记报告 v25")
        leakage_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        leakage_lines.append("=" * 70)
        leakage_lines.append(f"CPI训练集SMILES数: {len(cpi_smiles_set)} (非立体化学感知)")
        leakage_lines.append(f"TCM化合物池SMILES数: {len(tcm_smiles)}")
        leakage_lines.append(f"重叠SMILES数: {len(overlap_smiles)}")
        leakage_lines.append("")
        
        if overlap_smiles:
            leakage_lines.append("--- 重叠详情 ---")
            for i, smi in enumerate(overlap_smiles):
                smi_ns = get_canonical_nostereo(smi)
                mol_id = overlap_mol_ids[i]
                cpi_info = cpi_smiles_detail.get(smi_ns, [])
                genes = list({x["gene"] for x in cpi_info})
                sources = list({x["source"] for x in cpi_info})
                leakage_lines.append(f"  [{i+1}] MOL_ID={mol_id}")
                leakage_lines.append(f"      SMILES: {smi}")
                leakage_lines.append(f"      CPI训练集中关联基因: {genes}")
                leakage_lines.append(f"      CPI训练集数据来源: {sources}")
                leakage_lines.append("")
        else:
            leakage_lines.append("✅ 无重叠SMILES，无泄漏风险")
        
        leakage_lines.append("")
        leakage_lines.append("--- 建议 ---")
        leakage_lines.append("在预测时自动排除这些重叠化合物，避免数据泄漏影响评估。")
        leakage_lines.append("可将这些化合物的MOL_ID加入黑名单。")
        leakage_lines.append("")
        leakage_lines.append("--- 黑名单MOL_ID列表 ---")
        leakage_lines.append(", ".join([str(x) for x in overlap_mol_ids]))
        
        leakage_text = "\n".join(leakage_lines)
        with open(CPI_LEAK_PATH, "w", encoding="utf-8") as f:
            f.write(leakage_text)
        
        log(f"\n  ✅ 泄漏报告已保存: {CPI_LEAK_PATH}")
        log(f"  重叠MOL_ID: {overlap_mol_ids}")
        
        supplement_results["task2"] = {
            "status": "SUCCESS",
            "cpi_unique_smiles": len(cpi_smiles_set),
            "tcm_smiles_count": len(tcm_smiles),
            "overlap_count": len(overlap_smiles),
            "overlap_mol_ids": overlap_mol_ids
        }
        
    except Exception as e:
        log(f"  ❌ 任务2失败: {e}")
        log(traceback.format_exc())
        supplement_results["task2"] = {"status": "FAILED", "error": str(e)}

# ============================================================
# 任务3: KEGG通路缺失基因补充
# ============================================================

def task3_kegg_pathway_supplement():
    log_section("任务3: KEGG通路缺失基因补充")
    
    try:
        # 读取铁衰老96基因
        genes_df = pd.read_csv(GENES_PATH)
        ferroaging_genes = set(genes_df["gene_symbol"].tolist())
        log(f"  铁衰老基因总数: {len(ferroaging_genes)}")
        
        # 读取KEGG通路基因
        kegg_df = pd.read_csv(KEGG_PATHWAY_PATH, sep="\t")
        kegg_genes = set(kegg_df["gene_symbol"].tolist())
        log(f"  KEGG通路覆盖基因总数: {len(kegg_genes)}")
        log(f"  KEGG通路总数: {kegg_df['pathway_id'].nunique()}")
        
        # 找到缺失基因
        genes_in_kegg = ferroaging_genes & kegg_genes
        genes_missing_kegg = ferroaging_genes - kegg_genes
        
        log(f"\n  铁衰老基因在KEGG中: {len(genes_in_kegg)}")
        log(f"  铁衰老基因缺失KEGG: {len(genes_missing_kegg)}")
        log(f"  缺失基因列表: {sorted(genes_missing_kegg)}")
        
        # 对于在KEGG中的基因，记录通路信息
        kegg_pathway_info = {}
        for gene in genes_in_kegg:
            gene_pathways = kegg_df[kegg_df["gene_symbol"] == gene]
            pathways = list(zip(gene_pathways["pathway_id"], gene_pathways["pathway_name"], strict=False))
            kegg_pathway_info[gene] = pathways
        
        # 尝试从Reactome API补充
        log("\n  --- 尝试Reactome API补充 ---")
        reactome_supplemented = {}
        reactome_failed = []
        
        for gene in sorted(genes_missing_kegg):
            try:
                url = f"https://reactome.org/ContentService/data/query/genes/pathways/{gene}"
                resp = requests.get(url, timeout=10, headers={"Accept": "application/json"})
                if resp.status_code == 200:
                    pathways_data = resp.json()
                    if pathways_data and len(pathways_data) > 0:
                        reactome_pathways = []
                        for p in pathways_data[:5]:  # 取前5条
                            p_name = p.get("displayName", p.get("name", "N/A"))
                            p_id = p.get("stId", "N/A")
                            reactome_pathways.append((p_id, p_name))
                        reactome_supplemented[gene] = reactome_pathways
                        log(f"    ✅ {gene}: Reactome找到 {len(reactome_pathways)} 条通路")
                    else:
                        reactome_failed.append(gene)
                        log(f"    ⚠️ {gene}: Reactome无通路数据")
                else:
                    reactome_failed.append(gene)
                    log(f"    ⚠️ {gene}: Reactome API返回 {resp.status_code}")
            except Exception as e:
                reactome_failed.append(gene)
                log(f"    ⚠️ {gene}: Reactome请求失败 ({e})")
        
        # 尝试从WikiPathways API补充
        log("\n  --- 尝试WikiPathways API补充 ---")
        wikipathways_supplemented = {}
        wikipathways_failed = []
        
        for gene in reactome_failed:
            try:
                url = f"https://www.wikipathways.org/wikipathways/rest/findPathwaysByGene?gene={gene}&species=Homo sapiens"
                resp = requests.get(url, timeout=10, headers={"Accept": "application/json"})
                if resp.status_code == 200:
                    data = resp.json()
                    if "result" in data and len(data["result"]) > 0:
                        wp_pathways = []
                        for p in data["result"][:5]:
                            p_name = p.get("name", "N/A")
                            p_id = p.get("id", "N/A")
                            wp_pathways.append((p_id, p_name))
                        wikipathways_supplemented[gene] = wp_pathways
                        log(f"    ✅ {gene}: WikiPathways找到 {len(wp_pathways)} 条通路")
                    else:
                        wikipathways_failed.append(gene)
                        log(f"    ⚠️ {gene}: WikiPathways无通路数据")
                else:
                    wikipathways_failed.append(gene)
                    log(f"    ⚠️ {gene}: WikiPathways API返回 {resp.status_code}")
            except Exception as e:
                wikipathways_failed.append(gene)
                log(f"    ⚠️ {gene}: WikiPathways请求失败 ({e})")
        
        # 计算补充结果
        total_reactome = len(reactome_supplemented)
        total_wikipathways = len(wikipathways_supplemented)
        still_missing = wikipathways_failed  # 仍无法补充的
        
        log("\n  --- 补充结果汇总 ---")
        log(f"  Reactome补充: {total_reactome} 个基因")
        log(f"  WikiPathways补充: {total_wikipathways} 个基因")
        log(f"  仍缺失: {len(still_missing)} 个基因")
        if still_missing:
            log(f"  仍缺失基因: {sorted(still_missing)}")
        
        supplement_results["task3"] = {
            "status": "SUCCESS",
            "ferroaging_genes_total": len(ferroaging_genes),
            "in_kegg": len(genes_in_kegg),
            "missing_kegg": len(genes_missing_kegg),
            "missing_kegg_genes": sorted(genes_missing_kegg),
            "reactome_supplemented": total_reactome,
            "reactome_genes": {g: [p[1] for p in paths] for g, paths in reactome_supplemented.items()},
            "wikipathways_supplemented": total_wikipathways,
            "wikipathways_genes": {g: [p[1] for p in paths] for g, paths in wikipathways_supplemented.items()},
            "still_missing": len(still_missing),
            "still_missing_genes": sorted(still_missing),
            "kegg_pathway_info": {g: [p[1] for p in paths] for g, paths in kegg_pathway_info.items()}
        }
        
    except Exception as e:
        log(f"  ❌ 任务3失败: {e}")
        log(traceback.format_exc())
        supplement_results["task3"] = {"status": "FAILED", "error": str(e)}

# ============================================================
# 任务4: 蛋白特征缺失基因补充
# ============================================================

def task4_protein_feature_supplement():
    log_section("任务4: 蛋白特征缺失基因补充")
    
    try:
        # 读取铁衰老96基因
        genes_df = pd.read_csv(GENES_PATH)
        ferroaging_genes = set(genes_df["gene_symbol"].tolist())
        
        # 读取蛋白特征表
        protein_df = pd.read_csv(PROTEIN_FEAT_PATH)
        protein_genes = set(protein_df["gene_symbol"].tolist())
        log(f"  蛋白特征表基因数: {len(protein_genes)}")
        
        # 找到缺失基因
        genes_in_protein = ferroaging_genes & protein_genes
        genes_missing_protein = ferroaging_genes - protein_genes
        
        log(f"  铁衰老基因有蛋白特征: {len(genes_in_protein)}")
        log(f"  铁衰老基因缺失蛋白特征: {len(genes_missing_protein)}")
        log(f"  缺失基因列表(前20): {sorted(genes_missing_protein)[:20]}")
        
        # 尝试从UniProt API补充
        log("\n  --- 尝试UniProt REST API补充 ---")
        supplemented_features = {}
        failed_genes = []
        
        for gene in sorted(genes_missing_protein):
            try:
                # UniProt REST API: 通过基因名搜索人类蛋白
                url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene}+AND+organism_id:9606&format=json&size=5"
                resp = requests.get(url, timeout=15, headers={"Accept": "application/json"})
                
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    
                    if results:
                        # 取最佳匹配（通常是reviewed的优先）
                        best_entry = None
                        for entry in results:
                            entry_type = entry.get("entryType", "")
                            if entry_type == "UniProtKB reviewed (Swiss-Prot)":
                                best_entry = entry
                                break
                        if best_entry is None:
                            best_entry = results[0]
                        
                        uniprot_id = best_entry.get("primaryAccession", "N/A")
                        protein_name = best_entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                        
                        # 提取序列信息
                        sequence_info = best_entry.get("sequence", {})
                        seq_length = sequence_info.get("length", 0)
                        seq_mass = sequence_info.get("molWeight", 0)
                        
                        # 提取跨膜信息
                        features = best_entry.get("features", [])
                        n_transmembrane = sum(1 for f in features if f.get("type") == "Transmembrane")
                        n_signal = sum(1 for f in features if f.get("type") == "Signal")
                        n_domains = sum(1 for f in features if f.get("type") == "Domain")
                        
                        # 提取亚细胞定位
                        comments = best_entry.get("comments", [])
                        subcellular_locations = []
                        for c in comments:
                            if c.get("commentType") == "SUBCELLULAR LOCATION":
                                locs = c.get("subcellularLocations", [])
                                for loc in locs:
                                    subcellular_locations.append(loc.get("location", {}).get("value", "N/A"))
                        
                        supplemented_features[gene] = {
                            "uniprot_id": uniprot_id,
                            "protein_name": protein_name,
                            "length": seq_length,
                            "mass": seq_mass,
                            "n_domains": n_domains,
                            "n_transmembrane": n_transmembrane,
                            "has_signal_peptide": n_signal > 0,
                            "subcellular_main": subcellular_locations[0] if subcellular_locations else "N/A",
                            "reviewed": "Swiss-Prot" in best_entry.get("entryType", "")
                        }
                        
                        log(f"    ✅ {gene} -> {uniprot_id}: {protein_name} (长度={seq_length}, 质量={seq_mass:.0f}Da)")
                    else:
                        failed_genes.append(gene)
                        log(f"    ⚠️ {gene}: UniProt无搜索结果")
                else:
                    failed_genes.append(gene)
                    log(f"    ⚠️ {gene}: UniProt API返回 {resp.status_code}")
                    
            except Exception as e:
                failed_genes.append(gene)
                log(f"    ⚠️ {gene}: UniProt请求失败 ({e})")
            
            # 礼貌延迟，避免API限流
            time.sleep(0.3)
        
        log("\n  --- 补充结果汇总 ---")
        log(f"  UniProt补充成功: {len(supplemented_features)} 个基因")
        log(f"  补充失败: {len(failed_genes)} 个基因")
        if failed_genes:
            log(f"  失败基因列表: {sorted(failed_genes)}")
        
        supplement_results["task4"] = {
            "status": "SUCCESS",
            "ferroaging_genes_total": len(ferroaging_genes),
            "in_protein_features": len(genes_in_protein),
            "missing_protein_features": len(genes_missing_protein),
            "missing_protein_genes": sorted(genes_missing_protein),
            "uniprot_supplemented": len(supplemented_features),
            "uniprot_features": supplemented_features,
            "uniprot_failed": len(failed_genes),
            "uniprot_failed_genes": sorted(failed_genes)
        }
        
    except Exception as e:
        log(f"  ❌ 任务4失败: {e}")
        log(traceback.format_exc())
        supplement_results["task4"] = {"status": "FAILED", "error": str(e)}

# ============================================================
# 任务5: 生成补充报告
# ============================================================

def task5_generate_report():
    log_section("任务5: 生成补充报告")
    
    try:
        lines = []
        lines.append("=" * 70)
        lines.append("  铁衰老项目 - 数据补充报告 v25")
        lines.append(f"  生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")
        
        # 任务1总结
        lines.append("-" * 70)
        lines.append("【任务1】铁死亡表型数据去重修复")
        lines.append("-" * 70)
        t1 = supplement_results.get("task1", {})
        if t1.get("status") == "SUCCESS":
            lines.append("  状态: ✅ 成功")
            lines.append(f"  原始行数: {t1['original_rows']}")
            lines.append(f"  清洗后行数: {t1['cleaned_rows']}")
            lines.append(f"  删除行数: {t1['removed_rows']}")
            lines.append(f"  跨标签重复组数: {t1['cross_label_dupes']}")
            lines.append(f"  输出文件: {PHENO_CLEAN_PATH}")
        else:
            lines.append(f"  状态: ❌ 失败 - {t1.get('error', 'Unknown')}")
        
        lines.append("")
        
        # 任务2总结
        lines.append("-" * 70)
        lines.append("【任务2】CPI训练集泄漏标记")
        lines.append("-" * 70)
        t2 = supplement_results.get("task2", {})
        if t2.get("status") == "SUCCESS":
            lines.append("  状态: ✅ 成功")
            lines.append(f"  CPI训练集唯一SMILES: {t2['cpi_unique_smiles']}")
            lines.append(f"  TCM候选池SMILES数: {t2['tcm_smiles_count']}")
            lines.append(f"  重叠SMILES数: {t2['overlap_count']}")
            lines.append(f"  重叠MOL_ID: {t2['overlap_mol_ids']}")
            lines.append(f"  输出文件: {CPI_LEAK_PATH}")
        else:
            lines.append(f"  状态: ❌ 失败 - {t2.get('error', 'Unknown')}")
        
        lines.append("")
        
        # 任务3总结
        lines.append("-" * 70)
        lines.append("【任务3】KEGG通路缺失基因补充")
        lines.append("-" * 70)
        t3 = supplement_results.get("task3", {})
        if t3.get("status") == "SUCCESS":
            lines.append("  状态: ✅ 成功")
            lines.append(f"  铁衰老基因总数: {t3['ferroaging_genes_total']}")
            lines.append(f"  KEGG通路覆盖: {t3['in_kegg']} 个基因")
            lines.append(f"  KEGG通路缺失: {t3['missing_kegg']} 个基因")
            lines.append(f"  Reactome补充: {t3['reactome_supplemented']} 个基因")
            lines.append(f"  WikiPathways补充: {t3['wikipathways_supplemented']} 个基因")
            lines.append(f"  仍缺失: {t3['still_missing']} 个基因")
            if t3['still_missing'] > 0:
                lines.append(f"  仍缺失基因: {t3['still_missing_genes']}")
            if t3['reactome_supplemented'] > 0:
                lines.append("  Reactome补充详情:")
                for g, paths in t3['reactome_genes'].items():
                    lines.append(f"    {g}: {', '.join(paths)}")
            if t3['wikipathways_supplemented'] > 0:
                lines.append("  WikiPathways补充详情:")
                for g, paths in t3['wikipathways_genes'].items():
                    lines.append(f"    {g}: {', '.join(paths)}")
        else:
            lines.append(f"  状态: ❌ 失败 - {t3.get('error', 'Unknown')}")
        
        lines.append("")
        
        # 任务4总结
        lines.append("-" * 70)
        lines.append("【任务4】蛋白特征缺失基因补充")
        lines.append("-" * 70)
        t4 = supplement_results.get("task4", {})
        if t4.get("status") == "SUCCESS":
            lines.append("  状态: ✅ 成功")
            lines.append(f"  铁衰老基因总数: {t4['ferroaging_genes_total']}")
            lines.append(f"  已有蛋白特征: {t4['in_protein_features']} 个基因")
            lines.append(f"  缺失蛋白特征: {t4['missing_protein_features']} 个基因")
            lines.append(f"  UniProt API补充成功: {t4['uniprot_supplemented']} 个基因")
            lines.append(f"  UniProt API补充失败: {t4['uniprot_failed']} 个基因")
            if t4['uniprot_supplemented'] > 0:
                lines.append("  UniProt补充详情:")
                for g, feat in t4['uniprot_features'].items():
                    lines.append(f"    {g} ({feat['uniprot_id']}): {feat['protein_name']}, "
                               f"长度={feat['length']}, 质量={feat['mass']:.0f}Da, "
                               f"结构域={feat['n_domains']}, 跨膜={feat['n_transmembrane']}, "
                               f"亚细胞={feat['subcellular_main']}")
            if t4['uniprot_failed'] > 0:
                lines.append(f"  UniProt失败基因: {t4['uniprot_failed_genes']}")
        else:
            lines.append(f"  状态: ❌ 失败 - {t4.get('error', 'Unknown')}")
        
        lines.append("")
        
        # 总体建议
        lines.append("-" * 70)
        lines.append("【总体建议】")
        lines.append("-" * 70)
        lines.append("  1. 在模型训练前使用清洗后的表型数据集")
        lines.append("  2. 预测时自动排除CPI泄漏的TCM化合物")
        lines.append("  3. 将Reactome/WikiPathways补充的通路信息纳入基因功能注释")
        lines.append("  4. 将UniProt补充的蛋白特征纳入蛋白特征矩阵")
        lines.append("")
        
        report_text = "\n".join(lines)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        log(f"\n  ✅ 补充报告已保存: {REPORT_PATH}")
        
        # 也保存JSON版本
        json_path = REPORT_PATH.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(supplement_results, f, ensure_ascii=False, indent=2, default=str)
        log(f"  ✅ JSON结果已保存: {json_path}")
        
    except Exception as e:
        log(f"  ❌ 任务5失败: {e}")
        log(traceback.format_exc())

# ============================================================
# 主入口
# ============================================================

def main():
    log("=" * 70)
    log("  铁衰老项目 - 数据补充脚本 v25")
    log(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    
    # 任务1: 铁死亡表型数据去重
    task1_deduplicate_phenotype()
    
    # 任务2: CPI训练集泄漏标记
    task2_cpi_leakage()
    
    # 任务3: KEGG通路缺失基因补充
    task3_kegg_pathway_supplement()
    
    # 任务4: 蛋白特征缺失基因补充
    task4_protein_feature_supplement()
    
    # 任务5: 生成补充报告
    task5_generate_report()
    
    log(f"\n{'=' * 70}")
    log(f"  所有任务完成，结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 70}")

if __name__ == "__main__":
    main()