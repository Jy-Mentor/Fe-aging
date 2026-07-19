"""
补充铁衰老基因的传统蛋白特征
对缺失的61个基因，通过UniProt API获取蛋白序列，使用BioPython计算传统特征
"""
import pandas as pd
import numpy as np
import requests
import time
import sys
import traceback
import logging
from pathlib import Path
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent.parent / 'results' / 'supplement_protein_features.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# === 配置 ===
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_TAB_URL = "https://rest.uniprot.org/uniprotkb/search"
REQUEST_DELAY = 0.5  # 每次请求间隔秒数，避免被限流
MAX_RETRIES = 3

BASE_DIR = Path(__file__).parent.parent.parent
GENES_96_PATH = BASE_DIR / "L1" / "results" / "ferroaging_genes_96.csv"
EXISTING_FEATURES_PATH = BASE_DIR / "L2" / "results" / "target_protein_features.csv"
OUTPUT_PATH = BASE_DIR / "L2" / "results" / "target_protein_features_supplemented.csv"


def fetch_uniprot_by_gene(gene_symbol, organism="human", retries=MAX_RETRIES):
    """
    通过基因名搜索UniProt，获取人类蛋白的序列和注释信息
    返回 dict 或 None
    """
    query = f"gene:{gene_symbol}+AND+organism_id:9606&format=json&size=5"
    url = f"{UNIPROT_SEARCH_URL}?query={query}"
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if not results:
                    logger.warning(f"  [{gene_symbol}] UniProt搜索无结果")
                    return None
                
                # 优先选择reviewed (Swiss-Prot)条目，否则选第一个
                reviewed = [r for r in results if r.get("entryType") == "UniProtKB reviewed (Swiss-Prot)"]
                if reviewed:
                    entry = reviewed[0]
                else:
                    entry = results[0]
                
                return parse_uniprot_entry(entry, gene_symbol)
            elif resp.status_code == 429:
                wait = (attempt + 1) * 5
                logger.warning(f"  [{gene_symbol}] 速率限制，等待 {wait}s...")
                time.sleep(wait)
            else:
                logger.warning(f"  [{gene_symbol}] HTTP {resp.status_code}")
                time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning(f"  [{gene_symbol}] 请求异常 (attempt {attempt+1}): {e}")
            time.sleep(REQUEST_DELAY)
    
    return None


def parse_uniprot_entry(entry, gene_symbol):
    """解析UniProt JSON条目，提取蛋白信息"""
    info = {}
    info["gene_symbol"] = gene_symbol
    info["uniprot_id"] = entry.get("primaryAccession", "")
    info["reviewed"] = "Swiss-Prot" in entry.get("entryType", "")
    
    # 蛋白名称
    pd_entry = entry.get("proteinDescription", {})
    if "recommendedName" in pd_entry:
        info["protein_name"] = pd_entry["recommendedName"].get("fullName", {}).get("value", "")
    elif "submissionNames" in pd_entry and pd_entry["submissionNames"]:
        info["protein_name"] = pd_entry["submissionNames"][0].get("fullName", {}).get("value", "")
    else:
        info["protein_name"] = gene_symbol
    
    # 基因名称（标准名）
    genes = entry.get("genes", [])
    if genes:
        gene_names = genes[0].get("geneName", {})
        info["gene_name"] = gene_names.get("value", gene_symbol)
    else:
        info["gene_name"] = gene_symbol
    
    # 序列
    seq_dict = entry.get("sequence", {})
    info["sequence"] = seq_dict.get("value", "")
    info["sequence_length"] = seq_dict.get("length", 0)
    info["mass"] = seq_dict.get("molWeight", 0)
    
    # 亚细胞定位
    comments = entry.get("comments", [])
    subcellular = []
    for c in comments:
        if c.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in c.get("subcellularLocations", []):
                loc_name = loc.get("location", {}).get("value", "")
                if loc_name:
                    subcellular.append(loc_name)
    info["subcellular_main"] = "; ".join(subcellular[:3]) if subcellular else ""
    
    # 信号肽
    features = entry.get("features", [])
    info["has_signal_peptide"] = False
    for f in features:
        if f.get("type") == "Signal":
            info["has_signal_peptide"] = True
            break
    
    # 跨膜域
    info["has_transmembrane"] = False
    info["n_transmembrane"] = 0
    for f in features:
        if f.get("type") == "Transmembrane":
            info["has_transmembrane"] = True
            info["n_transmembrane"] += 1
    
    # 结构域
    info["n_domains"] = 0
    for f in features:
        if f.get("type") == "Domain":
            info["n_domains"] += 1
    
    # PTM修饰计数
    info["n_ptms"] = 0
    info["n_phospho"] = 0
    info["n_ubiquitination"] = 0
    info["n_acetylation"] = 0
    for f in features:
        ftype = f.get("type", "")
        if "Modified residue" in ftype or "Cross-link" in ftype or "Lipidation" in ftype:
            info["n_ptms"] += 1
        if "Phospho" in f.get("description", "") or "phospho" in ftype.lower():
            info["n_phospho"] += 1
        if "ubiquitin" in f.get("description", "").lower():
            info["n_ubiquitination"] += 1
        if "acetyl" in f.get("description", "").lower() or "acetyl" in ftype.lower():
            info["n_acetylation"] += 1
    
    return info


def calculate_protparam_features(sequence):
    """
    使用BioPython ProtParam计算蛋白传统特征
    返回 dict
    """
    if not sequence or len(sequence) < 10:
        return {}
    
    try:
        analysis = ProteinAnalysis(sequence)
        features = {}
        
        # 分子量
        features["molecular_weight"] = analysis.molecular_weight()
        
        # 等电点
        features["isoelectric_point"] = analysis.isoelectric_point()
        
        # 芳香性
        features["aromaticity"] = analysis.aromaticity()
        
        # 不稳定指数
        features["instability_index"] = analysis.instability_index()
        
        # 疏水性 (GRAVY)
        features["gravy"] = analysis.gravy()
        
        # 氨基酸组成 (AAC)
        aa_percent = analysis.amino_acids_percent
        for aa, pct in aa_percent.items():
            features[f"AAC_{aa}"] = pct
        
        # 二级结构分数
        helix, turn, sheet = analysis.secondary_structure_fraction()
        features["helix_fraction"] = helix
        features["turn_fraction"] = turn
        features["sheet_fraction"] = sheet
        
        # 消光系数 (还原态和氧化态)
        try:
            ext_coeff = analysis.molar_extinction_coefficient()
            features["extinction_coefficient_reduced"] = ext_coeff[0]
            features["extinction_coefficient_oxidized"] = ext_coeff[1]
        except Exception:
            features["extinction_coefficient_reduced"] = None
            features["extinction_coefficient_oxidized"] = None
        
        # 柔性 (flexibility)
        try:
            flex = analysis.flexibility()
            features["flexibility_mean"] = np.mean(flex) if len(flex) > 0 else None
        except Exception:
            features["flexibility_mean"] = None
        
        return features
    except Exception as e:
        logger.error(f"  ProtParam计算失败: {e}")
        traceback.print_exc()
        return {}


def main():
    logger.info("=" * 60)
    logger.info("开始补充铁衰老基因蛋白传统特征")
    logger.info("=" * 60)
    
    # 1. 读取数据
    genes_96 = pd.read_csv(GENES_96_PATH)
    all_genes = set(genes_96["gene_symbol"].unique())
    logger.info("96个铁衰老基因已加载")
    
    features_existing = pd.read_csv(EXISTING_FEATURES_PATH)
    existing_genes_in_96 = set(features_existing["gene_symbol"].unique()) & all_genes
    missing_genes = sorted(all_genes - existing_genes_in_96)
    logger.info(f"已有特征基因（在96列表中）: {len(existing_genes_in_96)}")
    logger.info(f"缺失特征基因: {len(missing_genes)}")
    logger.info(f"缺失基因列表: {missing_genes}")
    
    # 2. 对每个缺失基因获取序列和特征
    supplemented_records = []
    failed_genes = []
    success_count = 0
    
    for i, gene in enumerate(missing_genes):
        idx = i + 1
        logger.info(f"[{idx}/{len(missing_genes)}] 处理基因: {gene}")
        
        # 获取UniProt信息
        uniprot_info = fetch_uniprot_by_gene(gene)
        
        if uniprot_info is None or not uniprot_info.get("sequence"):
            logger.warning(f"  [{gene}] 无法获取蛋白序列，跳过")
            failed_genes.append(gene)
            time.sleep(REQUEST_DELAY)
            continue
        
        seq = uniprot_info["sequence"]
        logger.info(f"  [{gene}] UniProt ID: {uniprot_info['uniprot_id']}, "
                    f"长度: {len(seq)}, Reviewed: {uniprot_info['reviewed']}")
        
        # 计算ProtParam特征
        protparam_features = calculate_protparam_features(seq)
        
        if not protparam_features:
            logger.warning(f"  [{gene}] ProtParam特征计算失败，跳过")
            failed_genes.append(gene)
            time.sleep(REQUEST_DELAY)
            continue
        
        # 合并所有信息
        record = {
            "uniprot_id": uniprot_info["uniprot_id"],
            "protein_name": uniprot_info["protein_name"],
            "gene_name": uniprot_info["gene_name"],
            "length": len(seq),
            "mass": protparam_features.get("molecular_weight", uniprot_info.get("mass", 0)),
            "n_domains": uniprot_info.get("n_domains", 0),
            "n_ptms": uniprot_info.get("n_ptms", 0),
            "n_phospho": uniprot_info.get("n_phospho", 0),
            "n_ubiquitination": uniprot_info.get("n_ubiquitination", 0),
            "n_acetylation": uniprot_info.get("n_acetylation", 0),
            "subcellular_main": uniprot_info.get("subcellular_main", ""),
            "has_signal_peptide": uniprot_info.get("has_signal_peptide", False),
            "has_transmembrane": uniprot_info.get("has_transmembrane", False),
            "n_transmembrane": uniprot_info.get("n_transmembrane", 0),
            "reviewed": uniprot_info.get("reviewed", False),
            "gene_symbol": gene,
            "sequence": seq,
            "sequence_length": len(seq),
            # 新增传统特征
            "isoelectric_point": protparam_features.get("isoelectric_point"),
            "aromaticity": protparam_features.get("aromaticity"),
            "instability_index": protparam_features.get("instability_index"),
            "gravy": protparam_features.get("gravy"),
            "helix_fraction": protparam_features.get("helix_fraction"),
            "turn_fraction": protparam_features.get("turn_fraction"),
            "sheet_fraction": protparam_features.get("sheet_fraction"),
            "extinction_coefficient_reduced": protparam_features.get("extinction_coefficient_reduced"),
            "extinction_coefficient_oxidized": protparam_features.get("extinction_coefficient_oxidized"),
            "flexibility_mean": protparam_features.get("flexibility_mean"),
        }
        
        # 添加AAC
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            record[f"AAC_{aa}"] = protparam_features.get(f"AAC_{aa}", 0)
        
        supplemented_records.append(record)
        success_count += 1
        logger.info(f"  [{gene}] ✓ 成功获取特征 (MW={record['mass']:.1f}, "
                    f"pI={record['isoelectric_point']:.2f}, GRAVY={record['gravy']:.3f})")
        
        time.sleep(REQUEST_DELAY)
    
    # 3. 构建补充DataFrame
    if not supplemented_records:
        logger.error("没有成功获取任何基因的特征！")
        sys.exit(1)
    
    df_supplemented = pd.DataFrame(supplemented_records)
    
    # 确保列顺序与现有CSV一致（先放原有列，再放新增的传统特征列）
    existing_cols = [
        "uniprot_id", "protein_name", "gene_name", "length", "mass",
        "n_domains", "n_ptms", "n_phospho", "n_ubiquitination", "n_acetylation",
        "subcellular_main", "has_signal_peptide", "has_transmembrane", "n_transmembrane",
        "reviewed", "gene_symbol", "sequence", "sequence_length"
    ]
    new_cols = [
        "isoelectric_point", "aromaticity", "instability_index", "gravy",
        "helix_fraction", "turn_fraction", "sheet_fraction",
        "extinction_coefficient_reduced", "extinction_coefficient_oxidized",
        "flexibility_mean"
    ]
    aac_cols = [f"AAC_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]
    
    all_cols = existing_cols + new_cols + aac_cols
    # 只保留存在的列
    all_cols = [c for c in all_cols if c in df_supplemented.columns]
    df_supplemented = df_supplemented[all_cols]
    
    # 4. 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_supplemented.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"\n补充特征已保存到: {OUTPUT_PATH}")
    logger.info(f"补充记录数: {len(df_supplemented)}")
    
    # 5. 统计报告
    logger.info("\n" + "=" * 60)
    logger.info("统计报告")
    logger.info("=" * 60)
    logger.info("铁衰老基因总数: 96")
    logger.info(f"已有传统特征基因数: {len(existing_genes_in_96)}")
    logger.info(f"本次成功补充基因数: {success_count}")
    logger.info(f"补充失败基因数: {len(failed_genes)}")
    logger.info(f"补充后总覆盖: {len(existing_genes_in_96) + success_count}")
    logger.info(f"仍缺失: {len(failed_genes)}")
    
    if failed_genes:
        logger.info("\n仍缺失的基因列表:")
        for g in failed_genes:
            logger.info(f"  - {g}")
    
    # 新增特征与已有特征的对比统计
    logger.info(f"\n新增特征列: {new_cols + aac_cols}")
    logger.info(f"补充DataFrame列数: {len(df_supplemented.columns)}")
    logger.info(f"补充DataFrame行数: {len(df_supplemented)}")
    
    # 打印数值统计
    for col in ["isoelectric_point", "aromaticity", "instability_index", "gravy", 
                "helix_fraction", "turn_fraction", "sheet_fraction", "flexibility_mean"]:
        if col in df_supplemented.columns:
            vals = df_supplemented[col].dropna()
            if len(vals) > 0:
                logger.info(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                            f"min={vals.min():.4f}, max={vals.max():.4f}")
    
    print(f"\n{'='*60}")
    print("补充完成!")
    print(f"成功: {success_count}/{len(missing_genes)} 基因")
    print(f"失败: {len(failed_genes)} 基因")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"{'='*60}")
    
    return df_supplemented, failed_genes


if __name__ == "__main__":
    main()