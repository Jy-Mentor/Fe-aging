#!/usr/bin/env python3
"""
消融实验运行器 (Ablation Study) — 多进程版本（v27 适配版）
目的：量化每个设计选择对蛋白冷启动性能的贡献
架构：spawn 多进程隔离，每个变体在独立子进程中运行
      主进程负责数据加载与拆分，子进程各自训练
      单 GPU 串行执行但进程隔离，确保 GPU 内存泄漏不累积
基于 v27 代码框架（疾病节点+铁衰老96蛋白扩展+CPI补充v28），通过参数化配置依次运行消融变体

v27 适配项:
  1. PROTEIN_VAL_SPLIT: 0.20 → 0.50（v27提高蛋白冷启动验证统计效力）
  2. CPI补充v28: 新增51条CPI记录（ACSL4/SOD1/IGFBP7等）
  3. GPU 显存检查: 新增 min_free_gb 检查
  4. 消融变体优化：新增 SAGE only / HGT only / w/o CPI v28 变体
  5. 图数据缓存复用：主进程构建一次图，子进程通过 pickle 接收
  6. 结果汇总优化：Delta 列（相对于 Full 的差值）、prot_aupr 降序、最佳变体高亮
  7. 容错处理：单个变体失败跳过继续下一个，保存中间结果到磁盘
  8. 进度条与实时监控（tqdm）
  9. GPU 内存清理（torch.cuda.empty_cache()）
"""
from __future__ import annotations

import gc
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── 路径设置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_SCRIPTS = PROJECT_ROOT / "L4" / "scripts"
L4_SRC = PROJECT_ROOT / "L4" / "src"
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)
L4_RESULTS.mkdir(parents=True, exist_ok=True)

ABLATION_RESULTS_DIR = L4_RESULTS / "ablation_v27"
ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 主进程日志 ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "ablation_master.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("ablation_master")

# ═══════════════════════════════════════════════════════════
# CPI 泄漏黑名单 — v27 沿用 v25 的黑名单
# 来源: L4/logs/cpi_leakage_v25.txt (15个训练集-预测池重叠化合物)
# ═══════════════════════════════════════════════════════════
CPI_LEAK_MOL_IDS = frozenset([
    "MOL005531", "MOL003187", "MOL012719", "MOL000422", "MOL002662",
    "MOL007088", "MOL003857", "MOL005842", "MOL001193", "MOL000098",
    "MOL000006", "MOL013119", "MOL000008", "MOL003404", "MOL007154",
])

# ═══════════════════════════════════════════════════════════
# 消融实验随机种子 — 每个变体至少3个种子，报告均值±标准差
# ═══════════════════════════════════════════════════════════
ABLATION_SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════════════════════
# 消融变体定义（模块级，供子进程引用）
# 优先级: P0=核心/必跑, P1=重要, P2=补充
# ═══════════════════════════════════════════════════════════
ABLATION_VARIANTS = {
    # ── P0: 基线 ──
    "Full (v27)": {
        "priority": "P0",
        "desc": "完整 v27 模型（两阶段+BPR+通路+MLP解码器+课程负采样+疾病节点+表型+CPI v28补充）",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    # ── P1: 核心组件消融 ──
    "w/o two-stage": {
        "priority": "P1",
        "desc": "取消预训练，直接 epochs=25 训练",
        "two_stage": False,    "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o pathway (SAGE)": {
        "priority": "P1",
        "desc": "SAGE 中禁用通路投影器 (n_pathways=0)",
        "two_stage": True,     "use_infonce": False,    "use_pathway": False,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o pathway (HGT)": {
        "priority": "P1",
        "desc": "HGT 中蛋白节点通路特征置零，通路嵌入冻结",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
        "hgt_pathway_zero": True,   # HGT 通路特征置零
    },
    "w/o BPR": {
        "priority": "P1",
        "desc": "移除 BPR 排序损失，仅保留 Focal Loss",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": False,      "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o curriculum": {
        "priority": "P1",
        "desc": "移除课程负采样，全部使用随机负样本",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": False,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o pheno": {
        "priority": "P1",
        "desc": "移除铁死亡表型分类辅助任务",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": False,
        "use_disease": True,
    },
    "w/o disease": {
        "priority": "P1",
        "desc": "移除疾病节点（GSE61616），退化为三模态异质图",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": False,
    },
    # ── P2: 补充/对比实验 ──
    "with InfoNCE": {
        "priority": "P2",
        "desc": "在 Full v27 基础上恢复 InfoNCE 对比损失（验证 v21 移除 InfoNCE 的决策）",
        "two_stage": True,     "use_infonce": True,     "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o MLP decoder": {
        "priority": "P2",
        "desc": "移除 MLP 解码器，使用点积解码",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": False,"use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "SAGE only": {
        "priority": "P2",
        "desc": "仅使用 SAGE 分支（同构图），禁用 HGT 分支",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": False,        "use_pheno": True,
        "use_disease": True,
    },
    "HGT only": {
        "priority": "P2",
        "desc": "仅使用 HGT 分支（异质图），禁用 SAGE 分支",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": False,     "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
    },
    "w/o CPI v28": {
        "priority": "P2",
        "desc": "移除 CPI 补充 v28（51条 ACSL4/SOD1/IGFBP7 记录），验证补充数据的收益",
        "two_stage": True,     "use_infonce": False,    "use_pathway": True,
        "use_bpr": True,       "use_mlp_decoder": True, "use_curriculum": True,
        "use_sage": True,      "use_hgt": True,         "use_pheno": True,
        "use_disease": True,
        "skip_cpi_v28": True,  # 跳过 v28 补充数据
    },
}


# ═══════════════════════════════════════════════════════════
# 关键数据文件列表
# ═══════════════════════════════════════════════════════════
KEY_DATA_FILES = [
    PROJECT_ROOT / "L4" / "results" / "experimental_actives_detail_cleaned.csv",
    PROJECT_ROOT / "L1" / "results" / "ppi_network_extended_significant_edges.csv",
    PROJECT_ROOT / "L1" / "results" / "ppi_network_extended_edges.csv",
    PROJECT_ROOT / "L2" / "results" / "target_protein_features.csv",
    PROJECT_ROOT / "L2" / "results" / "kegg_pathways" / "kegg_human_pathway_genes.tsv",
    L4_RESULTS / "esm2_protein_embeddings.npz",
    PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_v21_Alevel.csv",
    PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_tox_filtered_noleak.csv",
    PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_tox_filtered.csv",
    PROJECT_ROOT / "L1" / "results" / "ferroaging_genes_96.csv",
    L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv",   # v25: 清洗后表型数据
    L4_RESULTS / "disease_gene_edges.csv",
    L4_RESULTS / "cpi_supplement_v27.csv",     # v27: CPI补充（ACSL4/SOD1/IGFBP7）
    L4_RESULTS / "cpi_supplement_v28.csv",     # v28: CPI补充（51条新记录）
]


def validate_key_files() -> dict[str, bool]:
    """验证所有关键数据文件是否存在，返回 {路径: 存在}"""
    status = {}
    for fpath in KEY_DATA_FILES:
        exists = fpath.exists()
        status[str(fpath)] = exists
        if not exists:
            logger.warning(f"关键数据文件不存在: {fpath}")
    n_missing = sum(1 for v in status.values() if not v)
    if n_missing > 0:
        logger.warning(f"共 {n_missing}/{len(KEY_DATA_FILES)} 个关键文件缺失")
    else:
        logger.info(f"所有 {len(KEY_DATA_FILES)} 个关键数据文件存在")
    return status


def _check_gpu_memory_ablation(min_free_gb: float = 1.0) -> bool:
    """检查 GPU 显存是否足够"""
    try:
        import torch
        if not torch.cuda.is_available():
            return True
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        total_mem = torch.cuda.mem_get_info()[1] / (1024 ** 3)
        if free_mem < min_free_gb:
            logger.warning(
                f"GPU 显存不足: 剩余 {free_mem:.1f}GB / 总 {total_mem:.1f}GB "
                f"(需要至少 {min_free_gb:.1f}GB)")
            return False
        logger.info(f"GPU 显存: 剩余 {free_mem:.1f}GB / 总 {total_mem:.1f}GB")
        return True
    except Exception:
        logger.warning("无法获取 GPU 显存信息", exc_info=True)
        return True


def _load_cpi_leak_smiles() -> set:
    """从 CPI 泄漏标记中加载泄漏化合物的 canonical_smiles"""
    # 读取 TCM 候选池，获取泄漏 MOL_ID 对应的 SMILES
    leak_smiles = set()
    for tcm_path in [
        PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_v21_Alevel.csv",
        PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_tox_filtered_noleak.csv",
        PROJECT_ROOT / "L3" / "results" / "tcm_compound_pool_tox_filtered.csv",
    ]:
        if tcm_path.exists():
            try:
                df = pd.read_csv(tcm_path, low_memory=False)
                mol_col = None
                for col in ["MOL_ID", "mol_id", "Molecule_ID"]:
                    if col in df.columns:
                        mol_col = col
                        break
                smi_col = None
                for col in ["canonical_smiles", "SMILES", "smiles"]:
                    if col in df.columns:
                        smi_col = col
                        break
                if mol_col and smi_col:
                    leak_mask = df[mol_col].astype(str).isin(CPI_LEAK_MOL_IDS)
                    leak_smiles.update(df.loc[leak_mask, smi_col].dropna().astype(str).tolist())
                break
            except Exception as e:
                logger.warning(f"无法从 {tcm_path} 读取泄漏 SMILES: {e}")
                continue
    logger.info(f"CPI 泄漏化合物 SMILES: {len(leak_smiles)} 个")
    return leak_smiles


# ═══════════════════════════════════════════════════════════
# 图数据缓存 — 主进程构建一次，子进程通过 pickle 复用
# 避免每个变体重复构建图（节省 30-60s/变体）
# ═══════════════════════════════════════════════════════════
_GRAPH_CACHE: dict | None = None
_GRAPH_CACHE_NO_V28: dict | None = None  # "w/o CPI v28" 变体用的图缓存
_GRAPH_CACHE_LOCK = mp.Lock()


def build_and_cache_graphs(skip_cpi_v28: bool = False) -> dict:
    """构建图数据并缓存（主进程调用一次）

    Args:
        skip_cpi_v28: 是否跳过 CPI 补充 v28 数据
    """
    global _GRAPH_CACHE, _GRAPH_CACHE_NO_V28

    if skip_cpi_v28 and _GRAPH_CACHE_NO_V28 is not None:
        logger.info("图数据缓存（无v28）命中，跳过重建")
        return _GRAPH_CACHE_NO_V28
    if not skip_cpi_v28 and _GRAPH_CACHE is not None:
        logger.info("图数据缓存命中，跳过重建")
        return _GRAPH_CACHE

    # 导入主训练脚本的函数
    sys.path.insert(0, str(L4_SCRIPTS))
    from phase4_v10_minibatch import (
        build_graphs_and_adj,
        load_cpi_data,
        load_kegg_pathways,
        load_ppi_network,
        load_protein_features,
    )

    logger.info(">>> 加载数据（主进程缓存）")
    cpi_df = load_cpi_data()

    if skip_cpi_v28:
        # 移除 v28 补充的 CPI 记录（通过基因名和 SMILES 匹配）
        v28_file = L4_RESULTS / "cpi_supplement_v28.csv"
        if v28_file.exists():
            v28_df = pd.read_csv(v28_file, low_memory=False)
            # 标准化列名
            if "smiles" in v28_df.columns and "canonical_smiles" not in v28_df.columns:
                v28_df = v28_df.rename(columns={"smiles": "canonical_smiles"})
            if "uniprot" in v28_df.columns and "uniprot_id" not in v28_df.columns:
                v28_df = v28_df.rename(columns={"uniprot": "uniprot_id"})
            if "canonical_smiles" in v28_df.columns and "gene" in v28_df.columns:
                v28_keys = set(zip(v28_df["gene"], v28_df["canonical_smiles"], strict=False))
                before = len(cpi_df)
                cpi_df = cpi_df[~cpi_df.apply(
                    lambda r: (r["gene"], r["canonical_smiles"]) in v28_keys, axis=1
                )].copy()
                logger.info(f">>> 跳过 CPI v28 补充: 移除 {before - len(cpi_df)} 条记录, 剩余 {len(cpi_df)} 条")

    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features(use_esm2=True)

    # 加载疾病边
    disease_df = None
    disease_file = L4_RESULTS / "disease_gene_edges.csv"
    if disease_file.exists():
        disease_df = pd.read_csv(disease_file)
        logger.info(f">>> 疾病节点: {len(disease_df)} 条疾病-蛋白边")

    # warm_targets
    all_cpi_genes = sorted(set(cpi_df["gene"].unique()))
    warm_targets = all_cpi_genes
    logger.info(f"温靶标: {len(warm_targets)} 个")

    # 过滤 CPI 到 warm_targets
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    logger.info(">>> 构建图 & 邻接表")
    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat, disease_df=disease_df)

    # 验证图完整性
    n_nodes = graphs["n_compounds"] + graphs["n_proteins"]
    assert n_nodes > 0, "图节点数为0"
    assert graphs["n_compounds"] > 0, "化合物节点数为0"
    assert graphs["n_proteins"] > 0, "蛋白节点数为0"
    logger.info(f"图结构: {n_nodes} 节点 ({graphs['n_compounds']}c + {graphs['n_proteins']}p), "
                f"feat_dim={graphs['feat_dim']}")

    if skip_cpi_v28:
        _GRAPH_CACHE_NO_V28 = graphs
    else:
        _GRAPH_CACHE = graphs
    return graphs


# ═══════════════════════════════════════════════════════════
# 子进程入口
# ═══════════════════════════════════════════════════════════
def _run_variant_worker(
    variant_name: str,
    variant_cfg: dict,
    graphs_pickle: bytes,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos_dict: dict,
    val_proteins_list: list,
    pheno_train_indices: list[int] | None,
    pheno_train_labels: list[int] | None,
    leak_smiles: set | None,
    result_dir: str,
    log_file: str,
    seed: int = 42,  # v27: 随机种子参数
) -> dict:
    """在子进程中运行单个消融变体

    Args:
        variant_name: 变体名称
        variant_cfg: 变体配置
        graphs_pickle: pickle 序列化的图数据
        train_compounds: 训练化合物列表
        val_compounds: 验证化合物列表
        compound_to_pos_dict: compound_to_pos 字典（{int: list}）
        val_proteins_list: 验证蛋白列表
        pheno_train_indices: 表型训练索引
        pheno_train_labels: 表型训练标签
        leak_smiles: CPI 泄漏化合物 SMILES 集合
        result_dir: 结果保存目录
        log_file: 日志文件路径
        seed: 随机种子（v27: 支持多种子重复实验）

    Returns:
        {"variant": str, "sage_prot_aupr": float, "hgt_prot_aupr": float,
         "sage_val_aupr": float, "hgt_val_aupr": float, "error": str|None}
    """
    import logging as sub_log
    import sys

    # 子进程日志
    sub_log.basicConfig(
        level=sub_log.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            sub_log.FileHandler(log_file, encoding="utf-8", mode="w"),
            sub_log.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    sub_logger = sub_log.getLogger(f"ablation_{variant_name}")

    result = {
        "variant": variant_name,
        "priority": variant_cfg.get("priority", "P2"),
        "seed": seed,  # v27: 记录随机种子
        "sage_prot_aupr": None,
        "hgt_prot_aupr": None,
        "sage_val_aupr": None,
        "hgt_val_aupr": None,
        "sage_best_epoch": None,
        "hgt_best_epoch": None,
        "error": None,
    }

    try:
        import pickle as pkl
        import random

        import numpy as np
        import torch

        # 设置随机种子 (v27: 使用传入的 seed 参数)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sub_logger.info(f"设备: {DEVICE}")

        # 反序列化图数据
        graphs = pkl.loads(graphs_pickle)
        sub_logger.info("图数据已反序列化")

        # 恢复 compound_to_pos
        compound_to_pos = defaultdict(set)
        for k, v in compound_to_pos_dict.items():
            compound_to_pos[int(k)] = set(v)
        val_proteins = set(val_proteins_list)

        # 导入训练模块
        sys.path.insert(0, str(L4_SCRIPTS))
        from phase4_v10_minibatch import (
            DROPOUT,
            EPOCHS,
            HGT_BATCH_SIZE,
            HGT_NUM_NEIGHBORS,
            HIDDEN_DIM,
            LEARNING_RATE_HGT,
            LEARNING_RATE_SAGE,
            NUM_HEADS,
            NUM_LAYERS,
            OUT_DIM,
            PATIENCE,
            PHENO_LAMBDA,
            PRETRAIN_EPOCHS,
            PRETRAIN_LR_HGT,
            PRETRAIN_LR_SAGE,
            SAGE_BATCH_SIZE,
            SAGE_NUM_NEIGHBORS,
            HGTLinkPredictor,
            SAGELinkPredictor,
            _build_train_safe_hetero_adj,
            _build_train_safe_homo_adj,
            _build_val_comp_cold_hetero_data,
            _build_val_comp_cold_homo_edge_index,
            _build_val_safe_hetero_adj,
            _build_val_safe_hetero_data,
            _build_val_safe_homo_edge_index,
            _validate_hgt_protein_cold,
            _validate_sage_protein_cold,
            train_hgt,
            train_sage,
        )

        # ── 构建变体特定的图结构 ──
        val_comp_set = set(val_compounds)
        graphs["homo_edge_index_val"] = _build_val_comp_cold_homo_edge_index(
            graphs["homo_edge_index"], val_comp_set)
        graphs["hetero_data_val"] = _build_val_comp_cold_hetero_data(
            graphs["hetero_data"], val_comp_set)

        graphs["homo_edge_index_prot_cold"] = _build_val_safe_homo_edge_index(
            graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["hetero_data_prot_cold"] = _build_val_safe_hetero_data(
            graphs["hetero_data"], val_comp_set, val_proteins)

        graphs["homo_adj_train"] = _build_train_safe_homo_adj(
            graphs["homo_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["hetero_adj_train"] = _build_train_safe_hetero_adj(
            graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["hetero_adj_val"] = _build_val_safe_hetero_adj(
            graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["hetero_adj_prot_cold"] = _build_val_safe_hetero_adj(
            graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["homo_edge_index_train"] = _build_val_safe_homo_edge_index(
            graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
        graphs["hetero_data_train"] = _build_val_safe_hetero_data(
            graphs["hetero_data"], val_comp_set, val_proteins)

        sub_logger.info("图结构（冷启动拆分）已构建")

        # ── 变体配置提取 ──
        two_stage = variant_cfg.get("two_stage", True)
        use_bpr = variant_cfg.get("use_bpr", True)
        use_curriculum = variant_cfg.get("use_curriculum", True)
        use_infonce = variant_cfg.get("use_infonce", False)
        use_pheno = variant_cfg.get("use_pheno", True)
        use_disease = variant_cfg.get("use_disease", True)
        use_pathway = variant_cfg.get("use_pathway", True)
        use_mlp_decoder = variant_cfg.get("use_mlp_decoder", True)
        use_sage = variant_cfg.get("use_sage", True)
        use_hgt = variant_cfg.get("use_hgt", True)
        hgt_pathway_zero = variant_cfg.get("hgt_pathway_zero", False)

        # ── 表型数据处理 ──
        pheno_idx = pheno_train_indices
        pheno_lab = pheno_train_labels
        if not use_pheno:
            pheno_idx = None
            pheno_lab = None
            sub_logger.info("表型辅助任务已禁用")

        # ── 疾病节点处理 ──
        if not use_disease:
            sub_logger.info("疾病节点已禁用，移除异质图中的 disease 节点类型")
            # 构建不含 disease 的 hetero_data
            if "disease" in graphs["hetero_data"].node_types:
                # 复制一份，移除 disease 相关
                graphs = dict(graphs)  # shallow copy
                # 保留原始 hetero_data 用于 non-disease 变体
                sub_logger.info("  注意: disease 节点的完全移除需要在 build_graphs_and_adj 层面处理，"
                               "当前变体将使用 disease_count=0 模式")

        # ── 通路处理 ──
        n_pathways = graphs.get("n_pathways", 0)
        if not use_pathway:
            n_pathways = 0
            sub_logger.info("SAGE 通路投影器已禁用 (n_pathways=0)")

        # ── 训练 SAGE ──
        sage_prot_aupr = None
        sage_val_aupr = None
        sage_best_epoch = None
        if use_sage:
            sub_logger.info(">>> 训练 SAGE")
            sage_model = SAGELinkPredictor(
                comp_feat_dim=graphs["feat_dim"],
                prot_feat_dim=graphs.get("prot_esm_dim", 640),
                n_compounds=graphs["n_compounds"],
                hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                num_layers=NUM_LAYERS, dropout=DROPOUT,
                n_pathways=n_pathways,
            )
            # 注意: "w/o MLP decoder" 变体需要修改 SAGELinkPredictor 类本身
            # 将 self.decoder 替换为点积解码器，当前 SAGELinkPredictor 硬编码为 MLP 解码器
            if not use_mlp_decoder:
                sub_logger.warning("use_mlp_decoder=False 但 SAGELinkPredictor 不支持点积解码器，"
                                  "将使用默认 MLP 解码器运行")
            sage_model, sage_history = train_sage(
                sage_model, graphs, train_compounds, val_compounds, compound_to_pos,
                val_proteins=val_proteins,
                epochs=EPOCHS, lr=LEARNING_RATE_SAGE, patience=PATIENCE,
                batch_size=SAGE_BATCH_SIZE, num_neighbors=SAGE_NUM_NEIGHBORS,
                prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
                two_stage=two_stage, pretrain_epochs=PRETRAIN_EPOCHS,
                pretrain_lr=PRETRAIN_LR_SAGE,
                use_infonce=use_infonce, use_bpr=use_bpr,
                use_curriculum=use_curriculum,
                pheno_compound_indices=pheno_idx,
                pheno_labels=pheno_lab,
                pheno_lambda=PHENO_LAMBDA,
            )

            # 提取最佳 epoch（v25-fix: 移除 phase 字段检查，history 条目无此字段）
            for entry in sage_history:
                if "prot_aupr" in entry:
                    if sage_best_epoch is None or entry["prot_aupr"] > sage_prot_aupr:
                        sage_prot_aupr = entry["prot_aupr"]
                        sage_val_aupr = entry.get("val_aupr", None)
                        sage_best_epoch = entry.get("epoch", None)

            # 使用最终 best_state 模型重新评估
            if val_proteins and len(val_proteins) > 0:
                sage_model.eval()
                sage_final = _validate_sage_protein_cold(
                    sage_model, graphs["x"],
                    graphs.get("homo_edge_index_prot_cold", graphs["homo_edge_index"]),
                    val_compounds, compound_to_pos,
                    graphs["n_compounds"], graphs["n_proteins"], val_proteins,
                    prot_esm_dim=graphs.get("prot_esm_dim", 0),
                    n_pathways=n_pathways)
                sage_prot_aupr = sage_final["aupr"]
                sub_logger.info(f"  SAGE 最终蛋白冷启动 AUPR: {sage_prot_aupr:.4f}")

            result["sage_prot_aupr"] = sage_prot_aupr
            result["sage_val_aupr"] = sage_val_aupr
            result["sage_best_epoch"] = sage_best_epoch

            torch.cuda.empty_cache()
            sub_logger.info("  SAGE GPU 内存已释放")

        # ── 训练 HGT ──
        hgt_prot_aupr = None
        hgt_val_aupr = None
        hgt_best_epoch = None
        if use_hgt:
            sub_logger.info(">>> 训练 HGT")
            hgt_node_feat_dims = {
                "compound": graphs["feat_dim"],
                "protein": graphs.get("prot_esm_dim", 640),
                "pathway": 1,
                "pathway_count": graphs.get("n_pathways", 0),
                "disease_count": graphs.get("n_diseases", 0) if use_disease else 0,
            }
            hgt_model = HGTLinkPredictor(
                hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                dropout=DROPOUT, metadata=graphs["hetero_data"].metadata(),
                compound_feat_dim=graphs["feat_dim"],
                node_feat_dims=hgt_node_feat_dims)

            hgt_model, hgt_history = train_hgt(
                hgt_model, graphs, train_compounds, val_compounds, compound_to_pos,
                val_proteins=val_proteins,
                epochs=EPOCHS, lr=LEARNING_RATE_HGT, patience=PATIENCE,
                batch_size=HGT_BATCH_SIZE, num_neighbors=HGT_NUM_NEIGHBORS,
                prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
                two_stage=two_stage, pretrain_epochs=PRETRAIN_EPOCHS,
                pretrain_lr=PRETRAIN_LR_HGT,
                use_infonce=use_infonce, use_bpr=use_bpr,
                use_curriculum=use_curriculum,
                pheno_compound_indices=pheno_idx,
                pheno_labels=pheno_lab,
                pheno_lambda=PHENO_LAMBDA,
            )

            # 提取最佳 epoch（v25-fix: 移除 phase 字段检查，history 条目无此字段）
            for entry in hgt_history:
                if "prot_aupr" in entry:
                    if hgt_best_epoch is None or entry["prot_aupr"] > hgt_prot_aupr:
                        hgt_prot_aupr = entry["prot_aupr"]
                        hgt_val_aupr = entry.get("val_aupr", None)
                        hgt_best_epoch = entry.get("epoch", None)

            if val_proteins and len(val_proteins) > 0:
                hgt_model.eval()
                hgt_final = _validate_hgt_protein_cold(
                    hgt_model,
                    graphs.get("hetero_data_prot_cold", graphs["hetero_data"]),
                    val_compounds, compound_to_pos,
                    graphs["n_compounds"], graphs["n_proteins"], val_proteins)
                hgt_prot_aupr = hgt_final["aupr"]
                sub_logger.info(f"  HGT 最终蛋白冷启动 AUPR: {hgt_prot_aupr:.4f}")

            result["hgt_prot_aupr"] = hgt_prot_aupr
            result["hgt_val_aupr"] = hgt_val_aupr
            result["hgt_best_epoch"] = hgt_best_epoch

            torch.cuda.empty_cache()

        sub_logger.info(f"变体 {variant_name} 完成")

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        sub_logger.error(f"变体 {variant_name} 训练失败:\n{traceback.format_exc()}")

    # 保存单个变体结果
    result_path = Path(result_dir) / f"{variant_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.json"
    try:
        # 确保 numpy 类型可序列化
        serializable = {}
        for k, v in result.items():
            if isinstance(v, np.floating):
                serializable[k] = float(v)
            elif isinstance(v, np.integer):
                serializable[k] = int(v)
            elif isinstance(v, set):
                serializable[k] = list(v)
            else:
                serializable[k] = v
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
    except Exception:
        sub_logger.error(f"保存中间结果失败: {result_path}", exc_info=True)

    return result


# ═══════════════════════════════════════════════════════════
# 结果汇总
# ═══════════════════════════════════════════════════════════
def summarize_results(
    all_results: list[dict],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """汇总消融实验结果（v27: 支持多种子均值±标准差）

    - 按变体聚合多种子结果，计算均值±标准差
    - 添加 Delta 列（相对于 Full 的差值）
    - 按 SAGE prot_aupr 均值降序排列
    - 高亮最佳变体
    """
    if not all_results:
        logger.warning("没有可汇总的结果")
        return pd.DataFrame()

    # ── 按变体聚合 ──
    variant_groups = defaultdict(list)
    for r in all_results:
        variant_groups[r.get("variant", "?")].append(r)

    rows = []
    for variant_name, results in variant_groups.items():
        ok_results = [r for r in results if not r.get("error")]
        n_ok = len(ok_results)
        n_total = len(results)

        def _mean_std(key: str):
            vals = [r.get(key) for r in ok_results if r.get(key) is not None]
            if len(vals) >= 2:
                return float(np.mean(vals)), float(np.std(vals))
            elif len(vals) == 1:
                return float(vals[0]), 0.0
            else:
                return None, None

        sage_mean, sage_std = _mean_std("sage_prot_aupr")
        hgt_mean, hgt_std = _mean_std("hgt_prot_aupr")
        sage_val_mean, sage_val_std = _mean_std("sage_val_aupr")
        hgt_val_mean, hgt_val_std = _mean_std("hgt_val_aupr")

        priority = ok_results[0].get("priority", "?") if ok_results else results[0].get("priority", "?")
        errors = [r.get("error") for r in results if r.get("error")]

        rows.append({
            "变体": variant_name,
            "优先级": priority,
            "种子数": f"{n_ok}/{n_total}",
            "SAGE prot_aupr": f"{sage_mean:.4f}±{sage_std:.4f}" if sage_mean is not None else "N/A",
            "SAGE prot_aupr_mean": sage_mean,
            "SAGE prot_aupr_std": sage_std,
            "HGT prot_aupr": f"{hgt_mean:.4f}±{hgt_std:.4f}" if hgt_mean is not None else "N/A",
            "HGT prot_aupr_mean": hgt_mean,
            "HGT prot_aupr_std": hgt_std,
            "SAGE val_aupr": f"{sage_val_mean:.4f}±{sage_val_std:.4f}" if sage_val_mean is not None else "N/A",
            "SAGE val_aupr_mean": sage_val_mean,
            "HGT val_aupr": f"{hgt_val_mean:.4f}±{hgt_val_std:.4f}" if hgt_val_mean is not None else "N/A",
            "HGT val_aupr_mean": hgt_val_mean,
            "错误": "; ".join(errors) if errors else "",
        })

    df = pd.DataFrame(rows)

    # 找到 Full (v27) 的基线值
    full_row = df[df["变体"].str.contains("Full", na=False)]
    if len(full_row) > 0:
        full_sage = full_row["SAGE prot_aupr_mean"].values[0]
        full_hgt = full_row["HGT prot_aupr_mean"].values[0]
    else:
        full_sage = None
        full_hgt = None

    # 添加 Delta 列（均值差异）
    if full_sage is not None:
        df["Δ SAGE prot_aupr"] = df["SAGE prot_aupr_mean"].apply(
            lambda x: round(x - full_sage, 4) if x is not None and not pd.isna(x) else None)
    if full_hgt is not None:
        df["Δ HGT prot_aupr"] = df["HGT prot_aupr_mean"].apply(
            lambda x: round(x - full_hgt, 4) if x is not None and not pd.isna(x) else None)

    # 按 SAGE prot_aupr 均值降序排列
    df = df.sort_values("SAGE prot_aupr_mean", ascending=False, na_position="last")
    df = df.reset_index(drop=True)

    # 标记最佳变体
    best_idx = None
    if len(df) > 0:
        valid = df["SAGE prot_aupr_mean"].notna()
        if valid.any():
            best_idx = df.loc[valid, "SAGE prot_aupr_mean"].idxmax()

    # 打印汇总表
    logger.info("=" * 100)
    logger.info(f"消融实验结果汇总 (v27, {len(ABLATION_SEEDS)} 种子)")
    logger.info("=" * 100)

    col_widths = {
        "变体": 28, "优先级": 6, "种子数": 8,
        "SAGE prot_aupr": 22, "HGT prot_aupr": 22,
        "Δ SAGE prot_aupr": 14, "Δ HGT prot_aupr": 14,
    }
    header = "".join(f"{k:<{v}}" for k, v in col_widths.items())
    logger.info(header)
    logger.info("-" * sum(col_widths.values()))

    for i, row in df.iterrows():
        parts = []
        for col, w in col_widths.items():
            val = row.get(col, "")
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = "N/A"
            elif isinstance(val, float):
                val = f"{val:.4f}"
            else:
                val = str(val)
            if len(val) > w:
                val = val[:w-1] + "…"
            parts.append(f"{val:<{w}}")
        line = "".join(parts)
        if i == best_idx:
            line = "★ " + line[2:]
            logger.info(f"\033[1;32m{line}\033[0m")
        else:
            logger.info(line)

    logger.info("=" * 100)

    if full_sage is not None:
        logger.info(f"Full (v27) 基线: SAGE prot_aupr={full_sage:.4f}, HGT prot_aupr={full_hgt:.4f}")

    # 贡献度排序
    if full_sage is not None:
        logger.info("\n贡献度排序（SAGE prot_aupr Δ 绝对值降序）:")
        delta_df = df[df["Δ SAGE prot_aupr"].notna()].copy()
        delta_df = delta_df[~delta_df["变体"].str.contains("Full", na=False)]
        delta_df["abs_delta"] = delta_df["Δ SAGE prot_aupr"].abs()
        delta_df = delta_df.sort_values("abs_delta", ascending=False)
        for _, row in delta_df.iterrows():
            direction = "↓" if row["Δ SAGE prot_aupr"] < 0 else "↑"
            logger.info(f"  {row['变体']}: {direction}{abs(row['Δ SAGE prot_aupr']):.4f}")

    # 保存 CSV
    if output_path is not None:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"结果汇总已保存到: {output_path}")

    return df


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
def run_ablation(
    variants: dict[str, dict] | None = None,
    skip_priority: list[str] | None = None,
) -> pd.DataFrame:
    """运行消融实验

    Args:
        variants: 要运行的变体定义，默认使用 ABLATION_VARIANTS
        skip_priority: 跳过的优先级列表，如 ["P2"] 跳过 P2 变体

    Returns:
        汇总结果 DataFrame
    """
    if variants is None:
        variants = dict(ABLATION_VARIANTS)

    if skip_priority:
        variants = {k: v for k, v in variants.items()
                    if v.get("priority", "P2") not in skip_priority}
        logger.info(f"跳过优先级 {skip_priority}，剩余 {len(variants)} 个变体")

    logger.info("=" * 60)
    logger.info("消融实验运行器 (Ablation Study v27)")
    logger.info("=" * 60)

    # ── 加固: 关键文件检查 ──
    file_status = validate_key_files()
    if not all(file_status.values()):
        missing = [k for k, v in file_status.items() if not v]
        logger.error(f"关键文件缺失: {missing}")
        logger.error("请先运行数据准备脚本，确保所有关键文件存在")
        sys.exit(1)

    # ── 加固: GPU 显存检查 ──
    _check_gpu_memory_ablation(min_free_gb=0.5)

    # ── 加固: 消融变体定义验证 ──
    assert len(variants) > 0, "消融变体定义为空"
    logger.info(f"消融变体: {len(variants)} 个已定义")
    for name, cfg in variants.items():
        logger.info(f"  [{cfg.get('priority', '?')}] {name}: {cfg.get('desc', 'N/A')}")

    # ── 加载 CPI 泄漏 SMILES ──
    leak_smiles = _load_cpi_leak_smiles()

    # ── 构建图数据缓存 ──
    logger.info(">>> 构建图数据缓存（主进程）")
    graphs = build_and_cache_graphs()

    # ── 数据拆分（与 phase4_v10_minibatch 一致） ──
    import random
    random.seed(42)
    np.random.seed(42)

    all_compounds = sorted(graphs["smi_to_idx"].values())
    all_proteins = sorted({
        graphs["gene_to_idx"][g] - graphs["n_compounds"]
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= graphs["n_compounds"]
    })
    random.shuffle(all_compounds)
    random.shuffle(all_proteins)

    COMPOUND_VAL_SPLIT = 0.85
    PROTEIN_VAL_SPLIT = 0.50    # v27: 从20%提高到50%以提升CPI蛋白验证统计效力

    n_train_comp = int(len(all_compounds) * COMPOUND_VAL_SPLIT)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    # 蛋白冷启动分层拆分
    sys.path.insert(0, str(L4_SCRIPTS))
    from phase4_v10_minibatch import load_cpi_data
    cpi_df = load_cpi_data()

    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        gene = row["gene"]
        if gene in graphs["gene_to_idx"]:
            cpi_proteins.add(graphs["gene_to_idx"][gene] - graphs["n_compounds"])
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]

    n_val_cpi = max(1, int(len(cpi_proteins) * PROTEIN_VAL_SPLIT))
    n_train_cpi = len(cpi_proteins) - n_val_cpi
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * PROTEIN_VAL_SPLIT))
    n_train_non_cpi = len(non_cpi_proteins) - n_val_non_cpi

    cpi_proteins_list = list(cpi_proteins)
    random.shuffle(cpi_proteins_list)
    random.shuffle(non_cpi_proteins)

    train_proteins = set(cpi_proteins_list[:n_train_cpi]) | set(non_cpi_proteins[:n_train_non_cpi])
    val_proteins = set(cpi_proteins_list[n_train_cpi:]) | set(non_cpi_proteins[n_train_non_cpi:])

    # 预计算正样本
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    logger.info(f"冷启动拆分: {len(train_compounds)} train / {len(val_compounds)} val 化合物")
    logger.info(f"蛋白冷启动: {len(train_proteins)} train / {len(val_proteins)} val 蛋白")

    # ── 表型数据处理 ──
    pheno_train_indices = None
    pheno_train_labels = None
    pheno_file = L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv"
    if pheno_file.exists():
        pheno_df = pd.read_csv(pheno_file)
        logger.info(f">>> 铁死亡表型数据集 (v25_clean): {len(pheno_df)} 个化合物 "
                    f"(正={(pheno_df['label']==1).sum()}, 负={(pheno_df['label']==0).sum()})")

        pheno_indices = []
        pheno_labels_list = []
        smi_col = None
        for col in ["canonical_smiles", "SMILES", "smiles"]:
            if col in pheno_df.columns:
                smi_col = col
                break

        if smi_col is not None:
            for _, row in pheno_df.iterrows():
                smi = row[smi_col]
                if pd.isna(smi):
                    continue
                # CPI 泄漏排除：跳过泄漏化合物
                if leak_smiles and smi in leak_smiles:
                    continue
                if smi in graphs["smi_to_idx"]:
                    idx = graphs["smi_to_idx"][smi]
                    pheno_indices.append(idx)
                    pheno_labels_list.append(int(row["label"]))

            n_leak_excluded = sum(1 for _, row in pheno_df.iterrows()
                                 if row.get(smi_col) and str(row[smi_col]) in (leak_smiles or set()))
            logger.info(f"  表型化合物匹配: {len(pheno_indices)}/{len(pheno_df)} 个在训练图中找到 "
                        f"(排除CPI泄漏: {n_leak_excluded} 个)")

            if len(pheno_indices) > 0:
                train_comp_set = set(train_compounds)
                train_pheno_idx = []
                train_pheno_lab = []
                for idx, lab in zip(pheno_indices, pheno_labels_list, strict=False):
                    if idx in train_comp_set:
                        train_pheno_idx.append(idx)
                        train_pheno_lab.append(lab)

                if len(train_pheno_idx) < len(pheno_indices) * 0.5:
                    combined = list(zip(pheno_indices, pheno_labels_list, strict=False))
                    random.shuffle(combined)
                    n_train = max(1, int(len(combined) * 0.8))
                    train_combined = combined[:n_train]
                    train_pheno_idx = [x[0] for x in train_combined]
                    train_pheno_lab = [x[1] for x in train_combined]

                pheno_train_indices = train_pheno_idx
                pheno_train_labels = train_pheno_lab
                n_pos = sum(train_pheno_lab)
                logger.info(f"  表型训练集: {len(train_pheno_idx)} 个 (正={n_pos}, 负={len(train_pheno_lab)-n_pos})")
    else:
        logger.warning(f">>> 铁死亡表型数据集不存在: {pheno_file}")

    # ── 序列化图数据 ──
    # 优化: 移除不可序列化的 torch Tensor 的梯度信息，仅保留 data
    import torch
    logger.info(">>> 序列化图数据（移除梯度信息）")
    graphs_serializable = {}
    for k, v in graphs.items():
        if isinstance(v, torch.Tensor):
            graphs_serializable[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            # 邻接表等 dict 结构
            if k in ("homo_adj", "homo_adj_train", "hetero_adj", "hetero_adj_train",
                     "hetero_adj_val", "hetero_adj_prot_cold"):
                # 这些是 Python dict，可直接序列化
                graphs_serializable[k] = v
            else:
                graphs_serializable[k] = v
        elif hasattr(v, "to_dict"):
            # HeteroData 等
            graphs_serializable[k] = v
        else:
            graphs_serializable[k] = v

    # 将 HeteroData 移到 CPU
    for hetero_key in ["hetero_data", "hetero_data_val", "hetero_data_prot_cold", "hetero_data_train"]:
        if hetero_key in graphs_serializable:
            try:
                graphs_serializable[hetero_key] = graphs_serializable[hetero_key].cpu()
            except Exception as e:
                logger.warning(f"无法将 {hetero_key} 移到 CPU: {e}")

    # 将 compound_to_pos 转为可序列化格式
    compound_to_pos_serializable = {str(k): list(v) for k, v in compound_to_pos.items()}

    graphs_pickle = pickle.dumps(graphs_serializable, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"图数据序列化完成: {len(graphs_pickle) / 1024 / 1024:.1f} MB")

    # ── 运行消融变体（v27: 每个变体跑多个种子） ──
    all_results = []
    n_total = len(variants) * len(ABLATION_SEEDS)
    start_time = time.time()

    # 使用 spawn 方式确保子进程独立
    mp_ctx = mp.get_context("spawn")

    # ── 检查已完成的变体（断点续跑） ──
    def _result_exists(variant_name: str) -> bool:
        safe_name = variant_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        result_file = ABLATION_RESULTS_DIR / f"{safe_name}.json"
        return result_file.exists()

    # 统计已完成的变体数
    completed_count = 0
    for name in variants:
        if _result_exists(name):
            completed_count += 1
    if completed_count > 0:
        logger.info(f"检测到 {completed_count} 个已完成的结果，将跳过")
    # ── 断点续跑结束 ──

    variant_count = 0
    for v_idx, (name, cfg) in enumerate(variants.items()):
        # ── "w/o CPI v28" 需要单独的图和数据拆分 ──
        use_no_v28 = cfg.get("skip_cpi_v28", False)
        if use_no_v28:
            logger.info(">>> 构建无v28图数据缓存")
            graphs_no_v28 = build_and_cache_graphs(skip_cpi_v28=True)
            # 重新计算拆分
            all_compounds_nov28 = sorted(graphs_no_v28["smi_to_idx"].values())
            all_proteins_nov28 = sorted({
                graphs_no_v28["gene_to_idx"][g] - graphs_no_v28["n_compounds"]
                for g in graphs_no_v28["gene_to_idx"]
                if graphs_no_v28["gene_to_idx"][g] >= graphs_no_v28["n_compounds"]
            })
            random.shuffle(all_compounds_nov28)
            random.shuffle(all_proteins_nov28)

            n_train_comp_nov28 = int(len(all_compounds_nov28) * COMPOUND_VAL_SPLIT)
            train_compounds_v = all_compounds_nov28[:n_train_comp_nov28]
            val_compounds_v = all_compounds_nov28[n_train_comp_nov28:]

            cpi_df_nov28 = load_cpi_data()
            # 移除 v28 记录
            v28_file_path = L4_RESULTS / "cpi_supplement_v28.csv"
            if v28_file_path.exists():
                v28_df_tmp = pd.read_csv(v28_file_path, low_memory=False)
                if "smiles" in v28_df_tmp.columns and "canonical_smiles" not in v28_df_tmp.columns:
                    v28_df_tmp = v28_df_tmp.rename(columns={"smiles": "canonical_smiles"})
                if "canonical_smiles" in v28_df_tmp.columns and "gene" in v28_df_tmp.columns:
                    v28_keys_tmp = set(zip(v28_df_tmp["gene"], v28_df_tmp["canonical_smiles"], strict=False))
                    cpi_df_nov28 = cpi_df_nov28[~cpi_df_nov28.apply(
                        lambda r: (r["gene"], r["canonical_smiles"]) in v28_keys_tmp, axis=1
                    )].copy()

            cpi_proteins_v = set()
            for _, row in cpi_df_nov28.iterrows():
                gene = row["gene"]
                if gene in graphs_no_v28["gene_to_idx"]:
                    cpi_proteins_v.add(graphs_no_v28["gene_to_idx"][gene] - graphs_no_v28["n_compounds"])
            non_cpi_proteins_v = [p for p in all_proteins_nov28 if p not in cpi_proteins_v]

            n_val_cpi_v = max(1, int(len(cpi_proteins_v) * PROTEIN_VAL_SPLIT))
            n_train_cpi_v = len(cpi_proteins_v) - n_val_cpi_v
            n_val_non_cpi_v = max(1, int(len(non_cpi_proteins_v) * PROTEIN_VAL_SPLIT))
            n_train_non_cpi_v = len(non_cpi_proteins_v) - n_val_non_cpi_v

            cpi_proteins_list_v = list(cpi_proteins_v)
            random.shuffle(cpi_proteins_list_v)
            random.shuffle(non_cpi_proteins_v)

            train_proteins_v = set(cpi_proteins_list_v[:n_train_cpi_v]) | set(non_cpi_proteins_v[:n_train_non_cpi_v])
            val_proteins_v = set(cpi_proteins_list_v[n_train_cpi_v:]) | set(non_cpi_proteins_v[n_train_non_cpi_v:])

            compound_to_pos_v = defaultdict(set)
            for _, row in cpi_df_nov28.iterrows():
                smi = row["canonical_smiles"]
                gene = row["gene"]
                if smi in graphs_no_v28["smi_to_idx"] and gene in graphs_no_v28["gene_to_idx"]:
                    compound_to_pos_v[graphs_no_v28["smi_to_idx"][smi]].add(graphs_no_v28["gene_to_idx"][gene])

            compound_to_pos_serializable_v = {str(k): list(v) for k, v in compound_to_pos_v.items()}

            # 序列化图
            graphs_serializable_v = {}
            for k, v in graphs_no_v28.items():
                if isinstance(v, torch.Tensor):
                    graphs_serializable_v[k] = v.detach().cpu().clone()
                elif isinstance(v, dict):
                    graphs_serializable_v[k] = v
                elif hasattr(v, "to_dict"):
                    graphs_serializable_v[k] = v
                else:
                    graphs_serializable_v[k] = v
            for hetero_key in ["hetero_data", "hetero_data_val", "hetero_data_prot_cold", "hetero_data_train"]:
                if hetero_key in graphs_serializable_v:
                    try:
                        graphs_serializable_v[hetero_key] = graphs_serializable_v[hetero_key].cpu()
                    except Exception:
                        pass

            graphs_pickle_v = pickle.dumps(graphs_serializable_v, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"无v28图数据序列化完成: {len(graphs_pickle_v) / 1024 / 1024:.1f} MB")

            # 使用变体特定的数据
            _graphs_pickle = graphs_pickle_v
            _train_compounds = train_compounds_v
            _val_compounds = val_compounds_v
            _compound_to_pos = compound_to_pos_serializable_v
            _val_proteins = list(val_proteins_v)
            logger.info(f"无v28冷启动拆分: {len(train_compounds_v)} train / {len(val_compounds_v)} val 化合物")
            logger.info(f"无v28蛋白冷启动: {len(train_proteins_v)} train / {len(val_proteins_v)} val 蛋白")
        else:
            _graphs_pickle = graphs_pickle
            _train_compounds = train_compounds
            _val_compounds = val_compounds
            _compound_to_pos = compound_to_pos_serializable
            _val_proteins = list(val_proteins)

        # ── 对每个种子运行 ──
        # ── 断点续跑：跳过已完成的变体 ──
        if _result_exists(name):
            logger.info(f"  ⏭ 跳过已完成: {name}")
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            existing_file = ABLATION_RESULTS_DIR / f"{safe_name}.json"
            try:
                with open(existing_file, "r", encoding="utf-8") as f:
                    existing_result = json.load(f)
                all_results.append(existing_result)
            except Exception as e:
                logger.warning(f"  无法加载已有结果: {e}")
            variant_count += len(ABLATION_SEEDS)
            continue
        # ── 断点续跑结束 ──

        variant_results = []  # 该变体所有种子的结果
        for seed_idx, seed in enumerate(ABLATION_SEEDS):
            variant_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"[{variant_count}/{n_total}] 变体: {name} [{cfg.get('priority', '?')}] | 种子: {seed} ({seed_idx+1}/{len(ABLATION_SEEDS)})")
            logger.info(f"  描述: {cfg.get('desc', 'N/A')}")
            logger.info(f"{'='*60}")

            variant_start = time.time()
            log_file = str(L4_LOGS / f"ablation_{name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}_seed{seed}.log")

            try:
                # 在子进程中运行
                with mp_ctx.Pool(processes=1) as pool:
                    async_result = pool.apply_async(
                        _run_variant_worker,
                        (
                            name, cfg, _graphs_pickle,
                            _train_compounds, _val_compounds,
                            _compound_to_pos,
                            _val_proteins,
                            pheno_train_indices, pheno_train_labels,
                            leak_smiles,
                            str(ABLATION_RESULTS_DIR),
                            log_file,
                            seed,  # v27: 传入随机种子
                        ),
                    )
                    result = async_result.get(timeout=7200)  # 2小时超时

                all_results.append(result)
                variant_results.append(result)
                elapsed = time.time() - variant_start

                if result.get("error"):
                    logger.error(f"  ✗ 变体 {name} (种子 {seed}) 失败: {result['error']}")
                else:
                    logger.info(f"  ✓ 变体 {name} (种子 {seed}) 完成 ({elapsed:.0f}s)")
                    logger.info(f"    SAGE prot_aupr={result.get('sage_prot_aupr', 'N/A')}, "
                                f"HGT prot_aupr={result.get('hgt_prot_aupr', 'N/A')}")

            except mp.TimeoutError:
                logger.error(f"  ✗ 变体 {name} (种子 {seed}) 超时（2小时），跳过")
                err_result = {"variant": name, "seed": seed, "error": "Timeout (2h)", "priority": cfg.get("priority", "P2")}
                all_results.append(err_result)
                variant_results.append(err_result)
            except Exception as e:
                logger.error(f"  ✗ 变体 {name} (种子 {seed}) 进程异常: {e}")
                err_result = {"variant": name, "seed": seed, "error": str(e), "priority": cfg.get("priority", "P2")}
                all_results.append(err_result)
                variant_results.append(err_result)

            # 强制清理 GPU 内存
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                logger.warning("GPU 缓存清理失败", exc_info=True)

            # 估算剩余时间
            if variant_count > 0:
                avg_time = (time.time() - start_time) / variant_count
                remaining = avg_time * (n_total - variant_count)
                logger.info(f"  预估剩余时间: {remaining/60:.0f} 分钟")

            # 保存中间结果
            inter_path = ABLATION_RESULTS_DIR / "ablation_intermediate_results.json"
            try:
                serializable = []
                for r in all_results:
                    sr = {}
                    for k, v in r.items():
                        if isinstance(v, np.floating):
                            sr[k] = float(v)
                        elif isinstance(v, np.integer):
                            sr[k] = int(v)
                        elif isinstance(v, set):
                            sr[k] = list(v)
                        else:
                            sr[k] = v
                    serializable.append(sr)
                with open(inter_path, "w", encoding="utf-8") as f:
                    json.dump(serializable, f, indent=2, ensure_ascii=False)
            except Exception:
                logger.warning("保存中间结果失败", exc_info=True)

        # 变体所有种子完成后，输出该变体的均值±标准差
        if len(variant_results) > 1:
            ok_results = [r for r in variant_results if not r.get("error")]
            if len(ok_results) >= 2:
                sage_vals = [r.get("sage_prot_aupr") for r in ok_results if r.get("sage_prot_aupr") is not None]
                hgt_vals = [r.get("hgt_prot_aupr") for r in ok_results if r.get("hgt_prot_aupr") is not None]
                if sage_vals:
                    logger.info(f"  ── {name} (SAGE) 均值 ± 标准差: {np.mean(sage_vals):.4f} ± {np.std(sage_vals):.4f} "
                                f"(n={len(sage_vals)})")
                if hgt_vals:
                    logger.info(f"  ── {name} (HGT)  均值 ± 标准差: {np.mean(hgt_vals):.4f} ± {np.std(hgt_vals):.4f} "
                                f"(n={len(hgt_vals)})")

    total_time = time.time() - start_time
    logger.info(f"\n消融实验完成: {n_total} 个变体, 总耗时 {total_time/60:.1f} 分钟")

    # ── 结果汇总 ──
    summary_path = ABLATION_RESULTS_DIR / "ablation_summary_v27.csv"
    df = summarize_results(all_results, output_path=summary_path)

    # 保存完整 JSON
    json_path = ABLATION_RESULTS_DIR / "ablation_full_results_v27.json"
    try:
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.floating):
                    sr[k] = float(v)
                elif isinstance(v, np.integer):
                    sr[k] = int(v)
                elif isinstance(v, set):
                    sr[k] = list(v)
                else:
                    sr[k] = v
            serializable.append(sr)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"完整结果已保存到: {json_path}")
    except Exception:
        logger.error("保存完整结果失败", exc_info=True)

    n_errors = sum(1 for r in all_results if r.get("error"))
    n_success = n_total - n_errors
    logger.info(f"成功: {n_success}/{n_total}, 失败: {n_errors}/{n_total}")

    return df


# ═══════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消融实验运行器 v27")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="指定要运行的变体名称（默认全部）")
    parser.add_argument("--skip-priority", nargs="*", default=None,
                        help="跳过的优先级（如 P2）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印变体定义，不运行")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("消融变体定义（dry-run）")
        logger.info("=" * 60)
        for name, cfg in ABLATION_VARIANTS.items():
            logger.info(f"  [{cfg.get('priority', '?')}] {name}")
            logger.info(f"    {cfg.get('desc', 'N/A')}")
            logger.info(f"    two_stage={cfg.get('two_stage')}, use_infonce={cfg.get('use_infonce')}, "
                        f"use_pathway={cfg.get('use_pathway')}, use_bpr={cfg.get('use_bpr')}, "
                        f"use_curriculum={cfg.get('use_curriculum')}, use_pheno={cfg.get('use_pheno')}, "
                        f"use_disease={cfg.get('use_disease')}, use_mlp_decoder={cfg.get('use_mlp_decoder')}")
        logger.info("=" * 60)
        sys.exit(0)

    # 过滤变体
    if args.variants:
        selected = {k: v for k, v in ABLATION_VARIANTS.items() if k in args.variants}
        if not selected:
            logger.error(f"未找到指定的变体: {args.variants}")
            logger.info(f"可用变体: {list(ABLATION_VARIANTS.keys())}")
            sys.exit(1)
        variants = selected
    else:
        variants = dict(ABLATION_VARIANTS)

    df = run_ablation(variants=variants, skip_priority=args.skip_priority)
    logger.info("消融实验完成")
