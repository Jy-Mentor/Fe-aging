#!/usr/bin/env python3
"""
统计显著性实验 — v27 5 种子重复运行
目的：报告 v27 在所有指标上的均值 ± 标准差，用于论文统计显著性。
v27 适配项:
  - PROTEIN_VAL_SPLIT: 0.20 → 0.50（提高蛋白冷启动验证统计效力）
  - 疾病节点（GSE61616）四模态异质图
  - CuDNN 确定性模式（确保完全可复现）
  - 排名指标（Precision@K, EF@1%, EF@5%）
  - CPI补充v28数据
架构：spawn 多进程，每个种子在独立子进程中运行完整 v27 训练。
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 路径 ──
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L4_SCRIPTS = PROJECT_ROOT / "L4" / "scripts"
L4_SRC = PROJECT_ROOT / "L4" / "src"
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v10_minibatch"
L4_LOGS = PROJECT_ROOT / "L4" / "logs"
L4_LOGS.mkdir(parents=True, exist_ok=True)
L4_RESULTS.mkdir(parents=True, exist_ok=True)

# ── v27: 5 个种子 ──
SEEDS = [42, 123, 456, 789, 1024]

# ── v27 配置 ──
COMPOUND_VAL_SPLIT = 0.85
PROTEIN_VAL_SPLIT = 0.50  # v27: 从20%提高到50%

# ── 主进程日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(L4_LOGS / "stat_significance_v27_master.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger("stat_sig")


def _prepare_data():
    """主进程加载数据、构建图、拆分，缓存到磁盘（v27: 含疾病节点+CPI v28补充）"""
    sys.path.insert(0, str(L4_SRC))
    sys.path.insert(0, str(L4_SCRIPTS))
    import phase4_v10_minibatch as p4

    logger.info(">>> 主进程: 加载数据")
    cpi_df = p4.load_cpi_data()
    ppi_df = p4.load_ppi_network()
    gene_to_pathways = p4.load_kegg_pathways()
    prot_feat, gene_to_seq = p4.load_protein_features()

    # v27: 加载疾病边
    disease_df = None
    disease_file = L4_RESULTS / "disease_gene_edges.csv"
    if disease_file.exists():
        disease_df = pd.read_csv(disease_file)
        logger.info(f">>> 疾病节点: {len(disease_df)} 条疾病-蛋白边")

    warm_targets = sorted(set(cpi_df["gene"].unique()) & set(p4.ALL_FERRORAGING_GENES))
    logger.info(f"  温靶标: {len(warm_targets)} 个")
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    logger.info(">>> 主进程: 构建图 & 邻接表（v27: 含疾病节点）")
    graphs = p4.build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat, disease_df=disease_df)

    # 拆分
    all_compounds = sorted(graphs["smi_to_idx"].values())
    all_proteins = sorted(set(
        graphs["gene_to_idx"][g] - graphs["n_compounds"]
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= graphs["n_compounds"]
    ))
    import random
    random.seed(42)
    random.shuffle(all_compounds)
    random.shuffle(all_proteins)

    n_train_comp = int(len(all_compounds) * COMPOUND_VAL_SPLIT)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    # v27: 蛋白冷启动分层拆分 (50% CPI蛋白)
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

    logger.info(f"  蛋白冷启动拆分: {len(cpi_proteins)} CPI蛋白, "
                f"训练CPI={n_train_cpi}, 验证CPI={n_val_cpi}")

    # 预计算正样本
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    # 验证安全图
    val_comp_set = set(val_compounds)
    graphs["homo_edge_index_val"] = p4._build_val_comp_cold_homo_edge_index(
        graphs["homo_edge_index"], val_comp_set)
    graphs["hetero_data_val"] = p4._build_val_comp_cold_hetero_data(
        graphs["hetero_data"], val_comp_set)
    graphs["homo_edge_index_prot_cold"] = p4._build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_data_prot_cold"] = p4._build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)
    graphs["homo_adj_train"] = p4._build_train_safe_homo_adj(
        graphs["homo_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_adj_train"] = p4._build_train_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_adj_val"] = p4._build_val_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_adj_prot_cold"] = p4._build_val_safe_hetero_adj(
        graphs["hetero_adj"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["homo_edge_index_train"] = p4._build_val_safe_homo_edge_index(
        graphs["homo_edge_index"], graphs["n_compounds"], val_comp_set, val_proteins)
    graphs["hetero_data_train"] = p4._build_val_safe_hetero_data(
        graphs["hetero_data"], val_comp_set, val_proteins)

    logger.info(f"  训练安全邻接表: SAGE {sum(len(v) for v in graphs['homo_adj_train'].values())} 条边, "
                f"HGT {sum(len(v) for v in graphs['hetero_adj_train'].values())} 条边")

    # 缓存
    import torch
    hetero_data = graphs.pop("hetero_data", None)
    hetero_data_val = graphs.pop("hetero_data_val", None)
    hetero_data_prot_cold = graphs.pop("hetero_data_prot_cold", None)
    hetero_data_train = graphs.pop("hetero_data_train", None)

    cache_path = L4_RESULTS / "graphs_cache_stat_sig_v27.pt"
    torch.save(graphs, cache_path)
    torch.save({
        "hetero_data": hetero_data,
        "hetero_data_val": hetero_data_val,
        "hetero_data_prot_cold": hetero_data_prot_cold,
        "hetero_data_train": hetero_data_train,
    }, L4_RESULTS / "hetero_cache_stat_sig_v27.pt")

    graphs["hetero_data"] = hetero_data
    graphs["hetero_data_val"] = hetero_data_val
    graphs["hetero_data_prot_cold"] = hetero_data_prot_cold
    graphs["hetero_data_train"] = hetero_data_train

    compound_to_pos_serializable = {str(k): sorted(list(v)) for k, v in compound_to_pos.items()}
    split_params = {
        "train_compounds": train_compounds,
        "val_compounds": val_compounds,
        "val_proteins": sorted(list(val_proteins)),
        "compound_to_pos": compound_to_pos_serializable,
        "n_compounds": graphs["n_compounds"],
        "n_proteins": graphs["n_proteins"],
        "n_pathways": graphs["n_pathways"],
        "n_diseases": graphs.get("n_diseases", 0),
        "feat_dim": graphs["feat_dim"],
        "prot_esm_dim": graphs["prot_esm_dim"],
    }
    split_path = L4_RESULTS / "split_params_stat_sig_v27.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_params, f, ensure_ascii=False, indent=2)
    logger.info(f"  拆分参数已保存: {split_path}")

    return split_params, cache_path


def _worker_run_seed(payload: int) -> dict:
    """子进程：用指定种子运行 v27 完整训练"""
    seed = payload
    worker_logger = logging.getLogger(f"stat_sig.seed{seed}")
    worker_logger.handlers.clear()
    worker_logger.setLevel(logging.INFO)
    _fh = logging.FileHandler(L4_LOGS / f"stat_sig_v27_seed{seed}.log", encoding="utf-8", mode="w")
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    worker_logger.addHandler(_fh)
    _sh = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    worker_logger.addHandler(_sh)
    worker_logger.propagate = False

    worker_logger.info(f"{'='*60}")
    worker_logger.info(f">>> 种子 {seed}: 开始 v27 训练")

    result = {"seed": seed, "status": "OK", "error": ""}

    try:
        import random
        import numpy as np
        import torch
        import torch.nn as nn

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # v27: CuDNN 确定性模式
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            worker_logger.info("  CuDNN 确定性模式已启用")

        sys.path.insert(0, str(L4_SRC))
        sys.path.insert(0, str(L4_SCRIPTS))
        import phase4_v10_minibatch as p4

        cache_path = L4_RESULTS / "graphs_cache_stat_sig_v27.pt"
        hetero_cache_path = L4_RESULTS / "hetero_cache_stat_sig_v27.pt"

        graphs = torch.load(cache_path, weights_only=False)
        hetero_dict = torch.load(hetero_cache_path, weights_only=False)
        graphs["hetero_data"] = hetero_dict["hetero_data"]
        graphs["hetero_data_val"] = hetero_dict["hetero_data_val"]
        graphs["hetero_data_prot_cold"] = hetero_dict["hetero_data_prot_cold"]
        graphs["hetero_data_train"] = hetero_dict["hetero_data_train"]

        split_path = L4_RESULTS / "split_params_stat_sig_v27.json"
        with open(split_path, "r", encoding="utf-8") as f:
            split_params = json.load(f)

        train_compounds = split_params["train_compounds"]
        val_compounds = split_params["val_compounds"]
        val_proteins = set(split_params["val_proteins"])
        compound_to_pos = {int(k): set(v) for k, v in split_params["compound_to_pos"].items()}

        # ── v27: 表型数据处理 ──
        pheno_train_indices = None
        pheno_train_labels = None
        pheno_file = L4_RESULTS / "phenotype_ferroptosis_dataset_v25_clean.csv"
        if pheno_file.exists():
            pheno_df = pd.read_csv(pheno_file)
            smi_col = None
            for col in ["canonical_smiles", "SMILES", "smiles"]:
                if col in pheno_df.columns:
                    smi_col = col
                    break
            if smi_col is not None:
                pheno_indices = []
                pheno_labels_list = []
                for _, row in pheno_df.iterrows():
                    smi = row[smi_col]
                    if pd.isna(smi):
                        continue
                    if smi in graphs["smi_to_idx"]:
                        pheno_indices.append(graphs["smi_to_idx"][smi])
                        pheno_labels_list.append(int(row["label"]))
                if pheno_indices:
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
                    worker_logger.info(f"  表型数据: {len(train_pheno_idx)} 训练样本")

        worker_logger.info(f"  训练 SAGE...")
        sage_model = p4.SAGELinkPredictor(
            comp_feat_dim=graphs["feat_dim"], prot_feat_dim=graphs["prot_esm_dim"],
            n_compounds=graphs["n_compounds"],
            hidden_dim=64, out_dim=64, num_layers=2, dropout=0.5,
            n_pathways=graphs["n_pathways"])
        sage_model, sage_history = p4.train_sage(
            sage_model, graphs, train_compounds, val_compounds, compound_to_pos,
            val_proteins=val_proteins,
            epochs=15, lr=5e-4, patience=5, batch_size=256, num_neighbors=[32, 16],
            prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
            two_stage=True, pretrain_epochs=10, pretrain_lr=7.5e-4,
            use_infonce=False, use_bpr=True, use_curriculum=True,  # v27: 无InfoNCE
            pheno_compound_indices=pheno_train_indices,
            pheno_labels=pheno_train_labels,
            pheno_lambda=0.05)

        sage_best_auc = max(h["auc"] for h in sage_history) if sage_history else 0.5
        sage_best_aupr = max(h.get("aupr", 0) for h in sage_history) if sage_history else 0.0

        # 蛋白冷启动评估
        sage_model.eval()
        sage_prot = p4._validate_sage_protein_cold(
            sage_model, graphs["x"], graphs.get("homo_edge_index_prot_cold", graphs["homo_edge_index"]),
            val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"], val_proteins,
            prot_esm_dim=graphs.get("prot_esm_dim", 0), n_pathways=graphs.get("n_pathways", 0))
        sage_prot_auc = sage_prot["auc"]
        sage_prot_aupr = sage_prot["aupr"]

        # v27: 排名指标
        sage_prot_p10 = sage_prot.get("precision@10", 0.0)
        sage_prot_p20 = sage_prot.get("precision@20", 0.0)
        sage_prot_p50 = sage_prot.get("precision@50", 0.0)
        sage_prot_ef1 = sage_prot.get("ef@1%", 0.0)
        sage_prot_ef5 = sage_prot.get("ef@5%", 0.0)

        torch.cuda.empty_cache()
        worker_logger.info("  SAGE GPU 内存已释放")

        worker_logger.info(f"  训练 HGT...")
        hgt_node_feat_dims = {
            "compound": graphs["feat_dim"],
            "protein": graphs["prot_esm_dim"],
            "pathway": 1,
            "pathway_count": graphs["n_pathways"],
            "disease_count": graphs.get("n_diseases", 0),
        }
        hgt_model = p4.HGTLinkPredictor(
            hidden_dim=64, out_dim=64, num_heads=2, num_layers=2,
            dropout=0.5, metadata=graphs["hetero_data"].metadata(),
            compound_feat_dim=graphs["feat_dim"], node_feat_dims=hgt_node_feat_dims)
        hgt_model, hgt_history = p4.train_hgt(
            hgt_model, graphs, train_compounds, val_compounds, compound_to_pos,
            val_proteins=val_proteins,
            epochs=15, lr=1e-3, patience=5, batch_size=128, num_neighbors=[32, 16],
            prot_to_path_neighbors=graphs.get("prot_to_path_neighbors"),
            two_stage=True, pretrain_epochs=10, pretrain_lr=1.5e-3,
            use_infonce=False, use_bpr=True, use_curriculum=True,  # v27: 无InfoNCE
            pheno_compound_indices=pheno_train_indices,
            pheno_labels=pheno_train_labels,
            pheno_lambda=0.05)

        hgt_best_auc = max(h["auc"] for h in hgt_history) if hgt_history else 0.5
        hgt_best_aupr = max(h.get("aupr", 0) for h in hgt_history) if hgt_history else 0.0

        hgt_model.eval()
        hgt_prot = p4._validate_hgt_protein_cold(
            hgt_model, graphs.get("hetero_data_prot_cold", graphs["hetero_data"]), val_compounds, compound_to_pos,
            graphs["n_compounds"], graphs["n_proteins"], val_proteins,
            hetero_adj=graphs.get("hetero_adj_prot_cold", graphs.get("hetero_adj")),
            prot_esm_dim=graphs.get("prot_esm_dim", 0), pathway_dim=graphs.get("n_pathways", 0))
        hgt_prot_auc = hgt_prot["auc"]
        hgt_prot_aupr = hgt_prot["aupr"]

        # v27: 排名指标
        hgt_prot_p10 = hgt_prot.get("precision@10", 0.0)
        hgt_prot_p20 = hgt_prot.get("precision@20", 0.0)
        hgt_prot_p50 = hgt_prot.get("precision@50", 0.0)
        hgt_prot_ef1 = hgt_prot.get("ef@1%", 0.0)
        hgt_prot_ef5 = hgt_prot.get("ef@5%", 0.0)

        torch.cuda.empty_cache()

        result.update({
            # 常规指标
            "sage_val_auc": sage_best_auc, "sage_val_aupr": sage_best_aupr,
            "sage_prot_auc": sage_prot_auc, "sage_prot_aupr": sage_prot_aupr,
            "hgt_val_auc": hgt_best_auc, "hgt_val_aupr": hgt_best_aupr,
            "hgt_prot_auc": hgt_prot_auc, "hgt_prot_aupr": hgt_prot_aupr,
            # v27: 排名指标
            "sage_prot_precision@10": sage_prot_p10,
            "sage_prot_precision@20": sage_prot_p20,
            "sage_prot_precision@50": sage_prot_p50,
            "sage_prot_ef@1%": sage_prot_ef1,
            "sage_prot_ef@5%": sage_prot_ef5,
            "hgt_prot_precision@10": hgt_prot_p10,
            "hgt_prot_precision@20": hgt_prot_p20,
            "hgt_prot_precision@50": hgt_prot_p50,
            "hgt_prot_ef@1%": hgt_prot_ef1,
            "hgt_prot_ef@5%": hgt_prot_ef5,
        })
        worker_logger.info(f"  SAGE: val_auc={sage_best_auc:.4f} val_aupr={sage_best_aupr:.4f} "
                           f"prot_auc={sage_prot_auc:.4f} prot_aupr={sage_prot_aupr:.4f} "
                           f"P@10={sage_prot_p10:.4f} EF@1%={sage_prot_ef1:.2f}")
        worker_logger.info(f"  HGT:  val_auc={hgt_best_auc:.4f} val_aupr={hgt_best_aupr:.4f} "
                           f"prot_auc={hgt_prot_auc:.4f} prot_aupr={hgt_prot_aupr:.4f} "
                           f"P@10={hgt_prot_p10:.4f} EF@1%={hgt_prot_ef1:.2f}")

    except Exception:
        worker_logger.error(f"种子 {seed} 失败:\n{traceback.format_exc()}")
        result["status"] = "ERROR"
        result["error"] = traceback.format_exc()

    return result


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"v27 统计显著性实验 — {len(SEEDS)} 个种子: {SEEDS}")
    logger.info(f"配置: PROTEIN_VAL_SPLIT={PROTEIN_VAL_SPLIT}, 疾病节点, CuDNN确定性, 排名指标")
    logger.info("=" * 60)

    # ── 主进程准备数据 ──
    _prepare_data()

    # ── 多进程运行 ──
    logger.info(f">>> 启动 {len(SEEDS)} 个子进程（串行执行，GPU 隔离）")
    mp.set_start_method("spawn", force=True)
    results = []

    for i, seed in enumerate(SEEDS):
        logger.info(f"\n{'='*40}")
        logger.info(f">>> [{i+1}/{len(SEEDS)}] 种子 {seed}")
        logger.info(f"{'='*40}")
        try:
            result = _worker_run_seed(seed)
            results.append(result)
        except Exception as e:
            logger.error(f"种子 {seed} 进程级异常: {e}")
            results.append({"seed": seed, "status": "ERROR", "error": str(e)})

        # 中间保存
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(L4_RESULTS / "stat_significance_v27_results_partial.csv", index=False)

    # ── 汇总分析 ──
    df = pd.DataFrame(results)
    df.to_csv(L4_RESULTS / "stat_significance_v27_results.csv", index=False)
    logger.info(f"\n原始结果已保存: {L4_RESULTS / 'stat_significance_v27_results.csv'}")

    ok_results = [r for r in results if r["status"] == "OK"]
    logger.info(f"\n成功: {len(ok_results)}/{len(SEEDS)}")

    if len(ok_results) >= 2:
        metrics = [
            # 常规指标
            "sage_val_auc", "sage_val_aupr", "sage_prot_auc", "sage_prot_aupr",
            "hgt_val_auc", "hgt_val_aupr", "hgt_prot_auc", "hgt_prot_aupr",
            # v27: 排名指标
            "sage_prot_precision@10", "sage_prot_precision@20", "sage_prot_precision@50",
            "sage_prot_ef@1%", "sage_prot_ef@5%",
            "hgt_prot_precision@10", "hgt_prot_precision@20", "hgt_prot_precision@50",
            "hgt_prot_ef@1%", "hgt_prot_ef@5%",
        ]
        summary = []
        for m in metrics:
            vals = [r[m] for r in ok_results if r.get(m) is not None]
            if vals:
                summary.append({
                    "metric": m,
                    "mean": np.mean(vals),
                    "std": np.std(vals),
                    "min": np.min(vals),
                    "max": np.max(vals),
                    "n": len(vals),
                })

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv(L4_RESULTS / "stat_significance_v27_summary.csv", index=False)

        logger.info("\n" + "=" * 70)
        logger.info("v27 统计显著性汇总 (均值 ± 标准差)")
        logger.info("=" * 70)
        for row in summary:
            logger.info(f"  {row['metric']:30s}: {row['mean']:.4f} ± {row['std']:.4f}  "
                        f"[{row['min']:.4f}, {row['max']:.4f}]  (n={row['n']})")
        logger.info("=" * 70)

    total_time = time.time() - start_time
    logger.info(f"\n总耗时: {total_time / 60:.1f} 分钟")


if __name__ == "__main__":
    main()