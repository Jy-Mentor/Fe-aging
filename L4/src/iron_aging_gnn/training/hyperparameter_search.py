"""超参数搜索模块 — 基于 Optuna 贝叶斯优化 + MLflow 实验追踪

从 configs/hyperparam_search.yaml 读取搜索空间和预算，
对 SAGE/HGT/SimpleHGN 模型进行系统化超参数调优。

用法:
    python -m iron_aging_gnn.training.hyperparameter_search --config configs/hyperparam_search.yaml
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


def _parse_search_space(search_space_cfg: dict) -> dict[str, dict]:
    """解析搜索空间配置，返回 Optuna 兼容的建议参数映射。"""
    suggestions = {}
    for param_name, param_cfg in search_space_cfg.items():
        ptype = param_cfg.get("type", "uniform")
        suggestions[param_name] = {
            "type": ptype,
            "low": param_cfg.get("low"),
            "high": param_cfg.get("high"),
            "choices": param_cfg.get("choices"),
        }
    return suggestions


def _sample_hyperparameters(trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """根据 Optuna trial 从搜索空间中采样超参数。"""
    params = {}
    for param_name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "log_uniform":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"], log=True
            )
        elif ptype == "uniform":
            params[param_name] = trial.suggest_float(
                param_name, spec["low"], spec["high"]
            )
        elif ptype == "categorical":
            params[param_name] = trial.suggest_categorical(
                param_name, spec["choices"]
            )
        elif ptype == "int":
            params[param_name] = trial.suggest_int(
                param_name, spec["low"], spec["high"]
            )
        else:
            logger.warning(f"未知搜索类型 {ptype}，跳过参数 {param_name}")
    return params


def _build_model_and_train(
    trial_params: dict[str, Any],
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    model_type: str,
    decoder_type: str,
    n_epochs: int,
    patience: int,
    val_proteins: set | None,
    random_seed: int,
    _validate_sage_fn=None,
    _validate_hgt_fn=None,
    _compute_cpi_loss_fn=None,
) -> tuple[float, float, list]:
    """根据采样参数构建模型并训练，返回最佳 AUPR 和 AUC。

    Returns:
        (best_aupr, best_auc, trial_history)
    """
    from ..models import SAGELinkPredictor, HGTLinkPredictor, SimpleHGNLinkPredictor
    from .trainer import train_sage, train_hgt, train_simplehgn

    hidden_dim = trial_params.get("hidden_dim", 64)
    out_dim = trial_params.get("out_dim", 64)
    num_layers = int(trial_params.get("num_layers", 2))
    dropout = trial_params.get("dropout", 0.5)
    lr = trial_params.get("lr", 1e-3)
    weight_decay = trial_params.get("weight_decay", 1e-4)
    batch_size = int(trial_params.get("batch_size", 128))
    num_neighbors_0 = int(trial_params.get("num_neighbors_0", 32))
    num_neighbors_1 = int(trial_params.get("num_neighbors_1", 16))
    temperature = trial_params.get("temperature", 1.0)
    focal_gamma = trial_params.get("focal_gamma", 2.0)
    focal_alpha = trial_params.get("focal_alpha", 0.75)

    num_neighbors = [num_neighbors_0, num_neighbors_1]

    comp_feat_dim = graphs["feat_dim"]
    prot_esm_dim = graphs["prot_esm_dim"]
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    n_pathways = graphs["n_pathways"]
    prot_to_path_neighbors = graphs.get("prot_to_path_neighbors", None)

    if model_type == "sage":
        model = SAGELinkPredictor(
            comp_feat_dim=comp_feat_dim,
            prot_feat_dim=prot_esm_dim,
            n_compounds=n_compounds,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            n_pathways=n_pathways,
            temperature=temperature,
            decoder_type=decoder_type,
        )
        model, history = train_sage(
            model=model,
            graphs=graphs,
            train_compounds=train_compounds,
            val_compounds=val_compounds,
            compound_to_pos=compound_to_pos,
            device=device,
            val_proteins=val_proteins,
            epochs=n_epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            prot_to_path_neighbors=prot_to_path_neighbors,
            weight_decay=weight_decay,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            random_seed=random_seed,
            _validate_sage_fn=_validate_sage_fn,
            _compute_cpi_loss_fn=_compute_cpi_loss_fn,
            two_stage=False,
            use_bpr=False,
            use_curriculum=False,
            use_plateau_scheduler=False,
        )
    elif model_type == "hgt":
        model = HGTLinkPredictor(
            comp_feat_dim=comp_feat_dim,
            prot_esm_dim=prot_esm_dim,
            n_compounds=n_compounds,
            n_proteins=n_proteins,
            n_pathways=n_pathways,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            temperature=temperature,
            decoder_type=decoder_type,
        )
        model, history = train_hgt(
            model=model,
            graphs=graphs,
            train_compounds=train_compounds,
            val_compounds=val_compounds,
            compound_to_pos=compound_to_pos,
            device=device,
            val_proteins=val_proteins,
            epochs=n_epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            prot_to_path_neighbors=prot_to_path_neighbors,
            weight_decay=weight_decay,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            random_seed=random_seed,
            _validate_hgt_fn=_validate_hgt_fn,
            _compute_cpi_loss_fn=_compute_cpi_loss_fn,
            two_stage=False,
            use_bpr=False,
            use_curriculum=False,
            use_plateau_scheduler=False,
        )
    elif model_type == "simplehgn":
        model = SimpleHGNLinkPredictor(
            comp_feat_dim=comp_feat_dim,
            prot_esm_dim=prot_esm_dim,
            n_compounds=n_compounds,
            n_proteins=n_proteins,
            n_pathways=n_pathways,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            temperature=temperature,
            decoder_type=decoder_type,
        )
        model, history = train_simplehgn(
            model=model,
            graphs=graphs,
            train_compounds=train_compounds,
            val_compounds=val_compounds,
            compound_to_pos=compound_to_pos,
            device=device,
            val_proteins=val_proteins,
            epochs=n_epochs,
            lr=lr,
            patience=patience,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            prot_to_path_neighbors=prot_to_path_neighbors,
            weight_decay=weight_decay,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            random_seed=random_seed,
            _validate_simplehgn_fn=_validate_hgt_fn,
            _compute_cpi_loss_fn=_compute_cpi_loss_fn,
            two_stage=False,
            use_bpr=False,
            use_curriculum=False,
            use_plateau_scheduler=False,
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if history:
        best_entry = max(history, key=lambda x: x.get("aupr", 0))
        best_aupr = best_entry.get("aupr", 0.0)
        best_auc = best_entry.get("auc", 0.0)
    else:
        best_aupr = 0.0
        best_auc = 0.0

    del model
    torch.cuda.empty_cache()

    return best_aupr, best_auc, history


def run_hyperparameter_search(
    config_path: str,
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    val_proteins: set | None = None,
    _validate_sage_fn=None,
    _validate_hgt_fn=None,
    _compute_cpi_loss_fn=None,
    random_seed: int = 42,
) -> dict[str, Any]:
    """执行超参数搜索。

    Args:
        config_path: hyperparam_search.yaml 配置文件路径。
        graphs: 图数据字典（来自 build_graphs_and_adj）。
        train_compounds: 训练化合物索引列表。
        val_compounds: 验证化合物索引列表。
        compound_to_pos: 化合物到正样本蛋白的映射。
        device: 计算设备。
        val_proteins: 验证蛋白索引集合。
        _validate_sage_fn: SAGE 验证函数。
        _validate_hgt_fn: HGT 验证函数。
        _compute_cpi_loss_fn: CPI 损失计算函数。
        random_seed: 随机种子。

    Returns:
        包含最优参数和结果的字典。
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"搜索配置文件不存在: {config_file}")

    with config_file.open(encoding="utf-8-sig") as f:
        search_cfg = yaml.safe_load(f)

    if search_cfg is None:
        raise ValueError(f"搜索配置文件为空: {config_file}")

    budget = search_cfg.get("budget", {})
    n_trials = budget.get("n_trials", 50)
    n_epochs = budget.get("n_epochs", 30)
    patience = budget.get("early_stopping_patience", 5)
    timeout = budget.get("timeout", 0) or None
    n_jobs = budget.get("n_jobs", 1)

    search_space_raw = search_cfg.get("search_space", {})
    search_space = _parse_search_space(search_space_raw)

    optimization_cfg = search_cfg.get("optimization", {})
    direction = optimization_cfg.get("direction", "maximize")
    metric_name = optimization_cfg.get("metric", "val/aupr")
    pruner_type = optimization_cfg.get("pruner", "median")
    pruner_startup = optimization_cfg.get("pruner_startup_trials", 5)
    pruner_warmup = optimization_cfg.get("pruner_warmup_steps", 3)

    mlflow_cfg = search_cfg.get("mlflow", {})
    mlflow_enabled = mlflow_cfg.get("enabled", False)
    experiment_name = mlflow_cfg.get("experiment_name", "iron_aging_gnn_hp_search")

    model_cfg = search_cfg.get("model", {})
    model_type = model_cfg.get("type", "sage")
    decoder_type = model_cfg.get("decoder_type", "residue_bilinear")

    logger.info("=" * 60)
    logger.info(f"超参数搜索启动: model={model_type}, n_trials={n_trials}, "
                f"n_epochs={n_epochs}, direction={direction}")
    logger.info(f"搜索空间: {list(search_space.keys())}")
    logger.info("=" * 60)

    try:
        import optuna
    except ImportError:
        logger.error("Optuna 未安装。请执行: pip install optuna")
        raise

    if pruner_type == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_startup,
            n_warmup_steps=pruner_warmup,
        )
    elif pruner_type == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        direction=direction,
        pruner=pruner,
        study_name=f"{model_type}_hp_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    if mlflow_enabled:
        try:
            import mlflow
            mlflow.set_experiment(experiment_name)
            mlflow_started = True
        except ImportError:
            logger.warning("MLflow 未安装，跳过 MLflow 追踪。pip install mlflow")
            mlflow_started = False
        except Exception as e:
            logger.warning(f"MLflow 初始化失败: {e}")
            mlflow_started = False
    else:
        mlflow_started = False

    def objective(trial: optuna.Trial) -> float:
        trial_params = _sample_hyperparameters(trial, search_space)
        trial_start = time.time()

        logger.info(
            f"--- Trial {trial.number} --- "
            + ", ".join(f"{k}={v}" for k, v in trial_params.items())
        )

        if mlflow_started:
            mlflow.start_run(run_name=f"trial_{trial.number}")
            mlflow.log_params(trial_params)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("decoder_type", decoder_type)
            mlflow.log_param("trial_number", trial.number)

        best_aupr, best_auc, history = _build_model_and_train(
            trial_params=trial_params,
            graphs=graphs,
            train_compounds=train_compounds,
            val_compounds=val_compounds,
            compound_to_pos=compound_to_pos,
            device=device,
            model_type=model_type,
            decoder_type=decoder_type,
            n_epochs=n_epochs,
            patience=patience,
            val_proteins=val_proteins,
            random_seed=random_seed + trial.number,
            _validate_sage_fn=_validate_sage_fn,
            _validate_hgt_fn=_validate_hgt_fn,
            _compute_cpi_loss_fn=_compute_cpi_loss_fn,
        )

        trial_elapsed = time.time() - trial_start
        logger.info(
            f"Trial {trial.number} 完成: best_aupr={best_aupr:.4f}, "
            f"best_auc={best_auc:.4f}, elapsed={trial_elapsed:.1f}s"
        )

        if mlflow_started:
            mlflow.log_metric("best_aupr", best_aupr)
            mlflow.log_metric("best_auc", best_auc)
            mlflow.log_metric("trial_elapsed_sec", trial_elapsed)
            mlflow.end_run()

        return best_aupr

    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs)

    logger.info("=" * 60)
    logger.info("超参数搜索完成")
    logger.info(f"最优试验: {study.best_trial.number}")
    logger.info(f"最优 AUPR: {study.best_value:.4f}")
    logger.info("最优参数:")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    result = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "direction": direction,
        "metric": metric_name,
        "model_type": model_type,
        "decoder_type": decoder_type,
    }

    return result


def main():
    """命令行入口 — 从 phase4_v10_minibatch 加载图数据并执行超参数搜索。"""
    parser = argparse.ArgumentParser(
        description="铁衰老 GNN 超参数搜索 (Optuna + MLflow)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/hyperparam_search.yaml",
        help="超参数搜索配置文件路径",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    scripts_dir = project_root / "scripts"
    sys.path.insert(0, str(scripts_dir))

    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)

    logger.info("从 phase4_v10_minibatch 加载图数据和训练基础设施...")

    from phase4_v10_minibatch import (
        _compute_cpi_loss,
        _validate_hgt,
        _validate_sage,
        build_graphs_and_adj,
        load_cpi_data,
        load_kegg_pathways,
        load_ppi_network,
        load_protein_features,
        load_tcm_pool,
    )

    random_seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"计算设备: {device}")

    # 加载数据
    cpi_df = load_cpi_data()
    ppi_df = load_ppi_network()
    gene_to_pathways = load_kegg_pathways()
    prot_feat, gene_to_seq = load_protein_features()
    tcm_df = load_tcm_pool()

    # TCM 与 CPI 隔离
    tcm_smiles_col = "SMILES_std" if "SMILES_std" in tcm_df.columns else (
        "SMILES" if "SMILES" in tcm_df.columns else "canonical_smiles")
    tcm_smiles_set = set(tcm_df[tcm_smiles_col].dropna().astype(str))
    cpi_df = cpi_df[~cpi_df["canonical_smiles"].isin(tcm_smiles_set)].copy()

    # warm_targets：所有有 CPI 数据的基因
    all_cpi_genes = sorted(set(cpi_df["gene"].unique()))
    warm_targets = all_cpi_genes
    cpi_df = cpi_df[cpi_df["gene"].isin(warm_targets)].copy()

    logger.info("数据加载完成，开始构建图...")

    # 构建图
    graphs = build_graphs_and_adj(cpi_df, ppi_df, gene_to_pathways, prot_feat)
    logger.info("图构建完成")

    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    n_pathways = graphs["n_pathways"]
    feat_dim = graphs["feat_dim"]
    prot_esm_dim = graphs["prot_esm_dim"]

    logger.info(
        f"图统计: compounds={n_compounds}, proteins={n_proteins}, "
        f"pathways={n_pathways}, feat_dim={feat_dim}, "
        f"prot_esm_dim={prot_esm_dim}"
    )

    # 化合物冷启动拆分: 85% train / 15% val
    all_compounds = sorted(graphs["smi_to_idx"].values())
    random.seed(random_seed)
    random.shuffle(all_compounds)
    n_train_comp = int(len(all_compounds) * 0.85)
    train_compounds = all_compounds[:n_train_comp]
    val_compounds = all_compounds[n_train_comp:]

    # 蛋白训练/验证拆分
    cpi_proteins = set()
    for _, row in cpi_df.iterrows():
        gene = row["gene"]
        if gene in graphs["gene_to_idx"]:
            cpi_proteins.add(graphs["gene_to_idx"][gene] - n_compounds)
    all_proteins = sorted({
        graphs["gene_to_idx"][g] - n_compounds
        for g in graphs["gene_to_idx"]
        if graphs["gene_to_idx"][g] >= n_compounds
    })
    non_cpi_proteins = [p for p in all_proteins if p not in cpi_proteins]
    cpi_proteins_list = list(cpi_proteins)
    random.shuffle(cpi_proteins_list)
    random.shuffle(non_cpi_proteins)
    n_val_cpi = max(1, int(len(cpi_proteins) * 0.5))
    n_train_cpi = len(cpi_proteins) - n_val_cpi
    n_val_non_cpi = max(1, int(len(non_cpi_proteins) * 0.5))
    n_train_non_cpi = len(non_cpi_proteins) - n_val_non_cpi
    val_proteins = set(cpi_proteins_list[n_train_cpi:]) | set(non_cpi_proteins[n_train_non_cpi:])

    # 预计算正样本
    compound_to_pos = defaultdict(set)
    for _, row in cpi_df.iterrows():
        smi = row["canonical_smiles"]
        gene = row["gene"]
        if smi in graphs["smi_to_idx"] and gene in graphs["gene_to_idx"]:
            compound_to_pos[graphs["smi_to_idx"][smi]].add(graphs["gene_to_idx"][gene])

    logger.info(
        f"训练/验证拆分: train_compounds={len(train_compounds)}, "
        f"val_compounds={len(val_compounds)}, "
        f"train_proteins={n_train_cpi + n_train_non_cpi}, "
        f"val_proteins={len(val_proteins)}"
    )

    result = run_hyperparameter_search(
        config_path=str(config_path),
        graphs=graphs,
        train_compounds=train_compounds,
        val_compounds=val_compounds,
        compound_to_pos=compound_to_pos,
        device=device,
        val_proteins=val_proteins,
        _validate_sage_fn=_validate_sage,
        _validate_hgt_fn=_validate_hgt,
        _compute_cpi_loss_fn=_compute_cpi_loss,
        random_seed=random_seed,
    )

    logger.info("最终结果:")
    for k, v in result.items():
        logger.info(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    main()