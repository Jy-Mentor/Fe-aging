"""训练模块：SAGE 和 HGT 的 mini-batch 训练函数

支持两阶段迁移学习、表型分类辅助任务、课程负采样、Memory Bank 等。

以下辅助函数从主脚本中注入：
  - _compute_cpi_loss (losses)
  - _validate_sage, _validate_hgt (validation)
"""

from __future__ import annotations

import logging
import random

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.amp import GradScaler, autocast

from ..graph import (
    drop_edge,
    sample_hetero_subgraph,
    sample_homo_subgraph,
    split_head_tail_nodes,
)
from ..models import HGTLinkPredictor, MemoryBank, RGCNLinkPredictor, SAGELinkPredictor, SimpleHGNLinkPredictor
from .training_components import (
    GradientMonitor,
    LRSchedulerFactory,
    MemoryBankManager,
    Validator,
)
from .training_config import TrainingConfig

logger = logging.getLogger(__name__)


def _check_tensor_nan(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """检查张量是否包含 NaN 或 Inf，返回 True 表示有问题"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        logger.warning(f"张量 {name} 包含 NaN 或 Inf: "
                       f"NaN={torch.isnan(tensor).sum().item()}, "
                       f"Inf={torch.isinf(tensor).sum().item()}")
        return True
    return False


def _check_gradient_norm(model: nn.Module, warn_threshold: float = 100.0) -> float:
    """计算模型梯度总范数，若超过阈值则记录警告"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    if total_norm > warn_threshold:
        logger.warning(f"梯度范数异常: {total_norm:.1f} > {warn_threshold:.1f}")
    return total_norm


def train_sage(
    model: SAGELinkPredictor,
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 256,
    num_neighbors: list[int] = None,
    prot_to_path_neighbors: dict[int, set] | None = None,
    two_stage: bool = False,
    pretrain_epochs: int = 0,
    pretrain_lr: float | None = None,
    random_seed: int = 42,
    use_infonce: bool = False,
    use_bpr: bool = True,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    pheno_compound_indices: list[int] = None,
    pheno_labels: list[int] = None,
    pheno_lambda: float = 0.3,
    bpr_weight: float = 0.4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.05,
    dropedge_ppi: float = 0.15,
    dropedge_pathway: float = 0.10,
    dropedge_cpi: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    memory_bank_size: int = 8192,
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    head_undersample_ratio: float = 0.6,
    grad_clip_norm: float = 1.0,
    pretrain_lr_multiplier: float = 1.5,
    pretrain_lr_decay: float = 0.5,
    finetune_lr_multiplier: float = 0.5,
    use_plateau_scheduler: bool = False,
    plateau_patience: int = 2,
    plateau_factor: float = 0.5,
    _validate_sage_fn=None,
    _compute_cpi_loss_fn=None,
    use_amp: bool = False,
) -> tuple[SAGELinkPredictor, list[dict]]:
    """GraphSAGE mini-batch 训练

    阶段1 (可选, two_stage): 在尾节点平衡子图上预训练，学习稀疏靶标表示
    阶段2: 在完整训练图上微调（CPI + 表型多任务联合训练）
    """
    if _validate_sage_fn is None or _compute_cpi_loss_fn is None:
        raise ValueError(
            "train_sage 需要注入 _validate_sage_fn, _compute_cpi_loss_fn 参数。"
        )

    if num_neighbors is None:
        num_neighbors = [32, 16]
    model = model.to(device)
    # v55-fix: RTX 50 系 / cu128 在 WDDM 下对 TF32 Tensor Core 存在稳定性问题，
    # 可能触发 CUBLAS_STATUS_EXECUTION_FAILED（错误信息中显示 CUDA_R_16F）。
    # 禁用 TF32 强制使用 float32 稳定路径。
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info("  禁用 CUDA TF32，使用 float32 稳定计算路径")
    # v49-fix: 禁止在此处无条件重初始化模型参数。
    # SAGE/HGT 模型构造函数与 ResidueAwareBilinearDecoder._init_weights 已实施受控初始化，
    # 此处 Xavier 重初始化会覆盖解码器的精心初始化（score_mlp 末层小增益等），
    # 导致训练初期数值不稳定、验证指标恶化，并破坏预训练权重加载。

    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            import importlib.util
            if importlib.util.find_spec("triton") is not None:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info('SAGE model compiled with torch.compile (reduce-overhead)')
            else:
                logger.info('Triton not available, skipping torch.compile for SAGE')
        except Exception as e:
            logger.warning(f'torch.compile failed for SAGE: {e}, continuing without compilation')

    # v54-fix: 完整节点特征矩阵保留在 CPU，仅按 batch 子图取到 GPU，
    # 避免 8GB 显存 (WDDM) 在启动阶段移动全图特征即 OOM。
    x = graphs["x"]
    homo_adj = graphs.get("homo_adj_train", graphs["homo_adj"])
    n_compounds = graphs["n_compounds"]
    all_compound_to_pos = compound_to_pos

    _homo_edge_index_val = graphs.get("homo_edge_index_val", graphs["homo_edge_index"])
    _homo_edge_index_train = graphs.get("homo_edge_index_train", graphs["homo_edge_index"])

    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}
    compound_to_prot_locals = {c: [p - n_compounds for p in pos_set] for c, pos_set in precomputed_pos.items()}

    prot_to_topo_medium_neighbors = graphs.get("prot_to_topo_medium_neighbors")
    prot_to_topo_hard_neighbors = graphs.get("prot_to_topo_hard_neighbors")

    use_pheno = (pheno_compound_indices is not None and pheno_labels is not None
                 and len(pheno_compound_indices) > 0 and len(pheno_labels) > 0)
    pheno_comp_set = set()
    pheno_idx_to_label = {}
    bce_loss_fn = None
    if use_pheno:
        pheno_comp_set = set(pheno_compound_indices)
        pheno_idx_to_label = dict(zip(pheno_compound_indices, pheno_labels, strict=False))
        n_pos_pheno = sum(pheno_labels)
        n_neg_pheno = len(pheno_labels) - n_pos_pheno
        pos_weight = n_neg_pheno / max(n_pos_pheno, 1)
        pheno_pos_weight = torch.tensor([pos_weight], device=device)
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pheno_pos_weight)
        logger.info(f"  表型分类任务: {len(pheno_compound_indices)} 个化合物, "
                    f"正样本={n_pos_pheno}, 负样本={n_neg_pheno}, "
                    f"pos_weight={pos_weight:.1f}, lambda={pheno_lambda}")

    finetune_lr_actual = lr * finetune_lr_multiplier if (two_stage and pretrain_epochs > 0) else lr
    logger.info(f"  SAGE 微调初始学习率: {finetune_lr_actual:.2e} (lr={lr:.2e}, multiplier={finetune_lr_multiplier})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr_actual, weight_decay=weight_decay, foreach=False)
    if use_plateau_scheduler:
        scheduler = LRSchedulerFactory.create_plateau(
            optimizer, patience=plateau_patience, factor=plateau_factor, mode="max", metric_name="aupr"
        )
    else:
        scheduler = LRSchedulerFactory.create_cosine_warmup(optimizer, epochs, warmup_ratio)
    scaler = GradScaler('cuda', enabled=use_amp)

    memory_bank_mgr = MemoryBankManager(memory_bank_size, model.out_dim, str(device))
    memory_bank = memory_bank_mgr.memory_bank
    gradient_monitor = GradientMonitor(grad_clip_norm)
    validator = Validator(_validate_sage_fn, patience=patience)

    history = []

    def _train_one_epoch(
        epoch: int,
        active_compounds: list[int],
        stage_optimizer: torch.optim.Optimizer,
        stage_memory_bank: MemoryBank,
        stage_epochs: int,
        stage_scaler: GradScaler = None,
        stage_use_residue_decoder: bool = True,
        stage_bpr_detach_neg: bool = True,
        cumulative_epoch_offset: int = 0,
        total_epochs: int = None,
    ) -> tuple[float, int]:
        model.train()
        total_loss = 0.0
        n_batches = 0

        shuffled_compounds = list(active_compounds)
        random.shuffle(shuffled_compounds)
        total_batches = (len(shuffled_compounds) + batch_size - 1) // batch_size
        for batch_start in range(0, len(shuffled_compounds), batch_size):
            if n_batches == 0 and batch_start % (batch_size * 50) == 0:
                logger.info(f"  SAGE epoch {epoch}: starting batch at compound {batch_start}/{len(shuffled_compounds)}")
            batch_seeds = shuffled_compounds[batch_start:batch_start + batch_size]

            node_list, node_to_local, edge_index = sample_homo_subgraph(
                batch_seeds, homo_adj, num_neighbors,
                seed=epoch * 10000 + batch_start)
            edge_index = edge_index.to(device)
            edge_index = drop_edge(edge_index, p=dropedge_ppi)

            sub_x = x[torch.tensor(node_list)].to(device).float()
            n_compounds_in_sub = sum(1 for n in node_list if n < n_compounds)
            with autocast('cuda', enabled=use_amp):
                node_emb = model(sub_x, edge_index, n_compounds=n_compounds_in_sub)

            if _check_tensor_nan(node_emb, "SAGE node_emb"):
                logger.warning("SAGE 前向传播产生 NaN 嵌入，跳过当前 batch")
                continue

            seed_local = []
            batch_idx_to_comp_idx = {}
            for bi, s in enumerate(batch_seeds):
                if s in node_to_local:
                    batch_idx_to_comp_idx[bi] = len(seed_local)
                    seed_local.append(node_to_local[s])
            if not seed_local:
                continue

            prot_local_indices = [i for i, n in enumerate(node_list) if n >= n_compounds]

            comp_emb = node_emb[torch.tensor(seed_local, device=device)]
            if not prot_local_indices:
                continue

            prot_emb = node_emb[torch.tensor(prot_local_indices, device=device)]
            n_batch_prots = len(prot_local_indices)

            local_to_prot_pos = {local_pos: i for i, local_pos in enumerate(prot_local_indices)}

            pos_src, pos_dst = [], []
            for bi, s in enumerate(batch_seeds):
                ci = batch_idx_to_comp_idx.get(bi)
                if ci is None or s not in precomputed_pos:
                    continue
                for p_global in precomputed_pos[s]:
                    if p_global in node_to_local:
                        local_pos = node_to_local[p_global]
                        if local_pos in local_to_prot_pos:
                            prot_pos = local_to_prot_pos[local_pos]
                            if 0 <= prot_pos < n_batch_prots:
                                pos_src.append(ci)
                                pos_dst.append(prot_pos)

            if not pos_src:
                continue

            pos_src_t = torch.tensor(pos_src, device=device)
            pos_dst_t = torch.tensor(pos_dst, device=device)

            prot_map = {}
            for j, local_pos in enumerate(prot_local_indices):
                n = node_list[local_pos]
                if n >= n_compounds:
                    prot_map[n - n_compounds] = j
            comp_sorted_batch = [s for s in batch_seeds if s in node_to_local]

            loss = _compute_cpi_loss_fn(
                model=model,
                comp_emb=comp_emb,
                prot_emb=prot_emb,
                pos_src=pos_src_t,
                pos_dst=pos_dst_t,
                comp_sorted=comp_sorted_batch,
                prot_map=prot_map,
                precomputed_pos=precomputed_pos,
                n_compounds=n_compounds,
                prot_to_path_neighbors=prot_to_path_neighbors,
                epoch=epoch + cumulative_epoch_offset,
                stage_epochs=total_epochs if total_epochs is not None else stage_epochs,
                memory_bank=stage_memory_bank,
                compound_to_prot_locals=compound_to_prot_locals,
                use_infonce=use_infonce,
                bpr_weight=bpr_weight if use_bpr else 0.0,
                use_curriculum=use_curriculum,
                use_topology_neg=use_topology_neg,
                prot_to_topo_medium_neighbors=prot_to_topo_medium_neighbors,
                prot_to_topo_hard_neighbors=prot_to_topo_hard_neighbors,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                use_residue_decoder=stage_use_residue_decoder,
                bpr_detach_neg=stage_bpr_detach_neg,
            )

            if use_pheno:
                pheno_local_indices = []
                pheno_batch_labels = []
                for global_idx in pheno_compound_indices:
                    if global_idx in node_to_local:
                        local_idx = node_to_local[global_idx]
                        if local_idx < n_compounds_in_sub:
                            pheno_local_indices.append(local_idx)
                            pheno_batch_labels.append(pheno_idx_to_label[global_idx])
                if len(pheno_local_indices) > 0:
                    pheno_emb = node_emb[torch.tensor(pheno_local_indices, device=device)]
                    pheno_logits = model.predict_phenotype(pheno_emb).squeeze(-1)
                    pheno_target = torch.tensor(pheno_batch_labels, dtype=torch.float32, device=device)
                    pheno_loss = bce_loss_fn(pheno_logits, pheno_target)
                    loss = loss + pheno_lambda * pheno_loss

            stage_optimizer.zero_grad()
            # 清缓存后再 backward，最大化可用显存
            torch.cuda.empty_cache()
            stage_scaler.scale(loss).backward()
            gradient_monitor.check_and_clip(model, scaler=stage_scaler, optimizer=stage_optimizer)
            torch.cuda.empty_cache()
            stage_scaler.step(stage_optimizer)
            stage_scaler.update()

            stage_memory_bank.update(prot_emb.detach())

            total_loss += loss.item()
            n_batches += 1

        return total_loss, n_batches

    # ============================================================
    # 阶段1 — 尾节点平衡子图预训练
    # ============================================================
    if two_stage and pretrain_epochs > 0:
        pretrain_compounds, tail_compounds = split_head_tail_nodes(
            train_compounds, compound_to_pos, head_ratio=head_ratio, lambda_hhi=lambda_hhi, seed=random_seed)
        n_head_kept = len(pretrain_compounds) - len(tail_compounds)
        logger.info(f"  两阶段预训练: tail={len(tail_compounds)}, head_kept={n_head_kept}, "
                    f"total_pretrain={len(pretrain_compounds)}")

        pretrain_lr_actual = pretrain_lr if pretrain_lr is not None else lr * pretrain_lr_multiplier
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr_actual, weight_decay=weight_decay, foreach=False)
        pretrain_scaler = GradScaler('cuda', enabled=use_amp)
        def pretrain_lr_lambda(e):
            return 1.0 - pretrain_lr_decay * (e / pretrain_epochs)
        pretrain_scheduler = torch.optim.lr_scheduler.LambdaLR(pretrain_optimizer, pretrain_lr_lambda)
        pretrain_memory_bank = MemoryBank(max_size=memory_bank_size, out_dim=model.out_dim, device=str(device))

        validator.reset()

        for epoch in range(1, pretrain_epochs + 1):
            total_loss, n_batches = _train_one_epoch(
                epoch, pretrain_compounds, pretrain_optimizer, pretrain_memory_bank, pretrain_epochs,
                stage_scaler=pretrain_scaler, stage_use_residue_decoder=False,
                stage_bpr_detach_neg=True,
                cumulative_epoch_offset=0, total_epochs=pretrain_epochs + epochs)
            if n_batches == 0:
                continue
            avg_loss = total_loss / n_batches

            if validator.should_validate(epoch, is_pretrain=True) and val_compounds:
                model.eval()
                with torch.no_grad(), autocast('cuda', enabled=use_amp):
                    val_metrics = _validate_sage_fn(
                        model, x, _homo_edge_index_val,
                        val_compounds, all_compound_to_pos, n_compounds)
                logger.info(
                    f"  SAGE pretrain epoch {epoch:3d} | loss={avg_loss:.4f} | "
                    f"val_auc={val_metrics['auc']:.4f} | val_aupr={val_metrics['aupr']:.4f}")
                if val_metrics["aupr"] > validator.best_val_aupr:
                    validator.best_val_aupr = val_metrics["aupr"]
                    validator.capture_best_state(model)
                model.train()
            elif epoch % 5 == 0:
                logger.info(f"  SAGE pretrain epoch {epoch:3d} | loss={avg_loss:.4f}")
            else:
                logger.info(f"  SAGE pretrain epoch {epoch:3d} | loss={avg_loss:.4f}")

            pretrain_scheduler.step()

        if validator.load_best_state(model):
            logger.info(f"  SAGE 预训练完成，加载最优 checkpoint (val_aupr={validator.best_val_aupr:.4f}) 进入微调阶段")
        else:
            logger.info("  SAGE 预训练完成，加载最终参数进入微调阶段")

    # 表型预训练阶段已移除，多任务联合训练已足够。
    # 消融实验如需启用，设置 pheno_pretrain_epochs > 0 即可恢复。

    # ============================================================
    # 阶段2 — 完整训练图微调
    # ============================================================
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = _train_one_epoch(
            epoch, train_compounds, optimizer, memory_bank, epochs, stage_scaler=scaler,
            cumulative_epoch_offset=pretrain_epochs, total_epochs=pretrain_epochs + epochs)

        if n_batches == 0:
            continue

        avg_loss = total_loss / n_batches

        if epoch % 2 == 0 and val_compounds:
            model.eval()
            val_metrics = _validate_sage_fn(model, x, _homo_edge_index_val, val_compounds, all_compound_to_pos,
                                         n_compounds)
            m = val_metrics

            pheno_auc = None
            if use_pheno:
                val_pheno_indices = [c for c in val_compounds if c in pheno_comp_set]
                if len(val_pheno_indices) > 5:
                    with torch.no_grad(), autocast('cuda', enabled=use_amp):
                        full_node_emb_val = model(x, _homo_edge_index_val,
                                                  n_compounds=n_compounds)
                        val_pheno_emb = full_node_emb_val[torch.tensor(val_pheno_indices, device=device)]
                        val_pheno_logits = model.predict_phenotype(val_pheno_emb).squeeze(-1)
                        val_pheno_labels = [pheno_idx_to_label[c] for c in val_pheno_indices]
                        val_pheno_labels_t = torch.tensor(val_pheno_labels, dtype=torch.float32, device=device)
                        try:
                            pheno_auc = roc_auc_score(val_pheno_labels_t.cpu().numpy(),
                                                      torch.sigmoid(val_pheno_logits).cpu().numpy())
                        except ValueError as e:
                            logger.warning(f"表型 AUC 计算退化（仅一类标签）: {e}，回退为 0.5")
                            pheno_auc = 0.5

            hist_entry = {"epoch": epoch, "loss": avg_loss, **m}
            if pheno_auc is not None:
                hist_entry["pheno_auc"] = pheno_auc
            history.append(hist_entry)

            log_str = (f"  SAGE epoch {epoch:3d} | loss={avg_loss:.4f} | val_auc={m['auc']:.4f} | val_aupr={m['aupr']:.4f}")
            if pheno_auc is not None:
                log_str += f" | pheno_auc={pheno_auc:.4f}"
            logger.info(log_str)

            # 早停基于 val_aupr（化合物冷启动）
            is_new_best = validator.update_best(m["aupr"], m["auc"])
            if is_new_best:
                validator.capture_best_state(model)

            if validator.should_stop_early():
                logger.info(f"  SAGE 早停 (epoch {epoch}, patience_counter={validator.patience_counter})")
                break

            if epoch % 5 == 0:
                memory_bank_mgr.refresh_global_sage(
                    model, x, _homo_edge_index_train, n_compounds,
                    val_proteins=val_proteins, use_amp=use_amp,
                )

            if use_plateau_scheduler:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(m["aupr"])
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    logger.info(f"  SAGE 微调 lr {prev_lr:.2e} -> {new_lr:.2e} (plateau, val_aupr={m['aupr']:.4f})")

    if not use_plateau_scheduler:
        scheduler.step()

    if validator.load_best_state(model):
        best_entry = validator.get_best_entry(history)
        logger.info(f"  SAGE best val_aupr={best_entry.get('aupr', 0):.4f}, val_auc={best_entry.get('auc', 0):.4f}")
        return model, history

    model = model.to("cpu")
    return model, history


def train_hgt(
    model: HGTLinkPredictor | RGCNLinkPredictor | nn.Module,
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 128,
    num_neighbors: list[int] = None,
    prot_to_path_neighbors: dict[int, set] | None = None,
    two_stage: bool = False,
    pretrain_epochs: int = 0,
    pretrain_lr: float | None = None,
    random_seed: int = 42,
    use_infonce: bool = False,
    use_bpr: bool = True,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    pheno_compound_indices: list[int] = None,
    pheno_labels: list[int] = None,
    pheno_lambda: float = 0.3,
    bpr_weight: float = 0.4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.05,
    dropedge_ppi: float = 0.15,
    dropedge_pathway: float = 0.10,
    dropedge_cpi: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    memory_bank_size: int = 8192,
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    head_undersample_ratio: float = 0.6,
    grad_clip_norm: float = 1.0,
    pretrain_lr_multiplier: float = 1.5,
    pretrain_lr_decay: float = 0.5,
    finetune_lr_multiplier: float = 0.5,
    use_plateau_scheduler: bool = False,
    plateau_patience: int = 2,
    plateau_factor: float = 0.5,
    _validate_hgt_fn=None,
    _compute_cpi_loss_fn=None,
    use_amp: bool = True,
) -> tuple[nn.Module, list[dict]]:
    """HGT/RGCN mini-batch 训练（适用于任何支持 x_dict/edge_index_dict 前向的模型）

    阶段1 (可选): 在尾节点平衡子图上预训练，学习稀疏靶标表示
    阶段2: 在完整训练图上微调
    """
    if _validate_hgt_fn is None or _compute_cpi_loss_fn is None:
        raise ValueError(
            "train_hgt 需要注入 _validate_hgt_fn, _compute_cpi_loss_fn 参数。"
        )

    if num_neighbors is None:
        num_neighbors = [32, 16]
    model = model.to(device)
    # v49-fix: 禁止在此处无条件重初始化模型参数（原因同 train_sage）。

    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            import importlib.util
            if importlib.util.find_spec("triton") is not None:
                model = torch.compile(model, mode='reduce-overhead')
                logger.info('HGT model compiled with torch.compile (reduce-overhead)')
            else:
                logger.info('Triton not available, skipping torch.compile for HGT')
        except Exception as e:
            logger.warning(f'torch.compile failed for HGT: {e}, continuing without compilation')

    hetero_adj = graphs.get("hetero_adj_train", graphs["hetero_adj"])
    hetero_data = graphs.get("hetero_data_train", graphs["hetero_data"]).to(device)
    n_compounds = graphs["n_compounds"]
    n_proteins = graphs["n_proteins"]
    n_pathways = graphs["n_pathways"]

    val_hetero = graphs.get("hetero_data_val")
    val_hetero_adj = graphs.get("hetero_adj_val", hetero_adj)

    logger.info(f"  HGT 通路嵌入: {max(n_pathways, 1)} 通路, dim={model.pathway_embed.embedding_dim}")

    all_compound_to_pos = compound_to_pos
    precomputed_pos = {src: sorted(pos_set) for src, pos_set in compound_to_pos.items() if pos_set}
    compound_to_prot_locals = {c: [p - n_compounds for p in pos_set] for c, pos_set in precomputed_pos.items()}

    prot_to_topo_medium_neighbors = graphs.get("prot_to_topo_medium_neighbors")
    prot_to_topo_hard_neighbors = graphs.get("prot_to_topo_hard_neighbors")

    finetune_lr_actual = lr * finetune_lr_multiplier if (two_stage and pretrain_epochs > 0) else lr
    logger.info(f"  HGT 微调初始学习率: {finetune_lr_actual:.2e} (lr={lr:.2e}, multiplier={finetune_lr_multiplier})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr_actual, weight_decay=weight_decay, foreach=False)
    if use_plateau_scheduler:
        scheduler = LRSchedulerFactory.create_plateau(
            optimizer, patience=plateau_patience, factor=plateau_factor, mode="max", metric_name="aupr"
        )
    else:
        scheduler = LRSchedulerFactory.create_cosine_warmup(optimizer, epochs, warmup_ratio)
    scaler = GradScaler('cuda', enabled=use_amp)

    memory_bank_mgr = MemoryBankManager(memory_bank_size, model.out_dim, str(device))
    memory_bank = memory_bank_mgr.memory_bank
    gradient_monitor = GradientMonitor(grad_clip_norm)
    validator = Validator(_validate_hgt_fn, patience=patience)

    history = []

    use_pheno = pheno_compound_indices is not None and pheno_labels is not None
    if use_pheno:
        pheno_idx_tensor = torch.tensor(pheno_compound_indices, device=device, dtype=torch.long)
        pheno_label_tensor = torch.tensor(pheno_labels, device=device, dtype=torch.float32)
        n_pos_p = sum(pheno_labels)
        n_neg_p = len(pheno_labels) - n_pos_p
        hgt_pos_weight = torch.tensor([n_neg_p / max(n_pos_p, 1)], device=device)
        pheno_bce = nn.BCEWithLogitsLoss(pos_weight=hgt_pos_weight)
        _, pheno_sort_pos = torch.sort(pheno_idx_tensor)
        pheno_sorted_idx = pheno_idx_tensor[pheno_sort_pos]
        pheno_sorted_labels = pheno_label_tensor[pheno_sort_pos]
        logger.info(f"  表型分类训练: {len(pheno_compound_indices)} 个化合物, lambda={pheno_lambda}")

    def _train_one_epoch(
        epoch: int,
        active_compounds: list[int],
        stage_optimizer: torch.optim.Optimizer,
        stage_memory_bank: MemoryBank,
        stage_epochs: int,
        stage_use_pheno: bool = False,
        stage_scaler: GradScaler = None,
        stage_use_residue_decoder: bool = True,
        stage_bpr_detach_neg: bool = True,
        cumulative_epoch_offset: int = 0,
        total_epochs: int = None,
    ) -> tuple[float, int]:
        model.train()
        total_loss = 0.0
        n_batches = 0

        shuffled_compounds = list(active_compounds)
        random.shuffle(shuffled_compounds)
        for batch_start in range(0, len(shuffled_compounds), batch_size):
            if n_batches == 0 and batch_start % (batch_size * 50) == 0:
                logger.info(f"  HGT epoch {epoch}: starting batch at compound {batch_start}/{len(shuffled_compounds)}")
            batch_seeds = shuffled_compounds[batch_start:batch_start + batch_size]

            sg, comp_sorted, prot_sorted, path_sorted, disease_sorted, comp_map, prot_map, disease_map = sample_hetero_subgraph(
                batch_seeds, hetero_adj, num_neighbors,
                seed=epoch * 10000 + batch_start)

            if not prot_sorted:
                continue

            torch.cuda.empty_cache()

            sg["compound"].x = hetero_data["compound"].x[torch.tensor(comp_sorted, device=device)]
            sg["protein"].x = hetero_data["protein"].x[torch.tensor(prot_sorted, device=device)]
            if path_sorted:
                path_global_tensor = torch.tensor(sg._path_global, device=device)
                path_global_tensor = torch.clamp(path_global_tensor, min=0,
                                                  max=model.pathway_embed.num_embeddings - 1)
                sg["pathway"].x = model.pathway_embed(path_global_tensor)
            else:
                sg["pathway"].x = torch.zeros(0, model.pathway_embed.embedding_dim, device=device)
            if disease_sorted:
                disease_global_tensor = torch.tensor(sg._disease_global, device=device).unsqueeze(-1)
                sg["disease"].x = disease_global_tensor
            else:
                sg["disease"].x = torch.zeros(0, 1, device=device)

            sg = sg.to(device)
            for et in list(sg.edge_index_dict.keys()):
                if "ppi" in str(et):
                    sg[et].edge_index = drop_edge(sg[et].edge_index, p=dropedge_ppi)
                elif "pathway" in str(et) or "belongs_to" in str(et) or "includes" in str(et):
                    sg[et].edge_index = drop_edge(sg[et].edge_index, p=dropedge_pathway)

            cpi_et = ("compound", "interacts", "protein")
            if cpi_et in sg.edge_index_dict and dropedge_cpi > 0:
                sg[cpi_et].edge_index = drop_edge(sg[cpi_et].edge_index, p=dropedge_cpi)

            stage_optimizer.zero_grad()

            with autocast('cuda', enabled=use_amp):
                hgt_out = model(sg.x_dict, sg.edge_index_dict)
            prot_emb = hgt_out["protein"]
            comp_emb = hgt_out["compound"]

            if torch.isnan(prot_emb).any() or torch.isnan(comp_emb).any():
                continue

            cpi_ei = sg[("compound", "interacts", "protein")].edge_index
            if cpi_ei.shape[1] < 1:
                continue

            pos_src = cpi_ei[0]
            pos_dst = cpi_ei[1]

            loss = _compute_cpi_loss_fn(
                model=model,
                comp_emb=comp_emb,
                prot_emb=prot_emb,
                pos_src=pos_src,
                pos_dst=pos_dst,
                comp_sorted=comp_sorted,
                prot_map=prot_map,
                precomputed_pos=precomputed_pos,
                n_compounds=n_compounds,
                prot_to_path_neighbors=prot_to_path_neighbors,
                epoch=epoch + cumulative_epoch_offset,
                stage_epochs=total_epochs if total_epochs is not None else stage_epochs,
                memory_bank=stage_memory_bank,
                compound_to_prot_locals=compound_to_prot_locals,
                use_infonce=use_infonce,
                bpr_weight=bpr_weight if use_bpr else 0.0,
                use_curriculum=use_curriculum,
                use_topology_neg=use_topology_neg,
                prot_to_topo_medium_neighbors=prot_to_topo_medium_neighbors,
                prot_to_topo_hard_neighbors=prot_to_topo_hard_neighbors,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                use_residue_decoder=stage_use_residue_decoder,
                bpr_detach_neg=stage_bpr_detach_neg,
            )

            if stage_use_pheno:
                comp_sorted_tensor = torch.tensor(comp_sorted, device=device, dtype=torch.long)
                pheno_mask = torch.isin(comp_sorted_tensor, pheno_idx_tensor)
                if pheno_mask.any():
                    pheno_comp_emb = comp_emb[pheno_mask]
                    pheno_global_indices = comp_sorted_tensor[pheno_mask]
                    matched_pos = torch.searchsorted(pheno_sorted_idx, pheno_global_indices)
                    pheno_label_batch = pheno_sorted_labels[matched_pos]
                    pheno_logits = model.predict_phenotype(pheno_comp_emb).squeeze(-1)
                    pheno_loss = pheno_bce(pheno_logits, pheno_label_batch)
                    loss = loss + pheno_lambda * pheno_loss

            # 清缓存后再 backward，最大化可用显存
            torch.cuda.empty_cache()
            stage_scaler.scale(loss).backward()
            gradient_monitor.check_and_clip(model, scaler=stage_scaler, optimizer=stage_optimizer)
            torch.cuda.empty_cache()
            stage_scaler.step(stage_optimizer)
            stage_scaler.update()

            stage_memory_bank.update(prot_emb.detach())

            total_loss += loss.item()
            n_batches += 1

        return total_loss, n_batches

    # ============================================================
    # 阶段1 — 尾节点平衡子图预训练 (HGT)
    # ============================================================
    if two_stage and pretrain_epochs > 0:
        pretrain_compounds, tail_compounds = split_head_tail_nodes(
            train_compounds, compound_to_pos, head_ratio=head_ratio, lambda_hhi=lambda_hhi, seed=random_seed)
        n_head_kept = len(pretrain_compounds) - len(tail_compounds)
        logger.info(f"  HGT 两阶段预训练: tail={len(tail_compounds)}, head_kept={n_head_kept}, "
                    f"total_pretrain={len(pretrain_compounds)}")

        pretrain_lr_actual = pretrain_lr if pretrain_lr is not None else lr * pretrain_lr_multiplier
        pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr_actual, weight_decay=weight_decay, foreach=False)
        pretrain_scaler = GradScaler('cuda', enabled=use_amp)
        def pretrain_lr_lambda(e):
            return 1.0 - pretrain_lr_decay * (e / pretrain_epochs)
        pretrain_scheduler = torch.optim.lr_scheduler.LambdaLR(pretrain_optimizer, pretrain_lr_lambda)
        pretrain_memory_bank = MemoryBank(max_size=memory_bank_size, out_dim=model.out_dim, device=str(device))

        validator.reset()

        for epoch in range(1, pretrain_epochs + 1):
            total_loss, n_batches = _train_one_epoch(
                epoch, pretrain_compounds, pretrain_optimizer, pretrain_memory_bank, pretrain_epochs,
                stage_scaler=pretrain_scaler, stage_use_residue_decoder=False,
                stage_bpr_detach_neg=True,
                cumulative_epoch_offset=0, total_epochs=pretrain_epochs + epochs)
            if n_batches == 0:
                continue
            avg_loss = total_loss / n_batches

            if validator.should_validate(epoch, is_pretrain=True) and val_compounds:
                model.eval()
                torch.cuda.empty_cache()
                with torch.no_grad(), autocast('cuda', enabled=use_amp):
                    hd = val_hetero if val_hetero is not None else hetero_data
                    val_metrics = _validate_hgt_fn(
                        model, hd, val_compounds, all_compound_to_pos,
                        n_compounds, n_proteins, hetero_adj=val_hetero_adj)
                logger.info(
                    f"  HGT pretrain epoch {epoch:3d} | loss={avg_loss:.4f} | "
                    f"val_auc={val_metrics['auc']:.4f} | val_aupr={val_metrics['aupr']:.4f}")
                if val_metrics["aupr"] > validator.best_val_aupr:
                    validator.best_val_aupr = val_metrics["aupr"]
                    validator.capture_best_state(model)
                model.train()
            elif epoch % 5 == 0:
                logger.info(f"  HGT pretrain epoch {epoch:3d} | loss={avg_loss:.4f}")

            pretrain_scheduler.step()

        if validator.load_best_state(model):
            logger.info(f"  HGT 预训练完成，加载最优 checkpoint (val_aupr={validator.best_val_aupr:.4f}) 进入微调阶段")
        else:
            logger.info("  HGT 预训练完成，加载最终参数进入微调阶段")

    # ============================================================
    # 阶段2 — 完整训练图微调
    # ============================================================
    for epoch in range(1, epochs + 1):
        total_loss, n_batches = _train_one_epoch(
            epoch, train_compounds, optimizer, memory_bank, epochs,
            stage_use_pheno=use_pheno, stage_scaler=scaler,
            cumulative_epoch_offset=pretrain_epochs, total_epochs=pretrain_epochs + epochs)

        if n_batches == 0:
            continue

        avg_loss = total_loss / n_batches

        if epoch % 2 == 0 and val_compounds:
            torch.cuda.empty_cache()
            hd = val_hetero if val_hetero is not None else hetero_data
            with autocast('cuda', enabled=use_amp):
                val_metrics = _validate_hgt_fn(
                model, hd, val_compounds,
                all_compound_to_pos, n_compounds, n_proteins,
                hetero_adj=val_hetero_adj)
            val_auc = val_metrics["auc"]
            val_aupr = val_metrics["aupr"]

            is_new_best = validator.update_best(val_aupr, val_auc)
            if is_new_best:
                validator.capture_best_state(model)

            history.append({"epoch": epoch, "loss": avg_loss, "auc": val_auc, "aupr": val_aupr})
            logger.info(f"  HGT epoch {epoch:3d} | loss={avg_loss:.4f} | val_auc={val_auc:.4f} | val_aupr={val_aupr:.4f}")

            if validator.should_stop_early():
                logger.info(f"  HGT 早停 (epoch {epoch}, patience_counter={validator.patience_counter})")
                break

            if memory_bank_mgr.should_refresh(epoch):
                hetero_data_dev = hetero_data.to(device)
                n_path = hetero_data_dev["pathway"].n_pathways
                hetero_data_dev["pathway"].x = model.pathway_embed(
                    torch.arange(max(n_path, 1), device=device))
                memory_bank_mgr.refresh_global_hgt(
                    model, hetero_data_dev,
                    val_proteins=val_proteins, use_amp=use_amp,
                )

            if use_plateau_scheduler:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_aupr)
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    logger.info(f"  HGT 微调 lr {prev_lr:.2e} -> {new_lr:.2e} (plateau, val_aupr={val_aupr:.4f})")

    if not use_plateau_scheduler:
        scheduler.step()

    if validator.load_best_state(model):
        best_entry = validator.get_best_entry(history)
        logger.info(f"  HGT best val_aupr={best_entry.get('aupr', 0):.4f}, val_auc={best_entry.get('auc', 0):.4f}")
        return model, history

    model = model.to("cpu")
    return model, history


def train_rgcn(
    model: RGCNLinkPredictor | nn.Module,
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 128,
    num_neighbors: list[int] = None,
    prot_to_path_neighbors: dict[int, set] | None = None,
    two_stage: bool = False,
    pretrain_epochs: int = 0,
    pretrain_lr: float | None = None,
    random_seed: int = 42,
    use_infonce: bool = False,
    use_bpr: bool = True,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    pheno_compound_indices: list[int] = None,
    pheno_labels: list[int] = None,
    pheno_lambda: float = 0.3,
    bpr_weight: float = 0.4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.05,
    dropedge_ppi: float = 0.15,
    dropedge_pathway: float = 0.10,
    dropedge_cpi: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    memory_bank_size: int = 8192,
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    head_undersample_ratio: float = 0.6,
    grad_clip_norm: float = 1.0,
    pretrain_lr_multiplier: float = 1.5,
    pretrain_lr_decay: float = 0.5,
    finetune_lr_multiplier: float = 0.5,
    use_plateau_scheduler: bool = False,
    plateau_patience: int = 2,
    plateau_factor: float = 0.5,
    _validate_rgcn_fn=None,
    _compute_cpi_loss_fn=None,
    use_amp: bool = True,
) -> tuple[RGCNLinkPredictor, list[dict]]:
    """RGCN mini-batch 训练（接口与 train_hgt 完全一致）

    将 train_hgt 的验证钩子函数 _validate_hgt_fn 映射为 _validate_rgcn_fn，
    其余参数与逻辑完全复用 train_hgt。

    阶段1 (可选): 在尾节点平衡子图上预训练，学习稀疏靶标表示
    阶段2: 在完整训练图上微调
    """
    if _validate_rgcn_fn is None or _compute_cpi_loss_fn is None:
        raise ValueError(
            "train_rgcn 需要注入 _validate_rgcn_fn, _compute_cpi_loss_fn 参数。"
        )

    # 将 _validate_rgcn_fn 映射为 _validate_hgt_fn 参数，复用 train_hgt
    return train_hgt(
        model=model, graphs=graphs,
        train_compounds=train_compounds, val_compounds=val_compounds,
        compound_to_pos=compound_to_pos, device=device,
        val_proteins=val_proteins, epochs=epochs, lr=lr, patience=patience,
        batch_size=batch_size, num_neighbors=num_neighbors,
        prot_to_path_neighbors=prot_to_path_neighbors,
        two_stage=two_stage, pretrain_epochs=pretrain_epochs,
        pretrain_lr=pretrain_lr, random_seed=random_seed,
        use_infonce=use_infonce, use_bpr=use_bpr,
        use_curriculum=use_curriculum, use_topology_neg=use_topology_neg,
        pheno_compound_indices=pheno_compound_indices,
        pheno_labels=pheno_labels, pheno_lambda=pheno_lambda,
        bpr_weight=bpr_weight, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        dropedge_ppi=dropedge_ppi, dropedge_pathway=dropedge_pathway,
        dropedge_cpi=dropedge_cpi,
        focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        memory_bank_size=memory_bank_size,
        head_ratio=head_ratio, lambda_hhi=lambda_hhi,
        head_undersample_ratio=head_undersample_ratio,
        grad_clip_norm=grad_clip_norm,
        pretrain_lr_multiplier=pretrain_lr_multiplier,
        pretrain_lr_decay=pretrain_lr_decay,
        finetune_lr_multiplier=finetune_lr_multiplier,
        use_plateau_scheduler=use_plateau_scheduler,
        plateau_patience=plateau_patience,
        plateau_factor=plateau_factor,
        _validate_hgt_fn=_validate_rgcn_fn,
        _compute_cpi_loss_fn=_compute_cpi_loss_fn,
        use_amp=use_amp,
    )


def train_simplehgn(
    model: SimpleHGNLinkPredictor | nn.Module,
    graphs: dict,
    train_compounds: list[int],
    val_compounds: list[int],
    compound_to_pos: dict[int, set],
    device: torch.device,
    val_proteins: set = None,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    batch_size: int = 128,
    num_neighbors: list[int] = None,
    prot_to_path_neighbors: dict[int, set] | None = None,
    two_stage: bool = False,
    pretrain_epochs: int = 0,
    pretrain_lr: float | None = None,
    random_seed: int = 42,
    use_infonce: bool = False,
    use_bpr: bool = True,
    use_curriculum: bool = True,
    use_topology_neg: bool = False,
    pheno_compound_indices: list[int] = None,
    pheno_labels: list[int] = None,
    pheno_lambda: float = 0.3,
    bpr_weight: float = 0.4,
    weight_decay: float = 1e-4,
    warmup_ratio: float = 0.05,
    dropedge_ppi: float = 0.15,
    dropedge_pathway: float = 0.10,
    dropedge_cpi: float = 0.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    memory_bank_size: int = 8192,
    head_ratio: float = 0.2,
    lambda_hhi: float = 1.0,
    head_undersample_ratio: float = 0.6,
    grad_clip_norm: float = 1.0,
    pretrain_lr_multiplier: float = 1.5,
    pretrain_lr_decay: float = 0.5,
    finetune_lr_multiplier: float = 0.5,
    use_plateau_scheduler: bool = False,
    plateau_patience: int = 2,
    plateau_factor: float = 0.5,
    _validate_simplehgn_fn=None,
    _compute_cpi_loss_fn=None,
    use_amp: bool = True,
) -> tuple[SimpleHGNLinkPredictor, list[dict]]:
    """SimpleHGN mini-batch 训练（接口与 train_hgt 完全一致）

    将 train_hgt 的验证钩子函数 _validate_hgt_fn 映射为 _validate_simplehgn_fn，
    其余参数与逻辑完全复用 train_hgt。

    阶段1 (可选): 在尾节点平衡子图上预训练，学习稀疏靶标表示
    阶段2: 在完整训练图上微调
    """
    if _validate_simplehgn_fn is None or _compute_cpi_loss_fn is None:
        raise ValueError(
            "train_simplehgn 需要注入 _validate_simplehgn_fn, _compute_cpi_loss_fn 参数。"
        )

    return train_hgt(
        model=model, graphs=graphs,
        train_compounds=train_compounds, val_compounds=val_compounds,
        compound_to_pos=compound_to_pos, device=device,
        val_proteins=val_proteins, epochs=epochs, lr=lr, patience=patience,
        batch_size=batch_size, num_neighbors=num_neighbors,
        prot_to_path_neighbors=prot_to_path_neighbors,
        two_stage=two_stage, pretrain_epochs=pretrain_epochs,
        pretrain_lr=pretrain_lr, random_seed=random_seed,
        use_infonce=use_infonce, use_bpr=use_bpr,
        use_curriculum=use_curriculum, use_topology_neg=use_topology_neg,
        pheno_compound_indices=pheno_compound_indices,
        pheno_labels=pheno_labels, pheno_lambda=pheno_lambda,
        bpr_weight=bpr_weight, weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        dropedge_ppi=dropedge_ppi, dropedge_pathway=dropedge_pathway,
        dropedge_cpi=dropedge_cpi,
        focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        memory_bank_size=memory_bank_size,
        head_ratio=head_ratio, lambda_hhi=lambda_hhi,
        head_undersample_ratio=head_undersample_ratio,
        grad_clip_norm=grad_clip_norm,
        pretrain_lr_multiplier=pretrain_lr_multiplier,
        pretrain_lr_decay=pretrain_lr_decay,
        finetune_lr_multiplier=finetune_lr_multiplier,
        use_plateau_scheduler=use_plateau_scheduler,
        plateau_patience=plateau_patience,
        plateau_factor=plateau_factor,
        _validate_hgt_fn=_validate_simplehgn_fn,
        _compute_cpi_loss_fn=_compute_cpi_loss_fn,
        use_amp=use_amp,
    )


__all__ = [
    "train_sage",
    "train_hgt",
    "train_rgcn",
    "train_simplehgn",
    "TrainingConfig",
    "Validator",
    "MemoryBankManager",
    "GradientMonitor",
    "LRSchedulerFactory",
]
