#!/usr/bin/env python3
"""提取铁衰老项目蛋白序列的残基级 ESM-2 嵌入。

输入: L2/results/target_protein_features.csv (gene_symbol, sequence)
输出:
  - L4/results_v10_minibatch/esm2_residue_embeddings_v31.npz
      每个 key 为基因名, value 为 (seq_len, esm_dim) 的 float32 数组
  - L4/results_v10_minibatch/esm2_residue_embeddings_padded_v31.npz
      embeddings: (N, max_len, esm_dim) padded 矩阵
      mask:       (N, max_len) bool 有效位掩码
      genes:      (N,) 基因名数组
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "L4" / "src"))

from iron_aging_gnn.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_residue_esm2")


def clean_sequence(seq: str) -> str:
    """将非标准氨基酸替换为 X, 以兼容 ESM-2 tokenizer。"""
    return seq.strip().upper().replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")


def load_sequences(csv_path: Path) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    gene_to_seq: dict[str, str] = {}
    for _, row in df.iterrows():
        gene = str(row["gene_symbol"]).strip().upper()
        seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
        gene_to_seq[gene] = clean_sequence(seq)
    return gene_to_seq


def main() -> None:
    cfg = Config()
    paths = cfg.get_resolved_paths()
    csv_path = paths.l2_results / "target_protein_features.csv"
    out_dir = paths.l4_results
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "esm2_residue_embeddings_v31.npz"
    out_padded = out_dir / "esm2_residue_embeddings_padded_v31.npz"

    if not csv_path.exists():
        raise FileNotFoundError(f"蛋白序列文件不存在: {csv_path}")

    gene_to_seq = load_sequences(csv_path)
    logger.info(f"读取蛋白序列: {len(gene_to_seq)} 条, 来自 {csv_path}")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import EsmModel, EsmTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg.esm2.model_name
    max_len = cfg.esm2.esm_max_len
    batch_size = cfg.esm2.esm_batch_size

    logger.info(f"加载 ESM-2 模型: {model_name} ...")
    tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    model = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    esm_dim = model.config.hidden_size
    logger.info(f"模型维度: {esm_dim}, 设备: {device}")

    genes = sorted(gene_to_seq.keys(), key=lambda g: len(gene_to_seq[g]), reverse=True)
    emb_dict: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i + batch_size]
            batch_seqs = [gene_to_seq[g] for g in batch_genes]
            truncated = [s[:max_len] for s in batch_seqs]

            inputs = tokenizer(
                truncated,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            attn = inputs["attention_mask"]

            for j, g in enumerate(batch_genes):
                seq_len_with_special = int(attn[j].sum().item())
                res_len = max(0, seq_len_with_special - 2)
                if res_len == 0:
                    emb = np.zeros((0, esm_dim), dtype=np.float32)
                else:
                    emb = hidden[j, 1:1 + res_len].cpu().numpy().astype(np.float32)
                emb_dict[g] = emb

            logger.info(
                f"ESM-2 进度: {min(i + batch_size, len(genes))}/{len(genes)}"
            )

    np.savez_compressed(out_npz, **emb_dict)
    logger.info(f"已保存变长残基嵌入: {out_npz} ({len(emb_dict)} 个蛋白)")

    max_l = max(e.shape[0] for e in emb_dict.values()) if emb_dict else 0
    n = len(emb_dict)
    padded = np.zeros((n, max_l, esm_dim), dtype=np.float32)
    mask = np.zeros((n, max_l), dtype=bool)
    gene_arr = np.empty(n, dtype=object)
    for idx, (g, e) in enumerate(emb_dict.items()):
        seq_len = e.shape[0]
        if seq_len > 0:
            padded[idx, :seq_len] = e
            mask[idx, :seq_len] = True
        gene_arr[idx] = g

    np.savez_compressed(out_padded, embeddings=padded, mask=mask, genes=gene_arr)
    logger.info(f"已保存 padded 残基嵌入: {out_padded}, shape={padded.shape}")

    prot_cache = paths.l4_results / "esm2_protein_embeddings.npz"
    if prot_cache.exists():
        prot = np.load(prot_cache, allow_pickle=True)
        sims = []
        for g in emb_dict:
            if g in prot:
                mean_res = emb_dict[g].mean(axis=0)
                p = prot[g].astype(np.float32)
                norm = np.linalg.norm(mean_res) * np.linalg.norm(p)
                sims.append(np.dot(mean_res, p) / (norm + 1e-12))
        if sims:
            logger.info(
                f"残基均值 vs 现有蛋白级嵌入的平均余弦相似度: {float(np.mean(sims)):.4f} (N={len(sims)})"
            )

    del model, tokenizer
    torch.cuda.empty_cache()
    logger.info("ESM-2 模型已释放 GPU 内存")


if __name__ == "__main__":
    main()
