"""预计算残基级 ESM-2 特征的统计池化表示，避免每次加载 8.86GB 完整张量。

生成：
  - esm2_residue_stats_mean_std_max.npz
    * mean: (n_proteins, 640)
    * std:  (n_proteins, 640)
    * maxv: (n_proteins, 640)
    * genes: list[str]
    * dim_names: ['mean', 'std', 'maxv']
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def precompute(
    src_path: Path = Path("L4/results_v10_minibatch/esm2_150M_residue_features.pt"),
    dst_path: Path = Path("L4/results_v10_minibatch/esm2_residue_stats_mean_std_max.npz"),
    batch_proteins: int = 64,
) -> None:
    logger.info(f"加载索引: {src_path}")
    data = torch.load(src_path, map_location="cpu", mmap=True, weights_only=False)
    embeddings = data["embeddings"]  # (total_residues, 640)
    offsets = data["offsets"]        # (n_proteins+1,)
    lengths = data["lengths"]        # (n_proteins,)
    genes = data.get("genes", [])
    n_proteins = len(genes)
    dim = embeddings.shape[1]

    logger.info(f"  蛋白数={n_proteins}, 总残基数={embeddings.shape[0]}, dim={dim}")

    mean_arr = np.zeros((n_proteins, dim), dtype=np.float32)
    std_arr = np.zeros((n_proteins, dim), dtype=np.float32)
    max_arr = np.zeros((n_proteins, dim), dtype=np.float32)

    for start in range(0, n_proteins, batch_proteins):
        end = min(start + batch_proteins, n_proteins)
        # 按蛋白切片：逐蛋白收集残基
        for i in range(start, end):
            off_s = int(offsets[i].item())
            off_e = int(offsets[i + 1].item())
            if off_e <= off_s:
                continue
            emb = embeddings[off_s:off_e].float().numpy()
            mean_arr[i] = emb.mean(axis=0)
            std_arr[i] = emb.std(axis=0)
            max_arr[i] = emb.max(axis=0)
        logger.info(f"  已完成 {end}/{n_proteins} 蛋白")

    np.savez_compressed(
        dst_path,
        mean=mean_arr,
        std=std_arr,
        maxv=max_arr,
        genes=np.array(genes, dtype=object),
    )
    logger.info(f"已保存: {dst_path} (mean/std/max 各 {dim}D)")


if __name__ == "__main__":
    precompute()
