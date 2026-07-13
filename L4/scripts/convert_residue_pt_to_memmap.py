#!/usr/bin/env python3
"""将 esm2_150M_residue_features.pt 转换为 numpy memmap 格式。

Windows + PyTorch 2.11 + cu128 下 torch.load(..., mmap=True) 会触发不可捕获的
ACCESS_VIOLATION；非 mmap 加载 8.86GB 又超出可用 RAM。numpy memmap 在 Windows
上稳定且按需分页，因此提供此转换脚本作为一劳永逸的存储格式。

输入:
  L4/results_v10_minibatch/esm2_150M_residue_features.pt
输出:
  L4/results_v10_minibatch/esm2_150M_residue_features.memmap  (float32, Nx640)
  L4/results_v10_minibatch/esm2_150M_residue_features_offsets.npy
  L4/results_v10_minibatch/esm2_150M_residue_features_lengths.npy
  L4/results_v10_minibatch/esm2_150M_residue_features_genes.npy
  L4/results_v10_minibatch/esm2_150M_residue_features_meta.json
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_residue_pt_to_memmap")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "L4" / "results_v10_minibatch"
INPUT_PT = RESULTS_DIR / "esm2_150M_residue_features.pt"
OUTPUT_MEMMAP = RESULTS_DIR / "esm2_150M_residue_features.memmap"
OUTPUT_OFFSETS = RESULTS_DIR / "esm2_150M_residue_features_offsets.npy"
OUTPUT_LENGTHS = RESULTS_DIR / "esm2_150M_residue_features_lengths.npy"
OUTPUT_GENES = RESULTS_DIR / "esm2_150M_residue_features_genes.npy"
OUTPUT_META = RESULTS_DIR / "esm2_150M_residue_features_meta.json"


def main() -> int:
    if not INPUT_PT.exists():
        logger.error(f"输入文件不存在: {INPUT_PT}")
        return 1

    if OUTPUT_MEMMAP.exists():
        logger.info(f"memmap 文件已存在: {OUTPUT_MEMMAP}，跳过转换")
        return 0

    logger.info(f"加载: {INPUT_PT} (大小 {INPUT_PT.stat().st_size / 1e9:.2f} GB)")
    # 非 mmap 加载；转换脚本独立运行，避免与其他大对象竞争 RAM。
    data = torch.load(INPUT_PT, map_location="cpu", mmap=False, weights_only=False)

    genes = data.get("genes", [])
    embeddings = data["embeddings"]
    offsets = data["offsets"]
    lengths = data["lengths"]

    n_proteins = len(genes)
    total_residues, dim = embeddings.shape
    logger.info(f"  蛋白数={n_proteins}, total_residues={total_residues}, dim={dim}")
    logger.info(f"  embeddings dtype={embeddings.dtype}, offsets dtype={offsets.dtype}")

    # 写入 memmap
    logger.info(f"写入 memmap: {OUTPUT_MEMMAP}")
    memmap_arr = np.memmap(
        OUTPUT_MEMMAP, dtype=np.float32, mode="w+", shape=(total_residues, dim)
    )
    # 分块复制，避免一次性创建大中间张量
    chunk_size = 100000
    for start in range(0, total_residues, chunk_size):
        end = min(start + chunk_size, total_residues)
        memmap_arr[start:end] = embeddings[start:end].numpy()
        if start % (chunk_size * 5) == 0:
            logger.info(f"  复制进度: {end}/{total_residues}")
    memmap_arr.flush()

    np.save(OUTPUT_OFFSETS, offsets.numpy().astype(np.int64))
    np.save(OUTPUT_LENGTHS, lengths.numpy().astype(np.int64))
    np.save(OUTPUT_GENES, np.array(genes, dtype=object))

    meta = {
        "n_proteins": n_proteins,
        "total_residues": int(total_residues),
        "dim": int(dim),
        "missing_genes": data.get("missing_genes", []),
        "esm_model_name": data.get("esm_model_name", ""),
        "n_requested": data.get("n_requested", n_proteins),
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("转换完成:")
    logger.info(f"  memmap: {OUTPUT_MEMMAP} ({OUTPUT_MEMMAP.stat().st_size / 1e9:.2f} GB)")
    logger.info(f"  offsets: {OUTPUT_OFFSETS}")
    logger.info(f"  lengths: {OUTPUT_LENGTHS}")
    logger.info(f"  genes: {OUTPUT_GENES}")
    logger.info(f"  meta: {OUTPUT_META}")

    # 验证读取
    logger.info("验证 memmap 可读性...")
    verify = np.memmap(OUTPUT_MEMMAP, dtype=np.float32, mode="r", shape=(total_residues, dim))
    sample = verify[:1024]
    logger.info(f"  前 1024 行形状={sample.shape}, mean={sample.mean():.4f}, std={sample.std():.4f}")

    del data, embeddings, offsets, lengths, memmap_arr, verify
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
