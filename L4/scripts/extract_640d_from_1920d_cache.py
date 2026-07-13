#!/usr/bin/env python3
"""从 1920D mean/max/std 缓存中提取 640D mean 分量，生成标准 ESM-2 嵌入文件。

动机:
  - esm2_residue_mean_max_std_cache.npz (6846蛋白 × 1920D) 是 mean/max/std 拼接，
    其中 mean 分量 (前 640D) 即为 ESM-2 残基层均值池化嵌入。
  - esm2_protein_embeddings.npz 目前仅含 103 蛋白，远少于 6846。
  - 提取 640D mean 分量可覆盖全部 6846 蛋白，同时避免 1920D 的冗余和噪声问题。

输出:
  - L4/results_v10_minibatch/esm2_protein_embeddings.npz (覆盖写入, 6846蛋白 × 640D)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_640d")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
L4_RESULTS = PROJECT_ROOT / "L4" / "results_v10_minibatch"

CACHE_1920D = L4_RESULTS / "esm2_residue_mean_max_std_cache.npz"
OUTPUT_NPZ = L4_RESULTS / "esm2_protein_embeddings.npz"
BACKUP_NPZ = L4_RESULTS / "esm2_protein_embeddings_backup_103.npz"


def main() -> None:
    if not CACHE_1920D.exists():
        logger.error(f"1920D 缓存文件不存在: {CACHE_1920D}")
        sys.exit(1)

    logger.info(f"加载 1920D 缓存: {CACHE_1920D}")
    data_1920d = np.load(CACHE_1920D, allow_pickle=True)

    n_total = len(data_1920d.files)
    result: dict[str, np.ndarray] = {}

    for gene in data_1920d.files:
        vec = data_1920d[gene]
        if hasattr(vec, "shape") and len(vec.shape) == 1 and vec.shape[0] == 1920:
            result[gene] = vec[:640].astype(np.float32)
        else:
            logger.warning(f"跳过非标准向量: {gene}, shape={getattr(vec, 'shape', 'N/A')}")

    n_extracted = len(result)
    logger.info(f"提取完成: {n_extracted}/{n_total} 蛋白, dim=640")

    if n_extracted == 0:
        logger.error("未提取到任何有效向量")
        sys.exit(1)

    # 备份旧文件
    if OUTPUT_NPZ.exists():
        logger.info(f"备份旧嵌入文件: {OUTPUT_NPZ} → {BACKUP_NPZ}")
        import shutil
        shutil.copy2(OUTPUT_NPZ, BACKUP_NPZ)

    np.savez_compressed(OUTPUT_NPZ, **result)
    logger.info(f"已保存: {OUTPUT_NPZ}")
    logger.info(f"文件大小: {OUTPUT_NPZ.stat().st_size / 1e6:.2f} MB")

    # 验证
    verify = np.load(OUTPUT_NPZ, allow_pickle=True)
    verify_dim = None
    for k in verify.files:
        v = verify[k]
        if hasattr(v, "shape"):
            verify_dim = v.shape[0]
            break
    logger.info(f"验证: {len(verify.files)} 蛋白, dim={verify_dim}")


if __name__ == "__main__":
    main()