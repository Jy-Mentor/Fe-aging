#!/usr/bin/env python3
"""
Phase 3 - 3D构象批量生成（独立脚本）
======================================
使用 ETKDGv3 + MMFF94 为候选化合物生成3D构象
输出合并 SDF 文件供 Phase 5 分子对接使用

注意：RDKit SDWriter 不支持中文路径，因此使用 D:/temp_conformers/ 作为临时目录
"""

import os
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
L3_ROOT = PROJECT_ROOT / "L3"
L3_RESULTS = L3_ROOT / "results"
COMPOUND_POOL = L3_RESULTS / "tcm_compound_pool_filtered.csv"
TEMP_DIR = Path("D:/temp_conformers")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SDF = L3_RESULTS / "conformers" / "tcm_compounds_3d.sdf"
OUTPUT_CSV = L3_RESULTS / "conformers" / "conformer_status.csv"
OUTPUT_SDF.parent.mkdir(parents=True, exist_ok=True)

LOG_FILE = L3_ROOT / "logs" / "phase3_conformers.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_conformer(smiles, mol_id, max_confs=10):
    """
    为单个化合物生成3D构象
    策略：在不含H的分子上生成构象 → 加H → MMFF94优化 → 去H → 保存
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        mol_h = Chem.AddHs(mol)

        # ETKDGv3 构象生成
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        params.maxIterations = 5000
        params.randomSeed = 42

        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=max_confs, params=params)
        if len(conf_ids) == 0:
            return None, "Embedding failed"

        # MMFF94 优化
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=0, maxIters=500)
        energies = [(i, energy) for i, (converged, energy) in enumerate(results) if converged == 0]
        if len(energies) == 0:
            return None, "MMFF94 optimization failed"

        # 选能量最低
        energies.sort(key=lambda x: x[1])
        best_idx, best_energy = energies[0]

        # 构建最终分子
        mol_out = Chem.Mol(mol_h)
        mol_out.RemoveAllConformers()
        mol_out.AddConformer(mol_h.GetConformer(best_idx), assignId=True)
        mol_out = Chem.RemoveHs(mol_out)

        mol_out.SetProp("_Name", str(mol_id))
        mol_out.SetProp("MOL_ID", str(mol_id))
        mol_out.SetProp("Energy_MMFF94", f"{best_energy:.2f}")

        return mol_out, f"OK (E={best_energy:.2f})"
    except Exception as e:
        return None, f"Error: {str(e)[:80]}"


def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Phase 3 - 3D构象批量生成")
    logger.info(f"启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 加载候选化合物
    if not COMPOUND_POOL.exists():
        logger.error(f"候选化合物池不存在: {COMPOUND_POOL}")
        return False

    df = pd.read_csv(COMPOUND_POOL)
    logger.info(f"候选化合物: {len(df)} 个")

    # 只生成 Top 200 高 OB 化合物的构象（足够 Phase 5 对接使用）
    df = df.nlargest(200, "ob")
    logger.info(f"聚焦 Top 200 高OB化合物: OB范围 {df['ob'].min():.1f} - {df['ob'].max():.1f}%")

    # 批量生成（分批写入SDF）
    batch_size = 10
    status_records = []
    success_count = 0
    fail_count = 0

    # 分批处理，每个批次写入一个临时SDF，最后合并
    temp_sdfs = []

    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch = df.iloc[batch_start:batch_end]

        temp_sdf = TEMP_DIR / f"batch_{batch_start:04d}.sdf"
        writer = Chem.SDWriter(str(temp_sdf))

        batch_success = 0
        for _, row in batch.iterrows():
            mol_id = row["MOL_ID"]
            smiles = row["SMILES_std"]
            mol_out, status = generate_conformer(smiles, mol_id, max_confs=5)

            if mol_out is not None:
                try:
                    writer.write(mol_out)
                    batch_success += 1
                    success_count += 1
                except Exception as e:
                    status = f"Write error: {str(e)[:50]}"
                    fail_count += 1
            else:
                fail_count += 1

            status_records.append({
                "MOL_ID": mol_id,
                "molecule_name": row.get("molecule_name", ""),
                "status": status,
                "success": mol_out is not None
            })

        writer.close()

        if batch_success > 0:
            temp_sdfs.append(temp_sdf)
            logger.info(f"  批次 {batch_start}-{batch_end}: {batch_success}/{len(batch)} 成功")

        # 如果没有成功，删除空文件
        if batch_success == 0 and temp_sdf.exists():
            temp_sdf.unlink()

    # 合并所有临时SDF
    if len(temp_sdfs) > 0:
        logger.info(f"合并 {len(temp_sdfs)} 个批次SDF...")
        with open(OUTPUT_SDF, "wb") as out_f:
            for temp_sdf in temp_sdfs:
                with open(temp_sdf, "rb") as in_f:
                    out_f.write(in_f.read())
                temp_sdf.unlink()  # 清理临时文件
        logger.info(f"  合并SDF: {OUTPUT_SDF}")
        logger.info(f"  文件大小: {OUTPUT_SDF.stat().st_size / 1024:.1f} KB")
    else:
        logger.warning("无成功构象，未生成SDF")

    # 保存状态
    status_df = pd.DataFrame(status_records)
    status_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"  状态文件: {OUTPUT_CSV}")

    # 清理临时目录
    for f in TEMP_DIR.glob("batch_*.sdf"):
        f.unlink()
    try:
        TEMP_DIR.rmdir()
    except OSError:
        pass

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"3D构象生成完成!")
    logger.info(f"  成功: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)")
    logger.info(f"  失败: {fail_count}/{len(df)}")
    logger.info(f"  耗时: {elapsed/60:.1f} 分钟")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"未捕获异常: {e}")
        traceback.print_exc()
        sys.exit(1)