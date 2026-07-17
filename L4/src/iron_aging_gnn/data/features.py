"""特征工程模块

化合物特征：ECFP4 指纹 + MACCS 密钥 + RDKit 分子描述符
蛋白特征：AAC 氨基酸组成 + ESM-2 预训练嵌入（可选）
缓存机制：TTL 过期清理、命中率监控、资源占用控制
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ..utils.config import Config
from .constants import RDKIT_DESCRIPTOR_NAMES

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


@dataclass
class CompoundFeatureConfig:
    """化合物特征工程配置。"""

    ecfp4_nbits: int = 2048
    ecfp4_radius: int = 2
    use_maccs: bool = True
    use_rdkit_descriptors: bool = True
    rdkit_descriptor_names: list[str] = field(default_factory=lambda: list(RDKIT_DESCRIPTOR_NAMES))
    enable_cache: bool = True
    cache_version: str = "v1"
    use_3d_conformer: bool = True

    def __post_init__(self):
        if self.ecfp4_nbits <= 0:
            raise ValueError(f"ecfp4_nbits 必须为正整数，当前: {self.ecfp4_nbits}")
        if self.ecfp4_radius < 1:
            raise ValueError(f"ecfp4_radius 必须 >= 1，当前: {self.ecfp4_radius}")
        if not self.rdkit_descriptor_names and self.use_rdkit_descriptors:
            logger.warning("rdkit_descriptor_names 为空，但 use_rdkit_descriptors=True，将不产生描述符特征")


@dataclass
class FeatureCacheKey:
    """特征缓存键，用于唯一标识一组特征计算输入。"""

    feature_type: str
    smiles_list: tuple[str, ...]
    config: CompoundFeatureConfig

    def to_string(self) -> str:
        config_dict = {
            "feature_type": self.feature_type,
            "ecfp4_nbits": self.config.ecfp4_nbits,
            "ecfp4_radius": self.config.ecfp4_radius,
            "use_maccs": self.config.use_maccs,
            "use_rdkit_descriptors": self.config.use_rdkit_descriptors,
            "rdkit_descriptor_names": self.config.rdkit_descriptor_names,
            "cache_version": self.config.cache_version,
        }
        config_str = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
        smiles_hash = hashlib.sha256(
            json.dumps(self.smiles_list, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.feature_type}_{self.config.cache_version}_{smiles_hash}_{hashlib.sha256(config_str.encode()).hexdigest()[:8]}"


@dataclass
class CacheStats:
    """缓存统计信息，用于命中率监控。"""

    hit_count: int = 0
    miss_count: int = 0
    total_files: int = 0
    total_size_bytes: int = 0
    expired_removed: int = 0
    size_limited_removed: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / _MB

    def summary(self) -> str:
        return (
            f"CacheStats(hits={self.hit_count}, misses={self.miss_count}, "
            f"hit_rate={self.hit_rate:.2%}, files={self.total_files}, "
            f"size={self.total_size_mb:.1f}MB, expired_del={self.expired_removed}, "
            f"size_del={self.size_limited_removed})"
        )


@dataclass
class CacheConfig:
    """缓存机制配置，控制过期清理、资源占用和命中率监控。"""

    ttl_days: float = 30.0
    max_size_mb: float = 2048.0
    max_files: int = 200
    cleanup_on_init: bool = True
    cleanup_on_write: bool = False
    stats_log_interval: int = 50


class FeatureCache:
    """增强的缓存管理器。

    支持 TTL 过期清理、命中率监控和资源占用控制：
    - TTL 过期：超过 ttl_days 的缓存文件自动清理
    - 资源控制：限制总缓存大小 (max_size_mb) 和文件数 (max_files)
    - 命中率监控：记录每次加载/计算事件，提供统计摘要
    - 线程安全：使用锁保护并发访问
    """

    def __init__(
        self,
        cache_dir: Path | str,
        version: str = "v1",
        cache_config: CacheConfig | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.cache_config = cache_config or CacheConfig()
        self._stats = CacheStats()
        self._lock = Lock()
        self._access_count = 0
        self._lru_tracker: dict[str, float] = {}

        if self.cache_config.cleanup_on_init:
            self._cleanup_expired()
            self._enforce_limits()

    def _get_shard_dir(self, feature_type: str) -> Path:
        shard_dir = self.cache_dir / feature_type
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir

    def _get_cache_path(self, key: FeatureCacheKey) -> Path:
        shard_dir = self._get_shard_dir(key.feature_type)
        return shard_dir / f"{key.to_string()}.npz"

    def _safe_load(self, cache_path: Path) -> dict | None:
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_version = str(data.get("version", ""))
            if cached_version != self.version:
                logger.warning(
                    f"缓存版本不匹配 (缓存 {cached_version!r} vs 当前 {self.version!r})，忽略缓存"
                )
                return None

            cached_ts = float(data.get("timestamp", 0))
            if cached_ts > 0:
                age_days = (time.time() - cached_ts) / 86400.0
                if age_days > self.cache_config.ttl_days:
                    logger.info(
                        f"  缓存已过期 ({age_days:.1f}d > {self.cache_config.ttl_days:.0f}d)，"
                        f"删除: {cache_path.name}"
                    )
                    try:
                        cache_path.unlink()
                    except Exception as unlink_err:
                        logger.warning(f"删除过期缓存文件失败 {cache_path}: {unlink_err}")
                    with self._lock:
                        self._stats.expired_removed += 1
                    return None
            cached_data = {k: data[k] for k in data.files}
            if "features" not in cached_data:
                logger.warning(f"缓存文件缺少 'features' 键: {cache_path.name}，将重新计算")
                return None
            return cached_data
        except Exception as e:
            logger.warning(f"加载缓存失败 {cache_path}: {e}，将重新计算")
            return None

    def _safe_save(self, cache_path: Path, features: np.ndarray) -> None:
        try:
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp_path, features=features, version=self.version,
                timestamp=time.time(),
            )
            with self._lock:
                tmp_path.replace(cache_path)
                self._lru_tracker[cache_path.stem] = time.time()
        except Exception as e:
            logger.warning(f"保存缓存失败 {cache_path}: {e}")
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
                    logger.debug(f"已清理临时文件: {tmp_path}")
            except Exception as cleanup_err:
                logger.warning(f"清理临时缓存文件失败 {tmp_path}: {cleanup_err}")

    def load_or_compute(
        self,
        feature_type: str,
        smiles_list: list[str],
        config: CompoundFeatureConfig,
        compute_fn: Callable[[list[str], CompoundFeatureConfig], np.ndarray],
    ) -> np.ndarray:
        if not config.enable_cache:
            return compute_fn(smiles_list, config)

        key = FeatureCacheKey(feature_type, tuple(smiles_list), config)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cached = self._safe_load(cache_path)
            if cached is not None:
                with self._lock:
                    self._stats.hit_count += 1
                    self._access_count += 1
                    self._lru_tracker[key.to_string()] = time.time()
                logger.info(f"  从缓存加载 {feature_type}: {cache_path.name}")
                self._maybe_log_stats()
                return cached["features"].astype(np.float32)

        with self._lock:
            self._stats.miss_count += 1
            self._access_count += 1

        logger.info(f"  计算 {feature_type} ({len(smiles_list)} compounds)...")
        features = compute_fn(smiles_list, config)
        self._safe_save(cache_path, features)

        if self.cache_config.cleanup_on_write:
            self._enforce_limits()

        self._maybe_log_stats()
        return features

    def _maybe_log_stats(self):
        if self._access_count > 0 and self._access_count % self.cache_config.stats_log_interval == 0:
            self._refresh_stats()
            logger.info(f"  {self._stats.summary()}")

    def _refresh_stats(self):
        files = list(self.cache_dir.rglob("*.npz"))
        with self._lock:
            self._stats.total_files = len(files)
            self._stats.total_size_bytes = sum(
                f.stat().st_size for f in files if f.is_file()
            )

    def _cleanup_expired(self) -> int:
        if self.cache_config.ttl_days <= 0:
            return 0
        removed = 0
        now = time.time()
        ttl_seconds = self.cache_config.ttl_days * 86400.0
        for f in self.cache_dir.rglob("*.npz"):
            try:
                age = now - f.stat().st_mtime
                if age > ttl_seconds:
                    with self._lock:
                        f.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"清理过期缓存失败 {f}: {e}")
        if removed > 0:
            logger.info(f"  过期缓存清理: 删除 {removed} 个文件")
            with self._lock:
                self._stats.expired_removed += removed
        return removed

    def _enforce_limits(self) -> int:
        files = sorted(
            self.cache_dir.rglob("*.npz"),
            key=lambda f: self._lru_tracker.get(f.stem, 0),
        )
        removed = 0

        if self.cache_config.max_size_mb > 0 and files:
            total_size = sum(f.stat().st_size for f in files)
            max_bytes = self.cache_config.max_size_mb * _MB
            while total_size > max_bytes and files:
                oldest = files.pop(0)
                total_size -= oldest.stat().st_size
                try:
                    with self._lock:
                        oldest.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"资源控制清理失败 {oldest}: {e}")

        if self.cache_config.max_files > 0:
            files = sorted(
                self.cache_dir.rglob("*.npz"),
                key=lambda f: self._lru_tracker.get(f.stem, 0),
            )
            while len(files) > self.cache_config.max_files:
                oldest = files.pop(0)
                try:
                    with self._lock:
                        oldest.unlink()
                    removed += 1
                except Exception as e:
                    logger.warning(f"文件数限制清理失败 {oldest}: {e}")

        if removed > 0:
            logger.info(f"  资源控制清理: 删除 {removed} 个文件")
            with self._lock:
                self._stats.size_limited_removed += removed
        return removed

    def get_stats(self) -> CacheStats:
        self._refresh_stats()
        return self._stats

    def clear_all(self) -> int:
        removed = 0
        for f in self.cache_dir.rglob("*.npz"):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {f}: {e}")
        with self._lock:
            self._stats = CacheStats()
        return removed

    def cleanup(self) -> dict:
        expired = self._cleanup_expired()
        limited = self._enforce_limits()
        self._refresh_stats()
        return {
            "expired_removed": expired,
            "size_limited_removed": limited,
            "stats": self._stats,
        }

    def warmup(
        self,
        smiles_list: list[str],
        feature_types: list[str] | None = None,
        config: CompoundFeatureConfig | None = None,
    ) -> dict:
        """批量预计算常用特征，加速后续查询。

        Args:
            smiles_list: 预定义的 SMILES 列表。
            feature_types: 要预热的特征类型列表，默认全部。
            config: 化合物特征配置。

        Returns:
            包含每种特征计算结果和统计信息的字典。
        """
        if config is None:
            config = CompoundFeatureConfig()

        if feature_types is None:
            feature_types = ["ecfp4", "maccs", "rdkit_descriptors", "3d_conformer"]

        _FEATURE_COMPUTE_FNS: dict[str, Callable] = {
            "ecfp4": _compute_ecfp4,
            "maccs": _compute_maccs,
            "rdkit_descriptors": _compute_rdkit_descriptors,
            "3d_conformer": _compute_3d_conformer_features,
        }

        for ft in feature_types:
            if ft not in _FEATURE_COMPUTE_FNS:
                raise ValueError(
                    f"不支持的特征类型: {ft!r}，可选: {list(_FEATURE_COMPUTE_FNS.keys())}"
                )

        start_time = time.time()
        results: dict[str, np.ndarray] = {}
        errors: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=min(len(feature_types), 4)) as executor:
            future_to_ft = {}
            for ft in feature_types:
                compute_fn = _FEATURE_COMPUTE_FNS[ft]
                future = executor.submit(
                    self.load_or_compute, ft, smiles_list, config, compute_fn
                )
                future_to_ft[future] = ft

            for future in as_completed(future_to_ft):
                ft = future_to_ft[future]
                try:
                    results[ft] = future.result()
                except Exception as e:
                    logger.error(f"预热特征 {ft} 失败: {e}")
                    errors[ft] = str(e)

        elapsed = time.time() - start_time
        self._refresh_stats()

        logger.info(
            f"缓存预热完成: {len(results)}/{len(feature_types)} 种特征成功, "
            f"耗时 {elapsed:.1f}s, 命中率 {self._stats.hit_rate:.2%}"
        )
        if errors:
            logger.warning(f"预热失败的特征: {list(errors.keys())}")

        return {
            "results": results,
            "errors": errors,
            "elapsed_seconds": elapsed,
            "stats": self._stats,
        }


def _compute_ecfp4(smiles_iter: list[str], config: CompoundFeatureConfig | None = None) -> np.ndarray:
    if config is None:
        config = CompoundFeatureConfig()

    n_bits = config.ecfp4_nbits
    radius = config.ecfp4_radius
    fps = np.zeros((len(smiles_iter), n_bits), dtype=np.float32)
    n_parse_fail = 0
    n_fp_fail = 0
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"ECFP4 SMILES 解析失败 索引 {i}: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            n_parse_fail += 1
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            for bit in fp.GetOnBits():
                fps[i, bit] = 1.0
        except Exception as e:
            logger.warning(f"ECFP4 指纹生成失败 索引 {i}: {smi!r}, 错误: {e}")
            n_fp_fail += 1
    total_fail = n_parse_fail + n_fp_fail
    if total_fail > 0:
        logger.warning(
            f"ECFP4 处理完成: {len(smiles_iter)} 个化合物, "
            f"SMILES 解析失败 {n_parse_fail} 个, 指纹生成失败 {n_fp_fail} 个"
        )
    if len(smiles_iter) > 0 and total_fail == len(smiles_iter):
        raise ValueError("ECFP4 指纹生成全部失败，请检查输入 SMILES 格式")
    return fps


def _compute_maccs(smiles_iter: list[str], config: CompoundFeatureConfig | None = None) -> np.ndarray:
    if config is None:
        config = CompoundFeatureConfig()

    fps = []
    n_parse_fail = 0
    n_fp_fail = 0
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"MACCS SMILES 解析失败 索引 {i}: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            n_parse_fail += 1
            fps.append(np.zeros(167, dtype=np.float32))
            continue
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            arr = np.zeros(167, dtype=np.float32)
            arr[list(fp.GetOnBits())] = 1.0
            fps.append(arr)
        except Exception as e:
            logger.warning(f"MACCS 指纹生成失败 索引 {i}: {smi!r}, 错误: {e}")
            n_fp_fail += 1
            fps.append(np.zeros(167, dtype=np.float32))
    total_fail = n_parse_fail + n_fp_fail
    if total_fail > 0:
        logger.warning(
            f"MACCS 处理完成: {len(smiles_iter)} 个化合物, "
            f"SMILES 解析失败 {n_parse_fail} 个, 指纹生成失败 {n_fp_fail} 个"
        )
    if len(smiles_iter) > 0 and total_fail == len(smiles_iter):
        raise ValueError("MACCS 指纹生成全部失败，请检查输入 SMILES 格式")
    return np.array(fps, dtype=np.float32)


def _compute_rdkit_descriptors(
    smiles_iter: list[str], config: CompoundFeatureConfig | None = None
) -> np.ndarray:
    if config is None:
        config = CompoundFeatureConfig()

    descriptor_names = config.rdkit_descriptor_names
    if not descriptor_names:
        return np.zeros((len(smiles_iter), 0), dtype=np.float32)

    desc_funcs = {name: getattr(Descriptors, name) for name in descriptor_names}
    rows = []
    n_parse_fail = 0
    n_desc_fail = 0
    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"RDKit 描述符 SMILES 解析失败 索引 {i}: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            n_parse_fail += 1
            rows.append([np.nan] * len(descriptor_names))
            continue
        vals = []
        for name in descriptor_names:
            try:
                vals.append(float(desc_funcs[name](mol)))
            except Exception as e:
                logger.warning(f"RDKit 描述符计算失败 索引 {i} 描述符 {name}: {e}")
                n_desc_fail += 1
                vals.append(np.nan)
        rows.append(vals)
    if n_parse_fail > 0 or n_desc_fail > 0:
        logger.warning(
            f"RDKit 描述符处理完成: {len(smiles_iter)} 个化合物, "
            f"SMILES 解析失败 {n_parse_fail} 个, 单个描述符计算失败 {n_desc_fail} 个"
        )
    if len(smiles_iter) > 0 and n_parse_fail == len(smiles_iter):
        raise ValueError("RDKit 描述符生成全部失败，请检查输入 SMILES 格式")
    return np.array(rows, dtype=np.float32)


def _compute_3d_conformer_features(
    smiles_iter: list[str], config: CompoundFeatureConfig | None = None
) -> np.ndarray:
    """计算3D构象分子特征（MsDGCN风格，PMID 42334640）。

    基于RDKit ETKDGv3构象生成，提取：
    - 3D分子描述符（PMI1/2/3、Asphericity、Eccentricity、InertialShapeFactor）
    - 构象能量
    - 回转半径
    共12维特征，补充二维指纹无法捕获的空间信息。
    """
    if config is None:
        config = CompoundFeatureConfig()

    n_features = 12
    rows = np.zeros((len(smiles_iter), n_features), dtype=np.float32)
    n_parse_fail = 0
    n_conf_fail = 0

    for i, smi in enumerate(smiles_iter):
        mol = None
        try:
            if pd.notna(smi):
                mol = Chem.MolFromSmiles(str(smi))
        except Exception as e:
            logger.warning(f"3D构象 SMILES解析失败 索引 {i}: {smi!r}, 错误: {e}")
            mol = None
        if mol is None:
            n_parse_fail += 1
            continue

        try:
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            status = AllChem.EmbedMolecule(mol, params)
            if status != 0:
                n_conf_fail += 1
                continue

            ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            if ff is None:
                n_conf_fail += 1
                continue

            energy = ff.CalcEnergy()
            opt_status = AllChem.MMFFOptimizeMolecule(mol)
            opt_energy = ff.CalcEnergy()
            if opt_status != 0:
                logger.debug(f"MMFF优化返回非零状态 索引 {i}: status={opt_status}")

            pmi1 = Descriptors.PMI1(mol)
            pmi2 = Descriptors.PMI2(mol)
            pmi3 = Descriptors.PMI3(mol)
            npr1 = Descriptors.NPR1(mol)
            npr2 = Descriptors.NPR2(mol)
            asphericity = Descriptors.Asphericity(mol)
            eccentricity = Descriptors.Eccentricity(mol)
            inertial_shape = Descriptors.InertialShapeFactor(mol)
            radius_of_gyration = Descriptors.RadiusOfGyration(mol)

            rows[i, 0] = float(energy) if energy is not None else 0.0
            rows[i, 1] = float(opt_energy) if opt_energy is not None else 0.0
            rows[i, 2] = float(pmi1) if pmi1 is not None else 0.0
            rows[i, 3] = float(pmi2) if pmi2 is not None else 0.0
            rows[i, 4] = float(pmi3) if pmi3 is not None else 0.0
            rows[i, 5] = float(npr1) if npr1 is not None else 0.0
            rows[i, 6] = float(npr2) if npr2 is not None else 0.0
            rows[i, 7] = float(asphericity) if asphericity is not None else 0.0
            rows[i, 8] = float(eccentricity) if eccentricity is not None else 0.0
            rows[i, 9] = float(inertial_shape) if inertial_shape is not None else 0.0
            rows[i, 10] = float(radius_of_gyration) if radius_of_gyration is not None else 0.0
            rows[i, 11] = float(Descriptors.FractionCSP3(mol))
        except Exception as e:
            logger.warning(f"3D构象特征计算失败 索引 {i}: {smi!r}, 错误: {e}")
            n_conf_fail += 1
            continue

    total_fail = n_parse_fail + n_conf_fail
    if total_fail > 0:
        logger.warning(
            f"3D构象特征处理完成: {len(smiles_iter)} 个化合物, "
            f"SMILES解析失败 {n_parse_fail} 个, 构象生成失败 {n_conf_fail} 个"
        )

    rows = np.nan_to_num(rows, nan=0.0, posinf=1e6, neginf=-1e6)
    valid_mask = ~(rows == 0).all(axis=1)
    n_valid = valid_mask.sum()
    if n_valid > 0:
        mean = rows[valid_mask].mean(axis=0)
        std = rows[valid_mask].std(axis=0) + 1e-8
        rows = (rows - mean) / std
    else:
        logger.warning("3D构象特征全部为零，无法标准化")
    return rows


def build_compound_features(
    smiles_list: list[str],
    stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    config: CompoundFeatureConfig | None = None,
    cache_manager: FeatureCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config is None:
        config = CompoundFeatureConfig()

    if cache_manager is not None:
        ecfp4 = cache_manager.load_or_compute("ecfp4", smiles_list, config, _compute_ecfp4)
    else:
        ecfp4 = _compute_ecfp4(smiles_list, config)

    if cache_manager is not None:
        maccs = cache_manager.load_or_compute("maccs", smiles_list, config, _compute_maccs)
    else:
        maccs = _compute_maccs(smiles_list, config)

    if cache_manager is not None:
        desc = cache_manager.load_or_compute(
            "rdkit_descriptors", smiles_list, config, _compute_rdkit_descriptors
        )
    else:
        desc = _compute_rdkit_descriptors(smiles_list, config)

    if stats is None:
        col_mean = np.nanmean(desc, axis=0)
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        mean = desc.mean(axis=0)
        std = desc.std(axis=0) + 1e-8
        desc = (desc - mean) / std
    else:
        mean, std, col_mean = stats
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(col_mean, inds[1])
        desc = np.nan_to_num(desc, nan=0.0, posinf=1e6, neginf=-1e6)
        desc = (desc - mean) / (std + 1e-8)

    feature_parts = [ecfp4]
    if config.use_maccs:
        feature_parts.append(maccs)
    if config.use_rdkit_descriptors:
        feature_parts.append(desc)
    if config.use_3d_conformer:
        if cache_manager is not None:
            conf3d = cache_manager.load_or_compute(
                "3d_conformer", smiles_list, config, _compute_3d_conformer_features
            )
        else:
            conf3d = _compute_3d_conformer_features(smiles_list, config)
        feature_parts.append(conf3d)

    features = np.hstack(feature_parts).astype(np.float32)
    return features, mean, std, col_mean


def compute_aac(sequences: list[str]) -> np.ndarray:
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    aac_matrix = np.zeros((len(sequences), 20), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if not seq or pd.isna(seq):
            continue
        seq = str(seq).upper().strip()
        total = len(seq)
        if total == 0:
            continue
        for aa in seq:
            if aa in aa_to_idx:
                aac_matrix[i, aa_to_idx[aa]] += 1
        aac_matrix[i] /= total
    return aac_matrix


def compute_esm2_embeddings(
    gene_to_seq: dict[str, str],
    cache_path: Path | None = None,
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        logger.info(f"  从缓存加载 ESM-2 嵌入: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        embeddings = {str(k): v.astype(np.float32) for k, v in cached.items()}
        logger.info(f"  ESM-2 嵌入已加载: {len(embeddings)} 蛋白, dim={next(iter(embeddings.values())).shape[0]}")
        return embeddings

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from transformers import EsmModel, EsmTokenizer

    import torch as _torch

    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    logger.info(f"  加载 ESM-2 模型: {model_name} (via HF_ENDPOINT={os.environ['HF_ENDPOINT']}) ...")
    tokenizer = EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    model = EsmModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()
    esm_dim = model.config.hidden_size
    logger.info(f"  ESM-2 嵌入维度: {esm_dim}")

    genes = sorted(gene_to_seq.keys(), key=lambda g: len(gene_to_seq.get(g, "")), reverse=True)
    embeddings: dict[str, np.ndarray] = {}

    with torch.no_grad():
        for i in range(0, len(genes), batch_size):
            batch_genes = genes[i:i + batch_size]
            batch_seqs = [gene_to_seq[g] for g in batch_genes]

            max_len = 1022
            truncated_seqs = [s[:max_len] for s in batch_seqs]

            inputs = tokenizer(
                truncated_seqs, return_tensors="pt", padding=True, truncation=True,
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state

            attention_mask = inputs["attention_mask"]
            for b in range(attention_mask.shape[0]):
                seq_len = attention_mask[b].sum().item()
                if seq_len > 1:
                    attention_mask[b, 0] = 0
                    attention_mask[b, seq_len - 1] = 0

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_emb = (hidden * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            pooled = sum_emb / count

            for j, g in enumerate(batch_genes):
                embeddings[g] = pooled[j].cpu().numpy().astype(np.float32)

            if (i + batch_size) % 20 == 0 or i + batch_size >= len(genes):
                logger.info(f"  ESM-2 嵌入进度: {min(i + batch_size, len(genes))}/{len(genes)}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **embeddings)
        logger.info(f"  ESM-2 嵌入已缓存: {cache_path}")

    del model, tokenizer
    _torch.cuda.empty_cache()
    logger.info("  ESM-2 模型已释放 GPU 内存")

    return embeddings


def load_protein_features(use_esm2: bool = True) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    _cfg = Config()
    _paths = _cfg.get_resolved_paths()
    pseaac_path = _paths.l2_results / "protein_pseaac.csv"
    esm_cache = _paths.l4_results / "esm2_protein_embeddings.npz"
    prot_feat: dict[str, np.ndarray] = {}
    gene_to_seq: dict[str, str] = {}

    genes = list(gene_to_seq.keys())

    esm2_embeddings = None
    if use_esm2:
        try:
            esm2_embeddings = compute_esm2_embeddings(
                gene_to_seq, cache_path=esm_cache,
                model_name="facebook/esm2_t30_150M_UR50D",
            )
        except Exception as e:
            logger.warning(f"ESM-2 嵌入计算失败 ({e})，降级为 AAC + PseAAC")

    if esm2_embeddings is not None:
        esm_dim = next(iter(esm2_embeddings.values())).shape[0]
        missing_genes = set(genes) - set(esm2_embeddings.keys())
        if missing_genes:
            logger.warning(f"ESM-2 缺失 {len(missing_genes)} 个基因的嵌入，已用零填充")
            for g in missing_genes:
                esm2_embeddings[g] = np.zeros(esm_dim, dtype=np.float32)

        prot_feat = esm2_embeddings
        logger.info(f"蛋白特征 (ESM-2): {len(prot_feat)} 基因, dim={esm_dim}")
    else:
        seqs = [gene_to_seq[g] for g in genes]
        aac = compute_aac(seqs)

        pseaac_data: dict[str, np.ndarray] = {}
        if pseaac_path.exists():
            df_pseaac = pd.read_csv(pseaac_path)
            if "Unnamed: 0" in df_pseaac.columns:
                df_pseaac = df_pseaac.drop(columns=["Unnamed: 0"])
            if "gene_symbol" in df_pseaac.columns:
                for _, row in df_pseaac.iterrows():
                    g = str(row["gene_symbol"]).strip().upper()
                    vals = row.drop("gene_symbol").values.astype(np.float32)
                    pseaac_data[g] = vals

        pseaac_dim = 0
        if pseaac_data:
            pseaac_dim = len(next(iter(pseaac_data.values())))

        n_missing_pseaac = 0
        for i, g in enumerate(genes):
            aac_vec = aac[i]
            if g in pseaac_data:
                prot_feat[g] = np.concatenate([aac_vec, pseaac_data[g]])
            elif pseaac_dim > 0:
                prot_feat[g] = np.concatenate([aac_vec, np.zeros(pseaac_dim, dtype=np.float32)])
                n_missing_pseaac += 1
            else:
                prot_feat[g] = aac_vec

        if n_missing_pseaac > 0:
            logger.warning(f"PseAAC 缺失 {n_missing_pseaac} 个基因的特征，已用零填充")

        if not prot_feat:
            for i, g in enumerate(genes):
                prot_feat[g] = aac[i]

        logger.info(f"蛋白特征 (AAC+PseAAC): {len(prot_feat)} 基因, dim={next(iter(prot_feat.values())).shape[0]}")

    return prot_feat, gene_to_seq