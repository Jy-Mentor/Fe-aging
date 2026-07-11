"""特征工程模块

化合物特征：ECFP4 指纹 + MACCS 密钥 + RDKit 分子描述符
蛋白特征：AAC 氨基酸组成 + ESM-2 预训练嵌入（可选）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class CompoundFeatureConfig:
    """化合物特征工程配置。

    统一管理化合物特征计算所需的各项参数，避免在脚本中硬编码。
    """

    ecfp4_nbits: int = 2048
    ecfp4_radius: int = 2
    use_maccs: bool = True
    use_rdkit_descriptors: bool = True
    rdkit_descriptor_names: list[str] = field(default_factory=lambda: list(RDKIT_DESCRIPTOR_NAMES))
    enable_cache: bool = True
    cache_version: str = "v1"

    def __post_init__(self):
        """校验关键参数。"""
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
        """生成稳定的字符串表示，用于哈希。"""
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
        # SMILES 列表可能很长，使用哈希避免文件名过长
        smiles_hash = hashlib.sha256(
            json.dumps(self.smiles_list, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{self.feature_type}_{self.config.cache_version}_{smiles_hash}_{hashlib.sha256(config_str.encode()).hexdigest()[:8]}"


class FeatureCache:
    """统一的特征缓存管理器。

    支持按特征类型、SMILES 列表和配置参数分别缓存，避免不同参数下的缓存冲突。
    """

    def __init__(self, cache_dir: Path | str, version: str = "v1"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version

    def _get_cache_path(self, key: FeatureCacheKey) -> Path:
        """根据缓存键生成缓存文件路径。"""
        return self.cache_dir / f"{key.to_string()}.npz"

    def _safe_load(self, cache_path: Path) -> dict | None:
        """安全加载缓存文件，失败时返回 None 并不抛出异常。"""
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_version = str(data.get("version", ""))
            if cached_version != self.version:
                logger.warning(
                    f"缓存版本不匹配 (缓存 {cached_version!r} vs 当前 {self.version!r})，忽略缓存"
                )
                return None
            return {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning(f"加载缓存失败 {cache_path}: {e}，将重新计算")
            return None

    def _safe_save(self, cache_path: Path, features: np.ndarray) -> None:
        """安全保存缓存文件，使用临时文件 + 重命名避免并发写入损坏。"""
        try:
            # np.savez_compressed 会在路径无 .npz 后缀时自动追加，因此临时文件必须
            # 显式使用 .npz 后缀，否则 replace 时源文件路径不一致导致 WinError 2
            tmp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp_path, features=features, version=self.version)
            tmp_path.replace(cache_path)
        except Exception as e:
            logger.warning(f"保存缓存失败 {cache_path}: {e}")

    def load_or_compute(
        self,
        feature_type: str,
        smiles_list: list[str],
        config: CompoundFeatureConfig,
        compute_fn: Callable[[list[str], CompoundFeatureConfig], np.ndarray],
    ) -> np.ndarray:
        """优先加载缓存，否则计算并缓存特征。

        Args:
            feature_type: 特征类型标识，如 "ecfp4", "maccs", "rdkit_descriptors"
            smiles_list: SMILES 字符串列表
            config: 特征工程配置
            compute_fn: 计算函数，签名为 (smiles_list, config) -> np.ndarray

        Returns:
            特征矩阵 np.ndarray
        """
        if not config.enable_cache:
            return compute_fn(smiles_list, config)

        key = FeatureCacheKey(feature_type, tuple(smiles_list), config)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            cached = self._safe_load(cache_path)
            if cached is not None:
                logger.info(f"  从缓存加载 {feature_type}: {cache_path.name}")
                return cached["features"].astype(np.float32)

        logger.info(f"  计算 {feature_type} ({len(smiles_list)} compounds)...")
        features = compute_fn(smiles_list, config)
        self._safe_save(cache_path, features)
        return features

    def clear_all(self) -> int:
        """清空所有缓存文件，返回删除数量。"""
        removed = 0
        for f in self.cache_dir.glob("*.npz"):
            try:
                f.unlink()
                removed += 1
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {f}: {e}")
        return removed


def _compute_ecfp4(smiles_iter: list[str], config: CompoundFeatureConfig | None = None) -> np.ndarray:
    """计算 ECFP4 (Morgan) 指纹。

    Args:
        smiles_iter: SMILES 字符串列表。
        config: 特征工程配置；为 None 时使用默认配置。

    Returns:
        (n_compounds, n_bits) 指纹矩阵。
    """
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
    """计算 MACCS 密钥指纹。

    Args:
        smiles_iter: SMILES 字符串列表。
        config: 特征工程配置（当前仅用于占位，保持接口一致性）。

    Returns:
        (n_compounds, 167) MACCS 指纹矩阵。
    """
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
    """计算 RDKit 分子描述符。

    Args:
        smiles_iter: SMILES 字符串列表。
        config: 特征工程配置，用于指定描述符名称列表。

    Returns:
        (n_compounds, n_descriptors) 描述符矩阵。
    """
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


def build_compound_features(
    smiles_list: list[str],
    stats: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    config: CompoundFeatureConfig | None = None,
    cache_manager: FeatureCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建化合物特征矩阵（ECFP4 + MACCS + RDKit 描述符）。

    Args:
        smiles_list: SMILES 字符串列表。
        stats: 预计算的 (mean, std, col_mean) 统计量；为 None 时基于输入数据计算。
        config: 特征工程配置；为 None 时使用默认配置。
        cache_manager: 特征缓存管理器；为 None 时不启用缓存。

    Returns:
        (features, mean, std, col_mean)
    """
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

    features = np.hstack(feature_parts).astype(np.float32)
    return features, mean, std, col_mean


def compute_aac(sequences: list[str]) -> np.ndarray:
    """计算氨基酸组成（Amino Acid Composition, AAC）。

    Args:
        sequences: 氨基酸序列列表。

    Returns:
        (n_sequences, 20) AAC 矩阵，每行加和为 1（非空序列）。
    """
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
    """使用 ESM-2 预训练蛋白质语言模型计算 per-protein 嵌入

    对每个蛋白序列通过 ESM-2 前向传播，取序列位置（排除特殊 token）的
    均值池化作为蛋白嵌入。结果缓存到磁盘避免重复计算。

    参考: Rives et al. (2021) "Biological structure and function emerge from
          scaling unsupervised learning to 250 million protein sequences", PNAS.

    Args:
        gene_to_seq: {基因符号: 氨基酸序列}
        cache_path: 缓存文件路径（.npz），None 则不缓存
        model_name: HuggingFace ESM-2 模型名
        batch_size: 推理批次大小

    Returns:
        {基因符号: embedding (np.ndarray, shape=(esm_dim,))}
    """
    if cache_path is not None and cache_path.exists():
        logger.info(f"  从缓存加载 ESM-2 嵌入: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = {str(k): v.astype(np.float32) for k, v in cached.items()}
        logger.info(f"  ESM-2 嵌入已加载: {len(embeddings)} 蛋白, dim={next(iter(embeddings.values())).shape[0]}")
        return embeddings

    # v17: 使用 HuggingFace 镜像解决国内网络不可达问题
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

            # 截断过长序列
            max_len = 1022
            truncated_seqs = [s[:max_len] for s in batch_seqs]

            inputs = tokenizer(
                truncated_seqs, return_tensors="pt", padding=True, truncation=True,
            ).to(device)

            outputs = model(**inputs)
            # last_hidden_state: (batch, seq_len, esm_dim)
            hidden = outputs.last_hidden_state

            # 均值池化：排除 [CLS] (pos 0) 和 [EOS] (最后一个有效 token)
            attention_mask = inputs["attention_mask"]
            # 将 [CLS] 和 [EOS] 位置 mask 掉
            for b in range(attention_mask.shape[0]):
                seq_len = attention_mask[b].sum().item()
                if seq_len > 1:
                    attention_mask[b, 0] = 0        # [CLS]
                    attention_mask[b, seq_len - 1] = 0  # [EOS]

            # 安全均值池化
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
    """加载蛋白特征

    v17: 默认使用 ESM-2 预训练嵌入（640维），远程同源检测能力远超 AAC。
    若 ESM-2 不可用，自动降级为 AAC + PseAAC。

    Args:
        use_esm2: 是否使用 ESM-2 嵌入（默认 True）

    Returns:
        prot_feat: {基因符号: np.ndarray}
        gene_to_seq: {基因符号: 序列字符串}
    """
    _cfg = Config()
    _paths = _cfg.get_resolved_paths()
    pf_path = _paths.l2_results / "target_protein_features.csv"
    pseaac_path = _paths.l2_results / "protein_pseaac.csv"
    esm_cache = _paths.l4_results / "esm2_protein_embeddings.npz"
    prot_feat: dict[str, np.ndarray] = {}
    gene_to_seq: dict[str, str] = {}

    if pf_path.exists():
        df = pd.read_csv(pf_path)
        for _, row in df.iterrows():
            gene = str(row["gene_symbol"]).strip().upper()
            seq = str(row["sequence"]) if pd.notna(row["sequence"]) else ""
            gene_to_seq[gene] = seq

    genes = list(gene_to_seq.keys())

    # ---- v17: 尝试 ESM-2 嵌入 ----
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
        # 使用 ESM-2 嵌入作为蛋白特征
        esm_dim = next(iter(esm2_embeddings.values())).shape[0]
        missing_genes = set(genes) - set(esm2_embeddings.keys())
        if missing_genes:
            logger.warning(f"ESM-2 缺失 {len(missing_genes)} 个基因的嵌入，已用零填充")
            for g in missing_genes:
                esm2_embeddings[g] = np.zeros(esm_dim, dtype=np.float32)

        prot_feat = esm2_embeddings
        logger.info(f"蛋白特征 (ESM-2): {len(prot_feat)} 基因, dim={esm_dim}")
    else:
        # ---- 降级: AAC + PseAAC ----
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
