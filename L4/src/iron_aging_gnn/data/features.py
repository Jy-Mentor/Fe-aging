"""特征工程模块

化合物特征：ECFP4 指纹 + MACCS 密钥 + RDKit 分子描述符
蛋白特征：AAC 氨基酸组成 + ESM-2 预训练嵌入（可选）
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ..utils.config import Config
from .constants import ECFP4_NBITS, RDKIT_DESCRIPTOR_NAMES

RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

logger = logging.getLogger(__name__)


def _compute_ecfp4(smiles_iter: list[str]) -> np.ndarray:
    fps = np.zeros((len(smiles_iter), ECFP4_NBITS), dtype=np.float32)
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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=ECFP4_NBITS)
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


def _compute_maccs(smiles_iter: list[str]) -> np.ndarray:
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


def _compute_rdkit_descriptors(smiles_iter: list[str]) -> np.ndarray:
    desc_funcs = {name: getattr(Descriptors, name) for name in RDKIT_DESCRIPTOR_NAMES}
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
            rows.append([np.nan] * len(RDKIT_DESCRIPTOR_NAMES))
            continue
        vals = []
        for name in RDKIT_DESCRIPTOR_NAMES:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建化合物特征矩阵（ECFP4 + MACCS + RDKit 描述符）。

    Args:
        smiles_list: SMILES 字符串列表。
        stats: 预计算的 (mean, std, col_mean) 统计量；为 None 时基于输入数据计算。

    Returns:
        (features, mean, std, col_mean)
    """
    logger.info(f"  computing ECFP4 ({len(smiles_list)} compounds)...")
    ecfp4 = _compute_ecfp4(smiles_list)
    logger.info(f"  computing MACCS ({len(smiles_list)} compounds)...")
    maccs = _compute_maccs(smiles_list)
    logger.info(f"  computing RDKit descriptors ({len(smiles_list)} compounds)...")
    desc = _compute_rdkit_descriptors(smiles_list)

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

    features = np.hstack([ecfp4, maccs, desc]).astype(np.float32)
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
