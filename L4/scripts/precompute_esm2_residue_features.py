#!/usr/bin/env python3
"""为 graph_cache_v31.pkl 中的 6,864 个蛋白节点预计算残基级 ESM-2 嵌入。

输入:
  - L4/results_v10_minibatch/graph_cache_v31.pkl (gene_to_idx)
  - UniProt REST API (获取蛋白序列)
  - facebook/esm2_t30_150M_UR50D (Hugging Face)

输出:
  - L4/results_v10_minibatch/protein_sequences_6864.csv
  - L4/results_v10_minibatch/esm2_150M_residue_features.pt
    {
      'genes': list[str],          # 长度 N
      'uniprot_acs': list[str],
      'reviewed': list[bool],
      'sequences': list[str],
      'lengths': torch.LongTensor, # 每条序列残基数
      'offsets': torch.LongTensor, # 在 concat 张量中的偏移, 长度 N+1
      'embeddings': torch.FloatTensor, # (total_residues, 640)
      'esm_model_name': str,
    }
"""
from __future__ import annotations

import io
import logging
import os
import pickle
import time
from pathlib import Path

import pandas as pd
import requests
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("precompute_esm2_residue_features")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "L4" / "results_v10_minibatch"
GRAPH_CACHE = RESULTS_DIR / "graph_cache_v31.pkl"
SEQ_CSV = RESULTS_DIR / "protein_sequences_6864.csv"
OUT_PT = RESULTS_DIR / "esm2_150M_residue_features.pt"

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
UNIPROT_CHUNK = 80
UNIPROT_SLEEP = 0.6
ESM_BATCH_SIZE = 4


def load_protein_gene_list(cache_path: Path) -> list[str]:
    """从 graph_cache 中按局部索引排序返回基因名列表。"""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    gene_to_idx = cache["graphs"]["gene_to_idx"]
    n_compounds = cache["graphs"]["n_compounds"]
    local = [(idx - n_compounds, gene) for gene, idx in gene_to_idx.items()]
    local.sort(key=lambda x: x[0])
    genes = [g for _, g in local]
    logger.info(f"从 {cache_path} 读取 {len(genes)} 个蛋白节点基因名")
    return genes


def clean_sequence(seq: str) -> str:
    return seq.strip().upper().replace("B", "X").replace("Z", "X").replace("J", "X").replace("U", "X").replace("O", "X")


def fetch_uniprot_sequences(genes: list[str]) -> pd.DataFrame:
    """分批从 UniProt 获取序列，优先选择 reviewed (Swiss-Prot) 条目。"""
    records: dict[str, dict] = {g: {"gene_symbol": g, "uniprot_ac": "", "reviewed": False, "sequence": ""} for g in genes}

    for i in range(0, len(genes), UNIPROT_CHUNK):
        chunk = genes[i : i + UNIPROT_CHUNK]
        gene_query = " OR ".join(f"gene:{g}" for g in chunk)
        query = f"organism_id:9606 AND ({gene_query})"
        url = "https://rest.uniprot.org/uniprotkb/stream"
        params = {"query": query, "format": "tsv", "fields": "accession,gene_names,sequence,reviewed"}
        logger.info(f"UniProt 查询批次 {i}-{i + len(chunk)} ...")
        for attempt in range(3):
            try:
                r = requests.get(url, params=params, timeout=120)
                r.raise_for_status()
                break
            except Exception as e:
                logger.warning(f"  UniProt 请求失败 (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"无法从 UniProt 获取序列: 批次 {i}")

        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        # 为该批次基因建立候选
        chunk_lower = {g.lower(): g for g in chunk}
        for _, row in df.iterrows():
            gene_names = str(row.get("Gene Names", ""))
            gene_list = [x.strip() for x in gene_names.replace(";", " ").split() if x.strip()]
            matched = None
            for name in gene_list:
                if name.lower() in chunk_lower:
                    matched = chunk_lower[name.lower()]
                    break
            if matched is None:
                continue
            reviewed = str(row.get("Reviewed", "")).strip().lower() == "reviewed"
            ac = str(row.get("Entry", "")).strip()
            seq = str(row.get("Sequence", "")).strip()
            if not seq:
                continue
            cur = records[matched]
            # 保留 reviewed；如果当前无记录或新的是 reviewed 而当前不是
            if (not cur["sequence"]) or (reviewed and not cur["reviewed"]):
                cur.update({"uniprot_ac": ac, "reviewed": reviewed, "sequence": seq})
        time.sleep(UNIPROT_SLEEP)

    df_out = pd.DataFrame.from_records(list(records.values()))
    found = df_out["sequence"].astype(bool).sum()
    logger.info(f"UniProt 序列获取完成: {found}/{len(genes)} 条有序列")
    return df_out


def ensure_sequences(genes: list[str]) -> pd.DataFrame:
    if SEQ_CSV.exists():
        df = pd.read_csv(SEQ_CSV)
        if set(df["gene_symbol"].tolist()) == set(genes) and df["sequence"].notna().sum() == len(genes):
            logger.info(f"复用已有序列文件: {SEQ_CSV}")
            return df
        logger.info(f"序列文件存在但不完整, 重新获取 (已有 {df['sequence'].notna().sum()} 条)")
    df = fetch_uniprot_sequences(genes)
    df.to_csv(SEQ_CSV, index=False)
    logger.info(f"序列已保存: {SEQ_CSV}")
    return df


def main() -> None:
    if not GRAPH_CACHE.exists():
        raise FileNotFoundError(f"图缓存不存在: {GRAPH_CACHE}")

    genes = load_protein_gene_list(GRAPH_CACHE)
    seq_df = ensure_sequences(genes)

    missing = seq_df[seq_df["sequence"].isna() | (seq_df["sequence"] == "")]
    if len(missing) > 0:
        logger.error(f"以下 {len(missing)} 个蛋白缺失 UniProt 序列:\n" + "\n".join(missing["gene_symbol"].tolist()[:20]))
        raise RuntimeError(f"存在 {len(missing)} 个蛋白无序列, 无法生成真实嵌入")

    seq_df = seq_df.set_index("gene_symbol").loc[genes].reset_index()
    sequences = [clean_sequence(s) for s in seq_df["sequence"].tolist()]

    # 下载/加载 ESM-2
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import EsmModel, EsmTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"加载 ESM-2 模型: {MODEL_NAME} (device={device}) ...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    esm_dim = model.config.hidden_size
    logger.info(f"ESM-2 hidden_dim={esm_dim}")

    # 按长度降序处理以减少 padding
    order = sorted(range(len(genes)), key=lambda i: len(sequences[i]), reverse=True)
    emb_by_idx: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for start in range(0, len(genes), ESM_BATCH_SIZE):
            batch_order = order[start : start + ESM_BATCH_SIZE]
            batch_genes = [genes[i] for i in batch_order]
            batch_seqs = [sequences[i][:1020] for i in batch_order]
            logger.info(f"ESM-2 进度 {start}-{min(start + ESM_BATCH_SIZE, len(genes))}/{len(genes)} (batch max len {max(len(s) for s in batch_seqs)})")

            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1022,
            ).to(device)

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, L, D)
            attn = inputs["attention_mask"]

            for j, orig_idx in enumerate(batch_order):
                tok_len = int(attn[j].sum().item())
                res_len = max(0, tok_len - 2)
                emb = hidden[j, 1 : 1 + res_len].cpu().to(torch.float32)
                emb_by_idx[orig_idx] = emb

            del inputs, outputs, hidden, attn
            torch.cuda.empty_cache()

    # 按原始基因顺序整理
    emb_list = [emb_by_idx[i] for i in range(len(genes))]
    lengths = torch.LongTensor([e.shape[0] for e in emb_list])
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    embeddings = torch.cat(emb_list, dim=0)  # (total_residues, esm_dim)

    payload = {
        "genes": genes,
        "uniprot_acs": seq_df["uniprot_ac"].tolist(),
        "reviewed": seq_df["reviewed"].astype(bool).tolist(),
        "sequences": sequences,
        "lengths": lengths,
        "offsets": offsets,
        "embeddings": embeddings,
        "esm_model_name": MODEL_NAME,
    }

    torch.save(payload, OUT_PT)
    logger.info(f"已保存残基级 ESM-2 特征: {OUT_PT}")
    logger.info(f"  蛋白数: {len(genes)}")
    logger.info(f"  总残基数: {embeddings.shape[0]}")
    logger.info(f"  嵌入维度: {esm_dim}")
    logger.info(f"  文件大小: {OUT_PT.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
