#!/usr/bin/env python3
from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

"""构建 rat → human 同源基因映射缓存（基于 mygene.info + NCBI HomoloGene）。

v32: 替换 map_probes_to_genes.py 中简单的 uppercase fallback，减少错误映射。
输入：L1/results/GSE61616_DE_gene_level.csv 中的 rat gene symbols
输出：L1/results/rat_to_human_ortholog_mygene.csv
"""

import pickle
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
L1_RESULTS = PROJECT_ROOT / "L1" / "results"
CACHE_FILE = L1_RESULTS / "rat_to_human_ortholog_mygene.csv"
RAW_CACHE = L1_RESULTS / "rat_to_human_ortholog_mygene_raw.pkl"

sys.path.insert(0, str(PROJECT_ROOT / "L1"))

import mygene


def load_rat_symbols(de_file: Path) -> set[str]:
    df = pd.read_csv(de_file)
    symbols: set[str] = set()
    for raw in df["GeneSymbol"].dropna().astype(str):
        for s in raw.split("///"):
            s = s.strip()
            if s and not s.startswith("LOC"):
                symbols.add(s)
    return symbols


def query_mygene(symbols: list[str], species: str, fields: list[str], batch_size: int = 1000):
    mg = mygene.MyGeneInfo()
    results = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        for attempt in range(3):
            try:
                res = mg.querymany(
                    batch,
                    scopes="symbol",
                    fields=",".join(fields),
                    species=species,
                    verbose=False,
                    returnall=True,
                )
                results.extend(res.get("out", []))
                break
            except Exception as e:
                print(f"  batch {i}-{i+len(batch)} attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"batch {i} failed after 3 attempts")
    return results


def extract_human_entrez_from_homologene(result: dict) -> int | None:
    hom = result.get("homologene")
    if not hom or "genes" not in hom:
        return None
    for taxid, entrez in hom["genes"]:
        if taxid == 9606:
            return int(entrez)
    return None


def build_map(de_file: Path) -> dict[str, str]:
    print(f"[1/4] 读取 rat symbols: {de_file}")
    symbols = sorted(load_rat_symbols(de_file))
    print(f"       共 {len(symbols)} 个唯一 rat symbols")

    print("[2/4] 查询 mygene.info (rat → homologene)")
    rat_results = query_mygene(symbols, "rat", ["symbol", "homologene"])

    rat_to_human_entrez: dict[str, int] = {}
    unmatched: list[str] = []
    for r in rat_results:
        query = r.get("query", "")
        if "notfound" in r:
            unmatched.append(query)
            continue
        human_entrez = extract_human_entrez_from_homologene(r)
        if human_entrez:
            rat_to_human_entrez[query] = human_entrez

    print(f"       HomoloGene 匹配: {len(rat_to_human_entrez)}/{len(symbols)}")
    print(f"       未匹配: {len(unmatched)}")

    print("[3/4] 查询 mygene.info (human entrez → symbol)")
    human_entrez_ids = sorted(set(rat_to_human_entrez.values()))
    human_results = query_mygene([str(x) for x in human_entrez_ids], "human", ["entrezgene", "symbol"])

    entrez_to_symbol: dict[int, str] = {}
    for r in human_results:
        if "notfound" in r:
            continue
        e = r.get("entrezgene")
        sym = r.get("symbol")
        if e and sym:
            entrez_to_symbol[int(e)] = str(sym).upper()

    rat_to_human: dict[str, str] = {}
    for rat_sym, he in rat_to_human_entrez.items():
        if he in entrez_to_symbol:
            rat_to_human[rat_sym] = entrez_to_symbol[he]

    print(f"       最终映射: {len(rat_to_human)} 个 rat → human")

    # 保存原始结果以便审计
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_CACHE, "wb") as f:
        pickle.dump({"rat_results": rat_results, "human_results": human_results}, f)

    out_df = pd.DataFrame(
        [{"rat_symbol": k, "human_symbol": v} for k, v in sorted(rat_to_human.items())]
    )
    out_df.to_csv(CACHE_FILE, index=False)
    print(f"[4/4] 已保存: {CACHE_FILE}")
    return rat_to_human


def load_map() -> dict[str, str]:
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"ortholog cache not found: {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE)
    return dict(zip(df["rat_symbol"], df["human_symbol"], strict=False))


if __name__ == "__main__":
    DE_FILE = L1_RESULTS / "GSE61616_DE_gene_level.csv"
    build_map(DE_FILE)
