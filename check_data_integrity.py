#!/usr/bin/env python3
"""检查项目关键数据文件完整性和一致性。"""
import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"d:\铁衰老 绝不重蹈覆辙")


def file_status(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "size_bytes": None}
    return {"exists": True, "size_bytes": path.stat().st_size}


def check_npz(path: Path):
    st = file_status(path)
    print(f"\n[1] {path}")
    print(f"    存在: {st['exists']}")
    if not st["exists"]:
        return
    print(f"    大小: {st['size_bytes']:,} bytes ({st['size_bytes']/1024/1024:.2f} MB)")
    try:
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        print(f"    键: {keys}")
        for k in keys:
            arr = data[k]
            print(f"      - {k}: shape={arr.shape}, dtype={arr.dtype}")
        if "version" in keys:
            print(f"    version: {data['version']}")
        if "smiles" in keys:
            smiles = data["smiles"]
            print(f"    smiles 数量: {len(smiles)}, 前3条: {list(smiles[:3])}")
            empty = sum(1 for s in smiles if str(s).strip() == "" or s is None)
            print(f"    空/无效 smiles 数量: {empty}")
    except Exception as e:
        print(f"    加载失败: {e}")


def check_pkl(path: Path):
    st = file_status(path)
    print(f"\n[2] {path}")
    print(f"    存在: {st['exists']}")
    if not st["exists"]:
        return
    print(f"    大小: {st['size_bytes']:,} bytes ({st['size_bytes']/1024/1024:.2f} MB)")
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            print(f"    顶层键: {list(data.keys())}")
            for k, v in data.items():
                info = f"type={type(v).__name__}"
                if hasattr(v, "shape"):
                    info += f", shape={v.shape}"
                elif hasattr(v, "__len__"):
                    info += f", len={len(v)}"
                print(f"      - {k}: {info}")
        else:
            print(f"    类型: {type(data).__name__}")
    except Exception as e:
        print(f"    加载失败: {e}")


def check_csv(path: Path, name: str):
    st = file_status(path)
    print(f"\n[{name}] {path}")
    print(f"    存在: {st['exists']}")
    if not st["exists"]:
        return None
    print(f"    大小: {st['size_bytes']:,} bytes ({st['size_bytes']/1024/1024:.2f} MB)")
    try:
        df = pd.read_csv(path)
        print(f"    行数(含表头): {len(df)+1}, 数据行: {len(df)}")
        print(f"    列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"    加载失败: {e}")
        return None


def check_pt(path: Path):
    st = file_status(path)
    print(f"\n[5] {path}")
    print(f"    存在: {st['exists']}")
    if not st["exists"]:
        return
    print(f"    大小: {st['size_bytes']:,} bytes ({st['size_bytes']/1024/1024:.2f} MB)")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        print(f"    键: {list(ckpt.keys())}")
        for k, v in ckpt.items():
            info = f"type={type(v).__name__}"
            if hasattr(v, "shape"):
                info += f", shape={tuple(v.shape)}"
            elif isinstance(v, dict):
                info += f", len={len(v)}"
            print(f"      - {k}: {info}")
    except Exception as e:
        print(f"    加载失败: {e}")


def check_log(path: Path):
    st = file_status(path)
    print(f"\n[6] 日志文件 {path}")
    print(f"    存在: {st['exists']}")
    if not st["exists"]:
        return
    print(f"    大小: {st['size_bytes']:,} bytes ({st['size_bytes']/1024/1024:.2f} MB)")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        print(f"    总行数: {len(lines)}")
        if lines:
            print(f"    前3行:\n" + "".join(lines[:3]))
            print(f"    后3行:\n" + "".join(lines[-3:]))
    except Exception as e:
        print(f"    读取失败: {e}")


def main():
    print("="*70)
    print("关键数据文件完整性与一致性检查")
    print("="*70)

    # 1. compound_features_v31.npz
    requested = ROOT / "L4" / "results" / "compound_features_v31.npz"
    actual = ROOT / "L4" / "results_v10_minibatch" / "compound_features_v31.npz"
    print(f"\n请求路径: {requested}")
    check_npz(requested)
    print(f"\n实际路径: {actual}")
    check_npz(actual)

    # 2. graph_cache_v31.pkl
    requested = ROOT / "L4" / "results" / "graph_cache_v31.pkl"
    actual = ROOT / "L4" / "results_v10_minibatch" / "graph_cache_v31.pkl"
    print(f"\n请求路径: {requested}")
    check_pkl(requested)
    print(f"\n实际路径: {actual}")
    check_pkl(actual)

    # 3. rat_to_human_ortholog_mygene.csv
    ortholog_path = ROOT / "L1" / "results" / "rat_to_human_ortholog_mygene.csv"
    check_csv(ortholog_path, "3")

    # 4. disease_gene_edges.csv
    requested = ROOT / "L1" / "results" / "disease_gene_edges.csv"
    actual_l4 = ROOT / "L4" / "results" / "disease_gene_edges.csv"
    actual_v10 = ROOT / "L4" / "results_v10_minibatch" / "disease_gene_edges.csv"
    print(f"\n请求路径: {requested}")
    check_csv(requested, "4")
    print(f"\nL4/results 路径: {actual_l4}")
    df_l4 = check_csv(actual_l4, "4-L4")
    print(f"\nL4/results_v10_minibatch 路径: {actual_v10}")
    df_v10 = check_csv(actual_v10, "4-V10")

    ferro96_path = ROOT / "L1" / "results" / "ferroaging_genes_96.csv"
    if ferro96_path.exists() and df_v10 is not None:
        ferro96 = set(pd.read_csv(ferro96_path).iloc[:, 0].dropna().astype(str).str.upper())
        genes_v10 = set(df_v10["gene_symbol"].dropna().astype(str).str.upper())
        overlap = genes_v10 & ferro96
        print(f"\n    Ferroaging96 基因数: {len(ferro96)}")
        print(f"    disease_gene_edges 基因数: {len(genes_v10)}")
        print(f"    与 Ferroaging96 交集: {len(overlap)}")
        print(f"    交集中不在 Ferroaging96 的: {len(genes_v10 - ferro96)}")
        print(f"    是否仅基于 Ferroaging96 交集: {'否' if len(genes_v10 - ferro96) > 0 else '是'}")
        if df_v10 is not None and "source" in df_v10.columns:
            print(f"    source 分布:\n{df_v10['source'].value_counts().to_string()}")
        if df_v10 is not None and "evidence" in df_v10.columns:
            print(f"    evidence 分布:\n{df_v10['evidence'].value_counts().to_string()}")

    # 5. sage_best_v28.pt
    requested = ROOT / "L4" / "results" / "sage_best_v28.pt"
    actual = ROOT / "L4" / "results_v10_minibatch" / "sage_best_v28.pt"
    print(f"\n请求路径: {requested}")
    check_pt(requested)
    print(f"\n实际路径: {actual}")
    check_pt(actual)

    # 6. log
    log_path = ROOT / "L4" / "logs" / "phase4_v28_train.log"
    check_log(log_path)

    print("\n" + "="*70)
    print("检查完成")
    print("="*70)


if __name__ == "__main__":
    main()
