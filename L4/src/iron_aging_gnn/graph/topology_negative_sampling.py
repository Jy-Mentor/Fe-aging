"""基于PPI网络拓扑的难负样本挖掘。

替代原 _compute_cpi_loss 中基于 KEGG 通路共现的中度负样本策略，
利用度中心性、聚类系数、Betweenness centrality 以及共同邻居/Jaccard相似度
对负样本进行拓扑难度分级。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TopologyNegativeSampler:
    """基于PPI拓扑的负样本难度分级采样器。

    输入PPI边表，自动计算以下拓扑指标：
    - 度中心性（degree centrality）
    - 聚类系数（clustering coefficient）
    - Betweenness centrality

    并基于这些指标将候选负样本分为易/中/难三级：
    - 易：与正样本拓扑距离远、度差异大。
    - 中：拓扑指标与正样本接近但无直接交互、无共同邻居。
    - 难：与正样本有共同邻居或高Jaccard相似度。
    """

    def __init__(
        self,
        ppi_df: pd.DataFrame,
        source_col: str = "gene_a",
        target_col: str = "gene_b",
        weight_col: str | None = "combined_score",
        exact_betweenness: bool = False,
        betweenness_samples: int = 500,
        precompute_distances: bool = True,
        seed: int = 42,
    ) -> None:
        """初始化并预计算拓扑指标。

        Args:
            ppi_df: PPI边表，至少包含source_col/target_col两列。
            source_col: 边起点列名。
            target_col: 边终点列名。
            weight_col: 边权重列名；若为None则构建无权图。
            exact_betweenness: 是否精确计算betweenness；大图上较慢。
            betweenness_samples: 近似betweenness的采样数，仅当
                exact_betweenness=False时生效。
            precompute_distances: 是否预计算全对最短路径，加速后续查询。
            seed: betweenness近似采样的随机种子。
        """
        self.ppi_df = ppi_df.copy()
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col

        self.graph = self._build_graph()
        self.nodes = sorted(self.graph.nodes())
        self.node_set: set[str] = set(self.nodes)
        self.neighbors: dict[str, set[str]] = {
            n: set(self.graph.neighbors(n)) for n in self.nodes
        }

        logger.info(
            f"TopologyNegativeSampler: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

        self.degree: dict[str, int] = dict(self.graph.degree())
        self.degree_centrality: dict[str, float] = nx.degree_centrality(self.graph)
        self.clustering: dict[str, float] = nx.clustering(self.graph)

        n_nodes = self.graph.number_of_nodes()
        if exact_betweenness or n_nodes <= 500:
            logger.info("Computing exact betweenness centrality ...")
            self.betweenness: dict[str, float] = nx.betweenness_centrality(
                self.graph, normalized=True
            )
        else:
            k = min(betweenness_samples, n_nodes)
            logger.info(f"Computing approximate betweenness centrality (k={k}) ...")
            self.betweenness = nx.betweenness_centrality(
                self.graph, k=k, normalized=True, seed=seed
            )

        self._feature_vector: dict[str, np.ndarray] = self._build_feature_vectors()

        self._distances: dict[str, dict[str, int]] | None = None
        if precompute_distances:
            logger.info("Precomputing all-pairs shortest path lengths ...")
            self._distances = dict(nx.all_pairs_shortest_path_length(self.graph))

    def _build_graph(self) -> nx.Graph:
        """根据PPI边表构建无向图。"""
        g = nx.Graph()
        for _, row in self.ppi_df.iterrows():
            src = str(row[self.source_col]).strip().upper()
            tgt = str(row[self.target_col]).strip().upper()
            if not src or not tgt or src == tgt:
                continue
            if self.weight_col and self.weight_col in row:
                w = float(row[self.weight_col])
                g.add_edge(src, tgt, weight=w)
            else:
                g.add_edge(src, tgt)
        return g

    def _build_feature_vectors(self) -> dict[str, np.ndarray]:
        """将三种拓扑指标拼接并标准化为特征向量。"""
        features: dict[str, np.ndarray] = {}
        values = np.array(
            [
                [self.degree_centrality[n], self.clustering[n], self.betweenness[n]]
                for n in self.nodes
            ],
            dtype=np.float64,
        )
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std[std == 0] = 1.0
        normalized = (values - mean) / std
        for i, n in enumerate(self.nodes):
            features[n] = normalized[i]
        return features

    def feature_distance(self, a: str, b: str) -> float:
        """计算两个蛋白拓扑特征向量的欧氏距离。"""
        if a not in self._feature_vector or b not in self._feature_vector:
            return np.inf
        return float(np.linalg.norm(self._feature_vector[a] - self._feature_vector[b]))

    def jaccard_similarity(self, a: str, b: str) -> float:
        """计算两个蛋白邻居集合的Jaccard相似度。"""
        na = self.neighbors.get(a, set())
        nb = self.neighbors.get(b, set())
        union = na | nb
        if not union:
            return 0.0
        return len(na & nb) / len(union)

    def common_neighbor_count(self, a: str, b: str) -> int:
        """计算两个蛋白的共同邻居数。"""
        return len(self.neighbors.get(a, set()) & self.neighbors.get(b, set()))

    def topological_distance(self, a: str, b: str) -> int:
        """计算最短路径长度；不连通返回一个极大值。"""
        if self._distances is not None and a in self._distances:
            return self._distances[a].get(b, 10_000)
        try:
            return nx.shortest_path_length(self.graph, source=a, target=b)
        except nx.NetworkXNoPath:
            return 10_000

    def difficulty_scores(self, positive_protein: str) -> pd.DataFrame:
        """为单个正样本蛋白计算全图候选负样本的拓扑难度分数。

        返回DataFrame包含：
        - protein: 候选负样本蛋白名
        - degree_centrality, clustering, betweenness: 候选蛋白拓扑指标
        - shortest_path: 与正样本的最短路径长度
        - jaccard: 邻居集合Jaccard相似度
        - common_neighbors: 共同邻居数
        - feature_distance: 拓扑特征向量距离
        - hard_score: 难负样本分数（共同邻居+Jaccard+路径倒数）
        - medium_score: 中度负样本分数（特征距离越小分数越高）
        - easy_score: 易负样本分数（距离越远、度差异越大分数越高）
        """
        if positive_protein not in self.node_set:
            logger.warning(
                f"正样本蛋白 {positive_protein} 不在PPI网络中，无法计算拓扑负样本"
            )
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        pos_deg = self.degree.get(positive_protein, 0)

        for neg in self.nodes:
            if neg == positive_protein:
                continue
            # 排除直接交互：直接交互的不是合法负样本
            if neg in self.neighbors.get(positive_protein, set()):
                continue

            sp = self.topological_distance(positive_protein, neg)
            jac = self.jaccard_similarity(positive_protein, neg)
            cn = self.common_neighbor_count(positive_protein, neg)
            fd = self.feature_distance(positive_protein, neg)
            neg_deg = self.degree.get(neg, 0)

            # 难：共同邻居越多、Jaccard越高、路径越短越难
            hard_score = cn + 10.0 * jac + 1.0 / max(sp, 1)

            # 中：拓扑特征越接近越中，但必须无共同邻居（否则应归为难）
            medium_score = 1.0 / (1.0 + fd) if fd != np.inf else 0.0
            if cn > 0:
                medium_score *= 0.1  # 有共同邻居的降低中度权重

            # 易：拓扑距离远、度差异大
            easy_score = float(sp) + abs(neg_deg - pos_deg) / max(pos_deg, 1)

            rows.append(
                {
                    "protein": neg,
                    "degree_centrality": self.degree_centrality.get(neg, 0.0),
                    "clustering": self.clustering.get(neg, 0.0),
                    "betweenness": self.betweenness.get(neg, 0.0),
                    "degree": neg_deg,
                    "shortest_path": 10_000 if sp == 10_000 else sp,
                    "jaccard": jac,
                    "common_neighbors": cn,
                    "feature_distance": fd,
                    "hard_score": hard_score,
                    "medium_score": medium_score,
                    "easy_score": easy_score,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values(
            by=["hard_score", "medium_score", "easy_score"],
            ascending=[False, False, False],
        )
        return df

    def classify_negatives(
        self,
        positive_protein: str,
        hard_top_k: int = 20,
        medium_top_k: int = 20,
        easy_top_k: int = 20,
    ) -> dict[str, pd.DataFrame]:
        """为单个正样本蛋白按拓扑难度分级返回负样本候选。

        Args:
            positive_protein: 正样本蛋白基因名。
            hard_top_k: 难负样本返回数量。
            medium_top_k: 中度负样本返回数量。
            easy_top_k: 易负样本返回数量。

        Returns:
            {'hard': DataFrame, 'medium': DataFrame, 'easy': DataFrame}
        """
        df = self.difficulty_scores(positive_protein)
        if df.empty:
            return {"hard": df.copy(), "medium": df.copy(), "easy": df.copy()}

        # 难：按 hard_score 排序
        hard_df = df.sort_values("hard_score", ascending=False).head(hard_top_k)

        # 中：无共同邻居且特征距离小
        medium_candidates = df[df["common_neighbors"] == 0].sort_values(
            "medium_score", ascending=False
        )
        medium_df = medium_candidates.head(medium_top_k)

        # 易：拓扑距离最远、度差异大，且排除已被中/难覆盖的
        easy_candidates = df.sort_values("easy_score", ascending=False)
        easy_df = easy_candidates.head(easy_top_k)

        return {"hard": hard_df, "medium": medium_df, "easy": easy_df}

    def summarize_topology(self) -> pd.DataFrame:
        """返回全图蛋白的拓扑指标分布统计。"""
        rows = []
        for n in self.nodes:
            rows.append(
                {
                    "protein": n,
                    "degree": self.degree[n],
                    "degree_centrality": self.degree_centrality[n],
                    "clustering": self.clustering[n],
                    "betweenness": self.betweenness[n],
                }
            )
        return pd.DataFrame(rows)


def build_topology_medium_neighbors(
    ppi_df: pd.DataFrame,
    gene_to_idx: dict[str, int],
    n_compounds: int,
    active_genes: set[str] | None = None,
    top_k: int = 50,
    sampler: TopologyNegativeSampler | None = None,
    exact_betweenness: bool = False,
    betweenness_samples: int = 500,
) -> dict[int, set[int]]:
    """构建拓扑中度负样本邻居表，接口与 build_pathway_neighbors 一致。

    对每个蛋白，返回与其拓扑特征最接近但无直接交互、无共同邻居的蛋白集合。

    Args:
        ppi_df: PPI边表。
        gene_to_idx: 基因名 -> 全局节点索引。
        n_compounds: 化合物数量，用于将全局索引转换为蛋白局部索引。
        active_genes: 若提供，仅针对这些正样本基因预计算邻居，减少计算量。
        top_k: 每个蛋白保留的中度候选数量。
        sampler: 若提供，复用已有采样器避免重复计算拓扑指标。
        exact_betweenness: 是否精确计算betweenness（新建sampler时生效）。
        betweenness_samples: 近似betweenness采样数（新建sampler时生效）。

    Returns:
        {蛋白局部索引: set(拓扑中度负样本局部索引)}
    """
    if sampler is None:
        sampler = TopologyNegativeSampler(
            ppi_df,
            exact_betweenness=exact_betweenness,
            betweenness_samples=betweenness_samples,
        )
    result: dict[int, set[int]] = defaultdict(set)

    genes_to_process = active_genes if active_genes else sampler.node_set
    for gene in genes_to_process:
        if gene not in gene_to_idx:
            continue
        g_local = gene_to_idx[gene] - n_compounds
        if g_local < 0:
            continue

        df = sampler.difficulty_scores(gene)
        if df.empty:
            continue

        medium_candidates = df[df["common_neighbors"] == 0].sort_values(
            "medium_score", ascending=False
        )
        for neg in medium_candidates.head(top_k)["protein"]:
            if neg not in gene_to_idx:
                continue
            neg_local = gene_to_idx[neg] - n_compounds
            if neg_local >= 0 and neg_local != g_local:
                result[g_local].add(neg_local)

    logger.info(
        f"Topology medium neighbors: {len(result)} proteins with median "
        f"{int(np.median([len(v) for v in result.values()])) if result else 0} candidates"
    )
    return result


def build_topology_hard_neighbors(
    ppi_df: pd.DataFrame,
    gene_to_idx: dict[str, int],
    n_compounds: int,
    active_genes: set[str] | None = None,
    top_k: int = 50,
    sampler: TopologyNegativeSampler | None = None,
    exact_betweenness: bool = False,
    betweenness_samples: int = 500,
) -> dict[int, set[int]]:
    """构建拓扑难负样本邻居表，接口与 build_pathway_neighbors 一致。

    对每个蛋白，返回与其有共同邻居或高Jaccard相似度的蛋白集合。

    Args:
        ppi_df: PPI边表。
        gene_to_idx: 基因名 -> 全局节点索引。
        n_compounds: 化合物数量。
        active_genes: 若提供，仅针对这些正样本基因预计算邻居。
        top_k: 每个蛋白保留的难候选数量。
        sampler: 若提供，复用已有采样器避免重复计算拓扑指标。
        exact_betweenness: 是否精确计算betweenness（新建sampler时生效）。
        betweenness_samples: 近似betweenness采样数（新建sampler时生效）。

    Returns:
        {蛋白局部索引: set(拓扑难负样本局部索引)}
    """
    if sampler is None:
        sampler = TopologyNegativeSampler(
            ppi_df,
            exact_betweenness=exact_betweenness,
            betweenness_samples=betweenness_samples,
        )
    result: dict[int, set[int]] = defaultdict(set)

    genes_to_process = active_genes if active_genes else sampler.node_set
    for gene in genes_to_process:
        if gene not in gene_to_idx:
            continue
        g_local = gene_to_idx[gene] - n_compounds
        if g_local < 0:
            continue

        df = sampler.difficulty_scores(gene)
        if df.empty:
            continue

        hard_candidates = df[df["common_neighbors"] > 0].sort_values(
            "hard_score", ascending=False
        )
        for neg in hard_candidates.head(top_k)["protein"]:
            if neg not in gene_to_idx:
                continue
            neg_local = gene_to_idx[neg] - n_compounds
            if neg_local >= 0 and neg_local != g_local:
                result[g_local].add(neg_local)

    logger.info(
        f"Topology hard neighbors: {len(result)} proteins with median "
        f"{int(np.median([len(v) for v in result.values()])) if result else 0} candidates"
    )
    return result
