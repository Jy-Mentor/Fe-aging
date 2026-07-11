"""Meta-path construction for heterogeneous graph neural networks.

Reference:
  - DHGT-DTI (2025): Global-view meta-path construction, filtering overly broad paths
  - MHGNN-DTI (2023): Intra-meta-path + inter-meta-path aggregation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

EdgeType = Tuple[str, str, str]

DEFAULT_DRUG_META_PATHS: Dict[str, List[EdgeType]] = {
    "C-P-C": [
        ("compound", "interacts", "protein"),
        ("protein", "interacts", "compound"),
    ],
    "C-P-P-C": [
        ("compound", "interacts", "protein"),
        ("protein", "ppi", "protein"),
        ("protein", "interacts", "compound"),
    ],
}

DEFAULT_PROTEIN_META_PATHS: Dict[str, List[EdgeType]] = {
    "P-P": [
        ("protein", "ppi", "protein"),
    ],
    "P-C-P": [
        ("protein", "interacts", "compound"),
        ("compound", "interacts", "protein"),
    ],
    "P-P-P": [
        ("protein", "ppi", "protein"),
        ("protein", "ppi", "protein"),
    ],
}


class MetaPathBuilder:
    """Build meta-path graphs for heterogeneous DTI networks.

    Constructs meta-path adjacency matrices via sparse matrix multiplication
    of edge type adjacency matrices. Filters overly dense meta-paths that
    would connect almost all node pairs (DHGT-DTI, 2025).

    Args:
        edge_index_dict: {edge_type: edge_index (2, E)} mapping.
        num_nodes_dict: {node_type: count} mapping.
        density_threshold: Maximum allowed density before filtering (default 0.1).
    """

    def __init__(
        self,
        edge_index_dict: Dict[EdgeType, torch.Tensor],
        num_nodes_dict: Dict[str, int],
        density_threshold: float = 0.1,
    ):
        self._edge_index_dict = edge_index_dict
        self._num_nodes_dict = num_nodes_dict
        self._density_threshold = density_threshold

        self._adj_cache: Dict[EdgeType, torch.Tensor] = {}
        self._meta_path_graphs: Dict[str, torch.Tensor] = {}

    def _get_adj(self, edge_type: EdgeType) -> torch.Tensor:
        """Resolve adjacency matrix for an edge type, with caching.

        If the edge type is not directly in edge_index_dict, attempts to
        use the transpose of the forward direction (swap src/dst).
        """
        if edge_type in self._adj_cache:
            return self._adj_cache[edge_type]

        src_type, rel, dst_type = edge_type
        n_src = self._num_nodes_dict.get(src_type, 0)
        n_dst = self._num_nodes_dict.get(dst_type, 0)

        if edge_type in self._edge_index_dict:
            ei = self._edge_index_dict[edge_type]
            if ei.shape[1] > 0:
                adj = torch.sparse_coo_tensor(
                    ei, torch.ones(ei.shape[1], device=ei.device), (n_src, n_dst)
                ).coalesce()
            else:
                adj = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0),
                    (n_src, n_dst),
                ).coalesce()
        else:
            fwd = (dst_type, rel, src_type)
            if fwd in self._edge_index_dict:
                ei = self._edge_index_dict[fwd]
                if ei.shape[1] > 0:
                    adj = torch.sparse_coo_tensor(
                        torch.stack([ei[1], ei[0]]),
                        torch.ones(ei.shape[1], device=ei.device),
                        (n_src, n_dst),
                    ).coalesce()
                else:
                    adj = torch.sparse_coo_tensor(
                        torch.zeros((2, 0), dtype=torch.long),
                        torch.zeros(0),
                        (n_src, n_dst),
                    ).coalesce()
            else:
                adj = torch.sparse_coo_tensor(
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros(0),
                    (n_src, n_dst),
                ).coalesce()

        self._adj_cache[edge_type] = adj
        return adj

    def _edge_type_exists(self, edge_type: EdgeType) -> bool:
        """Check whether an edge type has at least one edge."""
        if edge_type in self._edge_index_dict:
            return self._edge_index_dict[edge_type].shape[1] > 0
        src_type, rel, dst_type = edge_type
        fwd = (dst_type, rel, src_type)
        if fwd in self._edge_index_dict:
            return self._edge_index_dict[fwd].shape[1] > 0
        return False

    def _compute_density(self, adj: torch.Tensor) -> float:
        """Compute density (fraction of non-zero entries) of adjacency matrix."""
        nnz = adj._nnz()
        if nnz == 0:
            return 0.0
        total = adj.shape[0] * adj.shape[1]
        if total == 0:
            return 0.0
        return nnz / total

    def _build_meta_path_adj(self, meta_path: List[EdgeType]) -> torch.Tensor:
        """Build meta-path adjacency via sparse matrix chain multiplication.

        Args:
            meta_path: Ordered list of edge types defining the meta-path.

        Returns:
            Sparse adjacency matrix from first source type to last target type.
        """
        if not meta_path:
            raise ValueError("meta_path must not be empty")

        adj = self._get_adj(meta_path[0])
        for edge_type in meta_path[1:]:
            adj_next = self._get_adj(edge_type)
            adj = torch.sparse.mm(adj, adj_next).coalesce()
        return adj

    def build_meta_path_graph(
        self,
        meta_path: List[EdgeType],
        name: str,
        force: bool = False,
    ) -> Optional[torch.Tensor]:
        """Build a single meta-path graph.

        Args:
            meta_path: Ordered list of edge type tuples.
            name: Human-readable name (e.g. "C-P-C").
            force: If True, skip density filtering.

        Returns:
            Sparse adjacency matrix, or None if filtered out or failed.
        """
        meta_path_len = len(meta_path)
        if meta_path_len < 1 or meta_path_len > 5:
            logger.debug(
                "Skip meta-path '%s': length=%d (outside [1, 5])",
                name, meta_path_len,
            )
            return None

        for edge_type in meta_path:
            if not self._edge_type_exists(edge_type):
                logger.debug(
                    "Skip meta-path '%s': edge type %s has no edges",
                    name, edge_type,
                )
                return None

        try:
            adj = self._build_meta_path_adj(meta_path)
        except Exception:
            logger.exception("Failed to build meta-path '%s'", name)
            return None

        if not force:
            density = self._compute_density(adj)
            if density > self._density_threshold:
                logger.info(
                    "Filter meta-path '%s': density=%.4f > threshold=%.4f",
                    name, density, self._density_threshold,
                )
                return None

        self._meta_path_graphs[name] = adj
        nnz = adj._nnz()
        logger.info(
            "Meta-path '%s': %d edges, shape=%s, density=%.6f",
            name, nnz, adj.shape, self._compute_density(adj),
        )
        return adj

    def build_all_meta_paths(self) -> Dict[str, torch.Tensor]:
        """Build all predefined compound and protein meta-paths.

        Returns:
            Dict mapping meta-path names to sparse adjacency matrices.
        """
        for name, meta_path in DEFAULT_DRUG_META_PATHS.items():
            self.build_meta_path_graph(meta_path, name)

        for name, meta_path in DEFAULT_PROTEIN_META_PATHS.items():
            self.build_meta_path_graph(meta_path, name)

        return self._meta_path_graphs

    def get_meta_path_graph(self, name: str) -> Optional[torch.Tensor]:
        """Get the sparse adjacency matrix for a named meta-path."""
        return self._meta_path_graphs.get(name)

    def get_edge_index(self, name: str) -> Optional[torch.Tensor]:
        """Convert meta-path sparse adjacency to edge_index (2, E) format."""
        adj = self._meta_path_graphs.get(name)
        if adj is None:
            return None
        return adj.indices()

    @property
    def meta_path_names(self) -> List[str]:
        return list(self._meta_path_graphs.keys())