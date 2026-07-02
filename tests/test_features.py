"""测试特征工程模块。"""

from __future__ import annotations

import numpy as np

from L4.src.iron_aging_gnn.data.constants import ECFP4_NBITS, RDKIT_DESCRIPTOR_NAMES
from L4.src.iron_aging_gnn.data.features import (
    build_compound_features,
    compute_aac,
)


def test_build_compound_features_shape_and_validity():
    """化合物特征矩阵形状应正确，且不含 NaN/Inf。"""
    smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "invalid_smiles_string"]
    features, mean, std, col_mean = build_compound_features(smiles)

    assert features.shape[0] == len(smiles)
    assert features.shape[1] == ECFP4_NBITS + 167 + len(RDKIT_DESCRIPTOR_NAMES)  # ECFP4 + MACCS + RDKit
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
    assert mean is not None
    assert std is not None
    assert col_mean is not None


def test_build_compound_features_stats_reuse():
    """使用已有统计量进行标准化时应保持一致性。"""
    smiles = ["c1ccccc1", "CCO"]
    _, mean, std, col_mean = build_compound_features(smiles)
    features2, _, _, _ = build_compound_features(smiles, stats=(mean, std, col_mean))
    assert not np.isnan(features2).any()


def test_compute_aac_sums_to_one():
    """AAC 特征每行应加和为 1（非空序列）。"""
    seqs = ["ACDEFGHIKLMNPQRSTVWY", "AAAA", "XXX"]  # XXX 会被过滤
    aac = compute_aac(seqs)
    assert aac.shape == (len(seqs), 20)
    np.testing.assert_allclose(aac[0].sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(aac[1].sum(), 1.0, atol=1e-6)
    assert aac[2].sum() == 0.0
