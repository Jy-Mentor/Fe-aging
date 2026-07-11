"""分子图编码器：SMILES→RDKit 分子图→PyG Data 转换管道 + GIN/GROVER 编码器。

提供两种编码器：
  - GIN (Graph Isomorphism Network): 轻量级，纯 PyTorch Geometric 实现
  - GROVER: 预训练分子图 Transformer 封装（需要现有 grover_repo 权重）

SMILES 转换管道：
  SMILES → RDKit Mol → 原子特征 + 键特征 → PyG Data 对象 → 编码器 → 分子嵌入

参考:
  - Xu et al. (2019) "How Powerful are Graph Neural Networks?", ICLR
  - Rong et al. (2020) "Self-Supervised Graph Transformer on Large-Scale Molecular Data", NeurIPS
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_mean_pool

logger = logging.getLogger(__name__)

# RDKit 原子特征维度
_ATOM_FEAT_DIM = 74
# RDKit 键特征维度
_BOND_FEAT_DIM = 12


def _get_atom_features(atom) -> np.ndarray:
    """从 RDKit Atom 对象提取原子特征向量。

    特征维度 74:
      - 原子类型 (one-hot, 44): 前 44 个常见元素
      - 度数 (one-hot, 6): 0-5
      - 形式电荷 (one-hot, 5): -2, -1, 0, 1, 2
      - 手性 (one-hot, 4): unspecified, tetrahedral CW, tetrahedral CCW, other
      - 氢原子数 (one-hot, 5): 0-4
      - 杂化 (one-hot, 5): s, sp, sp2, sp3, sp3d
      - 芳香性 (1): 是/否
      - 环内原子 (1): 是/否
      - 原子质量 (1): 原子质量/100
      - 原子序号 (1): 原子序号/100
      - 自由基电子数 (1): 0-4/4
    """
    from rdkit import Chem

    feat = np.zeros(_ATOM_FEAT_DIM, dtype=np.float32)

    # 原子类型 (44 种常见元素)
    atomic_num = atom.GetAtomicNum()
    element_map = {
        5: 0, 6: 1, 7: 2, 8: 3, 9: 4, 11: 5, 12: 6, 13: 7, 14: 8, 15: 9,
        16: 10, 17: 11, 19: 12, 20: 13, 23: 14, 24: 15, 25: 16, 26: 17,
        27: 18, 28: 19, 29: 20, 30: 21, 33: 22, 34: 23, 35: 24, 38: 25,
        46: 26, 47: 27, 48: 28, 50: 29, 53: 30, 55: 31, 56: 32, 60: 33,
        78: 34, 79: 35, 80: 36, 82: 37, 83: 38, 92: 39, 1: 40, 3: 41,
        4: 42, 10: 43,
    }
    idx = element_map.get(atomic_num, 43)
    feat[idx] = 1.0

    # 度数 (6)
    degree = min(atom.GetDegree(), 5)
    feat[44 + degree] = 1.0

    # 形式电荷 (5)
    charge = atom.GetFormalCharge()
    charge_idx = max(0, min(4, charge + 2))
    feat[50 + charge_idx] = 1.0

    # 手性 (4)
    chiral_tag = atom.GetChiralTag()
    chiral_map = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.rdchem.ChiralType.CHI_OTHER: 3,
    }
    feat[55 + chiral_map.get(chiral_tag, 0)] = 1.0

    # 氢原子数 (5)
    n_h = min(atom.GetTotalNumHs(), 4)
    feat[59 + n_h] = 1.0

    # 杂化 (5)
    hybrid = atom.GetHybridization()
    hybrid_map = {
        Chem.rdchem.HybridizationType.S: 0,
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
    }
    feat[64 + hybrid_map.get(hybrid, 0)] = 1.0

    # 芳香性 (1)
    feat[69] = float(atom.GetIsAromatic())

    # 环内原子 (1)
    feat[70] = float(atom.IsInRing())

    # 原子质量 (1)
    feat[71] = atom.GetMass() / 100.0

    # 原子序号 (1)
    feat[72] = atomic_num / 100.0

    # 自由基电子数 (1)
    feat[73] = min(atom.GetNumRadicalElectrons(), 4) / 4.0

    return feat


def _get_bond_features(bond) -> np.ndarray:
    """从 RDKit Bond 对象提取键特征向量。

    特征维度 12:
      - 键类型 (one-hot, 5): single, double, triple, aromatic, other
      - 共轭 (1)
      - 环内键 (1)
      - 立体化学 (one-hot, 5): none, any, E, Z, cis/trans
    """
    from rdkit import Chem

    feat = np.zeros(_BOND_FEAT_DIM, dtype=np.float32)

    # 键类型 (5)
    bond_type = bond.GetBondType()
    type_map = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    feat[type_map.get(bond_type, 4)] = 1.0

    # 共轭 (1)
    feat[5] = float(bond.GetIsConjugated())

    # 环内键 (1)
    feat[6] = float(bond.IsInRing())

    # 立体化学 (5)
    stereo = bond.GetStereo()
    stereo_map = {
        Chem.rdchem.BondStereo.STEREONONE: 0,
        Chem.rdchem.BondStereo.STEREOANY: 1,
        Chem.rdchem.BondStereo.STEREOE: 2,
        Chem.rdchem.BondStereo.STEREOZ: 3,
        Chem.rdchem.BondStereo.STEREOCIS: 4,
        Chem.rdchem.BondStereo.STEREOTRANS: 4,
    }
    feat[7 + stereo_map.get(stereo, 0)] = 1.0

    return feat


def smiles_to_pyg_data(smiles: str) -> Optional[Data]:
    """将 SMILES 字符串转换为 PyG Data 对象（含原子/键特征）。

    Args:
        smiles: SMILES 字符串

    Returns:
        PyG Data 对象，若 SMILES 无效则返回 None
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"无效 SMILES: {smiles}")
        return None

    # 原子特征
    atom_feats = []
    for atom in mol.GetAtoms():
        atom_feats.append(_get_atom_features(atom))

    if not atom_feats:
        return None

    x = torch.tensor(np.array(atom_feats), dtype=torch.float32)

    # 边索引 + 键特征
    edge_index = [[], []]
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # 双向边
        edge_index[0].extend([i, j])
        edge_index[1].extend([j, i])
        bf = _get_bond_features(bond)
        edge_attr.extend([bf, bf])

    if edge_index[0]:
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)
        edge_attr_t = torch.tensor(np.array(edge_attr), dtype=torch.float32)
    else:
        # 单原子分子
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, _BOND_FEAT_DIM), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t)


def smiles_to_pyg_data_batch(smiles_list: list[str]) -> Optional[Batch]:
    """将 SMILES 列表转换为 PyG Batch 对象。

    Args:
        smiles_list: SMILES 字符串列表

    Returns:
        PyG Batch 对象，若全部无效则返回 None
    """
    data_list = []
    for smi in smiles_list:
        data = smiles_to_pyg_data(smi)
        if data is not None:
            data_list.append(data)

    if not data_list:
        logger.warning("smiles_to_pyg_data_batch: 所有 SMILES 均无效")
        return None

    return Batch.from_data_list(data_list)


class GINMolecularEncoder(nn.Module):
    """GIN 分子图编码器：将 SMILES→PyG Data→GIN 卷积→全局池化→分子嵌入。

    输入: SMILES 字符串或 PyG Data/Batch
    输出: (batch_size, out_dim) 分子嵌入向量

    用法:
        encoder = GINMolecularEncoder(hidden_dim=128, out_dim=128, num_layers=3)
        emb = encoder.encode_smiles(["CCO", "c1ccccc1"])
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        atom_feat_dim: int = _ATOM_FEAT_DIM,
        bond_feat_dim: int = _BOND_FEAT_DIM,
    ):
        """初始化 GIN 分子图编码器。

        Args:
            hidden_dim: 隐藏层维度
            out_dim: 输出嵌入维度
            num_layers: GIN 卷积层数
            dropout: Dropout 概率
            atom_feat_dim: 原子特征维度
            bond_feat_dim: 键特征维度
        """
        super().__init__()
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.out_dim = out_dim

        # 原子特征投影
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 键特征投影（已弃用：GINConv 不接受 edge_attr，保留以兼容旧检查点）
        self.bond_proj = nn.Sequential(
            nn.Linear(bond_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # GIN 卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """前向传播：GIN 卷积 + 全局平均池化。

        Args:
            data: PyG Data 或 Batch 对象

        Returns:
            (batch_size, out_dim) 分子嵌入
        """
        x = self.atom_proj(data.x)
        edge_index = data.edge_index

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = h + x  # 残差连接
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = h

        # 全局平均池化
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return self.out_proj(x)

    def encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """便捷方法：SMILES 列表 → 分子嵌入。

        Args:
            smiles_list: SMILES 字符串列表

        Returns:
            (batch_size, out_dim) 分子嵌入
        """
        data = smiles_to_pyg_data_batch(smiles_list)
        if data is None:
            raise ValueError("所有 SMILES 均无效，无法编码")
        device = next(self.parameters()).device
        data = data.to(device)
        return self.forward(data)


class GROVERMolecularEncoder(nn.Module):
    """GROVER 分子图 Transformer 编码器封装。

    封装现有 grover_repo 中的 GROVER 预训练模型，作为化合物编码器。
    需要 GROVER 预训练权重文件。

    用法:
        encoder = GROVERMolecularEncoder(pretrained_path="scripts/grover_repo/grover/pretrained/")
        emb = encoder.encode_smiles(["CCO", "c1ccccc1"])
    """

    def __init__(
        self,
        pretrained_path: str = "scripts/grover_repo/grover/pretrained/",
        out_dim: int = 640,
        freeze_pretrained: bool = True,
        model_size: str = "base",
    ):
        """初始化 GROVER 编码器。

        Args:
            pretrained_path: 预训练权重路径
            out_dim: 输出维度
            freeze_pretrained: 是否冻结预训练权重
            model_size: 模型大小 ("base" 或 "large")
        """
        super().__init__()
        self.out_dim = out_dim
        self.pretrained_path = pretrained_path
        self.freeze_pretrained = freeze_pretrained
        self.model_size = model_size

        self._grover_model = None
        self._output_proj = nn.Linear(out_dim, out_dim)

    def _load_grover(self):
        """延迟加载 GROVER 预训练模型。"""
        if self._grover_model is not None:
            return

        import sys
        from pathlib import Path

        grover_path = Path(__file__).resolve().parents[4] / "scripts" / "grover_repo"
        if str(grover_path) not in sys.path:
            sys.path.insert(0, str(grover_path))

        try:
            from grover.model.models import GROVEREmbedding
            self._grover_model = GROVEREmbedding(
                self.pretrained_path,
                model_size=self.model_size,
            )
            if self.freeze_pretrained:
                for param in self._grover_model.parameters():
                    param.requires_grad = False
            logger.info(f"GROVER 模型加载成功: {self.pretrained_path}")
        except Exception as e:
            logger.error(f"GROVER 模型加载失败: {e}")
            raise

    def forward(self, data: Batch) -> torch.Tensor:
        """前向传播：GROVER 编码 + 输出投影。

        Args:
            data: PyG Batch 对象

        Returns:
            (batch_size, out_dim) 分子嵌入
        """
        self._load_grover()
        if self._grover_model is None:
            raise RuntimeError("GROVER 模型未加载")

        emb = self._grover_model(data)
        return self._output_proj(emb)

    def encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """便捷方法：SMILES 列表 → GROVER 分子嵌入。

        Args:
            smiles_list: SMILES 字符串列表

        Returns:
            (batch_size, out_dim) 分子嵌入
        """
        data = smiles_to_pyg_data_batch(smiles_list)
        if data is None:
            raise ValueError("所有 SMILES 均无效，无法编码")
        device = next(self.parameters()).device
        data = data.to(device)
        return self.forward(data)


def create_molecular_encoder(
    encoder_type: str = "fingerprint",
    config: dict | None = None,
) -> nn.Module | None:
    """根据配置创建分子编码器。

    Args:
        encoder_type: "fingerprint" | "gin" | "grover"
        config: 编码器配置字典

    Returns:
        编码器模块，fingerprint 类型返回 None（使用传统特征提取）
    """
    cfg = config or {}

    if encoder_type == "fingerprint":
        return None

    if encoder_type == "gin":
        gin_cfg = cfg.get("gin", {})
        return GINMolecularEncoder(
            hidden_dim=gin_cfg.get("hidden_dim", 128),
            out_dim=gin_cfg.get("out_dim", 128),
            num_layers=gin_cfg.get("num_layers", 3),
            dropout=gin_cfg.get("dropout", 0.3),
            atom_feat_dim=gin_cfg.get("atom_feat_dim", _ATOM_FEAT_DIM),
            bond_feat_dim=gin_cfg.get("bond_feat_dim", _BOND_FEAT_DIM),
        )

    if encoder_type == "grover":
        grover_cfg = cfg.get("grover", {})
        return GROVERMolecularEncoder(
            pretrained_path=grover_cfg.get("pretrained_path", "scripts/grover_repo/grover/pretrained/"),
            out_dim=grover_cfg.get("out_dim", 640),
            freeze_pretrained=grover_cfg.get("freeze_pretrained", True),
            model_size=grover_cfg.get("model_size", "base"),
        )

    raise ValueError(f"不支持的 encoder_type: {encoder_type}")