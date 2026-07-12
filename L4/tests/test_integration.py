"""端到端集成测试 — 铁衰老 GNN 模块化系统

覆盖:
  1. 模块导入链
  2. 最小训练流水线（SAGE / HGT / SimpleHGN / RGCN）
  3. 验证流水线（validate_sage）
  4. 配置系统（YAML 加载 + 模型初始化）
  5. Memory Bank 集成
  6. 数据加载函数签名

所有测试使用合成数据，不依赖真实数据文件。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_VAL = -1e9
SCORE_CLAMP = 10.0

# 关闭日志噪音，加速测试
logging.basicConfig(level=logging.WARNING)
logging.getLogger("iron_aging_gnn").setLevel(logging.WARNING)


# ===========================================================================
# 1. 模块导入链
# ===========================================================================

class TestModuleImportChain:
    """验证所有关键模块可被导入。"""

    def test_import_data(self):
        import iron_aging_gnn.data as mod
        assert hasattr(mod, "loader")
        assert hasattr(mod, "features")
        assert hasattr(mod, "constants")
        assert hasattr(mod, "self_check")
        # 顶层导出
        assert callable(mod.load_cpi_data)
        assert callable(mod.load_ppi_network)
        assert callable(mod.load_kegg_pathways)
        assert callable(mod.load_tcm_pool)
        assert callable(mod.pipeline_self_check)

    def test_import_graph(self):
        import iron_aging_gnn.graph as mod
        assert callable(mod.build_graphs_and_adj)
        assert callable(mod.sample_hetero_subgraph)
        assert callable(mod.split_train_val)
        assert callable(mod.create_hetero_loader)
        assert callable(mod.create_homo_loader)
        assert hasattr(mod, "meta_path")
        assert hasattr(mod, "pyg_loaders")
        assert hasattr(mod, "validation_graphs")

    def test_import_models(self):
        import iron_aging_gnn.models as mod
        assert issubclass(mod.SAGELinkPredictor, nn.Module)
        assert issubclass(mod.HGTLinkPredictor, nn.Module)
        assert issubclass(mod.SimpleHGNLinkPredictor, nn.Module)
        assert issubclass(mod.RGCNLinkPredictor, nn.Module)
        assert callable(mod.focal_loss_with_logits)
        assert callable(mod.infonce_loss)
        assert callable(mod.compute_cpi_loss)
        assert issubclass(mod.MemoryBank, object)
        assert issubclass(mod.GraphTransformerEncoder, nn.Module)
        assert issubclass(mod.SemanticAttention, nn.Module)

    def test_import_model_submodules(self):
        import iron_aging_gnn.models.decoders as dec
        assert hasattr(dec, "MLPDecoder")
        assert hasattr(dec, "DotProductDecoder")
        assert hasattr(dec, "BilinearDecoder")
        assert hasattr(dec, "ResidueAwareBilinearDecoder")

        import iron_aging_gnn.models.ensemble_fusion as ef
        assert issubclass(ef.LearnableEnsembleFusion, nn.Module)

        import iron_aging_gnn.models.graph_transformer as gt
        assert issubclass(gt.GraphTransformerEncoder, nn.Module)
        assert issubclass(gt.GatedResidual, nn.Module)
        assert issubclass(gt.SemanticAttentionAggregation, nn.Module)

        import iron_aging_gnn.models.semantic_attention as sa
        assert issubclass(sa.SemanticAttention, nn.Module)

    def test_import_training(self):
        import iron_aging_gnn.training as mod
        assert callable(mod.train_sage)
        assert callable(mod.train_hgt)
        assert callable(mod.train_rgcn)
        assert callable(mod.train_simplehgn)
        assert hasattr(mod, "TrainingConfig")
        assert hasattr(mod, "Validator")
        assert hasattr(mod, "MemoryBankManager")
        assert hasattr(mod, "GradientMonitor")
        assert hasattr(mod, "LRSchedulerFactory")

    def test_import_pipeline(self):
        import iron_aging_gnn.pipeline as mod
        assert callable(mod.validate_sage)
        assert callable(mod.validate_hgt)
        assert callable(mod.validate_hgt_minibatch)
        assert callable(mod.validate_simplehgn)
        assert callable(mod.predict_hgt_scores)
        assert callable(mod.predict_tcm)

    def test_import_evaluation(self):
        import iron_aging_gnn.evaluation as mod
        assert callable(mod.compute_pairwise_metrics)
        assert callable(mod.compute_ranking_metrics)
        assert callable(mod.compute_roce)
        assert callable(mod.compute_bedroc)
        assert hasattr(mod, "ValidationProtocol")
        assert hasattr(mod, "ColdStartEvaluator")

    def test_import_utils(self):
        import iron_aging_gnn.utils as mod
        assert hasattr(mod, "Config")
        assert callable(mod.load_config)
        assert callable(mod.get_device)
        assert callable(mod.set_seed)
        assert callable(mod.setup_logger)
        assert hasattr(mod, "config")
        assert hasattr(mod, "device")
        assert hasattr(mod, "seed")
        assert hasattr(mod, "logging")
        import iron_aging_gnn.utils.reproducibility as rep
        assert callable(rep.generate_reproducibility_manifest)
        assert callable(rep.save_reproducibility_manifest)


# ===========================================================================
# 2. 最小训练流水线
# ===========================================================================

class TestMinimalTrainingPipeline:
    """用合成异质图构建并运行四个模型的前向 + 反向传播。"""

    # -- 合成数据尺寸 --
    N_COMPOUNDS = 10
    N_PROTEINS = 20
    N_PATHWAYS = 5
    COMP_FEAT_DIM = 200
    PROT_FEAT_DIM = 640
    HIDDEN_DIM = 64
    OUT_DIM = 64
    NUM_LAYERS = 2

    @pytest.fixture(scope="class")
    def synthetic_homo(self):
        """构建同构图张量（供 SAGE 使用）。"""
        torch.manual_seed(42)
        n_total = TestMinimalTrainingPipeline.N_COMPOUNDS + TestMinimalTrainingPipeline.N_PROTEINS
        feat_dim = max(
            TestMinimalTrainingPipeline.COMP_FEAT_DIM,
            TestMinimalTrainingPipeline.PROT_FEAT_DIM + TestMinimalTrainingPipeline.N_PATHWAYS,
        )
        x = torch.randn(n_total, feat_dim)
        # 构造随机 CPI 边
        src = torch.randint(0, TestMinimalTrainingPipeline.N_COMPOUNDS, (50,))
        dst = torch.randint(TestMinimalTrainingPipeline.N_COMPOUNDS, n_total, (50,))
        edge_index = torch.stack([src, dst], dim=0)
        return x, edge_index

    @pytest.fixture(scope="class")
    def synthetic_hetero(self):
        """构建 HeteroData（供 HGT / SimpleHGN / RGCN 使用）。"""
        torch.manual_seed(42)
        data = HeteroData()
        data["compound"].x = torch.randn(
            TestMinimalTrainingPipeline.N_COMPOUNDS,
            TestMinimalTrainingPipeline.COMP_FEAT_DIM,
        )
        data["protein"].x = torch.randn(
            TestMinimalTrainingPipeline.N_PROTEINS,
            TestMinimalTrainingPipeline.PROT_FEAT_DIM + TestMinimalTrainingPipeline.N_PATHWAYS,
        )
        data["pathway"].x = torch.randint(
            0, TestMinimalTrainingPipeline.N_PATHWAYS,
            (TestMinimalTrainingPipeline.N_PATHWAYS, 1),
        )
        # CPI 边
        data["compound", "interacts", "protein"].edge_index = torch.stack([
            torch.randint(0, TestMinimalTrainingPipeline.N_COMPOUNDS, (30,)),
            torch.randint(0, TestMinimalTrainingPipeline.N_PROTEINS, (30,)),
        ], dim=0)
        # PPI 边
        data["protein", "ppi", "protein"].edge_index = torch.stack([
            torch.randint(0, TestMinimalTrainingPipeline.N_PROTEINS, (40,)),
            torch.randint(0, TestMinimalTrainingPipeline.N_PROTEINS, (40,)),
        ], dim=0)
        # 蛋白-通路边
        data["protein", "belongs_to", "pathway"].edge_index = torch.stack([
            torch.randint(0, TestMinimalTrainingPipeline.N_PROTEINS, (15,)),
            torch.randint(0, TestMinimalTrainingPipeline.N_PATHWAYS, (15,)),
        ], dim=0)
        return data

    # ---- SAGE ----

    def test_sage_init_and_forward(self, synthetic_homo):
        from iron_aging_gnn.models import SAGELinkPredictor
        x, edge_index = synthetic_homo
        model = SAGELinkPredictor(
            comp_feat_dim=self.COMP_FEAT_DIM,
            prot_feat_dim=self.PROT_FEAT_DIM,
            n_compounds=self.N_COMPOUNDS,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            n_pathways=self.N_PATHWAYS,
            decoder_type="mlp",
        ).to(DEVICE)
        x_dev = x.to(DEVICE)
        ei_dev = edge_index.to(DEVICE)

        emb = model(x_dev, ei_dev, n_compounds=self.N_COMPOUNDS)
        assert emb.shape == (self.N_COMPOUNDS + self.N_PROTEINS, self.OUT_DIM)
        assert not torch.isnan(emb).any(), "SAGE 嵌入包含 NaN"

        # 解码测试
        comp_emb = emb[:self.N_COMPOUNDS]
        prot_emb = emb[self.N_COMPOUNDS:]
        scores = model.decode(comp_emb[:3], prot_emb[:3])
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any(), "SAGE decode 包含 NaN"

    def test_sage_backward(self, synthetic_homo):
        from iron_aging_gnn.models import SAGELinkPredictor, focal_loss_with_logits
        x, edge_index = synthetic_homo
        model = SAGELinkPredictor(
            comp_feat_dim=self.COMP_FEAT_DIM,
            prot_feat_dim=self.PROT_FEAT_DIM,
            n_compounds=self.N_COMPOUNDS,
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            n_pathways=self.N_PATHWAYS,
            decoder_type="mlp",
        ).to(DEVICE)
        x_dev = x.to(DEVICE)
        ei_dev = edge_index.to(DEVICE)

        emb = model(x_dev, ei_dev, n_compounds=self.N_COMPOUNDS)
        n_pairs = 5
        comp_idx = torch.randint(0, self.N_COMPOUNDS, (n_pairs,), device=DEVICE)
        prot_idx = torch.randint(0, self.N_PROTEINS, (n_pairs,), device=DEVICE)
        scores = model.decode(emb[comp_idx], emb[self.N_COMPOUNDS + prot_idx])
        targets = torch.rand(n_pairs, device=DEVICE).round()
        loss = focal_loss_with_logits(scores, targets, gamma=2.0, alpha=0.75)
        loss.backward()

        # 验证梯度流动（pheno_head 不在 decode 路径上，无梯度属正常）
        grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params += 1
                    assert not torch.isnan(param.grad).any(), f"SAGE 参数 {name} 梯度含 NaN"
        assert grad_params > 0, "至少应有部分参数获得梯度"

    # ---- HGT ----

    def test_hgt_init_and_forward(self, synthetic_hetero):
        from iron_aging_gnn.models import HGTLinkPredictor
        data = synthetic_hetero
        metadata = data.metadata()
        model = HGTLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_heads=2,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

        out = model(x_dict, edge_index_dict)
        assert "compound" in out
        assert "protein" in out
        assert out["compound"].shape == (self.N_COMPOUNDS, self.OUT_DIM)
        assert not torch.isnan(out["compound"]).any(), "HGT compound 嵌入含 NaN"
        assert not torch.isnan(out["protein"]).any(), "HGT protein 嵌入含 NaN"

        scores = model.decode(out["compound"][:3], out["protein"][:3])
        assert scores.shape == (3,)
        assert not torch.isnan(scores).any()

    def test_hgt_backward(self, synthetic_hetero):
        from iron_aging_gnn.models import HGTLinkPredictor, focal_loss_with_logits
        data = synthetic_hetero
        metadata = data.metadata()
        model = HGTLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_heads=2,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
        out = model(x_dict, edge_index_dict)

        n_pairs = 5
        comp_idx = torch.randint(0, self.N_COMPOUNDS, (n_pairs,), device=DEVICE)
        prot_idx = torch.randint(0, self.N_PROTEINS, (n_pairs,), device=DEVICE)
        scores = model.decode(out["compound"][comp_idx], out["protein"][prot_idx])
        targets = torch.rand(n_pairs, device=DEVICE).round()
        loss = focal_loss_with_logits(scores, targets)
        loss.backward()

        grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params += 1
                    assert not torch.isnan(param.grad).any(), f"HGT 参数 {name} 梯度含 NaN"
        assert grad_params > 0, "HGT 至少应有部分参数获得梯度"

    # ---- SimpleHGN ----

    def test_simplehgn_init_and_forward(self, synthetic_hetero):
        from iron_aging_gnn.models import SimpleHGNLinkPredictor
        data = synthetic_hetero
        metadata = data.metadata()
        model = SimpleHGNLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            num_heads=2,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

        out = model(x_dict, edge_index_dict)
        assert "compound" in out
        assert "protein" in out
        assert out["compound"].shape == (self.N_COMPOUNDS, self.OUT_DIM)
        assert not torch.isnan(out["compound"]).any(), "SimpleHGN compound 嵌入含 NaN"

        scores = model.decode(out["compound"][:3], out["protein"][:3])
        assert scores.shape == (3,)

    def test_simplehgn_backward(self, synthetic_hetero):
        from iron_aging_gnn.models import SimpleHGNLinkPredictor, focal_loss_with_logits
        data = synthetic_hetero
        metadata = data.metadata()
        model = SimpleHGNLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            num_heads=2,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
        out = model(x_dict, edge_index_dict)

        n_pairs = 5
        comp_idx = torch.randint(0, self.N_COMPOUNDS, (n_pairs,), device=DEVICE)
        prot_idx = torch.randint(0, self.N_PROTEINS, (n_pairs,), device=DEVICE)
        scores = model.decode(out["compound"][comp_idx], out["protein"][prot_idx])
        targets = torch.rand(n_pairs, device=DEVICE).round()
        loss = focal_loss_with_logits(scores, targets)
        loss.backward()

        grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params += 1
        assert grad_params > 0, "SimpleHGN 至少应有部分参数获得梯度"

    # ---- RGCN ----

    def test_rgcn_init_and_forward(self, synthetic_hetero):
        from iron_aging_gnn.models import RGCNLinkPredictor
        data = synthetic_hetero
        metadata = data.metadata()
        model = RGCNLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

        out = model(x_dict, edge_index_dict)
        assert "compound" in out
        assert "protein" in out
        assert out["compound"].shape == (self.N_COMPOUNDS, self.OUT_DIM)
        assert not torch.isnan(out["compound"]).any(), "RGCN compound 嵌入含 NaN"

        scores = model.decode(out["compound"][:3], out["protein"][:3])
        assert scores.shape == (3,)

    def test_rgcn_backward(self, synthetic_hetero):
        from iron_aging_gnn.models import RGCNLinkPredictor, focal_loss_with_logits
        data = synthetic_hetero
        metadata = data.metadata()
        model = RGCNLinkPredictor(
            hidden_dim=self.HIDDEN_DIM,
            out_dim=self.OUT_DIM,
            num_layers=self.NUM_LAYERS,
            dropout=0.3,
            metadata=metadata,
            compound_feat_dim=self.COMP_FEAT_DIM,
            node_feat_dims={"protein": self.PROT_FEAT_DIM, "pathway_count": self.N_PATHWAYS},
            decoder_type="mlp",
        ).to(DEVICE)

        x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
        out = model(x_dict, edge_index_dict)

        n_pairs = 5
        comp_idx = torch.randint(0, self.N_COMPOUNDS, (n_pairs,), device=DEVICE)
        prot_idx = torch.randint(0, self.N_PROTEINS, (n_pairs,), device=DEVICE)
        scores = model.decode(out["compound"][comp_idx], out["protein"][prot_idx])
        targets = torch.rand(n_pairs, device=DEVICE).round()
        loss = focal_loss_with_logits(scores, targets)
        loss.backward()

        grad_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    grad_params += 1
        assert grad_params > 0, "RGCN 至少应有部分参数获得梯度"


# ===========================================================================
# 3. 验证流水线
# ===========================================================================

class TestValidationPipeline:
    """测试 validate_sage 函数。"""

    N_COMPOUNDS = 10
    N_PROTEINS = 15
    COMP_FEAT_DIM = 200
    PROT_FEAT_DIM = 640
    N_PATHWAYS = 3
    HIDDEN_DIM = 64
    OUT_DIM = 64

    @pytest.fixture(scope="class")
    def sage_model_and_data(self):
        """构建 SAGE 模型 + 合成同构图 + 验证标签。"""
        from iron_aging_gnn.models import SAGELinkPredictor

        torch.manual_seed(42)
        n_total = TestValidationPipeline.N_COMPOUNDS + TestValidationPipeline.N_PROTEINS
        feat_dim = max(
            TestValidationPipeline.COMP_FEAT_DIM,
            TestValidationPipeline.PROT_FEAT_DIM + TestValidationPipeline.N_PATHWAYS,
        )
        x = torch.randn(n_total, feat_dim)
        # 构造 CPI 边（确保每个化合物至少有一条边）
        edge_list = []
        for c in range(TestValidationPipeline.N_COMPOUNDS):
            p = torch.randint(TestValidationPipeline.N_COMPOUNDS, n_total, (3,))
            for pp in p:
                edge_list.append([c, pp.item()])
        edge_index = torch.tensor(edge_list).T

        model = SAGELinkPredictor(
            comp_feat_dim=TestValidationPipeline.COMP_FEAT_DIM,
            prot_feat_dim=TestValidationPipeline.PROT_FEAT_DIM,
            n_compounds=TestValidationPipeline.N_COMPOUNDS,
            hidden_dim=TestValidationPipeline.HIDDEN_DIM,
            out_dim=TestValidationPipeline.OUT_DIM,
            num_layers=2,
            dropout=0.3,
            n_pathways=TestValidationPipeline.N_PATHWAYS,
            decoder_type="mlp",
        ).to(DEVICE)

        # 跑一次前向，让嵌入处于合理范围内
        with torch.no_grad():
            _ = model(x.to(DEVICE), edge_index.to(DEVICE), n_compounds=TestValidationPipeline.N_COMPOUNDS)

        return model, x, edge_index

    def test_validate_sage_returns_dict(self, sage_model_and_data):
        from iron_aging_gnn.pipeline import validate_sage

        model, x, edge_index = sage_model_and_data
        n_compounds = self.N_COMPOUNDS
        n_proteins = self.N_PROTEINS

        # 验证化合物索引（前 60%）
        val_compounds = set(range(0, int(n_compounds * 0.6)))
        # 每个化合物分配 1~2 个正样本蛋白（全局索引）
        all_compound_to_pos = {}
        for c in range(n_compounds):
            n_pos = 1 + (c % 2)
            all_compound_to_pos[c] = set(
                n_compounds + (c * 2 + i) % n_proteins for i in range(n_pos)
            )

        result = validate_sage(
            model=model,
            x=x,
            homo_edge_index=edge_index,
            val_compounds=val_compounds,
            all_compound_to_pos=all_compound_to_pos,
            n_compounds=n_compounds,
            device=DEVICE,
            score_clamp=SCORE_CLAMP,
            hard_neg_top_k=3,
            rand_neg_top_k=3,
            mask_val=MASK_VAL,
        )

        assert isinstance(result, dict), "validate_sage 应返回 dict"
        assert "auc" in result, "结果应包含 auc"
        assert "aupr" in result, "结果应包含 aupr"
        assert "n_valid_compounds" in result, "结果应包含 n_valid_compounds"
        assert result["n_valid_compounds"] > 0, "应有至少一个有效化合物"

        # AUC/AUPR 应在合理范围
        assert 0.0 <= result["auc"] <= 1.0, f"AUC 越界: {result['auc']}"
        assert 0.0 <= result["aupr"] <= 1.0, f"AUPR 越界: {result['aupr']}"

    def test_validate_sage_with_embeddings(self, sage_model_and_data):
        """测试 return_embeddings=True 模式。"""
        from iron_aging_gnn.pipeline import validate_sage

        model, x, edge_index = sage_model_and_data
        n_compounds = self.N_COMPOUNDS
        n_proteins = self.N_PROTEINS

        val_compounds = set(range(0, int(n_compounds * 0.6)))
        all_compound_to_pos = {
            c: {n_compounds + c % n_proteins} for c in range(n_compounds)
        }

        result = validate_sage(
            model=model,
            x=x,
            homo_edge_index=edge_index,
            val_compounds=val_compounds,
            all_compound_to_pos=all_compound_to_pos,
            n_compounds=n_compounds,
            device=DEVICE,
            score_clamp=SCORE_CLAMP,
            hard_neg_top_k=3,
            rand_neg_top_k=3,
            mask_val=MASK_VAL,
            return_embeddings=True,
        )

        assert isinstance(result, tuple) and len(result) == 2, "return_embeddings=True 应返回 (dict, tensor)"
        metrics, node_emb = result
        assert isinstance(metrics, dict)
        assert isinstance(node_emb, torch.Tensor)
        assert node_emb.shape[0] == n_compounds + n_proteins


# ===========================================================================
# 4. 配置系统
# ===========================================================================

class TestConfigurationSystem:
    """测试配置加载及模型初始化。"""

    def test_load_default_config(self):
        from iron_aging_gnn.utils import Config, load_config
        cfg = load_config(None)
        assert isinstance(cfg, Config)
        assert cfg.random_seed == 42
        assert cfg.model.hidden_dim > 0
        assert cfg.sage.epochs > 0
        assert cfg.hgt.epochs > 0
        assert cfg.simplehgn.epochs > 0

    def test_load_yaml_config(self):
        from iron_aging_gnn.utils import load_config
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip(f"配置文件不存在: {config_path}")
        cfg = load_config(str(config_path))
        assert cfg.random_seed == 42
        assert cfg.model.hidden_dim == 128
        assert cfg.model.out_dim == 128
        assert cfg.model.num_layers == 3
        assert cfg.model.decoder_type == "residue_bilinear"

    def test_create_sage_from_config(self):
        from iron_aging_gnn.utils import load_config
        from iron_aging_gnn.models import SAGELinkPredictor
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip(f"配置文件不存在: {config_path}")
        cfg = load_config(str(config_path))

        model = SAGELinkPredictor(
            comp_feat_dim=200,
            prot_feat_dim=640,
            n_compounds=10,
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            n_pathways=5,
            decoder_type=cfg.model.decoder_type,
            decoder_init_scheme=cfg.decoder.init_scheme,
            decoder_final_bias_init=cfg.decoder.final_bias_init,
            decoder_max_residue_batch=cfg.decoder.max_residue_batch,
        )
        assert model.out_dim == cfg.model.out_dim           # 128
        assert model.decoder_type == "residue_bilinear"

    def test_create_hgt_from_config(self):
        from iron_aging_gnn.utils import load_config
        from iron_aging_gnn.models import HGTLinkPredictor
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip(f"配置文件不存在: {config_path}")
        cfg = load_config(str(config_path))

        model = HGTLinkPredictor(
            hidden_dim=cfg.model.hidden_dim,
            out_dim=cfg.model.out_dim,
            num_heads=cfg.model.num_heads,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            metadata=None,
            compound_feat_dim=200,
            node_feat_dims={"protein": 640, "pathway_count": 5},
            decoder_type=cfg.model.decoder_type,
            decoder_init_scheme=cfg.decoder.init_scheme,
            decoder_final_bias_init=cfg.decoder.final_bias_init,
            decoder_max_residue_batch=cfg.decoder.max_residue_batch,
        )
        assert model.hidden_dim == cfg.model.hidden_dim  # 128
        assert model.out_dim == cfg.model.out_dim          # 128
        assert model.decoder_type == cfg.model.decoder_type

    def test_training_config_from_config(self):
        from iron_aging_gnn.utils import load_config
        from iron_aging_gnn.training import TrainingConfig
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            pytest.skip(f"配置文件不存在: {config_path}")
        cfg = load_config(str(config_path))
        tc = TrainingConfig.from_config(cfg)
        assert tc.epochs == cfg.sage.epochs
        assert tc.lr == cfg.sage.lr
        assert tc.batch_size == cfg.sage.batch_size
        assert tc.two_stage == cfg.sage.two_stage
        assert tc.memory_bank_size == cfg.memory_bank.memory_bank_size


# ===========================================================================
# 5. Memory Bank 集成
# ===========================================================================

class TestMemoryBankIntegration:
    """测试 MemoryBank 在训练场景下的行为。"""

    OUT_DIM = 64
    BANK_SIZE = 256

    @pytest.fixture
    def bank(self):
        from iron_aging_gnn.models import MemoryBank
        return MemoryBank(max_size=self.BANK_SIZE, out_dim=self.OUT_DIM, device="cpu")

    def test_init_empty(self, bank):
        assert bank.size() == 0
        assert bank.max_size == self.BANK_SIZE
        assert bank.bank.shape == (self.BANK_SIZE, self.OUT_DIM)

    def test_update_and_sample(self, bank):
        n = 50
        emb = torch.randn(n, self.OUT_DIM)
        bank.update(emb)
        assert bank.size() == n

        sampled = bank.sample(30)
        assert sampled.shape == (30, self.OUT_DIM)

        # 验证采样值来自 bank
        for i in range(30):
            found = False
            for j in range(n):
                if torch.allclose(sampled[i], emb[j]):
                    found = True
                    break
            assert found, f"采样值 {i} 不在 bank 中"

    def test_update_overflow(self, bank):
        """测试超过容量时的环绕行为。"""
        n = 300  # 超过 BANK_SIZE=256
        emb = torch.randn(n, self.OUT_DIM)
        bank.update(emb)
        assert bank.size() == self.BANK_SIZE
        assert bank.full is True

        sampled = bank.sample(100)
        assert sampled.shape == (100, self.OUT_DIM)
        assert not torch.isnan(sampled).any()

    def test_multiple_updates(self, bank):
        """测试多次更新后的 FIFO 行为。"""
        emb1 = torch.ones(10, self.OUT_DIM)
        emb2 = 2 * torch.ones(20, self.OUT_DIM)
        emb3 = 3 * torch.ones(30, self.OUT_DIM)

        bank.update(emb1)
        bank.update(emb2)
        bank.update(emb3)

        assert bank.size() == 60

        # 采样 60 个，应全部为非零
        sampled = bank.sample(60)
        assert sampled.shape == (60, self.OUT_DIM)
        assert not torch.isnan(sampled).any()
        assert (sampled.abs().sum(dim=1) > 0).all()

    def test_sample_empty_bank(self, bank):
        sampled = bank.sample(10)
        assert sampled.shape == (0, self.OUT_DIM)

    def test_training_scenario(self):
        """模拟训练循环中的 MemoryBank 更新与采样。"""
        from iron_aging_gnn.models import MemoryBank
        bank = MemoryBank(max_size=512, out_dim=self.OUT_DIM, device="cpu")

        for step in range(5):
            batch_emb = torch.randn(32, self.OUT_DIM)
            bank.update(batch_emb)

            if bank.size() > 0:
                mem = bank.sample(16)
                assert mem.shape == (min(16, bank.size()), self.OUT_DIM)

        assert bank.size() == 160


# ===========================================================================
# 6. 数据加载函数签名
# ===========================================================================

class TestDataPipelineSignatures:
    """验证数据加载函数可导入且符合预期签名。"""

    def test_load_cpi_data_signature(self):
        import inspect
        from iron_aging_gnn.data import load_cpi_data
        sig = inspect.signature(load_cpi_data)
        params = list(sig.parameters.keys())
        assert len(params) == 0, f"load_cpi_data 应为无参函数，实际参数: {params}"

    def test_load_ppi_network_signature(self):
        import inspect
        from iron_aging_gnn.data import load_ppi_network
        sig = inspect.signature(load_ppi_network)
        params = list(sig.parameters.keys())
        assert len(params) == 0, f"load_ppi_network 应为无参函数，实际参数: {params}"

    def test_load_kegg_pathways_signature(self):
        import inspect
        from iron_aging_gnn.data import load_kegg_pathways
        sig = inspect.signature(load_kegg_pathways)
        params = list(sig.parameters.keys())
        assert len(params) == 0, f"load_kegg_pathways 应为无参函数，实际参数: {params}"

    def test_load_tcm_pool_signature(self):
        import inspect
        from iron_aging_gnn.data import load_tcm_pool
        sig = inspect.signature(load_tcm_pool)
        params = list(sig.parameters.keys())
        assert len(params) == 0, f"load_tcm_pool 应为无参函数，实际参数: {params}"


# ===========================================================================
# 7. 端到端烟雾测试
# ===========================================================================

class TestEndToEndSmoke:
    """快速端到端烟雾测试：模型构建 → 前向 → 损失 → 反向 → 验证。"""

    def test_full_sage_pipeline(self):
        from iron_aging_gnn.models import SAGELinkPredictor, focal_loss_with_logits
        from iron_aging_gnn.pipeline import validate_sage

        torch.manual_seed(42)
        n_comp = 8
        n_prot = 12
        comp_dim = 200
        prot_dim = 640
        n_path = 3
        feat_dim = max(comp_dim, prot_dim + n_path)
        n_total = n_comp + n_prot

        x = torch.randn(n_total, feat_dim)
        edge_index = torch.stack([
            torch.randint(0, n_comp, (20,)),
            torch.randint(n_comp, n_total, (20,)),
        ], dim=0)

        model = SAGELinkPredictor(
            comp_feat_dim=comp_dim,
            prot_feat_dim=prot_dim,
            n_compounds=n_comp,
            hidden_dim=32,
            out_dim=32,
            num_layers=2,
            dropout=0.2,
            n_pathways=n_path,
            decoder_type="mlp",
        ).to(DEVICE)

        # 训练步骤
        x_dev = x.to(DEVICE)
        ei_dev = edge_index.to(DEVICE)
        emb = model(x_dev, ei_dev, n_compounds=n_comp)
        n_pairs = 4
        scores = model.decode(
            emb[torch.randint(0, n_comp, (n_pairs,), device=DEVICE)],
            emb[n_comp + torch.randint(0, n_prot, (n_pairs,), device=DEVICE)],
        )
        loss = focal_loss_with_logits(scores, torch.ones(n_pairs, device=DEVICE))
        loss.backward()

        # 验证步骤
        val_compounds = set(range(0, n_comp // 2))
        all_compound_to_pos = {c: {n_comp + c % n_prot} for c in range(n_comp)}
        result = validate_sage(
            model=model, x=x, homo_edge_index=edge_index,
            val_compounds=val_compounds,
            all_compound_to_pos=all_compound_to_pos,
            n_compounds=n_comp,
            device=DEVICE, score_clamp=SCORE_CLAMP,
            hard_neg_top_k=2, rand_neg_top_k=2, mask_val=MASK_VAL,
        )
        assert "auc" in result
        assert "aupr" in result
        assert result["n_valid_compounds"] > 0


# ===========================================================================
# 8. 工具函数集成
# ===========================================================================

class TestUtilsIntegration:
    """测试工具函数在集成场景下的行为。"""

    def test_seed_reproducibility(self):
        """set_seed 后两次运行应产生相同结果。"""
        from iron_aging_gnn.utils import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b), "set_seed 后应可复现"

    def test_get_device(self):
        from iron_aging_gnn.utils import get_device
        dev = get_device()
        assert isinstance(dev, torch.device)
        dev_cpu = get_device("cpu")
        assert dev_cpu == torch.device("cpu")

    def test_setup_logger(self):
        import tempfile
        from iron_aging_gnn.utils import setup_logger
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = setup_logger("test_integration_log", log_path, level=logging.DEBUG)
            logger.info("集成测试日志")
            assert log_path.exists()
            content = log_path.read_text(encoding="utf-8")
            assert "集成测试日志" in content
            # 关闭 handler 以释放文件句柄（Windows 下 tempdir 清理需要）
            for h in logger.handlers[:]:
                h.close()
                logger.removeHandler(h)

    def test_config_defaults(self):
        from iron_aging_gnn.utils import Config
        cfg = Config()
        assert cfg.random_seed == 42
        assert cfg.model.hidden_dim == 64
        assert cfg.model.out_dim == 64
        assert cfg.model.num_layers == 2
        assert cfg.loss.focal_gamma == 2.0
        assert cfg.loss.focal_alpha == 0.75
        assert cfg.validation.val_batch_size == 512
        assert cfg.memory_bank.memory_bank_size == 8192
        assert len(cfg.ferrogenesis_genes) == 96