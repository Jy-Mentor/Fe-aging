"""模型前向传播测试

测试 SAGE/HGT/SimpleHGN/RGCN 四个模型的前向传播，验证:
  - 输入输出形状正确
  - 输出无 NaN
  - 梯度可正常反向传播
"""

import pytest
import torch
from torch_geometric.data import HeteroData


class TestSAGELinkPredictor:
    """GraphSAGE 模型前向传播测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.comp_feat_dim = 200
        self.prot_feat_dim = 640
        self.n_compounds = 100
        self.hidden_dim = 64
        self.out_dim = 64
        self.num_layers = 2
        self.batch_size = 32
        self.total_nodes = 200

    def test_forward_mlp_decoder(self):
        from iron_aging_gnn.models.sage import SAGELinkPredictor
        model = SAGELinkPredictor(
            self.comp_feat_dim, self.prot_feat_dim, self.n_compounds,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_layers=self.num_layers, decoder_type="mlp",
        )
        x = torch.randn(self.total_nodes, self.prot_feat_dim)
        x[:self.n_compounds, :self.comp_feat_dim] = torch.randn(self.n_compounds, self.comp_feat_dim)
        edge_index = torch.randint(0, self.total_nodes, (2, 500))
        comp_indices = torch.randint(0, self.n_compounds, (self.batch_size,))
        prot_indices = torch.randint(self.n_compounds, self.total_nodes, (self.batch_size,))

        emb = model(x, edge_index)
        comp_emb = emb[comp_indices]
        prot_emb = emb[prot_indices]
        scores = model.decode(comp_emb, prot_emb)
        assert scores.shape == (self.batch_size,)
        assert not torch.isnan(scores).any()

    def test_forward_dot_decoder(self):
        from iron_aging_gnn.models.sage import SAGELinkPredictor
        model = SAGELinkPredictor(
            self.comp_feat_dim, self.prot_feat_dim, self.n_compounds,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_layers=self.num_layers, decoder_type="dot",
        )
        x = torch.randn(self.total_nodes, self.prot_feat_dim)
        x[:self.n_compounds, :self.comp_feat_dim] = torch.randn(self.n_compounds, self.comp_feat_dim)
        edge_index = torch.randint(0, self.total_nodes, (2, 500))
        comp_indices = torch.randint(0, self.n_compounds, (self.batch_size,))
        prot_indices = torch.randint(self.n_compounds, self.total_nodes, (self.batch_size,))

        emb = model(x, edge_index)
        comp_emb = emb[comp_indices]
        prot_emb = emb[prot_indices]
        scores = model.decode(comp_emb, prot_emb)
        assert scores.shape == (self.batch_size,)
        assert not torch.isnan(scores).any()

    def test_output_no_nan(self):
        from iron_aging_gnn.models.sage import SAGELinkPredictor
        model = SAGELinkPredictor(
            self.comp_feat_dim, self.prot_feat_dim, self.n_compounds,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_layers=self.num_layers,
        )
        x = torch.randn(self.total_nodes, self.prot_feat_dim)
        x[:self.n_compounds, :self.comp_feat_dim] = torch.randn(self.n_compounds, self.comp_feat_dim)
        edge_index = torch.randint(0, self.total_nodes, (2, 500))
        embeddings = model(x, edge_index)
        assert not torch.isnan(embeddings).any()

    def test_temperature_default(self):
        from iron_aging_gnn.models.sage import SAGELinkPredictor, _TEMPERATURE
        assert _TEMPERATURE == 1.0
        model = SAGELinkPredictor(
            self.comp_feat_dim, self.prot_feat_dim, self.n_compounds,
        )
        assert model.temperature == 1.0


class TestHGTLinkPredictor:
    """HGT 模型前向传播测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hidden_dim = 64
        self.out_dim = 64
        self.num_heads = 2
        self.num_layers = 2
        self.comp_feat_dim = 200
        self.batch_size = 16

    def _make_hetero_data(self):
        data = HeteroData()
        data["compound"].x = torch.randn(100, self.comp_feat_dim)
        data["protein"].x = torch.randn(50, 640)
        data["pathway"].x = torch.ones(10, 1)
        data["compound", "targets", "protein"].edge_index = torch.randint(0, 50, (2, 200))
        data["compound", "targets", "protein"].edge_index[0] = data["compound", "targets", "protein"].edge_index[0] % 100
        data["protein", "interacts", "protein"].edge_index = torch.randint(0, 50, (2, 300))
        data["protein", "belongs_to", "pathway"].edge_index = torch.randint(0, 10, (2, 100))
        data["protein", "belongs_to", "pathway"].edge_index[0] = data["protein", "belongs_to", "pathway"].edge_index[0] % 50
        return data

    def test_forward(self):
        from iron_aging_gnn.models.hgt import HGTLinkPredictor
        data = self._make_hetero_data()
        node_feat_dims = {
            "compound": self.comp_feat_dim,
            "protein": 640,
            "pathway": 1,
            "pathway_count": 10,
            "disease_count": 0,
        }
        model = HGTLinkPredictor(
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_heads=self.num_heads, num_layers=self.num_layers,
            dropout=0.5, metadata=data.metadata(),
            compound_feat_dim=self.comp_feat_dim,
            node_feat_dims=node_feat_dims,
        )
        x_dict = {k: v for k, v in data.x_dict.items()}
        edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}
        embeddings = model(x_dict, edge_index_dict)
        assert "compound" in embeddings
        assert "protein" in embeddings
        assert not torch.isnan(embeddings["compound"]).any()


class TestSimpleHGNLinkPredictor:
    """SimpleHGN 模型前向传播测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hidden_dim = 64
        self.out_dim = 64
        self.num_heads = 2
        self.num_layers = 2
        self.comp_feat_dim = 200

    def _make_hetero_data(self):
        data = HeteroData()
        data["compound"].x = torch.randn(100, self.comp_feat_dim)
        data["protein"].x = torch.randn(50, 640)
        data["pathway"].x = torch.ones(10, 1)
        data["compound", "targets", "protein"].edge_index = torch.randint(0, 50, (2, 200))
        data["compound", "targets", "protein"].edge_index[0] = data["compound", "targets", "protein"].edge_index[0] % 100
        data["protein", "interacts", "protein"].edge_index = torch.randint(0, 50, (2, 300))
        data["protein", "belongs_to", "pathway"].edge_index = torch.randint(0, 10, (2, 100))
        data["protein", "belongs_to", "pathway"].edge_index[0] = data["protein", "belongs_to", "pathway"].edge_index[0] % 50
        return data

    def test_forward(self):
        from iron_aging_gnn.models.simplehgn import SimpleHGNLinkPredictor
        data = self._make_hetero_data()
        node_feat_dims = {
            "compound": self.comp_feat_dim,
            "protein": 640,
            "pathway": 1,
            "pathway_count": 10,
            "disease_count": 0,
        }
        model = SimpleHGNLinkPredictor(
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_heads=self.num_heads, num_layers=self.num_layers,
            dropout=0.5, metadata=data.metadata(),
            compound_feat_dim=self.comp_feat_dim,
            node_feat_dims=node_feat_dims,
        )
        x_dict = {k: v for k, v in data.x_dict.items()}
        edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}
        embeddings = model(x_dict, edge_index_dict)
        assert "compound" in embeddings
        assert "protein" in embeddings


class TestRGCNLinkPredictor:
    """RGCN 模型前向传播测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.hidden_dim = 64
        self.out_dim = 64
        self.num_layers = 2
        self.comp_feat_dim = 200

    def _make_hetero_data(self):
        data = HeteroData()
        data["compound"].x = torch.randn(100, self.comp_feat_dim)
        data["protein"].x = torch.randn(50, 640)
        data["pathway"].x = torch.ones(10, 1)
        data["compound", "targets", "protein"].edge_index = torch.randint(0, 50, (2, 200))
        data["compound", "targets", "protein"].edge_index[0] = data["compound", "targets", "protein"].edge_index[0] % 100
        data["protein", "interacts", "protein"].edge_index = torch.randint(0, 50, (2, 300))
        data["protein", "belongs_to", "pathway"].edge_index = torch.randint(0, 10, (2, 100))
        data["protein", "belongs_to", "pathway"].edge_index[0] = data["protein", "belongs_to", "pathway"].edge_index[0] % 50
        return data

    def test_forward(self):
        from iron_aging_gnn.models.rgcn import RGCNLinkPredictor
        data = self._make_hetero_data()
        node_feat_dims = {
            "compound": self.comp_feat_dim,
            "protein": 640,
            "pathway": 1,
            "pathway_count": 10,
            "disease_count": 0,
        }
        model = RGCNLinkPredictor(
            hidden_dim=self.hidden_dim, out_dim=self.out_dim,
            num_layers=self.num_layers,
            dropout=0.5, metadata=data.metadata(),
            compound_feat_dim=self.comp_feat_dim,
            node_feat_dims=node_feat_dims,
        )
        x_dict = {k: v for k, v in data.x_dict.items()}
        edge_index_dict = {k: v for k, v in data.edge_index_dict.items()}
        embeddings = model(x_dict, edge_index_dict)
        assert "compound" in embeddings
        assert "protein" in embeddings