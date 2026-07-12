"""解码器测试

测试四种解码器 (MLP/Dot/Bilinear/ResidueBilinear) 的输出形状和数值稳定性。
"""

import torch


class TestMLPDecoder:
    def test_output_shape(self):
        from iron_aging_gnn.models.decoders import MLPDecoder
        decoder = MLPDecoder(out_dim=64, hidden_dim=128)
        comp_emb = torch.randn(32, 64)
        prot_emb = torch.randn(32, 64)
        scores = decoder(comp_emb, prot_emb)
        assert scores.shape == (32,)
        assert not torch.isnan(scores).any()

    def test_gradient_flow(self):
        from iron_aging_gnn.models.decoders import MLPDecoder
        decoder = MLPDecoder(out_dim=64, hidden_dim=128)
        comp_emb = torch.randn(32, 64, requires_grad=True)
        prot_emb = torch.randn(32, 64, requires_grad=True)
        scores = decoder(comp_emb, prot_emb)
        loss = scores.sum()
        loss.backward()
        assert comp_emb.grad is not None
        assert prot_emb.grad is not None


class TestDotProductDecoder:
    def test_output_shape(self):
        from iron_aging_gnn.models.decoders import DotProductDecoder
        decoder = DotProductDecoder()
        comp_emb = torch.randn(32, 64)
        prot_emb = torch.randn(32, 64)
        scores = decoder(comp_emb, prot_emb)
        assert scores.shape == (32,)
        assert not torch.isnan(scores).any()


class TestBilinearDecoder:
    def test_output_shape(self):
        from iron_aging_gnn.models.decoders import BilinearDecoder
        decoder = BilinearDecoder(out_dim=64, rank=32)
        comp_emb = torch.randn(32, 64)
        prot_emb = torch.randn(32, 64)
        scores = decoder(comp_emb, prot_emb)
        assert scores.shape == (32,)
        assert not torch.isnan(scores).any()


class TestResidueAwareBilinearDecoder:
    def test_output_shape_no_residue(self):
        from iron_aging_gnn.models.decoders import ResidueAwareBilinearDecoder
        decoder = ResidueAwareBilinearDecoder(comp_dim=64, residue_dim=640, rank=64)
        comp_emb = torch.randn(16, 64)
        prot_emb = torch.randn(16, 64)
        scores = decoder(comp_emb, prot_emb)
        assert scores.shape == (16,)
        assert not torch.isnan(scores).any()

    def test_output_shape_with_residue(self):
        from iron_aging_gnn.models.decoders import ResidueAwareBilinearDecoder
        decoder = ResidueAwareBilinearDecoder(comp_dim=64, residue_dim=640, rank=64, max_len=20)
        n_prot = 5
        max_len = 20
        total_residues = n_prot * max_len
        decoder.register_residue_buffers(
            torch.randn(total_residues, 640),
            torch.tensor([0, 20, 40, 60, 80, 100], dtype=torch.long),
            torch.full((n_prot,), max_len, dtype=torch.long),
            max_len=max_len,
            residue_device="cpu",
        )
        comp_emb = torch.randn(4, 64)
        prot_emb = torch.randn(4, 64)
        prot_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        scores = decoder(comp_emb, prot_emb, prot_indices=prot_indices)
        assert scores.shape == (4,)
        assert not torch.isnan(scores).any()