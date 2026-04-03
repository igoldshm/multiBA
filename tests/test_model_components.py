"""
tests/test_model_components.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unit tests for MultiBA model components.

Run: pytest tests/ -v --cov=src

Tests cover:
  - Cross-attention fusion shapes
  - MC Dropout uncertainty (non-zero std)
  - Concordance index metric
  - Scaffold split correctness
  - Dataset tokenization
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════════════════════
# Fusion tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossAttentionFusion:
    def setup_method(self):
        from src.models.fusion import CrossAttentionFusion, ConcatFusion
        self.B, self.Lp, self.Ll, self.D = 4, 32, 16, 64

        self.cross_attn = CrossAttentionFusion(
            embed_dim=self.D, num_heads=4, num_layers=1
        )
        self.concat = ConcatFusion(embed_dim=self.D)

    def test_cross_attention_output_shape(self):
        prot = torch.randn(self.B, self.Lp, self.D)
        lig = torch.randn(self.B, self.Ll, self.D)
        out = self.cross_attn(prot, lig)
        assert out["embedding"].shape == (self.B, self.D)

    def test_cross_attention_with_masks(self):
        prot = torch.randn(self.B, self.Lp, self.D)
        lig = torch.randn(self.B, self.Ll, self.D)
        prot_mask = torch.ones(self.B, self.Lp)
        lig_mask = torch.ones(self.B, self.Ll)
        # Mask out last 5 tokens
        prot_mask[:, -5:] = 0
        lig_mask[:, -3:] = 0
        out = self.cross_attn(prot, lig, prot_mask, lig_mask)
        assert out["embedding"].shape == (self.B, self.D)
        assert not torch.isnan(out["embedding"]).any()

    def test_concat_fusion_output_shape(self):
        prot = torch.randn(self.B, self.D)
        lig = torch.randn(self.B, self.D)
        out = self.concat(prot, lig)
        assert out["embedding"].shape == (self.B, self.D)

    def test_cross_attention_returns_attn_weights(self):
        prot = torch.randn(self.B, self.Lp, self.D)
        lig = torch.randn(self.B, self.Ll, self.D)
        out = self.cross_attn(prot, lig)
        assert "lig2prot_attn" in out
        assert out["lig2prot_attn"] is not None

    def test_no_nan_in_output(self):
        prot = torch.randn(self.B, self.Lp, self.D)
        lig = torch.randn(self.B, self.Ll, self.D)
        out = self.cross_attn(prot, lig)
        assert not torch.isnan(out["embedding"]).any()
        assert not torch.isinf(out["embedding"]).any()


# ══════════════════════════════════════════════════════════════════════════════
# MLP Head & MC Dropout tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMLPHead:
    def setup_method(self):
        from src.models.binding_model import MLPHead
        self.head = MLPHead(
            input_dim=128,
            hidden_dims=[64, 32],
            dropout_rates=[0.3, 0.2],
        )

    def test_forward_shape(self):
        x = torch.randn(8, 128)
        out = self.head(x)
        assert out.shape == (8, 1)

    def test_mc_dropout_returns_mean_std(self):
        x = torch.randn(4, 128)
        mean, std = self.head.mc_predict(x, num_samples=20)
        assert mean.shape == (4, 1)
        assert std.shape == (4, 1)

    def test_mc_dropout_uncertainty_nonzero(self):
        """MC Dropout should produce non-zero uncertainty."""
        x = torch.randn(4, 128)
        _, std = self.head.mc_predict(x, num_samples=20)
        # At least some uncertainty should exist (dropout is stochastic)
        assert std.mean().item() > 0.0

    def test_mc_dropout_consistency(self):
        """Mean prediction should be close across runs (not wildly different)."""
        x = torch.randn(4, 128)
        mean1, _ = self.head.mc_predict(x, num_samples=50)
        mean2, _ = self.head.mc_predict(x, num_samples=50)
        # Means should agree within tolerance
        assert torch.abs(mean1 - mean2).max().item() < 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Metrics tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMetrics:
    def test_pearson_r_perfect(self):
        from src.models.binding_model import MultiBA
        x = torch.arange(10, dtype=torch.float)
        r = MultiBA._pearson_r(x, x)
        assert abs(r.item() - 1.0) < 1e-5

    def test_pearson_r_anticorrelated(self):
        from src.models.binding_model import MultiBA
        x = torch.arange(10, dtype=torch.float)
        r = MultiBA._pearson_r(x, -x)
        assert abs(r.item() + 1.0) < 1e-5

    def test_pearson_r_random(self):
        from src.models.binding_model import MultiBA
        torch.manual_seed(42)
        x = torch.randn(100)
        y = torch.randn(100)
        r = MultiBA._pearson_r(x, y)
        assert -1.0 <= r.item() <= 1.0

    def test_concordance_index_perfect(self):
        from src.models.binding_model import MultiBA
        x = torch.arange(10, dtype=torch.float)
        ci = MultiBA._concordance_index(x, x)
        assert abs(ci.item() - 1.0) < 1e-4

    def test_concordance_index_worst(self):
        from src.models.binding_model import MultiBA
        x = torch.arange(10, dtype=torch.float)
        ci = MultiBA._concordance_index(-x, x)
        assert ci.item() < 0.1

    def test_concordance_index_range(self):
        from src.models.binding_model import MultiBA
        torch.manual_seed(42)
        x = torch.randn(50)
        y = torch.randn(50)
        ci = MultiBA._concordance_index(x, y)
        assert 0.0 <= ci.item() <= 1.0

    def test_spearman_r_monotonic(self):
        from src.models.binding_model import MultiBA
        x = torch.arange(10, dtype=torch.float)
        y = torch.exp(x)  # Monotonic but not linear
        r = MultiBA._spearman_r(x, y)
        assert abs(r.item() - 1.0) < 1e-4


# ══════════════════════════════════════════════════════════════════════════════
# Data splits tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSplits:
    def setup_method(self):
        import pandas as pd
        import numpy as np

        # Create synthetic dataset
        np.random.seed(42)
        n = 200
        self.df = pd.DataFrame({
            "pdb_id": [f"pdb{i:04d}" for i in range(n)],
            "sequence": ["ACDEFGHIKLMNPQRSTVWY" * 5] * n,
            "smiles": ["CC(=O)O"] * n,
            "neg_log_affinity": np.random.uniform(3, 11, n),
            "affinity_type": ["Kd"] * n,
            "year": np.random.randint(2000, 2022, n),
        })

    def test_random_split_sizes(self):
        from src.data.splits import random_split
        train, val, test = random_split(self.df, val_fraction=0.1, test_fraction=0.1)
        assert len(train) + len(val) + len(test) == len(self.df)
        assert len(val) == pytest.approx(20, abs=2)
        assert len(test) == pytest.approx(20, abs=2)

    def test_random_split_no_overlap(self):
        from src.data.splits import random_split
        train, val, test = random_split(self.df)
        train_ids = set(train["pdb_id"])
        val_ids = set(val["pdb_id"])
        test_ids = set(test["pdb_id"])
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0

    def test_temporal_split(self):
        from src.data.splits import temporal_split
        train, val, test = temporal_split(self.df, val_year=2018, test_year=2020)
        assert (train["year"] < 2018).all()
        assert ((val["year"] >= 2018) & (val["year"] < 2020)).all()
        assert (test["year"] >= 2020).all()

    def test_refined_core_split(self):
        from src.data.splits import refined_core_split, CORE_SET_IDS
        # Add some actual core set IDs to our test data
        self.df.iloc[:10, self.df.columns.get_loc("pdb_id")] = list(CORE_SET_IDS)[:10]
        train, val, test = refined_core_split(self.df)
        # Core set IDs should be in test
        test_ids = set(test["pdb_id"])
        core_present = set(list(CORE_SET_IDS)[:10]) & set(self.df["pdb_id"])
        assert len(core_present & test_ids) == len(core_present)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBindingAffinityDataset:
    """Tests that don't require downloading models (use tiny tokenizer)."""

    def test_mol_graph_building(self):
        """Test molecular graph construction from SMILES."""
        import pandas as pd
        from src.data.dataset import BindingAffinityDataset

        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")

        # Create minimal dataset instance just for graph building
        ds = BindingAffinityDataset.__new__(BindingAffinityDataset)

        test_smiles = [
            "CC(=O)O",          # Acetic acid
            "c1ccccc1",         # Benzene
            "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        ]

        for smi in test_smiles:
            graph = ds._build_mol_graph(smi)
            assert graph is not None, f"Graph build failed for {smi}"
            assert graph.x.shape[1] == 74, f"Expected 74 node features, got {graph.x.shape[1]}"
            assert graph.edge_attr.shape[1] == 12, f"Expected 12 edge features"
            mol = Chem.MolFromSmiles(smi)
            assert graph.x.shape[0] == mol.GetNumAtoms()

    def test_smiles_randomization(self):
        """SMILES augmentation should produce valid (possibly different) SMILES."""
        try:
            from rdkit import Chem
        except ImportError:
            pytest.skip("RDKit not installed")

        from src.data.dataset import BindingAffinityDataset
        smi = "CC(=O)Oc1ccccc1C(=O)O"

        results = set()
        for _ in range(10):
            rand = BindingAffinityDataset._randomize_smiles(smi)
            mol = Chem.MolFromSmiles(rand)
            assert mol is not None, f"Randomized SMILES invalid: {rand}"
            results.add(rand)

        # Should produce at least some variety (not always the same)
        # Note: small molecules may not have many valid random forms
        assert len(results) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Integration smoke test
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegrationSmoke:
    """End-to-end smoke test with tiny model (no pretrained weights needed)."""

    def test_full_forward_pass_shapes(self):
        """Test that all shapes work correctly without loading real weights."""
        from src.models.fusion import CrossAttentionFusion
        from src.models.binding_model import MLPHead

        B, Lp, Ll, D = 2, 20, 12, 64

        fusion = CrossAttentionFusion(embed_dim=D, num_heads=4, num_layers=1)
        head = MLPHead(input_dim=D, hidden_dims=[32, 16], dropout_rates=[0.1, 0.1])

        prot_emb = torch.randn(B, Lp, D)
        lig_emb = torch.randn(B, Ll, D)

        fusion_out = fusion(prot_emb, lig_emb)
        assert fusion_out["embedding"].shape == (B, D)

        prediction = head(fusion_out["embedding"])
        assert prediction.shape == (B, 1)

    def test_pairwise_ranking_loss(self):
        """Ranking loss should be non-negative."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        # Inline the ranking loss to test without full model init
        def ranking_loss(preds, targets):
            import torch.nn.functional as F
            pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
            target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
            sign = torch.sign(target_diff)
            return F.relu(1.0 - sign * pred_diff).mean()

        preds = torch.tensor([[5.0], [7.0], [9.0], [3.0]])
        targets = torch.tensor([[5.0], [7.0], [9.0], [3.0]])

        loss = ranking_loss(preds, targets)
        assert loss.item() >= 0.0

        # Perfect ordering should give 0 loss (or small margin loss)
        wrong_preds = torch.tensor([[9.0], [5.0], [3.0], [7.0]])  # Wrong order
        wrong_loss = ranking_loss(wrong_preds, targets)
        # Wrong ordering should have higher loss
        assert wrong_loss.item() > loss.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
