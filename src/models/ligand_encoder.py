"""
src/models/ligand_encoder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dual-path ligand encoder:
  Path A: ChemBERTa-2 (SMILES → Transformer)
  Path B: Graph Attention Network (SMILES → Molecular Graph → GAT)
  Path C: Ensemble (A + B concatenated and projected)

ChemBERTa-2 captures sequential chemical grammar from SMILES.
GAT captures 2D topology (atoms, bonds, rings) — complementary information.
Ensemble of both outperforms either alone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# Path A: ChemBERTa-2 Encoder
# ══════════════════════════════════════════════════════════════════════════════

class ChemBERTaEncoder(nn.Module):
    """
    ChemBERTa-2 based SMILES encoder.

    Model options:
      seyonec/ChemBERTa-zinc-base-v1   — 84M params, 77M ZINC SMILES pretrained
      seyonec/ChemBERTa-zinc-base-v2   — Same arch, improved pretraining
      seyonec/PubChem10M_SMILES_BPE_396_250   — 77M PubChem SMILES

    ChemBERTa-2 uses Byte-Pair Encoding (BPE) on SMILES tokens, learning
    chemical substructure-level representations.
    """

    def __init__(
        self,
        backbone_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        projection_dim: int = 512,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        try:
            from transformers import AutoModel, AutoConfig
        except ImportError:
            raise ImportError("Run: pip install transformers")

        self.backbone = AutoModel.from_pretrained(backbone_name)
        config = AutoConfig.from_pretrained(backbone_name)
        hidden_size = config.hidden_size  # 768 for ChemBERTa-2

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Projection: [CLS] representation → shared dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        logger.info(f"ChemBERTaEncoder: {backbone_name} → dim={projection_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_token_embeddings: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        input_ids : [B, L]
        attention_mask : [B, L]

        Returns
        -------
        dict:
          "embedding"  : [B, projection_dim]
          "token_emb"  : [B, L, projection_dim]  (optional, for cross-attn)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, hidden_size]

        # [CLS] pooling
        cls_repr = hidden_states[:, 0, :]           # [B, hidden_size]
        cls_repr = self.dropout(cls_repr)
        embedding = self.projection(cls_repr)        # [B, projection_dim]

        result = {"embedding": embedding}

        if return_token_embeddings:
            B, L, H = hidden_states.shape
            tok = self.projection(hidden_states.reshape(B * L, H)).reshape(B, L, self.projection_dim)
            result["token_emb"] = tok

        return result


# ══════════════════════════════════════════════════════════════════════════════
# Path B: Graph Attention Network (GAT) Encoder
# ══════════════════════════════════════════════════════════════════════════════

class GATEncoder(nn.Module):
    """
    Graph Attention Network (GATv2) for molecular graphs.

    Architecture:
      - 6 GATv2 convolutional layers with 8 heads each
      - Edge features incorporated at each layer
      - Global mean + sum pooling (captures both local/global topology)
      - Projection to shared embedding dim

    GATv2 (Brody et al., 2022) fixes the expressiveness limitations of
    original GAT by using dynamic attention (key depends on both source and target).

    Node features: 74-dim (atom type, degree, hybridization, aromaticity, etc.)
    Edge features: 12-dim (bond type, conjugation, ring, stereo)
    """

    def __init__(
        self,
        num_node_features: int = 74,
        num_edge_features: int = 12,
        hidden_channels: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        projection_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        try:
            from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
        except ImportError:
            raise ImportError("Run: pip install torch-geometric")

        self.node_embedding = nn.Sequential(
            nn.Linear(num_node_features, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
        )

        self.edge_embedding = nn.Sequential(
            nn.Linear(num_edge_features, hidden_channels // 4),
            nn.GELU(),
        )

        # Stack of GATv2 layers with residual connections
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_channels = hidden_channels
            # Each head produces hidden_channels // num_heads → concat → hidden_channels
            head_channels = hidden_channels // num_heads

            conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=head_channels,
                heads=num_heads,
                edge_dim=hidden_channels // 4,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))

        # Dual readout: global mean + global add
        # Concatenate both → 2 * hidden_channels
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        logger.info(
            f"GATEncoder: {num_layers}×GATv2(h={num_heads}, dim={hidden_channels})"
            f" → dim={projection_dim}"
        )

    def forward(self, batch) -> dict:
        """
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched molecular graph with:
              batch.x           : [N_total, 74]   node features
              batch.edge_index  : [2, E_total]     edge connectivity
              batch.edge_attr   : [E_total, 12]    edge features
              batch.batch       : [N_total]         batch assignment

        Returns
        -------
        dict:
          "embedding" : [B, projection_dim]
        """
        from torch_geometric.nn import global_mean_pool, global_add_pool

        x = self.node_embedding(batch.x)                    # [N, H]
        edge_attr = self.edge_embedding(batch.edge_attr)    # [E, H/4]

        for conv, norm, drop in zip(self.convs, self.norms, self.dropouts):
            x_new = conv(x, batch.edge_index, edge_attr)    # [N, H]
            x_new = norm(x_new)
            x_new = F.gelu(x_new)
            x_new = drop(x_new)
            x = x + x_new  # Residual connection

        # Global pooling
        x_mean = global_mean_pool(x, batch.batch)  # [B, H]
        x_add = global_add_pool(x, batch.batch)    # [B, H]
        x_global = torch.cat([x_mean, x_add], dim=-1)  # [B, 2H]

        embedding = self.projection(x_global)  # [B, projection_dim]

        return {"embedding": embedding}


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble Ligand Encoder (ChemBERTa + GAT)
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleLigandEncoder(nn.Module):
    """
    Combines ChemBERTa-2 (SMILES sequence) and GAT (molecular graph).

    The two representations are complementary:
      - ChemBERTa: sequential patterns in SMILES (functional groups as tokens)
      - GAT: topological structure (ring systems, branching, connectivity)

    Fusion: concatenate + linear projection with gating
    """

    def __init__(
        self,
        projection_dim: int = 512,
        chembert_config: Optional[dict] = None,
        gat_config: Optional[dict] = None,
    ):
        super().__init__()
        self.projection_dim = projection_dim

        chembert_config = chembert_config or {}
        gat_config = gat_config or {}

        self.chembert = ChemBERTaEncoder(
            projection_dim=projection_dim, **chembert_config
        )
        self.gat = GATEncoder(
            projection_dim=projection_dim, **gat_config
        )

        # Learnable gating: decides how much to weight each modality
        self.gate = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.Sigmoid(),
        )

        self.fusion_projection = nn.Sequential(
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU(),
            nn.LayerNorm(projection_dim),
        )

        logger.info(f"EnsembleLigandEncoder: ChemBERTa + GAT → dim={projection_dim}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mol_graph=None,
        return_token_embeddings: bool = False,
    ) -> dict:
        chembert_out = self.chembert(
            input_ids, attention_mask, return_token_embeddings=return_token_embeddings
        )
        chembert_emb = chembert_out["embedding"]  # [B, D]

        if mol_graph is not None:
            gat_out = self.gat(mol_graph)
            gat_emb = gat_out["embedding"]  # [B, D]
        else:
            # Fall back to zeros if graph not available
            gat_emb = torch.zeros_like(chembert_emb)

        # Gated fusion
        combined = torch.cat([chembert_emb, gat_emb], dim=-1)  # [B, 2D]
        gate = self.gate(combined)                               # [B, 2D]
        fused = gate * combined                                  # [B, 2D]
        embedding = self.fusion_projection(fused)               # [B, D]

        result = {"embedding": embedding}

        if return_token_embeddings and "token_emb" in chembert_out:
            result["token_emb"] = chembert_out["token_emb"]

        return result


def build_ligand_encoder(config: dict) -> nn.Module:
    """Factory function to build the appropriate ligand encoder."""
    mode = config.get("mode", "chembert")

    if mode == "chembert":
        cfg = config.get("chembert", {})
        return ChemBERTaEncoder(
            backbone_name=cfg.get("backbone", "seyonec/ChemBERTa-zinc-base-v1"),
            projection_dim=cfg.get("projection_dim", 512),
            freeze_backbone=cfg.get("freeze_backbone", False),
        )

    elif mode == "gat":
        cfg = config.get("gat", {})
        return GATEncoder(
            num_node_features=cfg.get("num_node_features", 74),
            num_edge_features=cfg.get("num_edge_features", 12),
            hidden_channels=cfg.get("hidden_channels", 256),
            num_heads=cfg.get("num_heads", 8),
            num_layers=cfg.get("num_layers", 6),
            projection_dim=cfg.get("projection_dim", 512),
        )

    elif mode == "ensemble":
        return EnsembleLigandEncoder(
            projection_dim=config.get("gat", {}).get("projection_dim", 512),
            chembert_config=config.get("chembert", {}),
            gat_config=config.get("gat", {}),
        )

    else:
        raise ValueError(f"Unknown ligand encoder mode: {mode}")
