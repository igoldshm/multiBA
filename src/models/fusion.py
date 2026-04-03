"""
src/models/fusion.py
━━━━━━━━━━━━━━━━━━━━
Cross-Attention Fusion Module.

WHY CROSS-ATTENTION OVER CONCATENATION?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Simple concatenation: embedding = f(concat[protein_emb, ligand_emb])
  → Treats protein and ligand independently, then fuses via MLP
  → Cannot model specific residue-atom interactions

Cross-attention: ligand "queries" specific protein residues
  Q = ligand representation
  K = protein residue representations
  V = protein residue representations

  Attention(Q, K, V) = softmax(QK^T / √d) · V

  Each ligand token attends to ALL protein residues simultaneously.
  The model learns which residues matter for this specific ligand.
  This mirrors the biophysical reality: specific amino acids in the
  binding pocket form hydrogen bonds, hydrophobic contacts, and
  electrostatic interactions with specific ligand atoms.

Architecture:
  1. Bidirectional cross-attention (ligand→protein AND protein→ligand)
  2. Add & Norm (residual connections + layer norm)
  3. FFN sublayer
  4. Global pooling → fixed-size vector
  5. Optional second cross-attention layer (deeper interaction modeling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from typing import Optional, Tuple
from loguru import logger


class CrossAttentionLayer(nn.Module):
    """
    Single cross-attention layer with residual connection.

    Allows sequence A to attend to sequence B.
    Output has the same shape as input A.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,       # [B, Lq, D]  — the "asker"
        key_value: torch.Tensor,   # [B, Lkv, D] — the "source of info"
        query_mask: Optional[torch.Tensor] = None,    # [B, Lq]
        kv_mask: Optional[torch.Tensor] = None,       # [B, Lkv]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        output : [B, Lq, D]  — query updated with cross-attended info
        attn_weights : [B, num_heads, Lq, Lkv]  — for interpretability
        """
        # key_padding_mask: True where padding (ignored)
        # nn.MultiheadAttention uses True = ignore convention
        kv_key_mask = (~kv_mask.bool()) if kv_mask is not None else None
        q_key_mask = (~query_mask.bool()) if query_mask is not None else None

        # Cross-attention: query attends to key_value
        attended, attn_weights = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=kv_key_mask,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )

        # Add & Norm
        query = self.norm1(query + self.dropout(attended))

        # FFN sublayer
        query = self.norm2(query + self.ffn(query))

        return query, attn_weights


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion of protein and ligand embeddings.

    The full fusion procedure:
      1. Project protein residue embeddings → [B, Lp, D]
      2. Project ligand token embeddings   → [B, Ll, D]
      3. Ligand → Protein cross-attention  (ligand queries protein)
         Output: ligand_updated [B, Ll, D]
      4. Protein → Ligand cross-attention  (protein queries ligand)
         Output: protein_updated [B, Lp, D]
      5. Mean-pool each → [B, D]
      6. Concatenate → [B, 2D]
      7. Final projection → [B, D]

    Bidirectionality captures:
      - Which protein residues this ligand binds to (step 3)
      - Which part of the ligand this protein recognizes (step 4)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Stacked bidirectional cross-attention
        self.lig2prot_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.prot2lig_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final aggregation: concat pooled representations → projection
        self.final_projection = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        logger.info(
            f"CrossAttentionFusion: {num_layers} layers, "
            f"{num_heads} heads, dim={embed_dim}"
        )

    def _masked_mean(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean pooling with mask."""
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).float()  # [B, L, 1]
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

    def forward(
        self,
        protein_emb: torch.Tensor,         # [B, Lp, D]  residue-level embeddings
        ligand_emb: torch.Tensor,          # [B, Ll, D]  token-level embeddings
        protein_mask: Optional[torch.Tensor] = None,  # [B, Lp]
        ligand_mask: Optional[torch.Tensor] = None,   # [B, Ll]
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          "embedding"        : [B, embed_dim]      — fused representation
          "lig2prot_attn"    : list of [B, H, Ll, Lp] — attention maps (last layer)
          "prot2lig_attn"    : list of [B, H, Lp, Ll]
        """
        lig = ligand_emb
        prot = protein_emb
        lig2prot_attns = []
        prot2lig_attns = []

        for lig2prot, prot2lig in zip(self.lig2prot_layers, self.prot2lig_layers):
            # Ligand queries protein
            lig_updated, l2p_attn = lig2prot(
                query=lig,
                key_value=prot,
                query_mask=ligand_mask,
                kv_mask=protein_mask,
            )
            # Protein queries ligand
            prot_updated, p2l_attn = prot2lig(
                query=prot,
                key_value=lig,
                query_mask=protein_mask,
                kv_mask=ligand_mask,
            )

            lig = lig_updated
            prot = prot_updated
            lig2prot_attns.append(l2p_attn)
            prot2lig_attns.append(p2l_attn)

        # Pool to fixed-size vectors
        lig_pooled = self._masked_mean(lig, ligand_mask)    # [B, D]
        prot_pooled = self._masked_mean(prot, protein_mask) # [B, D]

        # Fuse
        fused = torch.cat([lig_pooled, prot_pooled], dim=-1)  # [B, 2D]
        embedding = self.final_projection(fused)               # [B, D]

        return {
            "embedding": embedding,
            "lig2prot_attn": lig2prot_attns[-1] if lig2prot_attns else None,
            "prot2lig_attn": prot2lig_attns[-1] if prot2lig_attns else None,
        }


class ConcatFusion(nn.Module):
    """
    Simple concatenation baseline.
    Useful for ablation studies to measure the benefit of cross-attention.
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        protein_emb: torch.Tensor,
        ligand_emb: torch.Tensor,
        protein_mask=None,
        ligand_mask=None,
    ) -> dict:
        # If sequence-level inputs, pool first
        if protein_emb.dim() == 3:
            if protein_mask is not None:
                m = protein_mask.unsqueeze(-1).float()
                protein_emb = (protein_emb * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                protein_emb = protein_emb.mean(1)

        if ligand_emb.dim() == 3:
            if ligand_mask is not None:
                m = ligand_mask.unsqueeze(-1).float()
                ligand_emb = (ligand_emb * m).sum(1) / m.sum(1).clamp(min=1e-9)
            else:
                ligand_emb = ligand_emb.mean(1)

        fused = torch.cat([protein_emb, ligand_emb], dim=-1)
        return {"embedding": self.projection(fused)}


def build_fusion(config: dict) -> nn.Module:
    """Factory function to build the fusion module."""
    fusion_type = config.get("type", "cross_attention")
    embed_dim = config.get("embed_dim", 512)
    num_heads = config.get("num_heads", 8)
    dropout = config.get("dropout", 0.1)
    num_layers = config.get("num_layers", 2)

    if fusion_type == "cross_attention":
        return CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
    elif fusion_type == "concat":
        return ConcatFusion(embed_dim=embed_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
