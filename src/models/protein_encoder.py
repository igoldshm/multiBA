"""
src/models/protein_encoder.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESM-2 Protein Language Model encoder with:
  - LoRA fine-tuning (updates ~0.5% of parameters)
  - Multiple pooling strategies
  - Learnable projection to shared embedding dimension
  - Residue-level attention weights for interpretability

ESM-2 model sizes:
  facebook/esm2_t6_8M_UR50D       — 8M params   (dev/testing)
  facebook/esm2_t12_35M_UR50D     — 35M params  (lightweight)
  facebook/esm2_t30_150M_UR50D    — 150M params (balanced)
  facebook/esm2_t33_650M_UR50D    — 650M params (recommended ✓)
  facebook/esm2_t36_3B_UR50D      — 3B params   (max accuracy, needs A100)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from loguru import logger
from typing import Optional, Tuple


class ProteinEncoder(nn.Module):
    """
    ESM-2 based protein sequence encoder.

    Parameters
    ----------
    backbone_name : str
        HuggingFace model identifier for ESM-2
    projection_dim : int
        Output embedding dimension (shared with ligand encoder)
    freeze_backbone : bool
        If True, freeze all ESM-2 weights (only train LoRA + projection)
    lora_config : dict | None
        LoRA configuration. If None, no LoRA is applied.
    pooling : str
        How to aggregate per-residue representations:
        "mean" — mean over non-padding tokens (recommended)
        "cls"  — use [CLS] token only
        "max"  — max pooling
        "attention" — learned attention pooling
    """

    def __init__(
        self,
        backbone_name: str = "facebook/esm2_t33_650M_UR50D",
        projection_dim: int = 512,
        freeze_backbone: bool = True,
        lora_config: Optional[dict] = None,
        pooling: str = "mean",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.projection_dim = projection_dim
        self.pooling = pooling

        # ── Load ESM-2 ────────────────────────────────────────────────────
        self._load_backbone(backbone_name, freeze_backbone, lora_config)

        # ── Projection head ────────────────────────────────────────────────
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # ── Attention pooling (optional) ──────────────────────────────────
        if pooling == "attention":
            self.attn_pool = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )

        logger.info(f"ProteinEncoder: {backbone_name}")
        logger.info(f"  Pooling:     {pooling}")
        logger.info(f"  Output dim:  {projection_dim}")
        logger.info(f"  Trainable params: {self._count_trainable_params():,}")

    def _load_backbone(
        self,
        backbone_name: str,
        freeze_backbone: bool,
        lora_config: Optional[dict],
    ):
        """Load ESM-2 and optionally apply LoRA."""
        try:
            from transformers import EsmModel
        except ImportError:
            raise ImportError("Run: pip install transformers")

        self.backbone = EsmModel.from_pretrained(backbone_name, add_pooling_layer=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("ESM-2 backbone frozen")

        if lora_config and lora_config.get("enabled", False):
            self._apply_lora(lora_config)

    def _apply_lora(self, config: dict):
        """Inject LoRA adapters into ESM-2 attention layers."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Run: pip install peft")

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.get("r", 16),
            lora_alpha=config.get("alpha", 32),
            lora_dropout=config.get("dropout", 0.1),
            target_modules=config.get("target_modules", ["query", "key", "value"]),
            bias="none",
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        trainable, total = self.backbone.get_nb_trainable_parameters()
        logger.info(f"LoRA applied: {trainable:,} / {total:,} trainable ({100*trainable/total:.2f}%)")

    def _count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Pool per-residue representations to a fixed-size vector.

        Returns
        -------
        pooled : [B, hidden_size]
        attn_weights : [B, L] or None  (for interpretability)
        """
        attn_weights = None

        if self.pooling == "cls":
            pooled = hidden_states[:, 0, :]  # [CLS] token

        elif self.pooling == "mean":
            # Masked mean pooling (ignores padding tokens)
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            hidden_states_masked = hidden_states.masked_fill(~mask, -1e9)
            pooled = hidden_states_masked.max(dim=1).values

        elif self.pooling == "attention":
            # Learned attention pooling
            scores = self.attn_pool(hidden_states).squeeze(-1)  # [B, L]
            scores = scores.masked_fill(~attention_mask.bool(), -1e9)
            attn_weights = F.softmax(scores, dim=-1)             # [B, L]
            pooled = (attn_weights.unsqueeze(-1) * hidden_states).sum(dim=1)  # [B, H]

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled, attn_weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_residue_embeddings: bool = False,
    ) -> dict:
        """
        Forward pass through ESM-2 + projection.

        Parameters
        ----------
        input_ids : [B, L]
        attention_mask : [B, L]
        return_residue_embeddings : bool
            If True, also returns per-residue embeddings (needed for cross-attention)

        Returns
        -------
        dict with keys:
          "embedding"  : [B, projection_dim]  — pooled + projected
          "residue_emb": [B, L, projection_dim] — per-residue (optional)
          "attn_weights": [B, L] — pooling attention weights (optional)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state  # [B, L, hidden_size]

        # Pool to [B, hidden_size]
        pooled, attn_weights = self._pool(hidden_states, attention_mask)

        # Project to shared dim
        embedding = self.projection(pooled)  # [B, projection_dim]

        result = {"embedding": embedding}

        if attn_weights is not None:
            result["attn_weights"] = attn_weights

        if return_residue_embeddings:
            # Project each residue individually for cross-attention
            B, L, H = hidden_states.shape
            residue_proj = self.projection(hidden_states.reshape(B * L, H))
            result["residue_emb"] = residue_proj.reshape(B, L, self.projection_dim)

        return result
