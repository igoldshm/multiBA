"""
src/models/binding_model.py
━━━━━━━━━━━━━━━━━━━━━━━━━━
MultiBA: The complete Multimodal Binding Affinity model.

Integrates:
  1. ProteinEncoder  (ESM-2 + LoRA)
  2. LigandEncoder   (ChemBERTa-2 / GAT / Ensemble)
  3. CrossAttentionFusion
  4. MLP Regression Head with MC Dropout

This is the PyTorch Lightning module — handles training, validation,
logging, and optimization in a clean, reproducible way.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional, Dict, Tuple, List
from loguru import logger

from .protein_encoder import ProteinEncoder
from .ligand_encoder import build_ligand_encoder
from .fusion import build_fusion


class MLPHead(nn.Module):
    """
    Regression MLP head with MC Dropout for uncertainty quantification.

    During training: standard dropout (regularization)
    During inference with mc_dropout=True: dropout stays ON across T forward
    passes → sample distribution → get mean + std of predictions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = (512, 256, 128),
        dropout_rates: List[float] = (0.3, 0.2, 0.1),
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        activations = {"gelu": nn.GELU, "relu": nn.ReLU, "selu": nn.SELU}
        act_fn = activations.get(activation, nn.GELU)

        layers = []
        in_dim = input_dim
        for hidden_dim, drop_rate in zip(hidden_dims, dropout_rates):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
                act_fn(),
                nn.Dropout(p=drop_rate),
            ]
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def mc_predict(
        self, x: torch.Tensor, num_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout inference.

        Forces dropout to stay ON (train mode) for T passes.
        Returns mean and standard deviation of predictions.
        """
        self.train()  # Enable dropout
        preds = torch.stack([self.forward(x) for _ in range(num_samples)], dim=0)
        self.eval()
        mean = preds.mean(dim=0)       # [B, 1]
        std = preds.std(dim=0)         # [B, 1]
        return mean, std


class MultiBA(pl.LightningModule):
    """
    Multimodal Binding Affinity Predictor.

    PyTorch Lightning module for clean training, evaluation, and logging.

    Parameters
    ----------
    config : dict
        Full model configuration (see configs/base_config.yaml)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        model_cfg = config["model"]
        train_cfg = config["training"]

        # ── 1. Protein Tower ─────────────────────────────────────────────
        prot_cfg = model_cfg["protein_encoder"]
        self.protein_encoder = ProteinEncoder(
            backbone_name=prot_cfg["backbone"],
            projection_dim=prot_cfg["projection_dim"],
            freeze_backbone=prot_cfg["freeze_backbone"],
            lora_config=prot_cfg.get("lora"),
            pooling=prot_cfg.get("pooling", "mean"),
        )

        # ── 2. Ligand Tower ──────────────────────────────────────────────
        self.ligand_encoder = build_ligand_encoder(model_cfg["ligand_encoder"])

        # ── 3. Cross-Attention Fusion ────────────────────────────────────
        self.fusion = build_fusion(model_cfg["fusion"])

        # ── 4. MLP Regression Head ───────────────────────────────────────
        head_cfg = model_cfg["head"]
        self.head = MLPHead(
            input_dim=model_cfg["fusion"]["embed_dim"],
            hidden_dims=head_cfg["hidden_dims"],
            dropout_rates=head_cfg["dropout_rates"],
            activation=head_cfg.get("activation", "gelu"),
            use_layernorm=head_cfg.get("use_layernorm", True),
        )

        # ── Loss ─────────────────────────────────────────────────────────
        self.mse_loss = nn.MSELoss()
        self.ranking_loss_weight = train_cfg["loss"].get("ranking_loss_weight", 0.1)

        # ── Metrics storage ───────────────────────────────────────────────
        self._val_preds = []
        self._val_targets = []
        self._test_preds = []
        self._test_targets = []
        self._test_pdb_ids = []

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"MultiBA initialized:")
        logger.info(f"  Total params:     {total:,}")
        logger.info(f"  Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")

    def forward(
        self,
        protein_ids: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_ids: torch.Tensor,
        ligand_mask: torch.Tensor,
        mol_graph=None,
        return_attention: bool = False,
    ) -> dict:
        """
        Full forward pass.

        Returns
        -------
        dict:
          "prediction"    : [B, 1]  — predicted pKd/pKi
          "fusion_emb"    : [B, D]  — fused embedding (for SHAP)
          "lig2prot_attn" : attention maps (if return_attention=True)
        """
        # ── Protein Tower ─────────────────────────────────────────────────
        prot_out = self.protein_encoder(
            protein_ids,
            protein_mask,
            return_residue_embeddings=True,  # Needed for cross-attention
        )
        # residue_emb: [B, Lp, D] — per-residue projected embeddings
        protein_residue_emb = prot_out.get("residue_emb")
        protein_pooled = prot_out["embedding"]  # [B, D]

        # ── Ligand Tower ──────────────────────────────────────────────────
        lig_out = self.ligand_encoder(
            input_ids=ligand_ids,
            attention_mask=ligand_mask,
            mol_graph=mol_graph,
            return_token_embeddings=True,  # For cross-attention
        )
        ligand_token_emb = lig_out.get("token_emb")  # [B, Ll, D] or None
        ligand_pooled = lig_out["embedding"]          # [B, D]

        # ── Cross-Attention Fusion ────────────────────────────────────────
        if protein_residue_emb is not None and ligand_token_emb is not None:
            fusion_out = self.fusion(
                protein_emb=protein_residue_emb,
                ligand_emb=ligand_token_emb,
                protein_mask=protein_mask,
                ligand_mask=ligand_mask,
            )
        else:
            # Fallback: expand pooled to sequence dim=1
            prot_seq = protein_pooled.unsqueeze(1)  # [B, 1, D]
            lig_seq = ligand_pooled.unsqueeze(1)    # [B, 1, D]
            fusion_out = self.fusion(
                protein_emb=prot_seq,
                ligand_emb=lig_seq,
            )

        fused_emb = fusion_out["embedding"]  # [B, D]

        # ── MLP Head ─────────────────────────────────────────────────────
        prediction = self.head(fused_emb)  # [B, 1]

        result = {
            "prediction": prediction,
            "fusion_emb": fused_emb,
        }
        if return_attention:
            result["lig2prot_attn"] = fusion_out.get("lig2prot_attn")
            result["prot2lig_attn"] = fusion_out.get("prot2lig_attn")

        return result

    def predict_with_uncertainty(
        self,
        protein_ids: torch.Tensor,
        protein_mask: torch.Tensor,
        ligand_ids: torch.Tensor,
        ligand_mask: torch.Tensor,
        mol_graph=None,
        num_mc_samples: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout inference for uncertainty quantification.

        Returns
        -------
        mean : [B, 1] — mean prediction
        std  : [B, 1] — epistemic uncertainty (std across MC samples)
        """
        # Get the fused embedding first (encoder is deterministic in eval)
        self.eval()
        with torch.no_grad():
            prot_out = self.protein_encoder(
                protein_ids, protein_mask, return_residue_embeddings=True
            )
            lig_out = self.ligand_encoder(
                ligand_ids, ligand_mask, mol_graph=mol_graph, return_token_embeddings=True
            )

            protein_residue_emb = prot_out.get("residue_emb")
            ligand_token_emb = lig_out.get("token_emb")

            if protein_residue_emb is not None and ligand_token_emb is not None:
                fusion_out = self.fusion(
                    protein_emb=protein_residue_emb,
                    ligand_emb=ligand_token_emb,
                    protein_mask=protein_mask,
                    ligand_mask=ligand_mask,
                )
            else:
                fusion_out = self.fusion(
                    protein_emb=prot_out["embedding"].unsqueeze(1),
                    ligand_emb=lig_out["embedding"].unsqueeze(1),
                )

            fused_emb = fusion_out["embedding"]

        # MC Dropout over the head
        mean, std = self.head.mc_predict(fused_emb, num_samples=num_mc_samples)
        return mean, std

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _pairwise_ranking_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Pairwise ranking loss — penalizes wrong orderings.
        L = mean(max(0, -(pred_i - pred_j) * sign(target_i - target_j)))

        Encourages the model to get the relative ranking right,
        not just the absolute value.
        """
        B = preds.shape[0]
        pred_diff = preds.unsqueeze(1) - preds.unsqueeze(0)    # [B, B, 1]
        target_diff = targets.unsqueeze(1) - targets.unsqueeze(0)  # [B, B, 1]

        # Ranking consistent if sign matches
        sign = torch.sign(target_diff)
        loss = F.relu(1.0 - sign * pred_diff)  # Margin = 1
        return loss.mean()

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        mse = self.mse_loss(predictions, targets)
        rank = self._pairwise_ranking_loss(predictions, targets)
        total = mse + self.ranking_loss_weight * rank

        self.log(f"{stage}/mse_loss", mse, prog_bar=False)
        self.log(f"{stage}/rank_loss", rank, prog_bar=False)
        self.log(f"{stage}/total_loss", total, prog_bar=True)
        return total

    # ── Training Step ─────────────────────────────────────────────────────────

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self(
            batch["protein_ids"],
            batch["protein_mask"],
            batch["ligand_ids"],
            batch["ligand_mask"],
            batch.get("mol_graph"),
        )
        loss = self._compute_loss(out["prediction"], batch["affinity"], "train")
        return loss

    # ── Validation Step ───────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int):
        with torch.no_grad():
            out = self(
                batch["protein_ids"],
                batch["protein_mask"],
                batch["ligand_ids"],
                batch["ligand_mask"],
                batch.get("mol_graph"),
            )
        self._compute_loss(out["prediction"], batch["affinity"], "val")
        self._val_preds.append(out["prediction"].cpu())
        self._val_targets.append(batch["affinity"].cpu())

    def on_validation_epoch_end(self):
        preds = torch.cat(self._val_preds).squeeze()
        targets = torch.cat(self._val_targets).squeeze()
        pearson = self._pearson_r(preds, targets)
        rmse = torch.sqrt(F.mse_loss(preds, targets))

        self.log("val/pearson_r", pearson, prog_bar=True)
        self.log("val/rmse", rmse, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    # ── Test Step ─────────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int):
        with torch.no_grad():
            out = self(
                batch["protein_ids"],
                batch["protein_mask"],
                batch["ligand_ids"],
                batch["ligand_mask"],
                batch.get("mol_graph"),
            )
        self._test_preds.append(out["prediction"].cpu())
        self._test_targets.append(batch["affinity"].cpu())
        self._test_pdb_ids.extend(batch.get("pdb_id", []))

    def on_test_epoch_end(self):
        preds = torch.cat(self._test_preds).squeeze()
        targets = torch.cat(self._test_targets).squeeze()

        pearson = self._pearson_r(preds, targets)
        rmse = torch.sqrt(F.mse_loss(preds, targets))
        spearman = self._spearman_r(preds, targets)
        ci = self._concordance_index(preds, targets)

        self.log("test/pearson_r", pearson)
        self.log("test/spearman_r", spearman)
        self.log("test/rmse", rmse)
        self.log("test/ci", ci)

        logger.info(
            f"\n{'═'*50}\n"
            f"  TEST RESULTS (PDBbind Core Set / CASF-2016)\n"
            f"{'═'*50}\n"
            f"  Pearson R:  {pearson:.4f}\n"
            f"  Spearman R: {spearman:.4f}\n"
            f"  RMSE:       {rmse:.4f}\n"
            f"  CI:         {ci:.4f}\n"
            f"{'═'*50}"
        )

        self._test_preds.clear()
        self._test_targets.clear()
        self._test_pdb_ids.clear()

    # ── Optimizer ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        """
        Differential learning rates:
          - Protein backbone (ESM-2/LoRA): very low LR (1e-5)
          - Ligand backbone (ChemBERTa):   low LR (3e-5)
          - Fusion + head:                 normal LR (3e-4)
        """
        train_cfg = self.config["training"]
        lr_groups = train_cfg.get("lr_groups", {})
        wd = train_cfg["optimizer"].get("weight_decay", 0.01)
        base_lr = train_cfg["optimizer"]["lr"]

        # Group parameters
        protein_params = list(self.protein_encoder.parameters())
        ligand_params = list(self.ligand_encoder.parameters())
        fusion_head_params = list(self.fusion.parameters()) + list(self.head.parameters())

        param_groups = [
            {
                "params": protein_params,
                "lr": lr_groups.get("protein_backbone", 1e-5),
                "name": "protein",
            },
            {
                "params": ligand_params,
                "lr": lr_groups.get("ligand_backbone", 3e-5),
                "name": "ligand",
            },
            {
                "params": fusion_head_params,
                "lr": lr_groups.get("fusion_head", base_lr),
                "name": "fusion_head",
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=wd,
            betas=train_cfg["optimizer"].get("betas", [0.9, 0.999]),
            eps=float(train_cfg["optimizer"].get("eps", 1e-8)),
        )

        # Cosine schedule with warmup
        sched_cfg = train_cfg.get("scheduler", {})
        warmup_steps = sched_cfg.get("warmup_steps", 500)
        total_steps = self.trainer.estimated_stepping_batches if self.trainer else 10000

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ── Metric utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _pearson_r(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds - preds.mean()
        targets = targets - targets.mean()
        r = (preds * targets).sum() / (
            (preds**2).sum().sqrt() * (targets**2).sum().sqrt() + 1e-9
        )
        return r

    @staticmethod
    def _spearman_r(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        def rank(x):
            tmp = x.argsort()
            ranks = torch.zeros_like(tmp, dtype=torch.float)
            ranks[tmp] = torch.arange(len(x), dtype=torch.float)
            return ranks
        return MultiBA._pearson_r(rank(preds), rank(targets))

    @staticmethod
    def _concordance_index(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Harrell's Concordance Index (CI) — fraction of correctly ordered pairs."""
        pred_diff = preds.unsqueeze(0) - preds.unsqueeze(1)
        targ_diff = targets.unsqueeze(0) - targets.unsqueeze(1)
        concordant = ((pred_diff > 0) & (targ_diff > 0)).float()
        discordant = ((pred_diff < 0) & (targ_diff > 0)).float()
        valid = (targ_diff > 0).float()
        return concordant.sum() / (concordant.sum() + discordant.sum() + 1e-9)
