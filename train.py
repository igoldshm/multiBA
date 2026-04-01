"""
train.py
━━━━━━━━
Training entrypoint for MultiBA.

Usage:
  python train.py                                    # Use defaults
  python train.py training.batch_size=64            # Override a param
  python train.py model.ligand_encoder.mode=gat     # Use GAT ligand encoder
  python train.py +experiment=ablation_no_lora      # Load override config

Hydra manages config composition and experiment logging.
"""

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.binding_model import MultiBA
from src.data.dataset import create_dataloaders
from src.data.splits import refined_core_split, scaffold_split, random_split


def load_tokenizers(config: DictConfig):
    """Load ESM-2 and ChemBERTa-2 tokenizers."""
    from transformers import AutoTokenizer, EsmTokenizer

    prot_name = config.model.protein_encoder.backbone
    lig_name = config.model.ligand_encoder.chembert.backbone

    logger.info(f"Loading protein tokenizer: {prot_name}")
    protein_tokenizer = EsmTokenizer.from_pretrained(prot_name)

    logger.info(f"Loading ligand tokenizer: {lig_name}")
    ligand_tokenizer = AutoTokenizer.from_pretrained(lig_name)

    return protein_tokenizer, ligand_tokenizer


def build_callbacks(config: DictConfig) -> list:
    """Build PyTorch Lightning callbacks."""
    callbacks = []

    # Model checkpoint — save top-k by validation Pearson R
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=config.paths.checkpoints,
            filename="multiba-epoch{epoch:02d}-r{val/pearson_r:.4f}",
            monitor="val/pearson_r",
            mode="max",
            save_top_k=config.logging.save_top_k,
            save_last=True,
            verbose=True,
        )
    )

    # Early stopping
    es_cfg = config.training.early_stopping
    callbacks.append(
        pl.callbacks.EarlyStopping(
            monitor=es_cfg.monitor,
            mode=es_cfg.mode,
            patience=es_cfg.patience,
            min_delta=es_cfg.min_delta,
            verbose=True,
        )
    )

    # Learning rate monitor
    callbacks.append(
        pl.callbacks.LearningRateMonitor(logging_interval="step")
    )

    # Rich progress bar
    callbacks.append(pl.callbacks.RichProgressBar())

    return callbacks


def build_loggers(config: DictConfig) -> list:
    """Build experiment loggers."""
    loggers = []

    if config.logging.mlflow.enabled:
        from pytorch_lightning.loggers import MLFlowLogger
        loggers.append(
            MLFlowLogger(
                experiment_name=config.logging.mlflow.experiment_name,
                tracking_uri=config.logging.mlflow.tracking_uri,
                tags={
                    "model_version": config.project.version,
                    "ligand_encoder": config.model.ligand_encoder.mode,
                    "fusion": config.model.fusion.type,
                },
            )
        )

    if config.logging.wandb.enabled:
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(
            WandbLogger(
                project=config.logging.wandb.project,
                name=config.project.name,
                config=OmegaConf.to_container(config, resolve=True),
            )
        )

    if not loggers:
        # Fallback to CSV logger
        loggers.append(
            pl.loggers.CSVLogger(
                save_dir=config.paths.logs,
                name=config.project.name,
            )
        )

    return loggers


@hydra.main(config_path="configs", config_name="base_config", version_base=None)
def train(config: DictConfig):
    """
    Main training function.

    Hydra loads configs/base_config.yaml by default.
    Any config key can be overridden from the CLI.
    """
    # ── Reproducibility ───────────────────────────────────────────────────
    pl.seed_everything(config.project.seed, workers=True)
    torch.set_float32_matmul_precision("medium")  # Faster on Ampere+ GPUs

    # ── Log config ────────────────────────────────────────────────────────
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("Loading dataset...")

    # Try to load the full dataset; fall back to sample for testing
    data_path = Path(config.data.processed_dir) / "full_dataset.csv"
    if not data_path.exists():
        logger.warning(f"Dataset not found at {data_path}")
        logger.warning("Run: python data/download_pdbbind.py --use_kaggle")
        logger.warning("Falling back to sample dataset for demonstration...")
        data_path = Path("data/raw/sample_dataset.csv")
        if not data_path.exists():
            logger.info("Creating sample dataset...")
            from data.download_pdbbind import create_sample_dataset
            create_sample_dataset(Path("data/raw/"))

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} protein-ligand complexes")

    # ── Split data ────────────────────────────────────────────────────────
    split_strategy = config.data.split_strategy
    logger.info(f"Splitting data using strategy: {split_strategy}")

    if split_strategy == "refined_core":
        train_df, val_df, test_df = refined_core_split(
            df, val_fraction=config.data.val_fraction, seed=config.project.seed
        )
    elif split_strategy == "scaffold":
        train_df, val_df, test_df = scaffold_split(
            df, seed=config.project.seed
        )
    else:
        train_df, val_df, test_df = random_split(
            df, val_fraction=config.data.val_fraction, seed=config.project.seed
        )

    # Save splits for reproducibility
    splits_dir = Path(config.data.processed_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)
    logger.info(f"Splits saved to {splits_dir}")

    # ── Tokenizers ────────────────────────────────────────────────────────
    protein_tokenizer, ligand_tokenizer = load_tokenizers(config)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        protein_tokenizer=protein_tokenizer,
        ligand_tokenizer=ligand_tokenizer,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        cache_dir=config.paths.cache,
        include_graph=config.model.ligand_encoder.mode in ("gat", "ensemble"),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Initializing MultiBA model...")
    config_dict = OmegaConf.to_container(config, resolve=True)
    model = MultiBA(config_dict)

    # ── Callbacks & Loggers ───────────────────────────────────────────────
    callbacks = build_callbacks(config)
    loggers = build_loggers(config)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=False,  # True is slower; fine for production
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # ── Test on Core Set (unseen) ─────────────────────────────────────────
    logger.info("\nEvaluating on test set (PDBbind Core Set)...")
    trainer.test(model, test_loader, ckpt_path="best")

    # ── Save final checkpoint ─────────────────────────────────────────────
    final_ckpt = Path(config.paths.checkpoints) / "final_model.ckpt"
    trainer.save_checkpoint(final_ckpt)
    logger.success(f"Training complete! Best model saved to {final_ckpt}")

    return model


if __name__ == "__main__":
    train()
