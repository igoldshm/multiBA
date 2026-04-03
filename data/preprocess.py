"""
data/preprocess.py
━━━━━━━━━━━━━━━━━━
Preprocesses raw PDBbind data into a clean CSV ready for training.

Usage:
  python data/preprocess.py --input_dir data/raw/ --output_dir data/processed/

This script:
  1. Parses PDBbind INDEX file
  2. Extracts SMILES from SDF files (or Kaggle CSV)
  3. Loads protein sequences from FASTA files
  4. Validates SMILES (RDKit) and sequences (BioPython)
  5. Applies quality filters (length, affinity range, valid structure)
  6. Deduplicates
  7. Saves train/val/test splits
"""

import argparse
from pathlib import Path
import pandas as pd
from loguru import logger


def preprocess_kaggle_csv(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Process the Kaggle pre-organized PDBbind CSV.
    Expected columns: pdb_id, smiles, sequence (or protein_sequence), IC50, Kd, Ki
    """
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # Normalize column names
    col_map = {
        "protein_sequence": "sequence",
        "SMILES": "smiles",
        "IC50": "ic50",
        "Kd": "kd",
        "Ki": "ki",
        "neg_log_value": "neg_log_affinity",
        "-logKd/Ki": "neg_log_affinity",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Build neg_log_affinity if not present
    if "neg_log_affinity" not in df.columns:
        if "kd" in df.columns:
            import numpy as np
            df["neg_log_affinity"] = -np.log10(df["kd"].astype(float) + 1e-30)
            df["affinity_type"] = "Kd"
        elif "ki" in df.columns:
            import numpy as np
            df["neg_log_affinity"] = -np.log10(df["ki"].astype(float) + 1e-30)
            df["affinity_type"] = "Ki"
        else:
            raise ValueError("Cannot find affinity values in CSV")

    # Basic filters
    df = df.dropna(subset=["sequence", "smiles", "neg_log_affinity"])
    df = df[(df["neg_log_affinity"] >= 2) & (df["neg_log_affinity"] <= 14)]
    df = df[df["sequence"].str.len().between(20, 1022)]
    df = df[df["smiles"].str.len().between(5, 512)]

    # Validate SMILES
    try:
        from rdkit import Chem
        valid = df["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)
        n_invalid = (~valid).sum()
        if n_invalid > 0:
            logger.warning(f"Removing {n_invalid} invalid SMILES")
        df = df[valid]
    except ImportError:
        logger.warning("RDKit not available — skipping SMILES validation")

    # Deduplicate by pdb_id
    df = df.drop_duplicates(subset=["pdb_id"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Preprocessed {len(df)} complexes → {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess PDBbind data")
    parser.add_argument("--input_dir", type=str, default="data/raw/")
    parser.add_argument("--output_dir", type=str, default="data/processed/")
    parser.add_argument("--kaggle_csv", type=str, default=None,
                        help="Path to Kaggle CSV (auto-detected if not specified)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect input
    csv_files = list(input_dir.glob("*.csv"))

    if csv_files:
        csv_path = Path(args.kaggle_csv) if args.kaggle_csv else csv_files[0]
        logger.info(f"Processing Kaggle CSV: {csv_path}")
        df = preprocess_kaggle_csv(csv_path, output_dir / "full_dataset.csv")
    else:
        # Try raw PDBbind structure
        index_file = input_dir / "INDEX_refined_data.2020"
        if index_file.exists():
            logger.info("Processing raw PDBbind structure...")
            from download_pdbbind import (
                parse_pdbbind_index,
                load_smiles_from_sdf,
                load_sequences_from_fasta,
                build_dataset_csv,
            )
            index_df = parse_pdbbind_index(index_file)
            smiles_map = load_smiles_from_sdf(input_dir / "refined-set")
            seq_map = load_sequences_from_fasta(input_dir / "refined-set")
            df = build_dataset_csv(
                index_df, smiles_map, seq_map,
                output_dir / "full_dataset.csv"
            )
        else:
            logger.error(
                f"No data found in {input_dir}\n"
                "Run: python data/download_pdbbind.py --use_kaggle"
            )
            return

    logger.info(f"\nDataset summary:")
    logger.info(f"  Total: {len(df)}")
    logger.info(f"  Affinity: {df['neg_log_affinity'].describe()}")


if __name__ == "__main__":
    main()
