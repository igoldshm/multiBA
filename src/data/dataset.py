"""
src/data/dataset.py
━━━━━━━━━━━━━━━━━━
PyTorch Dataset for PDBbind protein-ligand binding affinity data.

Key features:
  - Lazy loading of ESM-2 and ChemBERTa-2 features
  - Disk-based caching (avoids re-tokenizing every epoch)
  - Scaffold splitting for realistic generalization estimates
  - Support for both Kd and Ki entries
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from loguru import logger


class BindingAffinityDataset(Dataset):
    """
    Dataset for protein-ligand binding affinity prediction.

    Each item returns:
      - protein_ids:    LongTensor[L_p]   ESM-2 input token IDs
      - protein_mask:   BoolTensor[L_p]   Attention mask for protein
      - ligand_ids:     LongTensor[L_l]   ChemBERTa-2 token IDs
      - ligand_mask:    BoolTensor[L_l]   Attention mask for ligand
      - mol_graph:      torch_geometric.data.Data (optional GAT path)
      - affinity:       FloatTensor[1]    pKd/pKi label
      - pdb_id:         str               Complex identifier

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: pdb_id, sequence, smiles, neg_log_affinity
    protein_tokenizer : transformers.PreTrainedTokenizer
        ESM-2 tokenizer
    ligand_tokenizer : transformers.PreTrainedTokenizer
        ChemBERTa-2 tokenizer
    max_protein_len : int
        Maximum protein sequence length (default 1022)
    max_smiles_len : int
        Maximum SMILES length (default 512)
    cache_dir : Optional[Path]
        Directory to cache tokenized inputs
    include_graph : bool
        Whether to build molecular graphs for GAT path
    augment : bool
        Whether to apply data augmentation (SMILES enumeration)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        protein_tokenizer,
        ligand_tokenizer,
        max_protein_len: int = 1022,
        max_smiles_len: int = 512,
        cache_dir: Optional[Path] = None,
        include_graph: bool = True,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.protein_tokenizer = protein_tokenizer
        self.ligand_tokenizer = ligand_tokenizer
        self.max_protein_len = max_protein_len
        self.max_smiles_len = max_smiles_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.include_graph = include_graph
        self.augment = augment

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset initialized with {len(self.df)} complexes")
        logger.info(f"  Affinity range: [{df['neg_log_affinity'].min():.2f}, {df['neg_log_affinity'].max():.2f}]")
        logger.info(f"  Graph features: {'enabled' if include_graph else 'disabled'}")

    def __len__(self) -> int:
        return len(self.df)

    def _cache_key(self, idx: int) -> str:
        row = self.df.iloc[idx]
        content = f"{row['pdb_id']}|{row['sequence'][:50]}|{row['smiles'][:50]}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[dict]:
        if self.cache_dir is None:
            return None
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: dict) -> None:
        if self.cache_dir is None:
            return
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as fh:
                pickle.dump(data, fh)
        except Exception:
            pass  # Non-critical if cache fails

    def _tokenize_protein(self, sequence: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize protein sequence with ESM-2 tokenizer."""
        # ESM-2 adds special tokens: [CLS] seq [EOS] — truncate to fit
        encoding = self.protein_tokenizer(
            sequence,
            max_length=self.max_protein_len + 2,  # +2 for special tokens
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    def _tokenize_ligand(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize SMILES string with ChemBERTa-2 tokenizer."""
        # Optional augmentation: randomize SMILES representation
        if self.augment:
            smiles = self._randomize_smiles(smiles)

        encoding = self.ligand_tokenizer(
            smiles,
            max_length=self.max_smiles_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)

    def _build_mol_graph(self, smiles: str):
        """
        Convert SMILES to a PyTorch Geometric graph.

        Node features (74-dim):
          - Atom type (one-hot, 44 types)
          - Degree (one-hot, 11 values)
          - Formal charge, num H, num radical e-
          - Hybridization (one-hot, 5 types)
          - Aromaticity, ring membership

        Edge features (12-dim):
          - Bond type (one-hot: single, double, triple, aromatic)
          - Conjugated, ring, stereo (one-hot, 6 types)
        """
        try:
            from rdkit import Chem
            from torch_geometric.data import Data

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # ── Node (atom) features ─────────────────────────────────────
            x = []
            for atom in mol.GetAtoms():
                feat = []

                # Atom type (one-hot, top-44 + other)
                atom_types = [
                    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
                    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
                    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn",
                    "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
                    "Zr", "Cr", "Pt", "Hg", "Pb", "other"
                ]
                atom_sym = atom.GetSymbol()
                idx = atom_types.index(atom_sym) if atom_sym in atom_types else atom_types.index("other")
                feat += [int(i == idx) for i in range(len(atom_types))]

                # Degree
                degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                degree = atom.GetDegree()
                feat += [int(i == degree) for i in degrees]

                # Other features
                feat.append(atom.GetFormalCharge() / 5.0)
                feat.append(atom.GetNumExplicitHs() / 8.0)
                feat.append(atom.GetNumRadicalElectrons() / 4.0)

                # Hybridization
                from rdkit.Chem import rdchem
                hyb_types = [
                    rdchem.HybridizationType.SP,
                    rdchem.HybridizationType.SP2,
                    rdchem.HybridizationType.SP3,
                    rdchem.HybridizationType.SP3D,
                    rdchem.HybridizationType.SP3D2,
                ]
                hyb = atom.GetHybridization()
                feat += [int(hyb == h) for h in hyb_types]

                feat.append(float(atom.GetIsAromatic()))
                feat.append(float(atom.IsInRing()))

                x.append(feat)

            x = torch.tensor(x, dtype=torch.float)

            # ── Edge (bond) features ─────────────────────────────────────
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # Add both directions (undirected graph)
                edge_index += [[i, j], [j, i]]

                from rdkit.Chem import rdchem
                bond_types = [
                    rdchem.BondType.SINGLE,
                    rdchem.BondType.DOUBLE,
                    rdchem.BondType.TRIPLE,
                    rdchem.BondType.AROMATIC,
                ]
                btype = bond.GetBondType()
                bfeat = [int(btype == b) for b in bond_types]
                bfeat.append(float(bond.GetIsConjugated()))
                bfeat.append(float(bond.IsInRing()))

                # Stereo (one-hot)
                stereo_types = [
                    rdchem.BondStereo.STEREONONE,
                    rdchem.BondStereo.STEREOANY,
                    rdchem.BondStereo.STEREOZ,
                    rdchem.BondStereo.STEREOE,
                    rdchem.BondStereo.STEREOCIS,
                    rdchem.BondStereo.STEREOTRANS,
                ]
                stereo = bond.GetStereo()
                bfeat += [int(stereo == s) for s in stereo_types]

                edge_attr.append(bfeat)
                edge_attr.append(bfeat)  # same for both directions

            if not edge_index:
                # Single atom molecule
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 12), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        except Exception as e:
            logger.debug(f"Graph build failed for SMILES: {e}")
            return None

    @staticmethod
    def _randomize_smiles(smiles: str) -> str:
        """Generate a random SMILES representation (augmentation)."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol, doRandom=True)
        except Exception:
            return smiles

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        cache_key = self._cache_key(idx)

        # Try cache first
        cached = self._load_from_cache(cache_key)
        if cached is not None and not self.augment:
            cached["affinity"] = torch.tensor(
                [row["neg_log_affinity"]], dtype=torch.float32
            )
            return cached

        # Tokenize protein
        protein_ids, protein_mask = self._tokenize_protein(row["sequence"])

        # Tokenize ligand (SMILES)
        ligand_ids, ligand_mask = self._tokenize_ligand(row["smiles"])

        item = {
            "pdb_id": row["pdb_id"],
            "protein_ids": protein_ids,
            "protein_mask": protein_mask,
            "ligand_ids": ligand_ids,
            "ligand_mask": ligand_mask,
            "affinity": torch.tensor([row["neg_log_affinity"]], dtype=torch.float32),
            "sequence": row["sequence"],
            "smiles": row["smiles"],
        }

        # Optional molecular graph
        if self.include_graph:
            graph = self._build_mol_graph(row["smiles"])
            item["mol_graph"] = graph

        # Cache (without affinity to allow label updates)
        cache_data = {k: v for k, v in item.items() if k != "affinity"}
        self._save_to_cache(cache_key, cache_data)

        return item


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    protein_tokenizer,
    ligand_tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    include_graph: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    The test set corresponds to PDBbind Core Set (CASF-2016 benchmark).
    Never train or tune on this set — treat it as a sealed envelope.
    """
    common_kwargs = dict(
        protein_tokenizer=protein_tokenizer,
        ligand_tokenizer=ligand_tokenizer,
        cache_dir=cache_dir,
        include_graph=include_graph,
    )

    train_ds = BindingAffinityDataset(train_df, augment=True, **common_kwargs)
    val_ds = BindingAffinityDataset(val_df, augment=False, **common_kwargs)
    test_ds = BindingAffinityDataset(test_df, augment=False, **common_kwargs)

    # Custom collate to handle optional mol_graph
    def collate_fn(batch):
        # Filter out None graphs
        if "mol_graph" in batch[0] and batch[0]["mol_graph"] is None:
            for item in batch:
                item.pop("mol_graph", None)

        collated = {}
        for key in batch[0]:
            if key == "mol_graph":
                from torch_geometric.data import Batch
                graphs = [item["mol_graph"] for item in batch if item.get("mol_graph") is not None]
                collated[key] = Batch.from_data_list(graphs) if graphs else None
            elif key in ("pdb_id", "sequence", "smiles"):
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        return collated

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Stability for batch norm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"DataLoaders created:")
    logger.info(f"  Train: {len(train_ds)} complexes, {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_ds)} complexes, {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_ds)} complexes, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
