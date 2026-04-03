"""
src/data/splits.py
━━━━━━━━━━━━━━━━━━
Train / validation / test splitting strategies for PDBbind.

Strategies:
  1. refined_core   — Official PDBbind split. Refined Set = train+val, Core Set = test.
                      This is the CASF-2016 benchmark used in all published papers.
                      USE THIS FOR FAIR COMPARISON.

  2. scaffold       — Scaffold-based split (Bemis-Murcko scaffolds).
                      Ensures ligands in val/test have novel scaffolds vs. train.
                      More realistic for prospective drug discovery.

  3. random         — Random split. Easiest, but overly optimistic (data leakage).

  4. temporal       — Split by PDB deposition year. Tests generalization to
                      newer complexes (relevant for real deployment).
"""

from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from loguru import logger
from pathlib import Path


# PDBbind v2020 Core Set PDB IDs (CASF-2016 benchmark — 285 complexes)
# Source: http://www.pdbbind.org.cn/casf.asp
CORE_SET_IDS = {
    "1a1e", "1a28", "1a9m", "1ai5", "1b9t", "1bcu", "1bma", "1c5z",
    "1cvu", "1d3p", "1f3d", "1fcx", "1fvt", "1g2k", "1gpk", "1gz8",
    "1h23", "1hnn", "1hp0", "1hq2", "1hwi", "1hww", "1ig3", "1j3j",
    "1k1i", "1ke5", "1kzk", "1l2s", "1l7f", "1lbk", "1lbm", "1lbs",
    "1lbu", "1lpg", "1m48", "1mq6", "1mzc", "1n1m", "1n2v", "1n46",
    "1nav", "1njs", "1noy", "1nxk", "1o5b", "1of1", "1of6", "1ohr",
    "1oit", "1opk", "1oq5", "1or0", "1osk", "1ow4", "1owh", "1p1n",
    "1p1q", "1pmn", "1q1g", "1q41", "1q5k", "1qau", "1qi0", "1r1h",
    "1r55", "1r9l", "1rnt", "1s19", "1s3v", "1sg0", "1sj0", "1sqa",
    "1sqn", "1t46", "1t7r", "1tp5", "1u4d", "1uml", "1unl", "1uou",
    "1v0p", "1v48", "1vcj", "1vso", "1w1p", "1w2g", "1xm6", "1xoq",
    "1xoz", "1yc1", "1ygb", "1ygc", "1ype", "1z95", "2b8t", "2br1",
    "2bsm", "2bvt", "2c3i", "2cbs", "2cej", "2clb", "2cnk", "2cqz",
    "2d1o", "2d3u", "2d4q", "2dq7", "2dri", "2e1w", "2f2h", "2f4j",
    "2fvd", "2goo", "2gtv", "2h4n", "2hiw", "2hn1", "2i0a", "2i1m",
    "2ica", "2j62", "2je4", "2jkm", "2nns", "2obs", "2p4y", "2p95",
    "2pcp", "2pm8", "2pq9", "2qbp", "2qbr", "2qbs", "2qnq", "2r23",
    "2r5p", "2reg", "2ums", "2v7a", "2vkm", "2vot", "2w66", "2wcg",
    "2wer", "2wn9", "2x97", "2xbv", "2xhb", "2xnb", "2y5h", "2ymd",
    "2zcl", "2zcq", "2zdt", "3ag9", "3b27", "3b5r", "3be9", "3bkk",
    "3bpc", "3cj4", "3clu", "3coy", "3cpa", "3cr4", "3cvo", "3d4q",
    "3dbs", "3dck", "3dge", "3dmt", "3dxg", "3ebp", "3ejr", "3eju",
    "3enz", "3f3c", "3f3d", "3f3e", "3fk1", "3fq6", "3fv1", "3g0w",
    "3g2y", "3g2z", "3g31", "3g6z", "3gbb", "3ge7", "3gr2", "3hbf",
    "3hpw", "3hz1", "3i3b", "3imc", "3in3", "3iod", "3kl6", "3kme",
    "3ko0", "3l4w", "3lka", "3lkf", "3m2w", "3mbp", "3mss", "3n73",
    "3n7a", "3n86", "3now", "3nq9", "3nv5", "3nw9", "3o0i", "3oe5",
    "3ozt", "3p3h", "3p5o", "3pap", "3phg", "3pmk", "3pxf", "3qgy",
    "3r88", "3rsx", "3s8o", "3sj0", "3skj", "3su2", "3szk", "3utu",
    "3vd4", "3vri", "3wz8", "3zmf", "4abd", "4cig", "4djv", "4e6q",
    "4f9w", "4gid", "4gqq", "4ivc", "4j21", "4k18", "4kzq", "4mdh",
    "4mme", "4ogj", "4ovn", "4pcs", "4q3f", "4rlu", "4tmn", "4twp",
    "4ty7", "4u4s", "4w52", "4w9h", "4wkq", "5a7b", "5c2h", "5cfo",
    "5dwr", "5edu", "5ei2", "5em1", "5er3", "5f7l", "5gjm", "5j18",
}


def refined_core_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Official PDBbind split:
      - Test  = Core Set (285 complexes, CASF-2016)
      - Train = Refined Set \ Core Set, minus val_fraction
      - Val   = val_fraction of remaining train

    This is the ONLY split that enables fair comparison to published results.
    """
    df = df.copy()
    test_mask = df["pdb_id"].str.lower().isin(CORE_SET_IDS)
    test_df = df[test_mask].reset_index(drop=True)
    trainval_df = df[~test_mask].reset_index(drop=True)

    # Random val split from remaining train data
    rng = np.random.RandomState(seed)
    val_idx = rng.choice(len(trainval_df), size=int(len(trainval_df) * val_fraction), replace=False)
    train_idx = np.setdiff1d(np.arange(len(trainval_df)), val_idx)

    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Refined-Core split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def scaffold_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scaffold-based split using Bemis-Murcko scaffolds (RDKit).

    Complexes sharing the same scaffold are kept together in the same split.
    This prevents the model from "memorizing" scaffold-level patterns —
    important for realistic prospective screening performance.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        logger.warning("RDKit not available. Falling back to random split.")
        return random_split(df, val_fraction, test_fraction, seed)

    df = df.copy()

    # Compute Murcko scaffold for each SMILES
    scaffolds = {}
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChiralCenters=False)
                scaffolds[idx] = scaffold
            else:
                scaffolds[idx] = row["smiles"]  # Fallback to full SMILES
        except Exception:
            scaffolds[idx] = row["smiles"]

    df["scaffold"] = pd.Series(scaffolds)

    # Group by scaffold, sort by size (ascending — put rare scaffolds in test)
    scaffold_groups = df.groupby("scaffold").indices
    scaffold_sizes = sorted(scaffold_groups.keys(), key=lambda s: len(scaffold_groups[s]))

    # Assign scaffolds to splits
    train_idx, val_idx, test_idx = [], [], []
    n = len(df)
    val_target = int(n * val_fraction)
    test_target = int(n * test_fraction)

    rng = np.random.RandomState(seed)
    scaffold_list = list(scaffold_sizes)
    rng.shuffle(scaffold_list)

    for scaffold in scaffold_list:
        idxs = list(scaffold_groups[scaffold])
        if len(test_idx) < test_target:
            test_idx.extend(idxs)
        elif len(val_idx) < val_target:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    train_df = df.iloc[train_idx].drop(columns=["scaffold"]).reset_index(drop=True)
    val_df = df.iloc[val_idx].drop(columns=["scaffold"]).reset_index(drop=True)
    test_df = df.iloc[test_idx].drop(columns=["scaffold"]).reset_index(drop=True)

    logger.info(f"Scaffold split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    _log_scaffold_overlap(train_df, val_df, test_df)
    return train_df, val_df, test_df


def temporal_split(
    df: pd.DataFrame,
    val_year: int = 2018,
    test_year: int = 2019,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal split by PDB deposition year.
    Mimics prospective deployment where we train on historical data.
    """
    df = df.copy()
    if "year" not in df.columns:
        raise ValueError("DataFrame must have a 'year' column for temporal split.")

    train_df = df[df["year"] < val_year].reset_index(drop=True)
    val_df = df[(df["year"] >= val_year) & (df["year"] < test_year)].reset_index(drop=True)
    test_df = df[df["year"] >= test_year].reset_index(drop=True)

    logger.info(f"Temporal split (train<{val_year}, val={val_year}-{test_year-1}, test>={test_year}):")
    logger.info(f"  train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def random_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random split (baseline; may overestimate performance)."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test_df = df[:n_test].reset_index(drop=True)
    val_df = df[n_test : n_test + n_val].reset_index(drop=True)
    train_df = df[n_test + n_val :].reset_index(drop=True)

    logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def _log_scaffold_overlap(train_df, val_df, test_df):
    """Check for scaffold leakage between splits."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        def get_scaffolds(df):
            scaffolds = set()
            for smi in df["smiles"]:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChiralCenters=False)
                        scaffolds.add(s)
                except Exception:
                    pass
            return scaffolds

        train_scaffolds = get_scaffolds(train_df)
        val_scaffolds = get_scaffolds(val_df)
        test_scaffolds = get_scaffolds(test_df)

        val_overlap = len(train_scaffolds & val_scaffolds) / max(len(val_scaffolds), 1)
        test_overlap = len(train_scaffolds & test_scaffolds) / max(len(test_scaffolds), 1)

        logger.info(f"Scaffold overlap — val: {val_overlap:.1%}, test: {test_overlap:.1%}")
        if val_overlap > 0.1:
            logger.warning("High scaffold overlap in val set — consider stricter splitting")

    except ImportError:
        pass
