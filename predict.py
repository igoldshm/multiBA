"""
predict.py
━━━━━━━━━━
Single-complex binding affinity prediction with uncertainty + attention.

Usage:
  python predict.py \\
    --protein "MKTAYIAKQRQISFVKSHFSRQ..." \\
    --smiles "CC(=O)Oc1ccccc1C(=O)O" \\
    --checkpoint checkpoints/best_model.ckpt

Output (JSON):
  {
    "predicted_pkd": 8.42,
    "uncertainty_std": 0.31,
    "confidence_interval_95": [7.81, 9.03],
    "interpretation": "Predicted strong binder (pKd > 7 = nanomolar range)",
    "molecule_info": { ... }
  }
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
from loguru import logger


def interpret_affinity(pkd: float) -> str:
    """Translate pKd to a human-readable binding strength."""
    if pkd >= 9:
        return f"Very strong binder (pKd={pkd:.2f}, Kd ≈ {10**(-pkd)*1e9:.2f} nM) — drug-like"
    elif pkd >= 7:
        return f"Strong binder (pKd={pkd:.2f}, Kd ≈ {10**(-pkd)*1e9:.2f} nM)"
    elif pkd >= 5:
        kd_um = 10**(-pkd) * 1e6
        return f"Moderate binder (pKd={pkd:.2f}, Kd ≈ {kd_um:.2f} µM)"
    else:
        kd_mm = 10**(-pkd) * 1e3
        return f"Weak binder (pKd={pkd:.2f}, Kd ≈ {kd_mm:.2f} mM)"


def get_molecule_info(smiles: str) -> dict:
    """Extract molecular properties using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),     # H-bond donors
            "hba": rdMolDescriptors.CalcNumHBA(mol),     # H-bond acceptors
            "tpsa": round(Descriptors.TPSA(mol), 2),     # Topological polar surface area
            "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "lipinski_ro5": all([
                Descriptors.MolWt(mol) <= 500,
                Descriptors.MolLogP(mol) <= 5,
                rdMolDescriptors.CalcNumHBD(mol) <= 5,
                rdMolDescriptors.CalcNumHBA(mol) <= 10,
            ]),
        }
    except ImportError:
        return {"error": "RDKit not installed"}
    except Exception as e:
        return {"error": str(e)}


def predict(
    protein_sequence: str,
    smiles: str,
    checkpoint_path: str,
    device: str = "auto",
    mc_samples: int = 30,
    return_attention: bool = True,
) -> dict:
    """
    Predict binding affinity for a single protein-ligand pair.

    Parameters
    ----------
    protein_sequence : str
        Single-letter amino acid sequence
    smiles : str
        Canonical SMILES string of the ligand
    checkpoint_path : str
        Path to trained MultiBA .ckpt file
    device : str
        "auto", "cpu", "cuda", or "mps"
    mc_samples : int
        Number of MC Dropout samples for uncertainty
    return_attention : bool
        Whether to include attention weight analysis

    Returns
    -------
    dict with prediction, uncertainty, molecule info, and optional attention
    """
    # ── Device ────────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    logger.info(f"Using device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.binding_model import MultiBA

    model = MultiBA.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model = model.to(device)

    # ── Tokenizers ────────────────────────────────────────────────────────
    from transformers import AutoTokenizer, EsmTokenizer
    prot_name = model.config["model"]["protein_encoder"]["backbone"]
    lig_name = model.config["model"]["ligand_encoder"]["chembert"]["backbone"]
    protein_tokenizer = EsmTokenizer.from_pretrained(prot_name)
    ligand_tokenizer = AutoTokenizer.from_pretrained(lig_name)

    # ── Tokenize ──────────────────────────────────────────────────────────
    max_prot = model.config["data"]["max_protein_len"]
    max_lig = model.config["data"]["max_smiles_len"]

    prot_enc = protein_tokenizer(
        protein_sequence, max_length=max_prot + 2, truncation=True,
        padding="max_length", return_tensors="pt"
    )
    lig_enc = ligand_tokenizer(
        smiles, max_length=max_lig, truncation=True,
        padding="max_length", return_tensors="pt"
    )

    protein_ids = prot_enc["input_ids"].to(device)
    protein_mask = prot_enc["attention_mask"].to(device)
    ligand_ids = lig_enc["input_ids"].to(device)
    ligand_mask = lig_enc["attention_mask"].to(device)

    # ── Build molecular graph (for GAT path) ─────────────────────────────
    mol_graph = None
    try:
        from src.data.dataset import BindingAffinityDataset
        dummy_ds = BindingAffinityDataset.__new__(BindingAffinityDataset)
        graph = dummy_ds._build_mol_graph(smiles)
        if graph is not None:
            from torch_geometric.data import Batch
            mol_graph = Batch.from_data_list([graph]).to(device)
    except Exception:
        pass

    # ── Predict with uncertainty ──────────────────────────────────────────
    with torch.no_grad():
        mean, std = model.predict_with_uncertainty(
            protein_ids, protein_mask, ligand_ids, ligand_mask,
            mol_graph=mol_graph, num_mc_samples=mc_samples
        )

    pkd_mean = mean.item()
    pkd_std = std.item()

    # 95% confidence interval (approximate, assuming normal distribution)
    ci_low = pkd_mean - 1.96 * pkd_std
    ci_high = pkd_mean + 1.96 * pkd_std

    # ── Attention analysis ────────────────────────────────────────────────
    attention_info = None
    if return_attention:
        try:
            with torch.no_grad():
                out = model(
                    protein_ids, protein_mask, ligand_ids, ligand_mask,
                    mol_graph=mol_graph, return_attention=True
                )

            l2p_attn = out.get("lig2prot_attn")  # [1, num_heads, Ll, Lp]
            if l2p_attn is not None:
                # Mean over heads, collapse over ligand tokens → protein residue importance
                residue_importance = l2p_attn[0].mean(0).mean(0)  # [Lp]
                prot_seq_len = protein_mask.sum().item() - 2  # Exclude special tokens
                residue_importance = residue_importance[1:int(prot_seq_len)+1].cpu().numpy()

                # Top binding residues
                top_k = 10
                top_idx = np.argsort(residue_importance)[-top_k:][::-1]
                top_residues = [
                    {
                        "position": int(i) + 1,  # 1-indexed
                        "amino_acid": protein_sequence[i] if i < len(protein_sequence) else "?",
                        "importance": float(residue_importance[i]),
                    }
                    for i in top_idx
                    if i < len(residue_importance)
                ]

                attention_info = {
                    "top_binding_residues": top_residues,
                    "note": "These residues are predicted to most strongly interact with the ligand",
                }
        except Exception as e:
            attention_info = {"error": f"Attention extraction failed: {str(e)}"}

    # ── Molecule properties ───────────────────────────────────────────────
    mol_info = get_molecule_info(smiles)

    # ── Compile results ───────────────────────────────────────────────────
    result = {
        "predicted_pkd": round(pkd_mean, 3),
        "uncertainty_std": round(pkd_std, 3),
        "confidence_interval_95": [round(ci_low, 3), round(ci_high, 3)],
        "interpretation": interpret_affinity(pkd_mean),
        "reliability": "high" if pkd_std < 0.3 else "medium" if pkd_std < 0.6 else "low",
        "protein_sequence_length": len(protein_sequence),
        "smiles": smiles,
        "molecule_properties": mol_info,
    }

    if attention_info:
        result["binding_site_analysis"] = attention_info

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict binding affinity with MultiBA")
    parser.add_argument("--protein", required=True, help="Amino acid sequence (single-letter)")
    parser.add_argument("--smiles", required=True, help="Canonical SMILES string")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt model file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--mc_samples", type=int, default=30, help="MC Dropout samples")
    parser.add_argument("--no_attention", action="store_true", help="Skip attention analysis")
    parser.add_argument("--output", default=None, help="Optional: save JSON to file")
    args = parser.parse_args()

    result = predict(
        protein_sequence=args.protein,
        smiles=args.smiles,
        checkpoint_path=args.checkpoint,
        device=args.device,
        mc_samples=args.mc_samples,
        return_attention=not args.no_attention,
    )

    output_json = json.dumps(result, indent=2)
    print("\n" + output_json)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
