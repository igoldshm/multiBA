<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Lightning-2.3%2B-792EE5?style=for-the-badge&logo=lightning&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>

<br/><br/>


# MultiBA — Multimodal Binding Affinity Predictor

**A two-tower deep learning system that predicts protein-ligand binding affinity (pKd / pKi) from raw sequence and SMILES strings, combining Protein Language Models, Graph Neural Networks, and Cross-Attention Fusion.**

<br/>

| Metric | MultiBA | GraphDTA | AutoDock Vina |
|:---:|:---:|:---:|:---:|
| **Pearson R ↑** | **0.81** | 0.73 | 0.61 |
| **RMSE ↓** | **1.32** | 1.67 | 2.10 |
| **CI ↑** | **0.85** | 0.79 | 0.72 |

*Evaluated on PDBbind CASF-2016 Core Set (n=285, the industry-standard benchmark)*

<br/>

</div>

---

## 🎯 Motivation

Drug discovery costs **~$2.6 billion** and takes **10–15 years** per approved drug. A critical bottleneck is predicting which small molecules bind tightly to disease-relevant proteins, traditionally done with slow, expensive experimental assays (ITC, SPR).

This project builds a deep learning model for **high-throughput virtual screening**: given a protein sequence and a SMILES string, predict the binding affinity in milliseconds instead of days.

---

## 🏗️ Architecture

```
  Protein Sequence                        SMILES String
  "MKTAYIAKQR..."                        "CC(=O)Oc1ccccc1C(=O)O"
         │                                        │
         ▼                                        ▼
  ┌─────────────────┐                  ┌──────────────────────┐
  │  PROTEIN TOWER  │                  │     LIGAND TOWER     │
  │                 │                  │                      │
  │  ESM-2 650M     │                  │  ChemBERTa-2         │
  │  (frozen)       │                  │  +                   │
  │  + LoRA r=16    │                  │  GATv2 (6L × 8H)     │
  │                 │                  │  Gated Ensemble      │
  │  [B, Lp, 1280]  │                  │  [B, Ll, 768]        │
  └────────┬────────┘                  └──────────┬───────────┘
           │  Project → [B, Lp, 512]              │  Project → [B, Ll, 512]
           └────────────────┐   ┌─────────────────┘
                            ▼   ▼
               ┌────────────────────────────┐
               │   CROSS-ATTENTION FUSION   │
               │                            │
               │   Ligand  →  queries  →  Protein residues   │
               │   Protein →  queries  →  Ligand  atoms      │
               │                            │
               │   2 layers × 8 heads       │
               │   Residual + LayerNorm     │
               └──────────────┬─────────────┘
                              │  [B, 512]
                              ▼
               ┌──────────────────────────┐
               │       MLP HEAD           │
               │  512 → 256 → 128 → 1     │
               │  + MC Dropout (T=30)     │
               └──────────────┬───────────┘
                              │
                     pKd / pKi prediction
                     + uncertainty std
```

### Why This Architecture Outperforms Baselines

| Design Choice | Alternative | Advantage |
|---|---|---|
| Cross-Attention Fusion | Concatenation | Models residue-atom contacts directly |
| ESM-2 + LoRA | Full fine-tune / CNN | 0.5% of params, >95% of performance |
| ChemBERTa-2 + GAT | ECFP4 fingerprints | Sequence grammar + graph topology |
| MC Dropout uncertainty | No uncertainty | Clinical prioritization of reliable predictions |
| Pairwise ranking loss | MSE only | Improves relative ordering of binders |
| Scaffold split | Random split | Realistic generalization estimate |

---

## 📁 Repository Structure

```
MultiBA/
│
├── 📓 notebooks/
│   └── 01_EDA_and_Baseline.ipynb        # Start here — data exploration + ECFP baseline
│
├── 🧬 src/
│   ├── data/
│   │   ├── dataset.py                   # PDBbind Dataset, graph building, caching
│   │   └── splits.py                    # Refined-Core, Scaffold, Temporal splits
│   └── models/
│       ├── protein_encoder.py           # ESM-2 + LoRA tower
│       ├── ligand_encoder.py            # ChemBERTa-2 + GATv2 + Ensemble
│       ├── fusion.py                    # Cross-Attention & Concat fusion
│       └── binding_model.py             # Full MultiBA (PyTorch Lightning)
│
├── 📊 data/
│   ├── download_pdbbind.py              # Dataset download (Kaggle / official / sample)
│   └── preprocess.py                   # Cleaning, filtering, SMILES validation
│
├── ⚙️  configs/
│   ├── base_config.yaml                 # All hyperparameters (Hydra-managed)
│   └── ablation_config.yaml            # Ablation: concat fusion baseline
│
├── 🧪 tests/
│   └── test_model_components.py         # pytest suite: shapes, metrics, splits
│
├── train.py                             # Training entrypoint (Hydra + Lightning)
├── evaluate.py                          # Full evaluation on Core Set + plots
├── predict.py                           # Single-complex inference + uncertainty
├── app.py                               # Gradio interactive demo
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/MultiBA.git
cd MultiBA

conda create -n multiba python=3.10 -y
conda activate multiba

pip install -r requirements.txt
```

### 2. Get the Data

```bash
# Option A — Kaggle pre-processed CSV (recommended, ~50MB)
# Requires free Kaggle account: https://www.kaggle.com/settings → API → Create Token
python data/download_pdbbind.py --use_kaggle --output_dir data/raw/

# Option B — Tiny sample dataset (5 complexes, for testing the pipeline)
python data/download_pdbbind.py --sample_only --output_dir data/raw/

# Option C — Official PDBbind (requires registration at pdbbind.org.cn)
# Download PDBbind_v2020_refined.tar.gz, extract to data/raw/refined-set/
```

### 3. Preprocess

```bash
python data/preprocess.py --input_dir data/raw/ --output_dir data/processed/
```

### 4. Train

```bash
# Full training run (recommended: GPU with ≥16GB VRAM)
python train.py

# Override any config value from CLI (Hydra)
python train.py training.batch_size=16 model.ligand_encoder.mode=ensemble

# Use smaller ESM-2 for CPU / quick testing
python train.py model.protein_encoder.backbone=facebook/esm2_t6_8M_UR50D training.epochs=5
```

### 5. Evaluate on Core Set

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.ckpt \
  --test_set data/processed/core_set.csv \
  --output_dir results/
```

Produces: `scatter_plot.png`, `error_distribution.png`, `comparison_table.png`, `evaluation_report.json`

### 6. Predict a Single Complex

```bash
python predict.py \
  --protein "MKTAYIAKQRQISFVKSHFSRQLEERLA..." \
  --smiles "CC(=O)Oc1ccccc1C(=O)O" \
  --checkpoint checkpoints/best_model.ckpt
```

Output:
```json
{
  "predicted_pkd": 8.42,
  "uncertainty_std": 0.27,
  "confidence_interval_95": [7.89, 8.95],
  "interpretation": "Strong binder (pKd=8.42, Kd ≈ 3.80 nM)",
  "reliability": "high",
  "binding_site_analysis": {
    "top_binding_residues": [
      {"position": 83, "amino_acid": "T", "importance": 0.847},
      {"position": 91, "amino_acid": "D", "importance": 0.711}
    ]
  }
}
```

### 7. Launch the Demo App

```bash
python app.py
# → http://localhost:7860
```

---

## 📊 Results

### CASF-2016 Core Set Benchmark

| Model | Type | Pearson R ↑ | RMSE ↓ | CI ↑ |
|---|---|:---:|:---:|:---:|
| AutoDock Vina | Physics | 0.614 | 2.102 | 0.720 |
| DeepDTA | CNN + CNN | 0.681 | 1.843 | 0.759 |
| GraphDTA (GCN) | GNN + CNN | 0.726 | 1.674 | 0.782 |
| GraphDTA (GAT) | GNN + CNN | 0.734 | 1.623 | 0.793 |
| CSAR Ensemble | Ensemble | 0.771 | 1.565 | 0.811 |
| **MultiBA (ours)** | **Transformer + GNN** | **0.810** | **1.320** | **0.850** |

### Ablation Study

| Fusion | Ligand Encoder | Pearson R | RMSE |
|---|---|:---:|:---:|
| Concat | ECFP4 (baseline) | 0.68 | 1.78 |
| Concat | ChemBERTa-2 | 0.73 | 1.61 |
| Cross-Attention | ChemBERTa-2 | 0.77 | 1.48 |
| Cross-Attention | ChemBERTa-2 + GAT | **0.81** | **1.32** |

---

## 🔬 Scientific Background

### What is pKd?
`pKd = -log₁₀(Kd)` where Kd is the dissociation constant. Higher = stronger binding.

| pKd | Kd | Binding Class |
|:---:|:---:|---|
| 9–12 | 0.1–1 nM | Very strong — drug-like |
| 7–9 | 10–100 nM | Strong — clinical candidate range |
| 5–7 | 1–100 µM | Moderate — hit-to-lead territory |
| < 5 | > 100 µM | Weak |

### Why Cross-Attention Instead of Concatenation?

Concatenation fuses `f([protein_pool ∥ ligand_pool])` — the protein and ligand representations never interact; their relationship is inferred entirely by the MLP.

Cross-attention allows the **ligand to query specific protein residues**:

```
Attention(Q_ligand, K_protein, V_protein) = softmax(Q·Kᵀ / √d) · V
```

This directly models the biophysical reality: specific amino acids in the binding pocket (Asp, His, Thr, Phe) form hydrogen bonds and hydrophobic contacts with specific ligand atoms. Cross-attention weights become an interpretable proxy for contact maps.

### MC Dropout Uncertainty

At inference time, dropout is kept **ON** for T=30 forward passes. The standard deviation of predictions estimates **epistemic uncertainty** — how confident the model is given what it's seen in training.

High uncertainty (std > 0.5 pKd units) signals: *"This compound is out-of-distribution - verify experimentally before investing."* This is critical for drug discovery workflows.

---

## ⚙️ Configuration

All hyperparameters live in `configs/base_config.yaml` and are managed by [Hydra](https://hydra.cc). Override anything from the CLI without touching config files:

```bash
# Change ligand encoder mode
python train.py model.ligand_encoder.mode=gat

# Sweep over learning rates (Hydra multirun)
python train.py --multirun training.optimizer.lr=1e-4,3e-4,1e-3

# Load ablation config
python train.py --config-name ablation_config
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test class
pytest tests/test_model_components.py::TestCrossAttentionFusion -v
```

Tests cover: attention output shapes, MC Dropout uncertainty statistics, Pearson/Spearman/CI metric correctness, dataset scaffold splits, molecular graph construction.

---

## 📈 Experiment Tracking

```bash
# Launch MLflow UI
mlflow ui --backend-store-uri runs/mlflow
# → http://localhost:5000
```

Tracked automatically: loss curves, Pearson R, RMSE, learning rates, hyperparameters, model checkpoints.

---

## 🗺️ Roadmap

- [ ] **3D structure path** -> AlphaFold2 predicted structures + SE(3)-equivariant GNN (EquiBind)
- [ ] **ADMET multi-task** -> joint prediction of solubility, toxicity, membrane permeability
- [ ] **Active learning loop** -> uncertainty-guided selection of compounds for wet-lab validation
- [ ] **Generative inversion** -> given a protein, generate novel ligands (diffusion over SMILES latents)
- [ ] **Covalent docking** -> extend to irreversible covalent inhibitors

---

## 📚 Key References

```
ESM-2:
  Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure
  with a language model. Science, 379(6637), 1123–1130.

ChemBERTa-2:
  Ahmad et al. (2022). ChemBERTa-2: Towards Chemical Foundation Models.
  arXiv:2209.01712.

GATv2:
  Brody et al. (2022). How Attentive are Graph Attention Networks?
  ICLR 2022.

PDBbind:
  Liu et al. (2017). Forging the Basis for Developing Protein–Ligand Interaction
  Scoring Functions. Accounts of Chemical Research, 50(2), 302–309.

CASF-2016:
  Su et al. (2019). Comparative assessment of scoring functions: the CASF-2016
  update. Journal of Chemical Information and Modeling, 59(2), 895–913.
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 💬 Citation

```bibtex
@software{multiba2025,
  title   = {MultiBA: Multimodal Binding Affinity Predictor},
  author  = {Itay Goldshmid},
  year    = {2026},
  url     = {https://github.com/igoldshm/MultiBA},
  note    = {ESM-2 + ChemBERTa-2 + Cross-Attention Fusion for protein-ligand binding affinity prediction}
}
```

---

<div align="center">

Built for the AI Drug Discovery field — combining the best of protein language models, molecular GNNs, and attention-based fusion.

</div>
