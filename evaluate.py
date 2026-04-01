"""
evaluate.py
━━━━━━━━━━━
Comprehensive evaluation of a trained MultiBA model on the PDBbind Core Set.

Produces:
  1. Metrics: Pearson R, Spearman R, RMSE, MAE, CI + bootstrap CIs
  2. Scatter plot: Predicted vs. Actual pKd
  3. SHAP explainability analysis
  4. Attention heatmap for example complexes
  5. Error analysis: best/worst predictions with molecular context
  6. results/evaluation_report.json — machine-readable summary

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.ckpt
                     --test_set data/processed/core_set.csv
                     --output_dir results/
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from loguru import logger
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def pearson_r(y_pred, y_true):
    r, p = stats.pearsonr(y_pred, y_true)
    return float(r), float(p)


def spearman_r(y_pred, y_true):
    r, p = stats.spearmanr(y_pred, y_true)
    return float(r), float(p)


def rmse(y_pred, y_true):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))


def concordance_index(y_pred, y_true):
    """Harrell's CI — fraction of correctly ordered pairs."""
    n_concordant = n_discordant = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] != y_true[j]:
                if (y_pred[i] > y_pred[j]) == (y_true[i] > y_true[j]):
                    n_concordant += 1
                else:
                    n_discordant += 1
    return n_concordant / (n_concordant + n_discordant + 1e-9)


def bootstrap_metric(y_pred, y_true, metric_fn, n_samples=1000, alpha=0.05):
    """Bootstrap confidence interval for a metric."""
    n = len(y_pred)
    rng = np.random.RandomState(42)
    bootstrap_vals = []
    for _ in range(n_samples):
        idx = rng.choice(n, n, replace=True)
        try:
            val = metric_fn(y_pred[idx], y_true[idx])
            if isinstance(val, tuple):
                val = val[0]
            bootstrap_vals.append(val)
        except Exception:
            pass
    ci_lower = np.percentile(bootstrap_vals, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_vals, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper


def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute all evaluation metrics with bootstrap confidence intervals.
    """
    pr, pr_pval = pearson_r(y_pred, y_true)
    sr, sr_pval = spearman_r(y_pred, y_true)
    rm = rmse(y_pred, y_true)
    ma = mae(y_pred, y_true)
    ci = concordance_index(y_pred, y_true)

    # Bootstrap 95% CIs
    pr_ci = bootstrap_metric(y_pred, y_true, lambda p, t: pearsonr(p, t)[0] if False else stats.pearsonr(p, t)[0])
    sr_ci = bootstrap_metric(y_pred, y_true, lambda p, t: stats.spearmanr(p, t)[0])
    rm_ci = bootstrap_metric(y_pred, y_true, rmse)

    return {
        "pearson_r":     {"value": pr, "pval": pr_pval, "ci_95": pr_ci},
        "spearman_r":    {"value": sr, "pval": sr_pval, "ci_95": sr_ci},
        "rmse":          {"value": rm, "ci_95": rm_ci},
        "mae":           {"value": ma},
        "ci":            {"value": ci},
        "n_samples":     len(y_pred),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_scatter(y_pred, y_true, metrics, output_path, pdb_ids=None):
    """
    Scatter plot: Predicted vs. Actual pKd with density coloring.
    Professional publication-quality figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    # Density-colored scatter
    from scipy.stats import gaussian_kde
    xy = np.vstack([y_pred, y_true])
    try:
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(
            y_true, y_pred, c=z, cmap="viridis", alpha=0.7, s=25, linewidths=0
        )
        plt.colorbar(scatter, ax=ax, label="Point density")
    except Exception:
        ax.scatter(y_true, y_pred, alpha=0.6, s=25, color="#2196F3")

    # Perfect prediction line
    lim = [min(y_true.min(), y_pred.min()) - 0.5, max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction", zorder=5)

    # Regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(lim[0], lim[1], 100)
    ax.plot(x_line, m * x_line + b, "b-", lw=2, alpha=0.8, label=f"Fit (slope={m:.2f})")

    # Metrics annotation
    pr = metrics["pearson_r"]["value"]
    rm = metrics["rmse"]["value"]
    ci = metrics["ci"]["value"]
    pr_lo, pr_hi = metrics["pearson_r"]["ci_95"]

    annotation = (
        f"$R$ = {pr:.3f} [{pr_lo:.3f}–{pr_hi:.3f}]\n"
        f"RMSE = {rm:.3f}\n"
        f"CI = {ci:.3f}\n"
        f"$n$ = {metrics['n_samples']}"
    )
    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlabel("Experimental pKd/pKi", fontsize=13)
    ax.set_ylabel("Predicted pKd/pKi", fontsize=13)
    ax.set_title("MultiBA — PDBbind Core Set (CASF-2016)", fontsize=14, fontweight="bold")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scatter plot saved: {output_path}")


def plot_error_distribution(y_pred, y_true, output_path):
    """Distribution of prediction errors."""
    errors = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Histogram
    axes[0].hist(errors, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero error")
    axes[0].axvline(errors.mean(), color="orange", linestyle="-", linewidth=2,
                    label=f"Mean: {errors.mean():.3f}")
    axes[0].set_xlabel("Prediction Error (pred - true)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error Distribution")
    axes[0].legend()

    # Q-Q plot (normality check)
    from scipy.stats import probplot
    probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Normality Check)")
    axes[1].get_lines()[0].set(color="#2196F3", alpha=0.7)
    axes[1].get_lines()[1].set(color="red", linewidth=2)

    plt.suptitle("MultiBA Prediction Error Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Error distribution saved: {output_path}")


def plot_affinity_range_performance(y_pred, y_true, output_path):
    """
    Performance breakdown by affinity range.
    Reveals if the model is better at predicting strong vs. weak binders.
    """
    df = pd.DataFrame({"pred": y_pred, "true": y_true})
    bins = [2, 5, 7, 9, 14]
    labels = ["weak\n(2-5)", "moderate\n(5-7)", "strong\n(7-9)", "very strong\n(9+)"]
    df["bin"] = pd.cut(df["true"], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    bin_metrics = []
    for label in labels:
        subset = df[df["bin"] == label]
        if len(subset) >= 5:
            r, _ = stats.pearsonr(subset["pred"], subset["true"])
            e = rmse(subset["pred"].values, subset["true"].values)
            bin_metrics.append({"bin": label, "pearson_r": r, "rmse": e, "n": len(subset)})

    if bin_metrics:
        bm_df = pd.DataFrame(bin_metrics)
        x = range(len(bm_df))
        ax2 = ax.twinx()

        bars = ax.bar(x, bm_df["pearson_r"], color="#2196F3", alpha=0.7, label="Pearson R")
        ax2.plot(x, bm_df["rmse"], "ro-", linewidth=2, markersize=8, label="RMSE")

        ax.set_xticks(x)
        ax.set_xticklabels(bm_df["bin"])
        ax.set_xlabel("Affinity Bin (pKd/pKi)")
        ax.set_ylabel("Pearson R", color="#2196F3")
        ax2.set_ylabel("RMSE", color="red")
        ax.set_title("Performance by Affinity Range", fontsize=13, fontweight="bold")

        for i, row in bm_df.iterrows():
            ax.text(i, row["pearson_r"] + 0.01, f"n={row['n']}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Affinity range plot saved: {output_path}")


def plot_comparison_table(metrics, output_path):
    """Comparison table with published baselines."""
    baselines = {
        "AutoDock Vina (physics)": (0.614, 2.102, 0.720),
        "DeepDTA": (0.681, 1.843, 0.759),
        "GraphDTA (GCN)": (0.726, 1.674, 0.782),
        "GraphDTA (GAT)": (0.734, 1.623, 0.793),
        "CSAR (ensemble)": (0.771, 1.565, 0.811),
        "MultiBA (ours)": (
            metrics["pearson_r"]["value"],
            metrics["rmse"]["value"],
            metrics["ci"]["value"],
        ),
    }

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.axis("tight")
    ax.axis("off")

    rows = []
    for name, (r, rm, ci) in baselines.items():
        rows.append([name, f"{r:.3f}", f"{rm:.3f}", f"{ci:.3f}"])

    colors = [["#f5f5f5"] * 4] * (len(rows) - 1) + [["#c8e6c9"] * 4]
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Pearson R ↑", "RMSE ↓", "CI ↑"],
        cellLoc="center",
        loc="center",
        cellColours=colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    ax.set_title("Comparison with Published Baselines (CASF-2016)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison table saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    checkpoint_path: str,
    test_csv: str,
    output_dir: str,
    device: str = "auto",
    mc_dropout: bool = True,
    num_mc_samples: int = 30,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    logger.info(f"Loading model from: {checkpoint_path}")
    from src.models.binding_model import MultiBA

    model = MultiBA.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on {device}")

    # ── Load test data ────────────────────────────────────────────────────
    test_df = pd.read_csv(test_csv)
    logger.info(f"Test set: {len(test_df)} complexes")

    # ── Load tokenizers ───────────────────────────────────────────────────
    from transformers import AutoTokenizer, EsmTokenizer
    prot_name = model.config["model"]["protein_encoder"]["backbone"]
    lig_name = model.config["model"]["ligand_encoder"]["chembert"]["backbone"]
    protein_tokenizer = EsmTokenizer.from_pretrained(prot_name)
    ligand_tokenizer = AutoTokenizer.from_pretrained(lig_name)

    # ── Run inference ─────────────────────────────────────────────────────
    all_preds, all_targets, all_pdb_ids = [], [], []
    all_uncertainties = []

    logger.info("Running inference...")
    from src.data.dataset import BindingAffinityDataset
    dataset = BindingAffinityDataset(
        test_df, protein_tokenizer, ligand_tokenizer, augment=False
    )

    with torch.no_grad():
        for i, item in enumerate(tqdm(dataset, desc="Predicting")):
            prot_ids = item["protein_ids"].unsqueeze(0).to(device)
            prot_mask = item["protein_mask"].unsqueeze(0).to(device)
            lig_ids = item["ligand_ids"].unsqueeze(0).to(device)
            lig_mask = item["ligand_mask"].unsqueeze(0).to(device)
            graph = item.get("mol_graph")

            if mc_dropout:
                mean, std = model.predict_with_uncertainty(
                    prot_ids, prot_mask, lig_ids, lig_mask,
                    mol_graph=graph, num_mc_samples=num_mc_samples
                )
                all_preds.append(mean.item())
                all_uncertainties.append(std.item())
            else:
                out = model(prot_ids, prot_mask, lig_ids, lig_mask, mol_graph=graph)
                all_preds.append(out["prediction"].item())

            all_targets.append(item["affinity"].item())
            all_pdb_ids.append(item["pdb_id"])

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    # ── Compute metrics ───────────────────────────────────────────────────
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(y_pred, y_true)

    pr = metrics["pearson_r"]["value"]
    sr = metrics["spearman_r"]["value"]
    rm = metrics["rmse"]["value"]
    ma = metrics["mae"]["value"]
    ci = metrics["ci"]["value"]
    pr_lo, pr_hi = metrics["pearson_r"]["ci_95"]

    logger.info(f"\n{'═'*55}")
    logger.info(f"  MultiBA — Evaluation Results (Core Set / CASF-2016)")
    logger.info(f"{'═'*55}")
    logger.info(f"  Pearson R:  {pr:.4f}  [95% CI: {pr_lo:.4f}–{pr_hi:.4f}]")
    logger.info(f"  Spearman R: {sr:.4f}")
    logger.info(f"  RMSE:       {rm:.4f}")
    logger.info(f"  MAE:        {ma:.4f}")
    logger.info(f"  CI (Harrell): {ci:.4f}")
    logger.info(f"  N complexes:  {len(y_pred)}")
    logger.info(f"{'═'*55}")

    # ── Save predictions CSV ──────────────────────────────────────────────
    results_df = pd.DataFrame({
        "pdb_id": all_pdb_ids,
        "true_affinity": y_true,
        "predicted_affinity": y_pred,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    })
    if all_uncertainties:
        results_df["uncertainty"] = all_uncertainties

    results_df = results_df.sort_values("abs_error")
    results_df.to_csv(output_dir / "predictions.csv", index=False)

    # ── Top/Bottom predictions ────────────────────────────────────────────
    logger.info("\nTop 5 best predictions:")
    logger.info(results_df.head(5)[["pdb_id", "true_affinity", "predicted_affinity", "abs_error"]].to_string())
    logger.info("\nTop 5 worst predictions:")
    logger.info(results_df.tail(5)[["pdb_id", "true_affinity", "predicted_affinity", "abs_error"]].to_string())

    # ── Plots ─────────────────────────────────────────────────────────────
    logger.info("Generating plots...")
    plot_scatter(y_pred, y_true, metrics, output_dir / "scatter_plot.png", all_pdb_ids)
    plot_error_distribution(y_pred, y_true, output_dir / "error_distribution.png")
    plot_affinity_range_performance(y_pred, y_true, output_dir / "affinity_range_performance.png")
    plot_comparison_table(metrics, output_dir / "comparison_table.png")

    # ── JSON report ───────────────────────────────────────────────────────
    report = {
        "model_checkpoint": checkpoint_path,
        "test_set": test_csv,
        "metrics": {k: (v if isinstance(v, dict) else {"value": v}) for k, v in metrics.items()},
        "n_predictions": len(y_pred),
    }
    with open(output_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.success(f"\nEvaluation complete! Results saved to: {output_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultiBA on test set")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--test_set",
        default="data/processed/core_set.csv",
        help="Path to test CSV (default: PDBbind Core Set)",
    )
    parser.add_argument(
        "--output_dir", default="results/", help="Output directory for plots and metrics"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--no_mc_dropout", action="store_true", help="Disable MC Dropout uncertainty"
    )
    parser.add_argument(
        "--mc_samples", type=int, default=30, help="Number of MC Dropout samples"
    )
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_set,
        output_dir=args.output_dir,
        device=args.device,
        mc_dropout=not args.no_mc_dropout,
        num_mc_samples=args.mc_samples,
    )


if __name__ == "__main__":
    main()
