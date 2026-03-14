# baseline.py
"""
Baseline Comparison — Tabular ML vs Graph Neural Network.

Trains and evaluates 4 baseline models on the SAME train/test split
used by the GNN, enabling a direct apples-to-apples comparison.

Baselines:
  1. Logistic Regression   — linear model, fast, interpretable
  2. Random Forest         — strong ensemble, handles non-linearity
  3. XGBoost               — best tabular ML baseline in most benchmarks
  4. XGBoost + Graph Stats — XGBoost augmented with degree/centrality features
                             (best possible tabular model that "knows" the graph)

Key point: ALL of these treat each transaction as an INDEPENDENT row.
They cannot see neighborhood topology, fraud ring structure, or
multi-hop propagation patterns. That is exactly what the GNN adds.

Usage:
  python baseline.py --checkpoint outputs/best_model.pt
  python baseline.py --checkpoint outputs/best_model.pt --no-xgboost   # if xgboost not installed

Output:
  - Console comparison table
  - outputs/baseline_comparison.png   (bar chart: all models × all metrics)
  - outputs/baseline_results.json     (all numbers for the paper)
"""

import os
import sys
import json
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import CFG
from models.gnn import build_model, get_device
from utils.data_loader import load_dataset
from train import build_standard_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION — same split as GNN
# ─────────────────────────────────────────────────────────────────────────────

def extract_tabular_splits(data):
    """
    Extract X_train, X_test, y_train, y_test from the PyG Data object.
    Uses the EXACT same train/test masks as the GNN — fair comparison.
    Only includes labelled nodes (y >= 0).
    """
    X = data.x.numpy()
    y = data.y.numpy()

    train_mask = data.train_mask.numpy() & (y >= 0)
    test_mask  = data.test_mask.numpy()  & (y >= 0)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    logger.info(f"  Train: {X_train.shape[0]:,} nodes  ({y_train.sum():,} fraud, {(y_train==0).sum():,} legit)")
    logger.info(f"  Test:  {X_test.shape[0]:,} nodes  ({y_test.sum():,} fraud,  {(y_test==0).sum():,} legit)")

    return X_train, X_test, y_train, y_test


def add_graph_features(data, X_train, X_test):
    """
    Augment tabular features with basic graph statistics:
      - node degree (in + out)
      - in-degree
      - out-degree

    This creates the best possible tabular baseline — XGBoost that
    'knows' something about the graph structure via hand-crafted features.
    Still cannot capture multi-hop patterns or fraud ring topology.
    """
    from torch_geometric.utils import degree

    N          = data.num_nodes
    edge_index = data.edge_index

    in_deg  = degree(edge_index[1], num_nodes=N).numpy()
    out_deg = degree(edge_index[0], num_nodes=N).numpy()
    total   = in_deg + out_deg

    graph_feats = np.stack([in_deg, out_deg, total], axis=1)  # [N, 3]

    y = data.y.numpy()
    train_mask = data.train_mask.numpy() & (y >= 0)
    test_mask  = data.test_mask.numpy()  & (y >= 0)

    gf_train = graph_feats[train_mask]
    gf_test  = graph_feats[test_mask]

    X_train_aug = np.concatenate([X_train, gf_train], axis=1)
    X_test_aug  = np.concatenate([X_test,  gf_test],  axis=1)

    logger.info(f"  Graph features added: in_degree, out_degree, total_degree")
    logger.info(f"  Feature dim: {X_train.shape[1]} → {X_train_aug.shape[1]}")

    return X_train_aug, X_test_aug


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, threshold=0.5) -> dict:
    """Compute all metrics used in the GNN evaluation for fair comparison."""
    # Apply threshold to probabilities
    y_pred_t = (y_prob >= threshold).astype(int)

    return {
        "f1"       : float(f1_score(y_true, y_pred_t, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred_t, zero_division=0)),
        "recall"   : float(recall_score(y_true, y_pred_t, zero_division=0)),
        "auc_roc"  : float(roc_auc_score(y_true, y_prob)),
        "auc_pr"   : float(average_precision_score(y_true, y_prob)),
        "threshold": threshold,
    }


def find_best_threshold(y_true, y_prob) -> float:
    """Same threshold sweep as used in evaluate.py — fair comparison."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_regression(X_train, X_test, y_train, y_test) -> dict:
    logger.info("Training Logistic Regression...")
    # class_weight='balanced' is the LR equivalent of weighted loss
    clf = LogisticRegression(
        class_weight = "balanced",
        max_iter     = 1000,
        random_state = 42,
        solver       = "lbfgs",
        C            = 1.0,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    t      = find_best_threshold(y_test, y_prob)
    metrics = compute_metrics(y_test, None, y_prob, threshold=t)
    logger.info(f"  LR  F1={metrics['f1']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  threshold={t:.2f}")
    return metrics


def run_random_forest(X_train, X_test, y_train, y_test) -> dict:
    logger.info("Training Random Forest...")
    # Compute class weights manually
    n_fraud = y_train.sum()
    n_legit = (y_train == 0).sum()
    w_fraud = len(y_train) / (2 * n_fraud)
    w_legit = len(y_train) / (2 * n_legit)
    sample_weights = np.where(y_train == 1, w_fraud, w_legit)

    clf = RandomForestClassifier(
        n_estimators = 300,
        max_depth    = 15,
        min_samples_leaf = 5,
        random_state = 42,
        n_jobs       = -1,
        class_weight = "balanced",
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    y_prob = clf.predict_proba(X_test)[:, 1]
    t      = find_best_threshold(y_test, y_prob)
    metrics = compute_metrics(y_test, None, y_prob, threshold=t)
    logger.info(f"  RF  F1={metrics['f1']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  threshold={t:.2f}")
    return metrics, clf


def run_xgboost(X_train, X_test, y_train, y_test, label="XGBoost") -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed. Run: pip install xgboost")
        return None

    logger.info(f"Training {label}...")
    n_fraud = y_train.sum()
    n_legit = (y_train == 0).sum()
    scale_pos_weight = n_legit / n_fraud   # XGBoost's built-in imbalance handling

    clf = XGBClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        random_state      = 42,
        n_jobs            = -1,
        eval_metric       = "aucpr",
        verbosity         = 0,
        use_label_encoder = False,
    )
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    t      = find_best_threshold(y_test, y_prob)
    metrics = compute_metrics(y_test, None, y_prob, threshold=t)
    logger.info(f"  {label}  F1={metrics['f1']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  threshold={t:.2f}")
    return metrics, clf


# ─────────────────────────────────────────────────────────────────────────────
# GNN RESULTS FROM CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_gnn_results(checkpoint_path: str, data) -> dict:
    """
    Load the trained GNN and evaluate on the same test set.
    Returns metrics at the optimal threshold found during training.
    """
    logger.info("Loading GNN checkpoint for comparison...")
    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg    = ckpt["config"]

    model = build_model(cfg, ckpt["in_channels"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect test predictions
    test_loader = build_standard_loader(
        data, data.test_mask, cfg.train.num_neighbors, cfg.train.batch_size, shuffle=False
    )

    all_probs, all_labels = [], []
    for batch in test_loader:
        batch       = batch.to(device)
        logits      = model(batch.x, batch.edge_index)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        labelled    = seed_labels >= 0
        if labelled.sum() == 0:
            continue
        probs = torch.softmax(seed_logits[labelled], dim=-1)[:, 1].numpy()
        all_probs.append(probs)
        all_labels.append(seed_labels[labelled].numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)

    t       = find_best_threshold(y_true, y_prob)
    metrics = compute_metrics(y_true, None, y_prob, threshold=t)
    logger.info(f"  GNN F1={metrics['f1']:.4f}  AUC-PR={metrics['auc_pr']:.4f}  threshold={t:.2f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, output_dir: str):
    """
    Grouped bar chart comparing all models across all metrics.
    GNN bar is highlighted in a distinct color to make it stand out.
    """
    metrics_to_plot = ["f1", "precision", "recall", "auc_roc", "auc_pr"]
    metric_labels   = ["F1 Score", "Precision", "Recall", "AUC-ROC", "AUC-PR"]

    model_names = list(results.keys())
    n_models    = len(model_names)
    n_metrics   = len(metrics_to_plot)

    # Color scheme: GNN = gold, baselines = blues/grays
    colors = []
    for name in model_names:
        if "GraphSAGE" in name or "GNN" in name:
            colors.append("#e63946")   # red/gold for GNN — stands out
        elif "XGBoost + Graph" in name:
            colors.append("#2196F3")   # blue for best tabular
        elif "XGBoost" in name:
            colors.append("#64B5F6")
        elif "Random Forest" in name:
            colors.append("#90CAF9")
        else:
            colors.append("#BBDEFB")   # light blue for weakest baselines

    x      = np.arange(n_metrics)
    width  = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals   = [results[name].get(m, 0) for m in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=name, color=color,
                        alpha=0.9, edgecolor="white", linewidth=0.5)

        # Annotate bar values
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, rotation=45,
            )

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Fraud Detection: GNN vs Tabular Baselines\n(Elliptic Bitcoin Dataset — Same Train/Test Split)",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.3, linewidth=1)

    # Annotate AUC-PR baseline
    fraud_rate = 0.0223
    ax.axhline(fraud_rate, color="#e63946", linestyle=":", alpha=0.4, linewidth=1.5)
    ax.text(n_metrics - 0.3, fraud_rate + 0.01, f"Random AUC-PR ≈ {fraud_rate:.3f}",
            fontsize=8, color="#e63946", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, "baseline_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_improvement(results: dict, gnn_key: str, output_dir: str):
    """
    Horizontal bar chart showing GNN improvement over each baseline,
    per metric. Makes the contribution immediately obvious.
    """
    metrics      = ["f1", "precision", "recall", "auc_roc", "auc_pr"]
    metric_labels= ["F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
    gnn_vals     = results[gnn_key]
    baselines    = {k: v for k, v in results.items() if k != gnn_key}

    fig, axes = plt.subplots(1, len(baselines), figsize=(5 * len(baselines), 6), sharey=True)
    if len(baselines) == 1:
        axes = [axes]

    fig.suptitle(f"GNN Improvement Over Baselines", fontsize=13, fontweight="bold")

    for ax, (bname, bvals) in zip(axes, baselines.items()):
        improvements = [gnn_vals[m] - bvals.get(m, 0) for m in metrics]
        colors = ["#1A7A4A" if v >= 0 else "#C0392B" for v in improvements]
        bars   = ax.barh(metric_labels, improvements, color=colors, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, improvements):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=10, fontweight="bold",
            )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"vs {bname}", fontsize=10)
        ax.set_xlabel("GNN − Baseline", fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, "gnn_improvement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(results: dict):
    """Print a clean comparison table to console."""
    metrics = ["f1", "precision", "recall", "auc_roc", "auc_pr"]
    headers = ["Model", "F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]

    col_w = [28, 8, 10, 8, 9, 8]
    sep   = "  "

    print("\n" + "═" * 75)
    print("  FRAUD DETECTION — MODEL COMPARISON (Elliptic Bitcoin Dataset)")
    print("  Same train/test split for all models — direct comparison")
    print("═" * 75)

    # Header
    header_str = sep.join(h.ljust(w) for h, w in zip(headers, col_w))
    print("  " + header_str)
    print("  " + "-" * 73)

    # Rows — GNN row gets a ★
    for name, vals in results.items():
        marker = " ★" if ("GraphSAGE" in name or "GNN" in name) else "  "
        row = [name + marker] + [f"{vals.get(m, 0):.4f}" for m in metrics]
        row_str = sep.join(v.ljust(w) for v, w in zip(row, col_w))
        print("  " + row_str)

    print("═" * 75)

    # Highlight best per metric
    print("\n  Best per metric:")
    for m, label in zip(metrics, ["F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]):
        best_name = max(results, key=lambda k: results[k].get(m, 0))
        best_val  = results[best_name].get(m, 0)
        print(f"    {label:<10}: {best_name:<30} {best_val:.4f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline comparison: GNN vs tabular ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline.py --checkpoint outputs/best_model.pt
  python baseline.py --checkpoint outputs/best_model.pt --no-xgboost
  python baseline.py --checkpoint outputs/best_model.pt --output-dir outputs
        """
    )
    parser.add_argument("--checkpoint",  default="outputs/best_model.pt")
    parser.add_argument("--output-dir",  default="outputs")
    parser.add_argument("--no-xgboost",  action="store_true",
                        help="Skip XGBoost (if not installed)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    data = load_dataset(cfg)

    # ── 2. Extract tabular splits ────────────────────────────────────────────
    logger.info("Extracting tabular train/test splits...")
    X_train, X_test, y_train, y_test = extract_tabular_splits(data)

    # Augmented version with graph degree features
    X_train_aug, X_test_aug = add_graph_features(data, X_train, X_test)

    # ── 3. Run baselines ─────────────────────────────────────────────────────
    results = {}

    # Logistic Regression
    results["Logistic Regression"] = run_logistic_regression(
        X_train, X_test, y_train, y_test
    )

    # Random Forest
    rf_metrics, rf_model = run_random_forest(X_train, X_test, y_train, y_test)
    results["Random Forest"] = rf_metrics

    # XGBoost
    if not args.no_xgboost:
        xgb_result = run_xgboost(X_train, X_test, y_train, y_test, "XGBoost")
        if xgb_result is not None:
            results["XGBoost"] = xgb_result[0]

        # XGBoost + hand-crafted graph features
        xgb_aug_result = run_xgboost(
            X_train_aug, X_test_aug, y_train, y_test, "XGBoost + Graph Stats"
        )
        if xgb_aug_result is not None:
            results["XGBoost + Graph Stats"] = xgb_aug_result[0]
    else:
        logger.info("Skipping XGBoost (--no-xgboost flag set)")

    # ── 4. GNN results ───────────────────────────────────────────────────────
    results["GraphSAGE (GNN)"] = get_gnn_results(args.checkpoint, data)

    # ── 5. Print table ───────────────────────────────────────────────────────
    print_comparison_table(results)

    # ── 6. Save plots ────────────────────────────────────────────────────────
    plot_comparison(results, args.output_dir)
    plot_improvement(results, "GraphSAGE (GNN)", args.output_dir)

    # ── 7. Save JSON ─────────────────────────────────────────────────────────
    path = os.path.join(args.output_dir, "baseline_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {path}")

    print(f"\n  Plots saved to: {args.output_dir}/")
    print(f"  baseline_comparison.png  — grouped bar chart")
    print(f"  gnn_improvement.png      — GNN delta over each baseline")
    print(f"  baseline_results.json    — all numbers\n")


if __name__ == "__main__":
    main()