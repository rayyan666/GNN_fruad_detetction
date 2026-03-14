# evaluate.py
"""
Step 6: Full evaluation suite.

Metrics computed:
  - Precision / Recall / F1 (fraud class)
  - AUC-ROC
  - AUC-PR (Precision-Recall curve) — more informative than ROC under imbalance
  - Confusion matrix
  - Threshold sensitivity analysis

Threshold selection:
  Default 0.5 is often wrong with class imbalance. We sweep thresholds
  and find the optimal one for F1 on the validation set, then report
  both default and optimised results on the test set.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

import wandb

sys.path.insert(0, os.path.dirname(__file__))
from config import CFG
from models.gnn import build_model, get_device
from utils.data_loader import load_dataset
from train import build_standard_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")


# ─────────────────────────────────────────────
# COLLECT ALL PREDICTIONS
# ─────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Run inference and collect (logits, labels) for all labelled nodes in loader.
    Returns:
        probs:  np.array [N]  fraud probabilities P(fraud)
        labels: np.array [N]  ground-truth labels
    """
    model.eval()
    all_probs  = []
    all_labels = []

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        labelled    = seed_labels >= 0

        if labelled.sum() == 0:
            continue

        probs = torch.softmax(seed_logits[labelled], dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(seed_labels[labelled].cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─────────────────────────────────────────────
# THRESHOLD OPTIMISATION
# ─────────────────────────────────────────────

def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Sweep classification thresholds from 0.1 to 0.9 and return the
    value that maximises F1 score on the given split.

    Important for fraud detection: a lower threshold increases recall
    (catches more fraud) at the cost of more false alarms.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_thresh = 0.0, 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            best_thresh = t

    logger.info(f"Optimal threshold: {best_thresh:.2f} (val F1={best_f1:.4f})")
    return best_thresh


# ─────────────────────────────────────────────
# FULL EVALUATION REPORT
# ─────────────────────────────────────────────

def evaluate_model(model, data, cfg, output_dir: str = "outputs"):
    """
    Full evaluation on the test set with all metrics and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(cfg)
    model = model.to(device)

    # Collect predictions on validation set (for threshold tuning)
    val_loader = build_standard_loader(
        data, data.val_mask, cfg.train.num_neighbors, cfg.train.batch_size, shuffle=False
    )
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    opt_threshold = find_optimal_threshold(val_probs, val_labels)

    # Collect predictions on test set
    test_loader = build_standard_loader(
        data, data.test_mask, cfg.train.num_neighbors, cfg.train.batch_size, shuffle=False
    )
    test_probs, test_labels = collect_predictions(model, test_loader, device)

    results = {}

    # ── Default threshold (0.5) ────────────────────────────────────────────────
    preds_default = (test_probs >= 0.5).astype(int)
    results["threshold_0.5"] = _compute_metrics(test_labels, preds_default, test_probs)

    # ── Optimised threshold ────────────────────────────────────────────────────
    preds_opt = (test_probs >= opt_threshold).astype(int)
    results[f"threshold_{opt_threshold:.2f}"] = _compute_metrics(test_labels, preds_opt, test_probs)
    results["optimal_threshold"] = float(opt_threshold)

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  FRAUD DETECTION — TEST SET EVALUATION")
    print("═" * 60)

    print(f"\n  Threshold = 0.50 (default):")
    _print_metrics(results["threshold_0.5"])

    print(f"\n  Threshold = {opt_threshold:.2f} (optimised for F1):")
    _print_metrics(results[f"threshold_{opt_threshold:.2f}"])

    print("\n  Full Classification Report (optimised threshold):")
    print(classification_report(test_labels, preds_opt, target_names=["Legit", "Fraud"]))
    print("═" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_pr_curves(test_labels, test_probs, output_dir)
    plot_confusion_matrix(test_labels, preds_opt, output_dir)
    plot_threshold_sweep(test_labels, test_probs, opt_threshold, output_dir)
    plot_training_history(output_dir)

    # Save results JSON
    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── W&B: log final test metrics + plots ───────────────────────────────────
    # The optimal threshold key is dynamic e.g. "threshold_0.27"
    opt_key = f"threshold_{opt_threshold:.2f}"
    opt_metrics = results[opt_key]

    wandb.log({
        # Test metrics at optimal threshold
        "test_f1":        opt_metrics["f1"],
        "test_precision": opt_metrics["precision"],
        "test_recall":    opt_metrics["recall"],
        "test_auc_roc":   opt_metrics["auc_roc"],
        "test_auc_pr":    opt_metrics["auc_pr"],
        "optimal_threshold": opt_threshold,

        # Test metrics at default 0.5 threshold (for comparison)
        "test_f1_at_0.5":  results["threshold_0.5"]["f1"],

        # All plots as images
        "roc_pr_curve":     wandb.Image(os.path.join(output_dir, "roc_pr_curves.png")),
        "confusion_matrix": wandb.Image(os.path.join(output_dir, "confusion_matrix.png")),
        "threshold_sweep":  wandb.Image(os.path.join(output_dir, "threshold_sweep.png")),
        "training_history": wandb.Image(os.path.join(output_dir, "training_history.png")),
    })

    return results


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────

def _compute_metrics(labels, preds, probs) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        "auc_pr":    float(average_precision_score(labels, probs)),
    }


def _print_metrics(metrics: dict):
    for k, v in metrics.items():
        print(f"    {k:<15}: {v:.4f}")


# ─────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────

def plot_roc_pr_curves(labels, probs, output_dir):
    """Side-by-side ROC and Precision-Recall curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Performance Curves", fontsize=14, fontweight="bold")

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#e63946", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    baseline = labels.mean()

    ax2.plot(recall, precision, color="#457b9d", lw=2, label=f"PR (AP = {ap:.4f})")
    ax2.axhline(baseline, color="gray", linestyle="--", alpha=0.6, label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: roc_pr_curves.png")


def plot_confusion_matrix(labels, preds, output_dir):
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in [
        (ax1, cm,      "d",    "Counts"),
        (ax2, cm_norm, ".2%",  "Normalised (row %)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=["Pred: Legit", "Pred: Fraud"],
                    yticklabels=["True: Legit", "True: Fraud"])
        ax.set_title(title)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: confusion_matrix.png")


def plot_threshold_sweep(labels, probs, opt_threshold, output_dir):
    """
    Show how Precision, Recall, and F1 change across thresholds.
    Helps practitioners choose a threshold based on business constraints
    (e.g., prioritise recall to minimise missed fraud, or precision to
    minimise false alarms to legitimate customers).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        precisions.append(precision_score_safe(labels, preds))
        recalls.append(recall_score_safe(labels, preds))
        f1s.append(f1_score(labels, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#2196F3", lw=2)
    ax.plot(thresholds, recalls,    label="Recall",    color="#FF5722", lw=2)
    ax.plot(thresholds, f1s,        label="F1 Score",  color="#4CAF50", lw=2)
    ax.axvline(opt_threshold, color="purple", linestyle="--", alpha=0.8,
               label=f"Optimal threshold = {opt_threshold:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.6, label="Default threshold = 0.50")

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Classification Threshold", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "threshold_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: threshold_sweep.png")


def plot_training_history(output_dir):
    """Plot training loss and validation metrics over epochs."""
    history_path = os.path.join(output_dir, "history.json")
    if not os.path.exists(history_path):
        logger.warning("No training history found; skipping history plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    f1s    = [h["f1"] for h in history]
    aucs   = [h["auc_roc"] for h in history]
    precs  = [h["precision"] for h in history]
    recs   = [h["recall"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    axes[0, 0].plot(epochs, losses, color="#e63946", lw=2)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, f1s, color="#2196F3", lw=2)
    axes[0, 1].set_title("Val F1 Score")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, precs, label="Precision", lw=2)
    axes[1, 0].plot(epochs, recs,  label="Recall",    lw=2)
    axes[1, 0].set_title("Val Precision / Recall")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, aucs, color="#4CAF50", lw=2)
    axes[1, 1].set_title("Val AUC-ROC")
    axes[1, 1].set_xlabel("Epoch"); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: training_history.png")


# ─────────────────────────────────────────────
# SAFE METRIC WRAPPERS
# ─────────────────────────────────────────────

from sklearn.metrics import precision_score as _prec, recall_score as _rec

def precision_score_safe(y_true, y_pred):
    return _prec(y_true, y_pred, zero_division=0)

def recall_score_safe(y_true, y_pred):
    return _rec(y_true, y_pred, zero_division=0)


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained GNN")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg        = checkpoint.get("config", CFG)
    data       = load_dataset(cfg)
    device     = get_device(cfg)

    model = build_model(cfg, checkpoint["in_channels"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluate_model(model, data, cfg, args.output_dir)   