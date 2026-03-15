# temporal_split.py
"""
Temporal Split Validation — the correct evaluation for fraud detection.

WHY THIS MATTERS:
  The random split used during training mixes transactions from ALL time steps
  in both train and test sets. This means the model sees future transaction
  patterns during training — a form of data leakage that inflates results.

  In production, you always train on PAST data and predict on FUTURE data.
  A model trained on time steps 1-34 has never seen transactions from 41-49.
  This is the only honest evaluation of real-world performance.

SPLIT USED (mirrors original Elliptic paper):
  Train : time steps  1 – 34  (~70% of labelled data)
  Val   : time steps 35 – 40  (~15%)
  Test  : time steps 41 – 49  (~15%)

  NOTE: This is a HARDER test than random split. Fraud patterns evolve
  over time, so the model must generalise to new fraud signatures it
  has never seen — not just recognise patterns from the same time period.

Usage:
  python temporal_split.py --train --compare
  python temporal_split.py --eval-only --checkpoint outputs/best_model.pt
"""

import os
import sys
import json
import logging
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, classification_report,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import CFG
from models.gnn import build_model, get_device
from utils.data_loader import load_dataset, compute_class_weights
from train import (
    build_standard_loader, build_oversampled_loader,
    train_one_epoch, evaluate, EarlyStopping, set_seed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRAIN_END = 34
VAL_END   = 40


# ─────────────────────────────────────────────────────────────────────────────
# W&B SAFE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

def wandb_log(metrics: dict):
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


def wandb_init(cfg):
    try:
        import wandb
        wandb.init(
            project = "fraud-gnn",
            name    = "graphsage_weighted_loss_temporal_ep150",
            config  = {
                "split"        : "temporal",
                "train_steps"  : f"1-{TRAIN_END}",
                "val_steps"    : f"{TRAIN_END+1}-{VAL_END}",
                "test_steps"   : f"{VAL_END+1}-49",
                "model"        : cfg.model.architecture,
                "strategy"     : cfg.data.imbalance_strategy,
                "epochs"       : cfg.train.epochs,
                "hidden"       : cfg.model.hidden_channels,
                "num_layers"   : cfg.model.num_layers,
                "lr"           : cfg.train.lr,
                "weight_decay" : cfg.train.weight_decay,
                "batch_size"   : cfg.train.batch_size,
                "patience"     : cfg.train.patience,
            }
        )
        logger.info("W&B run started: graphsage_weighted_loss_temporal_ep150")
    except Exception as e:
        logger.warning(f"W&B init failed: {e}")


def wandb_finish():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished.")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL MASKS
# ─────────────────────────────────────────────────────────────────────────────

def build_temporal_masks(data):
    time_steps = data.time_steps
    y          = data.y.numpy()
    N          = data.num_nodes
    labelled   = y >= 0

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)

    for i in range(N):
        if not labelled[i]:
            continue
        t = int(time_steps[i])
        if t <= TRAIN_END:
            train_mask[i] = True
        elif t <= VAL_END:
            val_mask[i]   = True
        else:
            test_mask[i]  = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    y_train = y[train_mask.numpy()]
    y_val   = y[val_mask.numpy()]
    y_test  = y[test_mask.numpy()]

    logger.info("  Temporal split applied:")
    logger.info(f"    Train  (steps  1–{TRAIN_END}): {train_mask.sum():>5} nodes  "
                f"| fraud: {(y_train==1).sum():>4}  ({100*(y_train==1).mean():.1f}%)")
    logger.info(f"    Val    (steps {TRAIN_END+1}–{VAL_END}): {val_mask.sum():>5} nodes  "
                f"| fraud: {(y_val==1).sum():>4}  ({100*(y_val==1).mean():.1f}%)")
    logger.info(f"    Test   (steps {VAL_END+1}–49): {test_mask.sum():>5} nodes  "
                f"| fraud: {(y_test==1).sum():>4}  ({100*(y_test==1).mean():.1f}%)")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ON TEMPORAL SPLIT (directly — does NOT reload data internally)
# ─────────────────────────────────────────────────────────────────────────────

def train_temporal(cfg, data, output_dir: str):
    """
    Train GNN using the already-masked data object.
    Unlike train.train(), this does NOT reload data internally,
    so the temporal masks are preserved throughout training.
    """
    set_seed(cfg.seed)
    os.makedirs(output_dir, exist_ok=True)

    device      = get_device(cfg)
    in_channels = data.x.shape[1]
    model       = build_model(cfg, in_channels).to(device)

    # Weighted loss
    class_weights = compute_class_weights(data).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Class weights → legit: {class_weights[0]:.2f}, fraud: {class_weights[1]:.2f}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    train_loader = build_standard_loader(
        data, data.train_mask, cfg.train.num_neighbors, cfg.train.batch_size
    )
    val_loader = build_standard_loader(
        data, data.val_mask, cfg.train.num_neighbors, cfg.train.batch_size, shuffle=False
    )

    early_stop = EarlyStopping(patience=cfg.train.patience)
    history    = []

    logger.info(f"\nStarting temporal training for {cfg.train.epochs} epochs...")
    logger.info("─" * 65)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["f1"])

        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        # W&B logging
        wandb_log({
            "epoch":         epoch,
            "train_loss":    train_loss,
            "val_f1":        val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall":    val_metrics["recall"],
            "val_auc_roc":   val_metrics["auc_roc"],
        })

        # Console logging
        if epoch % cfg.log_every == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d} | Loss: {train_loss:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | "
                f"Rec: {val_metrics['recall']:.4f} | "
                f"AUC: {val_metrics['auc_roc']:.4f}"
            )

        if early_stop.step(val_metrics["f1"], model):
            logger.info(f"\nEarly stopping at epoch {epoch} "
                        f"(best val F1={early_stop.best_score:.4f})")
            break

    early_stop.restore_best(model)

    checkpoint_path = os.path.join(output_dir, "best_model_temporal.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           cfg,
        "history":          history,
        "in_channels":      in_channels,
        "best_val_f1":      early_stop.best_score,
        "split":            "temporal",
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    with open(os.path.join(output_dir, "history_temporal.json"), "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete.  Best val F1: {early_stop.best_score:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_gnn(model, data, cfg, device, split="test"):
    model.eval()
    mask   = getattr(data, f"{split}_mask")
    loader = build_standard_loader(
        data, mask, cfg.train.num_neighbors, cfg.train.batch_size, shuffle=False
    )
    all_probs, all_labels = [], []
    for batch in loader:
        batch       = batch.to(device)
        logits      = model(batch.x, batch.edge_index)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]
        labelled    = seed_labels >= 0
        if labelled.sum() == 0:
            continue
        probs = F.softmax(seed_logits[labelled], dim=-1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(seed_labels[labelled].cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(y_true, y_prob):
    t      = find_best_threshold(y_true, y_prob)
    y_pred = (y_prob >= t).astype(int)
    return {
        "f1"        : float(f1_score(y_true, y_pred, zero_division=0)),
        "precision" : float(precision_score(y_true, y_pred, zero_division=0)),
        "recall"    : float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc"   : float(roc_auc_score(y_true, y_prob)),
        "auc_pr"    : float(average_precision_score(y_true, y_prob)),
        "threshold" : float(t),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TABULAR BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def run_tabular_baselines_temporal(data):
    X = data.x.numpy()
    y = data.y.numpy()

    X_train = X[data.train_mask.numpy()]
    y_train = y[data.train_mask.numpy()]
    X_test  = X[data.test_mask.numpy()]
    y_test  = y[data.test_mask.numpy()]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    results = {}

    logger.info("  Logistic Regression [temporal]...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=42, solver="lbfgs")
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = compute_metrics(y_test, lr.predict_proba(X_test)[:, 1])

    logger.info("  Random Forest [temporal]...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=15,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = compute_metrics(y_test, rf.predict_proba(X_test)[:, 1])

    try:
        from xgboost import XGBClassifier
        logger.info("  XGBoost [temporal]...")
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        xgb = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=scale_pos, random_state=42,
                            n_jobs=-1, verbosity=0, eval_metric="aucpr")
        xgb.fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb.predict_proba(X_test)[:, 1])
    except ImportError:
        logger.warning("XGBoost not installed, skipping.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_rate_over_time(data, output_dir: str):
    time_steps  = data.time_steps
    y           = data.y.numpy()
    fraud_rates = []
    steps       = sorted(set(time_steps.tolist()))

    for t in steps:
        mask = (time_steps == t) & (y >= 0)
        fraud_rates.append((y[mask] == 1).mean() if mask.sum() > 0 else 0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(steps, fraud_rates, color="#e63946", lw=2, marker="o", ms=4)
    ax.fill_between(steps, fraud_rates, alpha=0.15, color="#e63946")
    ax.axvspan(min(steps), TRAIN_END+0.5, alpha=0.08, color="#2196F3",
               label=f"Train (steps 1–{TRAIN_END})")
    ax.axvspan(TRAIN_END+0.5, VAL_END+0.5, alpha=0.08, color="#FF9800",
               label=f"Val (steps {TRAIN_END+1}–{VAL_END})")
    ax.axvspan(VAL_END+0.5, max(steps), alpha=0.08, color="#4CAF50",
               label=f"Test (steps {VAL_END+1}–49)")
    ax.axvline(TRAIN_END+0.5, color="#2196F3", linestyle="--", alpha=0.6, lw=1.5)
    ax.axvline(VAL_END+0.5,   color="#FF9800", linestyle="--", alpha=0.6, lw=1.5)
    ax.set_xlabel("Time Step (~2 weeks each)", fontsize=12)
    ax.set_ylabel("Fraud Rate", fontsize=12)
    ax.set_title("Fraud Rate Over Time — Elliptic Bitcoin Dataset\n"
                 "Temporal boundaries for train/val/test split shown",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(steps), max(steps))
    plt.tight_layout()
    path = os.path.join(output_dir, "fraud_rate_temporal_split.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_temporal_vs_random(temporal: dict, random: dict, output_dir: str):
    metrics       = ["f1", "precision", "recall", "auc_roc", "auc_pr"]
    metric_labels = ["F1", "Precision", "Recall", "AUC-ROC", "AUC-PR"]
    models        = [m for m in temporal if m in random]
    if not models:
        logger.warning("No overlapping models for comparison plot.")
        return

    n_models = len(models)
    x        = np.arange(len(metrics))
    width    = 0.35
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6), sharey=True)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        "Temporal Split vs Random Split — Performance Comparison\n"
        "Temporal = train steps 1–34, test steps 41–49  |  Random = shuffled 70/15/15",
        fontsize=11, fontweight="bold"
    )

    for ax, mname in zip(axes, models):
        rv = [random[mname].get(m, 0) for m in metrics]
        tv = [temporal[mname].get(m, 0) for m in metrics]

        b1 = ax.bar(x-width/2, rv, width, label="Random split",
                    color="#64B5F6", alpha=0.9, edgecolor="white")
        b2 = ax.bar(x+width/2, tv, width, label="Temporal split",
                    color="#e63946", alpha=0.9, edgecolor="white")

        for bar, val in zip(b1, rv):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(b2, tv):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "temporal_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLES
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: dict, title: str):
    metrics = ["f1", "precision", "recall", "auc_roc", "auc_pr"]
    print(f"\n{'═'*70}")
    print(f"  {title}")
    print(f"{'═'*70}")
    print(f"  {'Model':<30}  {'F1':>7}  {'Prec':>7}  {'Rec':>7}  {'ROC':>7}  {'PR':>7}")
    print(f"  {'-'*68}")
    for name, vals in results.items():
        marker = " ★" if "GNN" in name or "GraphSAGE" in name else "  "
        row = f"  {name+marker:<30}  " + "  ".join(f"{vals.get(m,0):>7.4f}" for m in metrics)
        print(row)
    print(f"{'═'*70}")


def print_delta_table(temporal: dict, random: dict):
    models = [m for m in temporal if m in random]
    print(f"\n{'═'*60}")
    print(f"  PERFORMANCE DROP: Temporal − Random Split")
    print(f"  (negative = harder test, as expected)")
    print(f"{'═'*60}")
    print(f"  {'Model':<30}  {'ΔF1':>8}  {'ΔAUC-PR':>8}")
    print(f"  {'-'*58}")
    for m in models:
        df1  = temporal[m].get("f1",     0) - random[m].get("f1",     0)
        dapr = temporal[m].get("auc_pr", 0) - random[m].get("auc_pr", 0)
        marker = " ★" if "GNN" in m or "GraphSAGE" in m else "  "
        print(f"  {m+marker:<30}  {df1:>+8.4f}  {dapr:>+8.4f}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Temporal split validation")
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument("--compare",        action="store_true")
    parser.add_argument("--checkpoint",     default="outputs/best_model.pt")
    parser.add_argument("--random-results", default="outputs/baseline_results.json")
    parser.add_argument("--output-dir",     default="outputs/temporal")
    parser.add_argument("--no-baselines",   action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.train and not args.eval_only:
        parser.error("Specify --train or --eval-only")

    # ── 1. Load data + apply temporal masks ──────────────────────────────────
    logger.info("Loading dataset...")
    data = load_dataset(CFG)
    data = build_temporal_masks(data)

    # ── 2. Fraud rate plot ───────────────────────────────────────────────────
    plot_fraud_rate_over_time(data, args.output_dir)

    temporal_results = {}

    # ── 3. Train GNN on temporal split ───────────────────────────────────────
    if args.train:
        wandb_init(CFG)

        logger.info("\nTraining GNN on temporal split (steps 1–34)...")
        model, history = train_temporal(CFG, data, args.output_dir)

        # Log test metrics to W&B after training
        logger.info("Evaluating GNN on temporal test set (steps 41–49)...")
        y_prob, y_true = evaluate_gnn(model, data, CFG, device, split="test")
        temporal_results["GraphSAGE (GNN)"] = compute_metrics(y_true, y_prob)

        gnn_m = temporal_results["GraphSAGE (GNN)"]
        wandb_log({
            "test_f1"        : gnn_m["f1"],
            "test_precision" : gnn_m["precision"],
            "test_recall"    : gnn_m["recall"],
            "test_auc_roc"   : gnn_m["auc_roc"],
            "test_auc_pr"    : gnn_m["auc_pr"],
            "optimal_threshold": gnn_m["threshold"],
        })

        logger.info("\n  Full Classification Report (temporal test set):")
        t      = gnn_m["threshold"]
        y_pred = (y_prob >= t).astype(int)
        print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))

        wandb_finish()

    # ── 4. Eval-only mode ────────────────────────────────────────────────────
    elif args.eval_only:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg   = ckpt["config"]
        model = build_model(cfg, ckpt["in_channels"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        data = load_dataset(cfg)
        data = build_temporal_masks(data)

        logger.info("Evaluating on temporal test set (steps 41–49)...")
        y_prob, y_true = evaluate_gnn(model, data, cfg, device, split="test")
        temporal_results["GraphSAGE (GNN)"] = compute_metrics(y_true, y_prob)

    # ── 5. Tabular baselines ─────────────────────────────────────────────────
    if not args.no_baselines:
        logger.info("\nRunning tabular baselines on temporal split...")
        baseline_results = run_tabular_baselines_temporal(data)
        temporal_results.update(baseline_results)

    # ── 6. Print results ─────────────────────────────────────────────────────
    print_table(temporal_results,
                "TEMPORAL SPLIT (train steps 1–34, test steps 41–49)")

    # ── 7. Compare with random split ─────────────────────────────────────────
    if args.compare and os.path.exists(args.random_results):
        with open(args.random_results) as f:
            random_results = json.load(f)
        print_table(random_results, "RANDOM SPLIT (shuffled 70/15/15)")
        print_delta_table(temporal_results, random_results)
        plot_temporal_vs_random(temporal_results, random_results, args.output_dir)
    elif args.compare:
        logger.warning(f"Random split results not found at {args.random_results}. "
                       f"Run baseline.py first.")

    # ── 8. Save ──────────────────────────────────────────────────────────────
    path = os.path.join(args.output_dir, "temporal_results.json")
    with open(path, "w") as f:
        json.dump(temporal_results, f, indent=2)
    logger.info(f"\nResults saved: {path}")

    print(f"\n  Output directory: {args.output_dir}/")
    print(f"  temporal_results.json")
    print(f"  fraud_rate_temporal_split.png")
    if args.compare:
        print(f"  temporal_comparison.png")


if __name__ == "__main__":
    main()