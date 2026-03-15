# train.py
"""
Step 5: Training pipeline with class imbalance handling.

Strategies for severe imbalance (~2% fraud rate):
  1. Weighted cross-entropy loss  — penalizes misclassification of rare fraud class more
  2. Node-level oversampling      — synthetically duplicates fraud nodes in mini-batches
  3. Combined                     — both strategies together

Uses PyTorch Geometric's NeighborLoader for scalable mini-batch training:
  - Samples k-hop neighborhoods around seed nodes
  - Allows training on graphs too large to fit in GPU memory
  - Fraud-biased sampling ensures each batch has a minimum number of fraud nodes

Training features:
  - Early stopping with patience
  - Learning rate scheduler (ReduceLROnPlateau)
  - Checkpoint saving (best val F1)
  - Epoch-level logging with metrics
  - W&B logging (safe — silently skipped if wandb.init() not called)
"""

import os
import sys
import json
import logging
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from config import CFG, Config
from models.gnn import build_model, get_device
from utils.data_loader import load_dataset, compute_class_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# W&B SAFE LOGGER
# ─────────────────────────────────────────────

def wandb_log(metrics: dict):
    """Log to W&B only if a run is active. Never crashes."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics)
    except Exception:
        pass


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# OVERSAMPLING (fraud-biased mini-batches)
# ─────────────────────────────────────────────

def build_oversampled_loader(
    data: Data,
    mask: torch.Tensor,
    num_neighbors: list,
    batch_size: int,
    oversample_ratio: float = 0.3,
) -> NeighborLoader:
    """
    Build a NeighborLoader where fraud nodes are oversampled as seed nodes.
    """
    idx    = mask.nonzero(as_tuple=True)[0]
    labels = data.y[idx].numpy()

    fraud_idx = idx[(labels == 1)].numpy()
    legit_idx = idx[(labels == 0)].numpy()

    n_fraud_target = int(batch_size * oversample_ratio)
    n_legit_target = batch_size - n_fraud_target

    fraud_seeds = np.random.choice(fraud_idx, size=n_fraud_target * 10, replace=True)
    legit_seeds = np.random.choice(legit_idx, size=n_legit_target * 10, replace=True)
    combined    = np.concatenate([fraud_seeds, legit_seeds])
    np.random.shuffle(combined)

    input_nodes = torch.tensor(combined, dtype=torch.long)

    return NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size,
        input_nodes   = input_nodes,
        shuffle       = True,
    )


def build_standard_loader(
    data: Data,
    mask: torch.Tensor,
    num_neighbors: list,
    batch_size: int,
    shuffle: bool = True,
) -> NeighborLoader:
    """Standard NeighborLoader without oversampling (for val/test)."""
    return NeighborLoader(
        data,
        num_neighbors = num_neighbors,
        batch_size    = batch_size,
        input_nodes   = mask,
        shuffle       = shuffle,
    )


# ─────────────────────────────────────────────
# SINGLE EPOCH TRAIN / EVAL
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: NeighborLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss  = 0.0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits      = model(batch.x, batch.edge_index)
        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]

        labelled_mask = seed_labels >= 0
        if labelled_mask.sum() == 0:
            continue

        loss = criterion(seed_logits[labelled_mask], seed_labels[labelled_mask])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss  += loss.item() * labelled_mask.sum().item()
        total_nodes += labelled_mask.sum().item()

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: NeighborLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        seed_logits = logits[:batch.batch_size]
        seed_labels = batch.y[:batch.batch_size]

        labelled = seed_labels >= 0
        if labelled.sum() == 0:
            continue

        all_logits.append(seed_logits[labelled].cpu())
        all_labels.append(seed_labels[labelled].cpu())

    if not all_logits:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "auc_roc": 0.0}

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs  = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds  = (probs >= 0.5).astype(int)

    return {
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "auc_roc":   roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
    }


# ─────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, score: float, model: nn.Module) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            logger.info(f"  Restored best model (val F1 = {self.best_score:.4f})")


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(cfg: Config = CFG):
    set_seed(cfg.seed)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    data   = load_dataset(cfg)
    device = get_device(cfg)
    logger.info(f"Device: {device}")

    # ── 2. Build model ─────────────────────────────────────────────────────────
    in_channels = data.x.shape[1]
    model       = build_model(cfg, in_channels).to(device)

    # ── 3. Loss function ───────────────────────────────────────────────────────
    strategy = cfg.data.imbalance_strategy

    if strategy in ("weighted_loss", "both"):
        class_weights = compute_class_weights(data).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Class weights → legit: {class_weights[0]:.2f}, fraud: {class_weights[1]:.2f}")
        logger.info(f"Using weighted loss: {class_weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using unweighted loss")

    # ── 4. Optimiser + Scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    # ── 5. Data loaders ────────────────────────────────────────────────────────
    use_oversample = strategy in ("oversample", "both")
    logger.info(f"Imbalance strategy: {strategy} | Oversampling: {use_oversample}")

    if use_oversample:
        train_loader = build_oversampled_loader(
            data, data.train_mask,
            cfg.train.num_neighbors, cfg.train.batch_size,
        )
    else:
        train_loader = build_standard_loader(
            data, data.train_mask,
            cfg.train.num_neighbors, cfg.train.batch_size,
        )

    val_loader = build_standard_loader(
        data, data.val_mask,
        cfg.train.num_neighbors, cfg.train.batch_size,
        shuffle=False,
    )

    # ── 6. Training loop ───────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=cfg.train.patience)
    history    = []

    logger.info(f"\nStarting training for {cfg.train.epochs} epochs...")
    logger.info("─" * 65)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["f1"])

        record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        history.append(record)

        # W&B logging — safe, skipped if no active run
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

        # Early stopping check
        if early_stop.step(val_metrics["f1"], model):
            logger.info(f"\nEarly stopping at epoch {epoch} (best val F1={early_stop.best_score:.4f})")
            break

    # ── 7. Restore best checkpoint and save ───────────────────────────────────
    early_stop.restore_best(model)

    checkpoint_path = os.path.join(cfg.train.checkpoint_dir, "best_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config":           cfg,
        "history":          history,
        "in_channels":      in_channels,
        "best_val_f1":      early_stop.best_score,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")

    history_path = os.path.join(cfg.train.checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\nTraining complete.")
    logger.info(f"  Best val F1:  {early_stop.best_score:.4f}")

    return model, data, history


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN Fraud Detector")
    parser.add_argument("--model",    choices=["graphsage", "gat"], default="graphsage")
    parser.add_argument("--epochs",   type=int, default=150)
    parser.add_argument("--dataset",  choices=["elliptic", "ieee_cis"], default="elliptic")
    parser.add_argument("--strategy", choices=["weighted_loss", "oversample", "both"],
                        default="weighted_loss")
    parser.add_argument("--hidden",   type=int, default=128)
    parser.add_argument("--layers",   type=int, default=3)
    parser.add_argument("--lr",       type=float, default=0.001)
    args = parser.parse_args()

    CFG.model.architecture      = args.model
    CFG.train.epochs            = args.epochs
    CFG.data.dataset            = args.dataset
    CFG.data.imbalance_strategy = args.strategy
    CFG.model.hidden_channels   = args.hidden
    CFG.model.num_layers        = args.layers
    CFG.train.lr                = args.lr

    train(CFG)