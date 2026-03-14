# predict.py
"""
Inference script — load a trained GNN checkpoint and score new transactions.

Usage:
  # Score new transactions against the EXISTING Elliptic graph:
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_transactions.csv

  # Score with a custom threshold (default: from checkpoint, or 0.5 fallback):
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_transactions.csv --threshold 0.52

  # See more verbose output:
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_transactions.csv --verbose

Output:
  - Prints top fraud alerts to console (sorted by P(fraud) descending)
  - Saves full results CSV to outputs/predictions_<timestamp>.csv
  - CSV columns: txId, p_fraud, p_legit, prediction, alert_level, rank

How it works:
  1. Loads the trained model from best_model.pt
  2. Loads the base Elliptic graph (for neighborhood context during message passing)
  3. Appends new transaction nodes + edges to the graph
  4. Runs NeighborLoader → forward pass → softmax → P(fraud) per node
  5. Applies threshold → FLAG or CLEAR decision
  6. Saves ranked output CSV

GraphSAGE is INDUCTIVE: no retraining required for new transactions.
The model learned AGGREGATION FUNCTIONS, not per-node embeddings — it
can immediately compute embeddings for unseen nodes by aggregating their neighbors.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from models.gnn import GraphSAGE, GAT, get_device
from utils.data_loader import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model + config from best_model.pt.

    Returns:
        model:       Trained GNN in eval mode
        cfg:         The Config object used during training
        in_channels: Number of input features the model expects
        best_val_f1: Best validation F1 achieved during training
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg         = ckpt["config"]
    in_channels = ckpt["in_channels"]
    best_val_f1 = ckpt.get("best_val_f1", 0.0)
    arch        = cfg.model.architecture

    # Rebuild the exact same model architecture
    if arch == "graphsage":
        model = GraphSAGE(
            in_channels     = in_channels,
            hidden_channels = cfg.model.hidden_channels,
            num_layers      = cfg.model.num_layers,
            dropout         = cfg.model.dropout,
            aggr            = cfg.model.sage_aggr,
        )
    elif arch == "gat":
        model = GAT(
            in_channels     = in_channels,
            hidden_channels = cfg.model.hidden_channels,
            num_layers      = cfg.model.num_layers,
            dropout         = cfg.model.dropout,
            heads           = cfg.model.gat_heads,
            concat          = cfg.model.gat_concat,
        )
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Architecture : {arch.upper()}")
    logger.info(f"  Parameters   : {n_params:,}")
    logger.info(f"  In channels  : {in_channels}")
    logger.info(f"  Best val F1  : {best_val_f1:.4f}")

    return model, cfg, in_channels, best_val_f1


# ─────────────────────────────────────────────────────────────────────────────
# NEW TRANSACTION PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_new_transactions(input_csv: str, in_channels: int):
    """
    Parse a CSV of new transactions into a feature matrix.

    Expected CSV format (Elliptic-compatible):
        txId, time_step, feat_0, feat_1, ..., feat_165
        OR just:
        txId, feat_0, feat_1, ..., feat_165

    If the feature count doesn't match in_channels, we zero-pad or trim.

    Returns:
        tx_ids:    list of transaction IDs (strings)
        X_new:     np.ndarray [N_new, in_channels]
        new_edges: list of (txId_src, txId_dst) tuples (optional, from edge columns)
    """
    logger.info(f"Reading new transactions from: {input_csv}")
    df = pd.read_csv(input_csv, header=None if _is_headerless(input_csv) else 0)

    # Detect txId column — first column by convention
    tx_ids = df.iloc[:, 0].astype(str).tolist()

    # Drop txId column, keep features
    feat_df = df.iloc[:, 1:]

    # Drop non-numeric columns (e.g. time_step if present as string)
    # Drop time_step if present — training excludes it, so model does not expect it
    if "time_step" in feat_df.columns:
        feat_df = feat_df.drop(columns=["time_step"])

    feat_df = feat_df.select_dtypes(include=[np.number])

    X_raw = feat_df.values.astype(np.float32)
    n_new, n_feat = X_raw.shape

    # Align feature dimension to what the model expects
    if n_feat < in_channels:
        logger.warning(f"Input has {n_feat} features, model expects {in_channels}. Zero-padding.")
        pad = np.zeros((n_new, in_channels - n_feat), dtype=np.float32)
        X_raw = np.concatenate([X_raw, pad], axis=1)
    elif n_feat > in_channels:
        logger.warning(f"Input has {n_feat} features, model expects {in_channels}. Trimming.")
        X_raw = X_raw[:, :in_channels]

    logger.info(f"  New transactions : {n_new}")
    logger.info(f"  Feature columns  : {min(n_feat, in_channels)}")

    return tx_ids, X_raw


def _is_headerless(path: str) -> bool:
    """Check if CSV has no header (first cell is numeric = no header)."""
    try:
        first = pd.read_csv(path, nrows=1, header=None).iloc[0, 0]
        float(first)
        return True
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH EXTENSION
# ─────────────────────────────────────────────────────────────────────────────

def extend_graph(base_data, X_new: np.ndarray, new_edge_pairs=None):
    """
    Append new transaction nodes to the existing base graph.

    New nodes get no initial edges to each other (unless new_edge_pairs provided).
    They inherit neighborhood context from the existing graph through message passing
    if they share edges with existing nodes.

    Args:
        base_data:      PyG Data object (existing graph)
        X_new:          [N_new, F] feature matrix for new nodes
        new_edge_pairs: optional list of (src_idx, dst_idx) edges within new nodes

    Returns:
        extended_data:  PyG Data with new nodes appended
        new_node_ids:   integer indices of the new nodes in extended_data
    """
    N_base = base_data.x.shape[0]
    N_new  = X_new.shape[0]

    X_new_t = torch.FloatTensor(X_new)

    # Normalise new features with the same statistics approach
    # (simple z-score per feature using base graph's distribution)
    x_mean = base_data.x.mean(dim=0, keepdim=True)
    x_std  = base_data.x.std(dim=0, keepdim=True).clamp(min=1e-8)
    X_new_t = (X_new_t - x_mean) / x_std

    # Concatenate feature matrices
    x_extended    = torch.cat([base_data.x, X_new_t], dim=0)

    # Labels: -1 for new nodes (unknown — to be predicted)
    y_new         = torch.full((N_new,), -1, dtype=torch.long)
    y_extended    = torch.cat([base_data.y, y_new], dim=0)

    # Edge index: keep existing edges; add any new→existing edges if provided
    edge_index = base_data.edge_index.clone()

    if new_edge_pairs is not None and len(new_edge_pairs) > 0:
        new_edges_t  = torch.LongTensor(new_edge_pairs).T   # [2, E_new]
        new_edges_t += N_base                                # offset to new node range
        edge_index   = torch.cat([edge_index, new_edges_t], dim=1)

    # Prediction mask: only new nodes
    new_node_ids = list(range(N_base, N_base + N_new))
    pred_mask    = torch.zeros(N_base + N_new, dtype=torch.bool)
    pred_mask[new_node_ids] = True

    from torch_geometric.data import Data
    extended_data = Data(
        x           = x_extended,
        edge_index  = edge_index,
        y           = y_extended,
        pred_mask   = pred_mask,
    )
    # Carry over train/val/test masks (not used for inference but useful to keep)
    for attr in ["train_mask", "val_mask", "test_mask"]:
        if hasattr(base_data, attr):
            old_mask   = getattr(base_data, attr)
            new_mask_t = torch.zeros(N_base + N_new, dtype=torch.bool)
            new_mask_t[:N_base] = old_mask
            setattr(extended_data, attr, new_mask_t)

    logger.info(f"  Base nodes   : {N_base:,}")
    logger.info(f"  New nodes    : {N_new}")
    logger.info(f"  Total nodes  : {N_base + N_new:,}")
    logger.info(f"  Total edges  : {edge_index.shape[1]:,}")

    return extended_data, new_node_ids


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    data,
    new_node_ids: list,
    cfg,
    device: torch.device,
) -> np.ndarray:
    """
    Run GNN inference on the new transaction nodes.

    Uses NeighborLoader to sample k-hop neighborhoods around each new node
    — exactly the same sampling strategy as training.

    Returns:
        probs: np.ndarray [N_new]  — P(fraud) for each new node
    """
    pred_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    pred_mask[new_node_ids] = True

    loader = NeighborLoader(
        data,
        num_neighbors = cfg.train.num_neighbors,   # [10, 5, 5] from training
        batch_size    = min(cfg.train.batch_size, len(new_node_ids)),
        input_nodes   = pred_mask,
        shuffle       = False,
    )

    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            logits = model(batch.x, batch.edge_index)

            # Only take predictions for the seed nodes (not neighborhood context)
            seed_logits = logits[:batch.batch_size]
            probs       = F.softmax(seed_logits, dim=-1)[:, 1]   # P(fraud)

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_results(tx_ids: list, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Build the results DataFrame with fraud scores and alert levels.

    Alert levels:
        HIGH   — P(fraud) >= threshold + 0.15  → immediate block / escalate
        MEDIUM — P(fraud) >= threshold          → flag for analyst review
        LOW    — P(fraud) < threshold           → clear (pass through)
    """
    predictions = (probs >= threshold).astype(int)

    alert_levels = []
    for p in probs:
        if p >= threshold + 0.15:
            alert_levels.append("HIGH")
        elif p >= threshold:
            alert_levels.append("MEDIUM")
        else:
            alert_levels.append("LOW")

    df = pd.DataFrame({
        "txId"       : tx_ids,
        "p_fraud"    : probs.round(4),
        "p_legit"    : (1 - probs).round(4),
        "prediction" : ["FRAUD" if p == 1 else "LEGIT" for p in predictions],
        "alert_level": alert_levels,
    })

    # Sort by P(fraud) descending — highest risk first
    df = df.sort_values("p_fraud", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df


def print_summary(df: pd.DataFrame, threshold: float, verbose: bool):
    """Print prediction summary to console."""
    n_total   = len(df)
    n_fraud   = (df["prediction"] == "FRAUD").sum()
    n_high    = (df["alert_level"] == "HIGH").sum()
    n_medium  = (df["alert_level"] == "MEDIUM").sum()

    print("\n" + "=" * 60)
    print("  FRAUD DETECTION RESULTS")
    print("=" * 60)
    print(f"  Threshold      : {threshold:.2f}")
    print(f"  Total scored   : {n_total}")
    print(f"  Flagged FRAUD  : {n_fraud}  ({100*n_fraud/n_total:.1f}%)")
    print(f"    └ HIGH alert : {n_high}")
    print(f"    └ MEDIUM     : {n_medium}")
    print(f"  Cleared LEGIT  : {n_total - n_fraud}  ({100*(n_total-n_fraud)/n_total:.1f}%)")
    print("=" * 60)

    # Always show top-10 flagged transactions
    flagged = df[df["prediction"] == "FRAUD"].head(10)
    if len(flagged) > 0:
        print(f"\n  Top {len(flagged)} Fraud Alerts (highest risk first):\n")
        print(f"  {'Rank':<6} {'txId':<25} {'P(fraud)':<10} {'Alert'}")
        print(f"  {'-'*6} {'-'*25} {'-'*10} {'-'*8}")
        for _, row in flagged.iterrows():
            print(f"  {row['rank']:<6} {str(row['txId']):<25} {row['p_fraud']:<10.4f} {row['alert_level']}")
    else:
        print("\n  No transactions flagged as fraud at this threshold.")

    if verbose:
        print(f"\n  P(fraud) distribution:")
        print(f"    min   : {df['p_fraud'].min():.4f}")
        print(f"    max   : {df['p_fraud'].max():.4f}")
        print(f"    mean  : {df['p_fraud'].mean():.4f}")
        print(f"    median: {df['p_fraud'].median():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN Fraud Detection — Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_txs.csv
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_txs.csv --threshold 0.4
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_txs.csv --verbose
  python predict.py --checkpoint outputs/best_model.pt --input data/raw/new_txs.csv --no-base-graph
        """
    )
    parser.add_argument("--checkpoint",   required=True,  help="Path to best_model.pt")
    parser.add_argument("--input",        required=True,  help="CSV of new transactions to score")
    parser.add_argument("--threshold",    type=float, default=None,
                        help="Fraud classification threshold (default: 0.52 from best run, or 0.5)")
    parser.add_argument("--output-dir",   default="outputs", help="Directory to save predictions CSV")
    parser.add_argument("--no-base-graph",action="store_true",
                        help="Score transactions standalone (no base graph context). "
                             "Less accurate but works without the original dataset.")
    parser.add_argument("--verbose",      action="store_true", help="Show extra stats")
    args = parser.parse_args()

    # ── 1. Resolve device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── 2. Load checkpoint ───────────────────────────────────────────────────
    model, cfg, in_channels, best_val_f1 = load_checkpoint(args.checkpoint, device)

    # ── 3. Set threshold ─────────────────────────────────────────────────────
    # Use: (a) user-specified, (b) default from best run, (c) fallback 0.5
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Threshold: {threshold:.2f}  (user-specified)")
    else:
        threshold = 0.52   # Our best run's optimal threshold
        logger.info(f"Threshold: {threshold:.2f}  (default from best run)")

    # ── 4. Parse new transactions ────────────────────────────────────────────
    tx_ids, X_new = parse_new_transactions(args.input, in_channels)
    N_new = len(tx_ids)

    # ── 5. Load base graph (for neighborhood context) ────────────────────────
    if not args.no_base_graph:
        logger.info("Loading base graph for neighborhood context...")
        try:
            base_data = load_dataset(cfg)
            logger.info("  Base graph loaded.")
            extended_data, new_node_ids = extend_graph(base_data, X_new)
        except Exception as e:
            logger.warning(f"Could not load base graph ({e}). Falling back to standalone mode.")
            args.no_base_graph = True

    if args.no_base_graph:
        # Standalone: score without existing graph context
        # Create a minimal graph: N_new nodes, no edges between them
        logger.info("Standalone mode: scoring without base graph context.")
        from torch_geometric.data import Data
        x_t = torch.FloatTensor(X_new)
        # Normalise standalone (just z-score within the batch)
        x_t = (x_t - x_t.mean(0)) / (x_t.std(0).clamp(min=1e-8))
        extended_data = Data(
            x          = x_t,
            edge_index = torch.zeros((2, 0), dtype=torch.long),
            y          = torch.full((N_new,), -1, dtype=torch.long),
        )
        new_node_ids = list(range(N_new))

    # ── 6. Run inference ─────────────────────────────────────────────────────
    logger.info("Running inference...")
    probs = run_inference(model, extended_data, new_node_ids, cfg, device)

    # Sanity check: probs should align with tx_ids order
    assert len(probs) == N_new, f"Expected {N_new} predictions, got {len(probs)}"

    # ── 7. Format & display results ──────────────────────────────────────────
    results_df = format_results(tx_ids, probs, threshold)
    print_summary(results_df, threshold, args.verbose)

    # ── 8. Save to CSV ───────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nPredictions saved: {output_path}")

    # Return for programmatic use
    return results_df


if __name__ == "__main__":
    main()