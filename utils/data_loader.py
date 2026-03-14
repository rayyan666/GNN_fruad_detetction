# utils/data_loader.py
"""
Step 2: Load raw datasets and convert them into PyTorch Geometric Data objects.

Supports:
  - Elliptic Bitcoin Dataset  (node classification on transaction graph)
  - IEEE-CIS Fraud Detection  (converted to bipartite graph: cards ↔ transactions)

The Elliptic dataset is the primary target. Each node is a Bitcoin transaction,
edges represent bitcoin flows, and ~2% of nodes are labelled 'illicit' (fraud).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ELLIPTIC BITCOIN DATASET LOADER
# ─────────────────────────────────────────────

def load_elliptic(raw_dir: str, local_features_only: bool = False) -> Data:
    """
    Load the Elliptic Bitcoin Dataset and construct a PyG Data object.

    Dataset layout (download from Kaggle 'elliptic-bitcoin-dataset'):
        elliptic_txs_features.csv  — 166 features per transaction node
        elliptic_txs_edgelist.csv  — directed edges (txId1 → txId2)
        elliptic_txs_classes.csv   — labels: 1=illicit, 2=licit, unknown

    Returns:
        PyG Data with:
            x            : [N, 166] node feature matrix
            edge_index   : [2, E]   directed edge list
            y            : [N]      binary labels (1=fraud, 0=legit, -1=unknown)
            train/val/test masks
    """
    logger.info("Loading Elliptic Bitcoin Dataset...")

    # ── Load raw CSVs ──────────────────────────────────────────────────────────
    feat_path  = os.path.join(raw_dir, "elliptic_txs_features.csv")
    edge_path  = os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    label_path = os.path.join(raw_dir, "elliptic_txs_classes.csv")

    # Features: first col = txId, second = time step, rest = anonymized features
    feat_df  = pd.read_csv(feat_path, header=None)
    feat_df.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(feat_df.shape[1] - 2)]

    edge_df  = pd.read_csv(edge_path)
    label_df = pd.read_csv(label_path)
    label_df.columns = ["txId", "class"]

    logger.info(f"  Nodes: {len(feat_df)}, Edges: {len(edge_df)}, Labelled: {(label_df['class'] != 'unknown').sum()}")

    # ── Build node index mapping  txId → sequential integer ───────────────────
    node_ids = feat_df["txId"].values
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(node_ids)}
    N = len(node_ids)

    # ── Node features (normalize) ──────────────────────────────────────────────
    feat_cols = [c for c in feat_df.columns if c.startswith("feat_")]

    # Ablation: optionally strip the 72 pre-aggregated neighborhood features (feat_94 onward)
    # keeping only the 94 raw local transaction features
    if local_features_only:
        feat_cols = feat_cols[:94]
        logger.info("  [Ablation] Using LOCAL features only: feat_0 to feat_93 (94 features)")
    else:
        logger.info(f"  Using ALL features: {len(feat_cols)} total")

    X_raw = feat_df[feat_cols].values.astype(np.float32)

    # Fit scaler only on training indices (we'll recompute splits shortly,
    # so here we do a full-fit as a practical simplification)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ── Labels ─────────────────────────────────────────────────────────────────
    # Map: '1' → 1 (illicit/fraud), '2' → 0 (licit/legit), 'unknown' → -1
    label_map = {"1": 1, "2": 0, "unknown": -1}
    label_df["label"] = label_df["class"].astype(str).map(label_map)

    # Align labels to feature ordering
    label_series = feat_df[["txId"]].merge(label_df[["txId", "label"]], on="txId", how="left")
    y = label_series["label"].fillna(-1).values.astype(np.int64)

    # ── Edge index ─────────────────────────────────────────────────────────────
    src = edge_df.iloc[:, 0].map(id_to_idx).dropna().astype(int).values
    dst = edge_df.iloc[:, 1].map(id_to_idx).dropna().astype(int).values

    # Keep only edges where both endpoints exist in our node set
    valid = (src < N) & (dst < N)
    src, dst = src[valid], dst[valid]
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── Train / Val / Test masks (only on labelled nodes) ─────────────────────
    labelled_idx = np.where(y >= 0)[0]
    labels_labelled = y[labelled_idx]

    # Stratified split to preserve fraud ratio
    train_idx, temp_idx = train_test_split(
        labelled_idx, test_size=0.30, stratify=labels_labelled, random_state=42
    )
    val_labels = y[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=val_labels, random_state=42
    )

    train_mask = _make_mask(N, train_idx)
    val_mask   = _make_mask(N, val_idx)
    test_mask  = _make_mask(N, test_idx)

    # ── Assemble PyG Data object ───────────────────────────────────────────────
    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = train_mask,
        val_mask   = val_mask,
        test_mask  = test_mask,
    )

    # Attach metadata for downstream use
    data.num_fraud   = int((y == 1).sum())
    data.num_legit   = int((y == 0).sum())
    data.num_unknown = int((y == -1).sum())
    data.node_ids    = node_ids
    data.time_steps  = feat_df["time_step"].values

    _log_dataset_stats(data)
    return data


# ─────────────────────────────────────────────
# IEEE-CIS FRAUD DETECTION LOADER  (adapter)
# ─────────────────────────────────────────────

def load_ieee_cis(raw_dir: str) -> Data:
    """
    Load the IEEE-CIS dataset and model it as a bipartite graph.

    Graph construction:
        Nodes : credit cards  +  transactions
        Edges : card ─── transaction  (card made this transaction)

    This models the payment network: fraud propagates through shared cards,
    emails, devices, and addresses.

    Files (from Kaggle 'ieee-fraud-detection'):
        train_transaction.csv
        train_identity.csv
    """
    logger.info("Loading IEEE-CIS Fraud Detection Dataset...")

    txn_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path  = os.path.join(raw_dir, "train_identity.csv")

    txn_df = pd.read_csv(txn_path)
    try:
        id_df = pd.read_csv(id_path)
        txn_df = txn_df.merge(id_df, on="TransactionID", how="left")
    except FileNotFoundError:
        logger.warning("train_identity.csv not found; proceeding without identity features.")

    logger.info(f"  Transactions: {len(txn_df)}, Fraud rate: {txn_df['isFraud'].mean():.3%}")

    # ── Build bipartite graph: cards (type-0 nodes) ↔ transactions (type-1 nodes) ─
    # Encode card as node; each unique card gets an integer ID
    txn_df["card_id"] = (
        txn_df["card1"].astype(str) + "_" +
        txn_df["card2"].astype(str) + "_" +
        txn_df["card4"].astype(str)
    )
    unique_cards = txn_df["card_id"].unique()
    card_to_idx  = {c: i for i, c in enumerate(unique_cards)}
    n_cards = len(unique_cards)
    n_txns  = len(txn_df)
    N = n_cards + n_txns

    # Transaction nodes are offset by n_cards
    txn_card_src = txn_df["card_id"].map(card_to_idx).values
    txn_node_dst = np.arange(n_txns) + n_cards

    src = np.concatenate([txn_card_src, txn_node_dst])
    dst = np.concatenate([txn_node_dst, txn_card_src])   # undirected
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── Node features ─────────────────────────────────────────────────────────
    # Card nodes: zero features (unknown aggregates)
    # Transaction nodes: numeric features from the dataset
    num_cols = txn_df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["TransactionID", "isFraud"]]

    X_txn = txn_df[num_cols].fillna(0).values.astype(np.float32)
    X_txn = StandardScaler().fit_transform(X_txn)

    X_card = np.zeros((n_cards, X_txn.shape[1]), dtype=np.float32)
    X = np.vstack([X_card, X_txn])

    # ── Labels ─────────────────────────────────────────────────────────────────
    y = np.full(N, -1, dtype=np.int64)
    y[n_cards:] = txn_df["isFraud"].values.astype(np.int64)   # only txn nodes labelled

    # ── Masks ──────────────────────────────────────────────────────────────────
    labelled_idx = np.where(y >= 0)[0]
    train_idx, temp_idx = train_test_split(labelled_idx, test_size=0.30, stratify=y[labelled_idx], random_state=42)
    val_idx, test_idx   = train_test_split(temp_idx,     test_size=0.50, stratify=y[temp_idx],     random_state=42)

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = _make_mask(N, train_idx),
        val_mask   = _make_mask(N, val_idx),
        test_mask  = _make_mask(N, test_idx),
    )
    data.num_fraud = int((y == 1).sum())
    data.num_legit = int((y == 0).sum())

    _log_dataset_stats(data)
    return data


# ─────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (for testing without Kaggle)
# ─────────────────────────────────────────────

def generate_synthetic_graph(
    n_nodes: int = 5000,
    n_edges: int = 15000,
    fraud_ratio: float = 0.02,
    n_features: int = 50,
    seed: int = 42,
) -> Data:
    """
    Generate a synthetic transaction graph for testing the full pipeline
    without needing to download any dataset.

    Fraud nodes are embedded in dense cliques (simulating mule networks).
    """
    rng = np.random.default_rng(seed)
    logger.info(f"Generating synthetic graph: {n_nodes} nodes, {n_edges} edges, {fraud_ratio:.1%} fraud")

    # ── Labels ─────────────────────────────────────────────────────────────────
    n_fraud = int(n_nodes * fraud_ratio)
    y = np.zeros(n_nodes, dtype=np.int64)
    fraud_nodes = rng.choice(n_nodes, size=n_fraud, replace=False)
    y[fraud_nodes] = 1

    # ── Features: fraud nodes have slightly different distribution ─────────────
    X = rng.standard_normal((n_nodes, n_features)).astype(np.float32)
    X[fraud_nodes] += rng.standard_normal((n_fraud, n_features)) * 0.5 + 0.3

    # ── Edges: random + dense cliques among fraud nodes ────────────────────────
    rand_src = rng.integers(0, n_nodes, size=n_edges)
    rand_dst = rng.integers(0, n_nodes, size=n_edges)

    # Fraud clique edges (fraud nodes are over-connected)
    clique_pairs = [(fraud_nodes[i], fraud_nodes[j])
                    for i in range(n_fraud)
                    for j in range(i + 1, min(i + 5, n_fraud))]
    if clique_pairs:
        clique_arr = np.array(clique_pairs)
        clique_src, clique_dst = clique_arr[:, 0], clique_arr[:, 1]
        src = np.concatenate([rand_src, clique_src, clique_dst])
        dst = np.concatenate([rand_dst, clique_dst, clique_src])
    else:
        src, dst = rand_src, rand_dst

    # Remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]

    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── Masks ──────────────────────────────────────────────────────────────────
    all_idx = np.arange(n_nodes)
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx,  test_idx  = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=seed)

    data = Data(
        x          = torch.tensor(X, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(y, dtype=torch.long),
        train_mask = _make_mask(n_nodes, train_idx),
        val_mask   = _make_mask(n_nodes, val_idx),
        test_mask  = _make_mask(n_nodes, test_idx),
    )
    data.num_fraud = int(n_fraud)
    data.num_legit = int(n_nodes - n_fraud)
    data.num_unknown = 0

    _log_dataset_stats(data)
    return data


# ─────────────────────────────────────────────
# DATASET FACTORY
# ─────────────────────────────────────────────

def load_dataset(cfg) -> Data:
    """
    Factory function: routes to the correct loader based on config.
    Falls back to synthetic data if raw files are not present.
    """
    dataset = cfg.data.dataset
    raw_dir = cfg.data.raw_dir

    feat_file = os.path.join(raw_dir, cfg.data.elliptic_features)

    if dataset == "elliptic" and os.path.exists(feat_file):
        return load_elliptic(raw_dir, local_features_only=getattr(cfg.data, "local_features_only", False))
    elif dataset == "ieee_cis" and os.path.exists(os.path.join(raw_dir, "train_transaction.csv")):
        return load_ieee_cis(raw_dir)
    else:
        logger.warning(
            f"Dataset files not found in '{raw_dir}'. "
            "Falling back to synthetic data for demonstration.\n"
            "To use real data, download from Kaggle and place CSVs in data/raw/."
        )
        return generate_synthetic_graph()


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_mask(n: int, indices: np.ndarray) -> torch.Tensor:
    """Create a boolean mask tensor of size n with True at given indices."""
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask


def compute_class_weights(data: Data) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for the training set.
    Used in weighted cross-entropy to combat class imbalance.

    Returns tensor([weight_legit, weight_fraud])
    """
    train_labels = data.y[data.train_mask]
    # Only consider labelled nodes
    labelled = train_labels[train_labels >= 0]
    n_total  = len(labelled)
    n_fraud  = (labelled == 1).sum().item()
    n_legit  = (labelled == 0).sum().item()

    # Inverse frequency: rarer class gets higher weight
    w_fraud = n_total / (2 * n_fraud + 1e-8)
    w_legit = n_total / (2 * n_legit + 1e-8)

    logger.info(f"Class weights → legit: {w_legit:.2f}, fraud: {w_fraud:.2f}")
    return torch.tensor([w_legit, w_fraud], dtype=torch.float)


def _log_dataset_stats(data: Data):
    logger.info(
        f"  Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges | "
        f"Fraud: {data.num_fraud} ({data.num_fraud/data.num_nodes:.2%}) | "
        f"Legit: {data.num_legit} | "
        f"Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}"
    )