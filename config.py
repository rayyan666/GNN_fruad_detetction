# config.py
"""
Central configuration for the Fraud GNN pipeline.
All hyperparameters, paths, and toggles live here.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, List


@dataclass
class DataConfig:
    # Dataset choice: 'elliptic' or 'ieee_cis'
    dataset: Literal["elliptic", "ieee_cis"] = "elliptic"

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # Elliptic-specific filenames (download from Kaggle)
    elliptic_features: str = "elliptic_txs_features.csv"
    elliptic_edges: str = "elliptic_txs_edgelist.csv"
    elliptic_labels: str = "elliptic_txs_classes.csv"

    # Class imbalance strategy: 'weighted_loss' | 'oversample' | 'both'
    imbalance_strategy: Literal["weighted_loss", "oversample", "both"] = "weighted_loss"

    # Ablation: use only local transaction features (feat_0-93), strip aggregated stats
    local_features_only: bool = False

    # Train/val/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio is implied as 1 - train - val


@dataclass
class ModelConfig:
    # Model architecture: 'graphsage' or 'gat'
    architecture: Literal["graphsage", "gat"] = "graphsage"

    # Layer sizes
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.3

    # GAT-specific
    gat_heads: int = 4
    gat_concat: bool = True         # If True, heads are concatenated; else averaged

    # GraphSAGE-specific
    sage_aggr: Literal["mean", "max", "lstm"] = "mean"


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 0.001
    weight_decay: float = 5e-4
    batch_size: int = 512           # For mini-batch training with NeighborLoader
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5, 5])
    patience: int = 20              # Early stopping patience
    checkpoint_dir: str = "outputs"
    device: str = "auto"            # 'auto' | 'cpu' | 'cuda' | 'mps'


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42
    log_every: int = 10             # Log metrics every N epochs


# Singleton config — import this everywhere
CFG = Config()