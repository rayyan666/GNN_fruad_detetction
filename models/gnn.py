# models/gnn.py
"""
Step 4: Graph Neural Network model definitions.

Implements two architectures, both interchangeable via config:

1. GraphSAGE (Hamilton et al., 2017)
   - Samples and aggregates features from local neighborhoods
   - Scales to large graphs via mini-batch training
   - Aggregation options: mean, max, LSTM

2. GAT (Graph Attention Network, Velickovic et al., 2018)
   - Learns attention weights over neighbors
   - Multiple attention heads capture different relationship types
   - Better at filtering noisy neighbors (important for fraud)

Both models output per-node class logits for binary fraud classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    SAGEConv,
    GATConv,
    BatchNorm,
    JumpingKnowledge,
)
from torch_geometric.nn import global_mean_pool
from typing import List, Optional


# ─────────────────────────────────────────────
# GRAPHSAGE MODEL
# ─────────────────────────────────────────────

class GraphSAGE(nn.Module):
    """
    GraphSAGE for node-level fraud classification.

    Architecture:
        Input → [SAGEConv → BatchNorm → ReLU → Dropout] × num_layers → Linear → Logits

    Key design decisions:
      - BatchNorm after each layer stabilizes training on heterogeneous transaction data
      - JumpingKnowledge (cat) concatenates all layer outputs, preserving both
        local (shallow) and structural (deep) neighborhood information
      - Dropout combats overfitting on imbalanced classes
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,          # binary: legit / fraud
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",             # 'mean' | 'max' | 'lstm'
        jk_mode: str = "cat",           # JumpingKnowledge mode
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(BatchNorm(hidden_channels))

        # JumpingKnowledge: concatenate outputs from all layers
        # This gives the classifier access to representations at different depths
        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)

        jk_out_channels = hidden_channels * num_layers if jk_mode == "cat" else hidden_channels

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(jk_out_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, in_channels]  node features
            edge_index: [2, E]            edge list

        Returns:
            logits:     [N, 2]            per-node class logits
        """
        layer_outputs = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)   # Message passing: aggregate neighbor features
            x = bn(x)                 # Normalize across batch
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

        # Concatenate representations from all depths
        x = self.jk(layer_outputs)

        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return node embeddings (before final classification head) for visualization."""
        layer_outputs = []
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            layer_outputs.append(x)
        return self.jk(layer_outputs)


# ─────────────────────────────────────────────
# GAT MODEL
# ─────────────────────────────────────────────

class GAT(nn.Module):
    """
    Graph Attention Network for node-level fraud classification.

    Architecture:
        Input → [GATConv (multi-head) → ELU → Dropout] × num_layers → Linear → Logits

    Key design decisions:
      - Multi-head attention: each head specializes in different neighborhood patterns
      - Heads are concatenated in hidden layers, averaged in the final layer
      - ELU activation (instead of ReLU) for smoother gradient flow in attention
      - Residual-style JK connection optional for deep networks
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        concat: bool = True,            # Concatenate heads (True) or average (False)
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.concat = concat

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input layer: concat heads → hidden_channels * heads output
        self.convs.append(GATConv(
            in_channels, hidden_channels, heads=heads, concat=concat, dropout=dropout
        ))
        in_ch = hidden_channels * heads if concat else hidden_channels
        self.bns.append(BatchNorm(in_ch))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                in_ch, hidden_channels, heads=heads, concat=concat, dropout=dropout
            ))
            in_ch = hidden_channels * heads if concat else hidden_channels
            self.bns.append(BatchNorm(in_ch))

        # Final GAT layer: always average heads to keep output size consistent
        self.convs.append(GATConv(in_ch, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          [N, in_channels]
            edge_index: [2, E]

        Returns:
            logits:     [N, 2]
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Node embeddings before the classifier head."""
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        return F.elu(x)


# ─────────────────────────────────────────────
# MODEL FACTORY
# ─────────────────────────────────────────────

def build_model(cfg, in_channels: int) -> nn.Module:
    """
    Factory function: builds the correct GNN model from config.

    Args:
        cfg:         Config object (from config.py)
        in_channels: Number of input node features

    Returns:
        Instantiated PyTorch model
    """
    m = cfg.model

    if m.architecture == "graphsage":
        model = GraphSAGE(
            in_channels     = in_channels,
            hidden_channels = m.hidden_channels,
            num_layers      = m.num_layers,
            dropout         = m.dropout,
            aggr            = m.sage_aggr,
        )
    elif m.architecture == "gat":
        model = GAT(
            in_channels     = in_channels,
            hidden_channels = m.hidden_channels,
            num_layers      = m.num_layers,
            dropout         = m.dropout,
            heads           = m.gat_heads,
            concat          = m.gat_concat,
        )
    else:
        raise ValueError(f"Unknown architecture: {m.architecture}. Choose 'graphsage' or 'gat'.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {m.architecture.upper()}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Layers: {m.num_layers}")
    print(f"  Hidden dim: {m.hidden_channels}")
    print(f"  Dropout: {m.dropout}\n")

    return model


def get_device(cfg) -> torch.device:
    """Resolve device from config, supporting 'auto' detection."""
    spec = cfg.train.device
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(spec)