# utils/graph_eda.py
"""
Step 3: Exploratory Data Analysis of the transaction graph.

Covers:
  - Degree distribution (fraud vs legit)
  - Connected component analysis
  - Temporal patterns (Elliptic time steps)
  - NetworkX subgraph sampling and visualization
  - Feature correlation with fraud label
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
import logging
import os

logger = logging.getLogger(__name__)

# Consistent color palette
PALETTE = {"fraud": "#e63946", "legit": "#457b9d", "unknown": "#adb5bd"}


# ─────────────────────────────────────────────
# MAIN EDA ENTRY POINT
# ─────────────────────────────────────────────

def run_eda(data: Data, output_dir: str = "outputs") -> dict:
    """
    Run full EDA suite and save all plots to output_dir.
    Returns a dict of computed statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats = {}

    logger.info("Running graph EDA...")

    stats["basic"]   = basic_graph_stats(data)
    plot_degree_distribution(data, output_dir)
    plot_feature_distributions(data, output_dir)
    plot_class_imbalance(data, output_dir)

    # NetworkX is expensive on large graphs — sample a subgraph
    nx_graph = sample_networkx_subgraph(data, max_nodes=2000)
    stats["nx"] = networkx_stats(nx_graph)
    plot_degree_centrality(nx_graph, data, output_dir)

    if hasattr(data, "time_steps"):
        plot_temporal_fraud_rate(data, output_dir)

    logger.info(f"EDA complete. Plots saved to {output_dir}/")
    return stats


# ─────────────────────────────────────────────
# BASIC STATS
# ─────────────────────────────────────────────

def basic_graph_stats(data: Data) -> dict:
    """Print and return basic graph statistics."""
    N = data.num_nodes
    E = data.edge_index.shape[1]
    y = data.y.numpy()

    n_fraud   = (y == 1).sum()
    n_legit   = (y == 0).sum()
    n_unknown = (y == -1).sum()

    node_deg = degree(data.edge_index[0], num_nodes=N).numpy()

    stats = {
        "num_nodes": N,
        "num_edges": E,
        "avg_degree": float(node_deg.mean()),
        "max_degree": float(node_deg.max()),
        "num_fraud": int(n_fraud),
        "num_legit": int(n_legit),
        "num_unknown": int(n_unknown),
        "fraud_ratio": float(n_fraud / (n_fraud + n_legit + 1e-8)),
        "density": float(E / (N * (N - 1) + 1e-8)),
    }

    print("\n" + "─" * 55)
    print("  GRAPH STATISTICS")
    print("─" * 55)
    for k, v in stats.items():
        print(f"  {k:<20}: {v:.4g}" if isinstance(v, float) else f"  {k:<20}: {v}")
    print("─" * 55 + "\n")

    return stats


# ─────────────────────────────────────────────
# DEGREE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_degree_distribution(data: Data, output_dir: str):
    """
    Compare degree distributions of fraud vs legit nodes.
    Fraud nodes in mule networks often have anomalously high or low degree.
    """
    N = data.num_nodes
    y = data.y.numpy()

    in_deg  = degree(data.edge_index[1], num_nodes=N).numpy()
    out_deg = degree(data.edge_index[0], num_nodes=N).numpy()

    labelled = y >= 0
    fraud_mask = (y == 1) & labelled
    legit_mask = (y == 0) & labelled

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Degree Distribution: Fraud vs Legit Nodes", fontsize=14, fontweight="bold")

    for ax, deg_arr, title in zip(axes, [in_deg, out_deg], ["In-Degree", "Out-Degree"]):
        # Use log-scale histogram
        bins = np.logspace(0, np.log10(max(deg_arr.max(), 2)), 40)
        ax.hist(deg_arr[legit_mask], bins=bins, alpha=0.6, color=PALETTE["legit"],
                label=f"Legit (n={legit_mask.sum():,})", density=True)
        ax.hist(deg_arr[fraud_mask], bins=bins, alpha=0.6, color=PALETTE["fraud"],
                label=f"Fraud (n={fraud_mask.sum():,})", density=True)
        ax.set_xscale("log")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate medians
        ax.axvline(np.median(deg_arr[legit_mask]), color=PALETTE["legit"],
                   linestyle="--", alpha=0.8, label=f"Legit median={np.median(deg_arr[legit_mask]):.1f}")
        ax.axvline(np.median(deg_arr[fraud_mask]), color=PALETTE["fraud"],
                   linestyle="--", alpha=0.8, label=f"Fraud median={np.median(deg_arr[fraud_mask]):.1f}")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degree_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: degree_distribution.png")


# ─────────────────────────────────────────────
# FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────

def plot_feature_distributions(data: Data, output_dir: str, top_k: int = 6):
    """
    Plot the top-K most discriminative features (highest absolute mean diff
    between fraud and legit distributions).
    """
    X = data.x.numpy()
    y = data.y.numpy()

    fraud_mask = y == 1
    legit_mask = y == 0

    if not (fraud_mask.any() and legit_mask.any()):
        logger.warning("Skipping feature distribution plot: missing fraud or legit labels.")
        return

    # Rank features by absolute mean difference (simple discrimination score)
    mean_diff = np.abs(X[fraud_mask].mean(0) - X[legit_mask].mean(0))
    top_features = np.argsort(mean_diff)[::-1][:top_k]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Top {top_k} Most Discriminative Features", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i, feat_idx in enumerate(top_features):
        ax = axes[i]
        ax.hist(X[legit_mask, feat_idx], bins=50, alpha=0.6, color=PALETTE["legit"],
                label="Legit", density=True)
        ax.hist(X[fraud_mask, feat_idx], bins=50, alpha=0.6, color=PALETTE["fraud"],
                label="Fraud", density=True)
        ax.set_title(f"Feature {feat_idx} (Δμ={mean_diff[feat_idx]:.3f})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: feature_distributions.png")


# ─────────────────────────────────────────────
# CLASS IMBALANCE
# ─────────────────────────────────────────────

def plot_class_imbalance(data: Data, output_dir: str):
    """Bar chart showing the severe class imbalance in the dataset."""
    y = data.y.numpy()
    counts = {
        "Legit\n(label=0)":   int((y == 0).sum()),
        "Fraud\n(label=1)":   int((y == 1).sum()),
        "Unknown\n(label=-1)": int((y == -1).sum()),
    }
    colors = [PALETTE["legit"], PALETTE["fraud"], PALETTE["unknown"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.keys(), counts.values(), color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("Class Distribution (Node Labels)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")

    for bar, (label, count) in zip(bars, counts.items()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{count:,}\n({count/sum(counts.values()):.1%})",
                ha="center", va="bottom", fontsize=9)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_imbalance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: class_imbalance.png")


# ─────────────────────────────────────────────
# NETWORKX SUBGRAPH SAMPLING
# ─────────────────────────────────────────────

def sample_networkx_subgraph(data: Data, max_nodes: int = 2000) -> nx.DiGraph:
    """
    Convert a sampled subgraph from PyG Data to NetworkX for structural analysis.
    Sampling is necessary for large graphs — NetworkX doesn't scale to millions of nodes.
    """
    N = data.num_nodes
    if N > max_nodes:
        # Sample a connected neighborhood: pick seed nodes, do BFS expansion
        logger.info(f"  Graph has {N} nodes; sampling {max_nodes}-node subgraph for NetworkX analysis.")
        y = data.y.numpy()
        # Prefer sampling around fraud nodes for richer analysis
        fraud_idx = np.where(y == 1)[0]
        seed = fraud_idx[:min(50, len(fraud_idx))].tolist() if len(fraud_idx) > 0 else [0]

        edge_index = data.edge_index.numpy()
        adjacency = {i: [] for i in range(N)}
        for s, d in zip(edge_index[0], edge_index[1]):
            adjacency[s].append(d)

        visited = set(seed)
        queue = list(seed)
        while queue and len(visited) < max_nodes:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        sampled_nodes = sorted(visited)[:max_nodes]
        node_set = set(sampled_nodes)
        node_map = {old: new for new, old in enumerate(sampled_nodes)}

        # Filter edges to sampled subgraph
        mask = np.array([(s in node_set and d in node_set) for s, d in
                         zip(edge_index[0], edge_index[1])])
        sub_src = edge_index[0][mask]
        sub_dst = edge_index[1][mask]

        G = nx.DiGraph()
        for old_id in sampled_nodes:
            label = int(data.y[old_id].item())
            G.add_node(node_map[old_id], label=label, original_id=old_id)
        for s, d in zip(sub_src, sub_dst):
            G.add_edge(node_map[s], node_map[d])
    else:
        G = to_networkx(data, node_attrs=["y"])
        nx.set_node_attributes(G, {i: int(data.y[i].item()) for i in range(N)}, "label")

    return G


def networkx_stats(G: nx.DiGraph) -> dict:
    """Compute structural graph statistics using NetworkX."""
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_components": len(components),
        "largest_component": max(len(c) for c in components),
        "avg_clustering": nx.average_clustering(undirected),
        "density": nx.density(G),
    }

    print("\n  NetworkX Stats (sampled subgraph):")
    for k, v in stats.items():
        print(f"    {k:<25}: {v:.4g}" if isinstance(v, float) else f"    {k:<25}: {v}")

    return stats


# ─────────────────────────────────────────────
# DEGREE CENTRALITY PLOT
# ─────────────────────────────────────────────

def plot_degree_centrality(G: nx.DiGraph, data: Data, output_dir: str):
    """
    Scatter plot of betweenness centrality colored by fraud/legit label.
    Fraud hubs often appear as high-centrality outliers.
    """
    undirected = G.to_undirected()

    # Betweenness is expensive → use degree centrality as proxy for large graphs
    deg_centrality = nx.degree_centrality(undirected)

    nodes  = list(G.nodes())
    labels = [G.nodes[n].get("label", -1) for n in nodes]
    centralities = [deg_centrality[n] for n in nodes]

    colors = [PALETTE["fraud"] if l == 1 else (PALETTE["legit"] if l == 0 else PALETTE["unknown"])
              for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(range(len(nodes)), centralities, c=colors, alpha=0.5, s=15)
    ax.set_title("Degree Centrality by Node Label", fontsize=13, fontweight="bold")
    ax.set_xlabel("Node Index (sorted)")
    ax.set_ylabel("Degree Centrality")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["fraud"],   label="Fraud"),
        Patch(facecolor=PALETTE["legit"],   label="Legit"),
        Patch(facecolor=PALETTE["unknown"], label="Unknown"),
    ]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degree_centrality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: degree_centrality.png")


# ─────────────────────────────────────────────
# TEMPORAL ANALYSIS (Elliptic-specific)
# ─────────────────────────────────────────────

def plot_temporal_fraud_rate(data: Data, output_dir: str):
    """
    Plot fraud rate over time steps (Elliptic dataset has 49 time steps,
    each representing ~2 weeks of Bitcoin transaction data).
    """
    if not hasattr(data, "time_steps"):
        return

    time_steps = data.time_steps
    y = data.y.numpy()

    rows = []
    for ts in np.unique(time_steps):
        mask = time_steps == ts
        ys = y[mask]
        labelled = ys[ys >= 0]
        fraud    = (ys == 1).sum()
        total    = (ys >= 0).sum()
        rows.append({"time_step": ts, "fraud": fraud, "total": total,
                     "fraud_rate": fraud / total if total > 0 else 0})

    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Temporal Analysis by Time Step (Elliptic)", fontsize=13, fontweight="bold")

    ax1.bar(df["time_step"], df["total"], color=PALETTE["legit"], alpha=0.7, label="Total labelled")
    ax1.bar(df["time_step"], df["fraud"], color=PALETTE["fraud"], alpha=0.9, label="Fraud")
    ax1.set_ylabel("Node Count")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(df["time_step"], df["fraud_rate"] * 100,
             color=PALETTE["fraud"], marker="o", linewidth=2, markersize=5)
    ax2.fill_between(df["time_step"], df["fraud_rate"] * 100, alpha=0.2, color=PALETTE["fraud"])
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_xlabel("Time Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_fraud_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: temporal_fraud_rate.png")


# Fix missing import
from typing import Dict