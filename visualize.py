# visualize.py
"""
Step 7: Fraud subgraph visualization.

Three visualization modes:
  1. fraud_subgraph  — Ego-network around predicted fraud nodes,
                       colored by true label, sized by fraud probability
  2. embedding_tsne  — t-SNE of GNN node embeddings (fraud vs legit clusters)
  3. attention_map   — GAT attention weights on a sampled subgraph
                       (shows which neighbors the model focuses on)
  4. fraud_heatmap   — Node-level fraud probability heatmap on full graph
"""

import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx, k_hop_subgraph
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import CFG
from models.gnn import build_model, get_device
from utils.data_loader import load_dataset
from train import build_standard_loader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# Color palette
FRAUD_COLOR  = "#e63946"
LEGIT_COLOR  = "#457b9d"
UNK_COLOR    = "#adb5bd"
EDGE_COLOR   = "#dee2e6"


# ─────────────────────────────────────────────
# COLLECT FRAUD PROBABILITIES
# ─────────────────────────────────────────────

@torch.no_grad()
def get_all_fraud_probs(model, data, cfg, device):
    """
    Run inference on the full graph and return fraud probability for every node.
    Uses NeighborLoader to handle large graphs.
    """
    model.eval()
    N      = data.num_nodes
    probs  = torch.zeros(N)
    counts = torch.zeros(N)

    # We need predictions for all nodes (not just masked ones)
    all_mask = torch.ones(N, dtype=torch.bool)
    loader   = build_standard_loader(data, all_mask, cfg.train.num_neighbors,
                                     cfg.train.batch_size, shuffle=False)

    for batch in loader:
        batch  = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        seed_logits = logits[:batch.batch_size]
        seed_probs  = torch.softmax(seed_logits, dim=-1)[:, 1].cpu()

        # Map batch-local indices back to global node indices
        global_ids = batch.n_id[:batch.batch_size].cpu()
        probs[global_ids]  += seed_probs
        counts[global_ids] += 1

    # Average (some nodes may appear in multiple batches)
    probs = probs / counts.clamp(min=1)
    return probs.numpy()


# ─────────────────────────────────────────────
# 1. FRAUD EGO-SUBGRAPH
# ─────────────────────────────────────────────

def visualize_fraud_subgraph(
    model, data, cfg,
    n_fraud_seeds: int = 5,
    hops: int = 2,
    output_dir: str = "outputs",
):
    """
    Extract k-hop neighborhoods around the top-N highest-probability fraud nodes
    and visualize them as a NetworkX graph.

    Node aesthetics:
      - Color: red = true fraud, blue = true legit, gray = unknown
      - Size:  proportional to fraud probability
      - Border: thick if predicted fraud (prob > 0.5)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(cfg)
    model  = model.to(device)

    fraud_probs = get_all_fraud_probs(model, data, cfg, device)
    y = data.y.numpy()

    # Select top-N predicted fraud nodes as seeds (ignoring unknowns)
    labelled_fraud = np.where(y == 1)[0]
    if len(labelled_fraud) == 0:
        # Fall back to highest-probability nodes
        labelled_fraud = np.argsort(fraud_probs)[::-1][:n_fraud_seeds * 3]

    # Sort by fraud probability and take top seeds
    seed_probs = fraud_probs[labelled_fraud]
    sorted_idx = np.argsort(seed_probs)[::-1]
    seeds = labelled_fraud[sorted_idx[:n_fraud_seeds]].tolist()

    logger.info(f"Building {hops}-hop subgraph around {len(seeds)} fraud seeds...")

    # Extract k-hop subgraph using PyG
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx   = seeds,
        num_hops   = hops,
        edge_index = data.edge_index,
        num_nodes  = data.num_nodes,
        relabel_nodes = True,
    )

    subset_np = subset.numpy()
    n_sub     = len(subset_np)

    # Build NetworkX graph
    G = nx.DiGraph()
    for local_id, global_id in enumerate(subset_np):
        G.add_node(
            local_id,
            label      = int(y[global_id]),
            prob       = float(fraud_probs[global_id]),
            is_seed    = global_id in seeds,
        )

    for src, dst in sub_edge_index.T.numpy():
        G.add_edge(int(src), int(dst))

    logger.info(f"  Subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Layout ────────────────────────────────────────────────────────────────
    # Use spring layout; seed nodes are given larger initial repulsion
    pos = nx.spring_layout(G, k=2.0 / np.sqrt(n_sub), seed=42, iterations=50)

    # ── Visual properties ─────────────────────────────────────────────────────
    node_colors = []
    node_sizes  = []
    edge_widths = []

    for n in G.nodes():
        label = G.nodes[n]["label"]
        prob  = G.nodes[n]["prob"]
        color = FRAUD_COLOR if label == 1 else (LEGIT_COLOR if label == 0 else UNK_COLOR)
        node_colors.append(color)
        node_sizes.append(100 + prob * 800)   # Larger = more likely fraud

    for u, v in G.edges():
        # Highlight edges between two fraud nodes
        u_fraud = G.nodes[u]["prob"] > 0.5
        v_fraud = G.nodes[v]["prob"] > 0.5
        edge_widths.append(2.5 if (u_fraud and v_fraud) else 0.5)

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#f8f9fa")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR, width=0.4, alpha=0.6,
        arrows=True, arrowsize=8, arrowstyle="->",
    )

    # Draw high-fraud edges separately (more visible)
    fraud_edges = [(u, v) for u, v in G.edges() if
                   G.nodes[u]["prob"] > 0.5 and G.nodes[v]["prob"] > 0.5]
    if fraud_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=fraud_edges, ax=ax,
            edge_color=FRAUD_COLOR, width=2.0, alpha=0.7,
            arrows=True, arrowsize=12,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.85,
        edgecolors="#212529", linewidths=1.2,
    )

    # Mark seed nodes with a star
    seed_local = [mapping[i].item() for i, s in enumerate(seeds) if mapping[i].item() < n_sub]
    if seed_local:
        nx.draw_networkx_nodes(
            G, pos, nodelist=seed_local, ax=ax,
            node_shape="*", node_color=FRAUD_COLOR, node_size=600, alpha=1.0,
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend = [
        Patch(color=FRAUD_COLOR,   label="True Fraud"),
        Patch(color=LEGIT_COLOR,   label="True Legit"),
        Patch(color=UNK_COLOR,     label="Unknown"),
        Line2D([0], [0], marker="*", color=FRAUD_COLOR, markersize=12, label="Seed (top fraud)"),
        Line2D([0], [0], color=FRAUD_COLOR, linewidth=2, label="Fraud-Fraud edge"),
    ]
    ax.legend(handles=legend, loc="upper left", framealpha=0.9, fontsize=9)
    ax.set_title(
        f"{hops}-Hop Fraud Subgraph | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}\n"
        f"Node size ∝ fraud probability | ★ = seed fraud nodes",
        fontsize=12, fontweight="bold", pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "fraud_subgraph.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: fraud_subgraph.png")


# ─────────────────────────────────────────────
# 2. t-SNE OF NODE EMBEDDINGS
# ─────────────────────────────────────────────

@torch.no_grad()
def visualize_embeddings_tsne(
    model, data, cfg,
    max_nodes: int = 3000,
    output_dir: str = "outputs",
):
    """
    Reduce GNN node embeddings to 2D via t-SNE and scatter-plot by label.

    Well-trained GNN: fraud and legit nodes form clearly separated clusters.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(cfg)
    model  = model.to(device)
    model.eval()

    # Collect embeddings on a sample of labelled nodes
    y = data.y.numpy()
    labelled_idx = np.where(y >= 0)[0]

    # Subsample if too many
    if len(labelled_idx) > max_nodes:
        # Stratified subsample
        fraud_idx = labelled_idx[y[labelled_idx] == 1]
        legit_idx = labelled_idx[y[labelled_idx] == 0]
        n_fraud = min(len(fraud_idx), max_nodes // 2)
        n_legit = min(len(legit_idx), max_nodes - n_fraud)
        sub_idx = np.concatenate([
            np.random.choice(fraud_idx, n_fraud, replace=False),
            np.random.choice(legit_idx, n_legit, replace=False),
        ])
    else:
        sub_idx = labelled_idx

    # Build a subgraph loader for these nodes
    sub_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    sub_mask[sub_idx] = True
    loader = build_standard_loader(data, sub_mask, cfg.train.num_neighbors,
                                   min(cfg.train.batch_size, 256), shuffle=False)

    embeddings  = []
    node_labels = []

    for batch in loader:
        batch = batch.to(device)
        emb   = model.get_embeddings(batch.x, batch.edge_index)
        seed_emb    = emb[:batch.batch_size].cpu().numpy()
        seed_labels = batch.y[:batch.batch_size].cpu().numpy()
        labelled    = seed_labels >= 0
        embeddings.append(seed_emb[labelled])
        node_labels.extend(seed_labels[labelled].tolist())

    embeddings  = np.vstack(embeddings)
    node_labels = np.array(node_labels)

    logger.info(f"Running t-SNE on {len(embeddings)} node embeddings...")

    tsne  = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    emb2d = tsne.fit_transform(embeddings)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    for label, color, name in [(0, LEGIT_COLOR, "Legit"), (1, FRAUD_COLOR, "Fraud")]:
        mask = node_labels == label
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1],
                   c=color, alpha=0.5, s=15 if label == 0 else 40,
                   label=f"{name} (n={mask.sum():,})",
                   edgecolors="none" if label == 0 else "#212529",
                   linewidths=0.5)

    ax.set_title("t-SNE of GNN Node Embeddings", fontsize=13, fontweight="bold")
    ax.legend(markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_tsne.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: embedding_tsne.png")


# ─────────────────────────────────────────────
# 3. FRAUD PROBABILITY HEATMAP
# ─────────────────────────────────────────────

def visualize_fraud_heatmap(
    model, data, cfg,
    output_dir: str = "outputs",
    max_nodes: int = 1000,
):
    """
    Adjacency-sorted heatmap of fraud probabilities.

    Nodes are sorted by their predicted fraud probability.
    True labels are shown as a color strip on the side.
    Helps reveal whether high-probability fraud nodes cluster together.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = get_device(cfg)
    model  = model.to(device)

    fraud_probs = get_all_fraud_probs(model, data, cfg, device)
    y = data.y.numpy()

    # Focus on labelled nodes
    labelled = np.where(y >= 0)[0]
    if len(labelled) > max_nodes:
        labelled = np.random.choice(labelled, max_nodes, replace=False)

    sorted_idx = labelled[np.argsort(fraud_probs[labelled])[::-1]]
    sorted_probs  = fraud_probs[sorted_idx]
    sorted_labels = y[sorted_idx]

    fig, (ax_prob, ax_label) = plt.subplots(
        1, 2, figsize=(14, max(4, len(sorted_idx) // 50)),
        gridspec_kw={"width_ratios": [15, 1]},
    )
    fig.suptitle("Node Fraud Probability (sorted)", fontsize=13, fontweight="bold")

    # Horizontal bar chart of fraud probs
    colors = [FRAUD_COLOR if l == 1 else LEGIT_COLOR for l in sorted_labels]
    ax_prob.barh(range(len(sorted_probs)), sorted_probs, color=colors, alpha=0.8, height=1.0)
    ax_prob.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Threshold 0.5")
    ax_prob.set_xlabel("Fraud Probability")
    ax_prob.set_ylabel("Node (sorted by P(fraud))")
    ax_prob.set_xlim([0, 1])
    ax_prob.grid(True, axis="x", alpha=0.3)
    ax_prob.legend()

    # True label color strip
    label_colors = np.array([
        mcolors.to_rgba(FRAUD_COLOR) if l == 1 else mcolors.to_rgba(LEGIT_COLOR)
        for l in sorted_labels
    ]).reshape(-1, 1, 4)
    ax_label.imshow(label_colors, aspect="auto")
    ax_label.set_xticks([0])
    ax_label.set_xticklabels(["True\nLabel"], fontsize=8)
    ax_label.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fraud_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: fraud_heatmap.png")


# ─────────────────────────────────────────────
# MAIN: RUN ALL VISUALIZATIONS
# ─────────────────────────────────────────────

def run_all_visualizations(model, data, cfg, output_dir: str = "outputs"):
    """Run all visualization modules."""
    logger.info("\nGenerating visualizations...")

    visualize_fraud_subgraph(model, data, cfg, output_dir=output_dir)
    visualize_embeddings_tsne(model, data, cfg, output_dir=output_dir)
    visualize_fraud_heatmap(model, data, cfg, output_dir=output_dir)

    # ── W&B: log visualization plots ──────────────────────────────────────────
    import wandb
    wandb.log({
        "fraud_subgraph": wandb.Image(os.path.join(output_dir, "fraud_subgraph.png")),
        "embedding_tsne": wandb.Image(os.path.join(output_dir, "embedding_tsne.png")),
        "fraud_heatmap":  wandb.Image(os.path.join(output_dir, "fraud_heatmap.png")),
    })

    logger.info(f"\nAll visualizations saved to: {output_dir}/")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--subgraph",   action="store_true")
    parser.add_argument("--tsne",       action="store_true")
    parser.add_argument("--heatmap",    action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg   = checkpoint.get("config", CFG)
    data  = load_dataset(cfg)
    model = build_model(cfg, checkpoint["in_channels"])
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.subgraph or not any([args.subgraph, args.tsne, args.heatmap]):
        visualize_fraud_subgraph(model, data, cfg, output_dir=args.output_dir)
    if args.tsne:
        visualize_embeddings_tsne(model, data, cfg, output_dir=args.output_dir)
    if args.heatmap:
        visualize_fraud_heatmap(model, data, cfg, output_dir=args.output_dir)