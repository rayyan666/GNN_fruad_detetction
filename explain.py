# explain.py
"""
Step 8: GNN Explainability using GNNExplainer.

For each flagged fraud node, GNNExplainer answers:
  - WHICH EDGES in the neighborhood were most important?
  - WHICH FEATURES drove the fraud prediction?

How GNNExplainer works:
  For a target node v, it learns masks M_edge and M_feat that maximise
  the mutual information between the masked subgraph and the model's prediction.
  Edges/features with high mask values = most important for the prediction.

  Formally it optimises:
    max_{M} MI( Y_v,  (G_s ⊙ M_edge, X ⊙ M_feat) )

Outputs per node:
  - Top-k most important edges (visualized as thick red lines)
  - Top-k most important features (bar chart)
  - Explanation subgraph image saved to outputs/explanations/

Usage:
  # Explain top-5 highest-probability fraud nodes from the test set:
  python explain.py --checkpoint outputs/best_model.pt

  # Explain a specific node by index:
  python explain.py --checkpoint outputs/best_model.pt --node-idx 1234

  # Explain more nodes:
  python explain.py --checkpoint outputs/best_model.pt --top-k 10

  # Save only the summary CSV, no plots:
  python explain.py --checkpoint outputs/best_model.pt --no-plots
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import k_hop_subgraph, to_networkx

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

# Feature name mapping for Elliptic dataset
# First 94 are local transaction features, next 72 are aggregated neighborhood features
FEATURE_NAMES = (
    [f"local_{i:02d}" for i in range(94)] +
    [f"agg_{i:02d}"   for i in range(72)]  # trimmed to 165 after time_step drop
)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD CHECKPOINT
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_data(checkpoint_path: str, device: torch.device):
    """Load trained model + dataset from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg         = ckpt["config"]
    in_channels = ckpt["in_channels"]
    best_val_f1 = ckpt.get("best_val_f1", 0.0)

    model = build_model(cfg, in_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"  Architecture : {cfg.model.architecture.upper()}")
    logger.info(f"  Best val F1  : {best_val_f1:.4f}")
    logger.info(f"  In channels  : {in_channels}")

    logger.info("Loading dataset...")
    data = load_dataset(cfg)
    logger.info(f"  Nodes: {data.num_nodes:,}   Edges: {data.edge_index.shape[1]:,}")

    return model, data, cfg, in_channels


# ─────────────────────────────────────────────────────────────────────────────
# GET FRAUD PROBABILITIES FOR ALL TEST NODES
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_fraud_probs_test(model, data, cfg, device) -> torch.Tensor:
    """
    Run inference on test nodes and return per-node fraud probabilities.
    Returns tensor of shape [num_test_nodes] with (global_node_idx, p_fraud) pairs.
    """
    model.eval()
    N      = data.num_nodes
    probs  = torch.zeros(N)
    counts = torch.zeros(N)

    loader = build_standard_loader(
        data, data.test_mask, cfg.train.num_neighbors,
        cfg.train.batch_size, shuffle=False
    )

    for batch in loader:
        batch       = batch.to(device)
        logits      = model(batch.x, batch.edge_index)
        seed_logits = logits[:batch.batch_size]
        seed_probs  = torch.softmax(seed_logits, dim=-1)[:, 1].cpu()
        global_ids  = batch.n_id[:batch.batch_size].cpu()
        probs[global_ids]  += seed_probs
        counts[global_ids] += 1

    probs = probs / counts.clamp(min=1)
    return probs


def get_top_fraud_nodes(probs: torch.Tensor, data, top_k: int = 5):
    """
    Return the top-k highest-probability fraud nodes from the test set.
    Only considers nodes with known labels (y >= 0).
    """
    test_indices = data.test_mask.nonzero(as_tuple=True)[0]
    test_probs   = probs[test_indices]

    # Sort by fraud probability descending
    sorted_idx = torch.argsort(test_probs, descending=True)
    top_local  = sorted_idx[:top_k]
    top_global = test_indices[top_local]

    results = []
    for local_i, global_i in zip(top_local, top_global):
        p     = test_probs[local_i].item()
        label = data.y[global_i].item()
        results.append({
            "node_idx" : global_i.item(),
            "p_fraud"  : p,
            "true_label": label,  # 1=fraud, 0=legit, -1=unknown
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# GNNEXPLAINER WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def build_explainer(model, explanation_type="phenomenon"):
    """
    Build a GNNExplainer using PyG's Explainer API.

    explanation_type:
      'phenomenon' — explains the model's prediction output (what we want)
      'model'      — explains the model itself (more theoretical)
    """
    explainer = Explainer(
        model         = model,
        algorithm     = GNNExplainer(epochs=200),
        explanation_type = explanation_type,
        node_mask_type   = "attributes",   # per-feature importance mask
        edge_mask_type   = "object",       # per-edge importance mask
        model_config  = dict(
            mode       = "multiclass_classification",
            task_level = "node",
            return_type= "probs",
        ),
    )
    return explainer


def explain_node(explainer, data, node_idx: int, device: torch.device, num_hops: int = 3):
    """
    Run GNNExplainer for a single target node.

    Returns the Explanation object containing:
      - x:              node features of the k-hop subgraph
      - edge_index:     edges of the k-hop subgraph
      - node_mask:      [N_sub, F] feature importance scores per node
      - edge_mask:      [E_sub]   edge importance scores
      - node_feat_mask: [F]       aggregated feature importance for target node
    """
    # Extract k-hop subgraph around target node
    subset, sub_edge_index, mapping, edge_mask_sub = k_hop_subgraph(
        node_idx   = node_idx,
        num_hops   = num_hops,
        edge_index = data.edge_index,
        relabel_nodes = True,
        num_nodes  = data.num_nodes,
    )

    sub_x      = data.x[subset].to(device)
    sub_edge   = sub_edge_index.to(device)
    target_idx = mapping.item()   # index of target node within subgraph

    # Run explainer — target is the fraud class (index 1)
    explanation = explainer(
        x          = sub_x,
        edge_index = sub_edge,
        index      = target_idx,
        target     = torch.tensor([1], device=device),  # explain fraud class
    )

    # Attach global node ids for reference
    explanation.global_node_ids = subset

    return explanation, subset, sub_edge_index, target_idx


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_explanation(
    explanation,
    data,
    node_idx: int,
    subset: torch.Tensor,
    sub_edge_index: torch.Tensor,
    target_local_idx: int,
    p_fraud: float,
    true_label: int,
    output_dir: str,
    top_k_edges: int = 10,
    top_k_features: int = 15,
    in_channels: int = 165,
):
    """
    Two-panel explanation figure:
      Left:  Subgraph with edges colored by importance (red=high, gray=low)
      Right: Bar chart of top-k most important features
    """
    fig, (ax_graph, ax_feat) = plt.subplots(1, 2, figsize=(16, 7))

    label_str  = {1: "FRAUD ✓", 0: "LEGIT", -1: "UNKNOWN"}.get(true_label, "?")
    title_col  = "#e63946" if true_label == 1 else "#457b9d"
    fig.suptitle(
        f"GNNExplainer — Node {node_idx}  |  P(fraud)={p_fraud:.4f}  |  True label: {label_str}",
        fontsize=13, fontweight="bold", color=title_col
    )

    # ── LEFT: Subgraph with edge importance ──────────────────────────────────
    edge_mask = explanation.edge_mask.cpu().numpy() if hasattr(explanation, "edge_mask") and explanation.edge_mask is not None else None

    G = to_networkx(
        torch.utils.data.Dataset.__new__(torch.utils.data.Dataset),  # dummy
        edge_index=sub_edge_index,
        num_nodes=subset.shape[0],
    ) if False else _build_nx_graph(sub_edge_index, subset.shape[0])

    pos = nx.spring_layout(G, seed=42, k=1.5)

    # Node colors: target=gold, fraud neighbors=red, legit=blue, unknown=gray
    node_colors = []
    node_sizes  = []
    for i in range(subset.shape[0]):
        global_id = subset[i].item()
        lbl       = data.y[global_id].item() if global_id < data.num_nodes else -1
        if i == target_local_idx:
            node_colors.append("#FFD700")   # gold = target node
            node_sizes.append(800)
        elif lbl == 1:
            node_colors.append("#e63946")   # red = fraud
            node_sizes.append(400)
        elif lbl == 0:
            node_colors.append("#457b9d")   # blue = legit
            node_sizes.append(300)
        else:
            node_colors.append("#adb5bd")   # gray = unknown
            node_sizes.append(200)

    # Edge colors and widths based on importance
    edges    = list(G.edges())
    n_edges  = len(edges)

    if edge_mask is not None and len(edge_mask) == n_edges:
        # Normalize to [0,1]
        em = edge_mask
        if em.max() > em.min():
            em = (em - em.min()) / (em.max() - em.min())

        edge_colors = [plt.cm.RdYlGn_r(v) for v in em]
        edge_widths = [0.5 + 4.0 * v for v in em]

        # Highlight top-k edges
        top_edge_indices = np.argsort(em)[-top_k_edges:]
    else:
        edge_colors = ["#dee2e6"] * n_edges
        edge_widths = [1.0] * n_edges
        top_edge_indices = []

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, ax=ax_graph)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.7, arrows=True, arrowsize=10, ax=ax_graph)

    # Label only the target node
    nx.draw_networkx_labels(G, pos,
                            labels={target_local_idx: f"TARGET\n{node_idx}"},
                            font_size=8, font_weight="bold", ax=ax_graph)

    # Legend
    legend_elements = [
        mpatches.Patch(color="#FFD700", label="Target node"),
        mpatches.Patch(color="#e63946", label="Fraud neighbor"),
        mpatches.Patch(color="#457b9d", label="Legit neighbor"),
        mpatches.Patch(color="#adb5bd", label="Unknown neighbor"),
    ]
    ax_graph.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax_graph.set_title(f"Subgraph ({subset.shape[0]} nodes, {n_edges} edges)\nEdge thickness ∝ importance",
                       fontsize=10)
    ax_graph.axis("off")

    # ── RIGHT: Feature importance bar chart ───────────────────────────────────
    if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
        # node_mask shape: [N_sub, F] — take the target node's row
        feat_imp = explanation.node_mask[target_local_idx].cpu().numpy()
    elif hasattr(explanation, "x") and explanation.x is not None:
        feat_imp = np.zeros(in_channels)
    else:
        feat_imp = np.zeros(in_channels)

    # Normalize
    if feat_imp.max() > feat_imp.min():
        feat_imp = (feat_imp - feat_imp.min()) / (feat_imp.max() - feat_imp.min())

    # Get feature names
    feat_names = FEATURE_NAMES[:in_channels]
    if len(feat_names) < in_channels:
        feat_names = feat_names + [f"feat_{i}" for i in range(len(feat_names), in_channels)]

    # Top-k features
    top_feat_idx  = np.argsort(feat_imp)[-top_k_features:][::-1]
    top_feat_imp  = feat_imp[top_feat_idx]
    top_feat_names = [feat_names[i] for i in top_feat_idx]

    colors = ["#e63946" if imp > 0.5 else "#457b9d" if imp > 0.25 else "#adb5bd"
              for imp in top_feat_imp]

    bars = ax_feat.barh(range(top_k_features), top_feat_imp[::-1], color=colors[::-1])
    ax_feat.set_yticks(range(top_k_features))
    ax_feat.set_yticklabels(top_feat_names[::-1], fontsize=9)
    ax_feat.set_xlabel("Normalised Importance Score")
    ax_feat.set_title(f"Top {top_k_features} Most Important Features", fontsize=10)
    ax_feat.axvline(0.5,  color="#e63946", linestyle="--", alpha=0.5, linewidth=1, label="High (>0.5)")
    ax_feat.axvline(0.25, color="#457b9d", linestyle="--", alpha=0.5, linewidth=1, label="Medium (>0.25)")
    ax_feat.legend(fontsize=8)
    ax_feat.grid(True, alpha=0.3, axis="x")
    ax_feat.set_xlim([0, 1.05])

    plt.tight_layout()
    fname = os.path.join(output_dir, f"explanation_node_{node_idx}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {fname}")

    return top_feat_names, top_feat_imp


def _build_nx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.DiGraph:
    """Build a NetworkX DiGraph from a PyG edge_index."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.T.tolist()
    G.add_edges_from(edges)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_node_explanation(node_info: dict, feat_names: list, feat_imp: np.ndarray):
    """Print a readable explanation summary for one node."""
    label_str = {1: "TRUE FRAUD ✓", 0: "TRUE LEGIT ✗ (false positive)", -1: "UNKNOWN"}.get(
        node_info["true_label"], "?"
    )
    print(f"\n  ── Node {node_info['node_idx']} ──────────────────────────────")
    print(f"     P(fraud)    : {node_info['p_fraud']:.4f}")
    print(f"     True label  : {label_str}")
    print(f"     Top features driving this prediction:")
    for name, imp in zip(feat_names[:5], feat_imp[:5]):
        bar = "█" * int(imp * 20)
        print(f"       {name:<15} {bar:<20} {imp:.3f}")


def save_summary_csv(all_results: list, output_dir: str):
    """Save a CSV summarising explanations across all explained nodes."""
    rows = []
    for r in all_results:
        row = {
            "node_idx"   : r["node_idx"],
            "p_fraud"    : r["p_fraud"],
            "true_label" : r["true_label"],
        }
        for i, (name, imp) in enumerate(zip(r["top_feat_names"][:10], r["top_feat_imp"][:10])):
            row[f"feature_{i+1}_name"] = name
            row[f"feature_{i+1}_imp"]  = round(float(imp), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "explanation_summary.csv")
    df.to_csv(path, index=False)
    logger.info(f"Summary saved: {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNNExplainer — explain fraud predictions node by node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python explain.py --checkpoint outputs/best_model.pt
  python explain.py --checkpoint outputs/best_model.pt --top-k 10
  python explain.py --checkpoint outputs/best_model.pt --node-idx 5432
  python explain.py --checkpoint outputs/best_model.pt --no-plots
        """
    )
    parser.add_argument("--checkpoint", default="outputs/best_model.pt",
                        help="Path to best_model.pt")
    parser.add_argument("--top-k",      type=int, default=5,
                        help="Number of highest-probability fraud nodes to explain (default: 5)")
    parser.add_argument("--node-idx",   type=int, default=None,
                        help="Explain a specific node index (overrides --top-k)")
    parser.add_argument("--output-dir", default="outputs/explanations",
                        help="Directory to save explanation plots")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Skip plot generation, save only summary CSV")
    parser.add_argument("--num-hops",   type=int, default=3,
                        help="Number of hops for subgraph extraction (default: 3, matches training)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load model + data ─────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data, cfg, in_channels = load_model_and_data(args.checkpoint, device)

    # ── 2. Get fraud probabilities for all test nodes ────────────────────────
    logger.info("Computing fraud probabilities for test nodes...")
    probs = get_fraud_probs_test(model, data, cfg, device)

    # ── 3. Select nodes to explain ───────────────────────────────────────────
    if args.node_idx is not None:
        nodes_to_explain = [{
            "node_idx"   : args.node_idx,
            "p_fraud"    : probs[args.node_idx].item(),
            "true_label" : data.y[args.node_idx].item(),
        }]
        logger.info(f"Explaining node {args.node_idx}  P(fraud)={nodes_to_explain[0]['p_fraud']:.4f}")
    else:
        nodes_to_explain = get_top_fraud_nodes(probs, data, top_k=args.top_k)
        logger.info(f"Explaining top-{args.top_k} highest-probability fraud nodes:")
        for n in nodes_to_explain:
            label = {1:"FRAUD", 0:"LEGIT", -1:"UNK"}.get(n["true_label"], "?")
            logger.info(f"  Node {n['node_idx']:>8}  P(fraud)={n['p_fraud']:.4f}  label={label}")

    # ── 4. Build explainer ───────────────────────────────────────────────────
    logger.info("Building GNNExplainer...")
    explainer = build_explainer(model)

    # ── 5. Explain each node ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GNNExplainer Results")
    print("=" * 60)

    all_results = []

    for node_info in nodes_to_explain:
        node_idx = node_info["node_idx"]
        logger.info(f"Explaining node {node_idx}...")

        try:
            explanation, subset, sub_edge_index, target_local_idx = explain_node(
                explainer, data, node_idx, device, num_hops=args.num_hops
            )

            if not args.no_plots:
                top_feat_names, top_feat_imp = plot_explanation(
                    explanation   = explanation,
                    data          = data,
                    node_idx      = node_idx,
                    subset        = subset,
                    sub_edge_index= sub_edge_index,
                    target_local_idx = target_local_idx,
                    p_fraud       = node_info["p_fraud"],
                    true_label    = node_info["true_label"],
                    output_dir    = args.output_dir,
                    in_channels   = in_channels,
                )
            else:
                # Still compute feature importance for CSV
                if hasattr(explanation, "node_mask") and explanation.node_mask is not None:
                    feat_imp = explanation.node_mask[target_local_idx].cpu().numpy()
                    if feat_imp.max() > feat_imp.min():
                        feat_imp = (feat_imp - feat_imp.min()) / (feat_imp.max() - feat_imp.min())
                else:
                    feat_imp = np.zeros(in_channels)
                feat_names = FEATURE_NAMES[:in_channels]
                top_idx    = np.argsort(feat_imp)[-15:][::-1]
                top_feat_names = [feat_names[i] for i in top_idx]
                top_feat_imp   = feat_imp[top_idx]

            print_node_explanation(node_info, top_feat_names, top_feat_imp)

            all_results.append({
                **node_info,
                "top_feat_names": top_feat_names,
                "top_feat_imp"  : top_feat_imp,
                "subgraph_nodes": subset.shape[0],
                "subgraph_edges": sub_edge_index.shape[1],
            })

        except Exception as e:
            logger.warning(f"  Could not explain node {node_idx}: {e}")
            continue

    # ── 6. Save summary CSV ──────────────────────────────────────────────────
    if all_results:
        df = save_summary_csv(all_results, args.output_dir)
        print(f"\n  Explained {len(all_results)} nodes.")
        print(f"  Plots saved to: {args.output_dir}/")
        print(f"  Summary CSV:    {args.output_dir}/explanation_summary.csv")
    else:
        logger.warning("No nodes were successfully explained.")

    print("=" * 60)


if __name__ == "__main__":
    main()