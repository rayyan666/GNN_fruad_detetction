# run_pipeline.py
"""
End-to-end pipeline runner.

Usage:
    python run_pipeline.py                                    # full run with defaults
    python run_pipeline.py --model gat                        # use GAT
    python run_pipeline.py --skip_train                       # eval + viz only
    python run_pipeline.py --quick                            # 20 epochs, fast test
    python run_pipeline.py --local-features-only              # ablation: 94 features
"""

import os
import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from config import CFG
from utils.data_loader import load_dataset
from utils.graph_eda import run_eda
from models.gnn import build_model, get_device
from train import train
from evaluate import evaluate_model
from visualize import run_all_visualizations

import torch


def main():
    parser = argparse.ArgumentParser(description="Fraud GNN Pipeline")
    parser.add_argument("--model",       choices=["graphsage", "gat"], default="graphsage")
    parser.add_argument("--dataset",     choices=["elliptic", "ieee_cis"], default="elliptic")
    parser.add_argument("--strategy",    choices=["weighted_loss", "oversample", "both"],
                        default="weighted_loss")
    parser.add_argument("--epochs",      type=int,   default=150)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--layers",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=0.001)
    parser.add_argument("--skip_eda",    action="store_true", help="Skip EDA step")
    parser.add_argument("--skip_train",  action="store_true", help="Skip training (use existing checkpoint)")
    parser.add_argument("--quick",       action="store_true", help="Fast run: 20 epochs, smaller model")
    parser.add_argument("--output_dir",  default="outputs")
    parser.add_argument("--local-features-only", action="store_true",
                        help="Use only 94 local features (ablation: strips pre-aggregated neighborhood stats)")
    parser.add_argument("--no-wandb",    action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    # ── Apply CLI args to config ───────────────────────────────────────────────
    CFG.model.architecture       = args.model
    CFG.data.dataset             = args.dataset
    CFG.data.imbalance_strategy  = args.strategy
    CFG.model.hidden_channels    = args.hidden
    CFG.model.num_layers         = args.layers
    CFG.train.lr                 = args.lr
    CFG.train.epochs             = args.epochs
    CFG.train.checkpoint_dir     = args.output_dir
    CFG.data.local_features_only = getattr(args, "local_features_only", False)

    if args.quick:
        CFG.train.epochs          = 20
        CFG.model.hidden_channels = 64
        CFG.train.patience        = 10
        logger.info("Quick mode: 20 epochs, hidden=64")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── W&B initialisation ────────────────────────────────────────────────────
    # Build a descriptive run name so each experiment is identifiable
    feat_tag  = "local94" if CFG.data.local_features_only else "full165"
    run_name  = f"{args.model}_{args.strategy}_{feat_tag}_ep{CFG.train.epochs}"

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project = "fraud-gnn",
                name    = run_name,
                config  = {
                    "model"               : CFG.model.architecture,
                    "dataset"             : CFG.data.dataset,
                    "strategy"            : CFG.data.imbalance_strategy,
                    "epochs"              : CFG.train.epochs,
                    "hidden_channels"     : CFG.model.hidden_channels,
                    "num_layers"          : CFG.model.num_layers,
                    "dropout"             : CFG.model.dropout,
                    "lr"                  : CFG.train.lr,
                    "weight_decay"        : CFG.train.weight_decay,
                    "batch_size"          : CFG.train.batch_size,
                    "num_neighbors"       : CFG.train.num_neighbors,
                    "patience"            : CFG.train.patience,
                    "local_features_only" : CFG.data.local_features_only,
                    "seed"               : CFG.seed,
                },
            )
            logger.info(f"W&B run: {run_name}")
        except Exception as e:
            logger.warning(f"W&B init failed ({e}). Continuing without logging.")
            use_wandb = False

    # ── STEP 1: Load dataset ───────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  STEP 1 — Loading Dataset")
    logger.info("═"*60)
    data = load_dataset(CFG)

    if use_wandb:
        try:
            import wandb
            wandb.config.update({
                "num_nodes"   : data.num_nodes,
                "num_edges"   : data.edge_index.shape[1],
                "num_features": data.x.shape[1],
                "fraud_rate"  : float((data.y == 1).sum() / (data.y >= 0).sum()),
            })
        except Exception:
            pass

    # ── STEP 2 & 3: EDA ───────────────────────────────────────────────────────
    if not args.skip_eda:
        logger.info("\n" + "═"*60)
        logger.info("  STEP 3 — Graph EDA")
        logger.info("═"*60)
        run_eda(data, output_dir=args.output_dir)

    # ── STEP 4 & 5: Train ─────────────────────────────────────────────────────
    if not args.skip_train:
        logger.info("\n" + "═"*60)
        logger.info("  STEP 4 & 5 — Build Model + Train")
        logger.info("═"*60)
        model, data, history = train(CFG)

        # Log per-epoch metrics to W&B
        if use_wandb and history:
            try:
                import wandb
                for entry in history:
                    wandb.log({
                        "epoch"      : entry["epoch"],
                        "train_loss" : entry["train_loss"],
                        "val_f1"     : entry["f1"],
                        "val_precision": entry["precision"],
                        "val_recall" : entry["recall"],
                        "val_auc_roc": entry["auc_roc"],
                    })
            except Exception:
                pass
    else:
        checkpoint_path = os.path.join(args.output_dir, "best_model.pt")
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(CFG, checkpoint["in_channels"])
        model.load_state_dict(checkpoint["model_state_dict"])

    # ── STEP 6: Evaluate ──────────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  STEP 6 — Evaluation")
    logger.info("═"*60)
    results = evaluate_model(model, data, CFG, output_dir=args.output_dir)

    # Log test metrics to W&B
    if use_wandb and results:
        try:
            import wandb
            # Pull best-threshold metrics
            opt_key = [k for k in results if k.startswith("threshold_") and k != "threshold_0.5"]
            if opt_key:
                m = results[opt_key[0]]
                wandb.log({
                    "test_f1"        : m.get("f1", 0),
                    "test_precision" : m.get("precision", 0),
                    "test_recall"    : m.get("recall", 0),
                    "test_auc_roc"   : m.get("auc_roc", 0),
                    "test_auc_pr"    : m.get("auc_pr", 0),
                    "optimal_threshold": results.get("optimal_threshold", 0.5),
                })

            # Upload evaluation plots
            plot_files = ["roc_pr_curves.png", "confusion_matrix.png",
                          "threshold_sweep.png", "training_history.png"]
            for fname in plot_files:
                fpath = os.path.join(args.output_dir, fname)
                if os.path.exists(fpath):
                    wandb.log({fname.replace(".png", ""): wandb.Image(fpath)})
        except Exception:
            pass

    # ── STEP 7: Visualize ─────────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  STEP 7 — Visualizations")
    logger.info("═"*60)
    run_all_visualizations(model, data, CFG, output_dir=args.output_dir)

    # Log visualization plots to W&B
    if use_wandb:
        try:
            import wandb
            viz_files = ["fraud_subgraph.png", "embedding_tsne.png", "fraud_heatmap.png"]
            for fname in viz_files:
                fpath = os.path.join(args.output_dir, fname)
                if os.path.exists(fpath):
                    wandb.log({fname.replace(".png", ""): wandb.Image(fpath)})
            wandb.finish()
            logger.info("W&B run finished.")
        except Exception:
            pass

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    logger.info("\n" + "═"*60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("═"*60)
    logger.info(f"  Run name:    {run_name}")
    logger.info(f"  Model:       {CFG.model.architecture.upper()}")
    logger.info(f"  Dataset:     {CFG.data.dataset}")
    logger.info(f"  Strategy:    {CFG.data.imbalance_strategy}")
    logger.info(f"  Features:    {'local 94' if CFG.data.local_features_only else 'full 165'}")
    logger.info(f"  Outputs:     {args.output_dir}/")
    logger.info("\n  Generated files:")
    for fname in sorted(os.listdir(args.output_dir)):
        logger.info(f"    {fname}")


if __name__ == "__main__":
    main()