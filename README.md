# Fraud Detection with Graph Neural Networks

A production-grade GNN pipeline for detecting fraud in transaction networks using PyTorch Geometric.

## Thesis paper
[https://docs.google.com/document/d/1bgi7PdoimHH-Y3SivMdgRmcxchl2HZth/edit?usp=sharing&ouid=117291735169605961484&rtpof=true&sd=true](https://docs.google.com/document/d/1bgi7PdoimHH-Y3SivMdgRmcxchl2HZth/edit?usp=sharing&ouid=117291735169605961484&rtpof=true&sd=true)

## Architecture
- **Dataset**: Elliptic Bitcoin Dataset (or IEEE-CIS via adapter)
- **Model**: GraphSAGE or GAT (switchable via config)
- **Graph**: Users/entities as nodes, transactions as edges
- **Handles**: Severe class imbalance via weighted loss + SMOTE-style oversampling

## Project Structure
```
fraud_gnn/
├── data/               # Raw and processed datasets
├── models/             # GNN model definitions
├── utils/              # Helpers: loaders, metrics, visualization
├── outputs/            # Saved models, plots, reports
├── config.py           # Central config
├── train.py            # Training pipeline
├── evaluate.py         # Evaluation + reporting
└── visualize.py        # Fraud subgraph visualization
```

## Quickstart
```bash
pip install torch torch-geometric networkx scikit-learn matplotlib seaborn pandas
python train.py --model graphsage --epochs 100
python evaluate.py --checkpoint outputs/best_model.pt
python visualize.py --subgraph fraud
```