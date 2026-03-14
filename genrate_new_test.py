"""
Generates a small CSV of fake new transactions for testing predict.py.
Produces 20 rows: 17 legit-like, 3 fraud-like.

Usage:
    python generate_test_transactions.py
Output:
    data/raw/test_new_transactions.csv
"""
import os
import csv
import random
import math

# NOTE: intentionally avoids numpy import to prevent shadowing issues.
# Uses only stdlib random + math.

random.seed(99)
F = 166
os.makedirs("data/raw", exist_ok=True)

def randn():
    """Box-Muller standard normal sample."""
    u1 = random.random() or 1e-10
    u2 = random.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

rows = []

# 17 legit-like: features near zero (standard normal)
for i in range(17):
    feats = [round(randn(), 6) for _ in range(F)]
    rows.append([f"new_tx_{i:04d}"] + feats)

# 3 fraud-like: features shifted to extremes (mean=3, std=2.5)
for i in range(3):
    feats = [round(randn() * 2.5 + 3.0, 6) for _ in range(F)]
    rows.append([f"new_tx_FRAUD_{i:04d}"] + feats)

path = os.path.join("data", "raw", "test_new_transactions.csv")
header = ["txId"] + [f"feat_{i}" for i in range(F)]

with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Saved {len(rows)} test transactions to {path}")
print(f"  - 17 legit-like  (feat mean ~0)")
print(f"  -  3 fraud-like  (feat mean ~3, shifted distribution)")