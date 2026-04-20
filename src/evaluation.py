import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from pathlib import Path
import pickle 

from visualize import show_top_results
from attacks import AttackResult

df = pd.read_csv("attack_results.csv")
print(df.head())

# --------------------------------
# PRECISION - RECALL CURVE
# --------------------------------
y_true = df["extracted"].astype(int)
scores = -df["adaptive_score"]   # lower score = higher confidence

precision, recall, _ = precision_recall_curve(y_true, scores)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Extraction Attack)")
plt.grid()
plt.show()

# --------------------------------
# DUPLICATE HISTOGRAM
# --------------------------------
with open("duplicate_counts.pkl", "rb") as f:
    duplicate_counts = pickle.load(f)

values = list(duplicate_counts.values())

plt.figure()
plt.hist(values, bins=50)
plt.xlabel("Duplicate Count")
plt.ylabel("Frequency")
plt.title("Duplicate Distribution")
plt.grid()
plt.show()

# -------------------------------
# ROC CURVE
# -------------------------------
labels = df["extracted"].astype(int)
scores = -df["adaptive_score"]

fpr, tpr, _ = roc_curve(labels, scores)

plt.figure()
plt.plot(fpr, tpr)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve (Extraction Attack)")
plt.grid()
plt.show()

# -------------------------------
# TOP EXTRACTED IMAGES
# -------------------------------
# convert df → AttackResult objects (simplified)
results = []

for _, row in df.iterrows():
    results.append(
        AttackResult(
            query_path=Path(row["query_path"]),
            match_path=row["match_path"],
            l2_norm=row["l2_norm"],
            mean_clique_dist=row["mean_clique_dist"],
            adaptive_score=row["adaptive_score"],
            clique_size=row["clique_size"],
            extracted=row["extracted"],
        )
    )

show_top_results(results, n=9, only_extracted=True)
