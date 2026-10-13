import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    ground_truth: List[str],
    predictions: List[str],
    labels: List[str]
) -> np.ndarray:
    """Computes square confusion matrix over the specified label list."""
    return confusion_matrix(ground_truth, predictions, labels=labels)


def find_top_confusions(
    cm: np.ndarray,
    labels: List[str],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Identifies the top misclassified pairs (excluding the diagonal)."""
    top_pairs = []
    n = len(labels)

    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                top_pairs.append({
                    "true_class": labels[i],
                    "predicted_class": labels[j],
                    "count": int(cm[i, j]),
                })

    top_pairs.sort(key=lambda x: x["count"], reverse=True)
    return top_pairs[:top_k]


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: str,
    title: str = "SAGE 15-Class Confusion Matrix"
) -> None:
    """Generates and saves a high-resolution confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add numeric annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, format(val, "d"),
                    ha="center", va="center",
                    color="white" if val > thresh else "black"
                )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
