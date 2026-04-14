"""
utils/metrics.py
=================
Precision / Recall / F1-Score / Accuracy — Paper Equations 16-19.
"""

from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report,
)

from utils.config import CFG


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = list(CFG.class_names),
) -> Dict[str, float]:
    """
    Compute all four paper metrics (macro-averaged across 7 classes).

    Paper Eqs 16-19:
        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1-Score  = 2 * (P * R) / (P + R)
        Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    """
    return {
        "precision": precision_score(y_true, y_pred, average="macro",
                                     zero_division=0) * 100,
        "recall":    recall_score(   y_true, y_pred, average="macro",
                                     zero_division=0) * 100,
        "f1":        f1_score(       y_true, y_pred, average="macro",
                                     zero_division=0) * 100,
        "accuracy":  accuracy_score( y_true, y_pred) * 100,
    }


def print_metrics(metrics: Dict[str, float], epoch: int = None, prefix: str = ""):
    tag = f"Epoch {epoch} " if epoch is not None else ""
    print(f"  {prefix}{tag}"
          f"Acc={metrics['accuracy']:.2f}%  "
          f"Prec={metrics['precision']:.2f}%  "
          f"Rec={metrics['recall']:.2f}%  "
          f"F1={metrics['f1']:.2f}%")


def full_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = list(CFG.class_names),
) -> str:
    return classification_report(y_true, y_pred, target_names=class_names,
                                 digits=4, zero_division=0)


class MetricsTracker:
    """
    Accumulates batch-level predictions and computes epoch-level metrics.
    Usage:
        tracker = MetricsTracker()
        for batch in loader:
            ...
            tracker.update(preds, labels)
        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self):
        self.preds:  List[int] = []
        self.labels: List[int] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        self.preds  += preds
        self.labels += labels.cpu().numpy().tolist()

    def compute(self) -> Dict[str, float]:
        return compute_metrics(
            np.array(self.labels),
            np.array(self.preds),
        )

    def reset(self):
        self.preds  = []
        self.labels = []

    @property
    def all_preds(self) -> np.ndarray:
        return np.array(self.preds)

    @property
    def all_labels(self) -> np.ndarray:
        return np.array(self.labels)
