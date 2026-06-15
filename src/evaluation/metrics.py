# Skill source: (none) – standard metrics from sklearn, ICBHI score computed manually
# This file implements evaluation metrics for the ICBHI 2017 lung sound classification task.
# We compute per-class sensitivity (recall), specificity, ICBHI official score,
# confusion matrix, macro/micro F1, and optionally ROC-AUC.

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)

from typing import Dict, Optional, List, Union


def compute_per_class_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute sensitivity (recall) for each class.

    Sensitivity = TP / (TP + FN)

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        labels: List of class labels to compute metrics for.
                Defaults to all unique values in y_true.

    Returns:
        Dictionary mapping class label to sensitivity value.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    sensitivities = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(sens) for label, sens in zip(labels, sensitivities)}


def compute_per_class_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[int, float]:
    """
    Compute specificity for each class.

    For each class i, we treat it as positive and all others as negative.
    Specificity = TN / (TN + FP)

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        labels: List of class labels. Defaults to sorted unique values in y_true.

    Returns:
        Dictionary mapping class label to specificity.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    # multilabel_confusion_matrix gives a 2x2 matrix per class:
    # [[TN, FP], [FN, TP]]
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    specificities = {}
    for i, label in enumerate(labels):
        tn = mcm[i, 0, 0]
        fp = mcm[i, 0, 1]
        denom = tn + fp
        specificities[label] = float(tn / denom) if denom > 0 else 0.0
    return specificities


def compute_icbhi_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> float:
    """
    Compute the official ICBHI score.

    The score is the average over all classes of (Sensitivity + Specificity) / 2.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        labels: List of class labels.

    Returns:
        ICBHI score (float).
    """
    sens = compute_per_class_sensitivity(y_true, y_pred, labels)
    spec = compute_per_class_specificity(y_true, y_pred, labels)
    if labels is None:
        labels = sorted(sens.keys())
    # Average of (Se+Sp)/2 per class
    per_class_scores = [(sens[label] + spec[label]) / 2.0 for label in labels]
    return float(np.mean(per_class_scores))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        labels: List of class labels. Order determines axis order.

    Returns:
        2D numpy array of shape (n_classes, n_classes).
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    return confusion_matrix(y_true, y_pred, labels=labels)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """
    Macro-averaged F1 score (unweighted average of per-class F1 scores).

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        labels: Class labels.

    Returns:
        Macro F1 (float).
    """
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def compute_micro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> float:
    """
    Micro-averaged F1 score (global counts of TP, FP, FN).

    Args:
        y_true: Ground truth.
        y_pred: Predictions.
        labels: Class labels.

    Returns:
        Micro F1 (float).
    """
    return f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)


def compute_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Optional[List[int]] = None,
    average: str = "macro",
    multi_class: str = "ovr",
) -> float:
    """
    Compute ROC AUC for multiclass classification.
    Requires predicted probabilities for each class.

    Args:
        y_true: Ground truth class indices.
        y_score: Array of shape (n_samples, n_classes) with predicted probabilities.
        labels: List of class labels.
        average: Averaging method ('macro', 'weighted', None).
        multi_class: How to handle multiclass ('ovr', 'ovo').

    Returns:
        ROC AUC score (float).
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())
    # Binarize y_true for multiclass ROC AUC
    return roc_auc_score(
        y_true,
        y_score,
        labels=labels,
        average=average,
        multi_class=multi_class,
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, Union[float, Dict[int, float], np.ndarray, None]]:
    """
    Compute all relevant evaluation metrics.

    Args:
        y_true: 1D array of ground truth class indices.
        y_pred: 1D array of predicted class indices.
        y_score: 2D array (n_samples, n_classes) of class probabilities, optional for ROC AUC.
        labels: List of class labels. If None, all unique values in y_true are used.

    Returns:
        Dictionary containing:
            'accuracy': overall accuracy (for reference only),
            'per_class_sensitivity': dict label -> sensitivity,
            'per_class_specificity': dict label -> specificity,
            'icbhi_score': ICBHI official score,
            'confusion_matrix': numpy array,
            'macro_f1': float,
            'micro_f1': float,
            'roc_auc': ROC AUC if y_score provided, else None.
    """
    if labels is None:
        labels = sorted(np.unique(y_true).tolist())

    acc = accuracy_score(y_true, y_pred)
    sens = compute_per_class_sensitivity(y_true, y_pred, labels)
    spec = compute_per_class_specificity(y_true, y_pred, labels)
    icbhi = compute_icbhi_score(y_true, y_pred, labels)
    cm = compute_confusion_matrix(y_true, y_pred, labels)
    macro_f1 = compute_macro_f1(y_true, y_pred, labels)
    micro_f1 = compute_micro_f1(y_true, y_pred, labels)

    roc_auc = None
    if y_score is not None:
        roc_auc = compute_roc_auc(y_true, y_score, labels)

    return {
        "accuracy": acc,
        "per_class_sensitivity": sens,
        "per_class_specificity": spec,
        "icbhi_score": icbhi,
        "confusion_matrix": cm,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "roc_auc": roc_auc,
    }


# ----------------------------------------------------------------------
# Simple test if run as main
if __name__ == "__main__":
    # Synthetic multi-class data (4 classes: 0,1,2,3)
    np.random.seed(42)
    n = 200
    y_t = np.random.choice([0, 1, 2, 3], size=n)
    # Create predictions with some noise
    y_p = np.copy(y_t)
    flip_mask = np.random.rand(n) < 0.2
    y_p[flip_mask] = (y_p[flip_mask] + np.random.randint(1, 4)) % 4

    # Generate dummy probabilities for ROC AUC
    y_prob = np.random.rand(n, 4)
    y_prob /= y_prob.sum(axis=1, keepdims=True)

    results = compute_all_metrics(y_t, y_p, y_score=y_prob)

    print("Sample evaluation results:")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    class {k}: {v:.4f}")
        elif isinstance(val, np.ndarray):
            print(f"  {key}:")
            print(val)
        else:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
