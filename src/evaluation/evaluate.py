# src/evaluation/evaluate.py
# Skill source: none – standard ML evaluation pipeline with PyTorch.

"""
Evaluation script for the ICBHI 2017 lung sound classification model.

Loads a trained model from a checkpoint, runs inference on the official test
split, computes all required metrics (per-class sensitivity/specificity,
ICBHI score, confusion matrix, F1 scores, ROC-AUC), prints them in a
readable table, and saves the results to a JSON file.

Usage:
    python src/evaluation/evaluate.py --model_path checkpoints/best_model.pth \
                                      --data_dir data/ \
                                      --output_file results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project imports (adjust as needed)
from src.data.icbhi_dataset import ICBHIDataset, get_test_dataloader
from src.models.cnn_baseline import CNNBaseline   # baseline model; replace with your own
from src.evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a model from a checkpoint file.

    Supports both cases:
      - checkpoint contains the whole model object (torch.save(model, path))
      - checkpoint contains only state_dict (torch.save(model.state_dict(), path))
    For state_dict loading we instantiate a default CNNBaseline for now;
    in the future this can be made configurable.

    Args:
        checkpoint_path: Path to the .pth file.
        device: torch device.

    Returns:
        Loaded model, set to eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # If checkpoint is an nn.Module, use it directly
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    # If it's a dict with a 'state_dict' key, use that
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model = CNNBaseline()  # TODO: make model class configurable (e.g., via config file)
        model.load_state_dict(checkpoint["state_dict"])
    # If it's a plain state_dict (dict of parameters)
    elif isinstance(checkpoint, dict):
        model = CNNBaseline()
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("Unknown checkpoint format. Expected nn.Module, dict with 'state_dict', or state_dict")

    model.to(device)
    model.eval()
    return model


def run_evaluation(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
) -> dict:
    """
    Run inference on the test set and gather predictions.

    Args:
        model: Trained model.
        test_loader: DataLoader for the test set.
        device: torch device.
        num_classes: Number of classes (default 4 for ICBHI).

    Returns:
        Tuple of (all_preds, all_targets, all_probs)
    """
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Assuming the dataset returns (inputs, labels) or (inputs, labels, ...)
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            outputs = model(inputs)  # raw logits
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds), np.concatenate(all_probs)


def print_metrics_table(metrics: dict, class_names: list = None):
    """
    Print all metrics in a human-readable table.

    Args:
        metrics: Dictionary returned by compute_all_metrics.
        class_names: Optional list of class name strings (length = number of classes).
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in sorted(metrics["per_class_sensitivity"].keys())]

    print("\n" + "=" * 60)
    print("ICBHI 2017 Evaluation Results")
    print("=" * 60)

    # Overall metrics
    print(f"Overall Accuracy (for reference): {metrics['accuracy']:.4f}")
    print(f"ICBHI Score (avg(Se+Sp)/2):        {metrics['icbhi_score']:.4f}")
    print(f"Macro F1 Score:                     {metrics['macro_f1']:.4f}")
    print(f"Micro F1 Score:                     {metrics['micro_f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC AUC (macro OvR):                {metrics['roc_auc']:.4f}")
    print()

    # Per-class metrics
    header = f"{'Class':<15} {'Sensitivity':>12} {'Specificity':>12}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        sens = metrics["per_class_sensitivity"].get(i, 0.0)
        spec = metrics["per_class_specificity"].get(i, 0.0)
        print(f"{name:<15} {sens:12.4f} {spec:12.4f}")

    # Confusion matrix
    print()
    print("Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    # Print with headers
    header_row = " " * 10 + "".join(f"{name:>8}" for name in class_names)
    print(header_row)
    for i, name in enumerate(class_names):
        row_str = f"{name:<10}" + "".join(f"{cm[i, j]:8d}" for j in range(len(class_names)))
        print(row_str)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ICBHI lung sound model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data/", help="Root directory of ICBHI dataset (contains audio/ and official_split.txt)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="JSON file to save results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', or 'auto'")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Class names in order, e.g., 'Normal' 'Wheeze' 'Crackle' 'Both'")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 1. Load model
    model = load_model(args.model_path, device)

    # 2. Build test DataLoader (use existing dataset infrastructure)
    #    We assume a get_test_dataloader function exists that reads the official split.
    #    It should handle the correct preprocessing, augmentation disabled, etc.
    test_loader = get_test_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3. Run inference
    logger.info("Running inference on test set...")
    y_true, y_pred, y_score = run_evaluation(model, test_loader, device, num_classes=4)

    # 4. Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_score)

    # 5. Print results
    class_names = args.class_names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(4)]
    print_metrics_table(metrics, class_names)

    # 6. Save to JSON
    # Convert numpy arrays to lists for JSON serialisation
    output = {
        "accuracy": metrics["accuracy"],
        "icbhi_score": metrics["icbhi_score"],
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "roc_auc": metrics["roc_auc"],
        "per_class_sensitivity": {str(k): v for k, v in metrics["per_class_sensitivity"].items()},
        "per_class_specificity": {str(k): v for k, v in metrics["per_class_specificity"].items()},
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Evaluation results saved to {args.output_file}")


if __name__ == "__main__":
    main()
