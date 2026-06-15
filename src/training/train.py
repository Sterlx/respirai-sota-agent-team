#!/usr/bin/env python3
"""
Training script for the RespirAI lung sound classification model.
Implements mixed-precision training, gradient accumulation, checkpointing,
early stopping, and per-class metric evaluation using the ICBHI 2017 official split.

Skill source: external/AI-research-SKILLs/03-fine-tuning/axolotl/references/other.md (mixed precision)
Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-full.md (general best practices)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.icbhi_dataset import ICBHIDataset  # custom dataset class
from src.evaluation.metrics import compute_classification_metrics  # per-class metrics


def setup_logging(log_file: str, console_level: int = logging.INFO) -> logging.Logger:
    """Configure logger with file and console handlers."""
    logger = logging.getLogger("RespirAI")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_model(config: dict) -> nn.Module:
    """
    Build the model specified in config.
    Currently only supports CNN baseline from src.models.cnn_baseline.
    """
    model_name = config["model"]["name"]
    if model_name == "cnn_baseline":
        from src.models.cnn_baseline import CNNBaseline
        model = CNNBaseline(num_classes=config["model"]["num_classes"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    scaler,
    accumulation_steps,
    logger,
    epoch: int,
) -> float:
    """
    Run one training epoch. Returns average loss.
    """
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    # Progress bar for batches
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch} [Train]")
    for batch_idx, (inputs, targets) in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # accumulate full loss
        pbar.set_postfix(loss=loss.item() * accumulation_steps)

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model,
    dataloader,
    criterion,
    device,
    logger,
) -> tuple[float, dict]:
    """
    Run validation: loss and per-class metrics.
    Returns average loss and metrics dict.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Use the pre-built metrics function (returns dict with per-class Se, Sp, etc.)
    metrics = compute_classification_metrics(all_targets, all_preds, num_classes=4)
    return avg_loss, metrics


def log_metrics(logger, phase: str, loss: float, metrics: dict, epoch: int = None):
    """Log loss and per-class metrics in a readable format."""
    if epoch is not None:
        logger.info(f"Epoch {epoch} {phase} Loss: {loss:.4f}")
    else:
        logger.info(f"{phase} Loss: {loss:.4f}")

    # Log per-class sensitivity and specificity
    for cls_name in ["normal", "crackle", "wheeze", "both"]:
        se = metrics.get(f"{cls_name}_sensitivity", "N/A")
        sp = metrics.get(f"{cls_name}_specificity", "N/A")
        logger.info(f"  Class {cls_name}: Sensitivity={se:.4f}, Specificity={sp:.4f}")

    # Log average scores
    avg_se = metrics.get("avg_sensitivity", "N/A")
    avg_sp = metrics.get("avg_specificity", "N/A")
    icbhi_score = metrics.get("icbhi_score", "N/A")
    logger.info(f"  Average Sensitivity: {avg_se:.4f}")
    logger.info(f"  Average Specificity: {avg_sp:.4f}")
    logger.info(f"  ICBHI Score (avg Se+Sp): {icbhi_score:.4f}")


def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str,
    logger,
):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train RespirAI lung sound classification model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(
        log_file=config["training"].get("log_file", "training.log"),
        console_level=logging.INFO,
    )
    logger.info("Configuration loaded successfully.")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model, optimizer, criterion
    model = get_model(config)
    model.to(device)
    logger.info(f"Model {config['model']['name']} initialized with {config['model']['num_classes']} classes.")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision scaler
    scaler = GradScaler(enabled=config["training"].get("use_amp", True))

    # Datasets and loaders
    data_config = config["data"]
    train_dataset = ICBHIDataset(
        data_dir=data_config["data_dir"],
        split_file=data_config["split_file"],
        split="train",
        use_preprocessed=data_config.get("use_preprocessed", False),
        preprocessed_dir=data_config.get("preprocessed_dir", None),
    )
    val_dataset = ICBHIDataset(
        data_dir=data_config["data_dir"],
        split_file=data_config["split_file"],
        split="test",
        use_preprocessed=data_config.get("use_preprocessed", False),
        preprocessed_dir=data_config.get("preprocessed_dir", None),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=True,
    )

    # Training parameters
    num_epochs = config["training"]["epochs"]
    accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)
    patience = config["training"].get("early_stopping_patience", 10)
    checkpoint_dir = config["training"].get("checkpoint_dir", "checkpoints")
    save_period = config["training"].get("save_period", 5)   # save every N epochs
    best_metric = 0.0
    patience_counter = 0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Starting epoch {epoch}/{num_epochs} ---")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            accumulation_steps, logger, epoch
        )
        val_loss, val_metrics = validate(model, val_loader, criterion, device, logger)

        # Log metrics
        log_metrics(logger, "Train", train_loss, None, epoch)
        log_metrics(logger, "Validation", val_loss, val_metrics, epoch)

        # Early stopping based on ICBHI score (average of sensitivity and specificity)
        current_score = val_metrics.get("icbhi_score", 0.0)
        if current_score > best_metric:
            best_metric = current_score
            patience_counter = 0
            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_icbhi_score": best_metric,
                    "config": config,
                },
                checkpoint_dir,
                "best_model.pth",
                logger,
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement in ICBHI score for {patience_counter} epoch(s).")

        # Periodic checkpoint
        if save_period > 0 and epoch % save_period == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoint_dir,
                f"epoch_{epoch}.pth",
                logger,
            )

        # Always save last checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            checkpoint_dir,
            "last_model.pth",
            logger,
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs with no improvement.")
            break

    logger.info("Training completed.")


if __name__ == "__main__":
    main()
