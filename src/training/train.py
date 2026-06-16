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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.icbhi_dataset import ICBHIDataset
from src.evaluation.metrics import compute_all_metrics

# Label mapping — must match CNNBaseline output order
LABEL_TO_INDEX = {"normal": 0, "crackles": 1, "wheezes": 2, "both": 3}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def make_preprocessing_transform(config: dict) -> callable:
    """Build a waveform→spectrogram transform using src/audio/preprocessing.py
    and optionally src/audio/augmentation.py."""
    from src.audio.preprocessing import log_mel_spectrogram
    from src.audio.augmentation import SpecAugment

    audio_cfg = config.get("audio", {})
    sample_rate = audio_cfg.get("sample_rate", 4000)
    n_fft = audio_cfg.get("n_fft", 256)
    hop_length = audio_cfg.get("hop_length", 64)
    n_mels = audio_cfg.get("n_mels", 32)
    f_min = audio_cfg.get("fmin", 50)
    f_max = audio_cfg.get("fmax", 2000)
    use_augment = audio_cfg.get("augmentation", False)

    class PreprocessingPipeline:
        """Composable pipeline using src/audio modules."""

        def __init__(self):
            self.sample_rate = sample_rate
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.n_mels = n_mels
            self.f_min = f_min
            self.f_max = f_max
            self.use_augment = use_augment
            if use_augment:
                self.augment = SpecAugment(
                    freq_mask_param=max(1, n_mels // 8),
                    time_mask_param=20,
                )

        def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
            # waveform: (1, samples)
            mel = log_mel_spectrogram(
                waveform, self.sample_rate,
                n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.n_mels, f_min=self.f_min, f_max=self.f_max,
            )
            # Ensure output is (1, n_mels, time)
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if self.use_augment:
                mel = self.augment(mel)
            return mel

    return PreprocessingPipeline()


def collate_fn(batch: list, max_time_frames: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate (waveform, labels_dict) → (stacked_inputs_padded, class_indices)."""
    inputs, label_dicts = zip(*batch)

    # Pad spectrograms to the same time dimension
    max_time = max(x.shape[-1] for x in inputs)
    if max_time_frames and max_time > max_time_frames:
        max_time = max_time_frames

    padded = []
    for x in inputs:
        if x.shape[-1] < max_time:
            pad = torch.zeros(x.shape[0], x.shape[1], max_time - x.shape[-1])
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > max_time:
            x = x[..., :max_time]
        padded.append(x)

    targets = []
    for ld in label_dicts:
        for label_name, is_active in ld.items():
            if is_active == 1:
                targets.append(LABEL_TO_INDEX[label_name])
                break

    return torch.stack(padded), torch.tensor(targets)


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

        with autocast("cuda"):
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

            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Use the pre-built metrics function
    metrics = compute_all_metrics(all_targets, all_preds)
    return avg_loss, metrics


def log_metrics(logger, phase: str, loss: float, metrics: dict, epoch: int = None):
    """Log loss and per-class metrics in a readable format."""
    if epoch is not None:
        logger.info(f"Epoch {epoch} {phase} Loss: {loss:.4f}")
    else:
        logger.info(f"{phase} Loss: {loss:.4f}")

    if metrics is None:
        return  # training phase has no metrics

    per_class_se = metrics.get("per_class_sensitivity", {})
    per_class_sp = metrics.get("per_class_specificity", {})

    for cls_idx, cls_name in INDEX_TO_LABEL.items():
        se = per_class_se.get(cls_idx, "N/A")
        sp = per_class_sp.get(cls_idx, "N/A")
        se_str = f"{se:.4f}" if isinstance(se, float) else str(se)
        sp_str = f"{sp:.4f}" if isinstance(sp, float) else str(sp)
        logger.info(f"  Class {cls_name}: Se={se_str}, Sp={sp_str}")

    icbhi = metrics.get("icbhi_score", "N/A")
    icbhi_str = f"{icbhi:.4f}" if isinstance(icbhi, float) else str(icbhi)
    logger.info(f"  ICBHI Score: {icbhi_str}")


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

    # Reproducibility
    torch.manual_seed(config["training"].get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["training"].get("seed", 42))

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

    # Mixed precision scaler
    scaler = GradScaler("cuda", enabled=config["training"].get("use_amp", True))

    # Datasets and loaders
    data_config = config["data"]
    transform = make_preprocessing_transform(config)

    train_dataset = ICBHIDataset(
        data_dir=data_config["data_dir"],
        split_file=data_config["split_file"],
        split="train",
        transform=transform,
    )
    val_dataset = ICBHIDataset(
        data_dir=data_config["data_dir"],
        split_file=data_config["split_file"],
        split="test",
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, config["audio"].get("max_time_frames", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, config["audio"].get("max_time_frames", 0)),
    )

    # Compute class weights from training set and create weighted loss
    class_weights = None
    if config["training"].get("use_class_weights", False):
        from collections import Counter
        label_counts = Counter()
        for _, label_dict in train_dataset:
            for name, val in label_dict.items():
                if val == 1:
                    label_counts[name] += 1
        total = sum(label_counts.values())
        weight_list = [
            total / max(label_counts.get(name, 1), 1)
            for name in ["normal", "crackles", "wheezes", "both"]
        ]
        class_weights = torch.tensor(weight_list, device=device)
        logger.info(f"Class weights: normal={weight_list[0]:.2f} crackles={weight_list[1]:.2f} "
                     f"wheezes={weight_list[2]:.2f} both={weight_list[3]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    # Log experiment to tracker
    from src.evaluation.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker(
        log_file=config["training"].get("experiment_log", "experiments.jsonl")
    )
    tracker.log_experiment(
        config=config,
        metrics={"best_icbhi_score": best_metric, "epochs_completed": epoch},
        name=config["model"]["name"],
    )
    logger.info(f"Experiment logged to {tracker.log_file}")


if __name__ == "__main__":
    main()
