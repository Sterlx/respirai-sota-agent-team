"""
CRNN (CNN + BiGRU) for ICBHI2017 4-class lung sound classification.

Architecture:
  1. CNN encoder (3 conv blocks) → local spectral features
  2. BiGRU → temporal dynamics (wheeze = continuous, crackles = transient)
  3. Attention pooling over time steps
  4. Classifier head → 4-class logits

Input:  [B, 1, n_mels, time_frames]  log-mel spectrogram
Output: [B, 4]  logits for [normal, crackles, wheezes, both]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """CNN encoder + BiGRU + attention for lung sound classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        cnn_channels: list[int] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            in_channels: Input channels (1 for mono spectrogram).
            num_classes: Number of output classes (4).
            cnn_channels: List of conv channel sizes. Default [32, 64, 128].
            gru_hidden: Hidden size of BiGRU.
            gru_layers: Number of BiGRU layers.
            dropout: Dropout rate after GRU.
        """
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # --- CNN Encoder ---
        cnn_blocks = []
        prev_ch = in_channels
        for ch in cnn_channels:
            cnn_blocks.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),  # halve freq and time
                nn.Dropout2d(0.1),
            ])
            prev_ch = ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # Frequency dimension after CNN: n_mels // (2 ** len(cnn_channels))
        # e.g., 32 → 16 → 8 → 4
        self.cnn_out_ch = cnn_channels[-1]

        # --- BiGRU ---
        self.gru = nn.GRU(
            input_size=self.cnn_out_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # --- Attention ---
        gru_out_dim = gru_hidden * 2  # bidirectional
        self.attention = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim // 2),
            nn.Tanh(),
            nn.Linear(gru_out_dim // 2, 1),
        )

        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, n_mels, time] log-mel spectrogram.

        Returns:
            logits: [B, num_classes]
        """
        # CNN: [B, 1, F, T] → [B, C, F', T']
        x = self.cnn(x)
        b, c, f, t = x.shape

        # Reshape for GRU: [B, T', C*F'] → actually [B, T', C] pooling over freq
        # Mean-pool frequency dimension: [B, C, T']
        x = x.mean(dim=2)  # [B, C, T']
        x = x.permute(0, 2, 1)  # [B, T', C]

        # BiGRU: [B, T', C] → [B, T', H*2]
        x, _ = self.gru(x)

        # Attention pooling over time
        attn_weights = self.attention(x)  # [B, T', 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        x = (x * attn_weights).sum(dim=1)  # [B, H*2]

        # Classifier
        return self.classifier(x)
