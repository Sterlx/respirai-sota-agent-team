"""
Audio Spectrogram Transformer (AST) for ICBHI 2017 lung sound classification.

AST applies Vision Transformer to spectrogram patches, pre-trained on AudioSet.
Reference: Gong et al. "AST: Audio Spectrogram Transformer" (Interspeech 2021).

Input:  [B, 1, 128, 1024]  log-mel spectrogram (128 mel × 1024 time frames)
Output: [B, 2]  logits for [crackles, wheezes] multi-label classification
"""

import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig


class ASTClassifier(nn.Module):
    """AST backbone + classification head for multi-label lung sound."""

    def __init__(
        self,
        num_labels: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            num_labels: Number of output logits (2 for crackles/wheezes).
            pretrained: Load AudioSet pre-trained weights.
            dropout: Dropout rate for classifier head.
        """
        super().__init__()

        if pretrained:
            self.ast = ASTModel.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
        else:
            config = ASTConfig(
                num_mel_bins=128,
                max_length=1024,
                num_labels=527,  # AudioSet classes for pretraining
            )
            self.ast = ASTModel(config)

        hidden_size = self.ast.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Log-mel spectrogram. Accepts [B, 1, n_mels, time] or
               [B, 1, n_mels, 1, time] (extra channel from MelSpectrogram).

        Returns:
            logits: [B, num_labels]
        """
        # AST expects [B, 1, n_mels, time] — preprocessing guarantees this
        outputs = self.ast(x).pooler_output  # [B, 768]
        return self.classifier(outputs)
