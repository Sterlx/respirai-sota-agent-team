# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-full.md
# (Inspired by the Unsloth library's emphasis on efficient and clear training pipelines.
# This baseline CNN is designed to be simple, trainable, and compatible with ICBHI2017
# preprocessing, following the project's requirement for a 4-class lung sound classifier.)

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    """
    A simple baseline CNN for 4-class lung sound classification.
    
    Input:
        x: log-mel spectrogram tensor of shape [B, 1, n_mels, time]
           where n_mels is the number of mel bands (e.g., 64) and 
           time is the number of time frames (variable length).
    
    Output:
        logits: tensor of shape [B, 4] corresponding to scores for
                classes 0: normal, 1: crackle, 2: wheeze, 3: both.
                No Softmax is applied; intended for CrossEntropyLoss.
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        """
        Args:
            in_channels (int): Number of input channels (default 1 for mono).
            num_classes (int): Number of output classes (default 4).
        """
        super(CNNBaseline, self).__init__()
        
        # Convolutional block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial dims
        )
        
        # Convolutional block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Convolutional block 4 (optional, can be kept for representational power)
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # collapse spatial dimensions to fixed size
        )
        
        # Classifier head
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (Tensor): Input of shape [B, 1, n_mels, time].
            
        Returns:
            Tensor: Logits of shape [B, num_classes].
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # output shape: [B, 256, 1, 1]
        
        # Flatten to [B, 256]
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x


def build_cnn_baseline(num_classes: int = 4, in_channels: int = 1) -> CNNBaseline:
    """
    Factory function to instantiate the baseline CNN model.
    
    Args:
        num_classes (int): Number of target classes (default 4: normal, crackle, wheeze, both).
        in_channels (int): Number of input channels (default 1 for mono log-mel spectrograms).
        
    Returns:
        CNNBaseline: Instantiated model.
    """
    return CNNBaseline(in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    # Quick sanity check
    batch_size = 2
    n_mels = 64
    time_steps = 128
    dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
    
    model = build_cnn_baseline()
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [2, 4]
    
    # Check compatibility with CrossEntropyLoss
    loss_fn = nn.CrossEntropyLoss()
    dummy_labels = torch.randint(0, 4, (batch_size,))
    loss = loss_fn(output, dummy_labels)
    print(f"CrossEntropyLoss: {loss.item():.4f}")
