"""
Spatial 2D Convolutional Neural Network (CNN) for Wildfire Risk Mapping.
Extracts spatial patterns from multi-channel environmental and topographic patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WildfireCNN(nn.Module):
    """
    Spatial CNN processing (batch_size, channels=5, height=16, width=16) patches:
      Ch 0: Elevation DEM
      Ch 1: Slope surface
      Ch 2: Temperature field
      Ch 3: Relative Humidity field
      Ch 4: Vegetation Dryness field
    Outputs: Binary logits (Fire Occurrence / High Wildfire Risk)
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 2, dropout: float = 0.3):
        super(WildfireCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial embedding vector for multimodal fusion."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # (B, 32, 8, 8)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (B, 64, 4, 4)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)  # (B, 128, 2, 2)

        x = x.view(x.size(0), -1)
        features = F.relu(self.fc1(self.dropout(x)))  # (B, 64)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.fc2(self.dropout(features))
        return logits
