"""
Multimodal CNN + Tabular Fusion Model for Wildfire Risk Mapping.
Fuses spatial satellite/topographic feature maps with tabular environmental indicators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.wildfire.cnn_model import WildfireCNN


class WildfireMultimodalFusion(nn.Module):
    """
    Multimodal Deep Learning Architecture:
      Branch A: Spatial CNN extracting 2D contextual patterns from environmental grids.
      Branch B: Tabular MLP processing meteorological & topographic scalar indicators.
      Fusion: Cross-modal feature aggregation with bottleneck classification head.
    """
    def __init__(self, tabular_dim: int = 13, in_channels: int = 5, num_classes: int = 2, dropout: float = 0.3):
        super(WildfireMultimodalFusion, self).__init__()

        # Spatial CNN Branch
        self.cnn_branch = WildfireCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

        # Tabular MLP Branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # Multimodal Fusion Head (64 spatial + 64 tabular = 128)
        self.fusion_fc1 = nn.Linear(64 + 64, 64)
        self.bn_fusion = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        # Extract spatial embedding from CNN
        spatial_emb = self.cnn_branch.extract_features(x_img)  # (B, 64)

        # Extract environmental embedding from Tabular MLP
        tabular_emb = self.tabular_mlp(x_tab)  # (B, 64)

        # Cross-modal concatenation
        fused = torch.cat([spatial_emb, tabular_emb], dim=1)  # (B, 128)

        # Final risk classification head
        h = F.relu(self.bn_fusion(self.fusion_fc1(fused)))
        logits = self.classifier(self.dropout(h))

        return logits
