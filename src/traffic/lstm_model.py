"""
LSTM Model for Multi-Horizon Highway Traffic Speed Forecasting.
"""

import torch
import torch.nn as nn


class TrafficLSTM(nn.Module):
    """
    Multi-horizon LSTM model for spatio-temporal road network traffic forecasting.
    Input shape: (batch_size, seq_len=12, num_sensors=207)
    Outputs:
      pred_30: (batch_size, num_sensors) -> 30-minute ahead forecast
      pred_60: (batch_size, num_sensors) -> 60-minute ahead forecast
    """
    def __init__(self, num_sensors: int = 207, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super(TrafficLSTM, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder LSTM
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Multi-horizon dedicated forecasting heads
        self.head_30min = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensors)
        )

        self.head_60min = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_sensors)
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, num_sensors)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        normed = self.layer_norm(self.dropout(last_hidden))

        pred_30 = self.head_30min(normed)
        pred_60 = self.head_60min(normed)

        return pred_30, pred_60
