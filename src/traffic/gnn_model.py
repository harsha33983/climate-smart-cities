"""
Graph Neural Network (STGNN / Spatial-Temporal GCN) for Traffic Flow Forecasting.
Optimized Node-Wise Spatio-Temporal Graph Recurrent architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Spatial Graph Convolution layer implementing:
    Z = A_norm * X * W + b
    where A_norm is the (N, N) normalized road network adjacency matrix.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor):
        # x: (batch_size, num_nodes, in_features)
        # norm_adj: (num_nodes, num_nodes)
        ax = torch.matmul(norm_adj, x)  # (batch_size, num_nodes, in_features)
        out = self.linear(ax)           # (batch_size, num_nodes, out_features)
        return F.relu(out)


class TrafficGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (ST-GNN).
    Combines Spatial Graph Convolutions with Node-Wise Temporal Recurrence.
    Scalable, fast, and mathematically rigorous.
    """
    def __init__(self, num_sensors: int = 207, gcn_dim: int = 16, rnn_dim: int = 32, dropout: float = 0.2):
        super(TrafficGNN, self).__init__()
        self.num_sensors = num_sensors
        self.gcn_dim = gcn_dim
        self.rnn_dim = rnn_dim

        # Spatial Graph Convolutions
        self.gcn1 = GraphConvLayer(in_features=1, out_features=gcn_dim)
        self.gcn2 = GraphConvLayer(in_features=gcn_dim, out_features=gcn_dim)

        # Node-Wise Temporal Recurrence across sequence (T=12)
        # Processes (batch_size * num_nodes, seq_len, gcn_dim)
        self.temporal_gru = nn.GRU(
            input_size=gcn_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(rnn_dim)

        # Forecasting heads for 30m and 60m horizons (node-wise)
        self.head_30min = nn.Sequential(
            nn.Linear(rnn_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.head_60min = nn.Sequential(
            nn.Linear(rnn_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor):
        # x: (batch_size, seq_len=12, num_sensors=207)
        batch_size, seq_len, num_nodes = x.size()

        # Reshape to apply GCN across each time step
        # (batch_size * seq_len, num_nodes, 1)
        x_reshaped = x.view(batch_size * seq_len, num_nodes, 1)

        # Spatial message passing
        g1 = self.gcn1(x_reshaped, norm_adj)  # (B*T, N, gcn_dim)
        g2 = self.gcn2(g1, norm_adj)          # (B*T, N, gcn_dim)

        # Rearrange for Node-Wise Temporal GRU: (batch_size, seq_len, num_nodes, gcn_dim)
        g_nodes = g2.view(batch_size, seq_len, num_nodes, self.gcn_dim)
        # Permute to: (batch_size, num_nodes, seq_len, gcn_dim) -> (batch_size * num_nodes, seq_len, gcn_dim)
        g_nodes = g_nodes.permute(0, 2, 1, 3).contiguous().view(batch_size * num_nodes, seq_len, self.gcn_dim)

        # Temporal sequence modeling
        gru_out, _ = self.temporal_gru(g_nodes)
        last_hidden = gru_out[:, -1, :]  # (B*N, rnn_dim)
        normed = self.layer_norm(self.dropout(last_hidden))

        # Project to 30-min and 60-min speeds per sensor
        pred_30 = self.head_30min(normed).view(batch_size, num_nodes)
        pred_60 = self.head_60min(normed).view(batch_size, num_nodes)

        return pred_30, pred_60
