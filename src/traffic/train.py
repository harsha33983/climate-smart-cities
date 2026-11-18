"""
Training Script for Traffic Forecasting Models: Baseline, LSTM, and ST-GNN.
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, set_seed, get_device, save_metrics
from src.traffic.lstm_model import TrafficLSTM
from src.traffic.gnn_model import TrafficGNN


def load_data(processed_dir: str):
    """Load preprocessed traffic data and graph topology."""
    data_path = os.path.join(processed_dir, "traffic_processed.pkl")
    graph_path = os.path.join(processed_dir, "traffic_graph.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    return data, graph


def train_baseline(data: dict) -> dict:
    """
    Train / Evaluate Baseline Model:
    Persistence forecasting: predicts future speed as the last observed value in input window.
    """
    print("\n--- Training Baseline Model (Persistence / Historical Last-Observed) ---")
    start_time = time.time()
    
    X_test = data['X_test']
    Y30_test = data['Y30_test']
    Y60_test = data['Y60_test']
    mean = data['mean']
    std = data['std']

    pred_norm = X_test[:, -1, :]
    y30_true = Y30_test * std + mean
    y60_true = Y60_test * std + mean
    y_pred = pred_norm * std + mean

    mae_30 = float(np.mean(np.abs(y_pred - y30_true)))
    rmse_30 = float(np.sqrt(np.mean((y_pred - y30_true) ** 2)))
    mape_30 = float(np.mean(np.abs((y_pred - y30_true) / np.clip(y30_true, 1.0, None))) * 100)

    mae_60 = float(np.mean(np.abs(y_pred - y60_true)))
    rmse_60 = float(np.sqrt(np.mean((y_pred - y60_true) ** 2)))
    mape_60 = float(np.mean(np.abs((y_pred - y60_true) / np.clip(y60_true, 1.0, None))) * 100)

    elapsed = time.time() - start_time
    print(f"[Baseline] 30-min Forecast: MAE = {mae_30:.2f} mph, RMSE = {rmse_30:.2f} mph, MAPE = {mape_30:.2f}%")
    print(f"[Baseline] 60-min Forecast: MAE = {mae_60:.2f} mph, RMSE = {rmse_60:.2f} mph, MAPE = {mape_60:.2f}%")

    metrics = {
        'model': 'Baseline (Persistence)',
        'horizon_30min': {'mae': round(mae_30, 3), 'rmse': round(rmse_30, 3), 'mape': round(mape_30, 3)},
        'horizon_60min': {'mae': round(mae_60, 3), 'rmse': round(rmse_60, 3), 'mape': round(mape_60, 3)},
        'inference_time_sec': round(elapsed, 4)
    }
    return metrics


def train_lstm(data: dict, config: dict, device: torch.device) -> tuple:
    """Train sequential LSTM model with early stopping."""
    print("\n--- Training Model A: Multi-Horizon LSTM ---")
    batch_size = config["traffic"]["training"]["batch_size"]
    epochs = config["traffic"]["training"]["epochs"]
    lr = config["traffic"]["training"]["lr"]
    patience = config["traffic"]["training"]["early_stopping_patience"]
    models_dir = config["traffic"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    train_ds = TensorDataset(
        torch.tensor(data['X_train'], dtype=torch.float32),
        torch.tensor(data['Y30_train'], dtype=torch.float32),
        torch.tensor(data['Y60_train'], dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(data['X_val'], dtype=torch.float32),
        torch.tensor(data['Y30_val'], dtype=torch.float32),
        torch.tensor(data['Y60_val'], dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_sensors = data['X_train'].shape[2]
    model = TrafficLSTM(num_sensors=num_sensors, hidden_dim=128, num_layers=2, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights_path = os.path.join(models_dir, "lstm_best.pth")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y30_b, y60_b in train_loader:
            x_b, y30_b, y60_b = x_b.to(device), y30_b.to(device), y60_b.to(device)
            optimizer.zero_grad()
            p30, p60 = model(x_b)
            loss = criterion(p30, y30_b) + criterion(p60, y60_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(x_b)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y30_b, y60_b in val_loader:
                x_b, y30_b, y60_b = x_b.to(device), y30_b.to(device), y60_b.to(device)
                p30, p60 = model(x_b)
                loss = criterion(p30, y30_b) + criterion(p60, y60_b)
                val_loss += loss.item() * len(x_b)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"LSTM Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config['traffic'],
                'epoch': epoch,
                'val_loss': val_loss
            }, best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[LSTM] Early stopping triggered at epoch {epoch}.")
                break

    training_time = time.time() - start_time
    print(f"[LSTM] Training finished in {training_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    return model, history, best_weights_path


def train_gnn(data: dict, graph: dict, config: dict, device: torch.device) -> tuple:
    """Train Spatio-Temporal Graph Neural Network."""
    print("\n--- Training Model B: Spatio-Temporal GNN ---")
    batch_size = config["traffic"]["training"]["batch_size"]
    epochs = config["traffic"]["training"]["epochs"]
    lr = config["traffic"]["training"]["lr"]
    patience = config["traffic"]["training"]["early_stopping_patience"]
    models_dir = config["traffic"]["models_dir"]

    norm_adj = torch.tensor(graph['norm_adj'], dtype=torch.float32).to(device)

    train_ds = TensorDataset(
        torch.tensor(data['X_train'], dtype=torch.float32),
        torch.tensor(data['Y30_train'], dtype=torch.float32),
        torch.tensor(data['Y60_train'], dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(data['X_val'], dtype=torch.float32),
        torch.tensor(data['Y30_val'], dtype=torch.float32),
        torch.tensor(data['Y60_val'], dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_sensors = data['X_train'].shape[2]
    model = TrafficGNN(num_sensors=num_sensors, gcn_dim=16, rnn_dim=32, dropout=0.2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights_path = os.path.join(models_dir, "gnn_best.pth")

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y30_b, y60_b in train_loader:
            x_b, y30_b, y60_b = x_b.to(device), y30_b.to(device), y60_b.to(device)
            optimizer.zero_grad()
            p30, p60 = model(x_b, norm_adj)
            loss = criterion(p30, y30_b) + criterion(p60, y60_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * len(x_b)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y30_b, y60_b in val_loader:
                x_b, y30_b, y60_b = x_b.to(device), y30_b.to(device), y60_b.to(device)
                p30, p60 = model(x_b, norm_adj)
                loss = criterion(p30, y30_b) + criterion(p60, y60_b)
                val_loss += loss.item() * len(x_b)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"ST-GNN Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config['traffic'],
                'epoch': epoch,
                'val_loss': val_loss
            }, best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ST-GNN] Early stopping triggered at epoch {epoch}.")
                break

    training_time = time.time() - start_time
    print(f"[ST-GNN] Training finished in {training_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    return model, history, best_weights_path


def main():
    set_seed(42)
    config = load_config()
    device = get_device(config["project"]["device"])
    print(f"[Traffic Training] Device: {device}")

    proc_dir = config["traffic"]["processed_dir"]
    data, graph = load_data(proc_dir)

    baseline_metrics = train_baseline(data)
    lstm_model, lstm_history, lstm_path = train_lstm(data, config, device)
    gnn_model, gnn_history, gnn_path = train_gnn(data, graph, config, device)

    history_out = os.path.join(config["traffic"]["outputs_dir"], "metrics", "traffic_training_history.json")
    save_metrics({
        'baseline': baseline_metrics,
        'lstm_history': lstm_history,
        'gnn_history': gnn_history
    }, history_out)

    print("\n[Traffic Training] All traffic models trained and checkpoints saved successfully!")


if __name__ == "__main__":
    main()
