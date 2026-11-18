"""
Comprehensive Evaluation of Traffic Forecasting Models on Unseen Test Split.
Computes MAE, RMSE, MAPE, R2, generates comparison plots, and saves test predictions.
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, get_device, save_metrics
from src.traffic.lstm_model import TrafficLSTM
from src.traffic.gnn_model import TrafficGNN


def evaluate_model(model_name: str, y30_pred: np.ndarray, y60_pred: np.ndarray,
                   y30_true: np.ndarray, y60_true: np.ndarray, elapsed_time: float) -> dict:
    """Compute MAE, RMSE, MAPE, R2 for 30m and 60m horizons."""
    def calc_metrics(y_pred, y_true):
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mape = float(np.mean(np.abs((y_pred - y_true) / np.clip(y_true, 1.0, None))) * 100)
        r2 = float(r2_score(y_true.flatten(), y_pred.flatten()))
        return {'mae': round(mae, 3), 'rmse': round(rmse, 3), 'mape': round(mape, 3), 'r2': round(r2, 4)}

    m30 = calc_metrics(y30_pred, y30_true)
    m60 = calc_metrics(y60_pred, y60_true)

    print(f"\n[{model_name} Test Results]")
    print(f"  30-min Horizon: MAE = {m30['mae']:.2f} mph | RMSE = {m30['rmse']:.2f} mph | MAPE = {m30['mape']:.2f}% | R2 = {m30['r2']:.4f}")
    print(f"  60-min Horizon: MAE = {m60['mae']:.2f} mph | RMSE = {m60['rmse']:.2f} mph | MAPE = {m60['mape']:.2f}% | R2 = {m60['r2']:.4f}")
    print(f"  Inference Latency: {elapsed_time:.3f}s ({elapsed_time / len(y30_true) * 1000:.2f} ms/sample)")

    return {
        'model': model_name,
        'horizon_30min': m30,
        'horizon_60min': m60,
        'inference_time_sec': round(elapsed_time, 4),
        'latency_ms_per_sample': round(elapsed_time / len(y30_true) * 1000, 3)
    }


def generate_evaluation_plots(data: dict, preds: dict, plots_dir: str):
    """Generate professional scientific plots comparing actual vs predicted speeds."""
    os.makedirs(plots_dir, exist_ok=True)
    sensor_idx = 0
    sensor_id = data['sensor_ids'][sensor_idx]
    y30_true = data['y30_true'][:, sensor_idx]
    y60_true = data['y60_true'][:, sensor_idx]
    time_steps = min(288, len(y30_true))

    # 1. Actual vs Predicted Speed Curves
    plt.figure(figsize=(14, 6), dpi=150)
    plt.plot(range(time_steps), y30_true[:time_steps], label="Actual Ground Truth Speed", color="#1f77b4", linewidth=2.0)
    plt.plot(range(time_steps), preds['Baseline']['y30'][:time_steps, sensor_idx], label="Baseline (Persistence)", color="#7f7f7f", linestyle="--", alpha=0.7)
    plt.plot(range(time_steps), preds['LSTM']['y30'][:time_steps, sensor_idx], label="Model A: Multi-Horizon LSTM", color="#ff7f0e", linestyle="-.", linewidth=1.8)
    plt.plot(range(time_steps), preds['ST-GNN']['y30'][:time_steps, sensor_idx], label="Model B: Spatio-Temporal GNN", color="#2ca02c", linewidth=2.0)

    plt.title(f"METR-LA Highway Traffic Forecasting (Sensor #{sensor_id} - 30-min Horizon)", fontsize=14, fontweight="bold")
    plt.xlabel("Time Step (5-minute intervals / 24-Hour Corridor)", fontsize=12)
    plt.ylabel("Traffic Speed (mph)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    plot_1 = os.path.join(plots_dir, "traffic_actual_vs_predicted.png")
    plt.savefig(plot_1)
    plt.close()
    print(f"[Traffic Evaluation] Saved plot: {plot_1}")

    # 2. 30-min vs 60-min Horizon Error Comparison
    plt.figure(figsize=(10, 5), dpi=150)
    models = ['Baseline', 'LSTM', 'ST-GNN']
    mae_30 = [preds[m]['metrics']['horizon_30min']['mae'] for m in models]
    mae_60 = [preds[m]['metrics']['horizon_60min']['mae'] for m in models]

    x = np.arange(len(models))
    width = 0.35
    plt.bar(x - width/2, mae_30, width, label='30-Minute Horizon', color='#3498db', edgecolor='black', alpha=0.9)
    plt.bar(x + width/2, mae_60, width, label='60-Minute Horizon', color='#e74c3c', edgecolor='black', alpha=0.9)

    plt.ylabel('Mean Absolute Error (mph)', fontsize=12)
    plt.title('Multi-Horizon Forecasting Error Comparison across Models', fontsize=13, fontweight='bold')
    plt.xticks(x, models, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot_2 = os.path.join(plots_dir, "traffic_horizon_comparison.png")
    plt.savefig(plot_2)
    plt.close()
    print(f"[Traffic Evaluation] Saved plot: {plot_2}")

    # 3. Training & Validation Loss Curves
    hist_path = os.path.join(os.path.dirname(plots_dir), "metrics", "traffic_training_history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r") as f:
            hist_data = json.load(f)
        plt.figure(figsize=(11, 5), dpi=150)
        if 'lstm_history' in hist_data and hist_data['lstm_history']['train_loss']:
            plt.plot(hist_data['lstm_history']['train_loss'], label='LSTM Train Loss', color='#ff7f0e', linestyle='--')
            plt.plot(hist_data['lstm_history']['val_loss'], label='LSTM Val Loss', color='#ff7f0e', linewidth=2)
        if 'gnn_history' in hist_data and hist_data['gnn_history']['train_loss']:
            plt.plot(hist_data['gnn_history']['train_loss'], label='ST-GNN Train Loss', color='#2ca02c', linestyle='--')
            plt.plot(hist_data['gnn_history']['val_loss'], label='ST-GNN Val Loss', color='#2ca02c', linewidth=2)

        plt.title('Training and Validation Loss Curves (MSE)', fontsize=13, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (Normalized MSE)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plot_3 = os.path.join(plots_dir, "traffic_loss_curves.png")
        plt.savefig(plot_3)
        plt.close()
        print(f"[Traffic Evaluation] Saved plot: {plot_3}")


def main():
    config = load_config()
    device = get_device(config["project"]["device"])
    proc_dir = config["traffic"]["processed_dir"]
    models_dir = config["traffic"]["models_dir"]
    outputs_dir = config["traffic"]["outputs_dir"]

    data_path = os.path.join(proc_dir, "traffic_processed.pkl")
    graph_path = os.path.join(proc_dir, "traffic_graph.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    mean = data['mean']
    std = data['std']
    y30_true = data['Y30_test'] * std + mean
    y60_true = data['Y60_test'] * std + mean
    data['y30_true'] = y30_true
    data['y60_true'] = y60_true

    num_sensors = data['X_train'].shape[2]
    norm_adj = torch.tensor(graph['norm_adj'], dtype=torch.float32).to(device)

    # 1. Baseline Evaluation
    t0 = time.time()
    base_pred_norm = data['X_test'][:, -1, :]
    base_pred = base_pred_norm * std + mean
    base_time = time.time() - t0
    base_metrics = evaluate_model("Baseline (Persistence)", base_pred, base_pred, y30_true, y60_true, base_time)

    # 2. LSTM Evaluation
    lstm_model = TrafficLSTM(num_sensors=num_sensors, hidden_dim=128, num_layers=2).to(device)
    lstm_ckpt = torch.load(os.path.join(models_dir, "lstm_best.pth"), map_location=device)
    lstm_model.load_state_dict(lstm_ckpt['model_state_dict'])
    lstm_model.eval()

    t0 = time.time()
    with torch.no_grad():
        p30_norm, p60_norm = lstm_model(X_test.to(device))
    lstm_time = time.time() - t0
    lstm_p30 = (p30_norm.cpu().numpy()) * std + mean
    lstm_p60 = (p60_norm.cpu().numpy()) * std + mean
    lstm_metrics = evaluate_model("LSTM", lstm_p30, lstm_p60, y30_true, y60_true, lstm_time)

    # 3. ST-GNN Evaluation
    gnn_model = TrafficGNN(num_sensors=num_sensors, gcn_dim=16, rnn_dim=32).to(device)
    gnn_ckpt = torch.load(os.path.join(models_dir, "gnn_best.pth"), map_location=device)
    gnn_model.load_state_dict(gnn_ckpt['model_state_dict'])
    gnn_model.eval()

    t0 = time.time()
    with torch.no_grad():
        p30_norm, p60_norm = gnn_model(X_test.to(device), norm_adj)
    gnn_time = time.time() - t0
    gnn_p30 = (p30_norm.cpu().numpy()) * std + mean
    gnn_p60 = (p60_norm.cpu().numpy()) * std + mean
    gnn_metrics = evaluate_model("ST-GNN", gnn_p30, gnn_p60, y30_true, y60_true, gnn_time)

    metrics_summary = {
        'Baseline': base_metrics,
        'LSTM': lstm_metrics,
        'ST-GNN': gnn_metrics
    }
    metrics_path = os.path.join(outputs_dir, "metrics", "traffic_metrics.json")
    save_metrics(metrics_summary, metrics_path)

    pred_bundle = {
        'timestamps': data['test_timestamps'],
        'sensor_ids': data['sensor_ids'],
        'sensor_locations': graph['sensor_locations'],
        'y30_true': y30_true,
        'y60_true': y60_true,
        'Baseline': {'y30': base_pred, 'y60': base_pred, 'metrics': base_metrics},
        'LSTM': {'y30': lstm_p30, 'y60': lstm_p60, 'metrics': lstm_metrics},
        'ST-GNN': {'y30': gnn_p30, 'y60': gnn_p60, 'metrics': gnn_metrics}
    }
    pred_path = os.path.join(outputs_dir, "predictions", "traffic_test_predictions.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred_bundle, f, protocol=4)
    print(f"[Traffic Evaluation] Saved test predictions to {pred_path}")

    plots_dir = os.path.join(outputs_dir, "plots")
    generate_evaluation_plots(data, pred_bundle, plots_dir)


if __name__ == "__main__":
    main()
