"""
Inference & Congestion Alert Pipeline for Urban Traffic Forecasting.
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, get_device
from src.traffic.lstm_model import TrafficLSTM
from src.traffic.gnn_model import TrafficGNN


class TrafficPredictor:
    def __init__(self, config_path: str = "config.yaml", model_type: str = "gnn"):
        self.config = load_config(config_path)
        self.device = get_device(self.config["project"]["device"])
        self.model_type = model_type.lower()
        self.models_dir = self.config["traffic"]["models_dir"]
        self.proc_dir = self.config["traffic"]["processed_dir"]

        with open(os.path.join(self.proc_dir, "traffic_processed.pkl"), "rb") as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.std = data['std']
            self.sensor_ids = data['sensor_ids']
            self.num_sensors = len(self.sensor_ids)

        with open(os.path.join(self.proc_dir, "traffic_graph.pkl"), "rb") as f:
            graph = pickle.load(f)
            self.norm_adj = torch.tensor(graph['norm_adj'], dtype=torch.float32).to(self.device)
            self.sensor_locs = graph['sensor_locations']

        if "lstm" in self.model_type:
            self.model = TrafficLSTM(num_sensors=self.num_sensors).to(self.device)
            ckpt = torch.load(os.path.join(self.models_dir, "lstm_best.pth"), map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model = TrafficGNN(num_sensors=self.num_sensors, gcn_dim=16, rnn_dim=32).to(self.device)
            ckpt = torch.load(os.path.join(self.models_dir, "gnn_best.pth"), map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])

        self.model.eval()

    def predict(self, recent_sequence: np.ndarray) -> dict:
        """
        Predict traffic speed 30 and 60 minutes into the future.
        recent_sequence: (12, num_sensors) in actual mph, or normalized
        """
        if recent_sequence.shape != (12, self.num_sensors):
            raise ValueError(f"Expected shape (12, {self.num_sensors}), got {recent_sequence.shape}")

        if recent_sequence.mean() > 10.0:
            seq_norm = (recent_sequence - self.mean) / (self.std + 1e-6)
        else:
            seq_norm = recent_sequence

        x_tensor = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if "lstm" in self.model_type:
                p30_norm, p60_norm = self.model(x_tensor)
            else:
                p30_norm, p60_norm = self.model(x_tensor, self.norm_adj)

        p30_mph = (p30_norm.cpu().numpy()[0]) * self.std + self.mean
        p60_mph = (p60_norm.cpu().numpy()[0]) * self.std + self.mean

        alerts = self._generate_alerts(p30_mph, p60_mph)

        return {
            'predicted_speed_30min': p30_mph,
            'predicted_speed_60min': p60_mph,
            'sensor_ids': self.sensor_ids,
            'alerts': alerts
        }

    def _generate_alerts(self, p30: np.ndarray, p60: np.ndarray) -> list:
        alerts = []
        for i, sid in enumerate(self.sensor_ids):
            spd30 = p30[i]
            spd60 = p60[i]
            if spd30 < 25.0 or spd60 < 25.0:
                severity = "SEVERE"
                color = "red"
            elif spd30 < 40.0 or spd60 < 40.0:
                severity = "MODERATE"
                color = "orange"
            else:
                continue

            alerts.append({
                'sensor_id': sid,
                'severity': severity,
                'color': color,
                'speed_30min': round(float(spd30), 1),
                'speed_60min': round(float(spd60), 1),
                'message': f"Sensor #{sid}: {severity} congestion predicted (30m: {spd30:.1f} mph, 60m: {spd60:.1f} mph)"
            })
        return alerts


def main():
    predictor = TrafficPredictor(model_type="gnn")
    with open("data/processed/traffic/traffic_processed.pkl", "rb") as f:
        data = pickle.load(f)
    sample_seq = data['X_test'][0] * data['std'] + data['mean']
    res = predictor.predict(sample_seq)
    print(f"[Traffic Predictor] Prediction successful across {len(res['sensor_ids'])} sensors!")
    print(f"  30-min mean speed: {np.mean(res['predicted_speed_30min']):.2f} mph")
    print(f"  60-min mean speed: {np.mean(res['predicted_speed_60min']):.2f} mph")
    print(f"  Active Congestion Alerts: {len(res['alerts'])}")


if __name__ == "__main__":
    main()
