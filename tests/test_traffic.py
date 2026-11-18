"""
Unit Tests for Traffic Module.
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common.utilities import load_config
from src.traffic.lstm_model import TrafficLSTM
from src.traffic.gnn_model import TrafficGNN
from src.traffic.predict import TrafficPredictor


class TestTrafficModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.num_sensors = 207
        cls.seq_len = 12

    def test_lstm_forward(self):
        model = TrafficLSTM(num_sensors=self.num_sensors, hidden_dim=64, num_layers=1)
        x = torch.randn(4, self.seq_len, self.num_sensors)
        p30, p60 = model(x)
        self.assertEqual(p30.shape, (4, self.num_sensors))
        self.assertEqual(p60.shape, (4, self.num_sensors))

    def test_gnn_forward(self):
        model = TrafficGNN(num_sensors=self.num_sensors, gcn_dim=16, rnn_dim=32)
        x = torch.randn(4, self.seq_len, self.num_sensors)
        norm_adj = torch.eye(self.num_sensors)
        p30, p60 = model(x, norm_adj)
        self.assertEqual(p30.shape, (4, self.num_sensors))
        self.assertEqual(p60.shape, (4, self.num_sensors))

    def test_predictor_api(self):
        predictor = TrafficPredictor(model_type="lstm")
        dummy_seq = np.full((12, self.num_sensors), 55.0, dtype=np.float32)
        res = predictor.predict(dummy_seq)
        self.assertIn('predicted_speed_30min', res)
        self.assertIn('predicted_speed_60min', res)
        self.assertEqual(len(res['predicted_speed_30min']), self.num_sensors)
        self.assertIsInstance(res['alerts'], list)


if __name__ == "__main__":
    unittest.main()
