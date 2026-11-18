"""
Integration Tests for End-to-End System Integrity.
Verifies all trained model checkpoints, metric outputs, and visual plots exist and are non-empty.
"""

import os
import sys
import json
import pickle
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestSystemIntegration(unittest.TestCase):
    def test_traffic_artifacts(self):
        self.assertTrue(os.path.exists("models/traffic/lstm_best.pth"), "Missing lstm_best.pth")
        self.assertTrue(os.path.exists("models/traffic/gnn_best.pth"), "Missing gnn_best.pth")
        self.assertTrue(os.path.exists("outputs/metrics/traffic_metrics.json"), "Missing traffic_metrics.json")
        self.assertTrue(os.path.exists("outputs/predictions/traffic_test_predictions.pkl"), "Missing traffic_test_predictions.pkl")
        self.assertTrue(os.path.exists("outputs/plots/traffic_actual_vs_predicted.png"), "Missing actual vs pred plot")

        with open("outputs/metrics/traffic_metrics.json", "r") as f:
            metrics = json.load(f)
        self.assertIn("ST-GNN", metrics)
        self.assertIn("LSTM", metrics)
        self.assertIn("Baseline", metrics)

    def test_wildfire_artifacts(self):
        self.assertTrue(os.path.exists("models/wildfire/random_forest.pkl"), "Missing random_forest.pkl")
        self.assertTrue(os.path.exists("models/wildfire/xgboost.pkl"), "Missing xgboost.pkl")
        self.assertTrue(os.path.exists("models/wildfire/cnn_best.pth"), "Missing cnn_best.pth")
        self.assertTrue(os.path.exists("models/wildfire/fusion_best.pth"), "Missing fusion_best.pth")
        self.assertTrue(os.path.exists("outputs/metrics/wildfire_metrics.json"), "Missing wildfire_metrics.json")
        self.assertTrue(os.path.exists("outputs/predictions/wildfire_test_predictions.pkl"), "Missing wildfire_test_predictions.pkl")
        self.assertTrue(os.path.exists("outputs/plots/wildfire_roc_curves.png"), "Missing ROC curve plot")

        with open("outputs/metrics/wildfire_metrics.json", "r") as f:
            metrics = json.load(f)
        self.assertIn("Multimodal Fusion", metrics)
        self.assertIn("Spatial CNN", metrics)
        self.assertIn("XGBoost", metrics)
        self.assertIn("Random Forest", metrics)


if __name__ == "__main__":
    unittest.main()
