"""
Unit Tests for Wildfire Module.
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.wildfire.cnn_model import WildfireCNN
from src.wildfire.fusion_model import WildfireMultimodalFusion
from src.wildfire.feature_engineering import generate_spatial_patch
from src.wildfire.predict import WildfirePredictor


class TestWildfireModule(unittest.TestCase):
    def test_spatial_patch_generation(self):
        sample = {
            'elevation_m': 800.0,
            'slope_deg': 15.0,
            'temp_celsius': 32.0,
            'relative_humidity_pct': 20.0,
            'vegetation_dryness': 0.8,
            'wind_direction_deg': 180.0,
            'fire_occurrence': 1
        }
        patch = generate_spatial_patch(sample, patch_size=16)
        self.assertEqual(patch.shape, (5, 16, 16))
        self.assertTrue(np.all(np.isfinite(patch)))

    def test_cnn_forward(self):
        model = WildfireCNN(in_channels=5, num_classes=2)
        x = torch.randn(4, 5, 16, 16)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 2))

    def test_fusion_forward(self):
        model = WildfireMultimodalFusion(tabular_dim=13, in_channels=5, num_classes=2)
        x_img = torch.randn(4, 5, 16, 16)
        x_tab = torch.randn(4, 13)
        logits = model(x_img, x_tab)
        self.assertEqual(logits.shape, (4, 2))

    def test_predictor_api(self):
        predictor = WildfirePredictor(model_type="fusion")
        test_env = {
            'temp_celsius': 30.0,
            'relative_humidity_pct': 25.0,
            'wind_speed_ms': 6.0,
            'wind_direction_deg': 90.0,
            'precipitation_mm': 0.0,
            'elevation_m': 500.0,
            'slope_deg': 10.0,
            'vegetation_dryness': 0.7,
            'ndvi': 0.25
        }
        res = predictor.predict_point(test_env)
        self.assertIn('risk_probability', res)
        self.assertIn('risk_category', res)
        self.assertIn(res['risk_category'], ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"])
        self.assertTrue(0.0 <= res['risk_probability'] <= 1.0)


if __name__ == "__main__":
    unittest.main()
