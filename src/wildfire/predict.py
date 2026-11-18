"""
Inference & Risk Mapping Pipeline for Wildfire Intelligence.
Predicts wildfire risk category (Low, Moderate, High, Critical) and probability.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, get_device
from src.wildfire.cnn_model import WildfireCNN
from src.wildfire.fusion_model import WildfireMultimodalFusion
from src.wildfire.feature_engineering import generate_spatial_patch


class WildfirePredictor:
    def __init__(self, config_path: str = "config.yaml", model_type: str = "fusion"):
        self.config = load_config(config_path)
        self.device = get_device(self.config["project"]["device"])
        self.model_type = model_type.lower()
        self.models_dir = self.config["wildfire"]["models_dir"]
        self.proc_dir = self.config["wildfire"]["processed_dir"]

        with open(os.path.join(self.proc_dir, "wildfire_engineered.pkl"), "rb") as f:
            data = pickle.load(f)
            self.tabular_cols = data['tabular_cols']
            self.tab_scaler = data['tab_scaler']
            self.tab_dim = len(self.tabular_cols)

        # Load models
        if "fusion" in self.model_type:
            self.model = WildfireMultimodalFusion(tabular_dim=self.tab_dim, in_channels=5, num_classes=2).to(self.device)
            ckpt = torch.load(os.path.join(self.models_dir, "fusion_best.pth"), map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
        elif "cnn" in self.model_type:
            self.model = WildfireCNN(in_channels=5, num_classes=2).to(self.device)
            ckpt = torch.load(os.path.join(self.models_dir, "cnn_best.pth"), map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
        elif "xgb" in self.model_type:
            with open(os.path.join(self.models_dir, "xgboost.pkl"), "rb") as f:
                self.model = pickle.load(f)
        else:  # Random Forest
            with open(os.path.join(self.models_dir, "random_forest.pkl"), "rb") as f:
                self.model = pickle.load(f)

    def classify_risk(self, prob: float) -> tuple:
        """Map continuous probability into standardized risk tier."""
        if prob < 0.25:
            return "Low Risk", "#2ecc71"
        elif prob < 0.50:
            return "Moderate Risk", "#f1c40f"
        elif prob < 0.75:
            return "High Risk", "#e67e22"
        else:
            return "Critical Risk", "#e74c3c"

    def predict_point(self, env_data: dict) -> dict:
        """
        Predict wildfire risk from raw environmental variables.
        env_data dict must contain:
          temp_celsius, relative_humidity_pct, wind_speed_ms, wind_direction_deg,
          precipitation_mm, elevation_m, slope_deg, vegetation_dryness, ndvi
        """
        # Calculate derived indices
        temp = env_data.get('temp_celsius', 28.0)
        rh = env_data.get('relative_humidity_pct', 25.0)
        ws = env_data.get('wind_speed_ms', 5.0)
        precip = env_data.get('precipitation_mm', 0.0)
        slope = env_data.get('slope_deg', 12.0)
        veg_dry = env_data.get('vegetation_dryness', 0.7)
        ndvi = env_data.get('ndvi', 0.3)

        e_s = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
        vpd = float(e_s * (1.0 - np.clip(rh, 1.0, 100.0) / 100.0))
        fwi_proxy = float((temp * ws) / (rh + 1.0) * np.exp(-0.15 * precip))
        slope_wind = float(slope * ws)
        fuel_dry = float(veg_dry * (1.0 - ndvi))

        features = [
            env_data.get('elevation_m', 600.0), slope, temp, rh, ws,
            env_data.get('wind_direction_deg', 180.0), precip,
            veg_dry, ndvi, vpd, fwi_proxy, slope_wind, fuel_dry
        ]

        scaled_tab = self.tab_scaler.transform([features]).astype(np.float32)

        if "fusion" in self.model_type or "cnn" in self.model_type:
            # Generate local spatial patch
            row_dict = {**env_data, 'fire_occurrence': 0}
            patch = generate_spatial_patch(row_dict, patch_size=16)
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if "fusion" in self.model_type:
                    tab_tensor = torch.tensor(scaled_tab, dtype=torch.float32).to(self.device)
                    logits = self.model(patch_tensor, tab_tensor)
                else:
                    logits = self.model(patch_tensor)
                prob = float(F.softmax(logits, dim=1)[0, 1].cpu().item())
        else:
            prob = float(self.model.predict_proba(scaled_tab)[0, 1])

        tier, color = self.classify_risk(prob)

        return {
            'risk_probability': round(prob, 4),
            'risk_percentage': round(prob * 100, 1),
            'risk_category': tier,
            'color': color,
            'derived_vpd_kpa': round(vpd, 2),
            'derived_fwi_proxy': round(fwi_proxy, 2),
            'active_fire_status': "Model Estimation (No Verified Ignition)" if prob < 0.75 else "Elevated Ignition Vulnerability"
        }


def main():
    predictor = WildfirePredictor(model_type="fusion")
    test_sample = {
        'temp_celsius': 36.5,
        'relative_humidity_pct': 14.0,
        'wind_speed_ms': 8.5,
        'wind_direction_deg': 240.0,
        'precipitation_mm': 0.0,
        'elevation_m': 1100.0,
        'slope_deg': 22.0,
        'vegetation_dryness': 0.88,
        'ndvi': 0.18
    }
    res = predictor.predict_point(test_sample)
    print("\n[Wildfire Predictor Test Inference]")
    print(f"  Risk Probability: {res['risk_probability']} ({res['risk_percentage']}%)")
    print(f"  Risk Category: {res['risk_category']}")
    print(f"  Vapor Pressure Deficit: {res['derived_vpd_kpa']} kPa")
    print(f"  Fire Weather Index Proxy: {res['derived_fwi_proxy']}")


if __name__ == "__main__":
    main()
