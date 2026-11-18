"""
Feature Engineering for Wildfire Occurrence & Risk Modeling.
Synthesizes 5-band multi-spectral satellite imagery patches:
  Band 1: Red Visible Reflectance (0.64 um)
  Band 2: Near-Infrared NIR (0.86 um - Canopy Biomass)
  Band 3: Short-Wave Infrared SWIR (1.6 um - Fuel Moisture)
  Band 4: Thermal Infrared Brightness (11.0 um - Surface Temperature)
  Band 5: Topographic Digital Elevation Model (DEM)
"""

import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, set_seed


def calculate_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute atmospheric and topographic wildfire risk indices."""
    df_feat = df.copy()

    # 1. Saturated Vapor Pressure e_s (kPa) and Vapor Pressure Deficit (VPD)
    temp = df_feat['temp_celsius'].values
    rh = df_feat['relative_humidity_pct'].values
    e_s = 0.61078 * np.exp((17.27 * temp) / (temp + 237.3))
    vpd = e_s * (1.0 - np.clip(rh, 1.0, 100.0) / 100.0)
    df_feat['vpd_kpa'] = np.clip(vpd, 0.0, 8.0)

    # 2. Fire Weather Index (FWI) proxy
    ws = df_feat['wind_speed_ms'].values
    precip = df_feat['precipitation_mm'].values
    fwi_proxy = (temp * ws) / (rh + 1.0) * np.exp(-0.15 * precip)
    df_feat['fwi_proxy'] = np.clip(fwi_proxy, 0.0, 50.0)

    # 3. Slope-Wind Interaction
    slope = df_feat['slope_deg'].values
    df_feat['slope_wind_interaction'] = slope * ws

    # 4. Fuel Moisture Deficit (Vegetation dryness combined with NDVI)
    veg_dry = df_feat['vegetation_dryness'].values
    ndvi = df_feat['ndvi'].values
    df_feat['fuel_dryness_index'] = veg_dry * (1.0 - ndvi)

    return df_feat


def generate_spatial_patch(row: dict, patch_size: int = 16) -> np.ndarray:
    """
    Construct a (5, patch_size, patch_size) multi-spectral satellite imagery patch:
      Band 1: Red Visible Reflectance (0.64 um)
      Band 2: Near-Infrared NIR (0.86 um - Vegetation Health)
      Band 3: Short-Wave Infrared SWIR (1.6 um - Fuel Moisture Absorption)
      Band 4: Thermal Infrared Brightness (11.0 um - MODIS Ch 31)
      Band 5: Topographic DEM Elevation Model
    """
    elev = float(row.get('elevation_m', 600.0))
    slope = float(row.get('slope_deg', 12.0))
    temp = float(row.get('temp_celsius', 28.0))
    rh = float(row.get('relative_humidity_pct', 25.0))
    veg_dry = float(row.get('vegetation_dryness', 0.65))
    ndvi = float(row.get('ndvi', 0.35))
    wind_dir = float(row.get('wind_direction_deg', 180.0))
    is_fire = int(row.get('fire_occurrence', 0))

    patch = np.zeros((5, patch_size, patch_size), dtype=np.float32)

    half = patch_size / 2.0
    y, x = np.ogrid[-half:half, -half:half]
    dist_sq = x*x + y*y

    # Spatial texture / Perlin-like background noise
    texture = np.sin(x * 0.7) * np.cos(y * 0.7) * 0.08 + np.sin(x * 1.5 + y * 1.2) * 0.04

    # Band 5: Topographic Elevation (DEM) with terrain gradient
    patch[4] = elev + slope * (x * 0.6 + y * 0.4) + np.random.normal(0, 3.0, (patch_size, patch_size))

    # Band 1: Red Visible Reflectance (Soil & dry vegetation have higher red reflectance ~ 0.15 - 0.35)
    base_red = 0.12 + 0.18 * veg_dry + texture
    patch[0] = np.clip(base_red, 0.05, 0.60)

    # Band 2: Near-Infrared NIR (Dense healthy vegetation has high NIR ~ 0.4 - 0.7; scorched/dry has lower NIR)
    base_nir = 0.18 + 0.50 * ndvi - 0.10 * veg_dry + texture
    patch[1] = np.clip(base_nir, 0.08, 0.85)

    # Band 3: SWIR (Short-Wave Infrared: dry canopy & active fire zones have high SWIR reflectance)
    base_swir = 0.15 + 0.35 * veg_dry - (rh / 200.0)
    if is_fire == 1:
        swir_anomaly = 0.35 * np.exp(-dist_sq / 16.0)
        patch[2] = np.clip(base_swir + swir_anomaly + texture, 0.10, 0.95)
    else:
        patch[2] = np.clip(base_swir + texture, 0.10, 0.80)

    # Band 4: Thermal Infrared Brightness (Kelvin)
    # Background temperature ~ 290 - 315 K; active fire center ~ 330 - 380 K
    rad = math.radians(wind_dir)
    dir_x, dir_y = math.sin(rad), -math.cos(rad)
    wind_plume = (x * dir_x + y * dir_y) / (half + 1e-5)

    kelvin_bg = 273.15 + temp
    if is_fire == 1:
        thermal_anomaly = 45.0 * np.exp(-dist_sq / 14.0) + 8.0 * np.clip(wind_plume, 0, 1)
        patch[3] = kelvin_bg + thermal_anomaly + np.random.normal(0, 1.5, (patch_size, patch_size))
    else:
        patch[3] = kelvin_bg + np.sin(y * 0.3) * 2.0 + np.random.normal(0, 1.2, (patch_size, patch_size))

    return patch


def build_engineered_dataset(proc_dir: str, patch_size: int = 16):
    """Generate domain features and multispectral spatial patches for training."""
    set_seed(42)
    prep_path = os.path.join(proc_dir, "wildfire_preprocessed.pkl")
    with open(prep_path, "rb") as f:
        prep_data = pickle.load(f)

    train_df = calculate_domain_features(prep_data['train_df'])
    val_df = calculate_domain_features(prep_data['val_df'])
    test_df = calculate_domain_features(prep_data['test_df'])

    tabular_cols = [
        'elevation_m', 'slope_deg', 'temp_celsius', 'relative_humidity_pct',
        'wind_speed_ms', 'wind_direction_deg', 'precipitation_mm',
        'vegetation_dryness', 'ndvi', 'vpd_kpa', 'fwi_proxy',
        'slope_wind_interaction', 'fuel_dryness_index'
    ]

    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(train_df[tabular_cols].values).astype(np.float32)
    X_val_tab = scaler.transform(val_df[tabular_cols].values).astype(np.float32)
    X_test_tab = scaler.transform(test_df[tabular_cols].values).astype(np.float32)

    print(f"[Wildfire Feature Eng] Synthesizing {len(train_df)} multispectral spatial patches (5 bands x {patch_size}x{patch_size})...")
    X_train_img = np.array([generate_spatial_patch(row, patch_size) for _, row in train_df.iterrows()], dtype=np.float32)
    print(f"[Wildfire Feature Eng] Synthesizing {len(val_df)} val multispectral patches...")
    X_val_img = np.array([generate_spatial_patch(row, patch_size) for _, row in val_df.iterrows()], dtype=np.float32)
    print(f"[Wildfire Feature Eng] Synthesizing {len(test_df)} test multispectral patches...")
    X_test_img = np.array([generate_spatial_patch(row, patch_size) for _, row in test_df.iterrows()], dtype=np.float32)

    # Standardize image channels across train statistics
    for c in range(5):
        c_mean = X_train_img[:, c].mean()
        c_std = X_train_img[:, c].std() + 1e-6
        X_train_img[:, c] = (X_train_img[:, c] - c_mean) / c_std
        X_val_img[:, c] = (X_val_img[:, c] - c_mean) / c_std
        X_test_img[:, c] = (X_test_img[:, c] - c_mean) / c_std

    engineered_data = {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'tabular_cols': tabular_cols,
        'tab_scaler': scaler,
        'X_train_tab': X_train_tab,
        'X_val_tab': X_val_tab,
        'X_test_tab': X_test_tab,
        'X_train_img': X_train_img,
        'X_val_img': X_val_img,
        'X_test_img': X_test_img,
        'y_train': prep_data['y_train'],
        'y_val': prep_data['y_val'],
        'y_test': prep_data['y_test']
    }

    out_file = os.path.join(proc_dir, "wildfire_engineered.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(engineered_data, f, protocol=4)

    print(f"[Wildfire Feature Eng] Saved engineered dataset to {out_file}")
    return out_file


def main():
    config = load_config()
    proc_dir = config["wildfire"]["processed_dir"]
    patch_size = config["wildfire"]["preprocessing"]["patch_size"]
    build_engineered_dataset(proc_dir, patch_size)


if __name__ == "__main__":
    main()
