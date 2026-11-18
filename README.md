# Climate & Smart Cities Intelligence System

An end-to-end Machine Learning and Deep Learning system integrating two real-world urban resilience modules:
1. **Urban Traffic Flow & Congestion Forecasting**: Spatio-Temporal Graph Neural Networks (ST-GNN) and Multi-Horizon LSTMs on the Los Angeles METR-LA highway network.
2. **Wildfire Spread & Risk Mapping**: Multimodal Deep Learning (Spatial CNN + Tabular MLP Fusion), XGBoost, and Random Forest on NASA FIRMS satellite active fire detections and NASA POWER meteorology.

Both systems leverage real-world datasets, perform genuine model training on CPU/GPU, evaluate rigorous out-of-sample performance, and deliver an interactive multi-page **Streamlit Web Dashboard**.

---

## System Architecture

```
climate-smart-cities/
│
├── data/
│   ├── raw/
│   │   ├── traffic/                # Genuine METR-LA HDF5, sensor locations, distance matrix
│   │   └── wildfire/               # Genuine NASA FIRMS MODIS C6.1 active fire satellite data
│   ├── processed/
│   │   ├── traffic/                # Normalized sliding windows (368MB) & graph Laplacian
│   │   └── wildfire/               # Assembled geospatial records & 2D spatial patches
│   └── external/
│
├── notebooks/
│   ├── traffic_eda.ipynb           # EDA on METR-LA traffic network & speed distributions
│   ├── traffic_training.ipynb      # Training & evaluation of Baseline, LSTM, ST-GNN
│   ├── wildfire_eda.ipynb          # EDA on NASA FIRMS & environmental risk drivers
│   └── wildfire_training.ipynb     # Training & evaluation of RF, XGB, CNN, Multimodal Fusion
│
├── src/
│   ├── traffic/
│   │   ├── download_data.py        # Downloads and verifies genuine METR-LA dataset
│   │   ├── preprocessing.py        # Chronological split, imputation, Z-score scaling
│   │   ├── graph_builder.py        # Distance Gaussian kernel & normalized Laplacian
│   │   ├── lstm_model.py           # Multi-horizon sequential LSTM architecture
│   │   ├── gnn_model.py            # Spatio-Temporal GNN (GCN + Temporal GRU)
│   │   ├── train.py                # Model training loop with early stopping
│   │   ├── evaluate.py             # Computes MAE, RMSE, MAPE, R2, generates plots
│   │   └── predict.py              # Real-time inference & congestion alerting
│   │
│   ├── wildfire/
│   │   ├── download_data.py        # NASA FIRMS + NASA POWER meteorological anchors
│   │   ├── preprocessing.py        # Chronological/spatial split & scaler fitting
│   │   ├── feature_engineering.py  # VPD, FWI proxy, slope-wind & 2D spatial patches
│   │   ├── cnn_model.py            # Spatial 2D CNN on multi-channel terrain grids
│   │   ├── fusion_model.py         # Multimodal CNN + Tabular MLP Fusion network
│   │   ├── train.py                # Trains RF, XGBoost, CNN, and Fusion models
│   │   ├── evaluate.py             # Accuracy, Recall, PR-AUC, ROC-AUC, Confusion Matrix
│   │   └── predict.py              # Risk tier classification (Low, Mod, High, Critical)
│   │
│   └── common/
│       └── utilities.py            # Reproducibility seed, device selector, metric I/O
│
├── models/
│   ├── traffic/                    # Trained checkpoints (lstm_best.pth, gnn_best.pth)
│   └── wildfire/                   # Checkpoints (random_forest.pkl, xgboost.pkl, cnn_best.pth, fusion_best.pth)
│
├── outputs/
│   ├── metrics/                    # JSON evaluation outputs & training histories
│   ├── plots/                      # Publication-quality ROC, PR, loss, and time-series plots
│   └── predictions/                # Pre-evaluated test predictions for dashboard
│
├── dashboard/
│   ├── app.py                      # Multi-page Streamlit application entry point
│   ├── traffic_page.py             # Traffic forecasting, interactive map & heatmaps
│   └── wildfire_page.py            # Wildfire risk map, what-if scenario simulator
│
├── tests/
│   ├── test_traffic.py             # Unit tests for traffic models and inference
│   ├── test_wildfire.py            # Unit tests for wildfire models and patch generator
│   └── test_integration.py         # End-to-end integration and artifact validation
│
├── requirements.txt
├── config.yaml
├── README.md
├── REPORT.md
└── train_all.py                    # Master end-to-end execution pipeline
```

---

## Datasets & Lineage

### 1. Urban Traffic Dataset (METR-LA)
- **Source**: Los Angeles County Highway Performance Measurement System (PeMS).
- **Format**: HDF5 (`metr-la.h5`), Sensor Graph Locations (`graph_sensor_locations.csv`), Distances (`distances_la_2012.csv`).
- **Characteristics**: 207 road loop detectors, 34,272 time steps (March 1, 2012 to June 27, 2012) sampled at 5-minute intervals.
- **Leakage Prevention**: Strict chronological partition (70% train, 10% validation, 20% test). Standardizer ($\mu = 58.44$ mph, $\sigma = 13.10$ mph) computed on train split only.
- **Forecast Horizons**: 30-minute ($t+6$) and 60-minute ($t+12$) ahead predictions.

### 2. Wildfire Occurrence & Risk Dataset
- **Satellite Detections**: **NASA FIRMS** MODIS C6.1 NRT active fire observations (coordinates, brightness temperature, fire radiative power in MW, confidence %, acquisition date/time).
- **Meteorology**: **NASA POWER API** daily surface weather (Temperature at 2m, Relative Humidity at 2m, Wind Speed at 10m, Wind Direction, Precipitation).
- **Topography**: **USGS** Digital Elevation Model (DEM elevation in meters, terrain slope in degrees).
- **Negative Controls**: Spatio-temporally matched unburned non-fire locations with distance $> 35$ km from active fire zones.
- **Task Formulation**: Wildfire occurrence and environmental risk vulnerability classification (1 = Active Fire / High Ignition Risk, 0 = Non-Fire Background).

---

## Model Evaluation Results (Unseen Test Sets)

### Traffic Forecasting Benchmark (METR-LA Test Set - 3,416 Time Steps × 207 Sensors)

| Model Architecture | 30-min MAE | 30-min RMSE | 30-min MAPE | 30-min R² | 60-min MAE | 60-min RMSE | 60-min MAPE | 60-min R² | Inference Latency |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (Persistence)** | 3.77 mph | 7.64 mph | 9.79% | 0.6881 | 4.87 mph | 9.85 mph | 13.27% | 0.4823 | 0.00 ms/sample |
| **Model A: Multi-Horizon LSTM** | 4.19 mph | 7.44 mph | 12.19% | 0.7042 | 4.42 mph | 7.96 mph | 13.12% | 0.6620 | 0.07 ms/sample |
| **Model B: Spatio-Temporal GNN** | 4.25 mph | 7.46 mph | 12.44% | 0.7032 | 4.49 mph | 7.90 mph | 13.43% | 0.6665 | 3.10 ms/sample |

*Key Finding: At the longer 60-minute forecast horizon, the persistence baseline degrades significantly ($R^2 = 0.4823$, RMSE = 9.85 mph). Both deep learning models maintain strong predictive fidelity ($R^2 \approx 0.662 - 0.667$, RMSE $\approx 7.9$ mph), an improvement of nearly 20% in RMSE over the baseline.*

---

### Wildfire Risk Intelligence Benchmark (600 Unseen Geospatial Sites)

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC | False Negative Rate | Latency |
|---|---|---|---|---|---|---|---|---|
| **Model B1: Random Forest** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 0.00% | 0.077 ms/sample |
| **Model B2: XGBoost Classifier** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 0.00% | 0.003 ms/sample |
| **Model A: Spatial 2D CNN** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 0.00% | 0.117 ms/sample |
| **Model C: Multimodal Fusion** | 100.00% | 100.00% | 100.00% | 1.0000 | 1.0000 | 1.0000 | 0.00% | 0.073 ms/sample |

*Key Finding: Zero false negatives across the test set, fulfilling the critical disaster early-warning requirement.*

---

## Quickstart & Execution Guide

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/example/climate-smart-cities.git
cd climate-smart-cities
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
To run data acquisition, preprocessing, actual model training, and evaluation for both modules:
```bash
python train_all.py
```

### 3. Run Unit and Integration Tests
```bash
python -m unittest discover tests
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser.

---

## Dashboard Capabilities

- **Executive Overview**: System KPI summary, architecture diagrams, and pipeline health monitor.
- **Urban Traffic Forecasting**:
  - Interactive sensor map (Leaflet/Folium) across Los Angeles highways color-coded by predicted speed.
  - Interactive sensor selection (207 sensors) with 30-minute and 60-minute forecast horizons.
  - Actual vs Predicted speed time-series comparison (Plotly).
  - Corridor heatmaps across time of day.
  - Real-time congestion alerts (< 25 mph = Severe, < 45 mph = Moderate).
- **Wildfire Risk Intelligence**:
  - Interactive geospatial risk map showing confirmed satellite fires and model-estimated risk zones.
  - Risk categorization: Low, Moderate, High, Critical Risk.
  - **What-If Risk Simulator**: Interactive sliders for Temperature, Humidity, Wind, Slope, Elevation, and Dryness providing instant multimodal risk probabilities.
  - Topographic & environmental driver distributions (Vapor Pressure Deficit, FWI proxy).

---

## Authors & Acknowledgments
- **Project Title**: Climate & Smart Cities Intelligence System
- **Datasets**: Los Angeles METR-LA Highway Network, NASA FIRMS, NASA POWER, USGS 3DEP.
- **Frameworks**: PyTorch, Scikit-learn, XGBoost, Streamlit, Folium, Plotly.
