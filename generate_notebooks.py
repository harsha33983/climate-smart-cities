"""
Script to generate the 4 required Jupyter Notebooks for the project.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }


def build_all_notebooks():
    os.makedirs("notebooks", exist_ok=True)

    # 1. Traffic EDA Notebook
    nb_traffic_eda = make_notebook([
        make_markdown_cell([
            "# Module 1: Urban Traffic Flow - Exploratory Data Analysis (EDA)",
            "### METR-LA Highway Traffic Sensor Network",
            "This notebook inspects the real-world Los Angeles traffic sensor dataset, exploring spatial topology, speed distributions, missing values, and temporal autocorrelation."
        ]),
        make_code_cell([
            "import os, sys, pickle",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "sys.path.append('..')",
            "from src.traffic.download_data import load_metr_la_hdf5"
        ]),
        make_markdown_cell([
            "## 1. Load Raw METR-LA Dataset & Road Network Metadata"
        ]),
        make_code_cell([
            "df = load_metr_la_hdf5('../data/raw/traffic/metr-la.h5')",
            "print('Dataset Shape:', df.shape)",
            "print('Sensors count:', df.shape[1])",
            "print('Time range:', df.index[0], 'to', df.index[-1])",
            "df.head()"
        ]),
        make_markdown_cell([
            "## 2. Traffic Speed Summary Statistics & Distributions"
        ]),
        make_code_cell([
            "print('Overall Mean Speed:', df.values.mean(), 'mph')",
            "print('Overall Std Speed:', df.values.std(), 'mph')",
            "print('Zeros (Sensor dropouts):', (df.values == 0).sum())",
            "plt.figure(figsize=(10, 4), dpi=120)",
            "sns.histplot(df.values.flatten()[:50000], bins=50, kde=True, color='#2980b9')",
            "plt.title('Traffic Speed Distribution across METR-LA Sensors (mph)')",
            "plt.xlabel('Speed (mph)')",
            "plt.ylabel('Frequency')",
            "plt.show()"
        ]),
        make_markdown_cell([
            "## 3. Spatial Road Network Graph Adjacency"
        ]),
        make_code_cell([
            "with open('../data/processed/traffic/traffic_graph.pkl', 'rb') as f:",
            "    graph = pickle.load(f)",
            "adj = graph['adj_matrix']",
            "plt.figure(figsize=(8, 6), dpi=120)",
            "plt.imshow(adj, cmap='viridis')",
            "plt.colorbar(label='Gaussian Distance Weight')",
            "plt.title('METR-LA Road Network Adjacency Matrix (207 x 207 Sensors)')",
            "plt.xlabel('Sensor Index')",
            "plt.ylabel('Sensor Index')",
            "plt.show()"
        ])
    ])

    with open("notebooks/traffic_eda.ipynb", "w") as f:
        json.dump(nb_traffic_eda, f, indent=2)

    # 2. Traffic Training Notebook
    nb_traffic_train = make_notebook([
        make_markdown_cell([
            "# Module 1: Traffic Forecasting Model Training & Comparative Benchmark",
            "### Training Baseline, Multi-Horizon LSTM, and Spatio-Temporal Graph Neural Network (ST-GNN)"
        ]),
        make_code_cell([
            "import os, sys, pickle, json",
            "import numpy as np",
            "import torch",
            "import matplotlib.pyplot as plt",
            "sys.path.append('..')",
            "from src.common.utilities import load_metrics"
        ]),
        make_markdown_cell([
            "## 1. Load Preprocessed Data Sequences & Graph Laplacian"
        ]),
        make_code_cell([
            "with open('../data/processed/traffic/traffic_processed.pkl', 'rb') as f:",
            "    data = pickle.load(f)",
            "with open('../data/processed/traffic/traffic_graph.pkl', 'rb') as f:",
            "    graph = pickle.load(f)",
            "print('Train sequences:', data['X_train'].shape)",
            "print('Val sequences:  ', data['X_val'].shape)",
            "print('Test sequences: ', data['X_test'].shape)"
        ]),
        make_markdown_cell([
            "## 2. Inspect Training Curves (MSE Loss)"
        ]),
        make_code_cell([
            "hist = load_metrics('../outputs/metrics/traffic_training_history.json')",
            "plt.figure(figsize=(10, 4), dpi=120)",
            "plt.plot(hist['lstm_history']['val_loss'], label='LSTM Val Loss', color='#ff7f0e', linewidth=2)",
            "plt.plot(hist['gnn_history']['val_loss'], label='ST-GNN Val Loss', color='#2ca02c', linewidth=2)",
            "plt.title('Validation Loss Convergence across Epochs')",
            "plt.xlabel('Epochs')",
            "plt.ylabel('MSE Loss')",
            "plt.legend()",
            "plt.grid(True, linestyle='--', alpha=0.5)",
            "plt.show()"
        ]),
        make_markdown_cell([
            "## 3. Test Set Evaluation Benchmark"
        ]),
        make_code_cell([
            "metrics = load_metrics('../outputs/metrics/traffic_metrics.json')",
            "import pandas as pd",
            "pd.DataFrame(metrics)"
        ])
    ])

    with open("notebooks/traffic_training.ipynb", "w") as f:
        json.dump(nb_traffic_train, f, indent=2)

    # 3. Wildfire EDA Notebook
    nb_wildfire_eda = make_notebook([
        make_markdown_cell([
            "# Module 2: Wildfire Spread & Risk - Exploratory Data Analysis (EDA)",
            "### NASA FIRMS Satellite Detections + Environmental & Topographic Analysis",
            "Exploration of genuine NASA FIRMS MODIS thermal detections, NASA POWER meteorology (temp, humidity, wind), and USGS topography."
        ]),
        make_code_cell([
            "import os, sys, pickle",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns"
        ]),
        make_markdown_cell([
            "## 1. Load Geospatial Wildfire Dataset"
        ]),
        make_code_cell([
            "df = pd.read_csv('../data/processed/wildfire/wildfire_dataset.csv')",
            "print('Dataset rows:', len(df))",
            "print('Class balance:', df['fire_occurrence'].value_counts())",
            "df.head()"
        ]),
        make_markdown_cell([
            "## 2. Environmental Factors: Fire vs Non-Fire Sites"
        ]),
        make_code_cell([
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)",
            "sns.boxplot(data=df, x='fire_occurrence', y='temp_celsius', ax=axes[0], palette=['#3498db', '#e74c3c'])",
            "axes[0].set_title('Temperature (°C)')",
            "sns.boxplot(data=df, x='fire_occurrence', y='relative_humidity_pct', ax=axes[1], palette=['#3498db', '#e74c3c'])",
            "axes[1].set_title('Relative Humidity (%)')",
            "sns.boxplot(data=df, x='fire_occurrence', y='vegetation_dryness', ax=axes[2], palette=['#3498db', '#e74c3c'])",
            "axes[2].set_title('Vegetation Dryness')",
            "plt.tight_layout()",
            "plt.show()"
        ]),
        make_markdown_cell([
            "## 3. Spatial Distribution of Detections"
        ]),
        make_code_cell([
            "plt.figure(figsize=(10, 5), dpi=120)",
            "plt.scatter(df['longitude'], df['latitude'], c=df['fire_occurrence'], cmap='coolwarm', s=25, alpha=0.7)",
            "plt.colorbar(label='Fire Occurrence (1=Fire, 0=Background)')",
            "plt.title('Geospatial Coordinates of Observations')",
            "plt.xlabel('Longitude')",
            "plt.ylabel('Latitude')",
            "plt.show()"
        ])
    ])

    with open("notebooks/wildfire_eda.ipynb", "w") as f:
        json.dump(nb_wildfire_eda, f, indent=2)

    # 4. Wildfire Training Notebook
    nb_wildfire_train = make_notebook([
        make_markdown_cell([
            "# Module 2: Wildfire Risk Model Training & Comparative Benchmark",
            "### Training Random Forest, XGBoost, Spatial CNN, and Multimodal Fusion"
        ]),
        make_code_cell([
            "import os, sys, pickle, json",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "sys.path.append('..')",
            "from src.common.utilities import load_metrics"
        ]),
        make_markdown_cell([
            "## 1. Load Engineered Tabular & Multi-Channel Spatial Patches"
        ]),
        make_code_cell([
            "with open('../data/processed/wildfire/wildfire_engineered.pkl', 'rb') as f:",
            "    data = pickle.load(f)",
            "print('Tabular Train shape:', data['X_train_tab'].shape)",
            "print('Image Train shape:  ', data['X_train_img'].shape)",
            "print('Tabular Features:   ', data['tabular_cols'])"
        ]),
        make_markdown_cell([
            "## 2. Test Set Evaluation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)"
        ]),
        make_code_cell([
            "metrics = load_metrics('../outputs/metrics/wildfire_metrics.json')",
            "df_res = pd.DataFrame(metrics).T",
            "df_res[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc', 'false_negative_rate']]"
        ]),
        make_markdown_cell([
            "## 3. Inspect ROC & Precision-Recall Diagnostic Curves"
        ]),
        make_code_cell([
            "from IPython.display import Image, display",
            "display(Image(filename='../outputs/plots/wildfire_roc_curves.png'))",
            "display(Image(filename='../outputs/plots/wildfire_pr_curves.png'))"
        ])
    ])

    with open("notebooks/wildfire_training.ipynb", "w") as f:
        json.dump(nb_wildfire_train, f, indent=2)

    print("All 4 Jupyter notebooks successfully created in notebooks/ directory!")


if __name__ == "__main__":
    build_all_notebooks()
