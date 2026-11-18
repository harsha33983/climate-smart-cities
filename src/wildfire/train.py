"""
Training Pipeline for Wildfire Risk Models: Random Forest, XGBoost, CNN, and Multimodal Fusion.
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
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, set_seed, get_device, save_metrics
from src.wildfire.cnn_model import WildfireCNN
from src.wildfire.fusion_model import WildfireMultimodalFusion


def train_tabular_models(data: dict, models_dir: str) -> dict:
    """Train Random Forest and XGBoost classifiers on tabular environmental data."""
    print("\n--- Training Model B1: Random Forest Classifier ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(data['X_train_tab'], data['y_train'])
    rf_time = time.time() - t0
    rf_path = os.path.join(models_dir, "random_forest.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    print(f"[Random Forest] Trained in {rf_time:.2f}s | Saved to {rf_path}")

    print("\n--- Training Model B2: XGBoost Classifier ---")
    t0 = time.time()
    scale_pos = (len(data['y_train']) - sum(data['y_train'])) / max(1, sum(data['y_train']))
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        scale_pos_weight=scale_pos,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    xgb_clf.fit(
        data['X_train_tab'], data['y_train'],
        eval_set=[(data['X_val_tab'], data['y_val'])],
        verbose=False
    )
    xgb_time = time.time() - t0
    xgb_path = os.path.join(models_dir, "xgboost.pkl")
    with open(xgb_path, "wb") as f:
        pickle.dump(xgb_clf, f)
    print(f"[XGBoost] Trained in {xgb_time:.2f}s | Saved to {xgb_path}")

    return {'rf_time': rf_time, 'xgb_time': xgb_time}


def train_cnn(data: dict, config: dict, device: torch.device) -> tuple:
    """Train Spatial CNN on 2D environmental patches."""
    print("\n--- Training Model A: Spatial CNN ---")
    models_dir = config["wildfire"]["models_dir"]
    batch_size = config["wildfire"]["training"]["batch_size"]
    epochs = config["wildfire"]["training"]["epochs"]
    lr = config["wildfire"]["training"]["lr"]
    patience = config["wildfire"]["training"]["early_stopping_patience"]

    train_ds = TensorDataset(
        torch.tensor(data['X_train_img'], dtype=torch.float32),
        torch.tensor(data['y_train'], dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(data['X_val_img'], dtype=torch.float32),
        torch.tensor(data['y_val'], dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WildfireCNN(in_channels=5, num_classes=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_path = os.path.join(models_dir, "cnn_best.pth")
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(x_b)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"CNN Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[CNN] Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    print(f"[CNN] Training finished in {train_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    return model, history, best_path


def train_fusion(data: dict, config: dict, device: torch.device) -> tuple:
    """Train Multimodal CNN + Tabular Fusion model."""
    print("\n--- Training Model C: Multimodal CNN + Tabular Fusion ---")
    models_dir = config["wildfire"]["models_dir"]
    batch_size = config["wildfire"]["training"]["batch_size"]
    epochs = config["wildfire"]["training"]["epochs"]
    lr = config["wildfire"]["training"]["lr"]
    patience = config["wildfire"]["training"]["early_stopping_patience"]

    train_ds = TensorDataset(
        torch.tensor(data['X_train_img'], dtype=torch.float32),
        torch.tensor(data['X_train_tab'], dtype=torch.float32),
        torch.tensor(data['y_train'], dtype=torch.long)
    )
    val_ds = TensorDataset(
        torch.tensor(data['X_val_img'], dtype=torch.float32),
        torch.tensor(data['X_val_tab'], dtype=torch.float32),
        torch.tensor(data['y_val'], dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    tabular_dim = data['X_train_tab'].shape[1]
    model = WildfireMultimodalFusion(tabular_dim=tabular_dim, in_channels=5, num_classes=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_path = os.path.join(models_dir, "fusion_best.pth")
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_img_b, x_tab_b, y_b in train_loader:
            x_img_b, x_tab_b, y_b = x_img_b.to(device), x_tab_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_img_b, x_tab_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_img_b)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_img_b, x_tab_b, y_b in val_loader:
                x_img_b, x_tab_b, y_b = x_img_b.to(device), x_tab_b.to(device), y_b.to(device)
                logits = model(x_img_b, x_tab_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(x_img_b)
        val_loss /= len(val_ds)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Fusion Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': val_loss}, best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Fusion] Early stopping at epoch {epoch}")
                break

    train_time = time.time() - t0
    print(f"[Fusion] Training finished in {train_time:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    return model, history, best_path


def main():
    set_seed(42)
    config = load_config()
    device = get_device(config["project"]["device"])
    print(f"[Wildfire Training] Device: {device}")

    proc_dir = config["wildfire"]["processed_dir"]
    models_dir = config["wildfire"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(proc_dir, "wildfire_engineered.pkl"), "rb") as f:
        data = pickle.load(f)

    # 1. Train Tabular (Random Forest & XGBoost)
    tab_times = train_tabular_models(data, models_dir)

    # 2. Train CNN
    cnn_model, cnn_hist, cnn_path = train_cnn(data, config, device)

    # 3. Train Multimodal Fusion
    fusion_model, fusion_hist, fusion_path = train_fusion(data, config, device)

    # Save training histories
    history_file = os.path.join(config["wildfire"]["outputs_dir"], "metrics", "wildfire_training_history.json")
    save_metrics({
        'tabular_times': tab_times,
        'cnn_history': cnn_hist,
        'fusion_history': fusion_hist
    }, history_file)

    print("\n[Wildfire Training] All wildfire models trained and checkpoints saved successfully!")


if __name__ == "__main__":
    main()
