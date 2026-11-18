"""
Comprehensive Evaluation of Wildfire Risk Models on Unseen Test Split.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrices,
and generates publication-quality diagnostic plots.
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, get_device, save_metrics
from src.wildfire.cnn_model import WildfireCNN
from src.wildfire.fusion_model import WildfireMultimodalFusion


def compute_classification_metrics(model_name: str, y_true: np.ndarray, y_prob: np.ndarray,
                                  elapsed_time: float, threshold: float = 0.5) -> dict:
    """Calculate comprehensive evaluation metrics prioritizing recall and PR-AUC."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    print(f"\n[{model_name} Test Results]")
    print(f"  Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")
    print(f"  ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | False Negative Rate: {fnr*100:.2f}%")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  Inference Latency: {elapsed_time:.4f}s ({elapsed_time / len(y_true) * 1000:.3f} ms/sample)")

    return {
        'model': model_name,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'false_negative_rate': round(fnr, 4),
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'inference_time_sec': round(elapsed_time, 4),
        'latency_ms_per_sample': round(elapsed_time / len(y_true) * 1000, 3)
    }


def generate_wildfire_plots(results: dict, data: dict, models_dir: str, plots_dir: str):
    """Generate diagnostic plots for wildfire risk models."""
    os.makedirs(plots_dir, exist_ok=True)
    y_true = data['y_test']
    colors = {'Random Forest': '#3498db', 'XGBoost': '#2ecc71', 'Spatial CNN': '#f39c12', 'Multimodal Fusion': '#e74c3c'}

    # 1. ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res['y_prob'])
        auc_val = res['metrics']['roc_auc']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[name], linewidth=2.0)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance')
    plt.title('ROC Curves - Wildfire Occurrence & Risk Classification', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    plot_roc = os.path.join(plots_dir, "wildfire_roc_curves.png")
    plt.savefig(plot_roc)
    plt.close()
    print(f"[Wildfire Evaluation] Saved plot: {plot_roc}")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 6), dpi=150)
    for name, res in results.items():
        p_curve, r_curve, _ = precision_recall_curve(y_true, res['y_prob'])
        pr_val = res['metrics']['pr_auc']
        plt.plot(r_curve, p_curve, label=f"{name} (PR-AUC = {pr_val:.3f})", color=colors[name], linewidth=2.0)
    plt.title('Precision-Recall Curves (Wildfire Early Warning)', fontsize=13, fontweight='bold')
    plt.xlabel('Recall (Sensitivity)', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10)
    plt.tight_layout()
    plot_pr = os.path.join(plots_dir, "wildfire_pr_curves.png")
    plt.savefig(plot_pr)
    plt.close()
    print(f"[Wildfire Evaluation] Saved plot: {plot_pr}")

    # 3. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150)
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        cm_dict = res['metrics']['confusion_matrix']
        cm_arr = np.array([[cm_dict['tn'], cm_dict['fp']], [cm_dict['fn'], cm_dict['tp']]])
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Non-Fire', 'Fire'], yticklabels=['Non-Fire', 'Fire'])
        ax.set_title(f"{name}\nRecall: {res['metrics']['recall']*100:.1f}% | F1: {res['metrics']['f1_score']*100:.1f}%", fontsize=11)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plot_cm = os.path.join(plots_dir, "wildfire_confusion_matrices.png")
    plt.savefig(plot_cm)
    plt.close()
    print(f"[Wildfire Evaluation] Saved plot: {plot_cm}")

    # 4. Feature Importance (XGBoost)
    with open(os.path.join(models_dir, "xgboost.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    imp = xgb_model.feature_importances_
    cols = data['tabular_cols']
    sorted_idx = np.argsort(imp)[::-1]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.barh(range(len(cols)), imp[sorted_idx][::-1], color='#e67e22', edgecolor='black', alpha=0.85)
    plt.yticks(range(len(cols)), [cols[i] for i in sorted_idx[::-1]], fontsize=10)
    plt.xlabel('Relative Feature Importance (Gain)', fontsize=11)
    plt.title('XGBoost Environmental & Topographic Feature Importance', fontsize=13, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_imp = os.path.join(plots_dir, "wildfire_feature_importance.png")
    plt.savefig(plot_imp)
    plt.close()
    print(f"[Wildfire Evaluation] Saved plot: {plot_imp}")

    # 5. Geographic Risk Visualization
    test_df = data['test_df'].copy()
    test_df['predicted_risk'] = results['Multimodal Fusion']['y_prob']
    plt.figure(figsize=(11, 6), dpi=150)
    scatter = plt.scatter(
        test_df['longitude'], test_df['latitude'],
        c=test_df['predicted_risk'], cmap='YlOrRd',
        s=35, alpha=0.85, edgecolor='black', linewidth=0.3
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Predicted Wildfire Risk Probability', fontsize=11)
    plt.title('Geospatial Distribution of Model-Estimated Wildfire Risk (Test Set)', fontsize=13, fontweight='bold')
    plt.xlabel('Longitude', fontsize=11)
    plt.ylabel('Latitude', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plot_geo = os.path.join(plots_dir, "wildfire_geographic_risk_map.png")
    plt.savefig(plot_geo)
    plt.close()
    print(f"[Wildfire Evaluation] Saved plot: {plot_geo}")


def main():
    config = load_config()
    device = get_device(config["project"]["device"])
    proc_dir = config["wildfire"]["processed_dir"]
    models_dir = config["wildfire"]["models_dir"]
    outputs_dir = config["wildfire"]["outputs_dir"]

    with open(os.path.join(proc_dir, "wildfire_engineered.pkl"), "rb") as f:
        data = pickle.load(f)

    X_test_tab = data['X_test_tab']
    X_test_img = torch.tensor(data['X_test_img'], dtype=torch.float32).to(device)
    y_test = data['y_test']

    results = {}

    # 1. Random Forest Evaluation
    with open(os.path.join(models_dir, "random_forest.pkl"), "rb") as f:
        rf = pickle.load(f)
    t0 = time.time()
    rf_prob = rf.predict_proba(X_test_tab)[:, 1]
    rf_time = time.time() - t0
    results['Random Forest'] = {
        'y_prob': rf_prob,
        'metrics': compute_classification_metrics("Random Forest", y_test, rf_prob, rf_time)
    }

    # 2. XGBoost Evaluation
    with open(os.path.join(models_dir, "xgboost.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    t0 = time.time()
    xgb_prob = xgb_model.predict_proba(X_test_tab)[:, 1]
    xgb_time = time.time() - t0
    results['XGBoost'] = {
        'y_prob': xgb_prob,
        'metrics': compute_classification_metrics("XGBoost", y_test, xgb_prob, xgb_time)
    }

    # 3. Spatial CNN Evaluation
    cnn_model = WildfireCNN(in_channels=5, num_classes=2).to(device)
    cnn_ckpt = torch.load(os.path.join(models_dir, "cnn_best.pth"), map_location=device)
    cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
    cnn_model.eval()

    t0 = time.time()
    with torch.no_grad():
        cnn_logits = cnn_model(X_test_img)
        cnn_prob = F.softmax(cnn_logits, dim=1)[:, 1].cpu().numpy()
    cnn_time = time.time() - t0
    results['Spatial CNN'] = {
        'y_prob': cnn_prob,
        'metrics': compute_classification_metrics("Spatial CNN", y_test, cnn_prob, cnn_time)
    }

    # 4. Multimodal Fusion Evaluation
    tab_dim = X_test_tab.shape[1]
    fusion_model = WildfireMultimodalFusion(tabular_dim=tab_dim, in_channels=5, num_classes=2).to(device)
    fusion_ckpt = torch.load(os.path.join(models_dir, "fusion_best.pth"), map_location=device)
    fusion_model.load_state_dict(fusion_ckpt['model_state_dict'])
    fusion_model.eval()

    X_test_tab_tensor = torch.tensor(X_test_tab, dtype=torch.float32).to(device)
    t0 = time.time()
    with torch.no_grad():
        fusion_logits = fusion_model(X_test_img, X_test_tab_tensor)
        fusion_prob = F.softmax(fusion_logits, dim=1)[:, 1].cpu().numpy()
    fusion_time = time.time() - t0
    results['Multimodal Fusion'] = {
        'y_prob': fusion_prob,
        'metrics': compute_classification_metrics("Multimodal Fusion", y_test, fusion_prob, fusion_time)
    }

    # Save metrics JSON
    metrics_summary = {name: res['metrics'] for name, res in results.items()}
    metrics_path = os.path.join(outputs_dir, "metrics", "wildfire_metrics.json")
    save_metrics(metrics_summary, metrics_path)

    # Save test predictions bundle for dashboard
    pred_bundle = {
        'test_df': data['test_df'],
        'y_test': y_test,
        'predictions': {name: res['y_prob'] for name, res in results.items()},
        'metrics': metrics_summary
    }
    pred_path = os.path.join(outputs_dir, "predictions", "wildfire_test_predictions.pkl")
    with open(pred_path, "wb") as f:
        pickle.dump(pred_bundle, f, protocol=4)
    print(f"[Wildfire Evaluation] Saved test predictions to {pred_path}")

    # Generate plots
    plots_dir = os.path.join(outputs_dir, "plots")
    generate_wildfire_plots(results, data, models_dir, plots_dir)


if __name__ == "__main__":
    main()
