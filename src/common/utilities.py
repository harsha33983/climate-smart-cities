"""
Common Utilities for Climate & Smart Cities Intelligence System.
"""

import os
import random
import json
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device(requested_device: str = "cpu") -> torch.device:
    """Get PyTorch device safely."""
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_metrics(metrics: dict, file_path: str):
    """Save metrics dictionary to a formatted JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Utilities] Saved metrics to {file_path}")


def load_metrics(file_path: str) -> dict:
    """Load metrics dictionary from JSON file."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)
