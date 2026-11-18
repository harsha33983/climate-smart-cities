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
    """Load YAML configuration file with robust absolute root resolution."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if not os.path.isabs(config_path):
        candidate = os.path.join(project_root, config_path)
        if os.path.exists(candidate):
            config_path = candidate
        elif not os.path.exists(config_path):
            alt_path = os.path.join(os.getcwd(), "climate-smart-cities", config_path)
            if os.path.exists(alt_path):
                config_path = alt_path

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert all module relative directories to absolute paths anchored to project_root
    for module in ["traffic", "wildfire"]:
        if module in config:
            for key in ["raw_dir", "processed_dir", "models_dir", "outputs_dir"]:
                if key in config[module]:
                    val = config[module][key]
                    if not os.path.isabs(val):
                        config[module][key] = os.path.normpath(os.path.join(project_root, val))

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


def resolve_path(rel_path: str) -> str:
    """Resolve a relative path robustly regardless of current working directory."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(project_root, rel_path),
        os.path.join(os.getcwd(), "climate-smart-cities", rel_path),
        os.path.join(os.getcwd(), rel_path),
        rel_path
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(project_root, rel_path)


def load_metrics(file_path: str) -> dict:
    """Load metrics dictionary from JSON file with multi-directory resolution."""
    resolved = resolve_path(file_path)
    if not os.path.exists(resolved):
        return {}
    with open(resolved, "r") as f:
        return json.load(f)
