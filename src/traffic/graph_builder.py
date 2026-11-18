"""
Graph Construction and Adjacency Matrix computation for METR-LA Sensor Network.
Constructs Gaussian distance graph Laplacian for Spatial Graph Neural Networks.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config


def build_sensor_graph(raw_dir: str, processed_dir: str, threshold: float = 0.1) -> tuple:
    """
    Construct road sensor graph adjacency matrix from real sensor distances.
    Outputs:
      adj_matrix: (N, N) normalized Gaussian adjacency matrix
      norm_laplacian: (N, N) symmetric normalized graph Laplacian
      sensor_ids: list of sensor ID strings
      sensor_locs: DataFrame of sensor lat/lon coordinates
    """
    os.makedirs(processed_dir, exist_ok=True)
    sensor_ids_path = os.path.join(raw_dir, "graph_sensor_ids.txt")
    distances_path = os.path.join(raw_dir, "distances_la_2012.csv")
    locations_path = os.path.join(raw_dir, "graph_sensor_locations.csv")
    adj_pickle_path = os.path.join(raw_dir, "adj_mx.pkl")

    # Read sensor IDs
    with open(sensor_ids_path, "r") as f:
        sensor_ids = [line.strip() for line in f.read().split(",") if line.strip()]

    num_sensors = len(sensor_ids)
    sensor_id_to_idx = {sid: idx for idx, sid in enumerate(sensor_ids)}
    print(f"[Graph Builder] Number of road sensors: {num_sensors}")

    # Check if precomputed official adj_mx.pkl exists
    if os.path.exists(adj_pickle_path):
        try:
            with open(adj_pickle_path, "rb") as f:
                pickle_data = pickle.load(f, encoding='latin1')
            if isinstance(pickle_data, tuple) and len(pickle_data) >= 3:
                _, _, adj_mx = pickle_data
            elif isinstance(pickle_data, np.ndarray):
                adj_mx = pickle_data
            else:
                adj_mx = None
            if adj_mx is not None and adj_mx.shape == (num_sensors, num_sensors):
                print(f"[Graph Builder] Loaded genuine official METR-LA adjacency matrix: {adj_mx.shape}")
            else:
                adj_mx = None
        except Exception as e:
            print(f"[Graph Builder] Recomputing from distances CSV: {e}")
            adj_mx = None
    else:
        adj_mx = None

    if adj_mx is None:
        # Compute from distances_la_2012.csv
        dist_df = pd.read_csv(distances_path)
        dist_mx = np.full((num_sensors, num_sensors), np.inf, dtype=np.float32)
        np.fill_diagonal(dist_mx, 0.0)

        for _, row in dist_df.iterrows():
            from_id = str(int(row['from']))
            to_id = str(int(row['to']))
            cost = float(row['cost'])
            if from_id in sensor_id_to_idx and to_id in sensor_id_to_idx:
                i = sensor_id_to_idx[from_id]
                j = sensor_id_to_idx[to_id]
                dist_mx[i, j] = cost

        # Gaussian kernel: A_ij = exp(- (dist / std)^2)
        finite_distances = dist_mx[np.isfinite(dist_mx) & (dist_mx > 0)]
        sigma = np.std(finite_distances)
        print(f"[Graph Builder] Computed finite distance std: {sigma:.2f}")

        adj_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
        for i in range(num_sensors):
            for j in range(num_sensors):
                if i == j:
                    adj_mx[i, j] = 1.0
                elif np.isfinite(dist_mx[i, j]):
                    w = np.exp(-((dist_mx[i, j] / sigma) ** 2))
                    if w >= threshold:
                        adj_mx[i, j] = w

    # Ensure self-loops
    np.fill_diagonal(adj_mx, 1.0)

    # Compute Symmetric Normalized Graph Laplacian: L_norm = D^(-1/2) * A * D^(-1/2)
    degrees = np.sum(adj_mx, axis=1)
    d_inv_sqrt = np.power(degrees, -0.5, where=degrees > 0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_mat = np.diag(d_inv_sqrt)
    norm_adj = D_mat @ adj_mx @ D_mat
    norm_adj = norm_adj.astype(np.float32)

    # Sensor Locations
    locs_df = pd.read_csv(locations_path)
    # Ensure matched ordering
    locs_df['sensor_id_str'] = locs_df['sensor_id'].astype(str)
    locs_df = locs_df.set_index('sensor_id_str').reindex(sensor_ids).reset_index()

    graph_data = {
        'adj_matrix': adj_mx.astype(np.float32),
        'norm_adj': norm_adj,
        'sensor_ids': sensor_ids,
        'sensor_locations': locs_df
    }

    out_file = os.path.join(processed_dir, "traffic_graph.pkl")
    with open(out_file, "wb") as f:
        pickle.dump(graph_data, f)

    print(f"[Graph Builder] Saved road network graph to {out_file}")
    print(f"  Adjacency shape: {adj_mx.shape}, Non-zero edges: {np.count_nonzero(adj_mx):,}")
    print(f"  Graph density: {np.count_nonzero(adj_mx) / (num_sensors * num_sensors) * 100:.2f}%")

    return adj_mx, norm_adj, sensor_ids, locs_df


def main():
    config = load_config()
    raw_dir = config["traffic"]["raw_dir"]
    proc_dir = config["traffic"]["processed_dir"]
    build_sensor_graph(raw_dir, proc_dir)


if __name__ == "__main__":
    main()
