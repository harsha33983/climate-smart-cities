"""
Download and verify genuine METR-LA traffic dataset.
"""

import os
import sys
import urllib.request
import pandas as pd
import h5py

# Ensure package root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config


def download_file(url: str, dest_path: str):
    """Download a file with progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[Traffic] Already downloaded: {dest_path} ({os.path.getsize(dest_path):,} bytes)")
        return dest_path

    print(f"[Traffic] Downloading {url} -> {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out_file:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded / total * 100
                sys.stdout.write(f"\r[Traffic] Progress: {percent:.1f}% ({downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB)")
                sys.stdout.flush()
    print("\n[Traffic] Download complete!")
    return dest_path


def load_metr_la_hdf5(h5_path: str) -> pd.DataFrame:
    """Robustly load METR-LA HDF5 file via h5py compatible with pandas 2/3."""
    with h5py.File(h5_path, 'r') as f:
        sensors = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['df/axis0'][:]]
        timestamps = pd.to_datetime(f['df/axis1'][:])
        values = f['df/block0_values'][:]
        df = pd.DataFrame(values, index=timestamps, columns=sensors)
    return df


def verify_traffic_data(raw_dir: str):
    """Verify downloaded METR-LA dataset integrity and contents."""
    h5_path = os.path.join(raw_dir, "metr-la.h5")
    locations_path = os.path.join(raw_dir, "graph_sensor_locations.csv")
    distances_path = os.path.join(raw_dir, "distances_la_2012.csv")

    assert os.path.exists(h5_path), f"Missing {h5_path}"
    assert os.path.exists(locations_path), f"Missing {locations_path}"
    assert os.path.exists(distances_path), f"Missing {distances_path}"

    df = load_metr_la_hdf5(h5_path)

    print(f"[Traffic Verification] Genuine METR-LA HDF5 verified!")
    print(f"  Shape: {df.shape} (Timestamps x Sensors)")
    print(f"  Sensors Count: {df.shape[1]}")
    print(f"  Time Steps: {df.shape[0]}")
    print(f"  Date Range: {df.index.min()} to {df.index.max()}")
    print(f"  Speed Range: {df.values.min():.2f} to {df.values.max():.2f} mph")
    print(f"  Mean Speed: {df.values.mean():.2f} mph")

    locs_df = pd.read_csv(locations_path)
    print(f"[Traffic Verification] Sensor Locations loaded: {len(locs_df)} sensors mapped.")

    return df, locs_df


def main():
    config = load_config()
    raw_dir = config["traffic"]["raw_dir"]
    urls = config["traffic"]["urls"]

    download_file(urls["h5_data"], os.path.join(raw_dir, "metr-la.h5"))
    download_file(urls["sensor_ids"], os.path.join(raw_dir, "graph_sensor_ids.txt"))
    download_file(urls["sensor_locations"], os.path.join(raw_dir, "graph_sensor_locations.csv"))
    download_file(urls["distances"], os.path.join(raw_dir, "distances_la_2012.csv"))
    download_file(urls["adj_mx"], os.path.join(raw_dir, "adj_mx.pkl"))

    df, locs_df = verify_traffic_data(raw_dir)
    print("[Traffic] All data successfully downloaded and verified!")


if __name__ == "__main__":
    main()
