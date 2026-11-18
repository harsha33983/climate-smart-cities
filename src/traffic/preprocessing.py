"""
Data Preprocessing Pipeline for METR-LA Traffic Forecasting.
Strict chronological train/val/test split to prevent temporal data leakage.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, set_seed
from src.traffic.download_data import load_metr_la_hdf5


class TrafficPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = config["traffic"]["raw_dir"]
        self.processed_dir = config["traffic"]["processed_dir"]
        self.seq_len = config["traffic"]["preprocessing"]["seq_len"]  # 12 steps (60 min)
        self.horizon_30min = config["traffic"]["preprocessing"]["horizon_30min"]  # 6 steps
        self.horizon_60min = config["traffic"]["preprocessing"]["horizon_60min"]  # 12 steps
        self.train_ratio = config["traffic"]["preprocessing"]["train_ratio"]
        self.val_ratio = config["traffic"]["preprocessing"]["val_ratio"]
        self.test_ratio = config["traffic"]["preprocessing"]["test_ratio"]

        self.mean = None
        self.std = None
        self.sensor_ids = None

    def load_raw_data(self) -> pd.DataFrame:
        """Load genuine METR-LA speed readings."""
        h5_path = os.path.join(self.raw_dir, "metr-la.h5")
        df = load_metr_la_hdf5(h5_path)
        return df

    def clean_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values, physical outliers, and zero readings.
        Sensors reporting 0 mph in METR-LA are missing sensor readouts.
        """
        print(f"[Traffic Preprocessor] Initial readings shape: {df.shape}")
        
        # Replace non-physical negative speeds and zeros (sensor dropouts) with NaN
        cleaned_values = df.values.copy()
        cleaned_values[cleaned_values <= 0.0] = np.nan
        cleaned_values[cleaned_values > 120.0] = np.nan  # Physical speed limit filter
        
        df_cleaned = pd.DataFrame(cleaned_values, index=df.index, columns=df.columns)
        
        # Forward-fill followed by backward-fill across time steps for temporal continuity
        df_cleaned = df_cleaned.ffill().bfill()
        
        # Any remaining NaNs replaced with sensor historical median
        df_cleaned = df_cleaned.fillna(df_cleaned.median())
        
        print(f"[Traffic Preprocessor] Cleaned data range: [{df_cleaned.values.min():.2f}, {df_cleaned.values.max():.2f}] mph")
        return df_cleaned

    def chronological_split(self, df: pd.DataFrame):
        """Split data chronologically into train, validation, and test partitions."""
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        print(f"[Traffic Preprocessor] Chronological Split (No Leakage):")
        print(f"  Train: {len(train_df)} timesteps ({train_df.index[0]} to {train_df.index[-1]})")
        print(f"  Val:   {len(val_df)} timesteps ({val_df.index[0]} to {val_df.index[-1]})")
        print(f"  Test:  {len(test_df)} timesteps ({test_df.index[0]} to {test_df.index[-1]})")

        return train_df, val_df, test_df

    def fit_normalizer(self, train_df: pd.DataFrame):
        """Fit Z-score standardizer on training split ONLY to prevent data leakage."""
        self.mean = train_df.values.mean()
        self.std = train_df.values.std()
        self.sensor_ids = list(train_df.columns)
        print(f"[Traffic Preprocessor] Fit Normalizer on TRAIN ONLY: Mean = {self.mean:.2f} mph, Std = {self.std:.2f} mph")

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Standardize using training statistics."""
        return (df.values - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """Denormalize back to actual mph."""
        return arr * (self.std + 1e-6) + self.mean

    def create_sliding_windows(self, data: np.ndarray, stride: int = 1):
        """
        Create (X, Y_30, Y_60, Y_all) sequences.
        X: (N, seq_len=12, num_sensors) -> past 60 min
        Y_30: (N, num_sensors) -> speed at t + 6 (30 min ahead)
        Y_60: (N, num_sensors) -> speed at t + 12 (60 min ahead)
        Y_all: (N, horizon=12, num_sensors) -> all future 12 steps
        """
        X, Y_30, Y_60, Y_all = [], [], [], []
        total_steps = len(data)
        max_idx = total_steps - self.seq_len - self.horizon_60min + 1

        for i in range(0, max_idx, stride):
            x_seq = data[i : i + self.seq_len]
            # Future steps: 1 to 12
            y_seq = data[i + self.seq_len : i + self.seq_len + self.horizon_60min]
            y_30 = y_seq[self.horizon_30min - 1]  # 6th future step (index 5)
            y_60 = y_seq[self.horizon_60min - 1]  # 12th future step (index 11)

            X.append(x_seq)
            Y_30.append(y_30)
            Y_60.append(y_60)
            Y_all.append(y_seq)

        return np.array(X, dtype=np.float32), np.array(Y_30, dtype=np.float32), np.array(Y_60, dtype=np.float32), np.array(Y_all, dtype=np.float32)

    def process_and_save(self, stride_train: int = 2, stride_test: int = 2):
        """Full preprocessing pipeline execution and serialization."""
        os.makedirs(self.processed_dir, exist_ok=True)
        raw_df = self.load_raw_data()
        clean_df = self.clean_and_impute(raw_df)

        train_df, val_df, test_df = self.chronological_split(clean_df)
        self.fit_normalizer(train_df)

        train_norm = self.transform(train_df)
        val_norm = self.transform(val_df)
        test_norm = self.transform(test_df)

        print("[Traffic Preprocessor] Generating sliding windows...")
        X_train, Y30_train, Y60_train, Yall_train = self.create_sliding_windows(train_norm, stride=stride_train)
        X_val, Y30_val, Y60_val, Yall_val = self.create_sliding_windows(val_norm, stride=stride_train)
        X_test, Y30_test, Y60_test, Yall_test = self.create_sliding_windows(test_norm, stride=stride_test)

        print(f"  Train: X shape {X_train.shape}, Y30 shape {Y30_train.shape}, Y60 shape {Y60_train.shape}")
        print(f"  Val:   X shape {X_val.shape}, Y30 shape {Y30_val.shape}, Y60 shape {Y60_val.shape}")
        print(f"  Test:  X shape {X_test.shape}, Y30 shape {Y30_test.shape}, Y60 shape {Y60_test.shape}")

        # Save preprocessed arrays
        processed_data = {
            'X_train': X_train, 'Y30_train': Y30_train, 'Y60_train': Y60_train, 'Yall_train': Yall_train,
            'X_val': X_val, 'Y30_val': Y30_val, 'Y60_val': Y60_val, 'Yall_val': Yall_val,
            'X_test': X_test, 'Y30_test': Y30_test, 'Y60_test': Y60_test, 'Yall_test': Yall_test,
            'mean': self.mean,
            'std': self.std,
            'sensor_ids': self.sensor_ids,
            'test_timestamps': test_df.index[self.seq_len + self.horizon_60min - 1 :: stride_test][:len(X_test)]
        }

        out_path = os.path.join(self.processed_dir, "traffic_processed.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(processed_data, f, protocol=4)

        print(f"[Traffic Preprocessor] Saved processed traffic data to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
        return out_path


def main():
    config = load_config()
    preprocessor = TrafficPreprocessor(config)
    preprocessor.process_and_save(stride_train=2, stride_test=2)


if __name__ == "__main__":
    main()
