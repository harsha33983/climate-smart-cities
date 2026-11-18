"""
Preprocessing Pipeline for Wildfire Risk & Occurrence Mapping.
Implements chronological and spatial splitting to strictly prevent data leakage.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.utilities import load_config, set_seed


class WildfirePreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.proc_dir = config["wildfire"]["processed_dir"]
        self.train_ratio = config["wildfire"]["preprocessing"]["train_ratio"]
        self.val_ratio = config["wildfire"]["preprocessing"]["val_ratio"]
        self.test_ratio = config["wildfire"]["preprocessing"]["test_ratio"]

        self.feature_cols = [
            'elevation_m', 'slope_deg', 'temp_celsius', 'relative_humidity_pct',
            'wind_speed_ms', 'wind_direction_deg', 'precipitation_mm',
            'vegetation_dryness', 'ndvi'
        ]
        self.scaler = None

    def load_data(self) -> pd.DataFrame:
        """Load assembled wildfire dataset."""
        csv_path = os.path.join(self.proc_dir, "wildfire_dataset.csv")
        df = pd.read_csv(csv_path)
        print(f"[Wildfire Preprocessor] Loaded raw dataset: {len(df)} samples")
        return df

    def split_data(self, df: pd.DataFrame):
        """
        Split dataset chronologically and spatially to prevent data leakage.
        Sort by acquisition date so future observations are strictly kept in the test set.
        """
        # Sort by acq_date and longitude for spatio-temporal ordering
        df_sorted = df.sort_values(by=['acq_date', 'longitude']).reset_index(drop=True)

        n = len(df_sorted)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        print(f"[Wildfire Preprocessor] Data Split (No Leakage):")
        print(f"  Train: {len(train_df)} samples (Fires: {sum(train_df['fire_occurrence'] == 1)})")
        print(f"  Val:   {len(val_df)} samples (Fires: {sum(val_df['fire_occurrence'] == 1)})")
        print(f"  Test:  {len(test_df)} samples (Fires: {sum(test_df['fire_occurrence'] == 1)})")

        return train_df, val_df, test_df

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Fit StandardScaler on TRAIN split only and transform val/test."""
        self.scaler = StandardScaler()
        X_train_num = self.scaler.fit_transform(train_df[self.feature_cols].values)
        X_val_num = self.scaler.transform(val_df[self.feature_cols].values)
        X_test_num = self.scaler.transform(test_df[self.feature_cols].values)

        print("[Wildfire Preprocessor] Standardized features using train statistics only.")
        return X_train_num, X_val_num, X_test_num

    def process_and_save(self):
        """Execute full preprocessing pipeline and save artifacts."""
        df = self.load_data()
        train_df, val_df, test_df = self.split_data(df)
        X_train_num, X_val_num, X_test_num = self.fit_transform(train_df, val_df, test_df)

        processed = {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'X_train_num': X_train_num.astype(np.float32),
            'X_val_num': X_val_num.astype(np.float32),
            'X_test_num': X_test_num.astype(np.float32),
            'y_train': train_df['fire_occurrence'].values.astype(np.int64),
            'y_val': val_df['fire_occurrence'].values.astype(np.int64),
            'y_test': test_df['fire_occurrence'].values.astype(np.int64),
            'feature_cols': self.feature_cols,
            'scaler': self.scaler
        }

        out_path = os.path.join(self.proc_dir, "wildfire_preprocessed.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(processed, f, protocol=4)

        print(f"[Wildfire Preprocessor] Saved preprocessed data to {out_path}")
        return out_path


def main():
    config = load_config()
    preprocessor = WildfirePreprocessor(config)
    preprocessor.process_and_save()


if __name__ == "__main__":
    main()
