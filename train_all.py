"""
Master Pipeline Orchestrator for Climate & Smart Cities Intelligence System.
Runs data acquisition, preprocessing, actual model training, and evaluation for both modules.
"""

import os
import sys
import time

# Ensure package root is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    print("=" * 80)
    print(" CLIMATE & SMART CITIES INTELLIGENCE SYSTEM - END-TO-END ORCHESTRATOR")
    print("=" * 80)
    start_total = time.time()

    # ----------------- MODULE 1: TRAFFIC -----------------
    print("\n>>> [MODULE 1: TRAFFIC FLOW & CONGESTION FORECASTING]")
    
    print("\n1.1 Checking & Downloading METR-LA Traffic Dataset...")
    from src.traffic import download_data as td
    td.main()

    print("\n1.2 Preprocessing Traffic Sequences & Chronological Splitting...")
    from src.traffic import preprocessing as tp
    tp.main()

    print("\n1.3 Constructing Highway Road Network Graph Adjacency Matrix...")
    from src.traffic import graph_builder as gb
    gb.main()

    print("\n1.4 Training Traffic Forecasting Models (Baseline, LSTM, ST-GNN)...")
    from src.traffic import train as tt
    tt.main()

    print("\n1.5 Evaluating Traffic Models on Unseen Test Split...")
    from src.traffic import evaluate as te
    te.main()

    # ----------------- MODULE 2: WILDFIRE -----------------
    print("\n>>> [MODULE 2: WILDFIRE SPREAD & RISK MAPPING]")

    print("\n2.1 Downloading & Assembling NASA FIRMS + Environmental Data...")
    from src.wildfire import download_data as wd
    wd.main()

    print("\n2.2 Preprocessing Geospatial Wildfire Records...")
    from src.wildfire import preprocessing as wp
    wp.main()

    print("\n2.3 Engineering Domain Indices & Multi-Channel Spatial Patches...")
    from src.wildfire import feature_engineering as wf
    wf.main()

    print("\n2.4 Training Wildfire Models (Random Forest, XGBoost, CNN, Multimodal Fusion)...")
    from src.wildfire import train as wt
    wt.main()

    print("\n2.5 Evaluating Wildfire Models on Unseen Test Split...")
    from src.wildfire import evaluate as we
    we.main()

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f" END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
