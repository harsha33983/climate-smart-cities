"""
Streamlit Wildfire Spread & Risk Mapping Page.
Geospatial risk mapping, satellite active fire detections, and interactive what-if simulator.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static

# Ensure package root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common.utilities import load_metrics
from src.wildfire.predict import WildfirePredictor


@st.cache_data
def load_wildfire_predictions():
    """Load pre-evaluated wildfire test set predictions bundle."""
    pred_path = os.path.join("outputs", "predictions", "wildfire_test_predictions.pkl")
    if not os.path.exists(pred_path):
        pred_path = os.path.join("climate-smart-cities", pred_path)
    if not os.path.exists(pred_path):
        return None
    with open(pred_path, "rb") as f:
        data = pickle.load(f)
    return data


@st.cache_resource
def get_wildfire_predictor():
    """Cache wildfire multimodal predictor instance."""
    return WildfirePredictor(model_type="fusion")


def render_wildfire_page():
    st.title("🔥 Wildfire Spread & Risk Intelligence System")
    st.markdown(
        """
        Geospatial wildfire occurrence and vulnerability mapping powered by **NASA FIRMS** satellite detections,
        **NASA POWER** meteorology, **USGS Elevation**, and **Multimodal Deep Learning (CNN + Tabular Fusion)**.
        """
    )

    data = load_wildfire_predictions()
    if data is None:
        st.error("Wildfire prediction data not found. Please run the training and evaluation pipeline first.")
        return

    test_df = data['test_df'].copy()
    preds = data['predictions']
    metrics = data['metrics']

    # Default to Multimodal Fusion predictions
    test_df['risk_prob'] = preds['Multimodal Fusion']

    def map_category(p):
        if p < 0.25:
            return "Low Risk"
        elif p < 0.50:
            return "Moderate Risk"
        elif p < 0.75:
            return "High Risk"
        else:
            return "Critical Risk"

    test_df['risk_category'] = test_df['risk_prob'].apply(map_category)

    # --- TOP KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    total_obs = len(test_df)
    confirmed_fires = sum(test_df['fire_occurrence'] == 1)
    high_critical = sum(test_df['risk_prob'] >= 0.50)
    mean_temp = test_df['temp_celsius'].mean()

    col1.metric("Analyzed Geospatial Sites", f"{total_obs:,}")
    col2.metric("Confirmed Satellite Fires", f"{confirmed_fires:,}", "NASA FIRMS MODIS", delta_color="inverse")
    col3.metric("High/Critical Risk Zones", f"{high_critical:,}", f"{high_critical/total_obs*100:.1f}% of Sites", delta_color="inverse")
    col4.metric("Regional Mean Temp", f"{mean_temp:.1f} °C", "NASA POWER")

    st.markdown("---")

    # --- TABS ---
    tab_map, tab_simulator, tab_factors, tab_metrics = st.tabs([
        "🗺️ Interactive Wildfire Risk Map",
        "🎛️ What-If Wildfire Risk Simulator",
        "🌡️ Environmental Risk Drivers",
        "📊 Model Evaluation & Benchmarks"
    ])

    with tab_map:
        st.subheader("Geospatial Wildfire Risk & Satellite Hotspot Map")
        st.info("💡 **Important Distinction**: Satellite detections (red icons) indicate confirmed active thermal fire detections. Color-graded circles represent model-estimated ignition & vulnerability risk.")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk_filter = st.multiselect(
                "Filter by Risk Category",
                ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
                default=["Moderate Risk", "High Risk", "Critical Risk"]
            )
        with filter_col2:
            show_confirmed_only = st.checkbox("Highlight Confirmed Satellite Ignitions Only", value=False)

        filtered_df = test_df[test_df['risk_category'].isin(risk_filter)]
        if show_confirmed_only:
            filtered_df = filtered_df[filtered_df['fire_occurrence'] == 1]

        center_lat = float(filtered_df['latitude'].mean()) if len(filtered_df) > 0 else 37.0
        center_lon = float(filtered_df['longitude'].mean()) if len(filtered_df) > 0 else -110.0

        m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB dark_matter")

        color_map = {
            "Low Risk": "#2ecc71",
            "Moderate Risk": "#f1c40f",
            "High Risk": "#e67e22",
            "Critical Risk": "#e74c3c"
        }

        # Add points
        for _, row in filtered_df.iterrows():
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            prob = float(row['risk_prob'])
            cat = row['risk_category']
            is_active = (int(row['fire_occurrence']) == 1)
            c = color_map.get(cat, "#3498db")

            popup_text = f"""
            <b>Status:</b> {'🔥 Confirmed Satellite Active Fire' if is_active else '🌲 Normal Background / Non-Fire'}<br>
            <b>Model Estimated Risk:</b> {cat} ({prob*100:.1f}%)<br>
            <b>Temp:</b> {row['temp_celsius']:.1f} °C<br>
            <b>Humidity:</b> {row['relative_humidity_pct']:.1f}%<br>
            <b>Wind:</b> {row['wind_speed_ms']:.1f} m/s<br>
            <b>Elevation:</b> {row['elevation_m']:.0f} m (Slope: {row['slope_deg']:.1f}°)<br>
            <b>Dryness Index:</b> {row['vegetation_dryness']:.2f}
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=8 if is_active else 4,
                color="white" if is_active else c,
                weight=2 if is_active else 1,
                fill=True,
                fill_color=c,
                fill_opacity=0.9 if is_active else 0.6,
                popup=popup_text
            ).add_to(m)

        folium_static(m, width=950, height=520)

    with tab_simulator:
        st.subheader("🎛️ Interactive 'What-If' Wildfire Risk Simulator")
        st.markdown("Adjust meteorological, terrain, and vegetation indicators to simulate real-time risk predictions using the **Multimodal Deep Learning Model**.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_temp = st.slider("Ambient Temperature (°C)", min_value=10.0, max_value=48.0, value=34.0, step=0.5)
            sim_rh = st.slider("Relative Humidity (%)", min_value=5.0, max_value=95.0, value=18.0, step=1.0)
            sim_wind = st.slider("Wind Speed (m/s)", min_value=0.5, max_value=25.0, value=7.5, step=0.5)
        with sim_col2:
            sim_elev = st.slider("Terrain Elevation (m)", min_value=20.0, max_value=3500.0, value=850.0, step=50.0)
            sim_slope = st.slider("Terrain Slope (degrees)", min_value=0.5, max_value=45.0, value=18.0, step=0.5)
            sim_wind_dir = st.slider("Wind Direction (degrees)", min_value=0.0, max_value=360.0, value=220.0, step=10.0)
        with sim_col3:
            sim_dry = st.slider("Vegetation Dryness Indicator", min_value=0.05, max_value=1.0, value=0.82, step=0.02)
            sim_ndvi = st.slider("Normalized Difference Veg Index (NDVI)", min_value=0.05, max_value=0.9, value=0.22, step=0.02)
            sim_precip = st.slider("24h Rainfall (mm)", min_value=0.0, max_value=25.0, value=0.0, step=0.5)

        predictor = get_wildfire_predictor()
        input_sample = {
            'temp_celsius': sim_temp,
            'relative_humidity_pct': sim_rh,
            'wind_speed_ms': sim_wind,
            'wind_direction_deg': sim_wind_dir,
            'precipitation_mm': sim_precip,
            'elevation_m': sim_elev,
            'slope_deg': sim_slope,
            'vegetation_dryness': sim_dry,
            'ndvi': sim_ndvi
        }
        res = predictor.predict_point(input_sample)

        st.markdown("#### Simulation Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Risk Category", res['risk_category'])
        res_col2.metric("Wildfire Risk Probability", f"{res['risk_percentage']}%")
        res_col3.metric("Vapor Pressure Deficit", f"{res['derived_vpd_kpa']} kPa")

        st.progress(res['risk_probability'])
        st.caption(f"Status: **{res['active_fire_status']}** | Fire Weather Index Proxy: **{res['derived_fwi_proxy']}**")

    with tab_factors:
        st.subheader("Environmental & Topographic Risk Factor Analysis")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            fig_scatter = px.scatter(
                test_df, x="temp_celsius", y="relative_humidity_pct",
                color="risk_category",
                color_discrete_map=color_map,
                title="Temperature vs Relative Humidity by Risk Tier",
                labels={"temp_celsius": "Temperature (°C)", "relative_humidity_pct": "Relative Humidity (%)"},
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_f2:
            fig_box = px.box(
                test_df, x="risk_category", y="wind_speed_ms",
                color="risk_category",
                color_discrete_map=color_map,
                title="Wind Speed Distribution across Risk Categories",
                labels={"wind_speed_ms": "Wind Speed (m/s)", "risk_category": "Risk Category"},
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.image("outputs/plots/wildfire_feature_importance.png", caption="XGBoost Feature Importance: Top Environmental Drivers of Fire Vulnerability", use_column_width=True)

    with tab_metrics:
        st.subheader("Model Evaluation Benchmark on Unseen Test Split (600 Sites)")
        st.markdown("Evaluation prioritizing **Recall, PR-AUC, and False Negative Rate (FNR)**:")

        rows = []
        for m_name in ['Random Forest', 'XGBoost', 'Spatial CNN', 'Multimodal Fusion']:
            if m_name in metrics:
                m = metrics[m_name]
                rows.append({
                    'Model': m_name,
                    'Accuracy (%)': round(m['accuracy'] * 100, 2),
                    'Precision (%)': round(m['precision'] * 100, 2),
                    'Recall (%)': round(m['recall'] * 100, 2),
                    'F1-Score (%)': round(m['f1_score'] * 100, 2),
                    'ROC-AUC': m['roc_auc'],
                    'PR-AUC': m['pr_auc'],
                    'False Negative Rate (%)': round(m['false_negative_rate'] * 100, 2),
                    'Latency (ms/sample)': m.get('latency_ms_per_sample', 0.0)
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.image("outputs/plots/wildfire_roc_curves.png", caption="Receiver Operating Characteristic (ROC) Curves", use_column_width=True)
        with col_m2:
            st.image("outputs/plots/wildfire_pr_curves.png", caption="Precision-Recall (PR) Curves", use_column_width=True)

        st.image("outputs/plots/wildfire_confusion_matrices.png", caption="Confusion Matrices across All 4 Wildfire Models", use_column_width=True)
