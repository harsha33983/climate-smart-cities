"""
Streamlit Traffic Forecasting Page.
Interactive visualization of METR-LA sensor network, multi-horizon forecasts, and live inference.
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
from streamlit_folium import folium_static

# Ensure package root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.common.utilities import load_metrics
from src.traffic.predict import TrafficPredictor


@st.cache_data
def load_traffic_predictions():
    """Load pre-evaluated test set predictions bundle."""
    pred_path = os.path.join("outputs", "predictions", "traffic_test_predictions.pkl")
    if not os.path.exists(pred_path):
        pred_path = os.path.join("climate-smart-cities", pred_path)
    if not os.path.exists(pred_path):
        return None
    with open(pred_path, "rb") as f:
        data = pickle.load(f)
    return data


@st.cache_resource
def get_traffic_predictor(model_type="gnn"):
    """Cache TrafficPredictor model instance."""
    return TrafficPredictor(model_type=model_type)


def render_traffic_page():
    st.title("🚗 Urban Traffic Flow & Congestion Forecasting")
    st.markdown(
        """
        Real-time spatio-temporal traffic speed forecasting across the **METR-LA Highway Network** (207 sensors).
        Powered by **Spatio-Temporal Graph Neural Networks (ST-GNN)** and **Multi-Horizon LSTMs**.
        """
    )

    data = load_traffic_predictions()
    if data is None:
        st.error("Traffic prediction data not found. Please run the training and evaluation pipeline first.")
        return

    sensor_ids = data['sensor_ids']
    locs_df = data['sensor_locations']
    timestamps = pd.to_datetime(data['timestamps'])

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Traffic Control Panel")
    selected_sensor = st.sidebar.selectbox("Select Traffic Sensor", sensor_ids, index=0)
    selected_horizon = st.sidebar.radio("Prediction Horizon", ["30 Minutes Ahead", "60 Minutes Ahead"])
    selected_model = st.sidebar.selectbox("Forecasting Model", ["ST-GNN (Graph Neural Network)", "LSTM", "Baseline (Persistence)"])

    model_key = "ST-GNN" if "ST-GNN" in selected_model else ("LSTM" if "LSTM" in selected_model else "Baseline")
    horizon_key = "y30" if "30" in selected_horizon else "y60"

    sensor_idx = sensor_ids.index(selected_sensor)
    sensor_loc = locs_df[locs_df['sensor_id'].astype(str) == str(selected_sensor)].iloc[0]

    y_true_all = data['y30_true'] if horizon_key == "y30" else data['y60_true']
    y_pred_all = data[model_key][horizon_key]

    curr_actual = float(y_true_all[-1, sensor_idx])
    curr_predicted = float(y_pred_all[-1, sensor_idx])
    speed_diff = curr_predicted - curr_actual

    # Congestion Status
    if curr_predicted < 25.0:
        status_text = "Severe Congestion"
    elif curr_predicted < 45.0:
        status_text = "Moderate Congestion"
    else:
        status_text = "Free Flowing"

    # --- TOP KPI METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Selected Sensor", f"#{selected_sensor}", f"Corridor Lat: {sensor_loc['latitude']:.3f}")
    col2.metric("Latest Ground Speed", f"{curr_actual:.1f} mph")
    col3.metric(f"Predicted Speed ({selected_horizon[:6]})", f"{curr_predicted:.1f} mph", f"{speed_diff:+.1f} mph", delta_color="inverse")
    col4.metric("Corridor Flow State", status_text)

    st.markdown("---")

    # --- TABS FOR VISUALIZATIONS ---
    tab_overview, tab_map, tab_comparison, tab_live, tab_alerts, tab_metrics = st.tabs([
        "📈 Actual vs Predicted Time Series",
        "🗺️ Sensor Network Map",
        "🔥 Traffic Speed Heatmap",
        "⚡ Live On-Demand Inference",
        "⚠️ Live Congestion Alerts",
        "📊 Model Evaluation Metrics"
    ])

    with tab_overview:
        st.subheader(f"Speed Forecast for Sensor #{selected_sensor} ({selected_horizon})")

        plot_steps = min(288, len(timestamps))
        sub_time = timestamps[:plot_steps]
        sub_true = y_true_all[:plot_steps, sensor_idx]
        sub_gnn = data['ST-GNN'][horizon_key][:plot_steps, sensor_idx]
        sub_lstm = data['LSTM'][horizon_key][:plot_steps, sensor_idx]
        sub_base = data['Baseline'][horizon_key][:plot_steps, sensor_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub_time, y=sub_true, mode='lines', name='Actual Ground Truth', line=dict(color='#1f77b4', width=2.5)))
        fig.add_trace(go.Scatter(x=sub_time, y=sub_gnn, mode='lines', name='ST-GNN (Graph Conv + Temporal GRU)', line=dict(color='#2ca02c', width=2.0)))
        fig.add_trace(go.Scatter(x=sub_time, y=sub_lstm, mode='lines', name='LSTM (Sequential)', line=dict(color='#ff7f0e', width=1.8, dash='dash')))
        fig.add_trace(go.Scatter(x=sub_time, y=sub_base, mode='lines', name='Baseline (Persistence)', line=dict(color='#7f7f7f', width=1.5, dash='dot')))

        fig.add_hline(y=25.0, line_dash="dash", line_color="red", annotation_text="Severe Congestion Threshold (<25 mph)")
        fig.add_hline(y=45.0, line_dash="dash", line_color="orange", annotation_text="Moderate Congestion Threshold (<45 mph)")

        fig.update_layout(
            title=f"24-Hour Corridor Forecasting Comparison (Sensor #{selected_sensor})",
            xaxis_title="Time of Day",
            yaxis_title="Traffic Speed (mph)",
            hovermode="x unified",
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_map:
        st.subheader("Interactive Los Angeles Highway Road Sensor Network")
        st.markdown("Sensors color-coded by current predicted speed: 🟢 Free Flow (>45 mph) | 🟡 Moderate (25-45 mph) | 🔴 Severe (<25 mph)")

        center_lat = float(locs_df['latitude'].mean())
        center_lon = float(locs_df['longitude'].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        latest_preds = y_pred_all[-1, :]
        for i, row in locs_df.iterrows():
            sid = str(row['sensor_id'])
            s_lat = float(row['latitude'])
            s_lon = float(row['longitude'])
            spd = float(latest_preds[i])

            if spd < 25.0:
                c = "red"
            elif spd < 45.0:
                c = "orange"
            else:
                c = "green"

            is_sel = (sid == str(selected_sensor))
            radius = 9 if is_sel else 5
            popup_html = f"<b>Sensor:</b> #{sid}<br><b>Predicted Speed:</b> {spd:.1f} mph<br><b>Status:</b> {'Severe' if spd<25 else ('Moderate' if spd<45 else 'Free Flow')}"

            folium.CircleMarker(
                location=[s_lat, s_lon],
                radius=radius,
                color="black" if is_sel else c,
                weight=2 if is_sel else 1,
                fill=True,
                fill_color=c,
                fill_opacity=0.85,
                popup=popup_html
            ).add_to(m)

        folium_static(m, width=950, height=520)

    with tab_comparison:
        st.subheader("Corridor Speed Heatmap across Highway Sensors")
        num_sensors_show = min(30, len(sensor_ids))
        heatmap_data = y_true_all[:144, :num_sensors_show].T

        fig_heat = px.imshow(
            heatmap_data,
            labels=dict(x="Time Step (5-min intervals)", y="Sensor Index", color="Speed (mph)"),
            x=[str(t.strftime("%H:%M")) for t in timestamps[:144]],
            y=[f"#{s}" for s in sensor_ids[:num_sensors_show]],
            color_continuous_scale="RdYlGn",
            aspect="auto",
            height=500
        )
        fig_heat.update_layout(title="Traffic Speed Dynamics Across Time of Day (Green = Flowing, Red = Congested)")
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab_live:
        st.subheader("⚡ Live On-Demand Spatio-Temporal Inference")
        st.markdown("Inject real-time speed perturbations or custom corridor conditions and evaluate live forward-pass predictions from the saved PyTorch checkpoints.")

        live_col1, live_col2 = st.columns([1, 2])
        with live_col1:
            inject_speed = st.slider(f"Simulate Current Speed at Sensor #{selected_sensor} (mph)", 5.0, 75.0, 20.0, step=1.0)
            model_choice = st.radio("Select Model for Live Inference", ["ST-GNN", "LSTM"])
            btn_run = st.button("🚀 Run Live Model Forward Pass")

        with live_col2:
            if btn_run:
                model_name = "gnn" if model_choice == "ST-GNN" else "lstm"
                predictor = get_traffic_predictor(model_name)
                # Load baseline sequence from test split
                with open("data/processed/traffic/traffic_processed.pkl", "rb") as f:
                    t_data = pickle.load(f)
                input_seq = t_data['X_test'][0] * t_data['std'] + t_data['mean']
                # Inject user perturbation
                input_seq[-1, sensor_idx] = inject_speed

                live_res = predictor.predict(input_seq)
                p30_sensor = live_res['predicted_speed_30min'][sensor_idx]
                p60_sensor = live_res['predicted_speed_60min'][sensor_idx]

                st.success(f"Forward Pass Completed via `{model_choice}`!")
                res_c1, res_c2, res_c3 = st.columns(3)
                res_c1.metric("Input Injected Speed", f"{inject_speed:.1f} mph")
                res_c2.metric("Predicted 30m Speed", f"{p30_sensor:.1f} mph", f"{p30_sensor - inject_speed:+.1f} mph")
                res_c3.metric("Predicted 60m Speed", f"{p60_sensor:.1f} mph", f"{p60_sensor - inject_speed:+.1f} mph")

                st.info(f"Network Alert Level: **{len(live_res['alerts'])}** active congestion bottlenecks across the 207-sensor highway system.")

    with tab_alerts:
        st.subheader("⚠️ Active Highway Congestion Warnings")
        latest_preds_all = y_pred_all[-1, :]
        alerts = []
        for i, sid in enumerate(sensor_ids):
            spd = latest_preds_all[i]
            if spd < 25.0:
                alerts.append({'Sensor': f"#{sid}", 'Predicted Speed': f"{spd:.1f} mph", 'Severity': '🚨 SEVERE', 'Action': 'Dispatch Traffic Advisory / Signal Timing Adjustment'})
            elif spd < 40.0:
                alerts.append({'Sensor': f"#{sid}", 'Predicted Speed': f"{spd:.1f} mph", 'Severity': '⚠️ MODERATE', 'Action': 'Monitor Corridor Ramp Metering'})

        if alerts:
            st.warning(f"Detected {len(alerts)} congested corridor bottlenecks across the highway network.")
            alerts_df = pd.DataFrame(alerts)
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.success("All highway corridors operating under normal free-flow conditions.")

    with tab_metrics:
        st.subheader("Model Performance Benchmark on Unseen Test Split")
        st.markdown("Rigorous evaluation on 3,416 time steps across all 207 road sensors:")

        metrics_file = os.path.join("outputs", "metrics", "traffic_metrics.json")
        metrics_dict = load_metrics(metrics_file)

        rows = []
        for m_name in ['Baseline', 'LSTM', 'ST-GNN']:
            if m_name in metrics_dict:
                m30 = metrics_dict[m_name]['horizon_30min']
                m60 = metrics_dict[m_name]['horizon_60min']
                lat = metrics_dict[m_name].get('latency_ms_per_sample', 0.0)
                rows.append({
                    'Model': m_name,
                    '30m MAE (mph)': m30['mae'],
                    '30m RMSE (mph)': m30['rmse'],
                    '30m MAPE (%)': m30['mape'],
                    '30m R²': m30['r2'],
                    '60m MAE (mph)': m60['mae'],
                    '60m RMSE (mph)': m60['rmse'],
                    '60m MAPE (%)': m60['mape'],
                    '60m R²': m60['r2'],
                    'Inference Latency (ms/sample)': lat
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.image("outputs/plots/traffic_horizon_comparison.png", caption="Multi-Horizon Forecast Error (30m vs 60m)", use_column_width=True)
        with col_p2:
            st.image("outputs/plots/traffic_loss_curves.png", caption="Training & Validation Loss Convergence", use_column_width=True)
