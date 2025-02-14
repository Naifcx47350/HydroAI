# gui_streamlit.py
# ------------------------------------------
# Enhanced Streamlit GUI for HydroAI
# ------------------------------------------

import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score

from data_pipeline.anomaly_detection import DataPreprocessor, AnomalyDetector
from data_pipeline.forecasting import ConsumptionForecaster
from data_pipeline.quality_classifier import WaterQualityClassifier
from data_pipeline.data_generators import generate_flow_data, generate_quality_data


def load_lottiefile(filepath: str):
    if "http" in filepath:
        r = requests.get(filepath)
        return r.json()
    else:
        with open(filepath) as f:
            
            return json.load(f)


lottie_file = load_lottiefile("water.json")

# * 1. Anomaly Detection


def run_anomaly_detection(seed_val):
    """Enhanced Anomaly Detection with data preview and plotting."""
    # * Generate or load data with random_state = seed_val
    flow_data = generate_flow_data(num_points=300, random_state=seed_val)
    df_flow = flow_data.to_frame(name="Flow_Rate")

    st.write("### Flow Data Preview")
    st.dataframe(df_flow.head(20))

    st.write("#### Flow Rate Over Time")
    st.line_chart(df_flow)

    st.write("Training anomaly detection model...")
    preprocessor = DataPreprocessor(window_size=30)
    X_flow = preprocessor.process_flow(flow_data)
    X_train = X_flow[:200]

    detector = AnomalyDetector(window_size=30, latent_dim=8)
    detector.train(X_train, epochs=20, batch_size=16)

    anomalies_array, threshold = detector.detect_anomalies(X_flow)
    count_anomalies = np.sum(anomalies_array)

    st.write(f"**Threshold:** {threshold:.4f}")
    st.write(f"**Number of anomalies detected:** {count_anomalies}")

    # Plot anomalies
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_flow.index, df_flow["Flow_Rate"], label="Flow Rate")

    # Indices where anomalies == True
    anomaly_indices = np.where(anomalies_array)[0]
    ax.scatter(
        anomaly_indices,
        df_flow["Flow_Rate"].iloc[anomaly_indices],
        color='red',
        label="Anomalies"
    )
    ax.set_title("Flow Rate with Detected Anomalies")
    ax.legend()
    st.pyplot(fig)

    return threshold, count_anomalies, flow_data, anomalies_array


# * 2. Consumption Forecasting


def run_forecasting(seed_val):
    """
    Demonstrates consumption forecasting with synthetic flow data 
    using log transform + MinMax scaling. 
    Shows daily usage preview, line charts,
    and displays multiple metrics + next-day prediction.
    """
    # 1) Generate ~120 days of data => 24 * 120 = 2880 points
    #    So the model sees enough examples for training
    flow_data = generate_flow_data(num_points=(2880), random_state=seed_val)

    # 2) Build forecaster with log transform
    forecaster = ConsumptionForecaster(
        lookback=7, hidden_units=16, use_log=True)
    X_train, X_test, y_train, y_test, daily_usage_log = forecaster.prepare_data(
        flow_data, daily_points=24
    )
    # 'daily_usage_log' is log1p(daily_usage) if use_log=True

    # We'll store a separate unscaled daily usage for display
    # We can do the reverse of the log transform if we want:
    daily_usage_unscaled = np.expm1(
        daily_usage_log) if forecaster.use_log else daily_usage_log
    daily_usage_unscaled = daily_usage_unscaled.flatten()

    # 3) Show daily usage preview
    df_daily = pd.DataFrame({
        "Day_Index": np.arange(len(daily_usage_unscaled)),
        "Daily_Usage": daily_usage_unscaled
    })
    st.write("### Daily Usage Preview (Unscaled)")
    st.dataframe(df_daily.head(20))

    st.write("#### Daily Usage Over Time")
    st.line_chart(df_daily.set_index("Day_Index")["Daily_Usage"])

    # 4) Train/test split
    st.write("Training the forecasting model...")

    X_train, X_test, y_train, y_test, daily_usage_log = forecaster.prepare_data(
        flow_data, daily_points=24
    )

    forecaster.train(X_train, y_train, epochs=100, batch_size=32)

    # 5) Predict on test set
    preds_scaled = forecaster.model.predict(X_test).flatten()
    # Convert from scaled (and possibly log) back to real daily usage
    preds_unscaled = forecaster.inverse_unscale(preds_scaled)
    y_test_unscaled = forecaster.inverse_unscale(y_test)

    # 6) Evaluate in unscaled space
    mse_test = np.mean((preds_unscaled - y_test_unscaled)**2)
    mae_test = mean_absolute_error(y_test_unscaled, preds_unscaled)
    rmse_test = np.sqrt(mse_test)
    r2 = r2_score(y_test_unscaled, preds_unscaled)

    # 7) Next day forecast
    # We take the last 7 daily usage points in unscaled form
    # Actually we need the same log transform approach as in predict_next
    recent_sequence = daily_usage_unscaled[-forecaster.lookback:
                                           ].reshape(-1, 1)
    next_day_pred = forecaster.predict_next(recent_sequence)

    # 8) Display metrics
    st.write("### Evaluation Metrics (Test Set)")
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{mse_test:.2f}")
    c2.metric("RMSE", f"{rmse_test:.2f}")
    c3.metric("MAE", f"{mae_test:.2f}")
    st.write(f"**R² Score**: {r2:.3f}")
    st.write(f"**Next-day Usage Prediction**: {next_day_pred:.2f}")

    # 9) Build DataFrame for test results
    df_results = pd.DataFrame({
        "Test_Day_Index": np.arange(len(y_test_unscaled)),
        "Actual_Usage": y_test_unscaled,
        "Predicted_Usage": preds_unscaled,
        "Absolute_Error": np.abs(y_test_unscaled - preds_unscaled)
    })

    # 10) Append next day
    last_test_idx = df_results["Test_Day_Index"].iloc[-1]
    future_idx = last_test_idx + 1

    future_row = pd.DataFrame({
        "Test_Day_Index": [future_idx],
        "Actual_Usage": [np.nan],
        "Predicted_Usage": [next_day_pred],
        "Absolute_Error": [np.nan]
    })
    df_extended = pd.concat([df_results, future_row], ignore_index=True)

    # 11) Plot
    fig_line, ax_line = plt.subplots(figsize=(8, 4))
    ax_line.plot(
        df_results["Test_Day_Index"],
        df_results["Actual_Usage"],
        label="Actual (Test)",
        marker='o'
    )
    ax_line.plot(
        df_results["Test_Day_Index"],
        df_results["Predicted_Usage"],
        label="Predicted (Test)",
        marker='x'
    )
    last_test_pred = df_results["Predicted_Usage"].iloc[-1]
    ax_line.plot(
        [last_test_idx, future_idx],
        [last_test_pred, next_day_pred],
        linestyle='--', color='orange', marker='o',
        label="Next-Day Forecast"
    )
    ax_line.set_title("Forecasting: Actual vs. Predicted (+ Next Day)")
    ax_line.set_xlabel("Test Day Index")
    ax_line.set_ylabel("Daily Usage (Unscaled)")
    ax_line.legend()
    st.pyplot(fig_line)

    st.write("### Extended Forecast (Test + Future)")
    st.dataframe(df_extended.tail(10))

    # 12) Error Distribution
    st.write("#### Error Distribution (Predicted - Actual)")
    errors = preds_unscaled - y_test_unscaled
    fig_err, ax_err = plt.subplots(figsize=(6, 4))
    sns.histplot(errors, kde=True, color='orange', ax=ax_err)
    ax_err.set_title("Distribution of Errors")
    ax_err.set_xlabel("Prediction Error")
    st.pyplot(fig_err)

    return mse_test, next_day_pred

# * 3. Water Quality Classification


def run_quality_classification(seed_val):
    """
    Demonstrates water quality classification with a simple feed-forward network.
    Includes data preview, confusion matrix, etc.
    """
    df = generate_quality_data(num_samples=(720*2), random_state=seed_val)

    st.write("### Water Quality Dataset Preview")
    st.dataframe(df.head(20))

    # ? Show distribution of pH
    st.write("#### Distribution of pH Values")
    fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
    sns.histplot(df["pH"], kde=True, color='green', ax=ax_dist)
    ax_dist.set_title("pH Distribution")
    st.pyplot(fig_dist)

    st.write("Training the water quality classifier...")
    classifier = WaterQualityClassifier()
    X, y = classifier.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed_val
    )
    classifier.train(X_train, y_train, epochs=10, batch_size=8)

    # Evaluate
    loss, acc = classifier.evaluate(X_test, y_test)

    # Predict sample
    # ? typical safe water(based on existing data)
    sample = np.array([[7.2, 280, 1.2]])
    pred = classifier.predict_quality(sample)
    label_str = "Safe" if pred[0][0] == 1 else "Unsafe"

    st.write(f"**Test Accuracy:** {acc:.2f}")
    st.write(f"**Sample Water Prediction:** {label_str}")

    # Confusion Matrix

    y_pred_probs = classifier.model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (Water Quality)")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    st.pyplot(fig_cm)

    # Additional summary below the matrix
    st.write("### Classification Summary")
    total_samples = len(y_test)
    predicted_safe = np.sum(y_pred == 1)
    predicted_unsafe = np.sum(y_pred == 0)

    # Actual distribution
    actual_safe = np.sum(y_test == 1)
    actual_unsafe = np.sum(y_test == 0)

    st.write(f"- **Total Test Samples**: {total_samples}")
    st.write(
        f"- **Predicted Safe**: {predicted_safe}, **Predicted Unsafe**: {predicted_unsafe}")
    st.write(
        f"- **Actual Safe**: {actual_safe}, **Actual Unsafe**: {actual_unsafe}")

    return acc, label_str


# * 4. Streamlit main
def main():
    st.title("HydroAI: Interactive Demo")
    st.write("A Streamlit UI to showcase Anomaly Detection, Forecasting, and Water Quality Classification.")

    # Let user choose a random seed or none
    st.sidebar.write("### Random Seed Selection")
    use_seed = st.sidebar.checkbox("Use a specific random seed?", value=False)
    seed_val = None
    if use_seed:
        seed_val = st.sidebar.number_input(
            "Enter Seed (e.g. 42)", min_value=0, max_value=999999, value=42)  # well be visible in the gui

    menu = ["Home", "Anomaly Detection",
            "Forecasting", "Quality Classification"]
    choice = st.sidebar.selectbox("Select a Demo", menu)

    if choice == "Home":
        st.subheader("Overview")
        st.write("""
        **HydroAI** is an AI-powered water monitoring system that:
        - Detects anomalies (e.g., leaks, bursts),
        - Forecasts future consumption,
        - Classifies water quality as safe or unsafe.

        Use the sidebar to navigate each demo.

        *Below, you can optionally specify a random seed.* 
        If unchecked, data is randomly generated each time.
        """)

        st_lottie(lottie_file, speed=1, height=150,
                  key="initial", loop=True, reverse=False)

    elif choice == "Anomaly Detection":
        st.subheader("Anomaly Detection Demo")
        if st.button("Run Anomaly Detection"):
            # expecting 4 values(would change with the final product)
            threshold, num_anomalies, flow_data, anomalies_array = run_anomaly_detection(
                seed_val)

    elif choice == "Forecasting":
        st.subheader("Consumption Forecasting Demo")
        if st.button("Run Forecasting"):
            mse_test, next_day = run_forecasting(seed_val)

    elif choice == "Quality Classification":
        st.subheader("Water Quality Classification Demo")
        if st.button("Run Quality Classification"):
            acc, label_str = run_quality_classification(seed_val)


if __name__ == "__main__":
    main()
