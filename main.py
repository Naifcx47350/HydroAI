# main.py
# -------------------------------
# Entry point for local demo
# -------------------------------

import numpy as np
from sklearn.model_selection import train_test_split

# Import from data_pipeline
from data_pipeline.data_generators import generate_flow_data, generate_quality_data
from data_pipeline.anomaly_detection import DataPreprocessor, AnomalyDetector
from data_pipeline.forecasting import ConsumptionForecaster
from data_pipeline.quality_classifier import WaterQualityClassifier


def demo_anomaly_detection():
    print("\n=== Anomaly Detection Demo ===")
    flow_data = generate_flow_data(num_points=1000)
    preprocessor = DataPreprocessor(window_size=30)
    X_flow = preprocessor.process_flow(flow_data)

    # Assume first 300 windows are normal for training
    X_train = X_flow[:300]

    detector = AnomalyDetector(window_size=30, latent_dim=8)
    detector.train(X_train, epochs=10, batch_size=32)

    anomalies, threshold = detector.detect_anomalies(X_flow)
    print(f"Threshold: {threshold:.4f}")
    print(f"Total anomalies detected: {anomalies.sum()}")

    detector.convert_to_tflite("models/anomaly_detector.tflite")


def demo_forecasting():
    print("\n=== Consumption Forecasting Demo ===")
    # e.g., 2400 hours for ~100 days
    flow_data = generate_flow_data(num_points=2400)
    forecaster = ConsumptionForecaster(lookback=7, hidden_units=32)
    X_days, y_days, daily_usage = forecaster.prepare_data(
        flow_data, daily_points=24)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_days, y_days,
        test_size=0.2,
        shuffle=False
    )
    forecaster.train(X_train, y_train, epochs=10, batch_size=8)

    preds = forecaster.model.predict(X_test)
    mse_test = np.mean((preds - y_test)**2)
    print(f"Forecast Test MSE: {mse_test:.4f}")

    # Predict next-day usage from last 'lookback' days
    last_seq = daily_usage[-forecaster.lookback:].reshape(-1, 1)
    next_day = forecaster.predict_next(last_seq)
    print(f"Predicted next-day usage: {next_day:.2f}")

    forecaster.convert_to_tflite("models/consumption_forecaster.tflite")


def demo_quality_classification():
    print("\n=== Water Quality Classification Demo ===")
    df = generate_quality_data(num_samples=1000)

    classifier = WaterQualityClassifier()
    X, y = classifier.prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    classifier.train(X_train, y_train, epochs=10, batch_size=16)
    loss, acc = classifier.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    sample = np.array([[7.2, 280, 1.2]])  # pH=7.2, TDS=280, turbidity=1.2
    pred = classifier.predict_quality(sample)
    label_str = "Safe" if pred[0][0] == 1 else "Unsafe"
    print(f"Sample water classified as: {label_str}")

    classifier.convert_to_tflite("models/water_quality_classifier.tflite")


if __name__ == "__main__":
    print("\n=== Running HydroAI Local Demo ===")
    demo_anomaly_detection()
    demo_forecasting()
    demo_quality_classification()

    print("\n=== All demos completed successfully ===")
