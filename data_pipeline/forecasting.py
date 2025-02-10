# forecasting.py
# --------------------------------
# LSTM for Water Consumption Forecasting (with optional log transform)
# --------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


class ConsumptionForecaster:
    """
    Predicts next-day usage from daily aggregated flow data.
    If use_log=True, we apply log1p transform before scaling
    to handle large variations more smoothly.
    """

    def __init__(self, lookback=14, hidden_units=128, use_log=True):
        """
        :param lookback: number of past daily points to use in each sequence.
        :param hidden_units: (not used as much now, we override in build_model).
        :param use_log: whether to apply log1p transform to daily usage.
        """
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.use_log = use_log
        self.scaler = MinMaxScaler()
        self.model = self.build_model()

    def prepare_data(self, flow_series, daily_points=24):
        """
        Sums hourly data into daily usage, optionally applies log transform,
        then scales with MinMax. Returns X (seqs), y (labels), daily_usage array
        (raw or log-transformed).
        """
        # 1) Sum hourly => daily
        num_days = len(flow_series) // daily_points
        daily_usage = [
            flow_series[i*daily_points: (i+1)*daily_points].sum()
            for i in range(num_days)
        ]
        daily_usage = np.array(daily_usage).reshape(-1, 1)

        # 2) (Optional) log transform => log1p
        if self.use_log:
            daily_usage = np.log1p(daily_usage)

        # Time-based split before scaling
        split_idx = int(len(daily_usage)*0.8)
        train_daily = daily_usage[:split_idx]
        test_daily = daily_usage[split_idx:]

        # Fit scaler ONLY on training data
        self.scaler.fit(train_daily)

        # Scale all data with train-based scaler
        scaled_daily = self.scaler.transform(daily_usage)

        # Build sequences from scaled data
        X, y = [], []
        for i in range(len(scaled_daily)-self.lookback):
            X.append(scaled_daily[i:i+self.lookback])
            y.append(scaled_daily[i+self.lookback])

        X = np.array(X)
        y = np.array(y)

        # Split sequences (first 80% for train)
        train_seq = split_idx - self.lookback
        return X[:train_seq], X[train_seq:], y[:train_seq], y[train_seq:], daily_usage.flatten()

    def build_model(self):
        model = keras.Sequential([
            layers.LSTM(128, input_shape=(self.lookback, 1)), layers.Dense(1)])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def inverse_unscale(self, scaled_values):
        """
        Inverse transform from MinMax, then expm1() if using log.
        scaled_values shape: (N,) or (N,1)
        Returns: array shape (N,)
        """
        if len(scaled_values.shape) == 1:
            scaled_values = scaled_values.reshape(-1, 1)

        unscaled = self.scaler.inverse_transform(scaled_values)
        if self.use_log:
            unscaled = np.expm1(unscaled)
        return unscaled.flatten()

    def predict_next(self, recent_sequence):
        """
        Predict next day's usage from the last 'lookback' days.
        If use_log=True, we log-transform and scale input,
        then invert transform after LSTM prediction.
        """
        seq = recent_sequence.copy()

        if self.use_log:
            seq = np.log1p(seq)
        seq_scaled = self.scaler.transform(seq)
        seq_scaled = seq_scaled.reshape(1, self.lookback, 1)

        pred_scaled = self.model.predict(seq_scaled)
        pred_unscaled = self.inverse_unscale(pred_scaled)
        return pred_unscaled[0]

    def convert_to_tflite(self, save_path="models/consumption_forecaster.tflite"):
        """
        Convert Keras model to TFLite for IoT deployment.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
