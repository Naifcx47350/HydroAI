# anomaly_detection.py
# -------------------------------
# Autoencoder-based Anomaly Detection
# -------------------------------

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """
    Scales and creates time-window sequences from flow data 
    for anomaly detection (e.g., LSTM Autoencoder).
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.flow_scaler = MinMaxScaler()

    def process_flow(self, series):
        """
        Scale flow data and transform into sliding windows of length 'window_size'.
        """
        scaled = self.flow_scaler.fit_transform(series.values.reshape(-1, 1))
        return self._create_sequences(scaled)

    def _create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.window_size):
            sequences.append(data[i: i + self.window_size])
        return np.array(sequences)


class AnomalyDetector:
    """
    An LSTM autoencoder that learns to reconstruct normal flow patterns, 
    then flags high reconstruction error as anomalies.
    """

    def __init__(self, window_size=30, latent_dim=8):
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.model = self._build_model()

    def _build_model(self):
        inputs = keras.Input(shape=(self.window_size, 1))
        # Encoder
        x = layers.LSTM(16, return_sequences=True)(inputs)
        x = layers.LSTM(self.latent_dim)(x)
        # Decoder
        x = layers.RepeatVector(self.window_size)(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True)(x)
        x = layers.LSTM(16, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(1))(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X_train, epochs=50, batch_size=32):
        """
        Trains the autoencoder to reconstruct normal flow sequences.
        """
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True
        )
        self.model.fit(
            X_train, X_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )

    def detect_anomalies(self, X_data, threshold_percentile=95):
        """
        Reconstruct input sequences, compute MSE, 
        and apply percentile-based threshold for anomalies.
        """
        reconstructions = self.model.predict(X_data)
        mse = np.mean((X_data - reconstructions)**2, axis=(1, 2))
        threshold = np.percentile(mse, threshold_percentile)
        anomalies = mse > threshold
        return anomalies, threshold

    def convert_to_tflite(self, save_path="models/anomaly_detector.tflite"):
        """
        Convert the Keras model to TensorFlow Lite for edge/IoT deployment.
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        with open(save_path, 'wb') as f:
            f.write(tflite_model)
